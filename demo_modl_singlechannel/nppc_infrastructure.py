import logging
from pathlib import Path

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple

from torch.utils.data import DataLoader

from alon.MoDLsinglechannel.demo_modl_singlechannel.MoDL_single import UnrolledModel
from nppc.nppc import PCWrapper
from utils import complex_utils as cplx

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class UNet(nn.Module):
    def __init__(
            self,
            in_channels=1,
            out_channels=1,
            channels_list=(32, 64, 128, 256),
            bottleneck_channels=512,
            min_channels_decoder=64,
            n_groups=8,
        ):

        super().__init__()
        ch = in_channels

        ## Encoder
        ## =======
        self.encoder_blocks = nn.ModuleList([])
        ch_hidden_list = []

        layers = []
        layers.append(nn.ZeroPad2d(2))
        ch_ = channels_list[0]
        layers.append(nn.Conv2d(ch, ch_, 3, padding=1))
        ch = ch_
        self.encoder_blocks.append(nn.Sequential(*layers))
        ch_hidden_list.append(ch)

        for i_level in range(len(channels_list)):
            ch_ = channels_list[i_level]
            downsample = i_level != 0

            layers = []
            if downsample:
                layers.append(nn.MaxPool2d(2))
            layers.append(nn.Conv2d(ch, ch_, 3, padding=1))
            ch = ch_
            layers.append(nn.GroupNorm(n_groups, ch))
            layers.append(nn.LeakyReLU(0.1))
            self.encoder_blocks.append(nn.Sequential(*layers))
            ch_hidden_list.append(ch)

        ## Bottleneck
        ## ==========
        ch_ = bottleneck_channels
        layers = []
        layers.append(nn.Conv2d(ch, ch_, 3, padding=1))
        ch = ch_
        layers.append(nn.GroupNorm(n_groups, ch))
        layers.append(nn.LeakyReLU(0.1))
        layers.append(nn.Conv2d(ch, ch, 3, padding=1))
        layers.append(nn.GroupNorm(n_groups, ch))
        layers.append(nn.LeakyReLU(0.1))
        self.bottleneck = nn.Sequential(*layers)

        ## Decoder
        ## =======
        self.decoder_blocks = nn.ModuleList([])
        for i_level in reversed(range(len(channels_list))):
            ch_ = max(channels_list[i_level], min_channels_decoder)
            downsample = i_level != 0
            ch = ch + ch_hidden_list.pop()
            layers = []

            layers.append(nn.Conv2d(ch, ch_, 3, padding=1))
            ch = ch_
            layers.append(nn.GroupNorm(n_groups, ch))
            layers.append(nn.LeakyReLU(0.1))
            if downsample:
                layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            self.decoder_blocks.append(nn.Sequential(*layers))

        ch = ch + ch_hidden_list.pop()
        ch_ = channels_list[0]
        layers = []
        layers.append(nn.Conv2d(ch, out_channels, 1))
        layers.append(nn.ZeroPad2d(-2))
        self.decoder_blocks.append(nn.Sequential(*layers))

    def forward(self, x):
        h = []
        for block in self.encoder_blocks:
            x = block(x)
            h.append(x)

        x = self.bottleneck(x)
        for block in self.decoder_blocks:
            x = torch.cat((x, h.pop()), dim=1)
            x = block(x)
        return x


@dataclass
class NPPCParams:
    # Inference params
    n_dirs: int = 5
    nppc_step: int = 0
    restoration_n_steps: int = 4000
    nppc_n_steps: int = 3000

    # Training params
    #   How many samples we insert into the network in parallel
    batch_size: int = 2
    #   How many such batches we use the losses thereof to calculate one loss that we backprop on
    batches_per_loss_calc: int = 4
    #   Loss
    max_loss_calc_batches: int = 16
    #   todo Add epochs (this was not originally in the code)

    #   Loss parameters
    second_moment_loss_lambda: float = 1e0
    second_moment_loss_grace: float = 500
    #       Data consistency loss component. Set to 0 if we don't want to calculate it
    dc_loss_lambda: float = 5e0  # 1e0


class NPPCForMoDLWrapper:
    def __init__(self, nppc_params: NPPCParams, modl: UnrolledModel, save_dir: Path):
        self.nppc_params = nppc_params
        self.nppc_net = PCWrapper(UNet(in_channels=1 + 1, out_channels=1 * self.nppc_params.n_dirs),
                                  n_dirs=self.nppc_params.n_dirs)  # , mask_ifft=mask_ifft)
        self.nppc_net.__setattr__('ddp', Namespace(size=1))
        self.nppc_net.to(device)
        self.nppc_net.train()
        self.nppc_optimizer = torch.optim.Adam(
            self.nppc_net.parameters(), lr=1e-4,  # lr=1e-5,
            betas=(0.9, 0.999)
        )

        self.modl = modl
        self.save_dir = save_dir

    def load(self, model_name: str = 'modl_nppc_dc'):
        net = torch.load(self.save_dir.joinpath(model_name + '.pth'))
        return net

    def train(self, data_loader: DataLoader, nppc_position, model_name: str = 'modl_nppc_dc', save_freq: int = 1):
        for loss_calc_batch in range(self.nppc_params.max_loss_calc_batches):
            objective = self._accumulate_losses(data_loader, nppc_position, loss_calc_batch)
            self.nppc_optimizer.zero_grad()
            objective.backward()
            self.nppc_optimizer.step()

            print(f'Loss: {objective.detach().item()}')
            if loss_calc_batch % save_freq == save_freq - 1:
                torch.save(self.nppc_net, self.save_dir.joinpath(model_name + '.pth'))

    def _accumulate_losses(self, data_loader: DataLoader, nppc_position, loss_calc_batch: int = 0):
        # Get the loss of this batch
        start = self.nppc_params.batch_size * loss_calc_batch
        losses = list()
        for iter, data in enumerate(data_loader, start=start):
            print(f'Iter {iter}...')
            losses.append(self._get_batch_loss(data, nppc_position))
            if len(losses) == self.nppc_params.batches_per_loss_calc:
                break
        # Combine together to a batch loss.
        objective = torch.stack(losses).mean()
        return objective

    def _get_batch_loss(self, data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], nppc_position: int = 0):
        """

        :param data:
        :param nppc_position:       Position of input to the NPPC (not position of the MMSE)
        :return:
        """
        assert nppc_position < self.modl.num_grad_steps,\
            f'Chosen NPPC placement ({nppc_position}) is out of bounds, not in [0, {self.modl.num_grad_steps} - 1]'

        # Unpack training data: subsampled k-space, target image, k-space mask
        y, x_true, mask = (obj.to(device) for obj in data)

        # Reconstruct, apply NPPC, calculate image-domain error
        with torch.no_grad():
            x_recon, x_intermed = self.modl(y.float(), mask=mask, return_steps=True)
        print('\tReconstructed')

        step_input = cplx.abs(x_intermed[nppc_position])[:, None, ...]
        step_mmse = cplx.abs(x_intermed[nppc_position + 1])[:, None, ...]
        w_mat = self.nppc_net(step_input, step_mmse)
        print('\tEstimated principal components')

        w_mat_ = w_mat.flatten(2)
        w_norms = w_mat_.norm(dim=2)
        w_hat_mat = w_mat_ / w_norms[:, :, None]

        x_true = cplx.abs(x_true.detach())[:, None, ...]
        err = (x_true - step_mmse).flatten(1)

        ## Normalizing by the error's norm
        ## -------------------------------
        err_norm = err.norm(dim=1)
        err = err / err_norm[:, None]
        w_norms = w_norms / err_norm[:, None]

        ## W hat loss
        ## ----------
        err_proj = torch.einsum('bki,bi->bk', w_hat_mat, err)
        reconst_err = 1 - err_proj.pow(2).sum(dim=1)

        ## W norms loss
        ## ------------
        second_moment_mse = (w_norms.pow(2) - err_proj.detach().pow(2)).pow(2)

        ## (Alon, ShimronLab) Data-consistency loss
        ## ----------------------------------------
        dc_loss = 0
        if self.nppc_params.dc_loss_lambda != 0:
            MFW = torch.einsum(
                'cije, edij -> ec',
                mask, abs(torch.fft.fftshift(torch.fft.fft2(w_mat[:, :, 0], axis=(1, 2)), axis=(1, 2)))
            ) / (2 * err.size()[1])  # c: Re/Im component, i,j: image dims, e: example, d: PC directions
            dc_loss = torch.linalg.norm(MFW) / (MFW.shape[0] * MFW.shape[1])

        second_moment_loss_lambda = -1 + 2 * self.nppc_params.nppc_step / self.nppc_params.second_moment_loss_grace
        second_moment_loss_lambda = max(min(second_moment_loss_lambda, 1), 1e-6)
        second_moment_loss_lambda *= second_moment_loss_lambda

        loss = (reconst_err.mean()
                + second_moment_loss_lambda * second_moment_mse.mean()
                + self.nppc_params.dc_loss_lambda * dc_loss)
        return loss

    def reset(self, nppc_params: NPPCParams = None):
        self.__init__(self.nppc_params if nppc_params is None else nppc_params, self.modl, self.save_dir)

    def test_run(self, nppc_net: UNet, data_loader: DataLoader, nppc_position: int = 0):
        for iter, data in enumerate(data_loader):
            # Unpack training data: subsampled k-space, target image, k-space mask
            y, x_true, mask = (obj.to(device) for obj in data)

            # Reconstruct, apply NPPC, calculate image-domain error
            with torch.no_grad():
                x_recon, x_intermed = self.modl(y.float(), mask=mask, return_steps=True)
                print('\tReconstructed')

                step_input = cplx.abs(x_intermed[nppc_position])[:, None, ...]
                step_mmse = cplx.abs(x_intermed[nppc_position + 1])[:, None, ...]
                w_mat = self.nppc_net(step_input, step_mmse)
            break

        while True:
            sample = int(input(
                f'Which sample (in range [0, {x_true.shape[0] - 1}]) would you like to display?'
                f'Insert -1 to exit'
            ))
            if sample == -1:
                return
            plot_type = int(input('0 - Plot PCs   |   1 - Create video'))

            if plot_type == 0:
                # Showing all PCs
                fig, ax = plt.subplots(1, 5)
                for c in range(5):
                    ax[c].set_title(f'PC {c + 1}')
                    ax[c].imshow(w_mat.detach()[sample, c, 0], vmin=-0.5, vmax=0.5, cmap='RdBu')
                fig.set_size_inches(20, 6)
                # fig.set_size_inches(10, 5)
                fig.tight_layout()
                # fig.savefig('/home/alon_granek/PythonProjects/NPPC/temp2.png')
                fig.show()

            action = int(input('0 - Save   |    1 - Continue   |   2 - Exit'))
            if action == 0:
                save_name = str(input('Name:'))
                path = Path(__file__).parent.parent.joinpath(save_name + '.png')
                fig.savefig(path)
                print(f'Saved figure in: {path}')
            elif action == 1:
                continue
            elif action == 2:
                return
