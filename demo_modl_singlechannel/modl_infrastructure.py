import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from MoDLsinglechannel.demo_modl_singlechannel.MoDL_single import UnrolledModel
from MoDLsinglechannel.demo_modl_singlechannel.utils import complex_utils as cplx
from alon.fastmri_preprocess import ifftc, fftc

import matplotlib.pyplot as plt


@dataclass
class MoDLParams:
    """ 8-step MoDL """
    # data_path: Path = Path(
    #     '/mnt/c/Users/along/brain_multicoil_train_batch_0/multicoil_train'
    # )
    # batch_size: int = 2 #2 #4
    # num_grad_steps: int = 8     #5 #3 #8 #3        #4
    # num_cg_steps: int = 8
    # share_weights: int = True
    # modl_lamda: int = 0.05
    # lr: int = 1e-3 #1e-5 #1e-4
    # weight_decay: int = 0
    # lr_step_size: int = 500
    # lr_gamma: int = 0.5
    # epoch: int = 21

    """ 4-step MoDL """
    data_path: Path = Path(
        '/mnt/c/Users/along/brain_multicoil_train_batch_0/multicoil_train'
    )
    batch_size: int = 2     #2 #4
    num_grad_steps: int = 4     #5 #3 #8 #3        #4
    num_cg_steps: int = 8
    share_weights: int = True
    modl_lamda: int = 0.01 #0.05
    lr: int = 1e-4 #1e-5 #1e-4
    weight_decay: int = 0
    lr_step_size: int = 500
    lr_gamma: int = 0.5
    epoch: int = 21


def build_optim(args, params):
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    return optimizer


class MoDLWrapper:
    def __init__(self, modl_params: MoDLParams, save_dir: Path, device: str = 'cpu', trained_model: UnrolledModel = None):
        """
        Handles the training and loading of MoDL.

        :param modl_params:
        :param device:
        """
        self.modl_params = modl_params
        self.device = device
        self.save_dir = save_dir

        self.single_MoDL = trained_model if trained_model is not None else  UnrolledModel(self.modl_params).to(self.device)
        self.optimizer = build_optim(self.modl_params, self.single_MoDL.parameters())
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.modl_params.lr_step_size,
                                                         self.modl_params.lr_gamma)
        self.criterion = nn.MSELoss()

    def checkpoint_naming(self, model_name: str, epoch: Union[int, str]):
        return f'Model {model_name}  epoch {epoch}'

    def load(self, checkpoints_dir: Path, model_name: str, epoch: Union[int, str] = 'last') -> UnrolledModel:
        """

        :param checkpoints_dir:
        :param model_name:
        :param epoch:               If "last", then chooses the latest epoch.
        :return:
        """
        assert (epoch == 'last') or isinstance(epoch, int)
        if epoch == 'last':
            search_str = self.checkpoint_naming(model_name, epoch='*')
            print('search_str: ', search_str)
            paths = list(checkpoints_dir.glob(f'{search_str}.pt'))
            extract_epoch = lambda _path: int(_path.stem.split('epoch ')[-1])
            path = paths[np.argmax(list(map(extract_epoch, paths)))]
        else:
            path = checkpoints_dir.joinpath(f'{self.checkpoint_naming(model_name, epoch)}.pt')

        self.single_MoDL.load_state_dict(
            torch.load(path)['model']
        )
        self.single_MoDL.eval()
        return self.single_MoDL

    def train(self, data_loader: DataLoader, model_name: str = 'model', save_freq: int = 1):
        for epoch in range(self.modl_params.epoch):
            print(f'Epoch {epoch}')
            self.single_MoDL.train()
            avg_loss = 0.

            for iter, data in enumerate(data_loader):
                print(f'{iter=}...')

                # Unpack data
                y, x_true, mask = (obj.to(self.device) for obj in data)

                # Reconstruct
                x_recon = self.single_MoDL(y.float(), mask=mask)

                # Backprop
                loss = self.criterion(x_recon, x_true)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Training state display
                rate = 0.1
                avg_loss = (1 - rate) * avg_loss + rate * loss.item() if iter > 0 else loss.item()
                print(f'\tInstant loss: {loss.item()}. \t\tAvg loss: {avg_loss}')

                if iter % save_freq == save_freq - 1:
                    # Saving the model
                    self.save_dir.mkdir(exist_ok=True)
                    torch.save(
                        {
                            'epoch': epoch,
                            'params': self.modl_params,
                            'model': self.single_MoDL.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'exp_dir': self.save_dir.__str__(),

                            'MASK': data_loader.MASK,
                        },
                        f=self.save_dir.joinpath(f'Model {model_name}  epoch {epoch}.pt')
                    )

    def test_run(self, modl: UnrolledModel, data_loader: DataLoader, show_steps: bool = True):
        random_start = np.random.randint(0, len(data_loader.dataset) - self.modl_params.batch_size - 1)
        for iter, data in enumerate(data_loader, start=random_start):
            print(f'{iter=}...')

            # Unpack data
            y, x_true, mask = (obj.to(self.device) for obj in data)

            # Reconstruct
            x_recon, x_intermed = modl(y.float(), mask=mask, return_steps=True)
            break

        recon_image = cplx.abs(x_recon.detach())
        gt_image = cplx.abs(x_true.detach())
        in_image = torch.tensor(abs(ifftc(torch.tensor(fftc(gt_image)) * mask[..., 0]))).to(self.device)  # abs(ifftc(in_kspace[..., 0]) + 1j * ifftc(in_kspace[..., 1]))

        while True:
            sample = int(input(
                f'Which sample (in range [0, {gt_image.shape[0] - 1}]) would you like to display?'
                f'Insert -1 to exit'
            ))
            if sample == -1:
                return

            if show_steps:
                fig, ax = plt.subplots(1, 2 + len(x_intermed), sharex=True, sharey=True)
                ax[0].set_title('Target')
                ax[0].imshow(torch.flipud(gt_image[sample]), cmap='Greys_r', vmin=0, vmax=1.2)
                ax[1].set_title('6x accelerated zero-filled')
                ax[1].imshow(torch.flipud(in_image[sample]), cmap='Greys_r', vmin=0, vmax=1.2)
                for i in range(2, 2 + len(x_intermed)):
                    ax[i].set_title(f'Recon step {i - 2}')
                    ax[i].imshow(torch.flipud(cplx.abs(x_intermed[i - 2][sample].detach())), cmap='Greys_r', vmin=0,
                                 vmax=1.2)
                fig.set_size_inches(30, 7)
                plt.tight_layout()
                fig.show()
            else:
                fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
                ax[0].set_title('Target')
                ax[0].imshow(gt_image[sample], cmap='Greys_r', vmin=0, vmax=1.2, origin='lower')
                ax[1].set_title('6x accelerated zero-filled')
                ax[1].imshow(in_image[sample], cmap='Greys_r', vmin=0, vmax=1.2, origin='lower')
                ax[2].set_title('Reconstructed')
                ax[2].imshow(recon_image[sample], cmap='Greys_r', vmin=0, vmax=1.2, origin='lower')
                fig.set_size_inches(13, 5)
                plt.tight_layout()
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
