import logging
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from MoDLsinglechannel.demo_modl_singlechannel.utils import complex_utils as cplx
from MoDLsinglechannel.demo_modl_singlechannel.utils.datasets import SliceData
from typing import Callable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from MoDLsinglechannel.demo_modl_singlechannel.subsample_fastmri import SaveableMask


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class DataTransform:
    """
    Data Transformer for training unrolled reconstruction models.
    """

    def __init__(self, mask_generator: Callable):
        self.mask_generator = mask_generator

    def __call__(self, kspace, target):
        """
        The forward model.

        :param kspace:
        :param target:
        :return:
        """
        # Normalize
        scale = self.normalization(target)
        kspace /= scale
        target /= scale

        # Convert to torch tensors
        kspace_torch = cplx.to_tensor(kspace).float()
        target_torch = cplx.to_tensor(target).float()

        # k-space masking
        mask = self.mask_generator()
        kspace_masked = kspace_torch * mask
        return kspace_masked, target_torch, mask

    def normalization(self, image: np.ndarray, norm_percentile: float = 95):
        vals = image.reshape(-1)
        n_taken = int(round((1 - norm_percentile * 1e-2) * vals.shape[0]))
        scale = vals[vals.argsort()[::-1][n_taken]]
        return scale


def create_data_loaders(args, mask: SaveableMask):
    # Set up training data
    train_data = SliceData(
        root=Path(str(args.data_path)),
        transform=DataTransform(mask.mask_generator),
        sample_rate=1
    )
    # Set up loader
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    # Include mask info
    train_loader.MASK = mask.name

    return train_loader



