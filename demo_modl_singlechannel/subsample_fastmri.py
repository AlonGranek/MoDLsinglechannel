"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
"""
Bank of sampling masks to choose
"""

import numpy as np
import torch
from sigpy.mri import poisson
from typing import Tuple, Union, Callable
from pathlib import Path

from pickle import load, dump


from uuid import uuid4


"""
With MoDL, appending MMSEs and an undersampling mask
"""


class SaveableMask:
    def __init__(self, masks_dir: Union[Path, str], mask_generator: Callable = None):
        self.masks_dir = Path(str(masks_dir))
        self.name = None
        self.mask_generator = mask_generator

    def save(self, name: str = None):
        assert self.mask_generator is not None, 'mask_generator should not be None'

        self.masks_dir.mkdir(exist_ok=True)

        # If saved name is None, set a UUID name
        self.name = name if name is not None else str(uuid4())

        path = self.masks_dir.joinpath(self.name + '.pkl')
        with open(path, 'wb') as file:
            dump(self.mask_generator, file)
            print(f'Successfully saved mask generator in {path}')

    def load(self, name: str):
        self.name = name
        path = self.masks_dir.joinpath(self.name + '.pkl')
        with open(path, 'rb') as file:
            self.mask_generator = load(file)
            print(f'Successfully loaded mask generator from {path}')


class MaskFuncGiven:
    def __init__(self, mask: torch.Tensor):
        """
        Simply return the given mask.
        This class exists for consistency with the other methods.
        """
        self.mask = mask

    def __call__(self):
        return self.mask


class MaskFuncPoisson2D:
    def __init__(
            self,
            image_shape: Tuple[int, int] = (372, 372),
            accel: float = 6,
            calib_shape: Tuple[int, int] = (56, 56)
    ):
        self.image_shape = image_shape
        self.accel = accel
        self.calib_shape = calib_shape

    def __call__(self) -> torch.Tensor:
        mask = poisson(
            img_shape=self.image_shape,
            accel=self.accel,  # 4,
            calib=self.calib_shape,
            dtype=float, crop_corner=True, return_density=False, seed=0, max_attempts=6, tol=0.1
        )
        mask_torch = torch.stack([torch.tensor(mask).float(), torch.tensor(mask).float()], dim=2)
        return mask_torch


class MaskFuncEquispacedLines:
    """
    MaskFunc creates a sub-sampling mask of a given shape.
    The mask selects a subset of columns from the input k-space data. If the k-space data has N
    columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center corresponding to
           low-frequencies
        2. The other columns are selected uniformly at random with a probability equal to:
           prob = (N / acceleration - N_low_freqs) / (N - N_low_freqs).
    This ensures that the expected number of columns selected is equal to (N / acceleration)
    It is possible to use multiple center_fractions and accelerations, in which case one possible
    (center_fraction, acceleration) is chosen uniformly at random each time the MaskFunc object is
    called.
    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04], then there
    is a 50% probability that 4-fold acceleration with 8% center fraction is selected and a 50%
    probability that 8-fold acceleration with 4% center fraction is selected.
    """

    def __init__(self, center_fractions, accelerations):
        """
        Args:
            center_fractions (List[float]): Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is chosen uniformly
                each time.
            accelerations (List[int]): Amount of under-sampling. This should have the same length
                as center_fractions. If multiple values are provided, then one of these is chosen
                uniformly each time. An acceleration of 4 retains 25% of the columns, but they may
                not be spaced evenly.
        """
        if len(center_fractions) != len(accelerations):
            raise ValueError('Number of center fractions should match number of accelerations')

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.rng = np.random.RandomState()

    def __call__(self, shape, seed=None):
        """
        Args:
            shape (iterable[int]): The shape of the mask to be created. The shape should have
                at least 3 dimensions. Samples are drawn along the second last dimension.
            seed (int, optional): Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same shape.
        Returns:
            torch.Tensor: A mask of the specified shape.
        """
        if len(shape) < 3:
            raise ValueError('Shape should have 3 or more dimensions')

        self.rng.seed(seed)
        num_cols = shape[-2]

        choice = self.rng.randint(0, len(self.accelerations))
        center_fraction = self.center_fractions[choice]
        acceleration = self.accelerations[choice]

        # Create the mask
        num_low_freqs = int(round(num_cols * center_fraction))
        prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
        mask = self.rng.uniform(size=num_cols) < prob
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad:pad + num_low_freqs] = True

        # Reshape the mask
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols
        mask = mask.reshape(*mask_shape).astype(np.float32)

        return mask