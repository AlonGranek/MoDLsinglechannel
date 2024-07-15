"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pathlib
import random

import h5py
import numpy as np
from torch.utils.data import Dataset
from alon.fastmri_preprocess import fftc, resize_image


class SliceData(Dataset):
    """
    A generic PyTorch Dataset class that provides access to 2D MR image slices.
    """

    def __init__(self, root, transform, sample_rate=1):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """

        self.transform = transform

        self.examples = []
        files = list(pathlib.Path(root).iterdir())
        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]
        for fname in sorted(files):
            print(f'\t{fname}...')
            # kspace = h5py.File(fname, 'r')['kspace']
            # print(f'\t\t{kspace.shape}')
            # num_slices = kspace.shape[0]
            self.examples += [fname] #[(fname, 3)]#[(fname, slice+8) for slice in range(num_slices-16)]

    def __len__(self):
        return len(self.examples)

    # def __getitem__(self, i):
    #     fname, slice = self.examples[i]
    #     # print(f'Loading {fname}...')
    #     with h5py.File(fname, 'r') as data:
    #         kspace = data['kspace'][slice]
    #         target = data['reconstruction'][slice]
    #         # target = data['reconstruction_rss'][slice]
    #
    #         #todo By alon: Perhaps do the processing here
    #
    #         return self.transform(kspace,target,slice)

    def __getitem__(self, item):
        # fname, slic = self.examples[item]
        fname = self.examples[item]
        data = np.load(fname, allow_pickle=True).tolist()
        target_origsize = data['reconstruction']
        target = resize_image(target_origsize, (372, 372))

        #todo Feed the SliceData object with a phase map (or a phase map function that receives the magnitude image)
        #   so that it could be included upon __getitem__()

        # k-space as simple Fourier
        kspace = fftc(target)
        return self.transform(kspace, target) #, slic)




