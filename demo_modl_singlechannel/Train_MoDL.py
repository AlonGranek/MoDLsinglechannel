import os, sys
import logging
# import random
# import h5py
# import shutil
# import time
# import argparse
import numpy as np
# import sigpy.plot as pl
import torch
# import sigpy as sp
import torchvision
# from torch import optim
# from tensorboardX import SummaryWriter
# from torch.nn import functional as F
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import matplotlib
# # import custom libraries
# from utils import transforms as T
# from utils import subsample as ss
from utils import complex_utils as cplx
# from utils.resnet2p1d import generate_model
# from utils.flare_utils import roll
# # import custom classes
from utils.datasets import SliceData
# from subsample_fastmri import MaskFunc
from MoDL_single import UnrolledModel
# import argparse
# from dataclasses import dataclass
from pathlib import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



from datasets import Namespace, create_data_loaders
from subsample_fastmri import MaskFuncGiven, MaskFuncPoisson2D, MaskFuncEquispacedLines

# Create a 2D Poisson mask, and freeze it
mask_generator = MaskFuncGiven(
    MaskFuncPoisson2D(
        accel=6, #14, #6
        calib_shape=(28, 28) #(11, 11),
    )()
)



from modl_infrastructure import MoDLParams, MoDLWrapper

modl_params = MoDLParams()
modl_train_loader = create_data_loaders(modl_params, mask_generator)
modl_checkpoint_dir = Path('/home/alon_granek/PythonProjects/NPPC/alon/checkpoints2')
modl_wrapper = MoDLWrapper(modl_params, modl_checkpoint_dir)



""" Training """
modl_wrapper.train(modl_train_loader, model_name='8-step MoDL') #save_dir = ....)



