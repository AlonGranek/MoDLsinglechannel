import os
import logging
# import random
# import h5py
# import shutil
# import time
# import argparse
# import sigpy.plot as pl
import torch
# import sigpy as sp
# from torch import optim
# from tensorboardX import SummaryWriter
# from torch.nn import functional as F
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import matplotlib
# # import custom libraries
# from utils import transforms as T
# from utils import subsample as ss
# from utils.resnet2p1d import generate_model
# from utils.flare_utils import roll
# # import custom classes
# from subsample_fastmri import MaskFunc
# import argparse
# from dataclasses import dataclass


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


from datasets import create_data_loaders
from subsample_fastmri import SaveableMask
from modl_infrastructure import MoDLParams, MoDLWrapper

from alon.config import *


# Load a mask
#   Ask for a UUID, else take the most recently-created one
mask_name = input('Enter mask name (if skipping, taking the most recently created): ')
if not mask_name.replace(' ', ''):
    latest_file = max(Path(MASKS_DIR).glob('*.pkl'), key=os.path.getctime)
    mask_name = latest_file.stem
    print(f'User skipped, thus loaded the latest mask {mask_name}')

mask = SaveableMask(masks_dir=MASKS_DIR)
mask.load(mask_name)


""" Loading """
modl_params = MoDLParams()
modl_train_loader = create_data_loaders(modl_params, mask)
modl_checkpoint_dir = Path('/home/alon_granek/PythonProjects/NPPC/alon/checkpoints2')
modl_wrapper = MoDLWrapper(modl_params, modl_checkpoint_dir)

single_MoDL = modl_wrapper.load(modl_checkpoint_dir,
                                model_name='MoDL 4-step regul 0.01', #'Initial 4-step MoDL (2)', #'8-step MoDL', #'test',
                                epoch='last')


from MoDLsinglechannel.demo_modl_singlechannel.nppc_infrastructure import NPPCParams, NPPCForMoDLWrapper

nppc_params = NPPCParams(
    dc_loss_lambda=0, #20e0, #5e0, #8e0,
#    n_dirs=8,

    #5e0, #0
    # dc_loss_lambda=0, #5e0, #0
)
nppc_checkpoint_dir = Path('/home/alon_granek/PythonProjects/NPPC/alon/nppc_checkpoints')
nppc_wrapper = NPPCForMoDLWrapper(nppc_params, single_MoDL, save_dir=nppc_checkpoint_dir)


# """ MERE ATTEMPT IN END-TO-END NPPC """
# nppc_wrapper.train(modl_train_loader, nppc_position=0, last_pos=-1, model_name=f'modl_nppc_dc END-TO-END DC test')
# """"""

""" DUAL-CHANNEL END-TO-END NPPC """
nppc_wrapper.train(modl_train_loader, nppc_position=0, last_pos=-1, model_name=f'modl_nppc_dc DUAL AlterProj')
""""""

