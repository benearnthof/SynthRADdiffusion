"""
Training wrapper for the 3d VQGAN that is used to obtain embeddings for the MRI data. 
"""

import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from synthraddiffusion.ddpm.diffusion import default
from synthraddiffusion.vqgan.model.vqgan import VQGAN

from synthraddiffusion.train.callbacks import ImageLogger, VideoLogger
from synthraddiffusion.train.get_dataset import get_dataset
