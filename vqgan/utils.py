# taken & adapted from Time-Agnostic VQGAN and Time-Sensitive Transformer (ECCV 2022)
# https://github.com/SongweiGe/

import warnings
import torch
import imageio

import math
import numpy as np
import skvideo.io

import sys
import pdb as pdb_original
import SimpleITK as sitk
import logging

import imageio.core.util
logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)

# shifts source dimension to destination dimension. 
# source 1, dest -1 is analogous to (batch, channel, time, height, width) -> (batch, time, height, width, channel)
def shift_dim(x, src_dim, dest_dim, make_contiguous):
  """
  Swaps source dimension with destination dimension. 
  Used to shift and flatten initial embeddings in the VQGAN codebook and to 
  pass codebook embeddings back to the decoder of the VQGAN.
  """
  pass


def adopt_weight(global_step, threshold, value):
  pass


def save_video_grid(video, fname, nrow, fps):
  """
  Helper function to save a video grid. 
  Used as a callback during training.
  """
  pass

def comp_getattr(args, attr_name, default=None):
  pass


def visualize_tensors(t, name=None, nest=0):
  """
  Helper function to pretty print nested tensors for debugging purposes.
  """
  pass