# taken & adapted from Time-Agnostic VQGAN and Time-Sensitive Transformer (ECCV 2022)
# https://github.com/SongweiGe/

import warnings
import torch
import imageio

import math
import numpy as np

import sys
import pdb as pdb_original
import logging

import imageio.core.util
logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)


def shift_dim(x, src_dim=-1, dest_dim=-1, make_contiguous=True):
  """
  Swaps source dimension with destination dimension. 
  Used to shift and flatten initial embeddings in the VQGAN codebook and to 
  pass codebook embeddings back to the decoder of the VQGAN.
  source 1, dest -1 is analogous to:
  (batch, channel, time, height, width) -> (batch, time, height, width, channel)
  """
  n_dims = len(x.shape)
  if src_dim < 0:
    src_dim = n_dims + src_dim
  if dest_dim < 0:
    dest_dim = n_dims + dest_dim

  assert 0 <= src_dim < n_dims and 0 <= dest_dim < n_dims

  dims = list(range(n_dims))
  del dims[src_dim]

  permutation = []
  ctr = 0
  for i in range(n_dims):
    if i == dest_dim:
        permutation.append(src_dim)
    else:
        permutation.append(dims[ctr])
        ctr += 1
  x = x.permute(permutation)
  if make_contiguous:
    x = x.contiguous()
  return x  


def adopt_weight(global_step, threshold=0, value=0.):
  """
  Used to obtain a factor used to scale the discriminator loss of the VQGAN:
  disc_factor = adopt_weight(global_step, threshold=discriminator_iter_start)
  disc_loss = disc_factor * (1 * d_image_loss + 1 * d_video_loss)
  global_step is the number of total training batches seen across all epochs
  We give the generator a head start to stabilize training.
  Discriminator_iter_start should be set between 10_000 and 50_000
  Refer to the .yaml files in /config/model for information on hyperparameters.
  """
  weight = 1
  if global_step < threshold:
    weight = value
  return weight

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


