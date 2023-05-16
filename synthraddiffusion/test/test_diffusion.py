from ..ddpm.diffusion import RelativePositionBias
import torch

rpb = RelativePositionBias()

assert rpb.forward(0, device="cpu").shape == torch.Size([8, 0, 0])
assert rpb.forward(1, device="cpu").shape == torch.Size([8, 1, 1])
assert rpb.forward(128, device="cpu").shape == torch.Size([8, 128, 128])

rpb4 = RelativePositionBias(heads=4)
assert rpb.forward(0, device="cpu").shape == torch.Size([4, 0, 0])
assert rpb.forward(1, device="cpu").shape == torch.Size([4, 1, 1])
assert rpb.forward(128, device="cpu").shape == torch.Size([4, 128, 128])

# GIVEN
from ..ddpm.diffusion import Upsample, Downsample

us = Upsample(1)
ds = Downsample(1)
dsc = Downsample(3)
# WHEN
image = torch.ones(1, 1, 4, 4, 4)  # batch, channel, depth, height, width
images = torch.ones(3, 1, 4, 4, 4)  # three images
images_3c = torch.ones(3, 3, 4, 4, 4)  # three images with three channels
usamp = us(image)
dsamp = ds(image)
dsamp_channels = dsc(images_3c)
# THEN
assert usamp.shape == torch.Size([1, 1, 4, 8, 8])
assert dsamp.shape == torch.Size([1, 1, 4, 2, 2])
assert dsamp_channels.shape == torch.Size([3, 3, 4, 2, 2])

#### Test Block
# GIVEN
from ..ddpm.diffusion import Block, ResnetBlock

blk = Block(3, 1, groups=1)  # 3 in channels, one out channel
# WHEN
images = torch.ones(3, 3, 4, 4, 4)  # three images with three channels
result = blk(images)
# THEN
assert result.shape == torch.Size([3, 1, 4, 4, 4])  # one out channel
