from ddpm.diffusion import RelativePositionBias
import torch

rpb = RelativePositionBias()

assert rpb.forward(0, device="cpu").shape == torch.Size([8, 0, 0])
assert rpb.forward(1, device="cpu").shape == torch.Size([8, 1, 1])
assert rpb.forward(128, device="cpu").shape == torch.Size([8, 128, 128])

rpb4 = RelativePositionBias(heads=4)
assert rpb.forward(0, device="cpu").shape == torch.Size([4, 0, 0])
assert rpb.forward(1, device="cpu").shape == torch.Size([4, 1, 1])
assert rpb.forward(128, device="cpu").shape == torch.Size([4, 128, 128])