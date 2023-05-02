import torch
from vqgan.utils import shift_dim

# test tensor shift function
torch.random.manual_seed(1)
# GIVEN
x = torch.rand(2, 3, 5)
y = shift_dim(x, 1, -1)

# THEN
assert x.size() == torch.Size([2, 3, 5])
assert y.size() == torch.Size([2, 5, 3])