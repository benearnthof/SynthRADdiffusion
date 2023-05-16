import torch
from vqgan.utils import shift_dim, adopt_weight
from vqgan.model.vqgan import ResBlock

### test tensor shift function
torch.random.manual_seed(1)
# GIVEN
x = torch.rand(2, 3, 5)
y = shift_dim(x, 1, -1)

# THEN
assert x.size() == torch.Size([2, 3, 5])
assert y.size() == torch.Size([2, 5, 3])

# test adopt_weight to stall discriminator training
# GIVEN
global_step = 9999
threshold = 10_000
value = 0.

# WHEN
result = adopt_weight(global_step, threshold, value)

# THEN
assert result == value

### test ResBlock Shape
# GIVEN
resblock = ResBlock(160)
x = torch.zeros([160, 160, 160, 1])

# WHEN
out = resblock(x)

# THEN
assert out.shape == torch.Size([160, 160, 160, 1])