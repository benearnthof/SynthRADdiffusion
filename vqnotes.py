mod = run(cfg) # run as in train_vqgan.py
# n_hiddens = 240 vs 64
n_params = sum(p.numel() for p in mod.parameters())  # 88 million vs 32 million
n_encoder = sum(
    p.numel() for p in mod.encoder.parameters()
)  # 19.8 million vs 1.4 million
n_decoder = sum(p.numel() for p in mod.decoder.parameters())  # 39.6 million
n_codebook = sum(p.numel() for p in mod.codebook.buffers())  # 1 million
n_image_disc = sum(
    p.numel() for p in mod.image_discriminator.parameters()
)  # 2.7 million
n_video_disc = sum(
    p.numel() for p in mod.video_discriminator.parameters()
)  # 11 million
n_perceptual_model = sum(
    p.numel() for p in mod.perceptual_model.parameters()
)  # 14 million

# calculating expected size of model in memory
param_size = 0
for param in mod.parameters():
    param_size += param.nelement() * param.element_size()

buffer_size = 0
for buffer in mod.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

total_size = (param_size + buffer_size) / (1024**2)
print("model size: {:.3f}MB".format(total_size))

from synthraddiffusion.vqgan.model.codebook import Codebook

cb = Codebook(n_codes=2048, embedding_dim=256)  # 32 * 32
# how do we find out the optimal number of codes and embedding dim?
# we don't have that much training data but could get more via ADNI
# 32000 embeddings seems overkill since we would obtain one distinct embedding
# for every image
# for the embedding dim 256 seems fine
# do we need to improve codebook usage? https://arxiv.org/pdf/2110.04627.pdf
# https://github.com/lucidrains/DALLE2-pytorch/blob/main/dalle2_pytorch/vqgan_vae.py

n_cb = sum(p.numel() for p in cb.buffers())

# VQGAN forward: self.encoder(x) is the first computation performed on x
# z = self.pre_vq_conv(self.encoder(x))
# encoder forward method: first step performed:
# self.conv_first(x)
conv_first = mod.encoder.conv_first
# Conv3d(1, 240, kernel_size=(3, 3, 3), stride=(1, 1, 1))
# in_ch, out_ch, kernel_size, stride
input = torch.randn(1, 1, 128, 128, 128)  # batch size 1, single channel images
B, C, T, H, W = input.shape
output = conv_first(input)

enc_out = mod.encoder(input)  # 1, 128, 64, 64, 64
# downsmaple 4,4,4 with 256**3 inputs results in 1, 256, 64, 64, 64 encodings
# with n_hiddens = 64

z = mod.pre_vq_conv(enc_out)
vq_output = mod.codebook(z)
vq_output["embeddings"].shape # 1, 256, 64, 64, 64
post_vq_output = mod.post_vq_conv(vq_output["embeddings"])
x_recon = mod.decoder(post_vq_output) # 1, 1, 128, 128, 128 for down 4,4,4 & 64 n hiddens
# to adjust the power of the vqgan we can both adjust downsampling to 2,2,2 or increase number of hidden channels



# problem: the intermediate tensors blow up to insane proportions.
# 1, 1, 128 ** 3 input results in 1, 240, 128, 128, 128 intermediate tensor
# after first convolution and
# [1, 480, 64, 64, 64] encoder output => 125 million bytes for single channel image

# adjusting n_hiddens from 240 to 64 yields:
# 1, 128, 64, 64, 64
# as shape of the encoder output

# encoder arch with downsample = [2, 2, 2]:
Encoder(
  (conv_blocks): ModuleList(
    (0): Module(
      (down): SamePadConv3d(
        (conv): Conv3d(64, 128, kernel_size=(4, 4, 4), stride=(2, 2, 2))
      )
      (res): ResBlock(
        (block): Sequential(
          (0): GroupNorm(32, 128, eps=1e-06, affine=True)
          (1): SiLU()
          (2): SamePadConv3d(
            (conv): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1))
          )
          (3): GroupNorm(32, 128, eps=1e-06, affine=True)
          (4): SiLU()
          (5): SamePadConv3d(
            (conv): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1))
          )
        )
      )
    )
  )
  (conv_first): SamePadConv3d(
    (conv): Conv3d(1, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1))
  )
  (final_block): Sequential(
    (0): GroupNorm(32, 128, eps=1e-06, affine=True)
    (1): SiLU()
  )
)

# encoder arch with downsample = [4, 4, 4]
Encoder(
  (conv_blocks): ModuleList(
    (0): Module(
      (down): SamePadConv3d(
        (conv): Conv3d(64, 128, kernel_size=(4, 4, 4), stride=(2, 2, 2))
      )
      (res): ResBlock(
        (block): Sequential(
          (0): GroupNorm(32, 128, eps=1e-06, affine=True)
          (1): SiLU()
          (2): SamePadConv3d(
            (conv): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1))
          )
          (3): GroupNorm(32, 128, eps=1e-06, affine=True)
          (4): SiLU()
          (5): SamePadConv3d(
            (conv): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1))
          )
        )
      )
    )
    (1): Module(
      (down): SamePadConv3d(
        (conv): Conv3d(128, 256, kernel_size=(4, 4, 4), stride=(2, 2, 2))
      )
      (res): ResBlock(
        (block): Sequential(
          (0): GroupNorm(32, 256, eps=1e-06, affine=True)
          (1): SiLU()
          (2): SamePadConv3d(
            (conv): Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1))
          )
          (3): GroupNorm(32, 256, eps=1e-06, affine=True)
          (4): SiLU()
          (5): SamePadConv3d(
            (conv): Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1))
          )
        )
      )
    )
  )
  (conv_first): SamePadConv3d(
    (conv): Conv3d(1, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1))
  )
  (final_block): Sequential(
    (0): GroupNorm(32, 256, eps=1e-06, affine=True)
    (1): SiLU()
  )
)


# TODO: set num_workers to 32 if possible
# TODO: discriminator_iter_start 10k for now 50k may be needed?
