"""Adapted from https://github.com/SongweiGe/TATS"""
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import math
import argparse
import numpy as np
import pickle as pkl

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from synthraddiffusion.vqgan.utils import shift_dim, adopt_weight  # , comp_getattr
from synthraddiffusion.vqgan.model.lpips import LPIPS
from synthraddiffusion.vqgan.model.codebook import Codebook


def silu(x):
    return x * torch.sigmoid(x)


class SiLU(nn.Module):
    """
    Sigmoid Linear Unit as presented in
    https://arxiv.org/pdf/1710.05941.pdf
    """

    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return silu(x)


def hinge_d_loss(logits_real, logits_fake):
    """
    Hinge loss for discriminators, Uses ReLU as hinge function.
    """
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    """
    Alternative for discriminator loss used in experiments.
    """
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real))
        + torch.mean(torch.nn.functional.softplus(logits_fake))
    )
    return d_loss


class VQGAN(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg  # obtained from config.yaml
        self.embedding_dim = cfg.model.embedding_dim  # 256
        self.n_codes = cfg.model.n_codes  # 2048

        self.encoder = Encoder(
            cfg.model.n_hiddens,  # 240
            cfg.model.downsample,  # [4, 4, 4]
            cfg.dataset.image_channels,  # 1, depending on dataset of course
            cfg.model.norm_type,  # group
            cfg.model.padding_type,  # replicate
            cfg.model.num_groups,  # 32
        )

        self.decoder = Decoder(
            cfg.model.n_hiddens,  # 240
            cfg.model.downsample,  # [4, 4, 4]
            cfg.dataset.image_channels,  # 1, depending on dataset of course
            cfg.model.norm_type,  # group
            cfg.model.num_groups,  # 32
        )
        self.enc_out_ch = self.encoder.out_channels  # 960
        # pre and post quantization conv layers. Used to learn a mapping from the quantized to the unquantized image and vice versa
        self.pre_vq_conv = SamePadConv3d(
            self.enc_out_ch, self.embedding_dim, 1, padding_type=cfg.model.padding_type
        )
        self.post_vq_conv = SamePadConv3d(self.embedding_dim, self.enc_out_ch, 1)
        # no_random_restart=False, restart_thres = 1.0
        self.codebook = Codebook(
            self.n_codes,
            self.embedding_dim,
            no_random_restart=cfg.model.no_random_restart,
            restart_thres=cfg.model.restart_thres,
        )

        self.gan_feat_weight = cfg.model.gan_feat_weight
        # TODO: Changed batchnorm from sync to normal
        self.image_discriminator = NLayerDiscriminator(
            cfg.dataset.image_channels,
            cfg.model.disc_channels,
            cfg.model.disc_layers,
            norm_layer=nn.BatchNorm2d,
        )
        self.video_discriminator = NLayerDiscriminator3D(
            cfg.dataset.image_channels,
            cfg.model.disc_channels,
            cfg.model.disc_layers,
            norm_layer=nn.BatchNorm3d,
        )

        if cfg.model.disc_loss_type == "vanilla":
            self.disc_loss = vanilla_d_loss
        elif cfg.model.disc_loss_type == "hinge":
            self.disc_loss = hinge_d_loss
        # TODO: Cache vgg16 weights to avoid having to redownload every time
        self.perceptual_model = LPIPS().eval()
        # TODO: Add calculate_lambda function that adjusts these weights during training (may be unstable)
        self.image_gan_weight = cfg.model.image_gan_weight  # 1
        self.video_gan_weight = cfg.model.video_gan_weight  # 1
        self.perceptual_weight = cfg.model.perceptual_weight  # 4
        self.l1_weight = cfg.model.l1_weight  # 4
        self.save_hyperparameters()

    def encode(self, x, include_embeddings=False, quantize=True):
        # encode first then apply pre quant convolution, then quantize
        h = self.pre_vq_conv(self.encoder(x))
        if quantize:
            vq_output = self.codebook(h)
            if include_embeddings:
                return vq_output["embeddings"], vq_output["encodings"]
            else:
                return vq_output["encodings"]
        return h

    def decode(self, latent, quantize=False):
        if quantize:
            vq_output = self.codebook(latent)
            latent = vq_output["encodings"]
        h = F.embedding(latent, self.codebook.embeddings)
        h = self.post_vq_conv(shift_dim(h, -1, 1))
        return self.decoder(h)

    def forward(self, x, optimizer_idx=None, log_image=False):
        B, C, T, H, W = x.shape

        z = self.pre_vq_conv(self.encoder(x))
        vq_output = self.codebook(z)
        x_recon = self.decoder(self.post_vq_conv(vq_output["embeddings"]))

        recon_loss = F.l1_loss(x_recon, x) * self.l1_weight

        # Selects one random 2D image from each 3D Image
        frame_idx = torch.randint(0, T, [B]).cuda()
        frame_idx_selected = frame_idx.reshape(-1, 1, 1, 1, 1).repeat(1, C, 1, H, W)
        frames = torch.gather(x, 2, frame_idx_selected).squeeze(2)
        frames_recon = torch.gather(x_recon, 2, frame_idx_selected).squeeze(2)

        if log_image:
            return frames, frames_recon, x, x_recon

        if optimizer_idx == 0:
            # Autoencoder - train the "generator"
            # Perceptual loss
            perceptual_loss = 0
            if self.perceptual_weight > 0:
                perceptual_loss = (
                    self.perceptual_model(frames, frames_recon).mean()
                    * self.perceptual_weight
                )

            # Discriminator loss (0 until discriminator_iter_start iterations have passed)
            logits_image_fake, pred_image_fake = self.image_discriminator(frames_recon)
            logits_video_fake, pred_video_fake = self.video_discriminator(x_recon)
            g_image_loss = -torch.mean(logits_image_fake)
            g_video_loss = -torch.mean(logits_video_fake)
            g_loss = (
                self.image_gan_weight * g_image_loss
                + self.video_gan_weight * g_video_loss
            )
            # TODO: Do more efficiently: only calculate weights if disc_factor > 0
            disc_factor = adopt_weight(
                self.global_step, threshold=self.cfg.model.discriminator_iter_start
            )
            aeloss = disc_factor * g_loss

            # GAN feature matching loss - tune features such that we get the same prediction result on the discriminator
            image_gan_feat_loss = 0
            video_gan_feat_loss = 0
            feat_weights = 4.0 / (3 + 1)
            if self.image_gan_weight > 0:
                logits_image_real, pred_image_real = self.image_discriminator(frames)
                for i in range(len(pred_image_fake) - 1):
                    image_gan_feat_loss += (
                        feat_weights
                        * F.l1_loss(pred_image_fake[i], pred_image_real[i].detach())
                        * (self.image_gan_weight > 0)
                    )
            if self.video_gan_weight > 0:
                logits_video_real, pred_video_real = self.video_discriminator(x)
                for i in range(len(pred_video_fake) - 1):
                    video_gan_feat_loss += (
                        feat_weights
                        * F.l1_loss(pred_video_fake[i], pred_video_real[i].detach())
                        * (self.video_gan_weight > 0)
                    )

            gan_feat_loss = (
                disc_factor
                * self.gan_feat_weight
                * (image_gan_feat_loss + video_gan_feat_loss)
            )

            self.log(
                "train/g_image_loss",
                g_image_loss,
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            self.log(
                "train/g_video_loss",
                g_video_loss,
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            self.log(
                "train/image_gan_feat_loss",
                image_gan_feat_loss,
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            self.log(
                "train/video_gan_feat_loss",
                video_gan_feat_loss,
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            self.log(
                "train/perceptual_loss",
                perceptual_loss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            self.log(
                "train/recon_loss",
                recon_loss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            self.log(
                "train/aeloss",
                aeloss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            self.log(
                "train/commitment_loss",
                vq_output["commitment_loss"],
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            self.log(
                "train/perplexity",
                vq_output["perplexity"],
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            return (
                recon_loss,
                x_recon,
                vq_output,
                aeloss,
                perceptual_loss,
                gan_feat_loss,
            )

        if optimizer_idx == 1:
            # discriminator
            logits_image_real, _ = self.image_discriminator(frames.detach())
            logits_video_real, _ = self.video_discriminator(x.detach())

            logits_image_fake, _ = self.image_discriminator(frames_recon.detach())
            logits_video_fake, _ = self.video_discriminator(x_recon.detach())

            d_image_loss = self.disc_loss(logits_image_real, logits_image_fake)
            d_video_loss = self.disc_loss(logits_video_real, logits_video_fake)
            disc_factor = adopt_weight(
                self.global_step, threshold=self.cfg.model.discriminator_iter_start
            )
            discloss = disc_factor * (
                self.image_gan_weight * d_image_loss
                + self.video_gan_weight * d_video_loss
            )

            self.log(
                "train/logits_image_real",
                logits_image_real.mean().detach(),
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            self.log(
                "train/logits_image_fake",
                logits_image_fake.mean().detach(),
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            self.log(
                "train/logits_video_real",
                logits_video_real.mean().detach(),
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            self.log(
                "train/logits_video_fake",
                logits_video_fake.mean().detach(),
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            self.log(
                "train/d_image_loss",
                d_image_loss,
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            self.log(
                "train/d_video_loss",
                d_video_loss,
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            self.log(
                "train/discloss",
                discloss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            return discloss

        perceptual_loss = (
            self.perceptual_model(frames, frames_recon) * self.perceptual_weight
        )
        return recon_loss, x_recon, vq_output, perceptual_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = batch["data"]
        if optimizer_idx == 0:
            (
                recon_loss,
                _,
                vq_output,
                aeloss,
                perceptual_loss,
                gan_feat_loss,
            ) = self.forward(x, optimizer_idx)
            commitment_loss = vq_output["commitment_loss"]
            loss = (
                recon_loss + commitment_loss + aeloss + perceptual_loss + gan_feat_loss
            )
        if optimizer_idx == 1:
            discloss = self.forward(x, optimizer_idx)
            loss = discloss
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["data"]
        recon_loss, _, vq_output, perceptual_loss = self.forward(x)
        self.log("val/recon_loss", recon_loss, prog_bar=True)
        self.log("val/perceptual_loss", perceptual_loss, prog_bar=True)
        self.log("val/perplexity", vq_output["perplexity"], prog_bar=True)
        self.log("val/commitment_loss", vq_output["commitment_loss"], prog_bar=True)

    def configure_optimizers(self):
        lr = self.cfg.model.lr
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.pre_vq_conv.parameters())
            + list(self.post_vq_conv.parameters())
            + list(self.codebook.parameters()),
            lr=lr,
            betas=(0.5, 0.9),
        )
        opt_disc = torch.optim.Adam(
            list(self.image_discriminator.parameters())
            + list(self.video_discriminator.parameters()),
            lr=lr,
            betas=(0.5, 0.9),
        )
        return [opt_ae, opt_disc], []

    def log_images(self, batch, **kwargs):
        log = dict()
        x = batch["data"]
        x = x.to(self.device)
        frames, frames_rec, _, _ = self(x, log_image=True)
        log["inputs"] = frames
        log["reconstructions"] = frames_rec
        return log

    def log_videos(self, batch, **kwargs):
        log = dict()
        x = batch["data"]
        _, _, x, x_rec = self(x, log_image=True)
        log["inputs"] = x
        log["reconstructions"] = x_rec
        return log


def Normalize(in_channels, norm_type="group", num_groups=32):
    """
    Allows us to switch between Batch Normalization for Images and Group Normalization for videos/3D data.
    In a nutshell, BNs error increases rapidly for shrinking batch sizes, because of larger inaccuracy in
    Batch statistic estimation. Because of the high memory demands for video/3D data we need small batches however.
    GN splits channels into groups and computes means and variances for normalization within each group.
    This is independent of batch size, thus more stable than BN for small batch sizes.
    https://arxiv.org/pdf/1803.08494.pdf
    """
    assert norm_type in ["group", "batch"]
    if norm_type == "group":
        return torch.nn.GroupNorm(
            num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
        )
    elif norm_type == "batch":
        return torch.nn.SyncBatchNorm(in_channels)


class Encoder(nn.Module):
    """
    Encodes Images into a latent representation.
    Has less model capacity than the decoder, used to obtain the conditioning vectors we need to pass to the DDPM.
    """

    def __init__(
        self,
        n_hiddens,
        downsample,
        image_channel=3,
        norm_type="group",
        padding_type="replicate",
        num_groups=32,
    ):
        super().__init__()
        n_times_downsample = np.array([int(math.log2(d)) for d in downsample])
        self.conv_blocks = nn.ModuleList()
        max_ds = n_times_downsample.max()

        self.conv_first = SamePadConv3d(
            image_channel, n_hiddens, kernel_size=3, padding_type=padding_type
        )

        for i in range(max_ds):
            block = (
                nn.Module()
            )  # Every Block has one downsampling Convolution followed by a ResBlock
            in_channels = (
                n_hiddens * 2**i
            )  # hidden dim defined in config * 2^downsample iterations)
            out_channels = n_hiddens * 2 ** (i + 1)
            stride = tuple(
                [2 if d > 0 else 1 for d in n_times_downsample]
            )  # tuple of strides for 3d conv, equals 2 for every axis if we downsample
            block.down = SamePadConv3d(
                in_channels, out_channels, 4, stride=stride, padding_type=padding_type
            )
            block.res = ResBlock(
                out_channels, out_channels, norm_type=norm_type, num_groups=num_groups
            )
            self.conv_blocks.append(block)
            n_times_downsample -= 1

        self.final_block = nn.Sequential(
            # Final Normalization & Activation layers
            Normalize(out_channels, norm_type, num_groups=num_groups),
            SiLU(),
        )

        self.out_channels = out_channels

    def forward(self, x):
        h = self.conv_first(x)
        for block in self.conv_blocks:
            h = block.down(h)
            h = block.res(h)
        h = self.final_block(h)
        return h


class Decoder(nn.Module):
    """
    Mirror Image of Encoder, swaps downsampling with upsampling layers.
    Has one ResBlock more than the Encoder, may need more model capacity for better performance.
    """

    def __init__(
        self, n_hiddens, upsample, image_channel, norm_type="group", num_groups=32
    ):
        super().__init__()

        n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        max_us = n_times_upsample.max()

        in_channels = n_hiddens * 2**max_us
        self.final_block = nn.Sequential(
            Normalize(in_channels, norm_type, num_groups=num_groups), SiLU()
        )

        self.conv_blocks = nn.ModuleList()
        for i in range(max_us):
            block = nn.Module()
            in_channels = in_channels if i == 0 else n_hiddens * 2 ** (max_us - i + 1)
            out_channels = n_hiddens * 2 ** (max_us - i)
            us = tuple([2 if d > 0 else 1 for d in n_times_upsample])
            block.up = SamePadConvTranspose3d(in_channels, out_channels, 4, stride=us)
            block.res1 = ResBlock(
                out_channels, out_channels, norm_type=norm_type, num_groups=num_groups
            )
            block.res2 = ResBlock(
                out_channels, out_channels, norm_type=norm_type, num_groups=num_groups
            )
            self.conv_blocks.append(block)
            n_times_upsample -= 1

        self.conv_last = SamePadConv3d(out_channels, image_channel, kernel_size=3)

    def forward(self, x):
        h = self.final_block(x)
        for i, block in enumerate(self.conv_blocks):
            h = block.up(h)
            h = block.res1(h)
            h = block.res2(h)
        h = self.conv_last(h)
        return h


class ResBlock(nn.Module):
    """
    Residual Block for VQGAN Encoder & Decoder
    Uses GroupNorm & Padding to keep tensors in correct shape
    """

    def __init__(
        self,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        norm_type="group",
        padding_type="replicate",
        num_groups=32,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.block = nn.Sequential(
            Normalize(in_channels, norm_type, num_groups=num_groups),
            SiLU(),
            SamePadConv3d(
                in_channels, out_channels, kernel_size=3, padding_type=padding_type
            ),
            Normalize(in_channels, norm_type, num_groups=num_groups),
            SiLU(),
            SamePadConv3d(
                out_channels, out_channels, kernel_size=3, padding_type=padding_type
            ),
        )

        if self.in_channels != self.out_channels:
            # Channel up in case of dimensionality mismatch
            self.conv_shortcut = SamePadConv3d(
                in_channels, out_channels, kernel_size=3, padding_type=padding_type
            )

    def forward(self, x):
        if self.in_channels != self.out_channels:
            return self.conv_shortcut(x) + self.block(x)
        else:
            return x + self.block(x)


# Does not support dilation
class SamePadConv3d(nn.Module):
    """
    3D Conv layer with added padding used in ResBlock & For Input Downsampling in the Encoder.
    Also Used as final layer of the decoder.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=True,
        padding_type="replicate",
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        # assumes that the input shape is divisible by stride
        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        self.padding_type = padding_type

        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=bias
        )

    def forward(self, x):
        return self.conv(F.pad(x, self.pad_input, mode=self.padding_type))


class SamePadConvTranspose3d(nn.Module):
    """
    3D ConvTranspose layer with added padding that is used for upsampling in decoder.
    Kernel Size will be set to 4, stride as 1 or 2 depending on position in network.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=True,
        padding_type="replicate",
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        self.padding_type = padding_type

        self.convt = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            bias=bias,
            padding=tuple([k - 1 for k in kernel_size]),
        )

    def forward(self, x):
        return self.convt(F.pad(x, self.pad_input, mode=self.padding_type))


class NLayerDiscriminator(nn.Module):
    def __init__(
        self,
        input_nc,
        ndf=64,
        n_layers=3,
        norm_layer=nn.SyncBatchNorm,
        use_sigmoid=False,
        getIntermFeat=True,
    ):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [
            [
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True),
            ]
        ]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [
                [
                    nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                    norm_layer(nf),
                    nn.LeakyReLU(0.2, True),
                ]
            ]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [
            [
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True),
            ]
        ]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, "model" + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, "model" + str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input), _


class NLayerDiscriminator3D(nn.Module):
    def __init__(
        self,
        input_nc,
        ndf=64,
        n_layers=3,
        norm_layer=nn.SyncBatchNorm,
        use_sigmoid=False,
        getIntermFeat=True,
    ):
        super(NLayerDiscriminator3D, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [
            [
                nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True),
            ]
        ]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [
                [
                    nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                    norm_layer(nf),
                    nn.LeakyReLU(0.2, True),
                ]
            ]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [
            [
                nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True),
            ]
        ]

        sequence += [[nn.Conv3d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, "model" + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, "model" + str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input), _
