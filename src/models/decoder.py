"""Attention-based stage-1 decoder."""

import torch.nn as nn
import torch.nn.functional as F

from .encoder import AttnBlock, Normalize, ResnetBlock, nonlinearity
from .encoder import SimpleResidualStack


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class SimpleDecoder(nn.Module):
    """Lightweight VQ-VAE-style convolutional decoder."""

    def __init__(
        self,
        in_channels,
        num_hiddens,
        num_residual_layers=None,
        num_residual_hiddens=None,
        num_channels=3,
        num_residual_blocks=None,
        out_channels=None,
        num_upsamples=2,
    ):
        super().__init__()
        if num_residual_layers is None:
            num_residual_layers = num_residual_blocks
        if out_channels is not None:
            num_channels = out_channels
        if num_residual_layers is None:
            raise ValueError("num_residual_layers or num_residual_blocks must be provided")
        if num_residual_hiddens is None:
            raise ValueError("num_residual_hiddens must be provided")
        self.num_upsamples = int(num_upsamples)
        num_hiddens = int(num_hiddens)
        if self.num_upsamples <= 0:
            raise ValueError(f"num_upsamples must be positive, got {self.num_upsamples}")
        if num_hiddens < 2:
            raise ValueError(f"num_hiddens must be at least 2, got {num_hiddens}")

        self.conv = nn.Conv2d(int(in_channels), num_hiddens, kernel_size=3, stride=1, padding=1)
        self.res = SimpleResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=int(num_residual_layers),
            num_residual_hiddens=int(num_residual_hiddens),
        )

        up = []
        ch = num_hiddens
        for level in range(self.num_upsamples):
            if level == self.num_upsamples - 1:
                out_ch = int(num_channels)
            elif level == self.num_upsamples - 2:
                out_ch = max(1, num_hiddens // 2)
            else:
                out_ch = num_hiddens
            up.append(nn.ConvTranspose2d(ch, out_ch, kernel_size=4, stride=2, padding=1))
            ch = out_ch
        self.up = nn.ModuleList(up)

    def forward(self, x):
        x = self.res(self.conv(x))
        for level, deconv in enumerate(self.up):
            x = deconv(x)
            if level != len(self.up) - 1:
                x = F.relu(x, inplace=False)
        return x


class Decoder(nn.Module):
    """DDPM-style ResNet-attention decoder."""

    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        use_mid_attention=True,
        extra_res_blocks=1,
        **ignore_kwargs,
    ):
        super().__init__()
        del in_channels, ignore_kwargs
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.give_pre_end = give_pre_end
        self.use_mid_attention = bool(use_mid_attention)
        self.extra_res_blocks = max(0, int(extra_res_blocks))

        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in) if self.use_mid_attention else nn.Identity()
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.blocks_per_level = max(1, self.num_res_blocks + self.extra_res_blocks)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.blocks_per_level):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        self.last_z_shape = z.shape
        temb = None
        h = self.conv_in(z)
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.blocks_per_level):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        if self.give_pre_end:
            return h
        h = self.norm_out(h)
        h = nonlinearity(h)
        return self.conv_out(h)
