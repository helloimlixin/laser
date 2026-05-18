"""Lightweight waveform encoder/decoder blocks for VQ-style audio autoencoders."""

from __future__ import annotations

import ast
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def canonical_int_tuple(values, *, default: Iterable[int]) -> Tuple[int, ...]:
    if values is None:
        values = tuple(default)
    if isinstance(values, str):
        raw = values.strip()
        if not raw:
            values = tuple(default)
        elif raw[0] in "[(":
            values = ast.literal_eval(raw)
        else:
            values = [part for part in raw.split(",") if part.strip()]
    out = tuple(int(value) for value in values)
    if not out:
        raise ValueError("Expected a non-empty integer tuple")
    if any(value <= 0 for value in out):
        raise ValueError(f"All tuple values must be positive, got {out}")
    return out


class AudioResidualUnit(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, *, dilation: int, kernel_size: int = 7):
        super().__init__()
        channels = int(channels)
        hidden_channels = int(hidden_channels)
        dilation = int(dilation)
        kernel_size = int(kernel_size)
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(
                channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class AudioResidualStack(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        *,
        num_layers: int,
        dilation_cycle=(1, 3, 9),
    ):
        super().__init__()
        dilations = canonical_int_tuple(dilation_cycle, default=(1, 3, 9))
        self.layers = nn.ModuleList(
            [
                AudioResidualUnit(
                    channels,
                    hidden_channels,
                    dilation=dilations[idx % len(dilations)],
                )
                for idx in range(int(num_layers))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class AudioEncoder(nn.Module):
    """SEANet/SoundStream-inspired 1D residual strided convolution encoder."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
        downsample_rates=(4, 4, 4),
        dilation_cycle=(1, 3, 9),
    ):
        super().__init__()
        rates = canonical_int_tuple(downsample_rates, default=(4, 4, 4))
        num_hiddens = int(num_hiddens)
        self.downsample_rates = rates
        self.conv_in = nn.Conv1d(int(in_channels), num_hiddens, kernel_size=7, padding=3)
        self.blocks = nn.ModuleList()
        for rate in rates:
            if rate % 2 != 0:
                raise ValueError(f"Audio downsample rates must be even for exact shape recovery, got {rates}")
            self.blocks.append(
                nn.Sequential(
                    AudioResidualStack(
                        num_hiddens,
                        int(num_residual_hiddens),
                        num_layers=int(num_residual_layers),
                        dilation_cycle=dilation_cycle,
                    ),
                    nn.SiLU(),
                    nn.Conv1d(
                        num_hiddens,
                        num_hiddens,
                        kernel_size=2 * int(rate),
                        stride=int(rate),
                        padding=int(rate) // 2,
                    ),
                )
            )
        self.conv_out = nn.Conv1d(num_hiddens, num_hiddens, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for block in self.blocks:
            x = block(x)
        return self.conv_out(F.silu(x))


class AudioDecoder(nn.Module):
    """1D residual decoder matching :class:`AudioEncoder` downsampling rates."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
        out_channels: int,
        upsample_rates=(4, 4, 4),
        dilation_cycle=(1, 3, 9),
    ):
        super().__init__()
        rates = canonical_int_tuple(upsample_rates, default=(4, 4, 4))
        num_hiddens = int(num_hiddens)
        self.upsample_rates = rates
        self.conv_in = nn.Conv1d(int(in_channels), num_hiddens, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList()
        for rate in reversed(rates):
            if rate % 2 != 0:
                raise ValueError(f"Audio upsample rates must be even for exact shape recovery, got {rates}")
            self.blocks.append(
                nn.Sequential(
                    AudioResidualStack(
                        num_hiddens,
                        int(num_residual_hiddens),
                        num_layers=int(num_residual_layers),
                        dilation_cycle=dilation_cycle,
                    ),
                    nn.SiLU(),
                    nn.ConvTranspose1d(
                        num_hiddens,
                        num_hiddens,
                        kernel_size=2 * int(rate),
                        stride=int(rate),
                        padding=int(rate) // 2,
                    ),
                )
            )
        self.res_out = AudioResidualStack(
            num_hiddens,
            int(num_residual_hiddens),
            num_layers=max(1, int(num_residual_layers)),
            dilation_cycle=dilation_cycle,
        )
        self.conv_out = nn.Conv1d(num_hiddens, int(out_channels), kernel_size=7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for block in self.blocks:
            x = block(x)
        x = self.res_out(x)
        return self.conv_out(F.silu(x))
