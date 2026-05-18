"""VQ-VAE-style convolutional encoder/decoder used by the LASER image stage.

This mirrors the simpler scratch implementation that was reconstructing more
faithfully than the heavier attention backbone, while keeping the downsample
depth configurable for 256x256 images and spectrogram inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, num_hiddens: int, num_residual_hiddens: int):
        super().__init__()
        if int(in_channels) != int(num_hiddens):
            raise ValueError(
                "ResidualBlock expects in_channels == num_hiddens, got "
                f"{in_channels} and {num_hiddens}"
            )
        self.conv1 = nn.Conv2d(
            int(in_channels),
            int(num_residual_hiddens),
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(int(num_residual_hiddens))
        self.conv2 = nn.Conv2d(
            int(num_residual_hiddens),
            int(num_hiddens),
            kernel_size=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(int(num_hiddens))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=False)
        out = self.bn2(self.conv2(out))
        return F.relu(out + x, inplace=False)


class ResidualStack(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
    ):
        super().__init__()
        num_residual_layers = int(num_residual_layers)
        if num_residual_layers <= 0:
            raise ValueError(f"num_residual_layers must be positive, got {num_residual_layers}")
        self.layers = nn.ModuleList(
            [
                ResidualBlock(
                    in_channels=int(in_channels),
                    num_hiddens=int(num_hiddens),
                    num_residual_hiddens=int(num_residual_hiddens),
                )
                for _ in range(num_residual_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
        num_downsamples: int = 4,
    ):
        super().__init__()
        num_downsamples = int(num_downsamples)
        num_hiddens = int(num_hiddens)
        if num_downsamples <= 0:
            raise ValueError(f"num_downsamples must be positive, got {num_downsamples}")
        if num_hiddens < 2:
            raise ValueError(f"num_hiddens must be at least 2, got {num_hiddens}")

        self.num_downsamples = num_downsamples
        down = []
        ch = int(in_channels)
        for level in range(num_downsamples):
            out_ch = max(1, num_hiddens // 2) if level == 0 else num_hiddens
            down.append(nn.Conv2d(ch, out_ch, kernel_size=4, stride=2, padding=1))
            ch = out_ch
        self.down = nn.ModuleList(down)
        self.conv = nn.Conv2d(ch, num_hiddens, kernel_size=3, stride=1, padding=1)
        self.res = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=int(num_residual_layers),
            num_residual_hiddens=int(num_residual_hiddens),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.down:
            x = F.relu(conv(x), inplace=False)
        return self.res(self.conv(x))


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
        out_channels: int = 3,
        num_upsamples: int = 4,
    ):
        super().__init__()
        num_upsamples = int(num_upsamples)
        num_hiddens = int(num_hiddens)
        if num_upsamples <= 0:
            raise ValueError(f"num_upsamples must be positive, got {num_upsamples}")
        if num_hiddens < 2:
            raise ValueError(f"num_hiddens must be at least 2, got {num_hiddens}")

        self.num_upsamples = num_upsamples
        self.conv = nn.Conv2d(int(in_channels), num_hiddens, kernel_size=3, stride=1, padding=1)
        self.res = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=int(num_residual_layers),
            num_residual_hiddens=int(num_residual_hiddens),
        )

        up = []
        ch = num_hiddens
        for level in range(num_upsamples):
            if level == num_upsamples - 1:
                out_ch = int(out_channels)
            elif level == num_upsamples - 2:
                out_ch = max(1, num_hiddens // 2)
            else:
                out_ch = num_hiddens
            up.append(nn.ConvTranspose2d(ch, out_ch, kernel_size=4, stride=2, padding=1))
            ch = out_ch
        self.up = nn.ModuleList(up)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res(self.conv(x))
        for level, deconv in enumerate(self.up):
            x = deconv(x)
            if level != len(self.up) - 1:
                x = F.relu(x, inplace=False)
        return x
