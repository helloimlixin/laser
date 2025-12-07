import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import ResidualStack


class Encoder(nn.Module):
    """
    Encoder matching the VQ-VAE-style layout: three strided convs followed by a residual stack.
    """

    def __init__(
        self,
        in_channels,
        num_hiddens,
        num_residual_layers=None,
        num_residual_hiddens=None,
        num_residual_blocks=None,  # backward compatibility alias
    ):
        super().__init__()

        # Backward compatibility: allow num_residual_blocks parameter name
        if num_residual_layers is None:
            num_residual_layers = num_residual_blocks

        self._conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self._conv_2 = nn.Conv2d(
            in_channels=num_hiddens // 2,
            out_channels=num_hiddens,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self._conv_3 = nn.Conv2d(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        x = self._conv_2(x)
        x = F.relu(x)
        x = self._conv_3(x)
        return self._residual_stack(x)


if __name__ == "__main__":
    encoder = Encoder(in_channels=3, num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32)
    x = torch.randn(4, 3, 256, 256)
    print(encoder(x).shape)
