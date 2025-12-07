import torch.nn as nn
import torch.nn.functional as F
from .utils import ResidualStack


class Decoder(nn.Module):
    """VQ-VAE style decoder with residual stack and two upsampling stages."""

    def __init__(
        self,
        in_channels,
        num_hiddens,
        num_residual_layers=None,
        num_residual_hiddens=None,
        num_channels=3,
        num_residual_blocks=None,  # backward compatibility alias
        out_channels=None,  # alias for num_channels
    ):
        super().__init__()

        # Allow legacy argument names so older checkpoints/configs still load
        if num_residual_layers is None:
            num_residual_layers = num_residual_blocks
        if out_channels is not None:
            num_channels = out_channels

        if num_residual_layers is None:
            raise ValueError("num_residual_layers or num_residual_blocks must be provided")
        if num_residual_hiddens is None:
            raise ValueError("num_residual_hiddens must be provided")

        self._conv_1 = nn.Conv2d(
            in_channels=in_channels,
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

        self._conv_trans_1 = nn.ConvTranspose2d(
            in_channels=num_hiddens,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self._conv_trans_2 = nn.ConvTranspose2d(
            in_channels=num_hiddens // 2,
            out_channels=num_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = self._residual_stack(x)
        x = self._conv_trans_1(x)
        x = F.relu(x)
        return self._conv_trans_2(x)
