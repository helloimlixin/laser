import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import ResidualBlock

class Encoder(nn.Module):
    """
    Encoder network inspired by Neural Discrete Representation Learning (VQ‑VAE).

    Reference:
        - Van Den Oord, A., & Vinyals, O. (2017). Neural discrete representation learning.
            Advances in Neural Information Processing Systems, 30.

    Input: batch_size x in_channels x H x W (e.g., 3 x 256 x 256)
       │
       └─► 4x4, stride 2 conv ──► num_hiddens // 2 channels (downsample by 2)
           │
           └─► 4x4, stride 2 conv ──► num_hiddens channels (downsample by 2 again)
                │
                └─► 3x3, stride 1 conv ──► num_hiddens channels (no spatial change)
                    │
                    └─► num_residual_blocks × ResidualBlock(in_channels=num_hiddens,
                        hidden=num_residual_hiddens → num_hiddens)

    Output: batch_size x num_hiddens x (H/4) x (W/4)
        - Example with num_hiddens=128 and input 256x256: 128 x 64 x 64

    Notes:
        - This implementation downsamples by a total factor of 4. To obtain (H/8, W/8)
          outputs (e.g., 32 x 32 from 256 x 256), add another downsampling stage
          (e.g., make the third conv stride=2 or insert an extra strided conv).
    """
    def __init__(self, in_channels, num_hiddens, num_residual_blocks, num_residual_hiddens):
        """
        Initialize the encoder network.

        Args:
            in_channels: Number of input channels
            num_hiddens: Number of hidden units
            num_residual_blocks: Number of residual blocks
            num_residual_hiddens: Number of residual hidden units
        """
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens // 2,
                                 kernel_size=4,
                                 stride=2, padding=1)

        self._conv_2 = nn.Conv2d(in_channels=num_hiddens // 2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)

        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        
        # Residual blocks
        self._residual_blocks = nn.Sequential(*[ResidualBlock(in_channels=num_hiddens,
                                                              num_hiddens=num_hiddens,
                                                              num_residual_hiddens=num_residual_hiddens)
                                               for _ in range(num_residual_blocks)])
    
    def forward(self, x):
        x = F.relu(self._conv_1(x))
        x = F.relu(self._conv_2(x))
        x = F.relu(self._conv_3(x))
        return F.relu(self._residual_blocks(x))


# test the encoder
if __name__ == "__main__":
    encoder = Encoder(in_channels=3, num_hiddens=128, num_residual_blocks=2, num_residual_hiddens=32)
    x = torch.randn(4, 3, 256, 256)
    print(encoder(x).shape)
