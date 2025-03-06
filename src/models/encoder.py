import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import ResidualBlock

class Encoder(nn.Module):
    """
    Encoder Network implementation followed from the original Neural Discrete Representation Learning paper.

    Reference:
        - Van Den Oord, A., & Vinyals, O. (2017). Neural discrete representation learning.
            Advances in neural information processing systems, 30.
    
    Input: batch_size x 3 x 256 x 256
       │
       └─► 4 x 4 conv ──► 128 channels (reduces dimensions)
           │
           └─► 4 x 4 conv ──► 128 channels (processes features)
                │
                └─► 2 x Residual Blocks ──► 256 channels

    Output: batch_size x 256 x 32 x 32

    The encoder network consists of 2 convolutional layers with stride 2 and window size 4 x 4, followed by a convolutional
    layer with stride 1 and window size 3 x 3, and then two residual blocks, which are implemented as ReLU, 3 x 3 conv, ReLU 
    and 1 x 1 conv, all having 256 hidden units.
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
        
        # Initial convolution layers
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
    x = torch.randn(4, 3, 256, 256)  # batch_size x 3 x 256 x 256
    print(encoder(x).shape)  # batch_size x 128 x 64 x 64

