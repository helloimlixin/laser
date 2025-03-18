import torch.nn as nn
import torch.nn.functional as F
from .utils import ResidualBlock

class Decoder(nn.Module):
    """
    Decoder Network implementation followed from the original Neural Discrete Representation Learning paper.

    Reference:
        - Van Den Oord, A., & Vinyals, O. (2017). Neural discrete representation learning.
            Advances in neural information processing systems, 30.
    
    Input: batch_size x 256 x 32 x 32
       │
       └─► 1 x 1 conv ──► 256 channels (processes features)
           │
           └─► 4 x 4 conv ──► 128 channels (reduces dimensions)
                │
                └─► 4 x 4 conv ──► 128 channels (processes features)
    
    Output: batch_size x 3 x 256 x 256

    The decoder network consists of two residual 3 x 3 blocks, followed by  two transposed convolutions with stride 2 and window size 4 x 4.
    """
    def __init__(self, in_channels, num_hiddens, num_residual_blocks, num_residual_hiddens):
        super().__init__()
        
        # Initial processing
        self.conv1 = nn.Conv2d(in_channels,
                               num_hiddens,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(*[ResidualBlock(in_channels=num_hiddens,
                                                             num_hiddens=num_hiddens,
                                                             num_residual_hiddens=num_residual_hiddens)
                                               for _ in range(num_residual_blocks)])
        
        # Upsampling layers
        self._conv_trans_1 = nn.ConvTranspose2d(
            in_channels=num_hiddens,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2,
            padding=1
        )
        
        self._conv_trans_2 = nn.ConvTranspose2d(
            in_channels=num_hiddens // 2,
            out_channels=3,  # Output RGB channels
            kernel_size=4,
            stride=2,
            padding=1
        )

    def forward(self, x):
        # Initial processing
        x = self.conv1(x)
        
        # Residual blocks
        x = self.residual_blocks(x)
        
        # Upsampling
        x = self._conv_trans_1(x)
        x = F.relu(x)
        
        return self._conv_trans_2(x)
