import torch.nn as nn
from src.models.utils import ResidualBlock

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dims, n_residual_blocks=2):
        super().__init__()
        
        # Initial convolution
        layers = [
            nn.Conv2d(in_channels, hidden_dims[0], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling layers
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Conv2d(hidden_dims[i], hidden_dims[i+1], kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(hidden_dims[i+1]),
                nn.ReLU(inplace=True)
            ])
        
        # Residual blocks
        current_channels = hidden_dims[-1]
        for _ in range(n_residual_blocks):
            # ResidualBlock expects out_channels to be 1/4 of the final output due to expansion=4
            layers.append(ResidualBlock(current_channels, current_channels // 4))
            current_channels = current_channels  # Stays the same due to ResidualBlock design
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.encoder(x)
