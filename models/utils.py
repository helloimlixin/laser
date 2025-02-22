import torch
from torch import nn

class ResidualBlock(nn.Module):
    """Residual Block

    References:
        - Deep Residual Learning for Image Recognition, Kaiming He, Xiangyu Zhang, Jian Sun.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()  # by default, it's an empty Sequential module (identity mapping)
        if stride != 1 or in_channels != out_channels:
            # do shortcut mapping when,
            # - the stride is not 1, the spatial dimensions will change
            # - the input channels don't match the output channels
            self.shortcut = nn.Sequential(
                # a 1 x 1 (kernel size) convolution to match the spatial dimensions
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # the shortcut connection helps solve the degradation problem in deep neural networks,
        # it allows x to bypass the main convolutional layers and be added directly to their output
        out += self.shortcut(x)
        out = self.relu(out)
        return out