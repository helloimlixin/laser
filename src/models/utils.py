import torch
from torch import nn
from tqdm.auto import tqdm

def compute_accuracy(eval_model, dataloader):
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader):
            eval_images, eval_labels = batch
            if torch.cuda.is_available():
                eval_images = eval_images.cuda()
                eval_labels = eval_labels.cuda()

            eval_outputs = eval_model(eval_images)
            _, predicted = torch.max(eval_outputs.data, 1)
            total += eval_labels.size(0)
            correct += predicted.eq(eval_labels).sum().item()

    return correct / total

class ResidualBlock(nn.Module):
    """Residual Block

    References:
        - Deep Residual Learning for Image Recognition, Kaiming He, Xiangyu Zhang, Jian Sun.

    Input: 256 channels
       │
       └─► 1×1 conv ──► 64 channels    (reduces dimensions)
           │
           └─► 3×3 conv ──► 64 channels (processes features)
               │
               └─► 1×1 conv ──► 256 channels (expands: 64 * expansion=4)

    Dramatically reduces the computation while maintaining the model capacity for very deep networks.
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # 1 x 1 convolution, reduces dimension: 256 -> 64 channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3 x 3 convolution, process features: 64 -> 64 channels
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1 x 1 convolution, expands to 256 channels: 64 -> 256 channels
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)  # ReLU activation

        self.shortcut = nn.Sequential()  # by default, it's an empty Sequential module (identity mapping)
        if stride != 1 or in_channels != out_channels * self.expansion:
            # do shortcut mapping when,
            # - the stride is not 1, the spatial dimensions will change
            # - the input channels don't match the output channels
            self.shortcut = nn.Sequential(
                # a 1 x 1 (kernel size) convolution to match the spatial dimensions
                nn.Conv2d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # the shortcut connection helps solve the degradation problem in deep neural networks,
        # it allows x to bypass the main convolutional layers and be added directly to their output
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """Residual Network

    References:
        - Deep Residual Learning for Image Recognition, Kaiming He, Xiangyu Zhang, Jian Sun.

    Recall the formula for the convolutional layer dimensions:

    O = [(N + 2P - K) / S] + 1, where
    - O: output dimension (height and width, aka. number of feature maps)
    - N: input dimension (height and width)
    - K: kernel size
    - P: padding size
    - S: stride length

    The divisions are rounded up to the nearest integer (floor division).

    Example: for a 224 x 224 input, using a 7 x 7 kernel with stride=2, padding=3
        O = [(224 + 2*3 - 7) / 2] + 1 = 112

    As for the pooling layers, the formula is similar but typically without the padding:
    O = [(N - K) / S] + 1

    Examples:
        1. for a 28 x 28 input, using a 2 x 2 kernel with stride=2 (pool size = stride length):
                O = [(28 - 2) / 2] + 1 = 14,
            which effectively reduces the spatial dimensions by half.
        2. for a 112 x 112 input, using a 3 x 3 kernel with stride=2 (pool size = stride length):
                O = [(112 - 3) / 2] + 1 = 55,

    Average pooling often used at the end of a network to reduce the spatial dimensions to 1 x 1.
    """
    def __init__(self, block, num_blocks, num_classes=1000):
        """
        initialize the ResNet model
        :param block: a residual block class
        :param num_blocks: number of residual blocks in each residual layer
        :param num_classes: number of classes for the classification task
        """
        super(ResNet, self).__init__()
        self.in_channels = 64

        # initial convolution layer, kernel size: 7 x 7, #channels: 3 (RGB) -> 64, stride: 2, padding: 3
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # max pooling layer, kernel size: 3 x 3, stride: 2, padding: 1
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # the residual layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # final layers

        # here instead of specifying the kernel size and stride, you specify the desired output size,
        # this layer will automatically calculate the necessary kernel size and stride length
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # initial convolution
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.max_pool(out)

        # residual layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # final layers for classification
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
