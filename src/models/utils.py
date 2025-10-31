import torch
from torch import nn
from tqdm.auto import tqdm
import torch.nn.functional as F

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
    """Residual block used in the encoder.
    Implemented as 3x3 conv → BN → ReLU → 1x1 conv → BN with a skip connection,
    followed by a ReLU on the summed output.

    References:
        - He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition.
            In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
        - Van Den Oord, A., & Vinyals, O. (2017). Neural discrete representation learning.
            Advances in neural information processing systems, 30.

    Input/Output: num_hiddens channels
       │
       └─► 3×3 conv ──► num_residual_hiddens channels (process features)
           │
           └─► 1×1 conv ──► num_hiddens channels (project back to input width)

    This keeps spatial resolution and channel width while adding capacity via a
    lightweight bottleneck.
    """
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, num_residual_hiddens, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_residual_hiddens)
        
        self.conv2 = nn.Conv2d(num_residual_hiddens, num_hiddens, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_hiddens)

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=False)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = F.relu(out, inplace=False)
        
        return out
