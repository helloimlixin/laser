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
    """Residual Block for the Encoder Network. Implemented as
        ReLU -> 3 x 3 conv -> ReLU -> 1 x 1 conv

    References:
        - He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition.
            In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
        - Van Den Oord, A., & Vinyals, O. (2017). Neural discrete representation learning.
            Advances in neural information processing systems, 30.

    Input: 256 channels
       │
       └─► 1×1 conv ──► 64 channels    (reduces dimensions)
           │
           └─► 3×3 conv ──► 64 channels (processes features)
               │
               └─► 1×1 conv ──► 256 channels (expands: 64 * expansion=4)

    Dramatically reduces the computation while maintaining the model capacity for very deep networks.
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
