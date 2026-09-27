import torch
from torch import nn
from tqdm.auto import tqdm
import torch.nn.functional as F


def fid_has_enough_samples(metric) -> bool:
    """Return True iff the FID metric has the >=2 samples per side that .compute() requires.

    torchmetrics' ``FrechetInceptionDistance.compute()`` raises a ``RuntimeError``
    if either the real or fake feature buffer holds fewer than 2 samples. That
    fires whenever ``compute_fid=True`` is set on an audio stage-1 run: the
    metric is lazily allocated but never receives ``update()`` calls because
    the audio code path skips them (rFID is meaningless for 1-channel audio
    and the Inception backbone expects 3 channels). Guard ``.compute()`` with
    this helper so the validation/test epoch end does not crash an otherwise
    healthy run.
    """
    if metric is None:
        return False
    real = getattr(metric, "real_features_num_samples", None)
    fake = getattr(metric, "fake_features_num_samples", None)
    if real is None or fake is None:
        # Unknown torchmetrics layout — be conservative and let .compute() raise so
        # the caller still sees the real diagnostic instead of a silent miss.
        return True
    try:
        return int(real) >= 2 and int(fake) >= 2
    except (TypeError, ValueError):
        return True


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


class ResidualStack(nn.Module):
    """Stack of residual blocks followed by a ReLU."""

    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ResidualBlock(
                    in_channels=in_channels,
                    num_hiddens=num_hiddens,
                    num_residual_hiddens=num_residual_hiddens,
                )
                for _ in range(num_residual_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return F.relu(x, inplace=False)
