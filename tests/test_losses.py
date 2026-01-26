import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path_str in (str(ROOT), str(SRC)):
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from src.models.losses import (
    multi_resolution_dct_loss,
    multi_resolution_gradient_loss,
)


def test_losses_smoke():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 16, 16)
    y = x.clone()

    dct_loss = multi_resolution_dct_loss(x, y, num_levels=2)
    grad_loss = multi_resolution_gradient_loss(x, y, num_levels=2)

    assert dct_loss.ndim == 0 and torch.isfinite(dct_loss)
    assert grad_loss.ndim == 0 and torch.isfinite(grad_loss)
