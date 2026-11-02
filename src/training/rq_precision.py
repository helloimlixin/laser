"""Select BF16 inside the native RQ model's own autocast guards."""
from contextlib import contextmanager
from functools import partial
import sys

import torch
from src import rq_upstream


@contextmanager
def native_bf16_autocast():
    """Make native amp=True use BF16 without changing model code or weights.

    Its legacy autocast(enabled=amp) overrides any outer autocast context.
    Passing amp=False therefore forces FP32 even inside an outer BF16 guard.
    """
    module = sys.modules[rq_upstream.RQTransformer.__module__]
    original = module.autocast
    module.autocast = partial(torch.autocast, 'cuda', dtype=torch.bfloat16)
    try:
        yield
    finally:
        module.autocast = original
