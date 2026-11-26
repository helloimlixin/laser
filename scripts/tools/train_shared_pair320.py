#!/usr/bin/env python3
"""Frozen-codec ImageNet stage 2 with BF16 training and native sampling."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys

import torch

BASE = Path(os.environ.get('SHARED_PAIR320_BASE', '/tmp/laser-sharedpair24577-d5-4gpu-20261112'))
ROOT = Path(os.environ.get('LASER_PROJECT_ROOT', '/workspace/Projects/laser'))
os.environ['RQ_UPSTREAM_ROOT'] = str(BASE / 'upstream')
sys.path[:0] = [str(BASE / 'source'), str(BASE), str(ROOT)]

from src import rq_upstream
from src.training.laser_joint_recipe import JointLaserRecipe
from src.training.rq_sdpa_backend import sdpa_attention
from src.training.rq_imagenet_shared import main


def autocast(*args, enabled=True, **kwargs):
    return torch.autocast('cuda', enabled=enabled,
        dtype=torch.bfloat16 if torch.is_grad_enabled() else torch.float16)


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    module = sys.modules[rq_upstream.RQTransformer.__module__]
    module.autocast = autocast
    original_adapt = JointLaserRecipe.adapt
    with ExitStack() as stack:
        def adapt(profile, model, aux):
            original_adapt(profile, model, aux)
            stack.enter_context(sdpa_attention(model, batch_limit=512, short_sequence_math=True))
        JointLaserRecipe.adapt = adapt
        main('laser_joint', BASE)
