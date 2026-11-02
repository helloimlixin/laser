"""Expand a saved four-rank RNG layout without reseeding existing ranks."""
import copy
import hashlib
import random

import numpy as np
import torch


def expand_four_to_eight(states, checkpoint_sha256, global_step, cuda_state_factory):
    if len(states) != 4 or int(global_step) != global_step or global_step < 0:
        raise ValueError('Expected four saved ranks and a nonnegative optimizer step')
    if (len(checkpoint_sha256) != 64 or
            any(c not in '0123456789abcdef' for c in checkpoint_sha256)):
        raise ValueError('A verified checkpoint SHA256 is required')
    required = {'python', 'numpy', 'cpu', 'cuda'}
    if any(set(state) != required for state in states):
        raise ValueError('Expected the native joint-training RNG state fields')
    expanded = copy.deepcopy(states)
    seeds = {}
    for rank in range(4, 8):
        identity = f'laser-ddp-layout-4-to-8:{checkpoint_sha256}:{global_step}:{rank}'
        seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], 'big') % (2**63-1)
        cuda = cuda_state_factory(seed)
        if cuda.dtype != torch.uint8 or cuda.ndim != 1 or cuda.device.type != 'cpu':
            raise ValueError('CUDA RNG factory must return a CPU byte state')
        expanded.append(dict(python=random.Random(seed).getstate(),
            numpy=np.random.RandomState(seed % 2**32).get_state(),
            cpu=torch.Generator().manual_seed(seed).get_state(), cuda=cuda.clone()))
        seeds[rank] = seed
    return expanded, seeds
