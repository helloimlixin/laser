"""Independent training RNG streams after shared model initialization."""
import hashlib
import random
from contextlib import contextmanager

import numpy as np
import torch
from torch.nn import functional as F


POLICY = 'independent_rank_v1'


@contextmanager
def isolated_rng(seed, device):
    """Seed temporary work without changing Python, NumPy, CPU or device RNG."""
    device = torch.device(device)
    devices = ([device.index if device.index is not None else torch.cuda.current_device()]
               if device.type == 'cuda' else [])
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(int(seed))
            np.random.seed(int(seed) % 2**32)
            # torch.manual_seed also reseeds every CUDA generator, including
            # devices outside fork_rng's saved set.
            torch.random.default_generator.manual_seed(int(seed))
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(int(seed))
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def configure_training_rng(base_seed, rank, *, resuming=False):
    """Seed fresh training per rank; a resumed run retains checkpoint RNG state.

    Call after model construction and DDP synchronization. Dataset shuffling
    uses its own generator and remains shared across ranks.
    """
    base_seed, rank = int(base_seed), int(rank)
    if base_seed < 0 or rank < 0 or base_seed + rank >= 2**63:
        raise ValueError('Training seeds require nonnegative base/rank below 2**63')
    seed = base_seed + rank
    if not resuming:
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    return dict(training_rng_policy=POLICY, training_rng_seed_base=base_seed,
                training_rng_seed=seed, training_rng_restored_from_checkpoint=bool(resuming))


def audit_training_rng(device):
    """Fingerprint a real dropout draw while preserving training RNG streams."""
    device = torch.device(device)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    devices = [device.index] if device.type == 'cuda' else []
    def state():
        return torch.cuda.get_rng_state(device) if device.type == 'cuda' else torch.get_rng_state()
    before = state().clone()
    with torch.random.fork_rng(devices=devices):
        mask = F.dropout(torch.ones(32, 256, device=device), p=.1, training=True).ne(0)
        fingerprint = hashlib.sha256(mask.cpu().numpy().tobytes()).hexdigest()
    unchanged = torch.equal(before, state())
    if not unchanged:
        raise RuntimeError('RNG audit changed the training random stream')
    return dict(rng_state_sha256=hashlib.sha256(before.numpy().tobytes()).hexdigest(),
                dropout_probe_sha256=fingerprint, training_rng_unchanged_by_audit=True)
