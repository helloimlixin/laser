"""Checkpoint recovery for deterministic cached-latent stage-2 training."""
from itertools import islice
import random

import numpy as np
import torch


def capture_rng_state(device):
    device = torch.device(device)
    return dict(torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
                numpy=np.random.get_state(), python=random.getstate())


def restore_rng_state(state, device):
    torch.set_rng_state(state['torch'].cpu())
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])
    device = torch.device(device)
    if device.type == 'cuda':
        if state['cuda'] is None:
            raise ValueError('CUDA training requires a saved CUDA RNG state')
        torch.cuda.set_rng_state(state['cuda'].cpu(), device)


def validate_resume_payload(payload, *, world, batch_size, cache, calibration,
                            loader_batches=None, accumulation=None):
    required = {'state_dict', 'optimizer', 'scheduler', 'scaler', 'rng_states',
                'epoch', 'batch_in_epoch', 'step', 'attempts', 'skipped_amp_updates',
                'tokenizer', 'temperature_calibration', 'config', 'initial_weights_sha256'}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f'Incomplete training checkpoint: missing {sorted(missing)}')
    if len(payload['rng_states']) != world:
        raise ValueError('Resume must retain the original world size')
    for state in payload['rng_states']:
        if not {'torch', 'cuda', 'numpy', 'python'} <= state.keys():
            raise ValueError('Incomplete per-rank RNG state')
    experiment = payload['config']['experiment']
    if experiment['batch_size'] != batch_size:
        raise ValueError('Resume must retain the original batch size per rank')
    if experiment['total_batch_size'] != world * batch_size * experiment['accumulation_steps']:
        raise ValueError('Resume batch/accumulation configuration is inconsistent')
    if accumulation is not None and experiment['accumulation_steps'] != accumulation:
        raise ValueError('Resume must retain the original accumulation count')
    for key in ('checkpoint_sha256', 'codebook_sha256', 'cache_sha256',
                'data_protocol_sha256', 'shape', 'images', 'world_size'):
        if key not in cache or payload['tokenizer'].get(key) != cache[key]:
            raise ValueError(f'Resume tokenizer/cache mismatch: {key}')
    if payload['temperature_calibration'] != calibration:
        raise ValueError('Resume temperature calibration differs from the checkpoint')
    for key in ('epoch', 'batch_in_epoch', 'step', 'attempts', 'skipped_amp_updates'):
        if not isinstance(payload[key], int) or payload[key] < 0:
            raise ValueError(f'Invalid resume counter: {key}')
    if payload['attempts'] != payload['step'] + payload['skipped_amp_updates']:
        raise ValueError('Resume optimizer counters are inconsistent')
    batch = payload['batch_in_epoch']
    if loader_batches is not None:
        if batch > loader_batches:
            raise ValueError('Saved batch offset exceeds the dataloader length')
        if accumulation and batch != loader_batches and batch % accumulation:
            raise ValueError('Checkpoint is not at an optimizer-step boundary')
    return payload['epoch'], batch


def restore_training_state(payload, model, optimizer, scheduler, scaler):
    """Call after constructing the optimizer and scheduler; restore RNG later."""
    model.load_state_dict(payload['state_dict'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    # Adam's CPU step counters can otherwise retain the entire mmap-backed
    # checkpoint (including an unlinked old last.pt) for the lifetime of a run.
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value) and value.device.type == 'cpu':
                state[key] = value.clone()
    scheduler.load_state_dict(payload['scheduler'])
    scaler.load_state_dict(payload['scaler'])


def resumed_iterator(loader, sampler, epoch, batch_offset, *, rng_state=None, device='cpu'):
    """Rebuild the deterministic sampler cursor before restoring training RNG.

    CachedLatents has no random transforms. Starting new workers and discarding
    previously consumed batches must not consume the saved model/target RNG.
    This is not a resume implementation for datasets with random augmentation.
    """
    sampler.set_epoch(epoch)
    iterator = iter(loader)
    consumed = sum(1 for _ in islice(iterator, batch_offset))
    if consumed != batch_offset:
        raise ValueError('Could not restore the saved dataloader offset')
    if rng_state is not None:
        restore_rng_state(rng_state, device)
    return iterator
