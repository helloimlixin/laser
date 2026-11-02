"""Preserve optimizer boundaries and sample coverage when moving this ImageNet run."""
import hashlib
import json
import math
import os
from pathlib import Path

import torch


def validate_layout(saved, world, accumulation, total_batch, batch_idx):
    if not saved.get('exact_global_batch') or total_batch != saved.get('total_batch_size'):
        raise ValueError('Resume requires the original exact global batch')
    if total_batch != 2048 or min(world, accumulation) < 1:
        raise ValueError('Invalid ImageNet recovery layout')
    if world * accumulation > total_batch:
        raise ValueError('Every rank must have a nonempty microbatch')
    if not isinstance(batch_idx, int) or batch_idx < 0 or batch_idx % accumulation:
        raise ValueError('Resume must be at an optimizer boundary')
    migration = saved.get('portable_resume_cursor')
    if migration is not None:
        if (migration['new_world'] != world or migration['new_accumulation'] != accumulation
                or migration['new_batch_idx'] != batch_idx):
            raise ValueError('Recovery cursor does not match the launch layout')
    elif saved.get('world_size') != world or saved.get('accumulation_steps') != accumulation:
        raise ValueError('Changed layout requires a remapped recovery cursor')
    return saved.get('world_size') != world or saved.get('accumulation_steps') != accumulation


def remap_payload(payload, *, world, accumulation, checkpoint_dir, reference, reference_sha256):
    """Modify only launch paths/cursor metadata, retaining weights and optimizer tensors."""
    payload = dict(payload)
    config = dict(payload['config'])
    old_accumulation = int(config['accumulation_steps'])
    old_batch = int(payload.get('batch_idx', 0))
    if old_batch < 0 or old_batch % old_accumulation:
        raise ValueError('Saved checkpoint is not at an optimizer boundary')
    epoch = int(payload['epoch'])
    completed_steps = old_batch // old_accumulation
    steps_per_epoch = math.ceil(int(config['training_images']) / int(config['total_batch_size']))
    if completed_steps > steps_per_epoch:
        raise ValueError('Checkpoint cursor exceeds the epoch')
    if int(payload['global_step']) != epoch * steps_per_epoch + completed_steps:
        raise ValueError('Checkpoint step and data cursor disagree')
    if completed_steps == steps_per_epoch:
        epoch += 1
        completed_steps = 0
    new_batch = completed_steps * accumulation
    cursor = dict(old_world=int(config['world_size']), new_world=world,
                  old_accumulation=old_accumulation, new_accumulation=accumulation,
                  old_epoch=int(payload['epoch']), old_batch_idx=old_batch,
                  new_epoch=epoch, new_batch_idx=new_batch,
                  completed_optimizer_steps_in_epoch=completed_steps)
    config['portable_resume_cursor'] = cursor
    config['portable_original_reference_path'] = str(config['fid_reference_stats'])
    config['fid_reference_stats'] = str(reference)
    # The trainer recognizes content-equivalent references through this field.
    config['amarel_continuation'] = dict(config.get('amarel_continuation', {}),
                                       reference_sha256=reference_sha256)
    checkpoint_dir = Path(checkpoint_dir)
    for field in ('best_fid', 'best_inception'):
        payload[field] = [(float(score), str(checkpoint_dir / Path(path).name))
                          for score, path in payload.get(field, [])]
        if any(not Path(path).is_file() for _, path in payload[field]):
            raise FileNotFoundError(f'Missing recovery winner: {field}')
    payload.update(config=config, epoch=epoch, batch_idx=new_batch)
    validate_layout(config, world, accumulation, config['total_batch_size'], new_batch)
    return payload


def restore_rng(payload, device, seed):
    states = payload['rng_state_by_rank']
    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if rank < len(states):
        local = states[rank]
        policy = 'retained checkpoint CPU/CUDA stream'
        derived_seed = None
    else:
        digest = hashlib.sha256(f"imagenet-migration:{len(states)}:{world}:{payload['global_step']}:{rank}".encode())
        for state in states:
            for key in ('torch_cpu', 'torch_cuda'):
                if key in state:
                    digest.update(state[key].cpu().numpy().tobytes())
        derived_seed = int.from_bytes(digest.digest()[:8], 'big') % (2**63 - 1)
        local = {'torch_cpu': torch.Generator().manual_seed(derived_seed).get_state()}
        if device.type == 'cuda':
            local['torch_cuda'] = torch.Generator(device=device).manual_seed(derived_seed).get_state()
        policy = 'new independent stream derived from saved RNG states, step, world and rank'
    torch.set_rng_state(local['torch_cpu'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(local['torch_cuda'], device=device)
    directory = os.environ.get('LASER_LAYOUT_MIGRATION_DIR')
    if directory:
        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)
        (p / f'rng-rank{rank}.json').write_text(json.dumps(dict(
            passed=True, rank=rank, old_world=len(states), new_world=world,
            global_step=payload['global_step'], policy=policy, derived_seed=derived_seed,
            saved_rank_streams_restored_exactly=len(states) == world,
            trajectory_bitwise_identical_guaranteed=False), indent=2) + '\n')
    return True if len(states) == world else 'resized'
