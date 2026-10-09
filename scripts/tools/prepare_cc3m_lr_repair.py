"""Lower a stalled CC3M continuation LR without discarding any training state."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

from omegaconf import OmegaConf
import torch

from scripts.tools.prepare_cc3m_best_resume import assert_identical, digest
from src.training.cc3m_text import (config_digest, create_lr_scheduler,
    verify_checkpoint_progress, verify_resume_config)
from src.training.fid_adaptive_schedule import FidAdaptiveSchedule
from src.training import rqtransformer as rq


def lower_lr_options(source, factor=.5, patience=3):
    """Keep the cosine timeline and halve the actual LR at the saved update."""
    if not 0 < factor < 1 or patience < 2:
        raise ValueError('LR repair requires a reduction and sustained observations')
    saved = source['config']
    target = copy.deepcopy(saved)
    for key in list(target):
        if 'migration' in key:
            target.pop(key)
    floor = saved['min_lr']
    before = source['optimizer']['param_groups'][0]['lr']
    after = max(floor, before * factor)
    old_policy = source['scheduler']['policy']
    excess = before - floor
    if excess <= 0 or after >= before:
        raise ValueError('The current LR is already at its floor')
    # The same multiplier and cosine position apply to every converted slot.
    # Scaling the excess peak makes the latest actual LR exactly half, while
    # retaining one common policy for last, best-FID, and best-CLIP resumes.
    target['lr'] = floor + (old_policy['initial_lr'] - floor) * (after - floor) / excess
    policy = target['fid_lr_policy']
    policy.pop('adaptive_reductions', None)  # True is the backward-compatible default.
    policy['patience'] = patience
    target.update(warmup_epochs=0, evaluate_on_resume=False,
        checkpoint_on_resume=False, resume_expected_metrics=None,
        stage2_initialization='latest_full_state_with_lower_lr',
        wandb_name='CC3M LASER | lower LR with plateau reductions | 100 epochs | 8 H100')
    return target


def convert_lr_checkpoint(source, target):
    updates = target['train_items'] // target['total_batch_size']
    migration = dict(source_config_sha256=config_digest(source['config']),
        source_step=source['global_step'],
        source_scheduler_sha256=config_digest(source['scheduler']))
    checked = dict(target, lr_schedule_migration=migration)
    verify_resume_config(source['config'], checked)
    optimizer = SimpleNamespace(param_groups=copy.deepcopy(source['optimizer']['param_groups']))
    scheduler = create_lr_scheduler(optimizer, checked, updates, source['global_step'],
        source['scheduler'], source['config'])
    before = source['optimizer']['param_groups'][0]['lr']
    after = optimizer.param_groups[0]['lr']
    assert target['min_lr'] <= after < before
    # The manual cut has already acted on the accumulated stall. Give the
    # lower LR a cooldown evaluation before counting new stalled epochs.
    scheduler.bad_epochs = 0
    scheduler.cooldown_remaining = target['fid_lr_policy']['cooldown']
    scheduler.reductions += 1
    payload = dict(source, config=target, scheduler=scheduler.state_dict(),
        optimizer=dict(source['optimizer'], param_groups=optimizer.param_groups),
        checkpoint_migration=dict(timestamp=time.time(), source_step=source['global_step'],
            source_lr=before, target_lr=after, source_config_sha256=migration['source_config_sha256'],
            reason='manual LR cut after three evaluations above the run best',
            stalled_counter_reset=True, cooldown_after_manual_cut=True))
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--source-local', type=Path, required=True)
    parser.add_argument('--local', type=Path, required=True)
    parser.add_argument('--checkpoint', choices=['last.pt', 'best-fid.pt'], default='last.pt')
    parser.add_argument('--source-checkpoint', type=Path,
        help='A preserved earlier best; current run best slots remain protected')
    args = parser.parse_args()
    base, previous, local = args.base.resolve(), args.source_local.resolve(), args.local.resolve()
    root = Path(__file__).resolve().parents[2]
    assert not local.exists(), 'Use a new local directory'
    source_path = args.source_checkpoint or previous/'checkpoints'/args.checkpoint
    source = torch.load(source_path, map_location='cpu', mmap=True, weights_only=True)
    latest = torch.load(previous/'checkpoints/last.pt', map_location='cpu', mmap=True, weights_only=True)
    saved = source['config']
    updates = saved['train_items'] // saved['total_batch_size']
    verify_checkpoint_progress(source, updates, saved['accumulation'], 8)
    target = lower_lr_options(source)
    if args.checkpoint == 'best-fid.pt':
        assert source['metrics']['fid'] == source['best_fid']
        target.update(evaluate_on_resume=True, resume_expected_step=source['global_step'],
            resume_expected_metrics=source['metrics'], stage2_initialization='original_best_fid_with_lower_lr')
    assert target['epochs'] == saved['epochs'] == 100
    runtime = local/'runtime'
    runtime.mkdir(parents=True)
    with tarfile.open(base/'runtime.tar.gz') as archive:
        archive.extractall(runtime, filter='data')
    manifest = dict(latest['config']['runtime_sha256'])
    for name, expected in manifest.items():
        assert digest(runtime/name) == expected, name
    helper = 'scripts/tools/prepare_cc3m_lr_repair.py'
    shutil.copyfile(root/helper, runtime/helper)
    manifest[helper] = digest(runtime/helper)
    subprocess.run([sys.executable, '-c', 'import src.training.cc3m_text; import train'],
        cwd=runtime, env=dict(os.environ, PYTHONPATH=f'{runtime/"runtime"}:{runtime}'), check=True)
    (local/'assets').mkdir()
    for name in ['stage1.pt', 'train.pt', 'validation.pt']:
        (local/'assets'/name).hardlink_to(previous/'assets'/name)
    for name in ['torch', 'inductor', 'triton']:
        (local/name).symlink_to(previous/name, target_is_directory=True)
    target.update(runtime_sha256=manifest, resume=True, resume_checkpoint=None,
        checkpoint=str(local/'assets/stage1.pt'), token_cache=str(local/'assets/train.pt'),
        validation_cache=str(local/'assets/validation.pt'), local_checkpoints=str(local/'checkpoints'))
    (local/'checkpoints').mkdir()
    archive = base/'fid-lr-repair-source-20261005'
    archive.mkdir(exist_ok=True)
    for name in ['recipe.yaml', 'runtime.tar.gz', 'runtime-manifest.json', 'resume.py']:
        shutil.copyfile(base/name, archive/name)
    for name in ['evaluation', 'lr-schedule']:
        # The shared mount stores content but does not implement copystat.
        (archive/name).mkdir(exist_ok=True)
        for path in (base/'train'/name).glob('*.json'):
            shutil.copyfile(path, archive/name/path.name)
    preserved = previous/'checkpoints/before-lr-repair'
    preserved.mkdir()
    for slot in ['last.pt', 'best-fid.pt', 'best-clip.pt']:
        (preserved/slot).hardlink_to(previous/'checkpoints'/slot)
    groups, records = {}, []
    for slot in ['last.pt', 'best-fid.pt', 'best-clip.pt']:
        path = source_path if slot == 'last.pt' else previous/'checkpoints'/slot
        identity = (path.stat().st_dev, path.stat().st_ino)
        if identity in groups:
            (local/'checkpoints'/slot).hardlink_to(groups[identity])
            records.append(dict(slot=slot, shared_with=groups[identity].name))
            continue
        state = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
        verify_checkpoint_progress(state, updates, saved['accumulation'], 8)
        payload = convert_lr_checkpoint(state, target)
        # Restoring an earlier point must never replace a later run-wide winner.
        payload.update(best_fid=latest['best_fid'], best_clip=latest['best_clip'])
        payload['checkpoint_slots'] = [slot]
        destination = local/'checkpoints'/slot
        rq.atomic_torch_save(payload, destination)
        restored = torch.load(destination, map_location='cpu', mmap=True, weights_only=True)
        verify_checkpoint_progress(restored, updates, saved['accumulation'], 8)
        verify_resume_config(restored['config'], target)
        for field in ['model', 'rng_state_by_rank', 'epoch', 'next_microbatch', 'global_step',
                      'world_size', 'metrics', 'cache_meta', 'bpe_policy']:
            assert_identical(state[field], restored[field])
        assert restored['best_fid'] == latest['best_fid']
        assert restored['best_clip'] == latest['best_clip']
        assert_identical(state['optimizer']['state'], restored['optimizer']['state'])
        assert_identical(payload['scheduler'], restored['scheduler'])
        schedule = create_lr_scheduler(SimpleNamespace(param_groups=copy.deepcopy(restored['optimizer']['param_groups'])),
            target, updates, restored['global_step'], restored['scheduler'], target)
        assert schedule.last_epoch == state['global_step']
        groups[identity] = destination
        record = dict(slot=slot, step=state['global_step'], epoch=state['epoch'],
            bytes=destination.stat().st_size, md5=digest(destination, 'md5'),
            lr_before=state['optimizer']['param_groups'][0]['lr'],
            lr_after=restored['optimizer']['param_groups'][0]['lr'],
            optimizer_states=len(restored['optimizer']['state']), rank_rng_states=len(restored['rng_state_by_rank']),
            weights_adam_rng_cursor_metrics_exact=True)
        records.append(record)
        print(json.dumps(record), flush=True)
        del state, payload, restored
    last = torch.load(local/'checkpoints/last.pt', map_location='cpu', mmap=True, weights_only=True)
    curve = [FidAdaptiveSchedule.lr_at_step(last['scheduler']['policy'], step, last['scheduler']['multiplier'])
             for step in range(last['global_step'], target['epochs']*updates+1)]
    assert all(a >= b >= target['min_lr'] for a, b in zip(curve, curve[1:]))
    assert abs(curve[0] - source['optimizer']['param_groups'][0]['lr']*.5) < 1e-15
    cfg = OmegaConf.load(base/'recipe.yaml')
    cfg.options = target
    OmegaConf.save(cfg, base/'recipe.yaml')
    OmegaConf.save(cfg, root/'configs/stage2'/(target['wandb_id']+'.yaml'))
    with tarfile.open(base/'runtime.tar.gz', 'w:gz') as archive_out:
        for name in manifest:
            archive_out.add(runtime/name, arcname=name)
    (base/'runtime-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    report = dict(timestamp=time.time(), source_step=source['global_step'], source_epoch=source['epoch'],
        source_checkpoint=str(source_path), source_metrics=source.get('metrics'),
        protected_run_best_fid=latest['best_fid'], protected_run_best_clip=latest['best_clip'],
        preserved_latest_step=latest['global_step'], preserved_latest_epoch=latest['epoch'],
        preserved_checkpoints=str(preserved),
        lr_before=source['optimizer']['param_groups'][0]['lr'], lr_after=curve[0], min_lr=target['min_lr'],
        target_epochs=100, local=str(local), fid_lr_policy=target['fid_lr_policy'],
        manual_reduction_factor=.5, adaptive_reductions=True, all_remaining_lr_steps_nonincreasing=True,
        weights_adam_rng_cursor_preserved=True, evaluation_protocol_unchanged=True, checkpoints=records)
    (base/'fid-lr-repair-plan.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
