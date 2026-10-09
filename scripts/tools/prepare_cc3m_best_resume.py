"""Prepare a verified original best checkpoint for a smooth full-state continuation."""
import argparse
import base64
import copy
import hashlib
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

from src.training.cc3m_text import (config_digest, create_lr_scheduler,
    verify_checkpoint_progress, verify_resume_config)
from src.training.fid_adaptive_schedule import FidAdaptiveSchedule
from src.training import rqtransformer as rq


def digest(path, algorithm='sha256'):
    with Path(path).open('rb') as stream:
        value = hashlib.file_digest(stream, algorithm).digest()
    return base64.b64encode(value).decode() if algorithm == 'md5' else value.hex()


def assert_identical(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_identical(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_identical(a, b)
    else:
        assert left == right


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--source-local', type=Path, required=True)
    parser.add_argument('--local', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    base, previous, local = args.base.resolve(), args.source_local.resolve(), args.local.resolve()
    root = Path(__file__).resolve().parents[2]
    source_path = previous/'checkpoints/best-fid.pt'
    source = torch.load(source_path, map_location='cpu', weights_only=True, mmap=True)
    saved = source['config']
    updates = saved['train_items']//saved['total_batch_size']
    verify_checkpoint_progress(source, updates, saved['accumulation'], 8)
    assert source['metrics']['fid'] == source['best_fid']
    assert args.epochs >= saved['epochs'] and args.epochs*updates > source['global_step']
    assert not (local/'checkpoints').exists(), 'Use a new continuation directory'
    source_md5 = digest(source_path, 'md5')
    expected = json.loads((base/'original-best-resume-cloud-source.json').read_text())
    assert expected['best-fid.pt']['md5'] == source_md5
    assert expected['best-fid.pt']['bytes'] == source_path.stat().st_size
    runtime = local/'runtime'
    runtime.mkdir(parents=True)
    preserved_runtime = base/'original-best-resume-source-20261005'
    original_base = preserved_runtime if preserved_runtime.exists() else base
    with tarfile.open(original_base/'runtime.tar.gz') as archive:
        archive.extractall(runtime, filter='data')
    original_manifest = saved['runtime_sha256']
    for name, expected_digest in original_manifest.items():
        assert digest(runtime/name) == expected_digest, name
    patches = ['src/training/cc3m_text.py', 'src/training/checkpoint_upload_queue.py',
               'src/training/fid_adaptive_schedule.py', 'src/training/warmup_cosine_schedule.py',
               'scripts/tools/resume_cc3m_text.py',
               'scripts/tools/prepare_cc3m_best_resume.py']
    for name in patches:
        target_path = runtime/name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/name, target_path)
    manifest = {name:digest(runtime/name) for name in sorted(set(original_manifest)|set(patches))}
    subprocess.run([sys.executable, '-c', 'import src.training.cc3m_text; import train'],
        cwd=runtime, env=dict(os.environ, PYTHONPATH=f'{runtime/"runtime"}:{runtime}'), check=True)
    (local/'assets').mkdir()
    for name, expected_digest in [('stage1.pt', saved['stage1_sha256']),
            ('train.pt', saved['cache_sha256']['train']),
            ('validation.pt', saved['cache_sha256']['validation'])]:
        path = previous/'assets'/name
        assert digest(path) == expected_digest, name
        (local/'assets'/name).hardlink_to(path)
    for name in ['torch','inductor','triton']:
        if (previous/name).exists():
            (local/name).symlink_to(previous/name, target_is_directory=True)
    target = copy.deepcopy(saved)
    for key in list(target):
        if 'migration' in key:
            target.pop(key)
    anchor = source['global_step']
    before_lr = source['optimizer']['param_groups'][0]['lr']
    multiplier = source['scheduler']['multiplier']
    new_peak = saved['min_lr']+(before_lr-saved['min_lr'])/multiplier
    target.update(epochs=args.epochs, lr=new_peak, warmup_epochs=0,
        stage2_initialization='resumed_original_best_fid',
        fid_lr_policy=dict(baseline_fid=saved['fid_lr_policy']['baseline_fid'],
            patience=saved['fid_lr_policy']['patience'], min_delta=saved['fid_lr_policy']['min_delta'],
            factor=saved['fid_lr_policy']['factor'], cooldown=saved['fid_lr_policy']['cooldown'],
            adaptive_reductions=False, decay_start_step=anchor,
            decay_steps=args.epochs*updates-anchor),
        checkpoint=str(local/'assets/stage1.pt'), token_cache=str(local/'assets/train.pt'),
        validation_cache=str(local/'assets/validation.pt'), local_checkpoints=str(local/'checkpoints'),
        runtime_sha256=manifest, resume=True, resume_checkpoint=None,
        evaluate_on_resume=True, resume_expected_step=anchor,
        resume_expected_metrics=source['metrics'], finish_epoch_on_resume=False,
        checkpoint_on_resume=False, checkpoint_best_fid_on_resume=False,
        checkpoint_async_upload=True, checkpoint_upload_retry_seconds=30,
        watchdog_timeout_seconds=900, supervisor_max_attempts=10,
        wandb_mode='online', track_inception_score=False,
        wandb_name='CC3M LASER | original best resumed | smooth cosine to 100 epochs | 8 H100')
    migration = dict(source_config_sha256=config_digest(saved), source_step=anchor,
        source_scheduler_sha256=config_digest(source['scheduler']))
    verified_target = dict(target, lr_schedule_migration=migration)
    verify_resume_config(saved, verified_target)
    optimizer = SimpleNamespace(param_groups=copy.deepcopy(source['optimizer']['param_groups']))
    scheduler = create_lr_scheduler(optimizer, verified_target, updates,
        anchor, source['scheduler'], saved)
    after_lr = optimizer.param_groups[0]['lr']
    assert abs(after_lr-before_lr) < 1e-15
    curve = [FidAdaptiveSchedule.lr_at_step(scheduler.policy, step, multiplier)
             for step in range(anchor, args.epochs*updates+1)]
    assert all(a >= b for a,b in zip(curve, curve[1:]))
    assert curve[-1] == target['min_lr']
    payload = dict(source, config=target, scheduler=scheduler.state_dict(),
        optimizer=dict(source['optimizer'], param_groups=optimizer.param_groups),
        checkpoint_migration=dict(timestamp=time.time(), source_md5=source_md5,
            source_step=anchor, source_lr=before_lr, target_lr=after_lr,
            source_config_sha256=migration['source_config_sha256']),
        checkpoint_slots=['last.pt','best-fid.pt','best-clip.pt'])
    (local/'checkpoints').mkdir()
    destination = local/'checkpoints/last.pt'
    rq.atomic_torch_save(payload, destination)
    restored = torch.load(destination, map_location='cpu', weights_only=True, mmap=True)
    verify_checkpoint_progress(restored, updates, saved['accumulation'], 8)
    verify_resume_config(restored['config'], target)
    for field in ['model','rng_state_by_rank','epoch','next_microbatch','global_step',
                  'world_size','best_fid','best_clip','metrics','cache_meta','bpe_policy']:
        assert_identical(source[field], restored[field])
    assert_identical(source['optimizer']['state'], restored['optimizer']['state'])
    assert_identical(scheduler.state_dict(), restored['scheduler'])
    assert restored['scheduler']['last_epoch'] == anchor
    for slot in ['best-fid.pt','best-clip.pt']:
        (local/'checkpoints'/slot).hardlink_to(destination)
    archive = base/'original-best-resume-source-20261005'
    if not archive.exists():
        archive.mkdir()
        for name in ['recipe.yaml','runtime-manifest.json','runtime.tar.gz','resume.py']:
            shutil.copyfile(base/name, archive/name)
    preserved = previous/'checkpoints/original-best-resume-source-20261005'
    preserved.mkdir(exist_ok=True)
    for name in ['last.pt','best-fid.pt','best-clip.pt']:
        if not (preserved/name).exists():
            (preserved/name).hardlink_to(previous/'checkpoints'/name)
    cfg = OmegaConf.load(base/'recipe.yaml')
    cfg.options = target
    OmegaConf.save(cfg,base/'recipe.yaml')
    OmegaConf.save(cfg,root/'configs/stage2'/(target['wandb_id']+'.yaml'))
    with tarfile.open(base/'runtime.tar.gz','w:gz') as output:
        for name in manifest:
            output.add(runtime/name,arcname=name)
    (base/'runtime-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    shutil.copyfile(root/'scripts/tools/resume_cc3m_text.py',base/'resume.py')
    milestones = {str(epoch):FidAdaptiveSchedule.lr_at_step(scheduler.policy,epoch*updates,multiplier)
                  for epoch in [15,16,20,30,50,75,100]}
    report = dict(timestamp=time.time(),source_checkpoint=str(source_path),
        source_step=anchor, source_epoch=source['epoch'], source_md5=source_md5,
        target_md5=digest(destination,'md5'),bytes=destination.stat().st_size,
        target_epochs=args.epochs, total_steps=args.epochs*updates,
        metrics=source['metrics'], lr_before=before_lr, lr_after=after_lr,
        lr_schedule_milestones=milestones, adaptive_reductions=False,
        weights_exact=True, adam_moments_and_steps_exact=True, all_rank_rng_exact=True,
        data_cursor_exact=True, best_metrics_exact=True, controller_history_exact=True,
        all_remaining_lr_steps_monotonic=True, source_migration=migration,
        checkpoints=['last.pt','best-fid.pt','best-clip.pt'],
        local=str(local), runtime_sha256=manifest)
    (base/'original-best-resume-plan.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key!='runtime_sha256'}),flush=True)


if __name__ == '__main__':
    main()
