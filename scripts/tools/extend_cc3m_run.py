"""Extend a preserved CC3M full state without changing weights or Adam moments."""
import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time
from types import SimpleNamespace

from omegaconf import OmegaConf
import torch

from src.training.cc3m_text import (config_digest, create_lr_scheduler,
    verify_checkpoint_progress, verify_resume_config)
from src.training import rqtransformer as rq


def digest(path, algorithm='sha256'):
    with Path(path).open('rb') as stream:
        value = hashlib.file_digest(stream, algorithm).digest()
    return base64.b64encode(value).decode() if algorithm == 'md5' else value.hex()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--source-local', type=Path, required=True)
    parser.add_argument('--local', type=Path, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    args = parser.parse_args()
    base, previous, local = args.base.resolve(), args.source_local.resolve(), args.local.resolve()
    root = Path(__file__).resolve().parents[2]
    source_dir = previous/'checkpoints/recovery-source-20261005'
    source = torch.load(source_dir/'last.pt', map_location='cpu', weights_only=True, mmap=True)
    saved = source['config']
    updates = saved['train_items']//saved['total_batch_size']
    verify_checkpoint_progress(source, updates, saved['accumulation'], 8)
    assert args.epochs > saved['epochs']
    assert not (local/'checkpoints/last.pt').exists()
    original = json.loads((base/'runtime-manifest.json').read_text())
    runtime = local/'runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    with tarfile.open(base/'runtime.tar.gz') as archive:
        archive.extractall(runtime, filter='data')
    for name, expected in original.items():
        assert digest(runtime/name) == expected
    patches = ['src/training/cc3m_text.py', 'src/training/checkpoint_upload_queue.py',
               'scripts/tools/resume_cc3m_text.py', 'scripts/tools/extend_cc3m_run.py']
    for name in patches:
        target = runtime/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/name, target)
    manifest = {name:digest(runtime/name) for name in sorted(set(original)|set(patches))}
    (local/'assets').mkdir(exist_ok=True)
    for name, expected in [('stage1.pt', saved['stage1_sha256']),
            ('train.pt', saved['cache_sha256']['train']),
            ('validation.pt', saved['cache_sha256']['validation'])]:
        path = previous/'assets'/name
        assert digest(path) == expected
        (local/'assets'/name).hardlink_to(path)
    for name in ['torch','inductor','triton']:
        if (previous/name).exists():
            (local/name).symlink_to(previous/name, target_is_directory=True)
    target = copy.deepcopy(saved)
    anchor = source['global_step']
    multiplier = source['scheduler']['multiplier']
    before_lr = source['optimizer']['param_groups'][0]['lr']
    new_peak = saved['min_lr'] + (before_lr-saved['min_lr'])/multiplier
    target.update(epochs=args.epochs, lr=new_peak, warmup_epochs=0,
        original_warmup_epochs=saved['warmup_epochs'], original_peak_lr=saved['lr'],
        fid_lr_policy=dict(baseline_fid=None, patience=3, min_delta=.1, factor=.5,
                           cooldown=2, decay_start_step=anchor,
                           decay_steps=args.epochs*updates-anchor),
        checkpoint=str(local/'assets/stage1.pt'), token_cache=str(local/'assets/train.pt'),
        validation_cache=str(local/'assets/validation.pt'),
        local_checkpoints=str(local/'checkpoints'), runtime_sha256=manifest,
        resume=True, resume_checkpoint=None, checkpoint_on_resume=True,
        checkpoint_upload_retry_seconds=30, watchdog_timeout_seconds=900,
        supervisor_max_attempts=10,
        wandb_name='CC3M LASER | matched sampler | 100 epochs | resume-safe cosine | 8 H100')
    for key in list(target):
        if 'migration' in key:
            target.pop(key)
    options_migration = {}
    groups = {}
    records = []
    (local/'checkpoints').mkdir(exist_ok=True)
    for name in ['last.pt','best-fid.pt','best-clip.pt']:
        path = source_dir/name
        source_md5 = digest(path, 'md5')
        if source_md5 in groups:
            (local/'checkpoints'/name).hardlink_to(groups[source_md5])
            records.append(dict(name=name,shared_with=groups[source_md5].name))
            continue
        state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        verify_checkpoint_progress(state, updates, saved['accumulation'], 8)
        assert config_digest(state['config']) == config_digest(saved)
        migration = dict(source_config_sha256=config_digest(state['config']),
            source_step=state['global_step'],
            source_scheduler_sha256=config_digest(state['scheduler']))
        options_migration[name] = migration
        verified_target = dict(target, lr_schedule_migration=migration)
        verify_resume_config(state['config'], verified_target)
        optimizer = SimpleNamespace(param_groups=copy.deepcopy(state['optimizer']['param_groups']))
        scheduler = create_lr_scheduler(optimizer, verified_target, updates,
            state['global_step'], state['scheduler'], state['config'])
        payload = dict(state, config=target, scheduler=scheduler.state_dict(),
            optimizer=dict(state['optimizer'], param_groups=optimizer.param_groups),
            checkpoint_migration=dict(timestamp=time.time(), source_md5=source_md5,
                source_epochs=state['config']['epochs'], target_epochs=args.epochs,
                source_lr=state['optimizer']['param_groups'][0]['lr'],
                target_lr=optimizer.param_groups[0]['lr'],
                source_config_sha256=migration['source_config_sha256']))
        destination = local/'checkpoints'/name
        rq.atomic_torch_save(payload, destination)
        restored = torch.load(destination, map_location='cpu', weights_only=True, mmap=True)
        verify_checkpoint_progress(restored, updates, saved['accumulation'], 8)
        verify_resume_config(restored['config'], target)
        for key, value in state['model'].items():
            assert torch.equal(value, restored['model'][key]), 'Model changed: '+key
        for key, values in state['optimizer']['state'].items():
            for field, value in values.items():
                other = restored['optimizer']['state'][key][field]
                assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other
        assert restored['rng_state_by_rank'] is not None
        assert restored['scheduler']['last_epoch'] == state['scheduler']['last_epoch']
        if name=='last.pt':
            assert abs(restored['optimizer']['param_groups'][0]['lr']-before_lr) < 1e-15
        groups[source_md5] = destination
        records.append(dict(name=name,step=state['global_step'],epoch=state['epoch'],
            source_md5=source_md5,target_md5=digest(destination,'md5'),
            bytes=destination.stat().st_size,weights_and_moments_identical=True,
            source_lr=state['optimizer']['param_groups'][0]['lr'],
            target_lr=optimizer.param_groups[0]['lr']))
        print(json.dumps(records[-1]), flush=True)
        del restored, payload, state
    archive = base/'recovery-source-20261005'
    archive.mkdir(exist_ok=True)
    for name in ['recipe.yaml','runtime-manifest.json','runtime.tar.gz','resume.py']:
        shutil.copyfile(base/name, archive/name)
    cfg = OmegaConf.load(base/'recipe.yaml')
    cfg.options = target
    OmegaConf.save(cfg,base/'recipe.yaml')
    OmegaConf.save(cfg,root/'configs/stage2'/(target['wandb_id']+'.yaml'))
    with tarfile.open(base/'runtime.tar.gz','w:gz') as output:
        for name in manifest:output.add(runtime/name,arcname=name)
    (base/'runtime-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    shutil.copyfile(root/'scripts/tools/resume_cc3m_text.py',base/'resume.py')
    report = dict(timestamp=time.time(),source_step=anchor,source_epochs=saved['epochs'],
        target_epochs=args.epochs,total_steps=args.epochs*updates,
        remaining_cosine_steps=args.epochs*updates-anchor,
        lr_before=before_lr,lr_after=before_lr,min_lr=target['min_lr'],
        checkpoints=records,source_migrations=options_migration,
        preserved=['model','Adam moments','all rank RNG','data cursor','FID controller'],
        fixes=['retry transient uploads','recover pending checkpoint slots',
               'log rank failures before teardown','watchdog stalled workers',
               'require requested epoch completion before reporting success'])
    (base/'extension-100-epochs.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(prepared=True,local=str(local),epochs=args.epochs,
                         step=anchor,lr=before_lr)),flush=True)


if __name__ == '__main__':
    main()
