"""Freeze and verify a FID-adaptive continuation from the run's best full state."""
import argparse
import base64
import difflib
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time

from omegaconf import OmegaConf
import torch
import wandb


def digest(path, algorithm='sha256'):
    with Path(path).open('rb') as reader:
        value = hashlib.file_digest(reader, algorithm)
    return base64.b64encode(value.digest()).decode() if algorithm == 'md5' else value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--local', type=Path, required=True)
    parser.add_argument('--api-key-file', type=Path, required=True)
    parser.add_argument('--source-checkpoint', type=Path)
    parser.add_argument('--runtime-dir', type=Path)
    parser.add_argument('--source-runtime-base', type=Path)
    parser.add_argument('--pending-evaluation', type=Path,
                        help='Replay a completed evaluation whose checkpoint commit was interrupted')
    parser.add_argument('--start-lr', type=float,
                        help='Exact LR at the saved global step, without resetting progress')
    parser.add_argument('--min-lr', type=float, default=.00002)
    parser.add_argument('--fid-patience', type=int, default=3)
    parser.add_argument('--decay-epochs', type=int,
                        help='Anneal over this many additional epochs, preserving the saved global step')
    parser.add_argument('--nonblocking-checkpoints', action='store_true')
    parser.add_argument('--evaluate-on-resume', action='store_true')
    args = parser.parse_args()
    os.environ['WANDB_API_KEY'] = args.api_key_file.read_text().strip()
    base, local = args.base.resolve(), args.local.resolve()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    checkpoint = (args.source_checkpoint or local / 'checkpoints/source-best-fid.pt').resolve()
    state = torch.load(checkpoint, weights_only=True, map_location='cpu', mmap=True)
    source = state['config']
    run_path = f"{source['wandb_entity']}/{source['wandb_project']}/{source['wandb_id']}"
    run = wandb.Api(timeout=60).run(run_path)
    remote = run.file('best-fid.pt')
    md5 = digest(checkpoint, 'md5')
    online_verified = remote.size == checkpoint.stat().st_size and remote.md5 == md5
    pending = json.loads(args.pending_evaluation.read_text()) if args.pending_evaluation else None
    updates = source['train_items'] // source['total_batch_size']
    if pending is not None:
        if (state['next_microbatch'] != updates * source['accumulation']
                or state['global_step'] != (state['epoch'] + 1) * updates
                or state['scheduler']['last_epoch'] != state['global_step']
                or pending['fid'] >= state['best_fid']
                or pending['items'] != source['validation_items']
                or any(float(s['step']) != state['global_step'] for s in state['optimizer']['state'].values())):
            raise ValueError('Pending best evaluation must match a full epoch-end optimizer checkpoint')
    if not online_verified and pending is None:
        raise ValueError('Local best FID checkpoint differs from W&B')
    if (not state['resume_capable'] or state['world_size'] != 8
            or len(state['rng_state_by_rank']) != 8 or not state['optimizer']['state']
            or (pending is None and state['metrics']['fid'] != state['best_fid'])):
        raise ValueError('Source is not the resumable best FID checkpoint')
    for name, expected in [('stage1.pt', source['stage1_sha256']),
                           ('train.pt', source['cache_sha256']['train']),
                           ('validation.pt', source['cache_sha256']['validation'])]:
        if digest(local / 'assets' / name) != expected:
            raise ValueError(f'Asset checksum mismatch: {name}')
    predecessor = base / ('continuation-source' if state['scheduler'].get('kind') is None
                          else f"continuation-source-step-{state['global_step']}")
    predecessor.mkdir(exist_ok=True)
    source_runtime_base = (args.source_runtime_base or base).resolve()
    for name in ['recipe.yaml', 'runtime.tar.gz', 'runtime-manifest.json', 'environment.json']:
        destination = predecessor / name
        if not destination.exists():
            shutil.copyfile(source_runtime_base / name, destination)
    runtime = (args.runtime_dir or local / 'runtime').resolve()
    runtime.mkdir(exist_ok=True)
    with tarfile.open(predecessor / 'runtime.tar.gz') as archive:
        archive.extractall(runtime, filter='data')
    original_manifest = source['runtime_sha256']
    for name, expected in original_manifest.items():
        if digest(runtime / name) != expected:
            raise ValueError(f'Original runtime checksum mismatch: {name}')
    patches = ['src/training/cc3m_text.py', 'src/training/cc3m_compound.py',
               'src/training/fid_adaptive_schedule.py', 'src/training/checkpoint_upload_queue.py',
               'scripts/tools/resume_cc3m_text.py']
    changes = []
    for name in patches:
        destination = runtime / name
        before = destination.read_text() if destination.exists() else ''
        after = (root / name).read_text()
        changes.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                           fromfile='source/' + name, tofile='continuation/' + name))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(after)
    manifest = {name: digest(runtime / name) for name in sorted(set(original_manifest) | set(patches))}
    completed = state['global_step']
    total_steps = source['epochs'] * (source['train_items'] // source['total_batch_size'])
    multiplier = state['scheduler'].get('multiplier', 1.)
    cosine = 1. if args.decay_epochs else .5 * (1 + math.cos(math.pi * completed / total_steps))
    initial_lr = .00025 if args.start_lr is None else (
        args.min_lr + (args.start_lr - args.min_lr) / (cosine * multiplier))
    fid_policy = dict(source.get('fid_lr_policy') or dict(
        baseline_fid=state['best_fid'], min_delta=.1, factor=.5, cooldown=1))
    fid_policy['patience'] = args.fid_patience
    if args.decay_epochs:
        fid_policy.update(decay_start_step=completed, decay_steps=args.decay_epochs * updates)
    migration = dict(source_config_sha256=hashlib.sha256(json.dumps(source,
        sort_keys=True, separators=(',', ':')).encode()).hexdigest(), source_step=completed)
    if state['scheduler'].get('kind') == 'fid-adaptive-cosine-v1':
        migration['source_scheduler_sha256'] = hashlib.sha256(json.dumps(state['scheduler'],
            sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    options = dict(source)
    for key in ('finish_epoch_on_resume', 'resume_expected_metrics',
                'checkpoint_best_fid_on_resume', 'resume_global_best_metrics'):
        options.pop(key, None)
    # Run-wide winners can be newer than the selected model checkpoint.
    # Preserve their thresholds while keeping all training state from the source.
    summary = run.summary._json_dict
    global_best = dict(fid=min(state['best_fid'], summary.get('best/fid', state['best_fid'])),
        clip_score=max(state['best_clip'], summary.get('best/clip_score', state['best_clip'])))
    options.update(checkpoint=str(local / 'assets/stage1.pt'),
        token_cache=str(local / 'assets/train.pt'), validation_cache=str(local / 'assets/validation.pt'),
        resume_checkpoint=str(checkpoint), local_checkpoints=str(local / 'checkpoints'),
        output=str(base / 'train'), official_stage2_config=str(base / 'official-stage2.yaml'),
        fid_reference_stats=str(base / 'assets/cc3m-validation-fid.npz'),
        lr=initial_lr, min_lr=args.min_lr, lr_schedule='fid_adaptive_cosine',
        fid_lr_policy=fid_policy, lr_schedule_migration=migration, runtime_sha256=manifest,
        track_inception_score=False, evaluate_on_resume=args.evaluate_on_resume,
        checkpoint_async_upload=args.nonblocking_checkpoints,
        preview_on_resume=False, checkpoint_on_resume=False,
        wandb_mode='online', upload_checkpoints=True, full_state_best_checkpoints=True)
    if pending is None and args.evaluate_on_resume:
        options.update(resume_expected_metrics={key: state['metrics'][key]
            for key in ('fid', 'clip_score', 'items')},
            resume_global_best_metrics=global_best, checkpoint_best_fid_on_resume=True)
    if pending is not None:
        options.update(evaluate_on_resume=True, finish_epoch_on_resume=True,
            resume_expected_metrics={key: pending[key] for key in ('fid', 'clip_score', 'items')})
    config = OmegaConf.create(dict(defaults=['_self_'], stage='stage2', backend='cc3m_text', options=options))
    from src.training.cc3m_text import verify_resume_config, create_lr_scheduler
    verify_resume_config(source, options)
    # Validate the actual saved scheduler against its optimizer LR without
    # copying the model or modifying any of its Adam state tensors.
    from types import SimpleNamespace
    optimizer = SimpleNamespace(param_groups=[dict(g) for g in state['optimizer']['param_groups']])
    schedule = create_lr_scheduler(optimizer, options,
        source['train_items'] // source['total_batch_size'], state['global_step'], state['scheduler'], source)
    with tarfile.open(base / 'runtime.tar.gz', 'w:gz') as archive:
        for name in manifest:
            archive.add(runtime / name, arcname=name)
    write_json(base / 'runtime-manifest.json', manifest)
    OmegaConf.save(config, base / 'recipe.yaml')
    OmegaConf.save(config, root / 'configs/stage2/cc3m-rfid421-physicalpairs-650m-8h100-20261004-fid-adaptive.yaml')
    shutil.copyfile(root / 'scripts/tools/resume_cc3m_text.py', base / 'resume.py')
    (base / 'continuation.diff').write_text(''.join(changes))
    packages = ['torch', 'torchvision', 'wandb', 'hydra-core', 'omegaconf', 'numpy', 'scipy',
                'tokenizers', 'openai-clip', 'torchmetrics', 'torch-fidelity', 'triton', 'einops', 'easydict', 'lmdb']
    write_json(base / 'environment.json', {name: importlib.metadata.version(name) for name in packages})
    provenance = dict(timestamp=time.time(), source_run=run_path,
        source_checkpoint=checkpoint.name if pending else 'best-fid.pt', source_local_checkpoint=str(checkpoint),
        source_checkpoint_sha256=digest(checkpoint), source_checkpoint_md5=md5,
        online_source_verified=online_verified, source_step=state['global_step'], source_epoch=state['epoch'],
        source_fid=pending['fid'] if pending else state['best_fid'],
        evaluation_replay_required=pending is not None,
        source_lr=state['optimizer']['param_groups'][0]['lr'],
        continuation_start_lr=optimizer.param_groups[0]['lr'], schedule=schedule.state_dict(),
        optimizer_states=len(state['optimizer']['state']), rank_rng_count=len(state['rng_state_by_rank']),
        checkpoint_slots=['last.pt', 'best-fid.pt', 'best-clip.pt'],
        preserved_run_best_metrics=global_best,
        upload_policy='full optimizer, scheduler, data cursor and all rank RNG; verify remote size and MD5',
        target_epochs=options['epochs'], changed_runtime_files=patches)
    write_json(base / 'continuation-provenance.json', provenance)
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
