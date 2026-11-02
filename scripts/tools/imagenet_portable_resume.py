#!/usr/bin/env python3
"""Download, verify and resume the frozen ImageNet low-noise recovery artifact."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile


def digest(path, algorithm='sha256'):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def checkpoint_metadata(path, *, finite=False):
    import torch
    payload = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    required = {'state_dict', 'optimizer', 'scheduler', 'rng_state_by_rank',
                'checkpoint_world_size', 'epoch', 'global_step', 'config'}
    if not required <= payload.keys():
        raise ValueError('Incomplete recovery checkpoint')
    config = payload['config']
    world = int(payload['checkpoint_world_size'])
    if world != config['world_size'] or len(payload['rng_state_by_rank']) != world:
        raise ValueError('Missing per-rank RNG states')
    for state in payload['rng_state_by_rank']:
        if not {'torch_cpu', 'torch_cuda'} <= state.keys():
            raise ValueError('Missing CPU/CUDA RNG stream')
    optimizer = payload['optimizer']
    parameters = [p for g in optimizer['param_groups'] for p in g['params']]
    if set(parameters) != set(optimizer['state']) or not parameters:
        raise ValueError('Missing optimizer moments')
    steps = set()
    for state in optimizer['state'].values():
        if not {'step', 'exp_avg', 'exp_avg_sq'} <= state.keys():
            raise ValueError('Incomplete Adam state')
        steps.add(int(state['step']))
        if finite and any(not bool(torch.isfinite(state[k]).all()) for k in ('exp_avg', 'exp_avg_sq')):
            raise ValueError('Nonfinite Adam state')
    if steps != {int(payload['global_step'])}:
        raise ValueError('Adam age and global step disagree')
    if finite and any(not bool(torch.isfinite(v).all()) for v in payload['state_dict'].values()
                      if torch.is_floating_point(v)):
        raise ValueError('Nonfinite model state')
    scheduler = payload['scheduler']
    if scheduler is None or scheduler['last_epoch'] != payload['global_step'] or scheduler['T_max'] != 62600:
        raise ValueError('Incomplete or restarted cosine scheduler')
    lr = [g['lr'] for g in optimizer['param_groups']]
    if lr != scheduler['_last_lr']:
        raise ValueError('Optimizer and scheduler learning rates disagree')
    batch = int(payload.get('batch_idx', 0))
    accumulation = int(config['accumulation_steps'])
    epoch = int(payload['epoch'])
    if batch % accumulation or epoch * 626 + batch // accumulation != payload['global_step']:
        raise ValueError('Invalid optimizer-boundary data cursor')
    if config['total_batch_size'] != 2048 or not config['exact_global_batch']:
        raise ValueError('Unexpected global batch policy')
    result = dict(epoch=epoch, next_microbatch=batch, global_step=payload['global_step'],
                  optimizer='AdamW', optimizer_parameter_states=len(parameters),
                  learning_rates=lr, scheduler=scheduler, rng_ranks=world,
                  saved_world_size=world, saved_accumulation=accumulation,
                  global_batch=2048, steps_per_epoch=626, training_images=config['training_images'],
                  coefficient_scales=config['coeff_scales'], target_temperature=config['coeff_target_temperature'],
                  coefficient_clipping=config['coefficient_clipping'],
                  training_precision='bfloat16', gradient_scaler_enabled=False,
                  ema_enabled=False, finite_checked=finite,
                  best_fid=payload.get('best_fid', []), best_inception=payload.get('best_inception', []))
    del payload
    return result


def verify(directory, *, finite=False):
    directory = Path(directory)
    manifest = json.loads((directory / 'recovery-manifest.json').read_text())
    for item in manifest['files']:
        path = directory / item['name']
        if path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
            raise ValueError(f"Recovery checksum mismatch: {item['name']}")
    states = {name: checkpoint_metadata(directory / filename, finite=finite)
              for name, filename in manifest['slots'].items()}
    for name, state in states.items():
        if state['global_step'] != manifest['checkpoints'][name]['global_step']:
            raise ValueError(f'Wrong recovery checkpoint: {name}')
    write_json(directory / 'download-verification.json', dict(passed=True, checkpoints=states,
                                                             verified_files=len(manifest['files'])))
    return manifest, states


def dataset_identity(root):
    from torchvision.datasets import ImageFolder
    dataset = ImageFolder(Path(root) / 'train')
    h = hashlib.sha256()
    train = Path(root) / 'train'
    for path, label in dataset.samples:
        h.update(f'{Path(path).relative_to(train).as_posix()}\t{label}\n'.encode())
    return dict(images=len(dataset), classes=dataset.classes, ordered_paths_sha256=h.hexdigest())


def make_entry(original):
    changes = {
        "assert world == 4 and backend == 'ddp'":
            "assert world == int(os.environ['LASER_EXPECTED_WORLD_SIZE']) and backend == 'ddp'",
        'native_checkpoint_load = torch.load':
            'from src.training import portable_imagenet_recovery as portable_recovery\n'
            'training.validate_resume_layout = portable_recovery.validate_layout\n'
            'training.restore_rank_rng_state = portable_recovery.restore_rng\n'
            'def portable_checkpoint_directory(output, checkpoint_dir):\n'
            "    expected = OUT/'train/checkpoints'\n"
            '    if checkpoint_dir is None or Path(checkpoint_dir) != expected:\n'
            "        raise ValueError('Checkpoint directory must match the explicit recovery output root')\n"
            '    expected.mkdir(parents=True, exist_ok=True)\n'
            '    return expected\n'
            'training.persistent_checkpoint_dir = portable_checkpoint_directory\n'
            'native_checkpoint_load = torch.load',
        "payload = original_logging.rebase_checkpoint_fids(payload, policy['previous_evaluations'])":
            "payload = original_logging.rebase_checkpoint_fids(payload, policy['previous_evaluations'])\n"
            "        payload = portable_recovery.remap_payload(payload,\n"
            "            world=int(os.environ['WORLD_SIZE']), accumulation=int(os.environ['LASER_ACCUMULATION']),\n"
            "            checkpoint_dir=OUT/'train/checkpoints', reference=ARGS.fid_reference_stats,\n"
            "            reference_sha256=os.environ['LASER_REFERENCE_SHA256'])\n"
            "        record('cursor-resume-rank'+os.environ['RANK']+'.json',\n"
            "            dict(passed=True, **payload['config']['portable_resume_cursor']))",
    }
    for before, after in changes.items():
        if original.count(before) != 1:
            raise ValueError(f'Unexpected frozen entry structure: {before}')
        original = original.replace(before, after)
    compile(original, 'resume_entry.py', 'exec')
    return original


def launch(args):
    import torch
    import yaml
    directory = args.directory.resolve()
    manifest, states = verify(directory)
    world = args.gpus or torch.cuda.device_count()
    if not 1 <= world <= torch.cuda.device_count():
        raise ValueError('Requested GPU count is unavailable')
    maximum = args.microbatch or (128 if min(torch.cuda.get_device_properties(r).total_memory
                                            for r in range(world)) >= 120 * 2**30 else 64)
    if maximum < 1:
        raise ValueError('Microbatch must be positive')
    accumulation = math.ceil(2048 / (world * maximum))
    microbatch = math.ceil(2048 / (world * accumulation))
    identity = dataset_identity(args.data.resolve())
    if identity != manifest['dataset']:
        raise ValueError('ImageNet image count, classes or ordered sample paths changed')
    frozen = directory / 'frozen'
    if not frozen.exists():
        with tarfile.open(directory / 'recovery-code.tar.gz') as archive:
            archive.extractall(directory, filter='data')
    base = args.work.resolve()
    out = args.output.resolve()
    base.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    source = base / 'source'
    if not source.exists():
        source.symlink_to(frozen / 'source', target_is_directory=True)
    inputs = base / 'inputs'
    inputs.mkdir(exist_ok=True)
    for name in ('stage1-tokenizer.pt', 'imagenet_256_train.npz'):
        p = inputs / name
        if not p.exists(): p.symlink_to(directory / name)
    data = base / 'imagenet'
    data.mkdir(exist_ok=True)
    if not (data / 'train').exists(): (data / 'train').symlink_to(args.data.resolve() / 'train')
    write_json(data / 'training-ready.json', dict(passed=True, training_images=1281167, classes=1000,
                                                ordered_paths_sha256=identity['ordered_paths_sha256']))
    checkpoint_dir = out / 'train/checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for slot, filename in manifest['slots'].items():
        alias = checkpoint_dir / ('last.pt' if slot == 'latest' else manifest['checkpoint_aliases'][slot])
        if not alias.exists(): alias.symlink_to(directory / filename)
    # Ranking lists may refer to both FID and IS winners sharing one full checkpoint.
    for name, filename in manifest['winner_aliases'].items():
        alias = checkpoint_dir / name
        if not alias.exists(): alias.symlink_to(directory / filename)
    selected = checkpoint_dir / ('last.pt' if args.slot == 'latest' else manifest['checkpoint_aliases'][args.slot])
    for name in ('coefficient-support-policy.json', 'support-validation.json', 'original-fid-only-policy.json',
                 'preflight-verification.json', 'restart-verification.json', 'source-manifest.json',
                 'plan.json', 'environment.txt', 'validation-reference-verification.json'):
        (out / name).write_bytes((directory / name).read_bytes())
    for name in ('loss-balance-policy.json', 'loss-balance-gradient-calibration.json',
                 'loss-balance-test-verification.json', 'loss-balance-resume-verification.json'):
        candidates = [directory/name, frozen/name]
        source = next((p for p in candidates if p.is_file()), None)
        if source is not None:
            (out/name).write_bytes(source.read_bytes())
    torch_home = base / 'torch-cache'
    hub = torch_home / 'hub/checkpoints'
    hub.mkdir(parents=True, exist_ok=True)
    for name in manifest['inception_weight_files']:
        if not (hub/name).exists(): (hub/name).symlink_to(directory/name)
    for name in ('checkpoint-staging','checkpoint-upload-cache','wandb','wandb-cache','wandb-data','inductor-cache'):
        (base/name).mkdir(exist_ok=True)
    (base/'entry.py').write_text((frozen/'resume_entry.py').read_text())
    recipe = yaml.safe_load((frozen/'production.yaml').read_text())
    opts = recipe['options']
    opts.update(checkpoint=str(inputs/'stage1-tokenizer.pt'), data=str(data), output=str(out/'train'),
                checkpoint_dir=str(checkpoint_dir), resume=True, resume_checkpoint=str(selected),
                init_stage2_checkpoint=None, batch_size=microbatch, total_batch_size=2048,
                fid_reference_stats=str(inputs/'imagenet_256_train.npz'),
                wandb_id=manifest['run_id'], wandb_mode='disabled' if args.validate_updates else 'online')
    if args.validate_updates:
        opts.update(max_optimizer_steps=args.validate_updates, fid_every=0,
                    sample_grid_every=0, save_step_freq=0, upload_checkpoints=False)
    else:
        opts.update(max_optimizer_steps=0)
    config = base/'resume.yaml'
    config.write_text(yaml.safe_dump(recipe, sort_keys=False))
    environment = dict(os.environ, LASER_RUN_BASE=str(base), LASER_PERSISTENT_BASE=str(out),
                       LASER_PHASE='production', LASER_EXPECTED_WORLD_SIZE=str(world),
                       LASER_ACCUMULATION=str(accumulation), LASER_FFHQ_DECODER_OBJECTIVE='1',
                       LASER_SUB_BIN_TARGETS='0', LASER_COEFFICIENT_SIGMA_BINS=str(.125/(6/2047)),
                       LASER_CHECKPOINT_STAGING_DIR=str(base/'checkpoint-staging'),
                       LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(base/'checkpoint-upload-cache'),
                       LASER_CHECKPOINT_IMMUTABLE_FILES='1',
                       LASER_REFERENCE_SHA256=manifest['reference_sha256'],
                       LASER_LAYOUT_MIGRATION_DIR=str(out/'verification/layout'),
                       PYTHONPATH=str(source)+':'+str(source/'runtime'),
                       TORCH_HOME=str(torch_home), TORCHINDUCTOR_CACHE_DIR=str(base/'inductor-cache'),
                       TORCHINDUCTOR_COMPILE_THREADS='4', WANDB_DIR=str(base/'wandb'),
                       WANDB_CACHE_DIR=str(base/'wandb-cache'), WANDB_DATA_DIR=str(base/'wandb-data'),
                       OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                       PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', NCCL_NVLS_ENABLE='0',
                       PYTHONUNBUFFERED='1')
    command = [sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc-per-node={world}',
               str(base/'entry.py'),'--config',str(config)]
    plan = dict(run_id=manifest['run_id'], checkpoint_slot=args.slot,
                resumed_step=states[args.slot]['global_step'], gpus=world,
                microbatch=microbatch, accumulation=accumulation, global_batch=2048,
                cosine_total_steps=62600, validate_updates=args.validate_updates,
                generation_metric_keys=['eval/fid_original_train50k','eval/inception_score'],
                saved_rng_rank_count_matches=world==states[args.slot]['saved_world_size'],
                trajectory_bitwise_identical_guaranteed=False,
                command=command)
    write_json(out/'portable-launch.json',plan)
    print(json.dumps(plan,indent=2),flush=True)
    if args.prepare_only: return
    with (out/'train.log').open('a') as log:
        result = subprocess.run(command, cwd=source, env=environment, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode: raise SystemExit(result.returncode)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    download=commands.add_parser('download')
    download.add_argument('--artifact',required=True)
    download.add_argument('--directory',type=Path,required=True)
    check=commands.add_parser('verify')
    check.add_argument('--directory',type=Path,required=True)
    check.add_argument('--finite',action='store_true')
    run=commands.add_parser('launch')
    run.add_argument('--directory',type=Path,required=True)
    run.add_argument('--data',type=Path,required=True,help='Original ImageNet root with train/<class>/*.JPEG')
    run.add_argument('--work',type=Path,required=True)
    run.add_argument('--output',type=Path,required=True)
    run.add_argument('--gpus',type=int)
    run.add_argument('--microbatch',type=int)
    run.add_argument('--slot',choices=['latest','best_fid','best_is'],default='latest')
    run.add_argument('--validate-updates',type=int,default=0)
    run.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if args.command=='download':
        import wandb
        artifact=wandb.Api(timeout=120).artifact(args.artifact,type='full-recovery')
        artifact.download(root=str(args.directory.resolve()))
        _,states=verify(args.directory,finite=True)
        print(json.dumps(dict(download_verified=True,checkpoints=states),indent=2))
    elif args.command=='verify':
        _,states=verify(args.directory,finite=args.finite)
        print(json.dumps(states,indent=2))
    else: launch(args)


if __name__=='__main__':main()
