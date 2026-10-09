"""Restore immutable run assets and supervise an exactly resumable CC3M run."""
import argparse
import fcntl
import hashlib
import json
import os
import pickle
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time

from omegaconf import OmegaConf


def newest_matching_checkpoint(options, paths):
    """Prefer the newest locally committed state from this exact continuation."""
    import torch
    candidates = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            if (state.get('resume_capable')
                    and state['config'].get('runtime_sha256') == options['runtime_sha256']
                    and state['config'].get('fid_lr_policy') == options.get('fid_lr_policy')
                    and state['config'].get('lr_schedule', 'cosine') == options.get('lr_schedule', 'cosine')
                    and state['config'].get('warmup_epochs', 0) == options.get('warmup_epochs', 0)
                    and state['config'].get('wandb_id') == options.get('wandb_id')
                    and state['scheduler']['last_epoch'] == state['global_step']):
                candidates.append((state['global_step'], path))
            del state
        except (OSError, EOFError, RuntimeError, KeyError, pickle.UnpicklingError) as error:
            print(f'Ignoring incomplete local checkpoint {path.name}: {type(error).__name__}', flush=True)
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--local', type=Path, default=Path('/mnt/laser-cc3m-20261004'))
    parser.add_argument('--checkpoint', choices=['last.pt', 'best-fid.pt', 'best-clip.pt'])
    parser.add_argument('--api-key-file', type=Path)
    parser.add_argument('--fresh', action='store_true',
                        help='Require an empty checkpoint directory and skip cloud restoration once')
    parser.add_argument('--local-only', action='store_true',
                        help='Use the validated local commit without restoring an older cloud file')
    args = parser.parse_args()
    if args.api_key_file:
        os.environ['WANDB_API_KEY'] = args.api_key_file.read_text().strip()
    base, local = args.base.resolve(), args.local.resolve()
    recipe = base / 'recipe.yaml'
    cfg = OmegaConf.load(recipe)
    options = OmegaConf.to_container(cfg.options, resolve=True)
    lock = open('/tmp/' + options['wandb_id'] + '.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    local.mkdir(parents=True, exist_ok=True)
    (local / 'assets').mkdir(exist_ok=True)
    for name in ['stage1.pt', 'train.pt', 'validation.pt']:
        target = local / 'assets' / name
        if not target.is_file():
            shutil.copyfile(base / 'assets' / name, target)
        if name != 'stage1.pt':
            split = 'train' if name == 'train.pt' else 'validation'
            with target.open('rb') as source:
                assert hashlib.file_digest(source, 'sha256').hexdigest() == options['cache_sha256'][split]
    runtime = local / 'runtime'
    runtime.mkdir(exist_ok=True)
    with tarfile.open(base / 'runtime.tar.gz') as archive:
        archive.extractall(runtime, filter='data')
    manifest = json.loads((base / 'runtime-manifest.json').read_text())
    for name, expected in manifest.items():
        assert hashlib.sha256((runtime / name).read_bytes()).hexdigest() == expected
    latest = Path(options['output']) / 'checkpoints/last.pt'
    restart_paths = [Path(options['local_checkpoints']) / 'last.pt', latest]
    if args.local_only:
        restart_paths = restart_paths[:1]
    if args.fresh and (args.checkpoint or cfg.options.get('resume_checkpoint')
                       or any(path.exists() for path in restart_paths)):
        raise ValueError('A fresh launch cannot contain an existing Stage 2 checkpoint')
    if not args.checkpoint and not cfg.options.get('resume_checkpoint'):
        matching = newest_matching_checkpoint(options, restart_paths)
        if matching is not None:
            cfg.options.resume_checkpoint = str(matching)
            OmegaConf.save(cfg, recipe)
    if args.checkpoint or (not args.fresh and not args.local_only and not latest.is_file()
                          and options.get('restore_from_wandb', True)):
        import wandb
        api = wandb.Api(timeout=60)
        run = api.run(f"{options['wandb_entity']}/{options['wandb_project']}/{options['wandb_id']}")
        name = args.checkpoint or 'last.pt'
        try:
            remote = run.file(name)
        except ValueError:
            remote = None
        if remote is not None and remote.size > 0:
            restored = base / 'restored'
            restored.mkdir(exist_ok=True)
            remote.download(root=str(restored), replace=True)
            cfg.options.resume_checkpoint = str(restored / name)
            OmegaConf.save(cfg, recipe)
        elif args.checkpoint:
            raise FileNotFoundError(f'{name} has no online checkpoint yet')
    env = os.environ.copy()
    env.update(PYTHONPATH=f'{runtime / "runtime"}:{runtime}',
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
        PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false',
        WANDB_MODE='online', WANDB_DIR=str(local), WANDB_CACHE_DIR=str(local / 'wandb-cache'),
        WANDB_DATA_DIR=str(local / 'wandb-data'), WANDB_ARTIFACT_DIR=str(local / 'wandb-artifacts'),
        TORCH_HOME=str(local / 'torch'), CC3M_LOCAL_SHARDS=str(local / 'validation-shards'),
        TORCHINDUCTOR_CACHE_DIR=str(local / 'inductor'), TRITON_CACHE_DIR=str(local / 'triton'),
        TORCHINDUCTOR_COMPILE_THREADS='2', NCCL_NVLS_ENABLE='0', TORCH_NCCL_ASYNC_ERROR_HANDLING='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    argv = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone',
        '--nproc-per-node=8', str(runtime / 'train.py'), '--config', str(recipe)]
    (base / 'supervisor.pid').write_text(str(os.getpid()) + '\n')
    for attempt in range(options.get('supervisor_max_attempts', 10)):
        status = dict(phase='starting_training', attempt=attempt, timestamp=time.time(),
            argv=argv, supervisor_pid=os.getpid())
        (base / 'status.json').write_text(json.dumps(status, indent=2))
        with (base / 'training.log').open('a') as log:
            child = subprocess.Popen(argv, cwd=runtime, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            (base / 'training.pid').write_text(str(child.pid) + '\n')
            print(f'Started torchrun PID {child.pid}, attempt {attempt}', flush=True)
            started = time.time()
            watchdog = options.get('watchdog_timeout_seconds', 900)
            while True:
                try:
                    result = child.wait(timeout=15)
                    break
                except subprocess.TimeoutExpired:
                    progress = Path(options['output']) / 'status.json'
                    updated = progress.stat().st_mtime if progress.exists() else started
                    # Ignore progress from a previous process until startup has
                    # had a chance to compile and restore the optimizer.
                    updated = max(started, updated)
                    completed = Path(options['output']) / 'complete.json'
                    if completed.is_file():
                        done = json.loads(completed.read_text())
                        if done.get('epochs') == options['epochs']:
                            continue  # Training finished; drain the online uploads.
                    if time.time() - updated > watchdog:
                        event = dict(phase='watchdog_restart', timestamp=time.time(),
                            attempt=attempt, torchrun_pid=child.pid,
                            seconds_without_progress=time.time()-updated)
                        (base / 'watchdog-restart.json').write_text(json.dumps(event, indent=2))
                        print(json.dumps(event), flush=True)
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            result = child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            result = child.wait()
                        break
        if result == 0:
            completed = Path(options['output']) / 'complete.json'
            if completed.is_file() and json.loads(completed.read_text()).get('epochs') == options['epochs']:
                (base / 'status.json').write_text(json.dumps(dict(phase='completed',
                    epochs=options['epochs'], timestamp=time.time()), indent=2))
                print('Training and online uploads completed', flush=True)
                return
            print('Process exited before the requested epochs; recovering full state', flush=True)
            result = 1
        print(f'Training exited {result}; restoring the last committed full state', flush=True)
        matching = newest_matching_checkpoint(options, restart_paths)
        if matching is not None:
            cfg.options.resume = True
            cfg.options.resume_checkpoint = str(matching)
            OmegaConf.save(cfg, recipe)
        elif cfg.options.get('resume_checkpoint'):
            # Before the continuation's first save, last.pt can still belong
            # to the earlier trajectory. Keep the selected source in that case.
            use_last = not cfg.options.get('lr_schedule_migration')
            if not use_last and latest.is_file():
                import torch
                committed = torch.load(latest, map_location='cpu', weights_only=True, mmap=True)
                use_last = (committed['config'].get('runtime_sha256') == options['runtime_sha256']
                            and committed['config'].get('fid_lr_policy') == options.get('fid_lr_policy')
                            and committed['scheduler'].get('kind') == 'fid-adaptive-cosine-v1')
                del committed
            if use_last:
                cfg.options.resume_checkpoint = None
                OmegaConf.save(cfg, recipe)
        time.sleep(10)
    (base / 'status.json').write_text(json.dumps(dict(phase='failed', exit_code=result,
        timestamp=time.time()), indent=2))
    raise SystemExit(result)


if __name__ == '__main__':
    main()
