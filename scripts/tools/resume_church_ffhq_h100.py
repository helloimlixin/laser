#!/usr/bin/env python3
"""Restore the frozen Church run and launch its detached four-H100 supervisor."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/church-ffhq-compound-350m-4a100-20260923'
LOCAL = Path('/mnt/laser-church')
RUNTIME = LOCAL / 'runtime'


def main(base: Path = BASE):
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL / 'resume-launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pid_file = base / 'supervisor.pid'
        if pid_file.is_file():
            process = Path('/proc') / pid_file.read_text().strip()
            if process.exists():
                command = (process / 'cmdline').read_bytes()
                if b'run_church_ffhq_compound.py' in command:
                    raise SystemExit('This run already has a live supervisor.')

        selection_path = base / 'resume-runtime.json'
        selection = json.loads(selection_path.read_text()) if selection_path.is_file() else {}
        runtime = Path(selection.get('runtime_root', str(RUNTIME)))
        manifest_path = base / selection.get('manifest', 'runtime-manifest.json')
        manifest = json.loads(manifest_path.read_text())
        if not (runtime / 'train.py').is_file():
            runtime.mkdir(parents=True, exist_ok=True)
            with tarfile.open(base / selection.get('source_archive', 'source.tar.gz')) as archive:
                archive.extractall(runtime, filter='data')
        for relative, expected in manifest.items():
            if hashlib.sha256((runtime / relative).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Frozen runtime checksum mismatch: {relative}')
        assets = LOCAL / 'assets'
        assets.mkdir(exist_ok=True)
        for name in ('tokenizer.pt', 'compound-cache.pt', 'compound-cache.validation.json',
                     'stage1-provenance.json', 'selection.json', 'lsun_256_church.npz'):
            source, target = base / 'assets' / name, assets / name
            if not target.is_file() or target.stat().st_size != source.stat().st_size:
                temporary = target.with_suffix(target.suffix + '.restore')
                shutil.copyfile(source, temporary)
                os.replace(temporary, target)

        sys.path.insert(0, str(runtime))
        os.environ['LASER_CHECKPOINT_UPLOAD_CACHE_DIR'] = str(LOCAL / 'upload-cache')
        from src.training.rqtransformer import (
            _checkpoint_file_identity, _local_checkpoint_paths,
        )
        import torch
        directory = base / 'train/checkpoints'
        latest = directory / 'last.pt'
        if not latest.is_file() or latest.stat().st_size == 0:
            raise RuntimeError('A complete last.pt is required; refusing to start fresh.')
        # Read each multi-GB recovery file from shared storage only once per host.
        sources = [latest, *directory.glob('best_fid_*.pt')]
        restored_latest = None
        for source in sources:
            source = source.resolve()
            local, receipt = _local_checkpoint_paths(source)
            # This mount rounds ctime after committing a file. Its immutable
            # payload name, inode, size and mtime still identify the same data.
            # Accept that metadata-only change after probing the stored bytes.
            identity = _checkpoint_file_identity(source)
            cached = False
            if local.is_file() and receipt.is_file():
                recorded = json.loads(receipt.read_text())['identity']
                stable = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns')
                cached = (local.stat().st_size == identity['st_size'] and
                          all(recorded.get(k) == identity[k] for k in stable))
                if cached:
                    with source.open('rb') as remote, local.open('rb') as staging:
                        cached = remote.read(64) == staging.read(64)
                        remote.seek(-64, 2)
                        staging.seek(-64, 2)
                        cached = cached and remote.read(64) == staging.read(64)
            if not cached:
                local.parent.mkdir(parents=True, exist_ok=True)
                temporary = local.with_suffix('.restore')
                shutil.copyfile(source, temporary)
                if temporary.stat().st_size != source.stat().st_size:
                    raise RuntimeError(f'Incomplete checkpoint restore: {source.name}')
                os.replace(temporary, local)
            receipt.write_text(json.dumps(dict(source=str(source),
                identity=_checkpoint_file_identity(source))) + '\n')
            if source == latest.resolve():
                restored_latest = local
        saved = torch.load(restored_latest, map_location='cpu',
                           weights_only=False, mmap=True)
        if not saved['optimizer']['state'] or len(saved['rng_state_by_rank']) != 4:
            raise RuntimeError('Recovery must include optimizer state and four RNG streams.')
        step = int(saved['global_step'])
        del saved

        config = LOCAL / 'resume-h100.yaml'
        shutil.copyfile(base / selection.get('config', 'resume-h100.yaml'), config)
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(config)
        # Select this host's copy of the newest committed checkpoint, rather
        # than making every rank traverse the network filesystem at startup.
        cfg.options.resume_checkpoint = str(restored_latest)
        cfg.options.resume = True
        OmegaConf.save(cfg, config)
        env = dict(os.environ, PYTHONUNBUFFERED='1', NCCL_NVLS_ENABLE='0')
        env.setdefault('TORCHINDUCTOR_CACHE_DIR', str(LOCAL / 'torchinductor'))
        credential = Path('/root/.config/laser/wandb-api-key')
        if not env.get('WANDB_API_KEY') and credential.is_file():
            env['WANDB_API_KEY'] = credential.read_text().strip()
        command = [sys.executable, str(runtime / 'scripts/tools/run_church_ffhq_compound.py'),
                   '--config', str(config), '--base', str(base),
                   '--runtime-manifest', str(manifest_path)]
        if selection.get('preflight_report'):
            command += ['--preflight-report', str(base / selection['preflight_report'])]
        with (base / 'resume-h100-supervisor.log').open('a') as log:
            process = subprocess.Popen(command, cwd=runtime, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        pid_file.write_text(str(process.pid) + '\n')
        receipt = dict(pid=process.pid, started_unix=time.time(), command=command,
            resume_step=step, runtime_root=str(runtime), runtime_manifest=str(manifest_path),
            gpus='4 x NVIDIA H100 80GB',
            global_batch=int(cfg.options.total_batch_size),
            per_gpu_batch=int(cfg.options.batch_size),
            accumulation_steps=int(cfg.options.total_batch_size) // (4 * int(cfg.options.batch_size)),
            save_step_freq=int(cfg.options.save_step_freq),
            save_ckpt_freq=int(cfg.options.save_ckpt_freq))
        (base / 'resume-h100-launch.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=BASE)
    main(parser.parse_args().base.resolve())
