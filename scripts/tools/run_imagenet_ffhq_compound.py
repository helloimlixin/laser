"""Supervise the verified ImageNet dataset -> unclipped cache -> compound prior."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    os.replace(temporary, path)


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dataset_progress(root):
    """Expose real HTTP progress instead of an indefinitely generic wait state."""
    result = {}
    if (root/'setup.pid').is_file():
        pid = int((root/'setup.pid').read_text())
        try:
            os.kill(pid, 0)
            state = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
            if state == 'Z':
                raise ProcessLookupError(pid)
        except (ProcessLookupError, FileNotFoundError):
            if (root/'READY.json').is_file():
                return result
            raise RuntimeError(f'ImageNet setup exited before READY.json; inspect {root}/logs/setup.log')
    try:
        request = urllib.request.Request('http://127.0.0.1:6802/jsonrpc',
            data=json.dumps({'jsonrpc': '2.0', 'id': 'progress', 'method': 'aria2.tellActive',
                'params': [['completedLength', 'totalLength', 'downloadSpeed', 'files']]}).encode(),
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=3) as response:
            items = json.load(response)['result']
        items = [x for x in items if any(Path(f['path']).is_relative_to(root) for f in x['files'])]
        for item in items:
            split = 'train' if 'img_train' in item['files'][0]['path'] else 'val'
            result[f'{split}_downloaded_bytes'] = int(item['completedLength'])
            result[f'{split}_total_bytes'] = int(item['totalLength'])
        result['download_bytes_per_second'] = sum(int(x['downloadSpeed']) for x in items)
    except (OSError, ValueError, KeyError):
        pass  # The HTTP process exits before checksum verification/extraction.
    markers = root/'.prepared_classes'
    if markers.is_dir():
        result['extracted_training_classes'] = sum(p.is_file() for p in markers.iterdir())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    args.config = args.config.resolve()
    from omegaconf import OmegaConf
    from src.training.cli import load_config
    cfg = load_config(args.config)
    options = OmegaConf.to_container(cfg.options, resolve=True)
    base = args.base.resolve()
    base.mkdir(parents=True, exist_ok=True)
    lock = open('/tmp/laser-imagenet-ffhq-compound.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stopped = {'signal': None}
    child = None
    def stop(number, _frame):
        stopped['signal'] = number
        if child is not None and child.poll() is None:
            child.send_signal(number)
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, stop)

    # Credentials come from the private .netrc, never from configs or snapshots.
    os.environ.setdefault('WANDB_DIR', '/tmp/laser-wandb')
    os.environ.setdefault('LASER_CHECKPOINT_STAGING_DIR', '/tmp/laser-checkpoint-staging')
    os.environ.setdefault('LASER_CHECKPOINT_UPLOAD_CACHE_DIR', '/mnt/laser-checkpoint-upload-cache')
    os.environ.setdefault('OMP_NUM_THREADS', '8')
    os.environ.setdefault('MKL_NUM_THREADS', '8')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
    os.environ['PYTHONUNBUFFERED'] = '1'
    Path(os.environ['WANDB_DIR']).mkdir(parents=True, exist_ok=True)
    import wandb
    wb = wandb.init(entity=options['wandb_entity'], project=options['wandb_project'],
        id=options['wandb_id'], name=options['wandb_name'], resume='allow', mode='online', allow_val_change=True,
        config={'pipeline': 'ImageNet FFHQ-v4 compound', 'stage1_rfid': 4.210914134979248,
            'recipe_reference': 'helloimlixin-rutgers/laser/ffhqcmp0804205803',
            'recipe_reference_fid': 8.174392700195312,
            'options': options, 'runtime_root': str(ROOT)})

    def status(phase, **values):
        row = dict(phase=phase, timestamp=time.time(), pid=os.getpid(),
            child_pid=None if child is None else child.pid, **values)
        write_json(base/'status.json', row)
        print(json.dumps(row), flush=True)
        if wb is not None:
            wb.summary['pipeline/phase'] = phase
            wb.log({'pipeline/heartbeat_unix': row['timestamp'], **{
                f'pipeline/{k}': v for k, v in values.items() if isinstance(v, (int, float))}})

    def command(phase, argv):
        nonlocal child
        with (base/f'{phase}.log').open('a') as output:
            child = subprocess.Popen(argv, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
            status(phase, command=argv)
            last = time.monotonic()
            while child.poll() is None:
                time.sleep(5)
                if time.monotonic() - last >= 60:
                    status(phase)
                    last = time.monotonic()
            code = child.returncode
            child = None
            if code:
                raise RuntimeError(f'{phase} exited {code}; see {base / (phase + ".log")}')
            if stopped['signal'] is not None:
                raise InterruptedError('Supervisor stopped by signal')

    try:
        manifest = json.loads((base/'runtime-manifest.json').read_text())
        for relative, expected in manifest.items():
            if sha256(ROOT/relative) != expected:
                raise RuntimeError(f'Runtime source changed: {relative}')
        while not (base/'preflight/complete.json').is_file():
            status('waiting_for_preflight')
            if stopped['signal'] is not None:
                raise InterruptedError('Supervisor stopped by signal')
            time.sleep(30)
        check = json.loads((base/'preflight/complete.json').read_text())
        if not check['passed']:
            raise RuntimeError('Full model preflight did not pass')
        expected_stage1 = 'dd28db9306d526bdc9fbf8016403e106c4f317fd8f5c04792f0953e53310fdab'
        if sha256(Path(options['checkpoint'])) != expected_stage1:
            raise RuntimeError('Stage-1 checkpoint provenance mismatch')
        wb.summary.update({'model/parameters': check['parameters'],
            'preflight/passed': True, 'coefficient_clipping': False,
            'checkpoints/latest_online_file': 'last.pt',
            'checkpoints/best_fid_online_file': 'best-fid-01.pt'})
        artifact = wandb.Artifact(options['wandb_id']+'-launch', type='run-config')
        for name, path in [('recipe.yaml', args.config),
                           ('stage1-provenance.json', base/'assets/stage1-provenance.json'),
                           ('preflight.json', base/'preflight/complete.json'),
                           ('capacity.json', base/'preflight/capacity.json'),
                           ('runtime-manifest.json', base/'runtime-manifest.json')]:
            artifact.add_file(str(path), name=name)
        wb.log_artifact(artifact, aliases=['latest']).wait()
        ready = Path(options['data'])/'READY.json'
        while not ready.is_file():
            progress = dataset_progress(ready.parent)
            phase = ('extracting_imagenet' if 'extracted_training_classes' in progress
                else 'downloading_imagenet' if 'train_downloaded_bytes' in progress
                else 'waiting_for_verified_imagenet')
            status(phase, dataset_ready=False, **progress)
            for _ in range(12):
                if stopped['signal'] is not None:
                    raise InterruptedError('Supervisor stopped by signal')
                if ready.is_file():
                    break
                time.sleep(5)
        counts = json.loads(ready.read_text())
        if counts != {'train': 1281167, 'val': 50000}:
            raise RuntimeError(f'Unexpected verified ImageNet counts: {counts}')
        while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                '--format=csv,noheader'], text=True).strip():
            status('waiting_for_idle_gpu')
            if stopped['signal'] is not None:
                raise InterruptedError('Supervisor stopped by signal')
            time.sleep(30)
        cache = Path(options['token_cache'])
        validation = cache.with_suffix('.validation.json')
        if not cache.is_file() or not validation.is_file():
            command('cache', [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                '--nproc-per-node=1', str(ROOT/'scripts/tools/build_official_imagenet_token_cache.py'),
                '--checkpoint', options['checkpoint'], '--data', options['data'],
                '--output', str(cache), '--dataset', 'imagenet', '--compound',
                '--sparsity-level', '4', '--num-atoms', '16384', '--coeff-vocab-size', '2048',
                '--coeff-max', '3', '--auto-coeff-scales-percentile', '100',
                '--no-clip-coefficients', '--coefficient-storage', 'fp32',
                '--encoder-precision', 'fp32', '--batch-size', '32', '--num-workers', '8'])
        result = json.loads(validation.read_text())
        if not result['passed'] or result['items'] != 1281167:
            raise RuntimeError('Full ImageNet cache verification failed')
        import torch
        payload = torch.load(cache, weights_only=True, map_location='cpu', mmap=True)
        meta = payload['meta']
        if (meta['stage1_checkpoint_sha256'] != expected_stage1 or meta['clip_coefficients']
                or meta['coefficient_storage'] != 'fp32' or meta['encoder_precision'] != 'fp32'
                or payload['labels'].unique().numel() != 1000):
            raise RuntimeError('Cache provenance, precision, clipping, or classes mismatch')
        wb.summary['cache/verification'] = result
        wb.summary['cache/coeff_scales'] = meta['coeff_scales']
        del payload
        status('starting_training')
        wb.summary['pipeline/phase'] = 'training'
        wb.finish()
        wb = None
        command('training', [sys.executable, str(ROOT/'train.py'), '--config', str(args.config)])
        status('completed')
    except BaseException as error:
        status('stopped' if isinstance(error, InterruptedError) else 'failed', error=str(error))
        (base/'failure.txt').write_text(traceback.format_exc())
        if wb is not None:
            wb.finish(exit_code=1)
        raise


if __name__ == '__main__':
    main()
