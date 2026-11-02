#!/usr/bin/env python3
"""Run CC3M GPU preflight, then train to the configured epoch target."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(data, time=time.time()), indent=2)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    config = args.config.resolve()
    cfg = yaml.safe_load(config.read_text())
    base = Path(cfg['local_checkpoints']).parent
    lock = (base/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sys.path.insert(0, str(base/'source'))
    from src.training.cc3m_storage import WorkspaceMirror
    from src.training.cc3m_paper_metrics import load_reference
    load_reference(cfg)
    logs, status = base/'logs', base/'supervisor'
    logs.mkdir(exist_ok=True)
    status.mkdir(exist_ok=True)
    mirror = WorkspaceMirror([(status, cfg['output']), (logs, cfg['output'])]).start()
    secret = os.environ.get('WANDB_API_KEY') or Path('/root/.config/laser/wandb.key').read_text().strip()
    env = dict(os.environ, WANDB_API_KEY=secret, WANDB_DISABLE_GIT='true',
        WANDB_CACHE_DIR=str(base/'wandb-cache'), WANDB_DATA_DIR=str(base/'wandb-data'),
        WANDB_ARTIFACT_DIR=str(base/'wandb-artifacts'), TORCH_HOME=str(base/'torch-cache'),
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false',
        PYTHONWARNINGS='ignore::FutureWarning',
        RQ_UPSTREAM_ROOT=str(base/'upstream'),
        PYTHONPATH=str(base/'source')+':'+str(base/'upstream'),
        CUDA_VISIBLE_DEVICES=','.join(map(str, range(cfg['world_size']))),
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    child, stopping, stop_at = None, False, None

    def stop(*_):
        nonlocal stopping, stop_at
        stopping, stop_at = True, time.monotonic()
        workers = set()
        for directory in ('train', 'preflight'):
            for path in (base/'live'/directory).glob('*rank*.json'):
                try:
                    pid = json.loads(path.read_text()).get('pid')
                    if pid and b'src.training.cc3m_joint' in Path(f'/proc/{pid}/cmdline').read_bytes():
                        workers.add(pid)
                except (OSError, ValueError):
                    pass
        for pid in workers:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if not workers and child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
        f'--nproc-per-node={cfg["world_size"]}', '-m', 'src.training.cc3m_joint',
        '--config', str(config)]

    def execute(phase, extra=()):
        nonlocal child
        with (logs/(phase+'.log')).open('ab', buffering=0) as log:
            child = subprocess.Popen(command+list(extra), cwd=base, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            write(status/'status.json', dict(phase=phase, pid=os.getpid(),
                child_pid=child.pid, run_id=cfg['run_id'], epochs=cfg['epochs']))
            while True:
                try:
                    return child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if stopping and time.monotonic()-stop_at > 300:
                        os.killpg(child.pid, signal.SIGKILL)

    try:
        passed = base/'live/preflight/passed.json'
        if not passed.exists():
            code = execute('gpu_preflight', ['--smoke'])
            if code or stopping:
                write(status/'status.json', dict(phase='stopped' if stopping else 'failed',
                    failed_phase='gpu_preflight', exit_code=code))
                return 0 if stopping else 1
        assert json.loads(passed.read_text())['passed']
        updates = cfg['train_items']//cfg['global_batch']
        write(status/'launch.json', dict(run_id=cfg['run_id'], epochs=cfg['epochs'],
            total_updates=updates*cfg['epochs'], preflight_passed=True,
            production_starts_from_zero=not (Path(cfg['local_checkpoints'])/'last.pt').exists(),
            evaluation=cfg['evaluation'], sampling=cfg['sampling']))
        for attempt in range(1, 11):
            if stopping:
                break
            code = execute('training')
            completed = base/'live/completed.json'
            if completed.exists():
                result = json.loads(completed.read_text())
                assert result == dict(epochs=cfg['epochs'], global_step=updates*cfg['epochs'])
                write(status/'status.json', dict(phase='completed', attempt=attempt, **result))
                return 0
            if stopping:
                break
            write(status/'status.json', dict(phase='retry_wait' if attempt < 10 else 'failed',
                attempt=attempt, exit_code=code, run_id=cfg['run_id']))
            if attempt == 10:
                return 1
            for _ in range(30):
                if stopping:
                    break
                time.sleep(1)
        write(status/'status.json', dict(phase='stopped', run_id=cfg['run_id']))
        return 0
    finally:
        mirror.close()


if __name__ == '__main__':
    sys.exit(main())
