#!/usr/bin/env python3
"""Supervise the two fresh hard-6kbps codec arms within a shared compute ceiling."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / 'outputs/mdctcodec_k4_a4096_hard6k_20260913'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    os.chdir(REPO)
    root = args.root.resolve()
    lock = (root / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'status.json').exists() or any((root / arm / 'run.json').exists() for arm in ('laser', 'rvq')):
        raise RuntimeError('Pair already launched; inspect status and resume each checkpoint explicitly')
    protocol = json.loads((root / 'protocol.json').read_text())
    for name in ('preflight_verified.json', 'restore_verified.json'):
        if not json.loads((root / name).read_text())['passed']:
            raise RuntimeError(f'Preflight did not pass: {name}')
    ceiling = protocol['compute_ceiling_gpu_hours_per_arm'] * 3600
    target = protocol['generator_updates']
    jobs, children, logs = [], {}, {}
    stopping = None

    def stop(signum, _frame):
        nonlocal stopping
        stopping = f'signal_{signum}'

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for gpu, arm in enumerate(('laser', 'rvq')):
            if stopping:
                break
            command = [sys.executable, '-u', 'scripts/tools/train_mdctcodec_hard6k.py',
                       '--root', str(root), '--arm', arm]
            logs[arm] = (root / f'{arm}.log').open('a')
            child = subprocess.Popen(command, stdout=logs[arm], stderr=subprocess.STDOUT,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONPATH': str(REPO),
                     'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}, start_new_session=True)
            children[arm] = child
            jobs.append({'arm': arm, 'gpu': gpu, 'pid': child.pid, 'command': command,
                         'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
                         'started_monotonic': time.monotonic()})
        while True:
            now = time.monotonic()
            for job in jobs:
                if job['status'] != 'running':
                    continue
                arm = job['arm']; child = children[arm]
                job['assigned_gpu_hours'] = (now - job['started_monotonic']) / 3600
                code = child.poll()
                if code is None:
                    if job['assigned_gpu_hours'] * 3600 >= ceiling:
                        stopping = stopping or f'{arm}_compute_ceiling'
                    continue
                completion_path = root / arm / 'completion.json'
                completion = json.loads(completion_path.read_text()) if completion_path.exists() else {}
                complete = code == 0 and completion.get('status') == 'complete' and completion.get('generator_updates') == target
                job.update(status='complete' if complete else 'stopped_incomplete', returncode=code,
                           completion=completion, finished_utc=datetime.now(timezone.utc).isoformat())
                logs[arm].close()
                if not complete:
                    stopping = stopping or f'{arm}_incomplete_or_failed'
            if (root / 'STOP').exists():
                stopping = stopping or 'STOP_file'
            if stopping:
                for job in jobs:
                    if job['status'] == 'running' and not job.get('stop_sent'):
                        child = children[job['arm']]
                        if child.poll() is None:
                            child.send_signal(signal.SIGTERM)
                        job['stop_sent'] = True
            done = all(job['status'] != 'running' for job in jobs)
            state = {'status': ('stopped_incomplete' if stopping else 'complete') if done else 'running',
                     'pid': os.getpid(), 'stop_reason': stopping,
                     'updated_utc': datetime.now(timezone.utc).isoformat(),
                     'budget_gpu_hours': ceiling * 2 / 3600,
                     'assigned_gpu_hours': sum(job.get('assigned_gpu_hours', 0) for job in jobs),
                     'generator_update_target_per_arm': target, 'hard_rate_cap_bps': 6000,
                     'scope': 'Assigned GPU wall time including validation/uploads; graceful checkpoint shutdown can add time.',
                     'jobs': [{k: v for k, v in job.items() if k != 'started_monotonic'} for job in jobs]}
            temp = root / 'status.tmp'
            temp.write_text(json.dumps(state, indent=2)); temp.replace(root / 'status.json')
            if done:
                break
            time.sleep(10)
    finally:
        for child in children.values():
            if child.poll() is None:
                child.send_signal(signal.SIGTERM)


if __name__ == '__main__':
    main()
