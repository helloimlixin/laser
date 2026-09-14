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
    parser.add_argument('--evaluate', action='store_true', help='Evaluate validation-selected models after both budgets complete')
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
    audited_epochs = set()
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
            command = [sys.executable, '-u', protocol.get('training_entrypoint','scripts/tools/train_mdctcodec_hard6k.py'),
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
            streams = {}
            for arm in ('laser', 'rvq'):
                path = root / arm / 'data_order.jsonl'
                if path.exists():
                    streams[arm] = {row['epoch']: row for row in
                        (json.loads(line) for line in path.read_text().splitlines(keepends=True) if line.endswith('\n'))}
            if len(streams) == 2:
                for epoch in (streams['laser'].keys() & streams['rvq'].keys()) - audited_epochs:
                    if streams['laser'][epoch] != streams['rvq'][epoch]:
                        stopping = stopping or f'paired_data_audit_mismatch_epoch_{epoch}'
                    else:
                        audited_epochs.add(epoch)
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
                     'paired_epochs_audited': len(audited_epochs),
                     'scope': 'Assigned GPU wall time including validation/uploads; graceful checkpoint shutdown can add time.',
                     'jobs': [{k: v for k, v in job.items() if k != 'started_monotonic'} for job in jobs]}
            temp = root / 'status.tmp'
            temp.write_text(json.dumps(state, indent=2)); temp.replace(root / 'status.json')
            if done:
                break
            time.sleep(10)
        if args.evaluate and not stopping and len(jobs)==2 and all(job['status']=='complete' for job in jobs):
            remaining = max(0.,ceiling*2-sum(job['assigned_gpu_hours']*3600 for job in jobs))
            comparison = {'status':'skipped_compute_ceiling' if remaining<60 else 'running',
                          'remaining_gpu_seconds':remaining}
            if remaining>=60:
                command=[sys.executable,'-u','archive/scripts/report_mdctcodec_matched.py',
                         '--root',str(root),'--device','cuda:0','--workers','8']
                with (root/'comparison.log').open('a') as log:
                    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                        env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})
                    children['comparison']=child
                    began=time.monotonic()
                    comparison.update(pid=child.pid,command=command)
                    while child.poll() is None:
                        elapsed=time.monotonic()-began
                        if stopping or (root/'STOP').exists() or elapsed>=remaining:
                            os.killpg(child.pid,signal.SIGTERM)
                            comparison['stop_reason']=stopping or 'stop_file_or_compute_ceiling'
                        comparison['assigned_gpu_hours']=elapsed/3600
                        state.update(comparison=dict(comparison),assigned_gpu_hours=
                            sum(job['assigned_gpu_hours'] for job in jobs)+elapsed/3600)
                        temp=root/'status.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(root/'status.json')
                        time.sleep(10)
                    comparison.update(status='complete' if child.returncode==0 and (root/'comparison_complete.json').exists()
                                      else 'stopped_or_failed',returncode=child.returncode)
                    if comparison['status']=='complete':
                        comparison.update(json.loads((root/'comparison_complete.json').read_text()))
            state.update(comparison=comparison,updated_utc=datetime.now(timezone.utc).isoformat())
            temp=root/'status.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(root/'status.json')
    finally:
        for child in children.values():
            if child.poll() is None:
                child.send_signal(signal.SIGTERM)


if __name__ == '__main__':
    main()
