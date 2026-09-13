#!/usr/bin/env python3
"""Run the larger codec/prior budgets and dependent benchmarks on two GPUs."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu-hours', type=float, default=24)
    p.add_argument('--resume-campaign', action='store_true')
    args = p.parse_args(); os.chdir(REPO)
    root = Path('outputs/mdctcodec_long_campaign'); root.mkdir(exist_ok=True)
    import fcntl
    lock = (root/'supervisor.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root/'status.json').exists() and not args.resume_campaign:
        raise RuntimeError('Campaign already launched; inspect its status before creating another')
    python = sys.executable
    jobs = [
        {'name': 'stage1_laser', 'deps': [], 'command': [python, '-u', 'scripts/tools/continue_mdctcodec_matched.py', '--arm', 'laser']},
        {'name': 'stage2', 'deps': [], 'command': [python, '-u', 'scripts/tools/train_mdctcodec_tts.py',
            '--config', 'configs/research/mdctcodec_tts_long.yaml', '--new-run', '--resume',
            'outputs/mdctcodec_tts_rangefix/stage2/checkpoints/last.pt']},
        {'name': 'stage1_rvq', 'deps': [], 'command': [python, '-u', 'scripts/tools/continue_mdctcodec_matched.py', '--arm', 'rvq']},
        {'name': 'tts_benchmark', 'deps': ['stage2'], 'command': [python, '-u', 'scripts/tools/evaluate_mdctcodec_long.py', '--stage', 'tts']},
        {'name': 'codec_benchmark', 'deps': ['stage1_laser', 'stage1_rvq'], 'command': [python, '-u',
            'scripts/tools/evaluate_mdctcodec_long.py', '--stage', 'codec']},
    ]
    for job in jobs: job['status'] = 'queued'
    processes, logs, completed = {}, {}, set()
    used = 0.; last = time.monotonic(); stopping = False
    if args.resume_campaign:
        previous = json.loads((root/'status.json').read_text())
        assert previous['status'] != 'running', 'Wait for the preceding supervisor to stop'
        assert args.gpu_hours == previous['budget_gpu_hours'], 'Recovery must preserve the campaign ceiling'
        recovery = json.loads((root/'recovery_plan.json').read_text())
        used = previous['active_gpu_hours'] * 3600
        for job in jobs:
            old = next(j for j in previous['jobs'] if j['name'] == job['name'])
            if job['name'] in recovery['jobs']:
                item = recovery['jobs'][job['name']]
                assert Path(item['checkpoint']).is_file()
                job['command'] += ['--resume', item['checkpoint']]
                job['previous_attempt'] = old
            elif old['status'] == 'complete' or (job['name'] == 'tts_benchmark' and
                    json.loads((root/'tts_evaluation_status.json').read_text())['status'] == 'complete'):
                job.update(old); job['status'] = 'complete'; completed.add(job['name'])
        (root/'status_before_recovery.json').write_text(json.dumps(previous, indent=2))
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    try:
        while True:
            now = time.monotonic(); used += (now-last)*len(processes); last=now
            for gpu, (process, job) in list(processes.items()):
                code = process.poll()
                if code is None: continue
                job.update(status='complete' if code == 0 else 'failed', returncode=code,
                           finished_utc=datetime.now(timezone.utc).isoformat())
                if code == 0: completed.add(job['name'])
                elif job['name'].startswith('stage'): stopping=True
                logs.pop(gpu).close(); del processes[gpu]
            if used >= args.gpu_hours*3600: stopping=True
            if stopping:
                for process, job in processes.values():
                    if job.get('stop_sent'): continue
                    if job['name'].startswith('stage'): process.send_signal(signal.SIGTERM)
                    else: os.killpg(process.pid, signal.SIGTERM)
                    job['stop_sent']=True
            else:
                for gpu in [0, 1]:
                    if gpu in processes: continue
                    pending = next((j for j in jobs if j['status']=='queued' and set(j['deps']) <= completed), None)
                    if pending is None: continue
                    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONPATH': str(REPO),
                        'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
                        'HF_HOME': '/workspace/tts-benchmark-models', 'HF_HUB_DISABLE_XET': '1'}
                    log = (root/f'{pending["name"]}.log').open('a')
                    child = subprocess.Popen(pending['command'], stdout=log, stderr=subprocess.STDOUT,
                        env=env, start_new_session=True)
                    logs[gpu]=log; processes[gpu]=(child,pending)
                    pending.update(status='running', pid=child.pid, gpu=gpu,
                                   started_utc=datetime.now(timezone.utc).isoformat())
            done = not processes and (stopping or all(j['status'] in ('complete','failed') for j in jobs))
            status = {'status': 'stopped_budget_or_failure' if done and stopping else 'complete' if done else 'running',
                'pid': os.getpid(), 'active_gpu_hours': used/3600, 'budget_gpu_hours': args.gpu_hours,
                'updated_utc': datetime.now(timezone.utc).isoformat(), 'jobs': jobs,
                'budget_scope': 'Sum of assigned-GPU job wall time, including validation and benchmark; graceful checkpoint shutdown may add several minutes'}
            temp=root/'status.tmp'; temp.write_text(json.dumps(status,indent=2)); temp.replace(root/'status.json')
            if done: break
            time.sleep(15)
    finally:
        for process, job in processes.values():
            if process.poll() is None: process.send_signal(signal.SIGTERM)


if __name__ == '__main__':
    main()
