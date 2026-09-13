#!/usr/bin/env python3
"""Pause the assigned RVQ GPU for the queued speech benchmark, then resume it."""
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
    os.chdir(REPO)
    root = Path('outputs/mdctcodec_long_campaign')
    import fcntl
    lock = (root/'priority_evaluation.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if Path('outputs/mdctcodec_tts_long/benchmark/complete.json').exists():
        print('TTS benchmark already complete; leaving training uninterrupted.', flush=True)
        return
    state = json.loads((root/'status.json').read_text())
    job = next(j for j in state['jobs'] if j['name'] == 'stage1_rvq')
    assert job['status'] == 'running'
    pid = job['pid']
    assert b'continue_mdctcodec_matched.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
    assert os.getpgid(pid) == pid
    stop = False
    child = None
    def interrupted(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    record = {'status': 'pausing_rvq', 'wrapper_pid': os.getpid(), 'rvq_pid': pid, 'gpu': job['gpu'],
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'reason': 'User requested the completed stage-2 comparison now',
        'timing_scope': 'One active compute workload on the benchmark GPU; paused RVQ retains resident GPU memory',
        'budget': 'Uses the already assigned RVQ GPU; its wall time remains counted by the campaign supervisor'}
    path = root/'priority_tts_evaluation.json'
    path.write_text(json.dumps(record, indent=2))
    started = time.monotonic()
    try:
        os.killpg(pid, signal.SIGSTOP)
        time.sleep(3)  # Let already submitted CUDA work drain before timing synthesis.
        assert Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1][0] == 'T'
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(job['gpu']), 'PYTHONPATH': str(REPO),
            'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
            'HF_HOME': '/workspace/tts-benchmark-models', 'HF_HUB_DISABLE_XET': '1'}
        with (root/'tts_benchmark_priority.log').open('a') as log:
            child = subprocess.Popen([sys.executable, '-u', 'scripts/tools/evaluate_mdctcodec_long.py',
                '--stage', 'tts'], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record.update(status='evaluating_rvq_paused', evaluation_pid=child.pid)
            path.write_text(json.dumps(record, indent=2))
            while child.poll() is None:
                if stop or time.monotonic()-started > 3600:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait(timeout=30)
                    raise RuntimeError('Priority evaluation interrupted or exceeded one hour')
                time.sleep(5)
            if child.returncode:
                raise RuntimeError(f'Priority evaluation exited with {child.returncode}')
            record['status'] = 'complete'
    except Exception as error:
        record.update(status='failed', error=str(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        if Path(f'/proc/{pid}').exists():
            os.killpg(pid, signal.SIGCONT)
            record['rvq_resumed'] = True
        record['finished_utc'] = datetime.now(timezone.utc).isoformat()
        record['elapsed_seconds'] = time.monotonic()-started
        path.write_text(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
