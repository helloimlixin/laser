#!/usr/bin/env python3
"""Run the fixed benchmark sequentially on a GPU; preserve progress on restart."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid', type=int)
    parser.add_argument('--root', type=Path, default=Path('outputs/mdctcodec_tts_benchmark'))
    parser.add_argument('--baseline-python', default='/workspace/tts-benchmark-env/bin/python')
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    root = args.root
    env = {**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
           'HF_HOME': '/workspace/tts-benchmark-models', 'HF_HUB_DISABLE_XET': '1'}
    if args.wait_pid:
        while True:
            proc = Path(f'/proc/{args.wait_pid}/stat')
            if not proc.exists() or proc.read_text().split()[2] == 'Z': break
            time.sleep(5)
    script = 'scripts/tools/benchmark_mdctcodec_tts.py'
    jobs = [(sys.executable, 'generate', 'reference'), (sys.executable, 'generate', 'codec'),
            (args.baseline_python, 'generate', 'f5'), (args.baseline_python, 'generate', 'chatterbox'),
            (args.baseline_python, 'score', 'reference,codec,laser_epoch40_selected,f5,chatterbox'),
            (args.baseline_python, 'report', 'reference,codec,laser_epoch40_selected,f5,chatterbox')]
    assert json.loads((root / 'generated/laser_epoch40_selected/complete.json').read_text())['items'] == 100
    for python, phase, arm in jobs:
        command = [python, '-u', script, '--phase', phase, '--arm', arm, '--root', str(root), '--device', args.device]
        log = root / f'{phase}_{arm.replace(",", "_")}.log'
        state = {'status': 'running', 'phase': phase, 'arm': arm, 'pid': os.getpid(),
                 'command': command, 'log': str(log), 'updated_utc': datetime.now(timezone.utc).isoformat()}
        (root / 'pipeline_status.json').write_text(json.dumps(state, indent=2))
        print('START', phase, arm, flush=True)
        with log.open('a') as f: result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, env=env)
        if result.returncode:
            state.update(status='failed', returncode=result.returncode)
            (root / 'pipeline_status.json').write_text(json.dumps(state, indent=2))
            raise RuntimeError(f'{phase} {arm} failed: inspect {log}')
    state.update(status='complete')
    (root / 'pipeline_status.json').write_text(json.dumps(state, indent=2))
    print('BENCHMARK_COMPLETE', flush=True)


if __name__ == '__main__': main()
