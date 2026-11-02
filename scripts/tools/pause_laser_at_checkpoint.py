"""Pause one owned LASER launcher only after a committed checkpoint."""
import argparse
import json
import os
from pathlib import Path
import signal
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--minimum-step', type=int, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    launcher = json.loads((args.output / 'launch-status.json').read_text())['pid']
    command = Path(f'/proc/{launcher}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    assert 'launch_laser_causal_ar_imagenet.py' in command
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(900):
        path = args.output / 'checkpoint.json'
        if path.exists():
            committed = json.loads(path.read_text())
            if committed['global_step'] >= args.minimum_step:
                checkpoint = Path(committed['path']).resolve()
                assert checkpoint.stat().st_size == committed['bytes'] > 10_000_000_000
                os.kill(launcher, signal.SIGINT)
                record = dict(passed=True, launcher_pid=launcher, checkpoint=str(checkpoint),
                              global_step=committed['global_step'], epoch=committed['epoch'],
                              checkpoint_bytes=committed['bytes'], reason='matched pair-decoder ablation',
                              time=time.time())
                args.receipt.write_text(json.dumps(record, indent=2) + '\n')
                print(json.dumps(record), flush=True)
                return
        if not Path(f'/proc/{launcher}').exists():
            raise RuntimeError('Launcher exited before requested checkpoint')
        time.sleep(4)
    raise TimeoutError('No requested checkpoint within one hour; training was left running')


if __name__ == '__main__':
    main()
