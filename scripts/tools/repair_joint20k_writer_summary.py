#!/usr/bin/env python3
"""Correct summary values in the existing W&B service without logging history."""
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path('/workspace/Projects/laser')
RUN = 'imagenet-laser-joint20k-depth5-uniformce-k512-scratch100-4b200-20261018'
TRAIN = ROOT / 'outputs' / RUN
OUT = TRAIN / 'diagnostics/logging-audit-20261018'


def main():
    import wandb
    rank0 = json.loads((TRAIN / 'train/progress-rank0.json').read_text())
    pid = rank0['pid']
    assert b'train.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
    # Use the already running writer's local service, never initialize a new run.
    environment = dict(row.split(b'=', 1) for row in
        Path(f'/proc/{pid}/environ').read_bytes().split(b'\0') if b'=' in row)
    service_key = wandb.env.SERVICE.encode()
    assert service_key in environment
    os.environ[wandb.env.SERVICE] = environment[service_key].decode()
    os.environ['WANDB_API_KEY'] = Path('/root/.config/laser/wandb_api_key').read_text().strip()
    run = wandb.attach(run_id=RUN)
    assert run is not None and run.id == RUN and run._is_attached
    before = run.summary._as_dict()
    evaluations = [json.loads(path.read_text()) for path in sorted((TRAIN / 'evaluations').glob('epoch*.json'))]
    latest = max(evaluations, key=lambda row:row['train/global_step'])
    values = {'eval/inception_score.last':latest['eval/inception_score'],
              'eval/inception_score.max':max(row['eval/inception_score'] for row in evaluations)}
    run.summary.update(values)
    after = run.summary._as_dict()  # Request/response flushes the preceding summary update.
    assert all(after[key] == value for key,value in values.items())
    assert isinstance(after['samples/fixed'], dict)
    result = dict(passed=True, time=time.time(), attached_to_existing_run=True,
        writer_pid=pid, no_history_log_calls=True, training_restarted=False,
        before={key:before.get(key) for key in values}, after=values)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'writer-summary-verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    exit_code = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        # The attached run belongs to training. Do not run SDK exit hooks or finish it.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
