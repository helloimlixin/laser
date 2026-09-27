"""Operational support for the isolated ImageNet compound scratch launch."""
import json
import os
from pathlib import Path
import signal
import time

import torch
import torch.distributed as dist

_STOP = False


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def install_stop_handler():
    def stop(signum, frame):
        global _STOP
        _STOP = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)


def should_stop(device):
    requested = _STOP or time.time() >= float(os.environ.get('LASER_STOP_TIME_UNIX', 'inf'))
    value = torch.tensor(int(requested), device=device)
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return bool(value.item())


def record_progress(payload):
    phase = os.environ.get('LASER_LAUNCH_PHASE', 'train')
    record = dict(payload, updated_unix=time.time(), job_id=os.environ.get('SLURM_JOB_ID'))
    atomic_json(Path(os.environ['RUN_DIR']) / phase / 'status.json', record)
    print(json.dumps(dict(phase=phase, **record)), flush=True)


def adapt_resume_payload(payload, args):
    # This launch deliberately has no stage-two initialization or resume input.
    raise RuntimeError('Scratch launch must not load any stage-two checkpoint')
