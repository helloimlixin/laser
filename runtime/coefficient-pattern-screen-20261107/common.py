"""Shared, frozen inputs for the coefficient-pattern reconstruction screen."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path('/tmp/laser-coefficient-pattern-screen-20261107')
BASE = Path('/tmp/laser-compound128-20261107/frozen')
sys.path[:0] = [str(ROOT/'frozen-code'), str(BASE/'source'), str(BASE)]
os.environ['RQ_UPSTREAM_ROOT'] = str(BASE/'upstream')
os.environ['TORCH_HOME'] = '/tmp/laser-compound128-20261107/torch-cache'


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.pending')
    temp.write_text(json.dumps(data, indent=2, default=str)+'\n')
    temp.replace(path)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def distributed():
    import torch
    import torch.distributed as dist
    torch.set_num_threads(4)
    rank, world, local = [int(os.environ[k]) for k in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK']]
    device = torch.device('cuda', local)
    torch.cuda.set_device(device)
    # Bound this temporary screen while the 56-GiB baseline keeps training.
    torch.cuda.set_per_process_memory_fraction(.15, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('highest')
    from datetime import timedelta
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(minutes=30))
    return rank, world, device
