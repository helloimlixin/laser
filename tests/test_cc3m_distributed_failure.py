import json
import os
from pathlib import Path
import subprocess
import sys


def test_rank_failure_logs_and_exits_without_waiting_for_peer_collective(tmp_path):
    script = tmp_path / 'fail_rank.py'
    script.write_text('''import os, sys
import torch.distributed as dist
from src.training.cc3m_text import abort_training
dist.init_process_group('gloo')
rank=int(os.environ['RANK'])
if rank == 0:
    try:
        raise RuntimeError('injected_training_failure')
    except RuntimeError as error:
        abort_training(error, sys.argv[1], rank, 12)
else:
    dist.barrier()
''')
    result = subprocess.run([sys.executable, '-m', 'torch.distributed.run',
        '--standalone', '--nproc-per-node=2', str(script), str(tmp_path)],
        capture_output=True, text=True, timeout=45, env=dict(os.environ, OMP_NUM_THREADS='1'))
    assert result.returncode != 0
    assert 'injected_training_failure' in result.stderr
    report = json.loads((tmp_path/'failure-rank-000.json').read_text())
    assert report['phase'] == 'failed' and report['global_step'] == 12
    assert report['error_type'] == 'RuntimeError'
