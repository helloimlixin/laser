import json
import os
from pathlib import Path
import subprocess
import sys


def test_worker_failure_exits_even_when_output_is_unwritable(tmp_path):
    script = '''
from src.training.distributed_failure import fatal_worker_error
try:
    raise OSError('injected metadata write failure')
except OSError as error:
    fatal_worker_error(error, '/proc/laser-unwritable-test', 0)
'''
    env = dict(os.environ, TMPDIR=str(tmp_path))
    result = subprocess.run([sys.executable, '-c', script], env=env,
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 1
    assert 'injected metadata write failure' in result.stderr
    reports = list((tmp_path/'laser-worker-failures').glob('*.json'))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text())['type'] == 'OSError'


def test_failure_handler_does_not_enter_finally(tmp_path):
    marker = tmp_path/'collective-teardown-would-block'
    script = f'''
from pathlib import Path
from src.training.distributed_failure import fatal_worker_error
try:
    try:
        raise RuntimeError('rank-local failure')
    except RuntimeError as error:
        fatal_worker_error(error, {str(tmp_path)!r}, 1)
finally:
    Path({str(marker)!r}).touch()
'''
    result = subprocess.run([sys.executable, '-c', script], text=True,
                            capture_output=True, timeout=10)
    assert result.returncode == 1
    assert not marker.exists()
    assert json.loads((tmp_path/'failure-rank1.json').read_text())['rank'] == 1
