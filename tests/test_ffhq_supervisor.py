import json
from pathlib import Path
import sys
import time

import pytest
from scripts.tools import run_ffhq_var341_pipeline as runner


def pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1,2')
    value = runner.FFHQPipeline.__new__(runner.FFHQPipeline)
    value.base = tmp_path
    value.runtime = tmp_path/'runtime'
    value.runtime.mkdir()
    value.stopping = False
    value.child = None
    return value


def test_phase_retries_failed_worker_then_requires_receipt(tmp_path, monkeypatch):
    value = pipeline(tmp_path, monkeypatch)
    receipt = tmp_path/'complete.json'
    counter = tmp_path/'attempts'
    code = f'''
from pathlib import Path
p=Path({str(counter)!r})
n=int(p.read_text())+1 if p.exists() else 1
p.write_text(str(n))
if n == 1: raise SystemExit(1)
Path({str(receipt)!r}).write_text('{{}}')
'''
    real_sleep = time.sleep
    monkeypatch.setattr(runner.time, 'sleep', lambda _: real_sleep(.01))
    value.phase('test', [sys.executable, '-c', code], receipt)
    assert counter.read_text() == '2'
    assert receipt.exists()


def test_stale_training_progress_triggers_watchdog(tmp_path, monkeypatch):
    value = pipeline(tmp_path, monkeypatch)
    (tmp_path/'tokenizer').mkdir()
    (tmp_path/'tokenizer/status.json').write_text(json.dumps(dict(time=0, phase='tokenizer')))
    ticks = iter([1000, 1000, 1701, 1701, 1701])
    monkeypatch.setattr(runner.time, 'time', lambda: next(ticks))
    class Child:
        pid = 12345
        returncode = None
        def poll(self):
            return self.returncode
    child = Child()
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *a, **k: child)
    def terminate(process):
        assert process is child
        process.returncode = -9
        value.stopping = True
    monkeypatch.setattr(runner, 'terminate_tree', terminate)
    with pytest.raises(InterruptedError):
        value.phase('tokenizer_training', ['unused'], tmp_path/'complete.json')
    assert json.loads((tmp_path/'pipeline-status.json').read_text())['phase'] == 'watchdog_restart'


def test_workspace_mirror_failure_does_not_stop_live_metadata(tmp_path):
    (tmp_path/'status.json').write_text('{"step": 100}')
    mirror = runner.MetadataMirror(tmp_path, '/proc/laser-mirror-unwritable-test')
    try:
        report = tmp_path/'workspace-mirror-status.json'
        deadline = time.monotonic()+3
        while not report.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert json.loads(report.read_text())['errors']
        assert mirror.thread.is_alive()
        assert json.loads((tmp_path/'status.json').read_text())['step'] == 100
    finally:
        mirror.stop.set()
        mirror.thread.join(timeout=3)
