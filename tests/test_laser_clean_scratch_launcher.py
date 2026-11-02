"""The six-epoch experiment must neither resume nor silently run twenty epochs."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def load_launcher():
    path = Path(__file__).resolve().parents[1]/'scripts/tools/launch_laser_hard_ar_imagenet.py'
    spec = importlib.util.spec_from_file_location('bounded_laser_launcher', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fresh_six_epoch_command_discards_smoke_state_and_honors_stop(monkeypatch, tmp_path):
    module = load_launcher()
    module.BASE, module.OUT = tmp_path/'runtime', tmp_path/'output'
    module.BASE.mkdir(); module.OUT.mkdir()
    prepared = dict(prepared=True, recipe=dict(epochs=6, microbatch=32,
        tokenizer=dict(sha256='verified-stage1')))
    for name, data in [('prepared.json', prepared), ('cpu-verification.json', dict(passed=True)),
                       ('smoke-reload-verification.json', dict(passed=True)),
                       ('reproducibility/code-manifest.json', {})]:
        path = module.OUT/name; path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(data))
    (module.BASE/'environment.json').write_text('{}')
    monkeypatch.setattr(module, 'digest', lambda path:'verified-stage1')
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.setattr(module.subprocess, 'check_output',
        lambda command, **kwargs:'' if '--query-compute-apps=pid' in command else '81559\n'*4)
    calls = []
    class Child:
        pid = 1
        def __init__(self, command, **kwargs): calls.append(command)
        def wait(self): return 0
    monkeypatch.setattr(module.subprocess, 'Popen', Child)
    monkeypatch.setattr(module.signal, 'signal', lambda *args:None)
    assert module.run(resume=False) == 0
    assert len(calls) == 3
    assert '--smoke-updates' in calls[0] and calls[1][-1].endswith('verify.py')
    assert calls[2][calls[2].index('--epochs')+1] == '6'
    assert calls[2][calls[2].index('--microbatch')+1] == '32'
    assert all('--resume' not in call for call in calls)
    assert json.loads((module.OUT/'launch-status.json').read_text())['accumulation_steps'] == 16
