import importlib.util
import json
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / 'scripts/tools/supervise_var_critic_pipeline.py'
    spec = importlib.util.spec_from_file_location('var_critic_pipeline_supervisor', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def marker(base, arm, epoch=10, batch=0):
    folder = base / 'stage1' / arm
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'complete.json').write_text(json.dumps(dict(
        progress=dict(epoch=epoch, batch=batch, step=100090), checkpoint_sha256=arm)))


def test_stage2_waits_for_both_epoch10_tokenizers(tmp_path):
    supervisor = module()
    assert supervisor.stage2_gate(tmp_path) is None
    marker(tmp_path, 'laser')
    assert supervisor.stage2_gate(tmp_path) is None
    marker(tmp_path, 'vq')
    gate = supervisor.stage2_gate(tmp_path)
    assert gate['ready'] and set(gate['runs']) == {'laser', 'vq'}


@pytest.mark.parametrize('epoch,batch', [(9, 0), (10, 2), (11, 0)])
def test_stage2_rejects_incomplete_or_wrong_epoch(tmp_path, epoch, batch):
    marker(tmp_path, 'laser')
    marker(tmp_path, 'vq', epoch, batch)
    with pytest.raises(ValueError, match='incomplete'):
        module().stage2_gate(tmp_path)
