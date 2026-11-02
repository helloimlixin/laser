import json
import sys
from types import SimpleNamespace

import pytest
import torch

from src.training.rq_sample_panels import CLASSES, MEDIA_KEYS
from src.training.rqtransformer import sample_class_grid


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.labels = []
    def sample_sparse(self, count, aux, *, cond, **kwargs):
        self.labels.extend(cond.tolist())
        return cond[:, None, None, None]


def test_actual_preview_conditions_and_media_match_recent_imagenet_runs(tmp_path, monkeypatch):
    model = Model().train()
    aux = SimpleNamespace(decode_tokens=lambda tokens:
        (tokens.float()/1000*2-1).expand(-1, 3, 4, 4))
    logged = []
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(Image=lambda path: {'path': path}))
    path = sample_class_grid(model, aux, [str(i) for i in range(1000)], tmp_path, 42,
                             wb=SimpleNamespace(log=logged.append), sample_batch_size=7)
    expected = [label for _, label in CLASSES for _ in range(8)]
    assert model.labels == expected
    assert model.training
    manifest = json.loads(path.with_suffix('.json').read_text())
    assert manifest['class_ids'] == [label for _, label in CLASSES]
    assert manifest['num_samples'] == 64 and manifest['samples_per_class'] == 8
    assert all(key in logged[0] for key in MEDIA_KEYS)
    assert logged[0]['train/global_step'] == logged[0]['samples/global_step'] == 42
    assert all(__import__('pathlib').Path(manifest['media'][key]).is_file() for key in MEDIA_KEYS)


def test_designated_64_per_class_uses_existing_overview_and_class_grid_layout(tmp_path, monkeypatch):
    from src.training import rq_sample_panels
    model = Model()
    aux = SimpleNamespace(decode_tokens=lambda tokens:
        (tokens.float()/1000*2-1).expand(-1, 3, 256, 256))
    calls = []
    def panels(images, directory, step):
        calls.append((images.shape, images.dtype, step))
        paths = {key: directory/(key.replace('/', '-')+'.png') for key in MEDIA_KEYS}
        for path in paths.values(): path.write_bytes(b'test renderer output')
        return paths
    monkeypatch.setattr(rq_sample_panels, 'save_panels', panels)
    path = sample_class_grid(model, aux, None, tmp_path, 44,
        num_samples=512, samples_per_class=64, sample_batch_size=32)
    assert calls == [(torch.Size([512, 3, 256, 256]), torch.uint8, 44)]
    assert model.labels == [label for _, label in CLASSES for _ in range(64)]
    assert json.loads(path.with_suffix('.json').read_text())['num_samples'] == 512


@pytest.mark.parametrize('labels', [[1], [1]*8, [-1,1,2,3,4,5,6,7], [1000,1,2,3,4,5,6,7]])
def test_reject_invalid_preview_class_overrides(tmp_path, labels):
    with pytest.raises(ValueError, match='preview classes'):
        sample_class_grid(Model(), None, None, tmp_path, 0, class_ids=labels)
