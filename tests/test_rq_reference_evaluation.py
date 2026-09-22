from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from src.training import rq_reference_evaluation as evaluation


def fixture(tmp_path, monkeypatch):
    reference = tmp_path / 'reference.npz'
    np.savez(reference, mu=np.zeros(3), sigma=np.eye(3))
    observed = []

    class Moments:
        def __init__(self, device):
            self.n = 0

        def update(self, features):
            observed.append(features.clone())
            self.n += len(features)

        def finish(self):
            return self.n, np.zeros(3), np.eye(3)

    class Decoder:
        def fhat_to_img(self, latent):
            assert latent.dtype == torch.float32
            assert not torch.is_autocast_enabled('cpu')
            return torch.tensor([-1.8, -.75, .24691, 2.]).reshape(1, 1, 1, 4).expand(len(latent), 3, 1, 4)

    class Model:
        def eval(self):
            return self

        def sample(self, labels, **kwargs):
            assert labels.eq(0).all()
            return torch.ones(len(labels), 3, 1, 4, dtype=torch.bfloat16)

    monkeypatch.setattr(evaluation, 'FeatureMoments', Moments)
    monkeypatch.setattr(evaluation.dist, 'barrier', lambda: None)
    monkeypatch.setattr(evaluation, 'save_image', lambda *args, **kwargs: None)
    cfg = OmegaConf.create(dict(prior=dict(cfg=0., top_k=250, top_p=1.),
                                evaluation=dict(batch_size=3, preview_samples=64, grid_columns=8)))
    experiment = SimpleNamespace(cfg=cfg, inception=lambda pixels: pixels, vae=Decoder(),
                                 device=torch.device('cpu'), rank=0, world=1, amp=nullcontext,
                                 sampling_options=lambda: {},
                                 media_path=lambda name: tmp_path / name, out=tmp_path,
                                 log=lambda *args, **kwargs: None)
    return experiment, Model(), reference, evaluation.file_sha256(reference), observed


def test_rq_protocol_retains_continuous_pixels_and_exact_sample_count(tmp_path, monkeypatch):
    experiment, model, reference, checksum, observed = fixture(tmp_path, monkeypatch)
    result = evaluation.evaluate_rq_reference(experiment, model, 50, 7, reference, checksum, 'baseline')
    pixels = torch.cat(observed)
    assert len(pixels) == result['count'] == 7
    assert pixels.dtype == torch.float32
    torch.testing.assert_close(pixels[0, 0, 0], torch.tensor([0., .125, .623455, 1.]))
    assert not torch.equal(pixels, pixels.mul(255).round().div(255))
    assert result['fid'] == pytest.approx(0.)
    assert result['reference_sha256'] == checksum


def test_rq_reference_rejects_changed_statistics_before_generation(tmp_path, monkeypatch):
    experiment, model, reference, _, observed = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='checksum mismatch'):
        evaluation.evaluate_rq_reference(experiment, model, 50, 7, reference, 'wrong', 'best')
    assert not observed


def test_rq_reference_fails_if_distributed_count_is_incomplete(tmp_path, monkeypatch):
    experiment, model, reference, checksum, _ = fixture(tmp_path, monkeypatch)
    experiment.world = 3  # This stub intentionally omits the other two ranks.
    with pytest.raises(RuntimeError, match='sample count mismatch'):
        evaluation.evaluate_rq_reference(experiment, model, 50, 7, reference, checksum, 'best')
