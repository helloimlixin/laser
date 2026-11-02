import hashlib
import importlib

import numpy as np
from PIL import Image
import pytest
import torch

from src import rq_upstream
from src.training.cc3m_paper_metrics import (
    PROTOCOL, PAPER_SAMPLING, clip_scores, fid_from_moments, load_clip, load_reference)


def test_fid_moments_match_the_analytic_unbiased_covariance_across_shards():
    features = torch.tensor([[1., 2.], [1., -2.], [-1., 2.], [-1., -2.]], dtype=torch.float64)
    shards = features.chunk(2)
    total = sum(s.sum(0) for s in shards)
    cross = sum(s.T @ s for s in shards)
    reference = (np.array([.25, -.5]), np.eye(2))
    variances = np.array([4/3, 16/3])
    expected = (.25**2 + .5**2) + np.sum(variances + 1 - 2*np.sqrt(variances))
    assert fid_from_moments(total, cross, 4, reference) == pytest.approx(expected, abs=1e-12)
    with pytest.raises(ValueError, match='at least two'):
        fid_from_moments(total, cross, 1, reference)


def test_reference_requires_verified_bytes_and_a_declared_paper_protocol(tmp_path):
    path = tmp_path/'reference.npz'
    np.savez(path, mu=np.zeros(2048), sigma=np.eye(2048))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    cfg = dict(sampling=dict(PAPER_SAMPLING), evaluation=dict(
        protocol=PROTOCOL, fid_reference=str(path), fid_reference_sha256=digest))
    mean, covariance = load_reference(cfg)
    assert mean.shape == (2048,) and covariance.shape == (2048, 2048)
    cfg['evaluation']['fid_reference_sha256'] = '0'*64
    with pytest.raises(ValueError, match='SHA256'):
        load_reference(cfg)
    cfg['evaluation']['protocol'] = 'old_subset_reference'
    with pytest.raises(ValueError, match='explicit'):
        load_reference(cfg)


def test_reference_rejects_the_previous_sampler_before_evaluation():
    cfg = dict(sampling=dict(temperature=1., top_k=2048, top_p=.95),
               evaluation=dict(protocol=PROTOCOL))
    with pytest.raises(ValueError, match='k=1024'):
        load_reference(cfg)


def test_clip_uses_uint8_pil_preprocessing_and_unscaled_cosine():
    from clip.clip import _transform
    preprocess = _transform(224)
    pixels = torch.linspace(.0009, .9991, 2*3*256*256).reshape(2, 3, 256, 256)

    class RecordingClip:
        def encode_image(self, images):
            self.images = images
            return torch.tensor([[3., 4.], [0., 1.]])

        def encode_text(self, tokens):
            self.tokens = tokens
            return torch.tensor([[1., 0.], [0., -2.]])

    model = RecordingClip()
    scores = clip_scores(pixels, ['a landscape', 'a dog'], model, preprocess, torch.device('cpu'))
    expected = torch.stack([preprocess(Image.fromarray(
        (p.permute(1, 2, 0).numpy()*255).astype(np.uint8))) for p in pixels])
    assert torch.equal(model.images, expected)
    assert model.tokens.shape == (2, 77)
    torch.testing.assert_close(scores, torch.tensor([.6, -1.]))
    mean = torch.tensor([.48145466, .4578275, .40821073])[None, :, None, None]
    std = torch.tensor([.26862954, .26130258, .27577711])[None, :, None, None]
    legacy = (torch.nn.functional.interpolate(pixels, (224, 224), mode='bicubic',
              align_corners=False, antialias=True)-mean)/std
    assert (model.images-legacy).abs().max() > .001


def test_clip_construction_keeps_the_released_cpu_fp32_weights(monkeypatch):
    module = importlib.import_module('rqvae.metrics.clip_score')
    model, preprocess = torch.nn.Linear(2, 2).float().train(), object()
    monkeypatch.setattr(module, 'get_clip', lambda: (model, preprocess))
    actual, transform = load_clip('cpu')
    assert actual is model and transform is preprocess
    assert not actual.training and next(actual.parameters()).dtype == torch.float32
