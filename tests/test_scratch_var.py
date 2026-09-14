from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.models.scratch_var import (
    build_scratch_tokenizer, build_scratch_prior, state_digest, shared_prior_digest,
)
from src.training.cli import load_config


class Config(SimpleNamespace):
    def get(self, name, default=None):
        return getattr(self, name, default)


def config(kind):
    return Config(bottleneck=kind, pretrained_vae=None, channels=4, atoms=8,
                  vae_width=32, patch_nums=(1,2), sparsity=2, coefficient_bins=33,
                  coefficient_range_decay=.9)


def test_scratch_never_loads_weights_and_backbones_match(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Scratch model attempted to load external weights')
    monkeypatch.setattr(torch, 'load', forbidden)
    vq = build_scratch_tokenizer(config('vq'), 41)
    laser = build_scratch_tokenizer(config('laser'), 41)
    assert state_digest(vq, shared_only=True) == state_digest(laser, shared_only=True)
    torch.testing.assert_close(vq.quantize.embedding.weight.T, laser.quantize.normalized_dictionary())
    assert all(p.requires_grad for p in vq.parameters())
    assert all(p.requires_grad for p in laser.parameters())
    different = build_scratch_tokenizer(config('laser'), 42)
    assert state_digest(laser, shared_only=True) != state_digest(different, shared_only=True)


def test_scratch_rejects_any_pretrained_path():
    cfg=config('laser')
    cfg.pretrained_vae='even-a-missing-file.pth'
    with pytest.raises(ValueError,match='forbids'):
        build_scratch_tokenizer(cfg, 0)


def test_prior_shared_weights_are_identical_and_vq_forward_is_official(monkeypatch):
    import dist
    monkeypatch.setattr(dist,'get_device',lambda:'cpu')
    vq = build_scratch_tokenizer(config('vq'), 41)
    laser = build_scratch_tokenizer(config('laser'), 41)
    baseline = build_scratch_prior(vq, 'vq', 71, depth=2, width=32, heads=2, num_classes=3)
    candidate = build_scratch_prior(laser, 'laser', 71, depth=2, width=32, heads=2, num_classes=3)
    assert shared_prior_digest(baseline) == shared_prior_digest(candidate)
    baseline.eval(); baseline.cond_drop_rate=0.
    atoms=torch.randint(8,(2,5,1)); inputs=torch.randn(2,4,4); labels=torch.tensor([0,1])
    from src.models.multiscale_laser_var import VAR
    torch.manual_seed(3)
    logits=VAR.forward(baseline,labels,inputs)
    expected=torch.nn.functional.cross_entropy(logits.flatten(0,1),atoms.flatten())
    torch.manual_seed(3)
    loss,parts=baseline(labels,inputs,atoms,torch.zeros_like(atoms))
    torch.testing.assert_close(expected,loss)
    loss.backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in baseline.parameters())


def test_ranges_change_only_in_training_and_codes_still_roundtrip():
    q=build_scratch_tokenizer(config('laser'),41).quantize
    z=torch.randn(2,4,2,2)
    original=q.coefficient_max.clone()
    q.train(); q.decompose(z)
    assert not torch.equal(original,q.coefficient_max)
    frozen=q.coefficient_max.clone()
    q.eval(); codes=q.decompose(z)
    assert torch.equal(frozen,q.coefficient_max)
    decoded,inputs=q.from_codes(codes['atoms'],codes['coefficients'])
    torch.testing.assert_close(decoded,codes['latent'])
    torch.testing.assert_close(inputs,codes['inputs'])


def test_pair_recipes_match_and_have_no_pretrained_initialization():
    root=Path(__file__).resolve().parents[1]/'configs/experiments'
    vq=load_config(root/'imagenet-var-vq-scratch.yaml')
    laser=load_config(root/'imagenet-var-scratch.yaml')
    assert vq.model.initialization==laser.model.initialization=='scratch'
    assert vq.model.pretrained_vae is None and laser.model.pretrained_vae is None
    assert vq.tokenizer==laser.tokenizer and vq.prior==laser.prior
    assert vq.data==laser.data and vq.evaluation==laser.evaluation
    assert vq.seed==laser.seed
    assert vq.output_dir != laser.output_dir
    assert vq.wandb.id != laser.wandb.id
