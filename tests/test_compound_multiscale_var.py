from types import SimpleNamespace

import pytest
import torch

from src.models.multiscale_laser_var import MultiScaleLaser, VAR
from src.models.compound_multiscale_var import CompoundLaserVAR


@pytest.fixture
def setup_model(monkeypatch):
    import dist
    monkeypatch.setattr(dist, 'get_device', lambda: 'cpu')
    torch.manual_seed(31)
    q = MultiScaleLaser(channels=4, atoms=8, sparsity=2,
                        patch_nums=(1, 2, 3), coefficient_bins=512).eval()
    model = CompoundLaserVAR(SimpleNamespace(quantize=q), depth=2, width=32,
                             heads=2, num_classes=3, token_width=16,
                             token_heads=2, token_layers=2).eval()
    model.cond_drop_rate = 0.
    codes = q.decompose(torch.randn(2, 4, 3, 3))
    return q, model, codes


def test_512_bin_codec_and_offset_vocabulary(setup_model):
    q, model, codes = setup_model
    ids = torch.arange(512)
    assert torch.equal(q.coefficient_ids(q.coefficient_values(ids, 0), 0), ids)
    latent, inputs = q.from_codes(codes['atoms'], codes['coefficients'])
    torch.testing.assert_close(latent, codes['latent'])
    torch.testing.assert_close(inputs, codes['inputs'])
    tokens = model.pack_tokens(codes['atoms'], codes['coefficients'])
    assert model.V == 520 and model.L == 14
    assert (tokens[..., ::2] < 8).all()
    assert ((tokens[..., 1::2] >= 8) & (tokens[..., 1::2] < 520)).all()
    torch.testing.assert_close(tokens[..., 1::2] - 8, codes['coefficients'])


def test_shifted_causal_tokens_masks_and_all_gradients(setup_model):
    q, model, codes = setup_model
    features = torch.randn(2, 14, 32)
    tokens = model.pack_tokens(codes['atoms'], codes['coefficients'])
    logits = model.token_logits(features, tokens)
    assert torch.isneginf(logits[:, :, ::2, 8:]).all()
    assert torch.isneginf(logits[:, :, 1::2, :8]).all()
    changed = tokens.clone()
    changed[..., 1] = 8 + (changed[..., 1] - 8 + 9) % 512
    later = model.token_logits(features, changed)
    torch.testing.assert_close(logits[..., :2, :], later[..., :2, :])
    assert not torch.allclose(logits[..., 2:, :], later[..., 2:, :])
    for slot in range(4):
        next_scores = model.prefix_logits(features, tokens[..., :slot])
        expected = logits[..., slot, :]
        finite = expected.isfinite()
        torch.testing.assert_close(next_scores[finite], expected[finite], atol=1e-6, rtol=1e-5)
    loss, parts = model(torch.tensor([0, 1]), codes['inputs'], codes['atoms'], codes['coefficients'])
    torch.testing.assert_close(loss.detach(), parts.sum())
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None and param.grad.isfinite().all(), name


@pytest.mark.parametrize('guidance', [0., 1.5])
def test_cached_sampling_matches_training_and_reconstructs(setup_model, monkeypatch, guidance):
    import src.models.compound_multiscale_var as module
    q, model, codes = setup_model
    labels = torch.tensor([0, 1])
    doubled = torch.cat((labels, torch.full_like(labels, model.num_classes)))
    features = VAR.forward(model, doubled, codes['inputs'].repeat(2, 1, 1))
    tokens = model.pack_tokens(codes['atoms'], codes['coefficients'])
    calls = []

    def force_targets(scores, **kwargs):
        scale, slot = divmod(len(calls), 4)
        lo, hi = model.begin_ends[scale]
        prefix = tokens[:, lo:hi, :slot].repeat(2, 1, 1)
        expected = model.prefix_logits(features[:, lo:hi], prefix)
        expected = expected[..., :8] if slot % 2 == 0 else expected[..., 8:]
        strength = guidance * scale / model.num_stages_minus_1
        expected = (1 + strength) * expected[:2] - strength * expected[2:]
        if slot == 2:
            expected.scatter_(-1, tokens[:, lo:hi, :slot:2], -torch.inf)
        torch.testing.assert_close(scores, expected, atol=2e-6, rtol=1e-4)
        calls.append((scale, slot))
        target = tokens[:, lo:hi, slot, None]
        return target if slot % 2 == 0 else target - 8

    monkeypatch.setattr(module, 'sample_with_top_k_top_p_', force_targets)
    latent = model.sample(labels, cfg=guidance)
    torch.testing.assert_close(latent, codes['latent'])
    assert len(calls) == 12
    assert all(not block.attn.caching for block in model.blocks)


def test_sampling_repeatable_and_cleans_cache_on_failure(setup_model, monkeypatch):
    import src.models.compound_multiscale_var as module
    _, model, _ = setup_model
    labels = torch.tensor([0, 1])
    torch.testing.assert_close(model.sample(labels, seed=41, top_k=4),
                               model.sample(labels, seed=41, top_k=4))
    def fail(*args, **kwargs):
        raise RuntimeError('injected sampling failure')
    monkeypatch.setattr(module, 'sample_with_top_k_top_p_', fail)
    with pytest.raises(RuntimeError, match='injected'):
        model.sample(labels)
    assert all(not block.attn.caching for block in model.blocks)
