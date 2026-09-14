from types import SimpleNamespace

import torch
from torch import nn

from src.models.multiscale_laser_var import MultiScaleLaser, LaserVAR


def tokenizer():
    torch.manual_seed(3)
    q = MultiScaleLaser(channels=4, atoms=8, sparsity=2,
                        patch_nums=(1, 2, 3), coefficient_bins=33, coefficient_max=3.)
    return q


def prior(q):
    import dist
    original = dist.get_device
    dist.get_device = lambda: "cpu"
    try:
        result = LaserVAR(SimpleNamespace(quantize=q), depth=2, width=32, heads=2, num_classes=3)
    finally:
        dist.get_device = original
    result.cond_drop_rate = 0
    return result.eval()


def test_quantized_roundtrip_and_dictionary_gradient():
    q = tokenizer()
    z = torch.randn(2, 4, 3, 3, requires_grad=True)
    codes = q.decompose(z)
    recovered, inputs = q.from_codes(codes['atoms'], codes['coefficients'])
    torch.testing.assert_close(recovered, codes['latent'])
    torch.testing.assert_close(inputs, codes['inputs'])
    assert not (codes['atoms'][..., 0] == codes['atoms'][..., 1]).any()
    out, _, loss = q(z)
    (out.square().mean() + loss).backward()
    assert z.grad.isfinite().all() and z.grad.abs().sum() > 0
    assert q.dictionary.dictionary.grad.isfinite().all()
    assert q.dictionary.dictionary.grad.abs().sum() > 0


def test_coefficient_vocabulary_has_exact_zero_and_closed_roundtrip():
    q = tokenizer()
    ids = torch.arange(q.coefficient_bins)
    values = q.coefficient_values(ids, 0)
    assert values[q.coefficient_bins // 2].item() == 0
    assert torch.equal(q.coefficient_ids(values, 0), ids)
    torch.testing.assert_close(values, -values.flip(0))


def test_current_and_future_scale_targets_do_not_leak_into_inputs():
    q = tokenizer()
    codes = q.decompose(torch.randn(2, 4, 3, 3))
    atoms = codes['atoms'].clone()
    atoms[:, 1:] = (atoms[:, 1:]+1) % q.vocab_size
    _, changed = q.from_codes(atoms, codes['coefficients'])
    # Input for all four positions of scale 2 uses only scale 1.
    torch.testing.assert_close(changed[:, :4], codes['inputs'][:, :4])
    model = prior(q)
    features = super(LaserVAR, model).forward(torch.tensor([0, 1]), codes['inputs'])
    changed_inputs = codes['inputs'].clone()
    changed_inputs[:, 4:] += 10
    changed_features = super(LaserVAR, model).forward(torch.tensor([0, 1]), changed_inputs)
    torch.testing.assert_close(features[:, :5], changed_features[:, :5])


def test_sparse_depth_is_causal_and_joint_loss_has_gradients():
    q = tokenizer()
    model = prior(q)
    codes = q.decompose(torch.randn(2, 4, 3, 3))
    features = torch.randn(2, 14, 32)
    a, c = model.token_logits(features, codes['atoms'], codes['coefficients'])
    modified = (codes['atoms'] + 1) % q.vocab_size
    aa, cc = model.token_logits(features, modified, codes['coefficients'])
    torch.testing.assert_close(a[:, :, 0], aa[:, :, 0])
    assert not torch.allclose(c[:, :, 0], cc[:, :, 0])
    loss, parts = model(torch.tensor([0, 1]), codes['inputs'], codes['atoms'], codes['coefficients'])
    torch.testing.assert_close(loss.detach(), parts.sum())
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and p.grad.isfinite().all(), name


def test_sampling_is_repeatable_and_clears_kv_cache():
    q = tokenizer()
    model = prior(q)
    labels = torch.tensor([0, 1])
    first = model.sample(labels, seed=7, top_k=4)
    second = model.sample(labels, seed=7, top_k=4)
    torch.testing.assert_close(first, second)
    assert first.shape == (2, 4, 3, 3) and first.isfinite().all()
    assert all(not block.attn.caching for block in model.blocks)


def test_public_recipe_and_backend_dispatch(monkeypatch):
    from pathlib import Path
    from src.training import cli
    recipe = Path(__file__).resolve().parents[1] / 'configs/experiments/imagenet-laser-var.yaml'
    cfg = cli.load_config(recipe)
    assert cfg.model.patch_nums[-1] == cfg.data.image_size // 16
    assert sum(p*p for p in cfg.model.patch_nums) == 680
    calls = []
    monkeypatch.setattr(cli, 'import_module', lambda name: SimpleNamespace(run=lambda cfg: calls.append(name)))
    cli.run(cfg)
    assert calls == ['src.training.var_laser']
