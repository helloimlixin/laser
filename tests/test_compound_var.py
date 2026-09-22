from types import SimpleNamespace

import torch

from src.models.multiscale_laser_var import MultiScaleLaser, VAR
from src.models.compound_var import CompoundLaserVAR, compound_decompose


def test_short_attention_matches_reference_values_and_gradients():
    from src.models.compound_var import short_causal_attention
    for length in (1, 2, 4):
        torch.manual_seed(13)
        q, k, v = [torch.randn(3, 2, length, 8, dtype=torch.float64, requires_grad=True) for _ in range(3)]
        reference = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        actual = short_causal_attention(q, k, v)
        torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
        upstream = torch.randn_like(actual)
        expected_grads = torch.autograd.grad(reference, (q,k,v), upstream)
        grads = torch.autograd.grad(actual, (q,k,v), upstream, allow_unused=True)
        for tensor, expected, result in zip((q,k,v), expected_grads, grads):
            torch.testing.assert_close(torch.zeros_like(tensor) if result is None else result,
                                       expected, rtol=1e-11, atol=1e-12)


def fixture(monkeypatch):
    import dist
    monkeypatch.setattr(dist, 'get_device', lambda: 'cpu')
    torch.manual_seed(17)
    q = MultiScaleLaser(channels=4, atoms=8, sparsity=2, patch_nums=(1, 2, 3),
                        coefficient_bins=33, coefficient_max=3.).eval()
    model = CompoundLaserVAR(SimpleNamespace(quantize=q), depth=2, width=32, heads=2,
                             local_width=16, local_heads=2, num_classes=2, dropout=0.).eval()
    model.cond_drop_rate = 0
    return q, model


def test_stochastic_multiscale_codes_reconstruct_the_actual_context(monkeypatch):
    q, _ = fixture(monkeypatch)
    z = torch.randn(2, 4, 3, 3)
    options = dict(atom_temperatures=[.01] * 3, coefficient_temperatures=[.005] * 3,
                   stochastic=True)
    codes = compound_decompose(q, z, generator=torch.Generator().manual_seed(3), **options)
    again = compound_decompose(q, z, generator=torch.Generator().manual_seed(3), **options)
    for key in ('atoms', 'coefficients', 'inputs'):
        torch.testing.assert_close(codes[key], again[key])
    latent, inputs = q.from_codes(codes['atoms'], codes['coefficients'])
    torch.testing.assert_close(codes['latent'], latent)
    torch.testing.assert_close(codes['inputs'], inputs)
    assert not (codes['atoms'][..., 0] == codes['atoms'][..., 1]).any()
    assert torch.isfinite(codes['coefficient_probabilities']).all()
    torch.testing.assert_close(codes['coefficient_probabilities'].sum(-1), torch.ones_like(codes['coefficients']).float())


def test_deterministic_codec_preserves_original_tokenizer(monkeypatch):
    q, _ = fixture(monkeypatch)
    z = torch.randn(2, 4, 3, 3)
    native, compound = q.decompose(z), compound_decompose(q, z)
    for key in ('atoms', 'coefficients', 'inputs', 'latent'):
        torch.testing.assert_close(native[key], compound[key])


def test_pair_causality_and_nonlinear_atom_context_interaction(monkeypatch):
    q, model = fixture(monkeypatch)
    codes = compound_decompose(q, torch.randn(2, 4, 3, 3))
    features = torch.randn(2, 14, 32)
    a, c = model.token_logits(features, codes['atoms'], codes['coefficients'])
    ids = codes['coefficients'].clone()
    ids[:, :, 1] = (ids[:, :, 1] + 3) % q.coefficient_bins
    aa, cc = model.token_logits(features, codes['atoms'], ids)
    torch.testing.assert_close(a, aa)
    torch.testing.assert_close(c, cc)
    ids[:, :, 0] = (ids[:, :, 0] + 8) % q.coefficient_bins
    aa, cc = model.token_logits(features, codes['atoms'], ids)
    torch.testing.assert_close(a[:, :, 0], aa[:, :, 0])
    assert not torch.allclose(c[:, :, 1], cc[:, :, 1])
    h0, h1 = torch.randn(2, 3, 16), torch.randn(2, 3, 16)
    v0, v1 = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    interaction = (model.coefficient_logits(h0, v0, 0) - model.coefficient_logits(h0, v1, 0)
                   - model.coefficient_logits(h1, v0, 0) + model.coefficient_logits(h1, v1, 0))
    assert interaction.abs().max() > 1e-4


def test_teacher_forcing_equals_cached_scale_and_pair_predictions(monkeypatch):
    q, model = fixture(monkeypatch)
    codes = compound_decompose(q, torch.randn(2, 4, 3, 3))
    labels = torch.tensor([0, 1])
    with torch.no_grad():
        features = VAR.forward(model, labels, codes['inputs'])
        a, c = model.token_logits(features, codes['atoms'], codes['coefficients'])
        cached = model.sample(labels, cfg=0, teacher_codes=codes, return_details=True)
    torch.testing.assert_close(a, cached['atom_logits'], atol=3e-6, rtol=2e-5)
    torch.testing.assert_close(c, cached['coefficient_logits'], atol=3e-6, rtol=2e-5)
    torch.testing.assert_close(codes['latent'], cached['latent'])
    assert all(not b.attn.caching for b in model.blocks)


def test_soft_compound_objective_and_finite_gradients(monkeypatch):
    q, model = fixture(monkeypatch)
    codes = compound_decompose(q, torch.randn(2, 4, 3, 3),
                               coefficient_temperatures=[.01] * 3, stochastic=True)
    loss, parts = model(torch.tensor([0, 1]), codes['inputs'], codes['atoms'],
                        codes['coefficients'], codes['coefficient_probabilities'])
    torch.testing.assert_close(loss.detach(), (1.5 * parts[0] + parts[1]) / 2.5)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert parameter.grad.isfinite().all(), name
    first = model.sample(torch.tensor([0, 1]), seed=19, top_k=4)
    second = model.sample(torch.tensor([0, 1]), seed=19, top_k=4)
    torch.testing.assert_close(first, second)


def test_partial_teacher_prefix_is_exact_and_never_leaks_future_codes(monkeypatch):
    q, model = fixture(monkeypatch)
    codes = compound_decompose(q, torch.randn(2, 4, 3, 3))
    labels = torch.tensor([0, 1])
    options = dict(cfg=0, seed=42, teacher_scales=2, return_details=True)
    result = model.sample(labels, teacher_codes=codes, **options)
    prefix = 1 + 4
    for key in ('atoms', 'coefficients'):
        torch.testing.assert_close(result[key][:, :prefix], codes[key][:, :prefix])
    changed = {k: v.clone() for k, v in codes.items()}
    changed['atoms'][:, prefix:] = (changed['atoms'][:, prefix:] + 1) % q.vocab_size
    changed['coefficients'][:, prefix:] = (changed['coefficients'][:, prefix:] + 7) % q.coefficient_bins
    again = model.sample(labels, teacher_codes=changed, **options)
    for key in ('atoms', 'coefficients', 'latent'):
        torch.testing.assert_close(result[key], again[key], rtol=0, atol=0)
    ordinary = model.sample(labels, cfg=0, seed=42)
    unused = model.sample(labels, cfg=0, seed=42, teacher_codes=codes, teacher_scales=0,
                          atom_temperature=[1., 1., 1.], coefficient_temperature=1., coefficient_top_p=1.)
    torch.testing.assert_close(ordinary, unused, rtol=0, atol=0)


def test_sampling_controls_reject_invalid_settings_and_preserve_teacher_roundtrip(monkeypatch):
    import pytest
    q, model = fixture(monkeypatch)
    codes = compound_decompose(q, torch.randn(2, 4, 3, 3))
    labels = torch.tensor([0, 1])
    for kwargs in (dict(atom_temperature=0), dict(coefficient_temperature=float('nan')),
                   dict(atom_temperature=[1., 1.]), dict(coefficient_top_p=1.1),
                   dict(teacher_scales=1), dict(teacher_scales=4)):
        with pytest.raises(ValueError):
            model.sample(labels, **kwargs)
    result = model.sample(labels, cfg=0, teacher_codes=codes, teacher_scales=3,
                          atom_temperature=.8, coefficient_temperature=.7)
    torch.testing.assert_close(result, codes['latent'])


def test_scale_loss_weighting_preserves_normalization_and_checkpoint_compatibility(monkeypatch):
    import pytest
    q, model = fixture(monkeypatch)
    options = dict(depth=2, width=32, heads=2, local_width=16, local_heads=2, num_classes=2, dropout=0.)
    weighted = CompoundLaserVAR(SimpleNamespace(quantize=q), scale_loss_weights=[1., 4., 2.], **options).eval()
    weighted.cond_drop_rate = 0
    weighted.load_state_dict(model.state_dict(), strict=True)
    assert 'position_loss_weights' not in weighted.state_dict()
    assert weighted.position_loss_weights.mean() == pytest.approx(1.)
    codes = compound_decompose(q, torch.randn(2, 4, 3, 3))
    labels = torch.tensor([0, 1])
    loss, _ = weighted(labels, codes['inputs'], codes['atoms'], codes['coefficients'])
    features = VAR.forward(model, labels, codes['inputs'])
    a, c = model.token_logits(features, codes['atoms'], codes['coefficients'])
    nll_a = torch.nn.functional.cross_entropy(a.float().flatten(0, 2), codes['atoms'].flatten(), reduction='none').reshape_as(codes['atoms'])
    nll_c = torch.nn.functional.cross_entropy(c.float().flatten(0, 2), codes['coefficients'].flatten(), reduction='none').reshape_as(codes['coefficients'])
    expected = sum(w * (1.5 * nll_a[:, lo:hi] + nll_c[:, lo:hi]).sum()
                   for w, (lo, hi) in zip([1., 4., 2.], model.begin_ends)) / (2 * 2 * (1 + 4*4 + 2*9) * 2.5)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in weighted.parameters())
    weighted.position_loss_weights.fill_(1.)
    baseline, _ = model(labels, codes['inputs'], codes['atoms'], codes['coefficients'])
    uniform, _ = weighted(labels, codes['inputs'], codes['atoms'], codes['coefficients'])
    torch.testing.assert_close(baseline, uniform)
    with pytest.raises(ValueError, match='Scale loss weights'):
        CompoundLaserVAR(SimpleNamespace(quantize=q), scale_loss_weights=[1., 0., 2.], **options)


def test_production_sampling_options_cannot_enable_teacher_forcing():
    import pytest
    from omegaconf import OmegaConf
    from src.training.compound_var import CompoundExperiment
    experiment = CompoundExperiment.__new__(CompoundExperiment)
    experiment.cfg = OmegaConf.create(dict(sampling=dict(atom_temperature=[1., .7, .7, 1., 1.])))
    assert experiment.sampling_options() == {'atom_temperature': [1., .7, .7, 1., 1.]}
    experiment.cfg.sampling.teacher_scales = 3
    with pytest.raises(ValueError, match='Unknown unconditional'):
        experiment.sampling_options()


def test_changed_sampling_policy_is_rejected_by_training_resume_contract():
    import pytest
    from omegaconf import OmegaConf
    from src.training.compound_var import CompoundExperiment
    from src.training.compound_resume import prepare_resume
    experiment = CompoundExperiment.__new__(CompoundExperiment)
    experiment.cfg = OmegaConf.create(dict(model={}, prior=dict(batch_size=128, accumulation=1, epochs=50),
                                           compound={}, seed=0, data={}))
    old = experiment.contract()
    assert 'sampling' not in old
    saved = dict(world_size=3, training_config=old, progress=dict(epoch=50, batch=0, step=7800))
    experiment.cfg.sampling = dict(atom_temperature=.6)
    with pytest.raises(ValueError, match='changed on resume'):
        prepare_resume(saved, experiment.contract(), 3, 60000)


def test_rq_selection_uses_full_reference_and_protects_resume(monkeypatch):
    import pytest
    from copy import deepcopy
    from omegaconf import OmegaConf
    from src.training.compound_var import CompoundExperiment
    from src.training.compound_resume import prepare_resume
    from src.training import rq_reference_evaluation
    experiment = CompoundExperiment.__new__(CompoundExperiment)
    experiment.rank = 0
    experiment.cfg = OmegaConf.create(dict(model={}, prior=dict(batch_size=128, accumulation=1, epochs=100),
        compound={}, seed=0, data={}, sampling=dict(atom_temperature=.6), evaluation=dict(
            selection_protocol='rq_train_50k', full_samples=50000, batch_size=64,
            rq_fid_reference='/reference/ffhq_256_train.npz', rq_fid_reference_sha256='original-reference')))
    calls = []
    def evaluate(*args):
        calls.append(args)
        return dict(fid=23.5)
    monkeypatch.setattr(rq_reference_evaluation, 'evaluate_rq_reference', evaluate)
    model = object()
    assert experiment.selection_fid(model, 70) == 23.5
    assert calls[0][1:] == (model, 70, 50000, '/reference/ffhq_256_train.npz', 'original-reference', 'selection')
    saved = dict(world_size=3, training_config=deepcopy(experiment.contract()),
                 progress=dict(epoch=70, batch=0, step=10920))
    experiment.cfg.evaluation.rq_fid_reference_sha256 = 'changed-reference'
    with pytest.raises(ValueError, match='changed on resume'):
        prepare_resume(saved, experiment.contract(), 3, 60000)
    experiment.cfg.evaluation.full_samples = 2000
    with pytest.raises(ValueError, match='50000 samples'):
        experiment.selection_fid(model, 80)
    assert len(calls) == 1


def test_legacy_fid_selection_preserves_heldout_metric(tmp_path):
    import json
    from omegaconf import OmegaConf
    from src.training.compound_var import CompoundExperiment
    experiment = CompoundExperiment.__new__(CompoundExperiment)
    experiment.rank, experiment.out = 0, tmp_path
    experiment.cfg = OmegaConf.create(dict(evaluation=dict(full_samples=2000)))
    calls = []
    experiment.generate = lambda *args: calls.append(args)
    (tmp_path / 'generation-epoch060-2000.json').write_text(json.dumps(dict(pytorch_fid_diagnostic=37.3)))
    model = object()
    assert experiment.selection_fid(model, 60) == 37.3
    assert calls == [(model, 60, 2000)]
    assert experiment.fid_selection() is None


def test_recipe_routes_to_compound_backend_and_uses_all_three_gpus(monkeypatch):
    from pathlib import Path
    from src.training import cli
    recipe = Path(__file__).resolve().parents[1] / 'configs/experiments/celebahq256-var-compound.yaml'
    cfg = cli.load_config(recipe)
    assert cfg.prior.batch_size * cfg.prior.accumulation * 3 == 384
    assert cfg.evaluation.full_samples == 2000
    assert cfg.compound.atom_loss_weight == 1.5
    assert cfg.model.coefficient_bins == 257
    calls = []
    monkeypatch.setattr(cli, 'import_module', lambda name: SimpleNamespace(run=lambda cfg: calls.append(name)))
    cli.run(cfg)
    assert calls == ['src.training.compound_var']


def test_cached_images_accept_numpy_evaluation_indices():
    import numpy as np
    from src.training.var_laser import HFSquareImages

    class StrictDataset:
        def __getitem__(self, index):
            assert type(index) is int
            return {'image': torch.zeros(3, 4, 4), 'label': 1}

    images = HFSquareImages.__new__(HFSquareImages)
    images.dataset, images.huggingface = StrictDataset(), True
    images.seed, images.epoch, images.transform = 0, 0, lambda x: x
    image, label = images[np.int64(2)]
    assert image.shape == (3, 4, 4) and label == 1
