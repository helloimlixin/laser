import copy

from omegaconf import OmegaConf
import pytest
import torch
from torch.nn import functional as F

from src.models.discretized_coefficient_mixture import (
    DiscretizedCoefficientMixture, discretized_logistic_log_probs, install_coefficient_mixture,
)
from src.training.imagenet_compound_runtime import validate_resume_recipe
from src.training.rqtransformer import build_model
from tests.test_offset_pair_prior import case


@pytest.mark.parametrize('centers', [[-3., 3.], [-3., -1.5, -.1, .3, 3.], torch.linspace(-3, 3, 512)])
def test_probabilities_match_cdf_integrals_including_both_clamped_tails(centers):
    centers = torch.as_tensor(centers, dtype=torch.float64)
    logits = torch.tensor([[.2, -.5, .9]], dtype=torch.float64)
    means = torch.tensor([[-2., .1, 2.]], dtype=torch.float64)
    scales = torch.tensor([[.5, .8, .3]], dtype=torch.float64)
    edges = (centers[:-1] + centers[1:]) * .5
    cdf = torch.sigmoid((edges - means[..., None]) / scales[..., None])
    cdf = torch.cat((torch.zeros_like(cdf[..., :1]), cdf, torch.ones_like(cdf[..., :1])), -1)
    expected = ((cdf[..., 1:] - cdf[..., :-1]) * logits.softmax(-1)[..., None]).sum(-2)
    actual = discretized_logistic_log_probs(logits, means, scales, centers).exp()
    torch.testing.assert_close(actual, expected, atol=2.e-16, rtol=2.e-11)
    torch.testing.assert_close(actual.sum(-1), torch.ones(1, dtype=torch.float64))


def test_exact_bin_likelihood_has_correct_gradients():
    torch.manual_seed(3)
    inputs = tuple(x.requires_grad_() for x in (
        torch.randn(2, 3, dtype=torch.float64), torch.randn(2, 3, dtype=torch.float64),
        torch.rand(2, 3, dtype=torch.float64) + .2,
    ))
    centers = torch.tensor([-3., -.2, .4, 3.], dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda *p: discretized_logistic_log_probs(*p, centers), inputs)


def test_extreme_tail_targets_have_finite_loss_and_gradients_without_probability_floor():
    logits = torch.tensor([[0., -1000.]], requires_grad=True)
    means = torch.tensor([[100., -100.]], requires_grad=True)
    scales = torch.tensor([[.001, 100.]], requires_grad=True)
    p = discretized_logistic_log_probs(logits, means, scales, torch.linspace(-3, 3, 512))
    loss = -p[0, 0] - p[0, 255]
    assert torch.isfinite(p).all() and torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(x.grad).all() for x in (logits, means, scales))
    assert p[0, 255] < -1000  # genuine low probability, not clipped at log(epsilon)


def test_autocast_preserves_float32_probability_math_and_optimizer_updates():
    torch.manual_seed(4)
    head = DiscretizedCoefficientMixture(12, bins=512, width=16, components=8)
    x = torch.randn(32, 12)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.01)
    before = head.parameters_projection.weight.detach().clone()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        logp = head(x)
        loss = F.cross_entropy(logp, torch.arange(32) * 16)
    assert logp.dtype == torch.float32
    torch.testing.assert_close(logp.exp().sum(-1), torch.ones(32), atol=3.e-7, rtol=0)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    optimizer.step()
    assert not torch.equal(before, head.parameters_projection.weight)


@torch.no_grad()
def test_replacement_retains_atoms_causality_and_full_cached_scalar_predictions():
    _, model, aux, target, cond = case()
    before = model(target, aux, cond)
    original = copy.deepcopy(model)
    install_coefficient_mixture(model, coefficient_max=1., components=3, width=8)
    output = model(target, aux, cond)
    torch.testing.assert_close(output['atom_logits'], before['atom_logits'], rtol=0, atol=0)
    for name, parameter in model.named_parameters():
        if not name.startswith('coeff_classifier.'):
            assert torch.equal(parameter, dict(original.named_parameters())[name])
    changed = target.clone()
    changed[:, -1, -1, -1] = 7
    for key, value in output.items():
        torch.testing.assert_close(value, model(changed, aux, cond)[key], rtol=0, atol=0)
    generated = torch.zeros_like(target)
    generated[..., 1::2] = 9
    model.init_cache()
    for h in range(2):
        for w in range(2):
            for event in range(8):
                logits = model.cached_forward(generated, aux, cond, amp=False, sample_loc=(h, w, event))
                expected = output['coeff_logits' if event % 2 else 'atom_logits'][:, h, w, event // 2]
                actual = logits[:, 7:] if event % 2 else logits[:, :7]
                torch.testing.assert_close(actual, expected, rtol=1.e-5, atol=2.e-6)
                generated[:, h, w, event] = target[:, h, w, event]
    model.init_cache()
    sampled = model.sample_sparse(2, aux, torch.tensor([1, 999]), amp=False)
    atoms, coefficients = model.fields(sampled)
    assert ((coefficients >= 0) & (coefficients < 5)).all()
    assert (atoms.sort(-1).values.diff(dim=-1) > 0).all()
    model.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)


def test_production_build_and_recovery_require_matching_head_architecture():
    with torch.device('meta'):
        model = build_model(16896, 16384, coeff_vocab_size=512, sparsity_level=4,
            physical_pair_context=True, efficient_pair_backbone=True,
            compound_micro_transformer_layers=2, compound_depth_specific_coeff_heads=True,
            coefficient_history_layers=2, coefficient_history_normalize_fields=True,
            coefficient_head='discretized-logistic-mixture')
    assert len(model.coeff_classifier) == 4
    assert all(isinstance(x, DiscretizedCoefficientMixture) for x in model.coeff_classifier)
    assert model.token_vocabulary_size == 16896 and model.token_block_size == (8, 8, 8)
    validate_resume_recipe({}, OmegaConf.create({}))
    with pytest.raises(ValueError, match='coefficient_head'):
        validate_resume_recipe({}, OmegaConf.create({'coefficient_head': 'discretized-logistic-mixture'}))
    config = dict(coefficient_head='discretized-logistic-mixture', coefficient_mixture_components=8)
    validate_resume_recipe(config, OmegaConf.create(config))
    with pytest.raises(ValueError, match='coefficient_mixture_components'):
        validate_resume_recipe(config, OmegaConf.create({**config, 'coefficient_mixture_components': 16}))
