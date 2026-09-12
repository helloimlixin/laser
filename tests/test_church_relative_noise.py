import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.church_relative_noise import RelativeChurchAux, relative_coefficient_distribution


def aux():
    return SimpleNamespace(coeff_bins=torch.linspace(-3, 3, 2048),
        coeff_scales=torch.tensor([7.4008713, 4.1602407, 2.4553516, 1.6484600]),
        coeff_vocab_size=2048, sigma_cap=.1875, relative_sigma=.05, truncate=3.)


def test_bounded_draws_cover_weak_zero_strong_and_boundary_coefficients():
    a = aux()
    torch.manual_seed(72)
    c = torch.cat([torch.rand(500, 4)*6-3,
        torch.tensor([0., 1e-10, -1e-10, .01, -.01, 3., -3.])[:, None].expand(-1, 4)])
    ids, p = RelativeChurchAux.compound_coeff_ids(a, c)
    physical = c*a.coeff_scales
    delta = a.coeff_bins*a.coeff_scales[:, None]-physical[..., None]
    nearest_error = delta.abs().min(-1).values
    radius = (.05*physical.abs()).clamp_max(.1875)*3
    bound = torch.maximum(radius, nearest_error)
    assert not (p[delta.abs() > bound[..., None]] > 0).any()
    assert torch.isfinite(p).all()
    torch.testing.assert_close(p.sum(-1), torch.ones_like(c))
    assert ((a.coeff_bins[ids]*a.coeff_scales-physical).abs() <= bound+1e-7).all()


def test_weak_values_shrink_noise_without_broadening_quantization_floor():
    a = aux()
    physical = torch.tensor([0., 1e-9, .5, 1., 2., 4., 8.])[:, None].expand(-1, 4)
    ids, p = RelativeChurchAux.compound_coeff_ids(a, physical/a.coeff_scales, stochastic=False)
    centers = a.coeff_bins*a.coeff_scales[:, None]
    # Actual finite ranges apply to large c; inspect in-range examples here.
    mse = (p*(centers-physical[..., None]).square()).sum(-1)
    assert (p[:2] > 0).sum(-1).eq(1).all()
    assert (p[2:5] > 0).sum(-1).min() > 3
    torch.testing.assert_close(mse[2:5].sqrt(), .05*physical[2:5], rtol=.035, atol=.0005)
    nearest = (centers-physical[..., None]).abs().argmin(-1)
    assert torch.equal(ids, nearest)


def test_context_and_labels_share_distribution_and_no_sign_flip_above_bin_floor():
    a = aux()
    physical = torch.tensor([.5, -.5, 2., -2.]).expand(6000, 4)
    torch.manual_seed(91)
    ids, p = RelativeChurchAux.compound_coeff_ids(a, physical/a.coeff_scales)
    deterministic, q = RelativeChurchAux.compound_coeff_ids(a, physical/a.coeff_scales, stochastic=False)
    assert torch.equal(p, q) and not torch.equal(ids, deterministic)
    sampled = a.coeff_bins[ids]*a.coeff_scales
    assert torch.equal(sampled >= 0, physical >= 0)
    torch.testing.assert_close((sampled-physical).square().mean(0).sqrt(), physical[0].abs()*.05, rtol=.05, atol=.001)


@pytest.mark.parametrize('sigma,relative,truncate', [(0,.05,3),(.1,0,3),(.1,.5,3),(.1,.05,float('nan'))])
def test_invalid_noise_parameters(sigma, relative, truncate):
    a = aux()
    with pytest.raises(ValueError):
        relative_coefficient_distribution(torch.zeros(4), a.coeff_bins, a.coeff_scales, sigma, relative, truncate)


def test_trainer_keeps_archived_training_evaluation_and_generation():
    root = Path(__file__).resolve().parents[1]
    old, new = [ast.parse((root/f'scripts/{name}.py').read_text()) for name in
                ('train_church_ffhq_noise', 'train_church_relative_noise')]
    def functions(tree):
        return {n.name:n for n in tree.body if isinstance(n, ast.FunctionDef)}
    a, b = functions(old), functions(new)
    for name in ('evaluate', 'generate'):
        assert ast.dump(a[name]) == ast.dump(b[name])
    for typ in (ast.Try,):
        assert [ast.dump(n) for n in a['main'].body if isinstance(n, typ)] == [ast.dump(n) for n in b['main'].body if isinstance(n, typ)]
