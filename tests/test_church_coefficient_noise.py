import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.church_coefficient_noise import CalibratedChurchAux, physical_coefficient_distribution


def aux(sigma=.1875):
    return SimpleNamespace(coeff_bins=torch.linspace(-3, 3, 2048),
        coeff_scales=torch.tensor([7.4, 4.16, 2.455, 1.648]),
        coeff_vocab_size=2048, coefficient_sigma=sigma)


def test_noise_std_is_physical_and_independent_of_depth_scale():
    a = aux()
    probabilities = physical_coefficient_distribution(torch.zeros(1, 4),
        a.coeff_bins, a.coeff_scales, a.coefficient_sigma)
    physical_bins = a.coeff_bins * a.coeff_scales[:, None]
    mean = (probabilities * physical_bins).sum(-1)
    std = (probabilities * (physical_bins-mean[..., None]).square()).sum(-1).sqrt()
    torch.testing.assert_close(mean, torch.zeros_like(mean), atol=2e-7, rtol=0.)
    torch.testing.assert_close(std, torch.full_like(std, a.coefficient_sigma), atol=2e-7, rtol=2e-6)
    # There are still multiple neighboring bins: calibration did not silently
    # turn stochastic soft-target training into nearest-bin classification.
    assert (probabilities > .001).sum(-1).min() > 10


def test_same_distribution_is_used_for_labels_and_sampled_context():
    a = aux()
    c = torch.tensor([.12, -.5, .05, 1.]).expand(4096, 4)
    torch.manual_seed(36)
    sampled, probs = CalibratedChurchAux.compound_coeff_ids(a, c)
    nearest, deterministic_probs = CalibratedChurchAux.compound_coeff_ids(a, c, stochastic=False)
    assert torch.equal(probs, deterministic_probs)
    expected = ((c+3)*(2047/6)).round().long()
    assert torch.equal(nearest, expected)
    assert not torch.equal(sampled, nearest)
    sampled_values = a.coeff_bins[sampled] * a.coeff_scales
    observed = (sampled_values-c*a.coeff_scales).std(0)
    torch.testing.assert_close(observed, torch.full_like(observed, a.coefficient_sigma), atol=.008, rtol=0.)


def test_boundaries_are_normalized_without_nan_or_out_of_range_ids():
    a = aux(.0625)
    c = torch.tensor([-3., 3., -2.999, 2.999]).expand(32, 4)
    ids, p = CalibratedChurchAux.compound_coeff_ids(a, c)
    assert torch.isfinite(p).all() and ids.min() >= 0 and ids.max() < 2048
    torch.testing.assert_close(p.sum(-1), torch.ones_like(c))
    for invalid in (0., -1., float('nan'), float('inf')):
        with pytest.raises(ValueError):
            physical_coefficient_distribution(c, a.coeff_bins, a.coeff_scales, invalid)


def test_calibrated_trainer_preserves_training_loop_and_generation():
    root = Path(__file__).resolve().parents[1]
    old, new = [ast.parse((root/f'scripts/{name}.py').read_text()) for name in
                ('train_church_ffhq_archived', 'train_church_ffhq_noise')]
    def functions(tree):
        return {n.name:n for n in tree.body if isinstance(n, ast.FunctionDef)}
    a, b = functions(old), functions(new)
    for name in ('evaluate', 'generate'):
        assert ast.dump(a[name]) == ast.dump(b[name])
    # All changes are confined to construction/configuration before logging.
    # The optimizer update, stream, LR, evaluation and checkpoint paths match.
    for typ in (ast.Try,):
        left = [n for n in a['main'].body if isinstance(n, typ)]
        right = [n for n in b['main'].body if isinstance(n, typ)]
        assert [ast.dump(n) for n in left] == [ast.dump(n) for n in right]
