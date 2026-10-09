import torch
import pytest

from src.training.coefficient_jitter import jittered_scalar_targets, jitter_moments, jitter_probabilities
from src.training.coefficient_diagnostics import coefficient_diagnostic_sums, coefficient_diagnostic_metrics


def test_scalar_target_path_matches_integrated_bounded_jitter_and_layout():
    bins = torch.linspace(-3, 3, 2048)
    width = 6 / 2047
    # Include bin centers, both sides of a boundary, and saturation tails.
    c = torch.tensor([bins[1024], (bins[1024]+bins[1025])/2-width*.004,
                      (bins[1024]+bins[1025])/2+width*.004, 3.2]).reshape(1, 1, 1, 4)
    atoms = torch.tensor([1, 3, 5, 7]).reshape_as(c)
    tokens, (ids, p) = jittered_scalar_targets(atoms, c, bins, num_atoms=16, stochastic=False)
    nearest, expected = jitter_probabilities(c, bins)
    assert torch.equal(tokens[..., 0::2], atoms)
    assert torch.equal(tokens[..., 1::2], nearest+16)
    assert torch.equal(ids, atoms)
    torch.testing.assert_close(p, expected)
    torch.testing.assert_close(p.sum(-1), torch.ones_like(c))
    assert (p > 0).sum(-1).max() <= 2
    assert p[0, 0, 0, 0, 1024] == 1
    assert 0 < p[0, 0, 0, 1, 1025] < .5
    assert p[0, 0, 0, 3, -1] == 1
    full_tokens, full = jittered_scalar_targets(atoms, c, bins, num_atoms=16, stochastic=False, compact=False)
    torch.testing.assert_close(tokens, full_tokens)
    torch.testing.assert_close(full[..., 1::2, 16:], p)
    assert torch.equal(full[..., 0::2, :16].argmax(-1), atoms)


def test_noise_cap_and_quantization_error_on_independent_continuous_draws():
    generator = torch.Generator().manual_seed(261010)
    bins = torch.linspace(-3, 3, 2048, dtype=torch.float64)
    width = 6 / 2047
    coefficients = torch.rand(20000, generator=generator, dtype=torch.float64)*5.9-2.95
    z = torch.randn(20000, generator=generator, dtype=torch.float64)
    rejected = z.abs() > 2.5
    while rejected.any():
        z[rejected] = torch.randn(int(rejected.sum()), generator=generator, dtype=torch.float64)
        rejected = z.abs() > 2.5
    jitter = z * .01 * width
    assert (jitter.abs()/width).max() <= .025
    observed = float(jitter.square().mean().sqrt()/width)
    assert abs(observed-jitter_moments()['actual_rms_bins']) < .0002
    quantized = bins[torch.searchsorted((bins[1:]+bins[:-1])/2, coefficients+jitter)]
    assert ((quantized-coefficients).abs()/width).max() <= .525
    # The integrated categorical distribution agrees with the independent draws.
    c = torch.full((1, 1, 1, 20000), float((bins[1024]+bins[1025])/2-.004*width), dtype=torch.float64)
    _, p = jitter_probabilities(c[..., :1], bins)
    empirical_upper = ((c.reshape(-1)+jitter) > (bins[1024]+bins[1025])/2).double().mean()
    assert abs(float(empirical_upper)-float(p[..., 1025])) < .015


def test_diagnostics_use_clean_coefficients_and_include_overflow():
    bins = torch.tensor([-2., -1., 0., 1., 2.])
    clean = torch.tensor([.25, 3.]).reshape(1, 1, 1, 2)
    logits = torch.full((*clean.shape, 5), -100.)
    logits[..., 0, 2] = 0
    logits[..., 1, 4] = 0
    target = logits.softmax(-1)
    sums = coefficient_diagnostic_sums(logits, clean, bins, target)
    metrics = coefficient_diagnostic_metrics(sums, bins, [2., 4.])
    assert metrics['train/coeff_mode_mae_bins'] == .625
    assert metrics['train/coeff_within_one_bin_fraction'] == .5
    assert metrics['train/coeff_out_of_range_fraction'] == .5
    assert metrics['train/coeff_mode_mae_bins_depth0'] == .25
    assert metrics['train/coeff_mode_physical_mae_depth1'] == 4.
    assert not sums.requires_grad
    # Accumulating identical microbatches/ranks must not alter the averages.
    assert coefficient_diagnostic_metrics(sums*8, bins, [2., 4.]) == metrics


@pytest.mark.parametrize('sigma,cap', [(0., .1), (.02, .01), (.1, .5), (1., 2.)])
def test_rejects_noise_not_strictly_sub_bin(sigma, cap):
    with pytest.raises(ValueError):
        jitter_moments(sigma, cap)
