import pytest
import torch
import torch.nn.functional as F

from src.training.physical_pair_crps import (
    crps_weight_at_step, physical_pair_objective_components,
    validate_coefficient_bins,
)


def inputs():
    generator = torch.Generator().manual_seed(42)
    a = torch.randn(2, 3, 4, 7, generator=generator, requires_grad=True)
    c = torch.randn(2, 3, 4, 5, generator=generator, requires_grad=True)
    ids = torch.randint(7, (2, 3, 4), generator=generator)
    q = torch.randn(2, 3, 4, 5, generator=generator).softmax(-1)
    return a, c, ids, q, torch.tensor([-3., -1., 0., .5, 3.])


def test_zero_weight_preserves_existing_ce_and_gradients():
    a, c, ids, q, bins = inputs()
    expected = (-a.log_softmax(-1).gather(-1, ids.unsqueeze(-1)).squeeze(-1).sum(-1)
                - (q * c.log_softmax(-1)).sum(-1).sum(-1)).mean() / (2 * 4 * 3)
    actual, _, _ = physical_pair_objective_components(a, c, ids, q, bins, torch.tensor(0.), 3)
    assert torch.equal(actual, expected)
    actual_grad = torch.autograd.grad(actual, (a, c), retain_graph=True)
    expected_grad = torch.autograd.grad(expected, (a, c))
    for x, y in zip(actual_grad, expected_grad):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_calibrated_soft_distribution_minimizes_crps():
    a, c, ids, q, bins = inputs()
    c = q.log().detach().requires_grad_(True)
    _, _, crps = physical_pair_objective_components(a, c, ids, q, bins, torch.tensor(.05), 1)
    assert crps.item() < 1e-14
    assert torch.autograd.grad(crps, c)[0].abs().max() < 1e-8


def test_ordered_bins_penalize_farther_misses():
    bins = torch.arange(5, dtype=torch.float32)
    target = F.one_hot(torch.tensor([[[1]]]), 5).float()
    atom = torch.zeros(1, 1, 1, 2)
    ids = torch.zeros(1, 1, 1, dtype=torch.long)
    scores = []
    for guess in (1, 2, 4):
        logits = torch.full((1, 1, 1, 5), -100.)
        logits[..., guess] = 0.
        scores.append(physical_pair_objective_components(atom, logits, ids, target, bins, torch.tensor(.05), 1)[2].item())
    assert scores == pytest.approx([0., .25, .75])


def test_auxiliary_contributes_gradients_and_accumulation_scales():
    a, c, ids, q, bins = inputs()
    total, ce, crps = physical_pair_objective_components(a, c, ids, q, bins, torch.tensor(.05), 3)
    torch.testing.assert_close(total, (ce + .05 * crps) / 3)
    grad = torch.autograd.grad(total - ce / 3, c)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    plain = physical_pair_objective_components(a, c, ids, q, bins, torch.tensor(.05), 1)[0]
    torch.testing.assert_close(total * 3, plain)
    # Range normalization is invariant to coefficient units and offsets.
    changed = physical_pair_objective_components(a, c, ids, q, bins * 8.2 + 17., torch.tensor(.05), 1)[2]
    torch.testing.assert_close(crps, changed)


def test_ramp_uses_saved_cursor_on_resume_and_rewind():
    assert crps_weight_at_step(.05, 40064, 40064, 626) == 0
    assert crps_weight_at_step(.05, 40084, 40064, 626) == pytest.approx(.05 * 20 / 626)
    assert crps_weight_at_step(.05, 40377, 40064, 626) == .025
    assert crps_weight_at_step(.05, 40690, 40064, 626) == .05
    assert crps_weight_at_step(.05, 41316, 40064, 626) == .05
    # Rewinding the full best epoch64 resets the objective clock with the model.
    assert crps_weight_at_step(.05, 40064, 40064, 626) == 0
    assert crps_weight_at_step(.05, 3, 4, 0) == 0
    assert crps_weight_at_step(.05, 4, 4, 0) == .05
    for weight in (-1., float('inf'), float('nan')):
        with pytest.raises(ValueError):
            crps_weight_at_step(weight, 1, 0, 1)


def test_bins_validation():
    validate_coefficient_bins(torch.tensor([-3., 0., 3.]), 3)
    for bins in (torch.tensor([0., 0.]), torch.tensor([1., 0.]), torch.tensor([0., float('nan')])):
        with pytest.raises(ValueError):
            validate_coefficient_bins(bins, 2)
    with pytest.raises(ValueError):
        validate_coefficient_bins(torch.tensor([0., 1.]), 3)
