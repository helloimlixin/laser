import math

import pytest
import torch

from src.training.physical_pair_cca import (
    condition_contrastive_alignment, physical_pair_sequence_log_probability,
)


def test_sequence_score_uses_sampled_coefficients_and_sums_every_event():
    atom = torch.tensor([[[[2., -1., 0.], [0., 1., 3.]]]], requires_grad=True)
    coefficient = torch.tensor([[[[0., 2.], [3., -1.]]]], requires_grad=True)
    atoms = torch.tensor([[[0, 2]]])
    ids = torch.tensor([[[1, 0]]])
    score = physical_pair_sequence_log_probability(atom, coefficient, atoms, ids)
    expected = (atom.log_softmax(-1)[0, 0, 0, 0] + atom.log_softmax(-1)[0, 0, 1, 2]
                + coefficient.log_softmax(-1)[0, 0, 0, 1]
                + coefficient.log_softmax(-1)[0, 0, 1, 0])
    torch.testing.assert_close(score, expected.reshape(1))
    changed = physical_pair_sequence_log_probability(atom, coefficient, atoms, 1 - ids)
    assert not torch.equal(score, changed)
    (-score.mean()).backward()
    assert atom.grad[0, 0, 0, 0] < 0 and coefficient.grad[0, 0, 0, 1] < 0


def test_masked_atoms_have_finite_scores_and_gradients():
    logits = torch.tensor([[[0., -torch.inf, 2.]]], requires_grad=True)
    coeff = torch.zeros(1, 1, 2, requires_grad=True)
    score = physical_pair_sequence_log_probability(logits, coeff,
        torch.tensor([[2]]), torch.tensor([[1]]))
    score.sum().backward()
    assert torch.isfinite(score).all() and torch.isfinite(logits.grad).all()
    assert logits.grad[..., 1].item() == 0


def test_alignment_increases_matched_and_decreases_mismatched_likelihood():
    pos = torch.tensor([-100., -100.], requires_grad=True)
    neg = pos.detach().clone().requires_grad_()
    ref_pos = pos.detach().clone().requires_grad_()
    ref_neg = neg.detach().clone().requires_grad_()
    loss, _ = condition_contrastive_alignment(pos, neg, ref_pos, ref_neg,
        torch.tensor([0, 1]), torch.tensor([1, 0]))
    assert loss.item() == pytest.approx(2 * math.log(2))
    loss.backward()
    assert (pos.grad < 0).all() and (neg.grad > 0).all()
    assert ref_pos.grad is None and ref_neg.grad is None
    with torch.no_grad():
        pos -= pos.grad; neg -= neg.grad
    improved, _ = condition_contrastive_alignment(pos, neg, ref_pos, ref_neg,
        torch.tensor([0, 1]), torch.tensor([1, 0]))
    assert improved < loss


def test_class_collisions_do_not_penalize_valid_conditions():
    pos = torch.zeros(3, requires_grad=True)
    neg = torch.zeros(3, requires_grad=True)
    loss, diagnostics = condition_contrastive_alignment(pos, neg,
        torch.zeros(3), torch.zeros(3), torch.tensor([0, 1, 2]), torch.tensor([1, 1, 2]),
        negative_weight=10.)
    loss.backward()
    assert neg.grad[0] > 0 and torch.equal(neg.grad[1:], torch.zeros(2))
    assert diagnostics['negative_pair_fraction'].item() == pytest.approx(1 / 3)
    # No division by the number of valid negatives: preserve image-batch weighting.
    assert neg.grad[0].item() == pytest.approx(.02 * .5 / 3)


def test_extreme_likelihood_ratios_remain_finite():
    pos = torch.tensor([-1e6, 1e6], requires_grad=True)
    neg = torch.tensor([1e6, -1e6], requires_grad=True)
    loss, _ = condition_contrastive_alignment(pos, neg, torch.zeros(2), torch.zeros(2),
        torch.tensor([0, 1]), torch.tensor([1, 0]), negative_weight=1000.)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pos.grad).all() and torch.isfinite(neg.grad).all()


def test_zero_negative_weight_preserves_positive_anchor_behavior():
    pos = torch.tensor([2.], requires_grad=True)
    neg = torch.tensor([3.], requires_grad=True)
    loss, _ = condition_contrastive_alignment(pos, neg, torch.zeros(1), torch.zeros(1),
        torch.tensor([0]), torch.tensor([1]), negative_weight=0.)
    loss.backward()
    assert pos.grad.item() < 0 and neg.grad.item() == 0


@pytest.mark.parametrize('kwargs', [{'beta':0}, {'beta':float('nan')},
    {'negative_weight':-1}, {'negative_weight':float('inf')}])
def test_invalid_hyperparameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        condition_contrastive_alignment(*[torch.zeros(1)] * 4,
            torch.zeros(1, dtype=torch.long), torch.ones(1, dtype=torch.long), **kwargs)
