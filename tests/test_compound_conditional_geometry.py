from types import SimpleNamespace

import pytest
import torch

from src.training.compound_geometry import (
    joint_candidate_contribution, conditional_geometry_prediction,
)
from src.training.rqtransformer import CompoundLaserRQTransformer, compound_objective
from tests.test_compound_pair_autoregressive import tiny_config, tiny_aux


def test_joint_expectation_and_gradients_match_probability_enumeration():
    vectors = torch.tensor([[1., .2], [-.3, -.7], [.2, .8]], dtype=torch.float64)
    bins = torch.tensor([-2., 0., 2.], dtype=torch.float64)
    a = torch.tensor([.2, -.1, .8], dtype=torch.float64, requires_grad=True)
    c = torch.tensor([[1., 0., -1.], [-.8, .2, 1.], [.3, -.4, .2]],
                     dtype=torch.float64, requires_grad=True)
    weights, probabilities = a.softmax(-1), c.softmax(-1)
    actual = joint_candidate_contribution(weights, vectors, (probabilities * bins).sum(-1))
    expected = sum(weights[i] * probabilities[i, j] * vectors[i] * bins[j]
                   for i in range(3) for j in range(3))
    torch.testing.assert_close(actual, expected)
    left = torch.autograd.grad(actual.square().sum(), (a, c), retain_graph=True)
    right = torch.autograd.grad(expected.square().sum(), (a, c))
    for x, y in zip(left, right):
        torch.testing.assert_close(x, y)


@pytest.mark.parametrize('top_k', [1, 2])
def test_signed_atom_candidates_condition_coefficients_and_include_teacher_once(top_k):
    aux = SimpleNamespace(dictionary=torch.tensor([[1., -1.]]),
                          coeff_bins=torch.tensor([-1., 1.]), coeff_scales=torch.tensor([2.]))
    table = torch.tensor([[-1., 1.], [1., -1.]], requires_grad=True)
    model = SimpleNamespace(coefficient_logits=lambda h, v, depth_index: table[(v[..., 0] < 0).long()])
    atoms = torch.zeros(1, 1, 1, 1, dtype=torch.long)
    # The teacher is excluded from the top-1 candidates but must still be included.
    atom_logits = torch.tensor([0., 1.]).expand(1, 1, 1, 1, 2).requires_grad_()
    coeff_logits = table[0].expand(1, 1, 1, 1, 2)
    actual = conditional_geometry_prediction(model, aux, torch.zeros(1, 1, 1, 1, 3),
                                            atom_logits, coeff_logits, atoms, top_k)
    expected = ((table.softmax(-1) * aux.coeff_bins).sum(-1)
                * aux.dictionary[0] * atom_logits[0, 0, 0, 0].softmax(-1)).sum() * 2
    torch.testing.assert_close(actual.flatten()[0], expected)
    gradient = torch.autograd.grad(actual.sum(), table)[0]
    assert (gradient.abs().sum(-1) > 0).all()


def test_conditional_forward_preserves_checkpoint_logits_and_causal_sampling():
    torch.manual_seed(21)
    model = CompoundLaserRQTransformer(tiny_config(), 7, 5, micro_transformer_layers=2,
        depth_specific_coeff_heads=True, pair_autoregressive=True, mask_seen_atoms_training=False).eval()
    aux = tiny_aux()
    packed = torch.tensor([[[[6, 13]]], [[[17, 22]]]])
    keys = set(model.state_dict())
    model.load_state_dict(model.state_dict(), strict=True)
    baseline = model(packed, model_aux=aux)
    corrected = model(packed, model_aux=aux, conditional_geometry_top_k=4)
    assert keys == set(model.state_dict())
    for key in baseline:
        torch.testing.assert_close(baseline[key], corrected[key], rtol=0, atol=0)
    assert model._conditional_geometry_top_k == 0
    loss, metrics = compound_objective(corrected['atom_logits'], corrected['coeff_logits'], None,
        packed // 5, torch.nn.functional.one_hot(packed % 5, 5).float(),
        aux.compound_embeddings(packed // 5, packed % 5), atom_weight=1.5,
        geometry_weight=.05, accumulation=1, distribution_geometry=True,
        geometry_prediction=corrected['geometry_prediction'])
    assert torch.isfinite(loss) and metrics['geometry'] > 0
    loss.backward()
    for module in (model.body_transformer, model.classifier, model.coeff_micro_transformer, model.coeff_classifier):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        assert sum(float(p.grad.abs().sum()) for p in module.parameters()) > 0
    torch.manual_seed(33)
    first = model.sample_compound(2, aux, atom_top_k=7, amp=False)
    model(packed, model_aux=aux, conditional_geometry_top_k=4)
    torch.manual_seed(33)
    second = model.sample_compound(2, aux, atom_top_k=7, amp=False)
    assert all(torch.equal(x, y) for x, y in zip(first, second))
