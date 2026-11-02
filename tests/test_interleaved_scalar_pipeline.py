from types import SimpleNamespace

import torch
import torch.nn.functional as F

from src.training.rqtransformer import LaserAux, build_model
from src.training.scalar_pair_loss import scalar_pair_objective, scalar_pair_metrics
from src.training.physical_pair_crps import physical_pair_objective_components
from src.training.compound_balance import shared_hidden_gradient_metrics
from tests.test_physical_pair_scalar_prior import case


def test_token_boundaries_use_sum_vocabulary_and_eight_scalar_slots():
    aux = SimpleNamespace(num_atoms=16384, coeff_vocab_size=2048, vocab_size=18432,
                          sparsity_level=4, soft_target_physical=False,
                          coeff_bins=torch.linspace(-3, 3, 2048))
    atoms = torch.tensor([[[[0, 1, 16382, 16383]]]])
    values = torch.tensor([[[[-3., -1., 1., 3.]]]])
    tokens, (actual_atoms, probabilities) = LaserAux.sparse_targets(
        aux, atoms, values, stochastic=False, compact=True, hard=True)
    assert tokens.shape == (1, 1, 1, 8)
    assert torch.equal(tokens[..., 0::2], atoms)
    assert torch.equal(tokens[..., 1::2] - 16384, probabilities.argmax(-1))
    assert tokens.min() == 0 and tokens.max() == 18431
    assert torch.equal(actual_atoms, atoms)
    with torch.device('meta'):
        model = build_model(18432, 16384, coeff_vocab_size=2048, sparsity_level=4,
                            physical_pair_context=True, compound=False)
    assert list(model.block_size) == [8, 8, 8]
    assert model.classifier.linear.weight.shape == (18432, 1536)
    assert list(model.config.vocab_size) == [18432] * 8


def inputs():
    torch.manual_seed(17)
    atoms = torch.tensor([[[0, 1, 2, 3]], [[3, 2, 1, 0]]])
    a = torch.randn(2, 1, 4, 7, requires_grad=True)
    c = torch.randn(2, 1, 4, 5, requires_grad=True)
    q = F.one_hot(torch.tensor([[[0, 1, 2, 3]], [[4, 3, 2, 1]]]), 5).float()
    return a, c, atoms, q, torch.linspace(-3, 3, 5)


def test_scalar_default_and_crps_match_previous_likelihood_and_gradients():
    for crps in [0., .2]:
        a, c, ids, q, bins = inputs()
        actual, details = scalar_pair_objective(a, c, ids, q, bins, crps_weight=crps, accumulation=3)
        expected, ce, ordered = physical_pair_objective_components(a, c, ids, q, bins, crps, 3)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(details['classification'], ce)
        torch.testing.assert_close(details['coefficient_crps'], ordered if crps else ordered*0)
        got = torch.autograd.grad(actual, (a, c), retain_graph=True)
        want = torch.autograd.grad(expected, (a, c))
        for x, y in zip(got, want):torch.testing.assert_close(x, y)


def test_scalar_field_and_depth_weights_change_the_correct_gradients():
    a, c, ids, q, bins = inputs()
    loss, details = scalar_pair_objective(a, c, ids, q, bins, atom_weight=.5,
        atom_depth_weights=[1, 2, 3, 4], coefficient_depth_weights=[4, 3, 2, 1])
    plain, _ = scalar_pair_objective(a, c, ids, q, bins)
    actual = torch.autograd.grad(loss, (a, c), retain_graph=True)
    original = torch.autograd.grad(plain, (a, c))
    wa = torch.tensor([.4, .8, 1.2, 1.6]).view(1, 1, 4, 1)
    wc = wa.flip(-2)
    torch.testing.assert_close(actual[0], original[0] * wa * (2*.5/1.5))
    torch.testing.assert_close(actual[1], original[1] * wc * (2/1.5))
    metrics = scalar_pair_metrics(a, c, ids, q, bins, torch.ones(4), details, atom_weight=.5)
    assert 'train/atom_nll_depth3' in metrics and 'train/coeff_cross_entropy_depth3' in metrics
    assert metrics['train/coeff_target_entropy'] == 0.


def test_scalar_depth_input_probe_keeps_final_parameter_gradients_unchanged():
    model, aux, tokens, cond = case()
    atoms = tokens[..., 0::2]
    q = F.one_hot(tokens[..., 1::2] - aux.num_atoms, aux.coeff_vocab_size).float()
    baseline = model(tokens, aux, cond)
    loss, _ = scalar_pair_objective(baseline['atom_logits'], baseline['coeff_logits'],
                                    atoms, q, aux.coeff_bins)
    parameters = list(model.parameters())
    expected = torch.autograd.grad(loss, parameters)
    model.capture_shared_hidden = True
    output = model(tokens, aux, cond)
    loss, details = scalar_pair_objective(output['atom_logits'], output['coeff_logits'],
                                         atoms, q, aux.coeff_bins)
    assert output['shared_hidden'].shape == (1, 2, 2, 4, 24)
    metrics = shared_hidden_gradient_metrics(details['depth_weighted_atom_cross_entropy'],
        details['depth_weighted_coeff_cross_entropy'], output['shared_hidden'])
    assert metrics['train/shared_atom_gradient_norm_rank0'] > 0
    assert metrics['train/shared_coefficient_gradient_norm_rank0'] > 0
    for d in range(4):
        assert metrics[f'train/shared_atom_gradient_norm_depth{d}_rank0'] > 0
        assert metrics[f'train/shared_coefficient_gradient_norm_depth{d}_rank0'] > 0
    assert all(p.grad is None for p in parameters)
    actual = torch.autograd.grad(loss, parameters)
    for x, y in zip(actual, expected):torch.testing.assert_close(x, y)
