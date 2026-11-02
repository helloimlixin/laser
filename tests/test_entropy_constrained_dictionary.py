import math

import pytest
import torch

from src.models.entropy_constrained_dictionary import (
    EntropyConstrainedBook, coding_statistics, fit_probabilities,
)


def test_exact_assignment_balances_distortion_against_train_fitted_rate():
    book = EntropyConstrainedBook([[[0.], [1.]]], [[.99, .01]])
    x = torch.tensor([[1.]])
    assert book.quantize(x)['codes'].item() == 1
    result = book.quantize(x, rate_weight=1.)
    assert result['codes'].item() == 0
    torch.testing.assert_close(result['quantized'], book.embed(result['codes']).sum(-2), rtol=0, atol=0)


def test_coordinate_search_recovers_from_a_greedy_residual_trap_without_more_tokens():
    book = EntropyConstrainedBook([[[0.], [3.], [2.]], [[0.], [2.2], [2.2]]])
    x = torch.tensor([[4.2]])
    greedy = book.quantize(x)
    refined = book.quantize(x, sweeps=2)
    assert abs(greedy['quantized'].item() - 5.2) < 1.e-6
    torch.testing.assert_close(refined['quantized'], x)
    assert refined['codes'].shape == greedy['codes'].shape


def test_coordinate_updates_do_not_increase_final_rate_distortion_objective():
    torch.manual_seed(4)
    vectors = torch.randn(3, 13, 5); vectors[:, 0] = 0
    book = EntropyConstrainedBook(vectors, torch.rand(3, 13).clamp_min(.01))
    x = torch.randn(80, 5)
    for beta in [0., .05, 1.]:
        old = book.quantize(x, rate_weight=beta)['codes']
        new = book.quantize(x, rate_weight=beta, sweeps=3, initial_codes=old)['codes']
        assert book.objective(x, new, rate_weight=beta) <= book.objective(x, old, rate_weight=beta) + 1.e-7


def test_heldout_rate_charges_unseen_symbols_and_separates_depth_from_marginal():
    train = [[10, 0], [0, 10]]
    same = coding_statistics(train, train)
    shifted = coding_statistics(train, [[0, 10], [10, 0]])
    assert same['depth_conditional_entropy_bits_per_token'] == 0
    assert same['marginal_entropy_bits_per_token'] == 1
    assert same['depth_conditional_entropy_bpp'] == 0
    assert same['marginal_entropy_bpp'] == 256 / 65536
    assert same['effective_vocabulary'] == 2
    assert same['heldout_cross_entropy_bits_per_token'] > 0
    assert shifted['heldout_cross_entropy_bits_per_token'] > 4
    assert math.isfinite(shifted['heldout_model_bpp'])
    assert not shifted['actual_bitstream_measured']
    assert not shifted['evaluation_used_for_entropy_model']


@pytest.mark.parametrize('counts', [[[0, 0]], [[-1, 3]], [[float('nan'), 1]]])
def test_invalid_counts_are_rejected(counts):
    with pytest.raises(ValueError):
        fit_probabilities(counts)
