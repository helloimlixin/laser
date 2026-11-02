import pytest
import torch

from src.models.laser_joint_tokenizer import JointAtomCoefficientRQ
from src.training.laser_shared_joint_vocabulary import assert_shared_codewords, refit_shared_book


def book():
    torch.manual_seed(817)
    dictionary = torch.nn.functional.normalize(torch.randn(7, 3), dim=0)
    atoms = torch.cat((torch.zeros(1, dtype=torch.long), torch.arange(3).repeat_interleave(4))).expand(5, -1)
    values = torch.cat((torch.zeros(1), torch.tensor([-2., -.7, .7, 2.]).repeat(3))).expand(5, -1)
    return JointAtomCoefficientRQ(dictionary, atoms, values)


def test_repeated_ids_fit_monotonically_with_exactly_shared_physical_vectors():
    q = book()
    codes = torch.randint(1, 13, (100, 5))
    codes[:30] = codes[:30, :1].expand(-1, 5)
    codes[30:35] = 0
    truth = q.dictionary + torch.randn_like(q.dictionary) * .05
    signals = sum(truth.T[q.atom_ids[0, codes[:, d]]] *
                  q.coefficients[0, codes[:, d], None] * 1.1 for d in range(5))
    result = refit_shared_book(q, signals, codes)
    assert result['fixed_assignment_mse_after'] < result['fixed_assignment_mse_before']
    assert result['fixed_assignment_mse_after'] <= result['fixed_assignment_mse_after_dictionary'] + 1.e-6
    assert_shared_codewords(q)
    for depth in range(5):
        assert torch.equal(q.expanded_codebook(depth), q.expanded_codebook(0))
    levels = q.coefficients[0, 1:].view(3, 4)
    assert (levels[:, :2] < 0).all() and (levels[:, 2:] > 0).all()
    assert (levels[:, 1:] >= levels[:, :-1]).all()
    assert (q.coefficients[:, 0] == 0).all()


def test_zero_signals_and_unused_dictionary_entries_are_preserved():
    q = book()
    before = {key:value.clone() for key,value in q.state_dict().items()}
    refit_shared_book(q, torch.zeros(20, 7), torch.zeros(20, 5, dtype=torch.long))
    for key,value in q.state_dict().items():
        torch.testing.assert_close(value, before[key])
    assert_shared_codewords(q)


def test_depth_dependent_values_cannot_silently_enter_shared_fitting():
    q = book()
    q.coefficients[1, 1] *= 2
    with pytest.raises(ValueError, match='same physical vector'):
        refit_shared_book(q, torch.zeros(20, 7), torch.zeros(20, 5, dtype=torch.long))
