import pytest
import torch

from src.stochastic_compound import sample_compound_bank, stochastic_omp


def test_zero_temperature_recovers_known_orthogonal_sparse_signal():
    dictionary = torch.eye(5, dtype=torch.float64)
    x = torch.tensor([[3., -4., .2, 0., 1.]], dtype=torch.float64)
    result = stochastic_omp(x, dictionary, depth=3)
    assert result['atoms'].tolist() == [[1, 0, 4]]
    torch.testing.assert_close(result['coefficients'], torch.tensor([[-4., 3., 1.]], dtype=torch.float64))
    torch.testing.assert_close(result['quantized'], torch.tensor([[3., -4., 0., 0., 1.]], dtype=torch.float64))


def test_sampled_supports_have_matching_least_squares_coefficients():
    g = torch.Generator().manual_seed(91)
    dictionary = torch.nn.functional.normalize(torch.randn(12, 24, generator=g, dtype=torch.float64), dim=0)
    signals = torch.randn(80, 12, generator=g, dtype=torch.float64)
    result = stochastic_omp(signals, dictionary, depth=4, temperature=.6, generator=g)
    active = dictionary.T[result['atoms']].transpose(-1, -2)
    expected = torch.linalg.lstsq(active, signals.unsqueeze(-1)).solution.squeeze(-1)
    torch.testing.assert_close(result['coefficients'], expected, atol=1e-10, rtol=1e-10)
    assert (result['atoms'].sort(-1).values.diff(dim=-1) != 0).all()
    residual = signals - result['quantized']
    torch.testing.assert_close(active.transpose(-1,-2) @ residual[...,None], torch.zeros(80,4,1,dtype=torch.float64), atol=1e-10, rtol=0)


def test_atom_sampling_distribution_matches_energy_and_changes_choices():
    x = torch.tensor([[1., .9, .1]]).expand(12000, -1)
    result = stochastic_omp(x, torch.eye(3), depth=1, temperature=.7,
                            generator=torch.Generator().manual_seed(12))
    measured = torch.bincount(result['atoms'].flatten(), minlength=3).float() / len(x)
    expected = (x[0].square() / .7).softmax(-1)
    torch.testing.assert_close(measured, expected, atol=.015, rtol=0)


def test_bank_selection_preserves_pairs_varies_and_replays_rng():
    atoms = torch.arange(2*3*4*8*4).reshape(2,3,4,8,4)
    coefficients = atoms.float() * .125 - 5
    g = torch.Generator().manual_seed(18)
    state = g.get_state()
    a,c,choices = sample_compound_bank(atoms, coefficients, generator=g)
    a2,c2,_ = sample_compound_bank(atoms, coefficients, generator=g)
    assert not torch.equal(a, a2)
    torch.testing.assert_close(c, a.float()*.125-5)
    torch.testing.assert_close(c2, a2.float()*.125-5)
    g.set_state(state)
    replay = sample_compound_bank(atoms, coefficients, generator=g)
    for original, repeated in zip((a,c,choices), replay):
        assert torch.equal(original, repeated)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError, match='temperature'):
        stochastic_omp(torch.ones(1,3),torch.eye(3),temperature=-1)
    with pytest.raises(ValueError, match='equal'):
        sample_compound_bank(torch.zeros(2,3,4,4),torch.zeros(2,3,4,4))
