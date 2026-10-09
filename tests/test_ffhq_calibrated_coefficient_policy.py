import pytest
import torch

from src.training.rqtransformer import LaserAux


def identity_aux():
    aux = LaserAux.__new__(LaserAux)
    torch.nn.Module.__init__(aux)
    aux.encoder = torch.nn.Identity()
    aux.quant_conv = torch.nn.Identity()
    aux.register_buffer('dictionary', torch.eye(4))
    aux.register_buffer('coeff_scales', torch.tensor([2., 5., 10., 1.]))
    aux.register_buffer('coeff_bins', torch.linspace(-3, 3, 2048))
    aux.sparsity_level, aux.coeff_vocab_size = 4, 2048
    aux.coeff_max, aux.clamp_coeffs = 3., True
    return aux


def test_normalization_precedes_clamping_and_geometry_uses_clamped_coefficients():
    aux = identity_aux()
    latent = torch.tensor([15., -10., 5., -20.]).view(1, 4, 1, 1)
    atoms, coefficients = aux.encode_sparse_components(latent)
    assert atoms.flatten().tolist() == [3, 0, 1, 2]
    torch.testing.assert_close(coefficients.flatten(), torch.tensor([-3., 3., -1., 3.]))
    contributions = aux.physical_contributions(atoms, coefficients).sum(-2)
    torch.testing.assert_close(contributions.flatten(), torch.tensor([15., -10., 3., -6.]))


def test_ffhq_temperature_matches_half_unit_gaussian_on_uniform_bins():
    aux = identity_aux()
    _, probabilities = aux.compound_coeff_ids(torch.zeros(1, 1, 1, 4), temp=.5, stochastic=False)
    p = probabilities[0, 0, 0, 0].double()
    bins = aux.coeff_bins.double()
    mean = (p*bins).sum()
    sigma = (p*(bins-mean).square()).sum().sqrt().item()
    assert sigma == pytest.approx(.5, rel=1e-4)
    assert sigma/(6/2047) == pytest.approx(170.5833333333, rel=1e-4)
