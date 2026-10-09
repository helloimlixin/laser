from types import SimpleNamespace

import torch

from src.training.rqtransformer import LaserAux, compound_objective


def test_requested_200_bin_gaussian_on_ffhq_uniform_centers():
    bins = torch.linspace(-3, 3, 2048)
    width = 6 / 2047
    temperature = 2 * (200 * width)**2
    aux = SimpleNamespace(coeff_bins=bins, coeff_scales=torch.tensor([8.2, 4.3, 3.1, 1.8]),
        sparsity_level=4, coeff_vocab_size=2048, num_atoms=16, vocab_size=2064,
        soft_target_physical=False)
    coefficients = torch.tensor([0., 0., 0., 0., 3., 3., 3., 3.]).reshape(2, 1, 1, 4)
    atoms = torch.arange(4).expand_as(coefficients)
    tokens, (_, p) = LaserAux.sparse_targets(aux, atoms, coefficients,
        stochastic=False, compact=True, temp=temperature)
    mean = (p.double() * bins.double()).sum(-1)
    std = (p.double() * (bins.double() - mean[..., None]).square()).sum(-1).sqrt()
    # Center widths match the requested ratio. Finite-range edge distributions
    # are deliberately truncated, as in the archived FFHQ Gaussian targets.
    torch.testing.assert_close(std[0]/width, torch.full_like(std[0], 200.), atol=.002, rtol=0)
    assert ((std[1]/width > 100) & (std[1]/width < 200)).all()
    torch.testing.assert_close(p.sum(-1), torch.ones_like(coefficients))
    assert torch.isfinite(p).all() and torch.equal(tokens[..., 0::2], atoms)
    # Physical depth scales multiply noise and spacing together.
    torch.testing.assert_close(std[0] * aux.coeff_scales / (width * aux.coeff_scales),
                               torch.full_like(std[0], 200.), atol=.002, rtol=0)


def test_physical_targets_have_depth_independent_quarter_unit_std():
    scales = torch.tensor([8.21, 4.26, 3.07, 1.83])
    aux = SimpleNamespace(coeff_bins=torch.linspace(-3, 3, 2048),
        coeff_scales=scales, sparsity_level=4, coeff_vocab_size=2048,
        soft_target_physical=True)
    coefficients = torch.ones(1, 1, 1, 4) / scales
    nearest, probabilities = LaserAux.compound_coeff_ids(
        aux, coefficients, stochastic=False, temp=.125)
    physical_bins = aux.coeff_bins * scales[:, None]
    mean = (probabilities * physical_bins).sum(-1)
    variance = (probabilities * (physical_bins - mean[..., None]).square()).sum(-1)
    torch.testing.assert_close(mean, torch.ones_like(mean), atol=1e-6, rtol=0)
    torch.testing.assert_close(variance.sqrt(), torch.full_like(mean, .25), atol=1e-6, rtol=0)
    aux.soft_target_physical = False
    old_nearest, _ = LaserAux.compound_coeff_ids(aux, coefficients, stochastic=False, temp=.5)
    assert torch.equal(nearest, old_nearest)


def test_geometry_disabled_classification_pushes_spurious_endpoint_down():
    atom_logits = torch.zeros(1, 1, 1, 4, 3, requires_grad=True)
    coeff_logits = torch.tensor([1., 0., 1.]).expand(1, 1, 1, 4, 3).clone().requires_grad_()
    atoms = torch.zeros(1, 1, 1, 4, dtype=torch.long)
    targets = torch.tensor([0., 1., 0.]).expand_as(coeff_logits)
    loss, metrics = compound_objective(atom_logits, coeff_logits, None, atoms, targets,
        None, atom_weight=1.5, geometry_weight=0., accumulation=3,
        distribution_geometry=False)
    gradient = torch.autograd.grad(loss, coeff_logits)[0]
    assert (gradient[..., [0, -1]] > 0).all()
    assert (gradient[..., 1] < 0).all()
    assert metrics['geometry'] == 0
