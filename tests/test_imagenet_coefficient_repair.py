from types import SimpleNamespace

import torch

from src.training.rqtransformer import LaserAux, compound_objective


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
