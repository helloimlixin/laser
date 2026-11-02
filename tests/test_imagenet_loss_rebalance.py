import torch
from torch.nn import functional as F

from src.training.imagenet_ffhq_adapter import compound_objective


def test_balanced_ce_gradients_and_disabled_geometry_are_independent_of_bad_geometry_inputs():
    torch.manual_seed(261011)
    atoms = torch.tensor([[[[0, 1, 2, 3]]], [[[4, 5, 6, 0]]]])
    atom_logits = torch.randn(*atoms.shape, 7, requires_grad=True)
    coeff_logits = torch.randn(*atoms.shape, 11, requires_grad=True)
    targets = torch.randn_like(coeff_logits).softmax(-1)
    dictionary = torch.full((3, 7), float('nan'), requires_grad=True)
    physical = torch.full((*atoms.shape, 3), float('nan'), requires_grad=True)
    loss, values = compound_objective(atom_logits, coeff_logits, None, atoms, targets,
        physical, atom_weight=.575, geometry_weight=0., accumulation=4,
        distribution_geometry=False, geometry_dictionary=dictionary,
        geometry_coeff_bins=torch.linspace(-3, 3, 11), geometry_coeff_scales=torch.ones(4),
        coeff_regression_weight=0., coeff_crps_weight=0., causal_prefix_weight=0.)
    assert torch.isfinite(loss)
    loss.backward()
    count = atoms.numel()
    expected_atom_gradient = (atom_logits.detach().softmax(-1)-F.one_hot(atoms, 7))*.575/(1.575*count*4)
    expected_coefficient_gradient = (coeff_logits.detach().softmax(-1)-targets)/(1.575*count*4)
    torch.testing.assert_close(atom_logits.grad, expected_atom_gradient)
    torch.testing.assert_close(coeff_logits.grad, expected_coefficient_gradient)
    assert atom_logits.grad.norm() > 0 and coeff_logits.grad.norm() > 0
    assert physical.grad is None and dictionary.grad is None
    for key in ('geometry', 'geometry_pair_mse', 'geometry_spatial_mse',
                'coefficient_regression', 'coefficient_crps', 'causal_prefix'):
        assert float(values[key]) == 0


def test_target_entropy_changes_reported_loss_but_not_the_coefficient_training_gradient():
    torch.manual_seed(261012)
    atoms = torch.zeros(1, 1, 1, 4, dtype=torch.long)
    atom_logits = torch.randn(*atoms.shape, 7, requires_grad=True)
    coeff_logits = torch.randn(*atoms.shape, 11, requires_grad=True)
    targets = torch.randn_like(coeff_logits).softmax(-1)
    ce_loss, _ = compound_objective(atom_logits, coeff_logits, None, atoms, targets,
        None, atom_weight=.575, geometry_weight=0., accumulation=1)
    entropy = -(targets*targets.log()).sum(-1).mean()
    kl_loss = ce_loss-entropy/1.575
    ce_gradient = torch.autograd.grad(ce_loss, coeff_logits, retain_graph=True)[0]
    kl_gradient = torch.autograd.grad(kl_loss, coeff_logits)[0]
    torch.testing.assert_close(ce_gradient, kl_gradient, rtol=0, atol=0)
    assert ce_loss > kl_loss
