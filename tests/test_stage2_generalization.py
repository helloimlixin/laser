import pytest
import torch

from src.training.stage2_generalization import reweight_depths


def test_depth_weighting_preserves_scale_auxiliary_loss_and_weights_gradients():
    atom = torch.ones(2, 4, requires_grad=True)
    coeff = torch.ones(2, 4, requires_grad=True)
    classification = (1.5*atom+coeff).mean()/2.5
    auxiliary = torch.tensor(.3, requires_grad=True)
    objective = dict(atom_nll=atom, coeff_cross_entropy=coeff, classification=classification)
    original = (classification+auxiliary)/2
    identical, _ = reweight_depths(original, objective, depth_weights=[1]*4, atom_weight=1.5, accumulation=2)
    assert torch.equal(identical, original)
    loss, _ = reweight_depths(original, objective, depth_weights=[2,1,.5,.5], atom_weight=1.5, accumulation=2)
    assert torch.equal(loss, original)
    loss.backward()
    assert torch.allclose(atom.grad[0]/atom.grad[0,-1], torch.tensor([4.,2.,1.,1.]))
    assert torch.allclose(atom.grad/coeff.grad, torch.full_like(atom,1.5))
    assert auxiliary.grad == .5


def test_invalid_depth_weight_is_rejected():
    values = torch.ones(1,4)
    with pytest.raises(ValueError):
        reweight_depths(values.mean(), dict(atom_nll=values,coeff_cross_entropy=values,classification=values.mean()),
                        depth_weights=[1,1,0,1],atom_weight=1.5,accumulation=1)
