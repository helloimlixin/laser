import math
import pytest
import torch
from src.training.compound_depth_loss import compound_depth_classification
from src.training.rqtransformer import compound_objective, build_parser


def case():
    torch.manual_seed(92)
    a=torch.randn(2,1,1,4,7,requires_grad=True)
    c=torch.randn(2,1,1,4,5,requires_grad=True)
    atoms=torch.tensor([0,1,2,3]).expand(2,1,1,4)
    probabilities=torch.randn_like(c).softmax(-1).detach()
    return a,c,atoms,probabilities


def objective(a,c,atoms,probs,**kwargs):
    return compound_objective(a,c,None,atoms,probs,None,atom_weight=1.2,
                              accumulation=3,geometry_weight=0.,**kwargs)


def test_defaults_exactly_preserve_previous_loss_and_gradients():
    a,c,atoms,probs=case()
    actual,metrics=objective(a,c,atoms,probs)
    ar=-a.float().log_softmax(-1).gather(-1,atoms[...,None]).squeeze(-1)
    cr=-(probs*c.float().log_softmax(-1)).sum(-1)
    expected=(1.2*ar.sum(-1)+cr.sum(-1)).mean()/((1.2+1.)*4)/3
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    left=torch.autograd.grad(actual,(a,c),retain_graph=True)
    right=torch.autograd.grad(expected,(a,c))
    for x,y in zip(left,right):torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_separate_depth_weights_normalize_and_match_explicit_chain_terms():
    a,c,atoms,probs=case()
    wa=torch.tensor([1.,1.,2.,3.]);wc=torch.tensor([1.,2.,1.,4.])
    loss,metrics=objective(a,c,atoms,probs,atom_depth_weights=wa.tolist(),
                           coefficient_depth_weights=wc.tolist())
    expected=(1.2*(metrics['atom_cross_entropy']*(wa/wa.mean())).mean()
              +(metrics['coeff_cross_entropy']*(wc/wc.mean())).mean())/2.2/3
    torch.testing.assert_close(loss,expected)
    scaled,_=objective(a,c,atoms,probs,atom_depth_weights=(wa*11).tolist(),
                        coefficient_depth_weights=(wc*5).tolist())
    torch.testing.assert_close(loss,scaled,rtol=0,atol=0)
    assert torch.isfinite(loss)


def test_later_atom_and_coefficient_gradients_receive_requested_emphasis():
    a=torch.zeros(1,1,1,4,7,requires_grad=True)
    c=torch.zeros(1,1,1,4,5,requires_grad=True)
    atoms=torch.tensor([[[[0,1,2,3]]]])
    probs=torch.nn.functional.one_hot(torch.tensor([[[[0,1,2,3]]]]),5).float()
    loss,_=objective(a,c,atoms,probs,atom_depth_weights=[1,1,2,3],
                      coefficient_depth_weights=[1,2,1,4])
    loss.backward()
    torch.testing.assert_close(a.grad[...,3,:].norm(),3*a.grad[...,0,:].norm())
    torch.testing.assert_close(c.grad[...,3,:].norm(),4*c.grad[...,0,:].norm())
    assert (a.grad.square().sum(-1)>0).all() and (c.grad.square().sum(-1)>0).all()


@pytest.mark.parametrize('weights',[[1,2],[1,1,0,2],[1,1,-1,2],[1,1,float('nan'),2],[1,1,float('inf'),2]])
def test_invalid_depth_weights_are_rejected(weights):
    with pytest.raises(ValueError):
        compound_depth_classification(torch.ones(2,4),torch.ones(2,4),atom_depth_weights=weights)


def test_cli_exposes_both_depth_weight_fields():
    options={action.dest for action in build_parser()._actions}
    assert {'atom_depth_loss_weights','coeff_depth_loss_weights'}<=options
