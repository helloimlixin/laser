from types import SimpleNamespace

import pytest
import torch

from src import ffhq_v4_archived as archived
from src.church_joint_geometry import joint_candidate_contribution,conditional_geometry,make_prior,objective,early_decay_lr
from tests.test_church_ffhq_archived import tiny


def test_signed_atom_mixture_has_correct_contribution():
    # Both alternatives encode exactly the same latent: +D with +c, -D with -c.
    weights=torch.tensor([.5,.5])
    atoms=torch.tensor([[1.,0.],[-1.,0.]])
    coefficients=torch.tensor([2.,-2.])
    actual=joint_candidate_contribution(weights,atoms,coefficients)
    torch.testing.assert_close(actual,torch.tensor([2.,0.]))
    archived_prediction=(weights[:,None]*atoms).sum(0)*coefficients[0]
    assert not torch.equal(actual,archived_prediction)


def test_joint_geometry_matches_explicit_joint_probability_enumeration_and_gradients():
    atoms=torch.tensor([[1.,.2],[.3,-.7],[-.2,.8]],dtype=torch.float64)
    bins=torch.tensor([-2.,0.,2.],dtype=torch.float64)
    atom_logits=torch.tensor([.2,-.1,.8],dtype=torch.float64,requires_grad=True)
    coefficient_logits=torch.tensor([[1.,0.,-1.],[-.8,.2,1.],[.3,-.4,.2]],dtype=torch.float64,requires_grad=True)
    weights=atom_logits.softmax(-1);p=coefficient_logits.softmax(-1)
    actual=joint_candidate_contribution(weights,atoms,(p*bins).sum(-1))
    expected=sum(weights[a]*p[a,c]*atoms[a]*bins[c] for a in range(3) for c in range(3))
    torch.testing.assert_close(actual,expected)
    ga=torch.autograd.grad(actual.square().sum(),(atom_logits,coefficient_logits),retain_graph=True)
    gb=torch.autograd.grad(expected.square().sum(),(atom_logits,coefficient_logits))
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b)


@pytest.mark.parametrize('sharp',[True,False])
def test_complete_geometry_path_conditions_each_candidate_and_trains_alternatives(sharp):
    table=torch.tensor([[-80.,80.],[80.,-80.]] if sharp else [[-1.,1.],[1.,-1.]],requires_grad=True)
    aux=SimpleNamespace(dictionary=torch.tensor([[1.,-1.]]),coeff_bins=torch.tensor([-1.,1.]),coeff_scales=torch.ones(1))
    model=SimpleNamespace(coefficient_logits=lambda hidden,vectors,depth_index:table[(vectors[...,0]<0).long()])
    atoms=torch.zeros(1,1,1,1,dtype=torch.long)
    physical=torch.ones_like(atoms,dtype=torch.float32)
    out={'atom_logits':torch.zeros(1,1,1,1,2),'coeff_logits':table[0].expand(1,1,1,1,2),
        'head_outputs':torch.zeros(1,1,1,1,3)}
    loss=conditional_geometry(model,aux,out,atoms,physical,top_k=2)
    if sharp:
        torch.testing.assert_close(loss,torch.tensor(0.))
        _,old=archived.compound_objective(out['atom_logits'],out['coeff_logits'],None,atoms,
            torch.tensor([0.,1.]).expand(1,1,1,1,2),torch.ones(1,1,1,1,1),atom_weight=1.5,
            geometry_weight=.05,accumulation=1,distribution_geometry=True,geometry_dictionary=aux.dictionary,
            geometry_coeff_bins=aux.coeff_bins,geometry_coeff_scales=aux.coeff_scales,geometry_top_k=2)
        torch.testing.assert_close(old['geometry'],torch.tensor(1.))
    else:
        gradient=torch.autograd.grad(loss,table)[0]
        assert (gradient.abs().sum(-1)>0).all()


def test_archived_outputs_samples_and_classification_unchanged_without_geometry():
    torch.manual_seed(19)
    old,aux,packed=tiny()
    new=make_prior(config=old.config,num_atoms=7,coeff_vocab_size=8)
    new.load_state_dict(old.state_dict(),strict=True)
    old.eval();new.eval()
    with torch.no_grad():
        a=old(packed,model_aux=aux);b=new(packed,model_aux=aux)
    for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    samples=[]
    for m in (old,new):
        torch.manual_seed(27)
        samples.append(m.sample_compound(2,aux,atom_top_k=7,atom_top_p=1.,coeff_top_p=.85,amp=False))
    assert all(torch.equal(a,b) for a,b in zip(*samples))
    from src.church_ffhq_archived import objective as previous
    atoms=packed//8;physical=aux.coeff_bins[packed%8]*aux.coeff_scales
    a,ma=previous(old,aux,atoms,physical,0.,stochastic=False)
    b,mb=objective(new,aux,atoms,physical,0.,stochastic=False)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert ma==mb


def test_corrected_geometry_backpropagates_through_candidate_coefficients():
    torch.manual_seed(20)
    old,aux,packed=tiny()
    model=make_prior(config=old.config,num_atoms=7,coeff_vocab_size=8).train()
    atoms=packed//8;physical=aux.coeff_bins[packed%8]*aux.coeff_scales
    loss,metrics=objective(model,aux,atoms,physical,.05,stochastic=False)
    loss.backward()
    assert torch.isfinite(loss) and metrics['geometry']>0
    for module in [model.classifier,model.coeff_classifier,model.coeff_micro_transformer]:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        assert sum(float(p.grad.abs().sum()) for p in module.parameters())>0


def test_early_decay_schedule_endpoints_monotonicity_and_resume_progress():
    total=300*493
    assert early_decay_lr(0,total,493)==pytest.approx(5e-4)
    assert early_decay_lr(5*493,total,493)==pytest.approx(2.75e-4)
    assert early_decay_lr(10*493,total,493)==pytest.approx(5e-5)
    assert early_decay_lr(total,total,493)==pytest.approx(1e-6)
    values=[early_decay_lr(s,total,493) for s in range(0,total+1,493)]
    assert all(a>=b for a,b in zip(values,values[1:]))
    for invalid in [-1]:
        with pytest.raises(ValueError):early_decay_lr(invalid,total,493)


def test_scratch_architecture_has_no_new_parameters_or_weight_loading(monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('No checkpoint loading during construction')
    monkeypatch.setattr(torch,'load',forbidden)
    monkeypatch.setattr(torch.nn.Module,'load_state_dict',forbidden)
    with torch.device('meta'):model=make_prior()
    assert sum(p.numel() for p in model.parameters())==404738048
