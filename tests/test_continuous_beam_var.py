from types import SimpleNamespace
import math
import pytest
import torch
from torch.nn import functional as F
from torch import nn
from src.models.continuous_beam_var import (ContinuousBeamQuantizer,ContinuousBeamVAR,
    ContinuousCoefficientMixture,continuous_conditionals)


def quantizer():
    torch.manual_seed(37)
    dictionary=F.normalize(torch.randn(4,9),dim=0)
    class Geometry(nn.Module):
        Cvae=4;sparsity=2;v_patch_nums=(1,2,3)
        def contribution(self,vectors,scale):
            pn=self.v_patch_nums[scale]
            x=vectors.transpose(1,2).reshape(len(vectors),4,pn,pn)
            return F.interpolate(x,(3,3),mode='bicubic',align_corners=False) if pn!=3 else x
        def next_input(self,accumulated,scale):
            return F.interpolate(accumulated,(self.v_patch_nums[scale],)*2,mode='area').flatten(2).transpose(1,2)
    source=SimpleNamespace(source=Geometry(),sparsity=2,v_patch_nums=(1,2,3),physical_tables=lambda:(dictionary,None))
    return ContinuousBeamQuantizer(source)


def test_beam16_matches_exhaustive_two_atom_ols_and_retains_continuous_values():
    q=quantizer();x=torch.randn(7,4)
    pool=q.beam_pool(x);a,c=pool['atoms'][:,0],pool['coefficients'][:,0]
    exhaustive=[]
    for i in range(9):
        for j in range(i+1,9):
            vectors=q.dictionary[:,[i,j]]
            values=torch.linalg.lstsq(vectors,x.T).solution.T
            exhaustive.append((x-values@vectors.T).square().sum(-1))
    expected=torch.stack(exhaustive,-1).amin(-1)
    torch.testing.assert_close(pool['costs'][:,0],expected,rtol=2e-4,atol=2e-6)
    residual=x-q.embed(a,c)
    torch.testing.assert_close((residual[...,None]*q.dictionary.T[a].transpose(-1,-2)).sum(-2),torch.zeros_like(c),atol=2e-5,rtol=0)
    assert c.dtype==torch.float32 and c.unique().numel()>4 and (a[:,0]!=a[:,1]).all()
    assert torch.isfinite(pool['weights']).all()
    torch.testing.assert_close(pool['weights'].sum(-1),torch.ones(7))


def test_multiscale_roundtrip_uses_the_same_continuous_teacher_pairs_and_inputs():
    q=quantizer();z=torch.randn(2,4,3,3)
    for teacher in [False,True]:
        result=q.decompose(z,teacher=teacher)
        decoded,inputs=q.from_codes(result['atoms'],result['coefficients'])
        torch.testing.assert_close(decoded,result['latent'],atol=0,rtol=0)
        torch.testing.assert_close(inputs,result['inputs'],atol=0,rtol=0)
        assert result['atoms'].shape==(2,14,2)
        assert result['coefficients'].dtype==torch.float32
        if teacher:
            torch.testing.assert_close(result['targets']['atom_weights'].sum(-1),torch.ones(2,14,2))
            torch.testing.assert_close(result['targets']['coefficient_weights'].sum(-1),torch.ones(2,14,2))


def test_beam_teacher_conditions_on_previous_atom_and_coefficient_without_future_leakage():
    a=torch.tensor([[[1,2],[1,3],[1,4],[6,7]]])
    c=torch.tensor([[[.1,.2],[.1,.5],[.7,.8],[.3,.4]]])
    w=torch.tensor([[.1,.2,.3,.4]])
    t=continuous_conditionals(a,c,w,torch.tensor([0]))
    torch.testing.assert_close(t['atom_weights'][0,1],torch.tensor([1/3,2/3,0.,0.]))
    other=continuous_conditionals(a,c,w,torch.tensor([1]))
    torch.testing.assert_close(t['atom_weights'][0,0],other['atom_weights'][0,0])
    torch.testing.assert_close(t['coefficient_weights'][0,0],other['coefficient_weights'][0,0])
    for selected in range(4):
        t=continuous_conditionals(a,c,w,torch.tensor([selected]));probability=torch.tensor(1.)
        for d in range(2):
            same_a=a[0,:,d]==a[0,selected,d]
            same_c=c[0,:,d]==c[0,selected,d]
            probability*=t['atom_weights'][0,d][same_a].sum()*t['coefficient_weights'][0,d][same_c].sum()
        torch.testing.assert_close(probability,w[0,selected])


def test_coefficient_head_is_a_normalized_continuous_density_with_physical_jacobian():
    weights=torch.tensor([[0.,1.]])
    means=torch.tensor([[-.3,.7]])
    std=torch.tensor([[.4,.5]])
    c=torch.linspace(-100,100,100001)[None]
    logp=ContinuousCoefficientMixture.log_prob((weights,means,std),c,torch.tensor(.8))
    integral=torch.trapezoid(logp.exp()[0],c[0])
    torch.testing.assert_close(integral,torch.tensor(1.),atol=2e-5,rtol=0)
    head=ContinuousCoefficientMixture(8)
    features=torch.randn(3,8,requires_grad=True);parameters=head(features)
    loss=-head.log_prob(parameters,torch.tensor([[.12345],[.78901],[-1.23456]]),torch.tensor(.8)).mean()
    loss.backward();assert features.grad.isfinite().all() and features.grad.abs().sum()>0
    values=head.sample(parameters,torch.tensor(.8),torch.Generator().manual_seed(3))
    assert values.dtype==torch.float32 and values.isfinite().all()


def model(q,monkeypatch):
    import dist
    monkeypatch.setattr(dist,'get_device',lambda:'cpu')
    m=ContinuousBeamVAR(q,depth=2,width=32,heads=2,num_classes=3)
    m.cond_drop_rate=0
    return m.eval()


def test_prior_coefficient_and_atom_causality_and_finite_soft_teacher_gradients(monkeypatch):
    q=quantizer();m=model(q,monkeypatch);result=q.decompose(torch.randn(2,4,3,3),teacher=True)
    features=torch.randn(2,14,m.C)
    logits,parameters=m.token_parameters(features,result['atoms'],result['coefficients'])
    changed=result['coefficients'].clone();changed[...,1]+=.71
    other_logits,other_parameters=m.token_parameters(features,result['atoms'],changed)
    torch.testing.assert_close(logits,other_logits,atol=0,rtol=0)
    for first,second in zip(parameters,other_parameters):torch.testing.assert_close(first,second,atol=0,rtol=0)
    changed=result['coefficients'].clone();changed[...,0]+=.71
    other_logits,other_parameters=m.token_parameters(features,result['atoms'],changed)
    torch.testing.assert_close(logits[...,0,:],other_logits[...,0,:],atol=0,rtol=0)
    for first,second in zip(parameters,other_parameters):torch.testing.assert_close(first[...,0,:],second[...,0,:],atol=0,rtol=0)
    loss=m(torch.tensor([0,1]),result['inputs'],result['atoms'],result['coefficients'],result['targets']);loss.backward()
    assert loss.isfinite()
    for name,p in m.named_parameters():assert p.grad is not None and p.grad.isfinite().all(),name
    assert all(p.grad is None for p in q.parameters())


@pytest.mark.parametrize('cfg',[0.,1.5])
def test_cached_sampling_matches_continuous_teacher_forcing_and_decoding(monkeypatch,cfg):
    import src.models.continuous_beam_var as module
    q=quantizer();m=model(q,monkeypatch);result=q.decompose(torch.randn(2,4,3,3))
    atoms,coefficients,labels=result['atoms'],result['coefficients'],torch.tensor([0,1])
    both=torch.cat((labels,torch.full_like(labels,m.num_classes)))
    features=super(ContinuousBeamVAR,m).forward(both,result['inputs'].repeat(2,1,1))
    expected_logits,expected_parameters=m.token_parameters(features,atoms.repeat(2,1,1),coefficients.repeat(2,1,1))
    calls=[]
    def force_atoms(logits,**kwargs):
        scale,d=divmod(len(calls),2);lo,hi=m.begin_ends[scale]
        expected=expected_logits[:,lo:hi,d].clone();mask=~expected[:2].isfinite();expected.masked_fill_(~expected.isfinite(),0.)
        guidance=cfg*scale/m.num_stages_minus_1
        expected=((1+guidance)*expected[:2]-guidance*expected[2:]).masked_fill(mask,-torch.inf)
        torch.testing.assert_close(logits,expected,atol=2e-6,rtol=3e-4);calls.append((scale,d))
        return atoms[:,lo:hi,d,None]
    def force_coefficients(parameters,scale,generator):
        stage,d=calls[-1];lo,hi=m.begin_ends[stage]
        for actual,expected in zip(parameters,expected_parameters):torch.testing.assert_close(actual,expected[:2,lo:hi,d],atol=2e-6,rtol=3e-4)
        return coefficients[:,lo:hi,d]
    monkeypatch.setattr(module,'sample_with_top_k_top_p_',force_atoms)
    monkeypatch.setattr(ContinuousCoefficientMixture,'sample',staticmethod(force_coefficients))
    latent,generated_a,generated_c=m.sample(labels,cfg=cfg,return_codes=True)
    torch.testing.assert_close(latent,result['latent'],atol=2e-6,rtol=3e-4)
    assert torch.equal(generated_a,atoms) and torch.equal(generated_c,coefficients)
    assert len(calls)==6 and all(not block.attn.caching for block in m.blocks)
