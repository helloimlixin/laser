from types import SimpleNamespace
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from src.models.compound_beam128_var import CompoundBeam128Quantizer,BeamAlignedCompound128VAR
from src.models.multiscale_laser_var import VAR


def quantizer():
    torch.manual_seed(73);dictionary=F.normalize(torch.randn(4,8),dim=0)
    class Geometry(nn.Module):
        Cvae=4;sparsity=2;v_patch_nums=(1,2,3)
        def contribution(self,vectors,scale):
            pn=self.v_patch_nums[scale];x=vectors.transpose(1,2).reshape(len(vectors),4,pn,pn)
            return F.interpolate(x,(3,3),mode='bicubic',align_corners=False) if pn!=3 else x
        def next_input(self,accumulated,scale):return F.interpolate(accumulated,(self.v_patch_nums[scale],)*2,mode='area').flatten(2).transpose(1,2)
    source=SimpleNamespace(source=Geometry(),sparsity=2,v_patch_nums=(1,2,3),physical_tables=lambda:(dictionary,None))
    q=CompoundBeam128Quantizer(source);q.coefficient_max.fill_(6.)
    return q


def test_128_bin_grid_and_packed_pairs_roundtrip():
    q=quantizer();ids=torch.arange(128)
    for s in range(3):
        torch.testing.assert_close(q.coefficient_ids(q.coefficient_values(ids,s),s),ids,atol=0,rtol=0)
        assert q.coefficient_values(ids,s).isfinite().all()
    atoms=torch.arange(8).repeat_interleave(128);bins=ids.repeat(8)
    packed=q.pack_pairs(atoms,bins);a,c=q.unpack_pairs(packed)
    assert packed.unique().numel()==1024 and torch.equal(a,atoms) and torch.equal(c,bins)
    assert q.coefficient_ids(torch.tensor([-100.,100.]),0).tolist()==[0,127]


def test_beam_scores_quantized_values_and_teacher_inputs_roundtrip_exactly():
    q=quantizer();signals=torch.randn(7,4);pool=q.quantized_pool(signals,1)
    expected=(signals[:,None]-q.embed(pool['atoms'],pool['coefficients'],1)).square().sum(-1)
    finite=pool['costs'].isfinite();torch.testing.assert_close(pool['costs'][finite],expected[finite])
    assert torch.equal(pool['canonical_index'],pool['costs'].argmin(-1))
    for teacher in [False,True]:
        result=q.decompose(torch.randn(2,4,3,3),teacher=teacher)
        z,inputs=q.from_codes(result['codes'])
        torch.testing.assert_close(z,result['latent'],atol=0,rtol=0);torch.testing.assert_close(inputs,result['inputs'],atol=0,rtol=0)
        assert result['coefficients'].dtype==torch.long and (result['coefficients']<128).all()
        if teacher:
            t=result['targets'];torch.testing.assert_close(t['joint_weights'].sum(-1),torch.ones(2,14,4))
            assert (t['joint_ids'][...,::2,:]<8).all()
            assert ((t['joint_ids'][...,1::2,:]>=8)&(t['joint_ids'][...,1::2,:]<136)).all()


def model(q,monkeypatch):
    import dist
    monkeypatch.setattr(dist,'get_device',lambda:'cpu')
    m=BeamAlignedCompound128VAR(SimpleNamespace(quantize=q),depth=2,width=32,heads=2,num_classes=3,
                               token_width=16,token_layers=2,token_heads=2)
    m.cond_drop_rate=0
    return m.eval()


def test_compound_soft_teacher_matches_dense_conditional_pair_nll_and_has_finite_gradients(monkeypatch):
    q=quantizer();m=model(q,monkeypatch);result=q.decompose(torch.randn(2,4,3,3),teacher=True)
    labels=torch.tensor([0,1]);events=m.pack_tokens(result['atoms'],result['coefficients'])
    features=VAR.forward(m,labels,result['inputs']);logits=m.token_logits(features,events)
    target=result['targets'];dense=torch.zeros_like(logits).scatter_add_(-1,target['joint_ids'],target['joint_weights'])
    logp=logits.log_softmax(-1);nll=-(dense*torch.where(dense>0,logp,torch.zeros_like(logp))).sum(-1)
    expected=nll[...,::2].mean()+nll[...,1::2].mean()
    loss=m(labels,result['inputs'],result['atoms'],result['coefficients'],target);torch.testing.assert_close(loss,expected)
    loss.backward()
    for name,p in m.named_parameters():assert p.grad is not None and p.grad.isfinite().all(),name
    changed=events.clone();changed[...,1]=8+(changed[...,1]-8+17)%128
    after=m.token_logits(features,changed)
    torch.testing.assert_close(logits[...,:2,:],after[...,:2,:])
    assert not torch.allclose(logits[...,2:,:],after[...,2:,:])


@pytest.mark.parametrize('cfg',[0.,1.5])
def test_compound_cached_sampling_matches_teacher_and_quantized_decode(monkeypatch,cfg):
    import src.models.compound_multiscale_var as module
    q=quantizer();m=model(q,monkeypatch);result=q.decompose(torch.randn(2,4,3,3),teacher=True)
    labels=torch.tensor([0,1]);both=torch.cat((labels,torch.full_like(labels,m.num_classes)))
    features=VAR.forward(m,both,result['inputs'].repeat(2,1,1));events=m.pack_tokens(result['atoms'],result['coefficients']);calls=[]
    def forced(logits,**kwargs):
        scale,slot=divmod(len(calls),4);lo,hi=m.begin_ends[scale]
        expected=m.prefix_logits(features[:,lo:hi],events[:,lo:hi,:slot].repeat(2,1,1))
        expected=expected[...,:8] if slot%2==0 else expected[...,8:]
        guidance=cfg*scale/m.num_stages_minus_1
        expected=(1+guidance)*expected[:2]-guidance*expected[2:]
        if slot%2==0 and slot:expected.scatter_(-1,events[:,lo:hi,:slot:2],-torch.inf)
        torch.testing.assert_close(logits,expected,atol=2e-6,rtol=2e-4);calls.append((scale,slot))
        target=events[:,lo:hi,slot,None]
        return target if slot%2==0 else target-8
    monkeypatch.setattr(module,'sample_with_top_k_top_p_',forced)
    generated=m.sample(labels,cfg=cfg,top_k=8)
    torch.testing.assert_close(generated,result['latent'],atol=2e-6,rtol=2e-4)
    assert len(calls)==12 and all(not b.attn.caching for b in m.blocks)
