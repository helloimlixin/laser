import pytest
import torch
from torch.nn import functional as F
from src.models.multiscale_laser_var import MultiScaleLaser,VAR
from src.models.joint_multiscale_var import JointMultiScaleQuantizer
from src.models.beam_joint_var import BeamJointQuantizer,BeamAlignedJointVAR,joint_prefix_conditionals


def quantizer(atoms=4,levels=2):
    torch.manual_seed(51)
    source=MultiScaleLaser(channels=4,atoms=atoms,sparsity=2,patch_nums=(1,2,3),coefficient_bins=33)
    table=torch.linspace(-1.1,1.1,levels).expand(3,atoms,levels).clone()
    q=JointMultiScaleQuantizer(source,source.normalized_dictionary(),table)
    return BeamJointQuantizer(q)


def test_all_joint4096_ids_decode_without_side_coefficients_and_quantize_real_values():
    q=quantizer(atoms=1024,levels=4);ids=torch.arange(4096)
    for scale in range(3):
        atoms,values=q.ids_to_pairs(ids,scale)
        assert torch.equal(q.pairs_to_ids(atoms,values,scale),ids)
        expected=F.embedding(atoms,q.dictionary.T)*values[...,None]
        torch.testing.assert_close(q.embed_joint(ids,scale),expected,atol=0,rtol=0)
        continuous=torch.linspace(-2,2,4096)
        encoded=q.pairs_to_ids(atoms,continuous,scale)
        _,decoded=q.ids_to_pairs(encoded,scale)
        error=(decoded-continuous).abs()
        torch.testing.assert_close(error,(continuous[:,None]-q.coefficients[scale,atoms]).abs().amin(-1))
        assert encoded.dtype==torch.long and int(encoded.min())>=0 and int(encoded.max())<4096


def test_beam_matches_exhaustive_tiny_vocabulary_and_never_worsens_greedy_anchor():
    q=quantizer();x=torch.randn(13,4);pool=q.beam_pool(x,1)
    ids=torch.arange(q.vocab_size);vectors=q.embed_joint(ids,1)
    sums=vectors[:,None]+vectors[None,:]
    costs=(x[:,None,None]-sums).square().sum(-1)
    same=(ids[:,None]//q.levels)==(ids[None,:]//q.levels)
    costs=costs.masked_fill(same,torch.inf)
    expected=costs.flatten(1).amin(-1)
    best=pool['costs'].gather(-1,pool['canonical_index'][...,None]).squeeze(-1)
    torch.testing.assert_close(best,expected,rtol=2e-5,atol=2e-6)
    assert (best<=pool['costs'][...,-1]+1e-6).all()
    atoms=pool['candidates']//q.levels
    assert (atoms[...,0]!=atoms[...,1]).all()
    torch.testing.assert_close(pool['weights'].sum(-1),torch.ones(len(x)))


def test_joint_prefix_conditionals_recover_complete_sequence_probabilities():
    candidates=torch.tensor([[[1,2],[1,3],[4,2],[5,6]]])
    weights=torch.tensor([[.1,.2,.3,.4]],dtype=torch.float64)
    for choice in range(4):
        codes=candidates[:,choice];target=joint_prefix_conditionals(candidates,weights,codes)
        probability=torch.ones((),dtype=torch.float64)
        for depth in range(2):
            same=target['joint_ids'][0,depth]==codes[0,depth]
            probability*=target['joint_weights'][0,depth,same].sum()
        torch.testing.assert_close(probability,weights[0,choice],atol=1e-12,rtol=1e-12)
    first=joint_prefix_conditionals(candidates,weights,candidates[:,0])
    second=joint_prefix_conditionals(candidates,weights,candidates[:,1])
    torch.testing.assert_close(first['joint_weights'],second['joint_weights'])
    torch.testing.assert_close(first['joint_weights'][0,1],torch.tensor([1/3,2/3,0,0],dtype=torch.float64))


def test_canonical_and_sampled_beam_codes_roundtrip_with_identical_multiscale_inputs():
    q=quantizer();z=torch.randn(2,4,3,3)
    for teacher in [False,True]:
        result=q.decompose(z,teacher=teacher);latent,inputs=q.from_codes(result['codes'])
        torch.testing.assert_close(latent,result['latent'],atol=0,rtol=0)
        torch.testing.assert_close(inputs,result['inputs'],atol=0,rtol=0)
        assert result['codes'].dtype==torch.long and result['codes'].shape==(2,14,2)
        if teacher:
            target=result['targets']
            torch.testing.assert_close(target['joint_weights'].sum(-1),torch.ones(2,14,2))
            assert target['joint_ids'].shape==target['joint_weights'].shape


def model(q,monkeypatch):
    import dist
    monkeypatch.setattr(dist,'get_device',lambda:'cpu')
    m=BeamAlignedJointVAR(q,depth=2,width=32,heads=2,num_classes=3)
    m.cond_drop_rate=0
    return m.eval()


def test_sparse_beam_loss_matches_dense_conditional_ce_and_all_gradients_are_finite(monkeypatch):
    q=quantizer();m=model(q,monkeypatch);result=q.decompose(torch.randn(2,4,3,3),teacher=True)
    labels=torch.tensor([0,1]);features=VAR.forward(m,labels,result['inputs']);logits=m.token_logits(features,result['codes'])
    targets=result['targets'];dense=torch.zeros_like(logits).scatter_add_(-1,targets['joint_ids'],targets['joint_weights'])
    logp=logits.log_softmax(-1)
    expected=-(dense*torch.where(dense>0,logp,torch.zeros_like(logp))).sum(-1).mean()
    actual=m(labels,result['inputs'],result['codes'],targets)
    torch.testing.assert_close(actual,expected)
    actual.backward()
    for name,p in m.named_parameters():assert p.grad is not None and p.grad.isfinite().all(),name
    assert not any('coefficient_head' in n for n,_ in m.named_parameters())
    assert all(p.grad is None for p in q.parameters())


@pytest.mark.parametrize('cfg',[0.,1.5])
def test_joint_cached_sampling_matches_beam_teacher_forcing(monkeypatch,cfg):
    import src.models.joint_multiscale_var as module
    q=quantizer();m=model(q,monkeypatch);result=q.decompose(torch.randn(2,4,3,3),teacher=True)
    codes=result['codes'];labels=torch.tensor([0,1]);both=torch.cat((labels,torch.full_like(labels,m.num_classes)))
    features=VAR.forward(m,both,result['inputs'].repeat(2,1,1));expected=m.token_logits(features,codes.repeat(2,1,1));calls=[]
    def force_ids(logits,**kwargs):
        scale,depth=divmod(len(calls),2);lo,hi=m.begin_ends[scale]
        score=expected[:,lo:hi,depth].clone();mask=~score[:2].isfinite();score.masked_fill_(~score.isfinite(),0.)
        guidance=cfg*scale/m.num_stages_minus_1
        score=((1+guidance)*score[:2]-guidance*score[2:]).masked_fill(mask,-torch.inf)
        torch.testing.assert_close(logits,score,atol=2e-6,rtol=3e-4);calls.append((scale,depth))
        return codes[:,lo:hi,depth,None]
    monkeypatch.setattr(module,'sample_with_top_k_top_p_',force_ids)
    latent=m.sample(labels,cfg=cfg,top_k=q.vocab_size)
    torch.testing.assert_close(latent,result['latent'],atol=2e-6,rtol=3e-4)
    assert len(calls)==6 and all(not block.attn.caching for block in m.blocks)
