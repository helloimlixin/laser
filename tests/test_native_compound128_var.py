from types import SimpleNamespace
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from src.models.native_compound128_var import NativeCompound128Quantizer,NativeCompound128VAR
from src.models.dictionary_learner import DictionaryLearning
from src.models.multiscale_laser_var import VAR


def quantizer():
    torch.manual_seed(89);dictionary=F.normalize(torch.randn(4,8),dim=0)
    class Geometry(nn.Module):
        Cvae=4;sparsity=2;v_patch_nums=(1,2,3)
        def contribution(self,vectors,scale):
            pn=self.v_patch_nums[scale];x=vectors.transpose(1,2).reshape(len(vectors),4,pn,pn)
            return F.interpolate(x,(3,3),mode='bicubic',align_corners=False) if pn!=3 else x
        def next_input(self,accumulated,scale):return F.interpolate(accumulated,(self.v_patch_nums[scale],)*2,mode='area').flatten(2).transpose(1,2)
    source=SimpleNamespace(source=Geometry(),sparsity=2,v_patch_nums=(1,2,3),physical_tables=lambda:(dictionary,None))
    q=NativeCompound128Quantizer(source);q.coefficient_max.fill_(6.)
    return q


def test_fast_two_atom_omp_matches_native_dictionary_learning_support_and_full_refit():
    q=quantizer();x=torch.randn(257,4)
    reference=DictionaryLearning(num_embeddings=8,embedding_dim=4,sparsity_level=2,omp_ridge=1e-5,omp_max_support_coherence=.999)
    atoms,coefficients=q.omp_codes(x);expected_a,expected_c=reference.batch_omp_with_support(x.T,q.dictionary)
    assert torch.equal(atoms,expected_a)
    torch.testing.assert_close(coefficients,expected_c,atol=3e-5,rtol=3e-5)
    assert (atoms[:,0]!=atoms[:,1]).all()


def test_native_target_generation_has_no_beam_or_candidate_sampling_and_preserves_rng(monkeypatch):
    q=quantizer();z=torch.randn(2,4,3,3);before=torch.get_rng_state().clone()
    def forbidden(*args,**kwargs):raise AssertionError('Candidate/trajectory sampling is forbidden in native OMP targets')
    monkeypatch.setattr(torch,'multinomial',forbidden);monkeypatch.setattr(torch.Tensor,'topk',forbidden)
    first=q.decompose(z);second=q.decompose(z)
    assert not hasattr(q,'beam_pool') and not hasattr(q,'quantized_pool')
    assert 'targets' not in first and 'weights' not in first
    assert torch.equal(first['codes'],second['codes']) and torch.equal(torch.get_rng_state(),before)


def test_native_128_bins_and_quantized_inputs_roundtrip_without_float_side_information():
    q=quantizer();bins=torch.arange(128)
    for s in range(3):assert torch.equal(q.coefficient_ids(q.coefficient_values(bins,s),s),bins)
    result=q.decompose(torch.randn(2,4,3,3));z,inputs=q.from_codes(result['codes'])
    torch.testing.assert_close(z,result['latent'],atol=0,rtol=0);torch.testing.assert_close(inputs,result['inputs'],atol=0,rtol=0)
    assert result['atoms'].dtype==torch.long and result['coefficients'].dtype==torch.long
    a,c=q.unpack_pairs(result['codes']);assert torch.equal(a,result['atoms']) and torch.equal(c,result['coefficients'])
    assert (c>=0).all() and (c<128).all()


def model(q,monkeypatch):
    import dist
    monkeypatch.setattr(dist,'get_device',lambda:'cpu')
    m=NativeCompound128VAR(SimpleNamespace(quantize=q),depth=2,width=32,heads=2,num_classes=3,
                           token_width=16,token_layers=2,token_heads=2)
    m.cond_drop_rate=0
    return m.eval()


def test_native_loss_is_plain_hard_atom_plus_coefficient_cross_entropy_with_causal_inputs(monkeypatch):
    q=quantizer();m=model(q,monkeypatch);result=q.decompose(torch.randn(2,4,3,3));labels=torch.tensor([0,1])
    events=m.pack_tokens(result['atoms'],result['coefficients']);features=VAR.forward(m,labels,result['inputs']);logits=m.token_logits(features,events)
    expected=F.cross_entropy(logits[:,:,::2,:8].reshape(-1,8),result['atoms'].reshape(-1))+F.cross_entropy(logits[:,:,1::2,8:].reshape(-1,128),result['coefficients'].reshape(-1))
    loss=m(labels,result['inputs'],result['atoms'],result['coefficients']);torch.testing.assert_close(loss,expected)
    loss.backward()
    for name,p in m.named_parameters():assert p.grad is not None and p.grad.isfinite().all(),name
    changed=events.clone();changed[...,1]=8+(changed[...,1]-8+17)%128
    after=m.token_logits(features,changed);torch.testing.assert_close(logits[...,:2,:],after[...,:2,:])
    assert not torch.allclose(logits[...,2:,:],after[...,2:,:])


@pytest.mark.parametrize('cfg',[0.,1.5])
def test_native_compound_sampling_matches_training_and_hard_bin_decode(monkeypatch,cfg):
    import src.models.compound_multiscale_var as module
    q=quantizer();m=model(q,monkeypatch);result=q.decompose(torch.randn(2,4,3,3));labels=torch.tensor([0,1])
    both=torch.cat((labels,torch.full_like(labels,m.num_classes)));features=VAR.forward(m,both,result['inputs'].repeat(2,1,1))
    events=m.pack_tokens(result['atoms'],result['coefficients']);calls=[]
    def forced(logits,**kwargs):
        scale,slot=divmod(len(calls),4);lo,hi=m.begin_ends[scale]
        expected=m.prefix_logits(features[:,lo:hi],events[:,lo:hi,:slot].repeat(2,1,1));expected=expected[...,:8] if slot%2==0 else expected[...,8:]
        g=cfg*scale/m.num_stages_minus_1;expected=(1+g)*expected[:2]-g*expected[2:]
        if slot%2==0 and slot:expected.scatter_(-1,events[:,lo:hi,:slot:2],-torch.inf)
        torch.testing.assert_close(logits,expected,atol=2e-6,rtol=2e-4);calls.append((scale,slot))
        target=events[:,lo:hi,slot,None];return target if slot%2==0 else target-8
    monkeypatch.setattr(module,'sample_with_top_k_top_p_',forced)
    generated=m.sample(labels,cfg=cfg,top_k=8);torch.testing.assert_close(generated,result['latent'],atol=2e-6,rtol=2e-4)
    assert len(calls)==12 and all(not b.attn.caching for b in m.blocks)
