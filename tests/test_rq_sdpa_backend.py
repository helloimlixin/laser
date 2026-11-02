import copy
from unittest.mock import patch

import pytest
import torch

from src.training.rq_sdpa_backend import sdpa_attention
from src.rq_upstream import RQTransformerConfig
from rqvae.models.rqtransformer.attentions import MultiSelfAttention
from tests.test_laser_native_support_order_adapter import case


def attention():
    return MultiSelfAttention(RQTransformerConfig.create(__import__('omegaconf').OmegaConf.create(
        dict(embed_dim=12,body=dict(block=dict(n_head=3,attn_pdrop=0.,resid_pdrop=0.))))).body.block).eval()


@pytest.mark.parametrize('batch_limit',[None,1])
def test_sdpa_matches_native_outputs_input_gradients_and_parameter_gradients(batch_limit):
    torch.manual_seed(31);native=attention();fused=copy.deepcopy(native)
    x=torch.randn(2,7,12,requires_grad=True);y=x.detach().clone().requires_grad_()
    expected=native(x);expected.square().sum().backward()
    with sdpa_attention(fused,batch_limit=batch_limit):
        actual=fused(y);actual.square().sum().backward()
    torch.testing.assert_close(actual,expected,rtol=1.e-5,atol=1.e-6)
    torch.testing.assert_close(y.grad,x.grad,rtol=1.e-5,atol=1.e-6)
    for a,b in zip(native.parameters(),fused.parameters()):
        torch.testing.assert_close(a.grad,b.grad,rtol=1.e-5,atol=2.e-6)
    assert 'forward' not in fused.__dict__
    assert '_sdpa_batch_limit' not in fused.__dict__


@pytest.mark.parametrize('batch_limit',[None,1])
@pytest.mark.parametrize('chunks',[[1,1,1,1,1,1,1],[3,2,2]])
@torch.no_grad()
def test_cached_offset_causality_matches_native_for_single_and_multiple_queries(chunks,batch_limit):
    torch.manual_seed(32);model=attention();x=torch.randn(2,7,12)
    reference=model(x);cache=None;outputs=[];first=0
    with sdpa_attention(model,batch_limit=batch_limit):
        for length in chunks:
            value,cache=model(x[:,first:first+length],caching=True,past_kv=cache)
            outputs.append(value);first+=length
    torch.testing.assert_close(torch.cat(outputs,1),reference,rtol=1.e-5,atol=1.e-6)


@torch.no_grad()
def test_full_model_and_cache_keep_causality_and_state_dict_with_sdpa():
    _,model,aux,truth,labels=case('support_first')
    before=copy.deepcopy(model.state_dict());reference=model(truth,aux,labels)
    with sdpa_attention(model):
        expected=model(truth,aux,labels)
        for name in expected:torch.testing.assert_close(expected[name],reference[name],rtol=1.e-5,atol=3.e-6)
        prefix=torch.zeros_like(truth);prefix[...,1::2]=11+16;model.init_cache()
        for h in range(2):
            for w in range(2):
                for event,external in enumerate(model.event_positions):
                    actual=model.cached_forward(prefix[:,:h+1],aux,labels,sample_loc=(h,w,event))
                    atom=external%2==0;actual=actual[:,:11] if atom else actual[:,11:]
                    target=expected['atom_logits' if atom else 'coeff_logits'][:,h,w,external//2]
                    torch.testing.assert_close(actual,target,rtol=1.e-5,atol=3.e-6)
                    prefix[:,h,w,external]=truth[:,h,w,external]
    for name,tensor in model.state_dict().items():torch.testing.assert_close(tensor,before[name],rtol=0,atol=0)


@pytest.mark.parametrize('model_kind',['support_first','interleaved','compound'])
@torch.no_grad()
def test_skipping_full_vocabulary_topk_preserves_entire_sample_bit_for_bit(model_kind):
    if model_kind=='compound':
        import src.models.laser_rq_adapter as adapter
        from tests.test_laser_native_compound_adapter import case as compound_case
        _,model,aux,_,labels=compound_case()
    else:
        import src.models.laser_native_support_order_adapter as adapter
        _,model,aux,_,labels=case(model_kind)
    original=adapter.sample_from_logits
    def redundant_topk(logits,**kwargs):
        if kwargs['top_k'] is None:kwargs['top_k']=logits.shape[-1]
        return original(logits,**kwargs)
    torch.manual_seed(55)
    with patch.object(adapter,'sample_from_logits',redundant_topk):
        before=model.sample_sparse(2,aux,labels,amp=False,atom_top_p=.9,coeff_top_p=.85)
    torch.manual_seed(55)
    after=model.sample_sparse(2,aux,labels,amp=False,atom_top_p=.9,coeff_top_p=.85)
    torch.testing.assert_close(after,before,rtol=0,atol=0)
