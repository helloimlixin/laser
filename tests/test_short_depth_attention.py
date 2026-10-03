import torch
import pytest
from src.models.rqtransformer.attentions import causal_short_attention

@pytest.mark.parametrize('scale',[.1,1.,10.])
def test_causal_short_outputs_and_gradients_match_reference(scale):
 torch.manual_seed(189)
 inputs=[(torch.randn(4,3,4,64,dtype=torch.float64)*scale).requires_grad_() for _ in range(3)]
 ref=torch.nn.functional.scaled_dot_product_attention(*inputs,is_causal=True)
 out=causal_short_attention(*inputs)
 torch.testing.assert_close(out,ref,rtol=1e-11,atol=1e-11)
 cotangent=torch.randn_like(out)
 expected=torch.autograd.grad(ref,inputs,cotangent,retain_graph=True)
 actual=torch.autograd.grad(out,inputs,cotangent)
 for got,wanted in zip(actual,expected):torch.testing.assert_close(got,wanted,rtol=1e-9,atol=1e-9)

def test_future_values_cannot_affect_earlier_depth_outputs():
 torch.manual_seed(17)
 q,k,v=[torch.randn(2,3,4,64) for _ in range(3)]
 baseline=causal_short_attention(q,k,v)
 for first in range(1,4):
  changed_k=k.clone();changed_v=v.clone();changed_k[...,first:,:]+=100;changed_v[...,first:,:]-=100
  updated=causal_short_attention(q,changed_k,changed_v)
  torch.testing.assert_close(updated[...,:first,:],baseline[...,:first,:],rtol=0,atol=0)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_compiled_short_attention_outputs_and_gradients_on_cuda(dtype):
 from src.models.rqtransformer.attentions import compiled_causal_short_attention
 torch.manual_seed(414)
 # Sequence-major source creates the same strides used in production.
 inputs=[torch.randn(4,128,24,64,device='cuda',dtype=dtype).permute(1,2,0,3).detach().requires_grad_() for _ in range(3)]
 with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
  expected=torch.nn.functional.scaled_dot_product_attention(*inputs,is_causal=True)
 actual=compiled_causal_short_attention()(*inputs)
 cotangent=torch.randn_like(actual)
 ref_grad=torch.autograd.grad(expected,inputs,cotangent,retain_graph=True)
 got_grad=torch.autograd.grad(actual,inputs,cotangent)
 tol=2e-5 if dtype==torch.float32 else .035
 torch.testing.assert_close(actual,expected,atol=tol,rtol=tol)
 for got,wanted in zip(got_grad,ref_grad):torch.testing.assert_close(got,wanted,atol=tol,rtol=tol)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('length,backend',[(64,'sdpa'),(4,'compiled'),(2,'sdpa')])
@pytest.mark.parametrize('amp',[False,True])
def test_compiled_block_preserves_state_and_gradients(length,backend,amp):
 import copy
 from src.models.rqtransformer.attentions import AttentionBlock, compile_training_forward
 from src.models.rqtransformer.configs import AttentionBlockConfig
 torch.manual_seed(64)
 reference=AttentionBlock(AttentionBlockConfig(embed_dim=128,n_head=2,resid_pdrop=0)).cuda()
 candidate=copy.deepcopy(reference)
 if length==4:candidate.attn.short_attention_backend=backend
 if length==2:
  reference.attn.pair_attention_backend='compiled';candidate.attn.pair_attention_backend='compiled'
 original_keys=list(candidate.state_dict());compile_training_forward(candidate)
 assert list(candidate.state_dict())==original_keys
 candidate.load_state_dict(reference.state_dict(),strict=True)
 inputs=[torch.randn(8,length,128,device='cuda',requires_grad=True)]
 inputs.append(inputs[0].detach().clone().requires_grad_())
 with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
  expected=reference(inputs[0]);actual=candidate(inputs[1])
 torch.testing.assert_close(actual,expected,rtol=.02 if amp else 1e-5,atol=.012 if amp else 1e-5)
 cotangent=torch.randn_like(actual)
 expected.backward(cotangent);actual.backward(cotangent)
 torch.testing.assert_close(inputs[0].grad,inputs[1].grad,rtol=.03,atol=.03)
 numerator=denominator=0.
 for (name,a),(other,b) in zip(reference.named_parameters(),candidate.named_parameters()):
  assert name==other
  # Key bias cancels exactly in attention; its near-zero reference gradient
  # needs an absolute BF16 tolerance rather than a relative-only test.
  if amp and name=='attn.key.bias':
   scale=reference.attn.key.weight.grad.float().norm()
   assert max(a.grad.float().norm(),b.grad.float().norm()) < .01*scale
  elif amp:
   error=(a.grad-b.grad).float()
   assert error.norm() <= .02*a.grad.float().norm()+.01,name
   assert error.abs().max() <= .01*a.grad.abs().max()+.002,name
  else:
   torch.testing.assert_close(a.grad,b.grad,rtol=1e-4,atol=1e-4)
  numerator+=(a.grad-b.grad).float().square().sum().item()
  denominator+=a.grad.float().square().sum().item()
 assert (numerator/denominator)**.5 < .02


def test_compiled_training_dispatch_preserves_eager_sampling(monkeypatch):
    from src.models.rqtransformer.attentions import (
        AttentionBlock, compile_training_forward,
    )
    from src.models.rqtransformer.configs import AttentionBlockConfig

    calls = []
    def fake_compile(forward, **kwargs):
        def compiled(*args, **kwargs):
            calls.append(True)
            return forward(*args, **kwargs)
        return compiled

    monkeypatch.setattr(torch, 'compile', fake_compile)
    block = AttentionBlock(AttentionBlockConfig(embed_dim=16, n_head=2))
    compile_training_forward(block)
    block(torch.randn(2, 4, 16))
    assert len(calls) == 1
    block.eval()(torch.randn(2, 4, 16))
    block.train()
    with torch.no_grad():
        block(torch.randn(2, 4, 16))
    assert len(calls) == 1


def test_short_attention_falls_back_for_sampling_and_attention_dropout(monkeypatch):
    from src.models.rqtransformer import attentions
    from src.models.rqtransformer.configs import AttentionBlockConfig

    attention = attentions.MultiSelfAttention(
        AttentionBlockConfig(embed_dim=16, n_head=2, attn_pdrop=.1)
    )
    attention.short_attention_backend = 'compiled'
    def forbidden(*args, **kwargs):
        raise AssertionError('Training shortcut used during sampling or attention dropout')
    monkeypatch.setattr(attentions, 'causal_short_attention', forbidden)
    attention(torch.randn(2, 4, 16))
    attention.attn_drop.p = 0.
    attention.eval()(torch.randn(2, 4, 16))
    attention.train()
    with torch.no_grad():
        attention(torch.randn(2, 4, 16))
        output, cache = attention(torch.randn(2, 1, 16), caching=True)
        next_output, _ = attention(torch.randn(2, 1, 16), caching=True, past_kv=cache)
    assert output.shape == next_output.shape == (2, 1, 16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('accumulation', [2, 4])
def test_compiled_compound_objective_preserves_loss_and_gradients(accumulation):
    from src.training.rqtransformer import compound_objective

    torch.manual_seed(39)
    atoms = torch.randn(2, 2, 2, 4, 37, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    coefficients = torch.randn(2, 2, 2, 4, 17, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    target_atoms = torch.randint(37, atoms.shape[:-1], device='cuda')
    target_probs = torch.randn_like(coefficients, dtype=torch.float32).softmax(-1)
    compiled = torch.compile(compound_objective, fullgraph=True, dynamic=False)
    args = (atoms, coefficients, None, target_atoms, target_probs, None)
    kwargs = dict(atom_weight=1.5, geometry_weight=0., accumulation=accumulation)
    expected, reference_metrics = compound_objective(*args, **kwargs)
    actual, candidate_metrics = compiled(*args, **kwargs)
    torch.testing.assert_close(actual, expected)
    for key, value in reference_metrics.items():
        if value is not None:
            torch.testing.assert_close(candidate_metrics[key], value)
    expected_grad = torch.autograd.grad(expected, (atoms, coefficients), retain_graph=True)
    actual_grad = torch.autograd.grad(actual, (atoms, coefficients))
    for got, wanted in zip(actual_grad, expected_grad):
        torch.testing.assert_close(got, wanted, rtol=.02, atol=2e-5)
