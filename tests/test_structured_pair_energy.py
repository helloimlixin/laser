from dataclasses import replace

import pytest
import torch

from src.models.structured_pair_energy import (
    StructuredPairEnergyRQ, coefficient_partition, structured_pair_cross_entropy,
)
from src.models.typed_pair_rq import TypedPairCodebook
from src.training.typed_pair_targets import typed_pair_targets
from tests.test_offset_pair_prior import case


@pytest.mark.parametrize('chunk', [1, 3, 20])
def test_streamed_cartesian_normalizer_and_gradients_match_dense_oracle(chunk):
    torch.manual_seed(210)
    coupling = torch.randn(5,7,dtype=torch.float64,requires_grad=True)
    bias = torch.randn(5,11,dtype=torch.float64,requires_grad=True)
    bins = torch.linspace(-3,2,11,dtype=torch.float64)
    actual = coefficient_partition(coupling,bias,bins,row_chunk=chunk)
    expected = (coupling[:,:,None]*bins+bias[:,None,:]).logsumexp(-1)
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    weight = torch.randn_like(actual)
    for a,b in zip(torch.autograd.grad((actual*weight).sum(),(coupling,bias),retain_graph=True),
                   torch.autograd.grad((expected*weight).sum(),(coupling,bias))):
        torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
    assert torch.autograd.gradcheck(lambda k,v:coefficient_partition(k,v,bins,row_chunk=chunk),
                                    (coupling,bias),eps=1e-6)


def test_extreme_scores_have_finite_partition_and_first_derivatives():
    k = torch.tensor([[1000.,-1000.,0.]],requires_grad=True)
    v = torch.tensor([[1000.,-1000.,0.,500.]],requires_grad=True)
    c = torch.tensor([-16.,-.1,.1,16.])
    g = coefficient_partition(k,v,c)
    g.sum().backward()
    assert torch.isfinite(g).all() and torch.isfinite(k.grad).all() and torch.isfinite(v.grad).all()


def fixture(physical=True):
    _,backbone,aux,tokens,cond = case()
    backbone.mask_seen_atoms_training=False
    book=TypedPairCodebook(aux.dictionary,aux.coeff_bins,depth=4)
    model=StructuredPairEnergyRQ(backbone,book.dictionary,book.coeff_bins,
                                  physical_interaction=physical,row_chunk=3).eval()
    return model,book,tokens,cond


def test_full_joint_loss_and_gradients_match_brute_pair_probabilities():
    torch.manual_seed(211)
    model,book,tokens,cond=fixture()
    with torch.no_grad(): model.query.weight.normal_(std=.05)
    targets=typed_pair_targets(book,torch.randn(1,2,2,4),top_k=13,relative_temperature=.1,
        generator=torch.Generator().manual_seed(212))
    output=model(targets.tokens,book,cond)
    actual,_,_=structured_pair_cross_entropy(output,targets,book.coeff_bins)
    score=(output['atom_bias'][...,None]+output['coefficient_bias'][...,None,:]
           +output['coupling'][...,None]*book.coeff_bins)
    joint_logp=score.flatten(-2).log_softmax(-1)
    pair_ids=targets.candidate_atoms*book.coeff_vocab_size+targets.candidate_bins
    expected=-(joint_logp.gather(-1,pair_ids)*targets.pair_weights).sum(-1).mean()
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
    for a,b in zip(torch.autograd.grad(actual,(model.query.weight,model.backbone.classifier.linear.weight),retain_graph=True),
                   torch.autograd.grad(expected,(model.query.weight,model.backbone.classifier.linear.weight))):
        torch.testing.assert_close(a,b,atol=2e-6,rtol=1e-5)
    atoms,_=book.fields(targets.tokens)
    torch.testing.assert_close(output['atom_logits'].softmax(-1),joint_logp.exp().reshape_as(score).sum(-1))
    selected=score.gather(-2,atoms[...,None,None].expand(*atoms.shape,1,book.coeff_vocab_size)).squeeze(-2)
    torch.testing.assert_close(output['coeff_logits'].softmax(-1),selected.softmax(-1))


@pytest.mark.parametrize('physical', [False,True])
@torch.no_grad()
def test_cached_pair_parity_causality_and_one_native_step_per_pair(physical):
    model,book,tokens,cond=fixture(physical)
    model.query.weight.normal_(std=.1)
    full=model(tokens,book,cond)
    changed=tokens.clone();changed[:,-1,-1,-1]=7
    future=model(changed,book,cond)
    for field in full:
        torch.testing.assert_close(full[field],future[field],atol=0,rtol=0)
    changed=tokens.clone();changed[:,0,0,0]=6
    current=model(changed,book,cond)
    torch.testing.assert_close(full['atom_logits'][:,0,0,0],current['atom_logits'][:,0,0,0],atol=0,rtol=0)
    if physical:
        assert not torch.equal(full['coeff_logits'][:,0,0,0],current['coeff_logits'][:,0,0,0])
    generated=book.empty_tokens(1,2,2)
    calls=[];cached=model.backbone.head_transformer.cached_forward
    def count(x): calls.append(1);return cached(x)
    model.backbone.head_transformer.cached_forward=count
    model.init_cache()
    for h in range(2):
        for w in range(2):
            for depth in range(4):
                # A cached pair can score arbitrary current atoms; oracle uses
                # the actual atom only for the conditional coefficient metric.
                generated[:,h,w,2*depth]=tokens[:,h,w,2*depth]
                output=model.cached_pair(generated,book,cond,sample_loc=(h,w,depth))
                for field in full:
                    torch.testing.assert_close(output[field],full[field][:,h,w,depth],atol=2e-6,rtol=1e-5)
                generated[:,h,w,2*depth+1]=tokens[:,h,w,2*depth+1]
    assert len(calls)==16
    sampled=model.sample_joint_sparse(1,book,cond,top_k=7,amp=False)
    a,b=book.fields(sampled)
    assert a.shape==(1,2,2,4) and b.shape==a.shape


def test_zero_query_reproduces_product_distribution_and_has_physical_gradient():
    model,book,tokens,cond=fixture()
    output=model(tokens,book,cond)
    torch.testing.assert_close(output['atom_logits'].softmax(-1),output['atom_bias'].softmax(-1))
    targets=typed_pair_targets(book,torch.randn(1,2,2,4),top_k=5,stochastic=False)
    output=model(targets.tokens,book,cond)
    loss,_,_=structured_pair_cross_entropy(output,targets,book.coeff_bins)
    loss.backward()
    assert model.query.weight.grad.norm()>0
