from pathlib import Path
import sys
import pytest
import torch

UPSTREAM=Path(__file__).resolve().parents[1]/'outputs/church-rq-baseline-scratch-20260912/upstream-source'
sys.path.insert(0,str(UPSTREAM))
from src.compact_rq_training import DepthAdaptiveScaledAtomRQ


def quantizer(levels=2):
    torch.manual_seed(228)
    return DepthAdaptiveScaledAtomRQ(torch.randn(7,6),
        ((torch.rand(4,6,levels)+.5)*torch.tensor([-1.]*(levels//2)+[1.]*(levels//2))).sort(-1).values)


@pytest.mark.parametrize('levels',[2,4])
def test_depth_compact_matches_explicit_unshared_rq_books(levels):
    from rqvae.models.rqvae.quantizations import RQBottleneck
    q=quantizer(levels)
    original=RQBottleneck([2,2,7],[2,2,4],q.vocab_size,
        shared_codebook=False,restart_unused_codes=False).eval()
    with torch.no_grad():
        for target,source in zip(original.codebooks,q.codebooks):
            target.weight[:-1].copy_(source.expanded_codebook())
    x=torch.randn(3,2,2,7)
    before=x.clone()
    expected,commitment,codes=original(x)
    actual,actual_commitment,actual_codes=q(x)
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(actual_commitment,commitment)
    assert torch.equal(codes,actual_codes)
    torch.manual_seed(893)
    p,c=original.get_soft_codes(x,temp=.125,stochastic=True)
    torch.manual_seed(893)
    p2,c2=q.get_soft_codes(x,temp=.125,stochastic=True)
    torch.testing.assert_close(p2,p,atol=4e-6,rtol=5e-5)
    assert torch.equal(c,c2) and torch.equal(x,before)
    assert not torch.equal(q.embed(torch.ones(1,4,dtype=torch.long))[:,0],
                           q.embed(torch.ones(1,4,dtype=torch.long))[:,1])


@pytest.mark.parametrize('levels',[2,4])
def test_depth_compact_cached_sampling_matches_full_causal_forward(levels):
    from rqvae.models import create_model
    from src.imagenet_scaled_stage2 import load_imagenet_config
    q=quantizer(levels)
    config=load_imagenet_config(UPSTREAM,vocab_size=q.vocab_size)
    config.arch.embed_dim=16;config.arch.input_embed_dim=7;config.arch.block_size=[2,2,4]
    for stack in [config.arch.body,config.arch.head]:
        stack.n_layer=1;stack.block.embed_dim=16;stack.block.n_head=2
    model,_=create_model(config.arch,ema=False)
    model.eval()
    labels=torch.tensor([9,22])
    codes=torch.zeros(2,2,2,4,dtype=torch.long)
    torch.manual_seed(1023)
    cached=model.sample(codes,model_aux=q,cond=labels,temperature=.9,top_k=None,top_p=.92,amp=False,cached=True)
    torch.manual_seed(1023)
    full=model.sample(codes,model_aux=q,cond=labels,temperature=.9,top_k=None,top_p=.92,amp=False,cached=False)
    assert torch.equal(cached,full)
