import pytest
import torch

from src.training.rq_contrastive_sampling import contrast_logits, ContrastiveCachedPrior


def test_zero_strength_retains_exact_native_logits_and_dtype():
    expert=torch.randn(4,8,dtype=torch.float16)
    assert contrast_logits(expert,torch.randn_like(expert),0,3) is expert


def test_guidance_is_log_probability_ratio_up_to_a_row_constant():
    expert=torch.tensor([[2.,1.,-3.]])
    amateur=torch.tensor([[1.,3.,-4.]])
    actual=contrast_logits(expert,amateur,.5,3).softmax(-1)
    expected=(1.5*expert.log_softmax(-1)-.5*amateur.log_softmax(-1)).softmax(-1)
    torch.testing.assert_close(actual,expected)


def test_amateur_cannot_promote_a_token_outside_expert_support():
    expert=torch.tensor([[3.,2.,1.,-10.]])
    amateur=torch.tensor([[0.,0.,0.,-1000.]])
    logits=contrast_logits(expert,amateur,1.,2)
    assert torch.isneginf(logits[0,2:]).all()
    torch.testing.assert_close(logits[0,:2],torch.tensor([6.,4.]))


@pytest.mark.parametrize('strength,k',[(-1,2),(float('nan'),2),(1,0),(1,5)])
def test_invalid_parameters_fail(strength,k):
    with pytest.raises(ValueError):contrast_logits(torch.zeros(1,4),torch.zeros(1,4),strength,k)


class ToyPrior:
    block_size=(1,1,2)
    vocab_size=[3,3]

    def __init__(self,offset):self.offset=offset;self.calls=[];self.resets=0

    def init_cache(self):self.resets+=1

    def cached_forward(self,x,**kw):
        self.calls.append((x.clone(),dict(kw)))
        return torch.tensor([[3.,2.,1.]])+self.offset


def test_both_caches_use_identical_prefixes_and_wrapper_can_be_removed():
    good,bad=ToyPrior(0),ToyPrior(1)
    native=good.cached_forward
    wrapper=ContrastiveCachedPrior(good,bad,top_k=2)
    wrapper.strength=.5
    x=torch.ones(1,1,1,2,dtype=torch.long)
    good.init_cache();good.cached_forward(x,sample_loc=(0,0,1))
    assert good.resets==bad.resets==1
    assert torch.equal(good.calls[0][0],bad.calls[0][0])
    assert good.calls[0][1]==bad.calls[0][1]
    wrapper.restore()
    assert good.cached_forward==native
