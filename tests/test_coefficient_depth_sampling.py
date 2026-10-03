"""Policies must preserve native RNG and complete-pair causal handoffs."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from src.coefficient_depth_sampling import depth_settings, sample_compound_depthwise


def sample_from_logits(logits, *, temperature=1., top_p=None, **kwargs):
    return torch.multinomial((logits/temperature).softmax(-1),1).squeeze(-1)


class RecordingSampler(nn.Module):
    interleaved_vector_decoder = True
    block_size = (1, 2, 4)
    num_atoms = 8
    coeff_vocab_size = 5

    def __init__(self, fail=False):
        super().__init__()
        self.register_parameter('anchor', nn.Parameter(torch.zeros(())))
        self.calls = []; self.fail=fail;self.cache=None

    def init_cache(self):
        self.cache=None

    def sample_compound(self, n, aux, *, temperature=1., coeff_temperature=None,
                        coeff_top_p=.92, atom_temperature=1., atom_top_p=1., **kwargs):
        aa=torch.zeros(n,*self.block_size,dtype=torch.long);cc=torch.zeros_like(aa)
        self.cache=[]
        try:
            for w in range(2):
                for d in range(4):
                    # Record the exact physical coefficients in earlier pairs.
                    self.calls.append((w,d,list(self.cache)))
                    a=sample_from_logits(torch.zeros(n,8),temperature=atom_temperature,top_p=atom_top_p)
                    if self.fail and w==0 and d==1:raise RuntimeError('injected failure')
                    c=sample_from_logits(torch.arange(5).float().expand(n,5),temperature=coeff_temperature,top_p=coeff_top_p)
                    aa[:,0,w,d]=a;cc[:,0,w,d]=c
                    self.cache.append((float(aux.coeff_bins[c[0]]*aux.coeff_scales[d]),int(a[0])))
        finally:self.cache=None
        return aa,cc


def setup(fail=False):
    return RecordingSampler(fail).eval(),SimpleNamespace(coeff_scales=torch.tensor([7.6,4.2,2.6,1.7]),coeff_bins=torch.linspace(-3,3,5))


def test_uniform_lists_preserve_samples_and_rng_exactly():
    model,aux=setup();torch.manual_seed(142)
    expected=model.sample_compound(8,aux,coeff_temperature=1.,coeff_top_p=.85)
    expected_rng=torch.get_rng_state().clone();torch.manual_seed(142)
    actual=sample_compound_depthwise(model,8,aux,coeff_temperature=[1.]*4,coeff_top_p=[.85]*4)
    assert all(torch.equal(a,b) for a,b in zip(actual,expected))
    assert torch.equal(torch.get_rng_state(),expected_rng)


def test_correct_depth_filters_and_atom_settings_across_spatial_sites():
    model,aux=setup();observed=[]
    def draw(logits,**settings):
        observed.append((logits.shape[-1],settings.copy()))
        return torch.full((len(logits),), (len(observed)//2)%logits.shape[-1],dtype=torch.long)
    import sys
    module=sys.modules[__name__]
    with patch.object(module,'sample_from_logits',side_effect=draw) as original:
        aa,cc=sample_compound_depthwise(model,2,aux,coeff_top_p=[.8,.9,.95,1.],
                                      coeff_temperature=[.9,1.,1.05,1.1],atom_temperature=.7,atom_top_p=.6)
        assert module.sample_from_logits is original
    for i,(vocab,spec) in enumerate(observed):
        d=(i//2)%4
        assert vocab==(8 if i%2==0 else 5)
        assert spec['temperature']==([.9,1.,1.05,1.1][d] if i%2 else .7)
        assert spec['top_p']==([.8,.9,.95,1.][d] if i%2 else .6)
    assert len(observed)==16 and model.cache is None
    for w,d,history in model.calls:
        assert len(history)==w*4+d
        for j,(physical,atom) in enumerate(history):
            assert physical==float(aux.coeff_bins[cc[0,0,j//4,j%4]]*aux.coeff_scales[j%4])
            assert atom==int(aa[0,0,j//4,j%4])


def test_sampling_failure_restores_helper_and_caches():
    model,aux=setup(fail=True);original=sample_from_logits
    with pytest.raises(RuntimeError,match='injected'):
        sample_compound_depthwise(model,2,aux,coeff_top_p=[.85,.9,.95,1.])
    assert sample_from_logits is original and model.cache is None
    assert '__laser_coefficient_depth_sampling_active__' not in globals()


@pytest.mark.parametrize('value,probability', [([1.,1.],False),(True,False),([1.,0.,1.,1.],False),
    ([1.,float('nan'),1.,1.],False),([1.,1.01,1.,1.],True),('0.9',False),([None]*4,False)])
def test_invalid_policy_is_rejected_before_sampling(value,probability):
    with pytest.raises(ValueError):depth_settings(value,4,name='setting',probability=probability)


def test_no_filter_and_scalar_policies_are_supported():
    assert depth_settings(None,4,name='p',probability=True)==(None,)*4
    assert depth_settings(.95,4,name='p',probability=True)==(.95,)*4
