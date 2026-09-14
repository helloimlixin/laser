from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

UPSTREAM=Path(__file__).resolve().parents[1]/'outputs/church-rq-baseline-scratch-20260912/upstream-source'
sys.path.insert(0,str(UPSTREAM))
from src.scaled_atom_sampling import factorized_logits,sample_codes,validate_sampler
from src.imagenet_scaled_stage2 import load_imagenet_config,enable_sdpa
from src.scaled_atom_training import TrainingScaledAtomRQ


def test_factorization_preserves_unfiltered_joint_probabilities_and_one_zero():
    torch.manual_seed(51)
    logits=torch.randn(7,13)
    atoms,bins=factorized_logits(logits,4)
    p=atoms.softmax(-1)
    rebuilt=torch.cat([p[:,:1],(p[:,1:,None]*bins.softmax(-1)).flatten(1)],-1)
    torch.testing.assert_close(rebuilt,logits.softmax(-1),atol=1e-7,rtol=1e-6)
    assert atoms.shape==(7,4) and bins.shape==(7,3,4)


@pytest.mark.parametrize('top_k',[10,None])
def test_joint_sampler_matches_released_samples_and_rng_exactly(top_k):
    from rqvae.models import create_model
    config=load_imagenet_config(UPSTREAM,vocab_size=13)
    config.arch.embed_dim=16;config.arch.input_embed_dim=7;config.arch.block_size=[2,2,2]
    for stack in [config.arch.body,config.arch.head]:
        stack.n_layer=1;stack.block.embed_dim=16;stack.block.n_head=2
    torch.manual_seed(31)
    model,_=create_model(config.arch,ema=False)
    enable_sdpa(model);model.eval()
    quantizer=TrainingScaledAtomRQ(torch.randn(7,3),torch.tensor([-2.,-.5,.5,2.]),depth=2)
    class Aux(torch.nn.Module):
        def __init__(self):super().__init__();self.quantizer=quantizer
        def get_code_emb_with_depth(self,codes):return self.quantizer.get_code_emb_with_depth(codes)
    aux=Aux();labels=torch.tensor([9,22])
    torch.manual_seed(103)
    expected=model.sample(torch.zeros(2,2,2,2,dtype=torch.long),model_aux=aux,cond=labels,
        temperature=.9,top_k=top_k,top_p=.92,amp=False)
    rng=torch.get_rng_state().clone()
    torch.manual_seed(103)
    actual=sample_codes(model,aux,labels,dict(mode='joint',temperature=.9,top_k=top_k,top_p=.92),amp=False)
    assert torch.equal(expected,actual) and torch.equal(rng,torch.get_rng_state())


def test_invalid_sampler_temperature_is_rejected():
    with pytest.raises(ValueError):validate_sampler(dict(mode='joint',temperature=0.,top_k=None,top_p=.92))


def test_atom_specific_sampler_uses_coefficient_count_instead_of_atom_count():
    # Three atoms with two levels each: a mistaken len(levels)==3 either fails
    # the factorization or groups categorical IDs under the wrong atoms.
    class Model:
        block_size=(1,1,1)
        def init_cache(self): pass
        def cached_forward(self,*args,**kwargs):
            return torch.tensor([[-100.,-100.,-100.,-100.,-100.,-100.,100.]])
    aux=SimpleNamespace(quantizer=SimpleNamespace(levels=torch.tensor([[-1.,1.],[-2.,2.],[-3.,3.]])))
    settings=dict(mode='factorized',atom_temperature=1.,atom_top_p=1.,coefficient_temperature=1.)
    codes=sample_codes(Model(),aux,torch.tensor([9]),settings,amp=False)
    assert codes.item()==6
