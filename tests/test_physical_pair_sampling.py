import pytest
import torch

from tests.test_physical_pair_scalar_prior import case
from src.physical_pair_sampling import (PairSamplingPolicy,calibrate_geometry,
    candidate_coefficient_logits,geometry_penalties,nucleus_probabilities,sample_physical_pairs,prior_atom_proposals)


@torch.no_grad()
def test_ancestral_control_reproduces_native_tokens_exactly():
    model,aux,_,cond=case()
    torch.manual_seed(714)
    expected=model.sample_sparse(1,aux,cond,amp=False,atom_temperature=.9,atom_top_p=.9,
        coeff_temperature=1.,coeff_top_p=.85)
    torch.manual_seed(714)
    actual=sample_physical_pairs(model,1,aux,cond,amp=False,
        policy=PairSamplingPolicy(mode='ancestral'))
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


@pytest.mark.parametrize('location',[(0,0,1),(0,0,3),(0,1,1),(1,0,5)])
@torch.no_grad()
def test_branch_logits_match_full_teacher_forcing_and_restore_cache(location):
    model,aux,tokens,cond=case()
    tokens=tokens.repeat(2,1,1,1);cond=cond.repeat(2)
    h,w,event=location
    model.init_cache()
    for row in range(2):
        for col in range(2):
            for previous_event in range(8):
                if (row,col,previous_event)>=(h,w,event):break
                model.cached_forward(tokens[:, :row+1],aux,cond,sample_loc=(row,col,previous_event))
            if (row,col)>=(h,w):break
        if row>=h:break
    caches=[block._cache['past_kv'] for block in model.head_transformer.blocks]
    body=[block._cache['past_kv'] for block in model.body_transformer.blocks]
    spatial=model._cache['spatial_ctx_hw']
    candidates=torch.tensor([[4,5,6],[6,5,4]])
    logits=candidate_coefficient_logits(model,aux,tokens,cond,candidates,location,amp=False)
    for index in range(3):
        truth=tokens.clone();truth[:,h,w,event-1]=candidates[:,index]
        expected=model(truth,aux,cond)['coeff_logits'][:,h,w,event//2]
        torch.testing.assert_close(logits[:,index],expected,rtol=3e-6,atol=8e-7)
    assert all(block._cache['past_kv'] is cache for block,cache in zip(model.head_transformer.blocks,caches))
    assert all(block._cache['past_kv'] is cache for block,cache in zip(model.body_transformer.blocks,body))
    assert model._cache['spatial_ctx_hw'] is spatial


@pytest.mark.parametrize('proposal',['top','prior'])
@torch.no_grad()
def test_joint_sampling_has_unique_support_and_leaves_weights_unchanged(proposal):
    model,aux,truth,cond=case()
    before={key:value.clone() for key,value in model.state_dict().items()}
    diagnostics={}
    sampled=sample_physical_pairs(model,1,aux,cond,amp=False,
        policy=PairSamplingPolicy(candidate_atoms=3,atom_proposal=proposal),diagnostics=diagnostics)
    assert (sampled[...,0::2].sort(-1).values.diff(dim=-1)>0).all()
    assert ((sampled[...,1::2]>=7)&(sampled[...,1::2]<12)).all()
    key='drawn_pool_coverage_mean' if proposal=='prior' else 'proposal_mass_mean'
    assert 0<diagnostics[key]<=1
    for key,value in model.state_dict().items():torch.testing.assert_close(value,before[key],rtol=0,atol=0)
    assert model._cache['spatial_ctx_hw'] is None
    assert all(block._cache['past_kv'] is None for block in model.head_transformer.blocks)


def test_geometry_matches_direct_reconstruction_and_detects_degenerate_support():
    model,aux,tokens,_=case()
    # Construct exact cancellation with a repeated-direction candidate.
    aux.dictionary[:,1]=aux.dictionary[:,0]
    aux.dictionary=torch.nn.functional.normalize(aux.dictionary,dim=0)
    stats=calibrate_geometry(aux.dictionary,tokens[...,0::2],
        aux.coeff_bins[tokens[...,1::2]-7]*aux.coeff_scales)
    bounds=dict(norm_median=.5,norm_upper=2.,cancellation_upper=2.,gram_minimum_lower=.1)
    stats['depths']=[bounds]*4
    candidates=torch.tensor([[1,2]])
    penalty=geometry_penalties(aux,torch.tensor([[0]]),torch.tensor([[1.]]),candidates,1,stats)
    assert torch.isfinite(penalty).all()
    # coefficient=-1 cancels the first vector and must receive a large penalty.
    assert penalty[0,0,1]>penalty[0,0,3]
    assert penalty.shape==(1,2,5)


def test_joint_geometry_requires_training_calibration():
    model,aux,_,cond=case()
    with pytest.raises(ValueError,match='training-only'):
        sample_physical_pairs(model,1,aux,cond,amp=False,
            policy=PairSamplingPolicy(geometry_weight=1.))


def test_nucleus_rejects_all_masked_rows_and_keeps_crossing_token():
    with pytest.raises(ValueError):nucleus_probabilities(torch.full((1,4),-torch.inf))
    probability=nucleus_probabilities(torch.tensor([[.6,.3,.1]]).log(),top_p=.7)
    torch.testing.assert_close(probability,torch.tensor([[2/3,1/3,0.]]))


def test_full_prior_proposals_preserve_mass_and_combine_duplicates():
    torch.manual_seed(16)
    probability=torch.tensor([[.7,.2,.1]]).expand(5000,-1)
    atoms,weight,_=prior_atom_proposals(probability,16)
    torch.testing.assert_close(weight.sum(-1),torch.ones(5000))
    marginal=torch.zeros(5000,3).scatter_add_(1,atoms,weight).mean(0)
    torch.testing.assert_close(marginal,probability[0],rtol=0,atol=.01)
    # Repeated draws add prior mass once per unique atom, not p(a)^2.
    for row in range(5):
        assert atoms[row,weight[row]>0].unique().numel()==int((weight[row]>0).sum())


def test_unfiltered_prior_mixture_matches_exact_joint_atom_coefficient_law():
    torch.manual_seed(513)
    atom_probability=torch.tensor([[.8,.2]]).expand(5000,-1)
    conditional=torch.tensor([[.9,.1],[.2,.8]])
    atoms,weight,_=prior_atom_proposals(atom_probability,16)
    pair_probability=weight[...,None]*conditional[atoms]
    indices=(2*atoms[...,None]+torch.arange(2)).flatten(1)
    law=torch.zeros(5000,4).scatter_add_(1,indices,pair_probability.flatten(1)).mean(0)
    torch.testing.assert_close(law,torch.tensor([.72,.08,.04,.16]),rtol=0,atol=.01)
