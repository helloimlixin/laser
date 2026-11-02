from dataclasses import replace

import torch
import pytest

from src.models.coarse_prefix_pair_refiner import CoarsePrefixPairRefiner
from src.models.typed_pair_rq import TypedPairCodebook
from src.models.structured_pair_energy import structured_pair_cross_entropy
from src.training.typed_pair_targets import typed_pair_targets,typed_pair_cross_entropy


def fixture(conditional=False):
    torch.manual_seed(261031)
    book=TypedPairCodebook(torch.randn(4,7),torch.linspace(-1,1,5),depth=4)
    model=CoarsePrefixPairRefiner(4,book.dictionary,book.coeff_bins,height=2,width=2,
        depth=4,hidden=12,layers=2,heads=3,classes=3,dropout=0.,conditional_coefficients=conditional).eval()
    with torch.no_grad(): model.readout.query.weight.normal_(std=.03)
    targets=typed_pair_targets(book,torch.randn(1,2,2,4),top_k=7,stochastic=False)
    return model,book,targets,torch.tensor([2])


@torch.no_grad()
@pytest.mark.parametrize('conditional',[False,True])
def test_current_and_future_maps_never_enter_current_prediction_but_coarse_neighbors_do(conditional):
    model,book,targets,cond=fixture(conditional)
    original=model(targets.tokens,book,cond)
    for depth in range(1,4):
        changed=targets.tokens.clone()
        changed[...,2*depth::2]=(changed[...,2*depth::2]+3)%7
        changed[...,2*depth+1::2]=7+(changed[...,2*depth+1::2]-7+2)%5
        output=model(changed,book,cond)
        # Atom-dependent coefficient logits refer to changed current atoms;
        # the joint distribution parameters themselves exclude current atoms.
        for field in (('atom_logits','hidden') if conditional else ('atom_logits','atom_bias','coefficient_bias','coupling')):
            torch.testing.assert_close(original[field][...,:depth,:],output[field][...,:depth,:],atol=0,rtol=0)
    changed=targets.tokens.clone();changed[:,0,0,0]=(changed[:,0,0,0]+1)%7
    output=model(changed,book,cond)
    assert not torch.equal(original['atom_logits'][:,1,1,0],output['atom_logits'][:,1,1,0])
    local=model(targets.tokens,book,cond,local_only=True)
    local_changed=model(changed,book,cond,local_only=True)
    torch.testing.assert_close(local['atom_logits'][:,1,1,0],local_changed['atom_logits'][:,1,1,0],atol=0,rtol=0)


@pytest.mark.parametrize('conditional',[False,True])
def test_joint_fine_loss_backpropagates_and_sampling_preserves_complete_coarse_pairs(conditional):
    model,book,targets,cond=fixture(conditional)
    fine=replace(targets,candidate_atoms=targets.candidate_atoms[...,1:,:],
        candidate_bins=targets.candidate_bins[...,1:,:],pair_weights=targets.pair_weights[...,1:,:],
        coefficient_weights_given_atom=targets.coefficient_weights_given_atom[...,1:,:],
        residual_energies=targets.residual_energies[...,1:])
    output=model(targets.tokens,book,cond)
    loss,_,_=(typed_pair_cross_entropy(output,fine) if conditional else structured_pair_cross_entropy(output,fine,book.coeff_bins))
    loss.backward()
    parameter=model.coefficient_atom_projection.weight if conditional else model.readout.query.weight
    assert model.input_projection.weight.grad.norm()>0 and parameter.grad.norm()>0
    sampled=model.sample_finer(targets.tokens,book,cond,top_k=7,amp=False)
    assert torch.equal(sampled[...,:2],targets.tokens[...,:2])
    book.fields(sampled)
    sampled=model.sample_finer(targets.tokens,book,cond,top_k=0,amp=False)
    assert torch.equal(sampled[...,:2],targets.tokens[...,:2])
    book.fields(sampled)
