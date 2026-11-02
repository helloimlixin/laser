"""Offline residual refinement with fixed pair IDs and unchanged earlier depths."""
import torch

from src.models.laser_joint_tokenizer import JointAtomCoefficientRQ
from src.training.laser_compact_joint_vocabulary import reconstruct_fixed_codes


@torch.no_grad()
def append_refinement_depth(quantizer,*,scale=.75):
    if scale<=0:raise ValueError('Positive initial refinement scale required')
    return JointAtomCoefficientRQ(quantizer.dictionary,
        torch.cat((quantizer.atom_ids,quantizer.atom_ids[-1:]),0),
        torch.cat((quantizer.coefficients,quantizer.coefficients[-1:]*scale),0)).to(quantizer.dictionary.device)


@torch.no_grad()
def refit_last_depth(quantizer,signals,codes,*,prior_weight=4.):
    if prior_weight<=0:raise ValueError('Positive regularization required')
    x=signals.reshape(-1,signals.shape[-1]).float();codes=codes.reshape(-1,quantizer.depth)
    error=x-reconstruct_fixed_codes(quantizer,codes)
    before=float(error.square().mean());d=quantizer.depth-1
    chosen=codes[:,d];atoms=quantizer.atom_ids[d,chosen];vectors=quantizer.dictionary.T[atoms]
    old=quantizer.coefficients[d].clone()
    optimum=old[chosen]+(error*vectors).sum(-1)/quantizer.norms[atoms]
    frequency=torch.bincount(chosen,minlength=quantizer.vocab_size)
    total=torch.bincount(chosen,weights=optimum,minlength=quantizer.vocab_size)
    proposal=(total+prior_weight*old)/(frequency+prior_weight)
    prior,values=old[1:],proposal[1:]
    same=quantizer.atom_ids[d,1:-1]==quantizer.atom_ids[d,2:]
    midpoint=(prior[:-1]+prior[1:])*.5
    values[:-1]=torch.minimum(values[:-1],torch.where(same,midpoint,torch.inf))
    values[1:]=torch.maximum(values[1:],torch.where(same,midpoint,-torch.inf))
    values.copy_(torch.where(prior<0,values.clamp(max=-1.e-6),values.clamp(min=1.e-6)))
    proposal[0]=0.
    error.sub_(vectors*(proposal[chosen]-old[chosen])[:,None])
    quantizer.coefficients[d].copy_(proposal)
    after=float(error.square().mean())
    assert after<=before+2.e-6*max(1.,before)
    return dict(fixed_assignment_mse_before=before,fixed_assignment_mse_after=after)
