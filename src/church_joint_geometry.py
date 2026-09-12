"""Atom-conditional geometry loss and early decay for the compound Church prior."""
import math

import torch
import torch.nn.functional as F

from src import ffhq_v4_archived as archived
from src.church_ffhq_archived import church_config, targets


def joint_candidate_contribution(weights, atom_vectors, coefficient_means):
    """Sum p(a|history) * D[a] * E[c|history,a] over candidate atoms."""
    if weights.shape != coefficient_means.shape or atom_vectors.shape[:-1] != weights.shape:
        raise ValueError('Candidate weights, atom vectors and coefficient means must align')
    return (weights[..., None]*atom_vectors*coefficient_means[..., None]).sum(-2)


class JointGeometryCompoundTransformer(archived.CompoundLaserRQTransformer):
    def classify_head_outputs(self, head_outputs):
        result = super().classify_head_outputs(head_outputs)
        # The causal hidden state is needed to condition coefficients on each
        # candidate atom. No extra parameters or inference path are introduced.
        result['head_outputs'] = head_outputs
        return result


def make_prior(variant='control', config=None, num_atoms=16384, coeff_vocab_size=2048):
    if variant != 'control':
        raise ValueError('This experiment uses the non-looped compound architecture')
    return JointGeometryCompoundTransformer(church_config() if config is None else config,
        num_atoms, coeff_vocab_size, micro_transformer_layers=2, depth_specific_coeff_heads=True)


def early_decay_lr(completed_steps, total_steps, steps_per_epoch, peak=5e-4, knee_epoch=10., knee_lr=5e-5, minimum=1e-6):
    if min(total_steps,steps_per_epoch,knee_epoch) <= 0 or completed_steps < 0:
        raise ValueError('Invalid learning-rate schedule progress')
    if not 0 <= minimum <= knee_lr <= peak:
        raise ValueError('Expected minimum <= knee_lr <= peak')
    knee_steps=min(knee_epoch*steps_per_epoch,total_steps)
    if completed_steps <= knee_steps:
        return knee_lr+(peak-knee_lr)*.5*(1+math.cos(math.pi*completed_steps/knee_steps))
    fraction=min(1.,(completed_steps-knee_steps)/max(total_steps-knee_steps,1))
    return minimum+(knee_lr-minimum)*.5*(1+math.cos(math.pi*fraction))


def conditional_geometry(model, aux, out, atoms, physical, top_k=4):
    atom_logits=out['atom_logits'].float()
    count=min(max(int(top_k),1),atom_logits.shape[-1])
    candidate_logits,candidate_atoms=atom_logits.topk(count,-1)
    gt_logits=atom_logits.gather(-1,atoms[...,None])
    gt_logits=gt_logits.masked_fill((candidate_atoms==atoms[...,None]).any(-1,keepdim=True),-torch.inf)
    weights=torch.cat([candidate_logits,gt_logits],-1).softmax(-1)
    vectors=aux.dictionary.t()[candidate_atoms]
    means=[]
    hidden=out['head_outputs']
    for d in range(atoms.shape[-1]):
        h=hidden[...,d,None,:].expand(*hidden.shape[:-2],count,hidden.shape[-1])
        candidate_coeff_logits=model.coefficient_logits(h,vectors[...,d,:,:],depth_index=d)
        # Reuse the teacher branch for matching candidates, including its
        # dropout realization. Other candidates get their own conditional head.
        candidate_coeff_logits=torch.where((candidate_atoms[...,d,:]==atoms[...,d,None])[...,None],
            out['coeff_logits'][...,d,None,:],candidate_coeff_logits)
        candidate_mean=(candidate_coeff_logits.float().softmax(-1)*aux.coeff_bins).sum(-1)*aux.coeff_scales[d]
        means.append(candidate_mean)
    means=torch.stack(means,-2)
    gt_mean=(out['coeff_logits'].float().softmax(-1)*aux.coeff_bins).sum(-1)*aux.coeff_scales
    means=torch.cat([means,gt_mean[...,None]],-1)
    vectors=torch.cat([vectors,aux.dictionary.t()[atoms][...,None,:]],-2)
    prediction=joint_candidate_contribution(weights,vectors,means)
    target=aux.dictionary.t()[atoms]*physical[...,None]
    pair_mse=F.mse_loss(prediction,target)
    spatial_mse=F.mse_loss(prediction.sum(-2),target.sum(-2))
    return .5*(pair_mse/target.square().mean().detach().clamp_min(1e-6)
        +spatial_mse/target.sum(-2).square().mean().detach().clamp_min(1e-6))


def objective(model, aux, atoms, physical, geometry_weight, stochastic=True):
    with torch.autocast(atoms.device.type,dtype=torch.bfloat16,enabled=atoms.is_cuda):
        packed,probabilities=targets(aux,atoms,physical,stochastic)
        out=model(packed,model_aux=aux,amp=False)
        classification,values=archived.compound_objective(out['atom_logits'],out['coeff_logits'],None,
            atoms,probabilities,None,atom_weight=1.5,geometry_weight=0.,accumulation=1)
        # Match the FP32 classifier/conditional-head arithmetic from teacher
        # forcing. Keep physical expectation arithmetic in FP32 as well.
        with torch.autocast(atoms.device.type,enabled=False):
            geometry=conditional_geometry(model,aux,out,atoms,physical) if geometry_weight>0 else classification.new_zeros(())
        loss=classification+geometry_weight*geometry
    with torch.no_grad():
        entropy=-(probabilities*probabilities.clamp_min(1e-30).log()).sum(-1)
        predicted=out['coeff_logits'].float().softmax(-1)
        mean=(predicted*aux.coeff_bins).sum(-1)*aux.coeff_scales
        sign=predicted[...,aux.coeff_vocab_size//2:].sum(-1)>=.5
        metrics={'loss':loss.detach(),'atom_nll':values['atom_nll'].mean(),
            'coefficient_cross_entropy':values['coeff_cross_entropy'].mean(),
            'coefficient_target_entropy':entropy.mean(),
            'coefficient_kl':(values['coeff_cross_entropy']-entropy).mean(),
            'geometry':geometry,'coefficient_mean_mae':(mean-physical).abs().mean(),
            'sign_accuracy':(sign==(physical>=0)).float().mean()}
    return loss,{k:float(v) for k,v in metrics.items()}
