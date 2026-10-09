"""Experimental joint sampling for the trained physical scalar-pair prior.

The native prior already models p(atom) p(coefficient | atom). This sampler
changes its decoding distribution: it evaluates coefficients for a finite
atom proposal mixture and applies one nucleus to complete pairs. Default
proposals come from the full atom prior; top-atom truncation is an ablation. Soft
geometry penalties use training-code statistics. There is no unknown image
residual to minimize, no coefficient refit, and no spatial smoothing.
"""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class PairSamplingPolicy:
    mode: str = 'joint'
    atom_temperature: float = .9
    atom_top_p: float = .9
    coefficient_temperatures: tuple = (1., 1., 1., 1.)
    coefficient_top_p: float = .85
    candidate_atoms: int = 32
    atom_proposal: str = 'prior'
    joint_top_p: float = .9
    geometry_weight: float = 0.


def nucleus_probabilities(logits, temperature=1., top_p=1.):
    if not math.isfinite(temperature) or temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError('Invalid sampling temperature or nucleus mass')
    if not torch.isfinite(logits).any(-1).all() or torch.isnan(logits).any():
        raise ValueError('Every categorical row needs finite logits')
    probabilities = (logits.float() / temperature).softmax(-1)
    if top_p < 1:
        sorted_p, indices = probabilities.sort(-1, descending=True)
        remove = sorted_p.cumsum(-1) >= top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        probabilities = probabilities.masked_fill(remove.scatter(-1, indices, remove), 0)
        probabilities = probabilities / probabilities.sum(-1, keepdim=True)
    return probabilities


def _expanded_kv(value, *, batch, candidates, heads):
    if value is None:
        return None
    # Native cache merges batch and attention heads. Expand batch, never heads.
    shape = value.shape
    if shape[0] != 2 or shape[1] != batch * heads:
        raise ValueError('Unexpected physical scalar attention cache layout')
    return value.reshape(2, batch, heads, *shape[2:]).repeat_interleave(
        candidates, dim=1).reshape(2, batch*candidates*heads, *shape[2:])


def prior_atom_proposals(probabilities, count, generator=None):
    """Monte Carlo mixture over the full atom prior, with duplicates combined.

    Samples are drawn from q(a)=p(a). Each draw has importance weight1/count;
    multiplying by p(a) again would incorrectly square the atom prior. The
    empirical mixture is unbiased before the optional joint nucleus/penalties.
    Keep the first copy of each atom, carrying its count/count_total weight.
    """
    atoms=torch.multinomial(probabilities,count,replacement=True,generator=generator)
    equal=atoms[:,:,None]==atoms[:,None,:]
    duplicate=(equal & torch.ones(count,count,device=atoms.device,dtype=torch.bool).tril(-1)).any(-1)
    first=~duplicate
    weight=equal.sum(-1).float()/count
    weight=weight.masked_fill(duplicate,0)
    coverage=(probabilities.gather(1,atoms)*first).sum(-1)
    return atoms,weight,coverage


@torch.no_grad()
def candidate_coefficient_logits(model, aux, tokens, cond, atom_ids, location, amp=True):
    """Evaluate candidate atoms without consuming the selected decoding cache."""
    h, w, event = location
    if event % 2 != 1 or atom_ids.ndim != 2 or len(atom_ids) != len(tokens):
        raise ValueError('Candidate lookahead must follow an atom event')
    batch, count = atom_ids.shape
    blocks = model.head_transformer.blocks
    spatial = model._cache['spatial_ctx_hw']
    caches = [block._cache['past_kv'] for block in blocks]
    if spatial is None:
        raise ValueError('Consume the atom event before coefficient lookahead')
    branches = tokens.repeat_interleave(count, dim=0)
    branches[:, h, w, event-1] = atom_ids.reshape(-1)
    branch_cond = None if cond is None else cond.repeat_interleave(count, dim=0)
    try:
        model._cache['spatial_ctx_hw'] = spatial.repeat_interleave(count, dim=0)
        for block, cache in zip(blocks, caches):
            block._cache['past_kv'] = _expanded_kv(cache, batch=batch,
                candidates=count, heads=block.attn.n_head)
        logits = model.cached_forward(branches[:, :h+1], aux, branch_cond,
            sample_loc=location, amp=amp)
        return logits[:, model.num_atoms:].reshape(batch, count, -1)
    finally:
        model._cache['spatial_ctx_hw'] = spatial
        for block, cache in zip(blocks, caches):
            block._cache['past_kv'] = cache


@torch.no_grad()
def calibrate_geometry(dictionary, atoms, physical_coefficients):
    """Summarize real training prefixes, including their signed cancellations."""
    atoms = atoms.reshape(-1, atoms.shape[-1]).long()
    coefficients = physical_coefficients.reshape_as(atoms).float()
    vectors = dictionary.T[atoms].float()
    gram = vectors @ vectors.transpose(-2, -1)
    prefixes = (vectors * coefficients[..., None]).cumsum(-2)
    norms = prefixes.norm(dim=-1)
    energy = coefficients.square().cumsum(-1)
    ratio = energy / norms.square().clamp_min(1e-8)
    depths = []
    for depth in range(atoms.shape[-1]):
        minimum = torch.linalg.eigvalsh(gram[:, :depth+1, :depth+1])[:, 0]
        depths.append(dict(
            norm_median=float(norms[:, depth].median()),
            norm_upper=float(norms[:, depth].quantile(.999)),
            cancellation_upper=float(ratio[:, depth].quantile(.999)),
            gram_minimum_lower=float(minimum.quantile(.001))))
    return dict(training_only=True, latent_sites=len(atoms),
        quantile_upper=.999, quantile_lower=.001, depths=depths)


def geometry_penalties(aux, previous_atoms, previous_coefficients, candidates, depth, calibration):
    """Compute candidate effects using c^T G c, without large channel grids."""
    vectors = aux.dictionary.T[candidates].float()
    batch, count, channels = vectors.shape
    if depth:
        old_vectors = aux.dictionary.T[previous_atoms.long()].float()
        prefix = (old_vectors * previous_coefficients[..., None]).sum(1)
        energy = previous_coefficients.square().sum(1)
        supports = torch.cat((old_vectors[:, None].expand(-1, count, -1, -1),
                              vectors[:, :, None]), dim=2)
    else:
        prefix = vectors.new_zeros(batch, channels)
        energy = vectors.new_zeros(batch)
        supports = vectors[:, :, None]
    gram = supports @ supports.transpose(-2, -1)
    minimum = torch.linalg.eigvalsh(gram)[..., 0].clamp_min(0)
    coefficients = (aux.coeff_bins.float() * aux.coeff_scales[depth].float())[None, None]
    norm2 = (prefix.square().sum(-1)[:, None, None]
             + 2 * (vectors*prefix[:, None]).sum(-1)[..., None] * coefficients
             + vectors.square().sum(-1)[..., None] * coefficients.square()).clamp_min(1e-8)
    ratio = (energy[:, None, None]+coefficients.square()) / norm2
    bounds = calibration['depths'][depth]
    width = max(bounds['norm_upper']-bounds['norm_median'], .1)
    norm_excess = ((norm2.sqrt()-bounds['norm_upper']) / width).clamp_min(0)
    cancellation_excess = (ratio/max(bounds['cancellation_upper'], 1.)-1).clamp_min(0)
    threshold = max(bounds['gram_minimum_lower'], 1e-6)
    conditioning_excess = ((threshold-minimum)/threshold).clamp_min(0)
    return norm_excess.square()+cancellation_excess.square()+conditioning_excess[..., None].square()


@torch.no_grad()
def sample_physical_pairs(model, batch_size, aux, cond=None, *, policy=None,
                          calibration=None, amp=True, generator=None, diagnostics=None):
    policy = policy or PairSamplingPolicy()
    if policy.mode not in ('ancestral', 'joint') or policy.atom_proposal not in ('top','prior'):
        raise ValueError('Expected ancestral or joint sampling')
    if model.decoder_type != 'shared_interleaved_physical_pairs_scalar_ce_v1' or model.training:
        raise ValueError('A trained physical scalar model in eval mode is required')
    height, width, events = model.block_size
    depths = events//2
    if (len(policy.coefficient_temperatures) != depths or policy.candidate_atoms < 1
            or policy.geometry_weight < 0 or not math.isfinite(policy.geometry_weight)):
        raise ValueError('Invalid pair policy or coefficient depth temperatures')
    if policy.geometry_weight and (calibration is None or not calibration.get('training_only')):
        raise ValueError('Geometry sampling requires training-only calibration')
    for temperature in (policy.atom_temperature, *policy.coefficient_temperatures):
        if not math.isfinite(temperature) or temperature <= 0:raise ValueError('Invalid temperature')
    for mass in (policy.atom_top_p,policy.coefficient_top_p,policy.joint_top_p):
        if not 0 < mass <= 1:raise ValueError('Invalid nucleus mass')
    device = next(model.parameters()).device
    tokens = torch.zeros(batch_size,height,width,events,device=device,dtype=torch.long)
    tokens[..., 1::2] = model.num_atoms + aux.coeff_vocab_size//2
    retained, chosen_energy = [], []
    model.init_cache()
    try:
        for h in range(height):
            for w in range(width):
                for depth in range(depths):
                    event=2*depth
                    atom_logits=model.cached_forward(tokens[:, :h+1],aux,cond,amp=amp,
                        sample_loc=(h,w,event))[:, :model.num_atoms]
                    atom_prob=nucleus_probabilities(atom_logits,policy.atom_temperature,policy.atom_top_p)
                    if policy.mode=='ancestral':
                        atom=torch.multinomial(atom_prob,1,generator=generator).squeeze(-1)
                        tokens[:,h,w,event]=atom
                        coefficient_logits=model.cached_forward(tokens[:, :h+1],aux,cond,amp=amp,
                            sample_loc=(h,w,event+1))[:,model.num_atoms:]
                        probabilities=nucleus_probabilities(coefficient_logits,
                            policy.coefficient_temperatures[depth],policy.coefficient_top_p)
                        coefficient=torch.multinomial(probabilities,1,generator=generator).squeeze(-1)
                    else:
                        if policy.atom_proposal=='prior':
                            candidates,probability,coverage=prior_atom_proposals(atom_prob,
                                policy.candidate_atoms,generator)
                            retained.append(coverage)
                        else:
                            probability,candidates=atom_prob.topk(min(policy.candidate_atoms,model.num_atoms),dim=-1)
                            retained.append(probability.sum(-1))
                        coefficient_logits=candidate_coefficient_logits(model,aux,tokens,cond,
                            candidates,(h,w,event+1),amp=amp)
                        log_coeff=(coefficient_logits.float()/policy.coefficient_temperatures[depth]).log_softmax(-1)
                        joint=probability.log()[...,None]+log_coeff
                        if policy.geometry_weight:
                            previous=tokens[:,h,w,:event]
                            values=aux.coeff_bins[previous[:,1::2]-model.num_atoms]*aux.coeff_scales[:depth]
                            penalty=geometry_penalties(aux,previous[:,0::2],values,
                                candidates,depth,calibration)
                            joint=joint-policy.geometry_weight*penalty
                        probabilities=nucleus_probabilities(joint.flatten(1),top_p=policy.joint_top_p)
                        selected=torch.multinomial(probabilities,1,generator=generator).squeeze(-1)
                        atom=candidates.gather(1,(selected//aux.coeff_vocab_size)[:,None]).squeeze(-1)
                        coefficient=selected%aux.coeff_vocab_size
                        if policy.geometry_weight:
                            chosen_energy.append(penalty.flatten(1).gather(1,selected[:,None]).squeeze(-1))
                        tokens[:,h,w,event]=atom
                        # Advance the unbranched cache with the selected atom.
                        model.cached_forward(tokens[:, :h+1],aux,cond,amp=amp,sample_loc=(h,w,event+1))
                    tokens[:,h,w,event+1]=coefficient+model.num_atoms
    finally:
        model.init_cache()
    if diagnostics is not None:
        diagnostics.update(spatial_smoothing=False,coefficient_refit=False)
        if policy.mode=='joint':
            diagnostics.update(atom_proposal=policy.atom_proposal,
                candidate_pool=('draws from full native atom nucleus; duplicate mass combined'
                    if policy.atom_proposal=='prior' else 'highest-probability atoms after native atom nucleus'),
                joint_law=('empirical prior mixture; one nucleus over atom/coefficient pairs'
                    if policy.atom_proposal=='prior' else 'truncated proposal pool; one nucleus over atom/coefficient pairs'))
        else:diagnostics['sampling_law']='native conditional atom/coefficient sampling with per-field nuclei'
        if retained:
            mass=torch.cat(retained).float()
            key='drawn_pool_coverage' if policy.atom_proposal=='prior' else 'proposal_mass'
            diagnostics.update({key+'_mean':float(mass.mean()),
                key+'_p10':float(mass.quantile(.1)),key+'_min':float(mass.min())})
        if chosen_energy:diagnostics['selected_geometry_penalty_mean']=float(torch.cat(chosen_energy).mean())
    return tokens
