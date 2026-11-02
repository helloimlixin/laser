"""Draw complete compound tokens from the globally tempered joint distribution.

For beta=1/T, the desired atom marginal is p(a)**beta Z_c(a), where
Z_c(a)=sum_c p(c|a)**beta. Independently tempering the two categoricals omits
Z_c(a). Propose from tempered p(a), accept with Z_c(a)/U, then draw the
coefficient from tempered p(c|a). U=max(1,C**(1-beta)) is a valid bound, so
this samples the full joint exactly without constructing A*C logits.

An optional joint support floor keeps p(a,c)>=rho*max_a p(a)/C. Its cutoff
uses a guaranteed lower bound on the largest joint probability, rather than
claiming to be global top-p. The retained conditional mass stays in Z_c.

The accept/repeat step normalizes each token draw. No images are selected or
discarded, and no future autoregressive events are consulted.
"""
import math

import torch


@torch.no_grad()
def draw_joint_pair(atom_logits, coefficient_logits, *, temperature=1., categorical=None,
                    max_rounds=256,proposal_width=8,support_floor_ratio=0.):
    """coefficient_logits(row_indices, atom_ids) returns conditional rows.

    Batch independent proposals and retain the first accepted one per row.
    This changes launch efficiency, never the target distribution.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Joint temperature must be positive and finite')
    if atom_logits.ndim != 2:
        raise ValueError('Expected a batch of atom logits')
    if not math.isfinite(support_floor_ratio) or not 0 <= support_floor_ratio <= 1:
        raise ValueError('Joint support floor must be finite and between zero and one')
    if categorical is None:
        def categorical(logits, temperature):
            return torch.multinomial((logits.float()/temperature).softmax(-1),1).squeeze(-1)
    batch = atom_logits.shape[0]
    if proposal_width < 1:
        raise ValueError('Proposal batch width must be positive')
    plain_ancestral = temperature==1. and support_floor_ratio==0.
    width = 1 if plain_ancestral else proposal_width
    device = atom_logits.device
    atoms = torch.empty(batch,dtype=torch.long,device=device)
    coefficients = torch.empty_like(atoms)
    pending = torch.arange(batch,device=device)
    atom_probabilities = None if plain_ancestral else (atom_logits.float()/temperature).softmax(-1)
    atom_log_probabilities = atom_logits.float().log_softmax(-1) if support_floor_ratio else None
    floor_base = (atom_log_probabilities.max(-1).values+math.log(support_floor_ratio)
        if support_floor_ratio else None)
    proposals, rounds = 0, 0
    while pending.numel():
        rounds += 1
        if rounds > max_rounds:
            raise RuntimeError('Joint normalization exceeded its draw budget; no biased fallback')
        if plain_ancestral:
            selected = categorical(atom_logits[pending],temperature)
        else:
            selected = torch.multinomial(atom_probabilities[pending],width,replacement=True).flatten()
        rows = pending.repeat_interleave(width)
        logits = coefficient_logits(rows,selected).float()
        proposals += rows.numel()
        if plain_ancestral:
            accepted = torch.ones_like(selected,dtype=torch.bool)
        else:
            beta = 1./temperature
            conditional_logp = logits.log_softmax(-1)
            weights = conditional_logp*beta
            if support_floor_ratio:
                # The cutoff acts on the joint probability, never on either
                # categorical alone. max_a p(a)/C is a guaranteed lower bound
                # on max_(a,c) p(a,c), ensuring a nonempty support for ratio<=1.
                log_threshold = floor_base[rows]-math.log(logits.shape[-1])
                allowed = (atom_log_probabilities[rows,selected,None]
                    +conditional_logp >= log_threshold[:,None])
                weights = weights.masked_fill(~allowed,-torch.inf)
                logits = logits.masked_fill(~allowed,-torch.inf)
            log_z = torch.logsumexp(weights,-1)
            log_upper = max(0.,(1.-beta)*math.log(logits.shape[-1]))
            probability = (log_z-log_upper).clamp(max=0.).exp()
            accepted = torch.rand(len(rows),device=device) < probability
        accepted = accepted.reshape(len(pending),width)
        any_accepted = accepted.any(-1)
        first = accepted.long().argmax(-1)
        positions = (torch.arange(len(pending),device=device)*width+first)[any_accepted]
        chosen = pending[any_accepted]
        if chosen.numel():
            atoms[chosen] = selected[positions]
            coefficients[chosen] = categorical(logits[positions],temperature)
        pending = pending[~any_accepted]
    return atoms,coefficients,dict(proposals=proposals,accepted=batch,max_rounds=rounds,
        proposal_width=width,support_floor_ratio=support_floor_ratio)


@torch.no_grad()
def sample_joint_compound(model,aux,batch_size,cond,*,temperature=1.,amp=True,
                          verify_conditionals=False,support_floor_ratio=0.):
    """Use the native RQ cache and atom-conditioned head, one whole pair at a time."""
    from rqvae.utils.utils import sample_from_logits
    height,width,events = model.block_size
    assert events == 2*aux.sparsity_level
    temperatures = ([float(temperature)]*(events//2) if isinstance(temperature,(int,float))
        else [float(t) for t in temperature])
    if len(temperatures)!=events//2 or any(not math.isfinite(t) or t<=0 for t in temperatures):
        raise ValueError('Expected one positive joint temperature per pair depth')
    device = next(model.parameters()).device
    tokens = torch.zeros(batch_size,height,width,events,dtype=torch.long,device=device)
    tokens[...,1::2] = model.num_atoms + aux.coeff_vocab_size//2
    stats = dict(proposals=0,accepted=0,max_rounds=0,maximum_conditional_kl=0.)

    def categorical(logits,temperature):
        return sample_from_logits(logits,temperature=temperature,top_k=None,top_p=None)

    model.init_cache()
    try:
        for h in range(height):
            for w in range(width):
                for depth in range(events//2):
                    event = depth*2
                    logits = model.cached_forward(tokens[:,:h+1],aux,cond,amp=amp,
                        sample_loc=(h,w,event))[:,:model.num_atoms]
                    hidden = model._pair_hidden

                    def conditional(rows,atoms):
                        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
                            vectors = aux.dictionary.T[atoms][:,None]
                            refined = model.coefficient_hidden(hidden[rows],vectors)
                            result = model.coeff_classifier[depth](refined)[:,0]
                        if (verify_conditionals and h==0 and w==0 and len(rows)>=batch_size
                                and len(rows)%batch_size==0
                                and torch.equal(rows.reshape(batch_size,-1)[:,0],
                                    torch.arange(batch_size,device=device))):
                            positions = torch.arange(0,len(rows),len(rows)//batch_size,device=device)
                            tokens[:,h,w,event] = atoms[positions]
                            expected = model.cached_forward(tokens[:,:h+1],aux,cond,amp=amp,
                                sample_loc=(h,w,event+1))[:,model.num_atoms:]
                            lp,lq = expected.float().log_softmax(-1),result[positions].float().log_softmax(-1)
                            maximum = float((lp.exp()*(lp-lq)).sum(-1).max())
                            if maximum >= 2.e-4:
                                raise AssertionError(f'Conditional head differs from native cache: {maximum}')
                            stats['maximum_conditional_kl'] = max(stats['maximum_conditional_kl'],maximum)
                        return result

                    atoms,coefficients,counts = draw_joint_pair(logits,conditional,
                        temperature=temperatures[depth],categorical=categorical,
                        support_floor_ratio=support_floor_ratio)
                    tokens[:,h,w,event] = atoms
                    tokens[:,h,w,event+1] = coefficients+model.num_atoms
                    stats['proposals'] += counts['proposals']
                    stats['accepted'] += counts['accepted']
                    stats['max_rounds'] = max(stats['max_rounds'],counts['max_rounds'])
                    stats['proposal_width'] = counts['proposal_width']
    finally:
        model.init_cache()
    stats['proposals_per_pair'] = stats['proposals']/stats['accepted']
    return tokens,stats
