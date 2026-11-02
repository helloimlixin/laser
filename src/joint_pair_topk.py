"""Certified global top-k over complete atom/coefficient contributions.

Each state is the physical vector bins[c] * scales[depth] * dictionary[:, a].
Its probability is p(a | history) p(c | a, history). Atom marginals are only
used as upper bounds while searching: p(a,c) <= p(a). No independent atom or
coefficient truncation is applied. Temperature and nucleus act on whole pairs.
"""
import math
from numbers import Integral

import torch


@torch.no_grad()
def joint_pair_topk(atom_logits, coefficient_logits, *, top_k, candidate_chunk=16384,
                    max_candidate_atoms=None, max_candidate_batch=32768):
    """Return globally highest joint log probabilities and flattened pair IDs.

    ``coefficient_logits(rows, atom_ids)`` returns [rows, candidates, bins].
    Stop only when every unseen atom's probability upper bound is below the
    smallest retained pair probability. An optional compute limit raises if
    certification is impossible; it never silently returns a biased shortlist.
    """
    if atom_logits.ndim != 2 or atom_logits.shape[0] < 1:
        raise ValueError('expected nonempty atom logits [batch, atoms]')
    for name, value in [('top_k', top_k), ('candidate_chunk', candidate_chunk)]:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if torch.isnan(atom_logits).any() or torch.isposinf(atom_logits).any() or not torch.isfinite(atom_logits).any(-1).all():
        raise ValueError('each atom row needs a finite, valid distribution')
    batch, atoms = atom_logits.shape
    if max_candidate_batch < batch:
        raise ValueError('candidate batch budget must cover at least one atom per image')
    limit = atoms if max_candidate_atoms is None else int(max_candidate_atoms)
    if limit < 1:
        raise ValueError('candidate limit must be positive')
    limit = min(limit, atoms)
    logpa = atom_logits.float().log_softmax(-1)
    sorted_logpa, order = logpa.sort(dim=-1, descending=True, stable=True)
    rows = torch.arange(batch, device=atom_logits.device)
    scores = ids = None
    evaluated = torch.zeros(batch, device=atom_logits.device, dtype=torch.long)
    active = rows
    bins = None
    cursor = 0
    while active.numel():
        chunk = min(candidate_chunk, max(1, max_candidate_batch // active.numel()))
        stop = min(cursor + chunk, limit)
        if stop <= cursor:
            raise RuntimeError('global joint top-k was not certified within the candidate budget')
        candidates = order[active, cursor:stop]
        conditional = coefficient_logits(active, candidates).float()
        if conditional.ndim != 3 or conditional.shape[:2] != candidates.shape:
            raise ValueError('conditional callback must return [rows, candidates, bins]')
        if bins is None:
            bins = conditional.shape[-1]
            count = min(top_k, atoms * bins)
            scores = torch.full((batch, count), -torch.inf, device=atom_logits.device)
            ids = torch.full((batch, count), -1, dtype=torch.long, device=atom_logits.device)
        elif conditional.shape[-1] != bins:
            raise ValueError('coefficient vocabulary changed during joint search')
        if torch.isnan(conditional).any() or torch.isposinf(conditional).any() or not torch.isfinite(conditional).any(-1).all():
            raise ValueError('each conditional row needs a finite, valid distribution')
        joint = sorted_logpa[active, cursor:stop, None] + conditional.log_softmax(-1)
        local_scores, local_indices = joint.flatten(1).topk(min(count, joint.shape[1] * bins), -1)
        local_atoms = candidates.gather(1, local_indices // bins)
        local_ids = local_atoms * bins + local_indices % bins
        if cursor == 0:
            # The first candidates already arrive sorted. In particular, full
            # enumeration needs no second top-k over K values plus -inf pads.
            size = local_scores.shape[-1]
            scores[active, :size] = local_scores
            ids[active, :size] = local_ids
        else:
            merged_scores = torch.cat((scores[active], local_scores), -1)
            merged_ids = torch.cat((ids[active], local_ids), -1)
            best, indices = merged_scores.topk(count, -1)
            scores[active] = best
            ids[active] = merged_ids.gather(1, indices)
        evaluated[active] = stop
        cursor = stop
        if cursor == atoms:
            break
        # This is a probability bound, not a marginal-atom top-k cutoff.
        certified = sorted_logpa[active, cursor] < scores[active, -1]
        active = active[~certified]
    return scores, ids, dict(evaluated_atoms=evaluated, coefficient_bins=bins,
                            retained_mass=scores.logsumexp(-1).exp(), certified=True)


@torch.no_grad()
def sample_joint_pair_topk(atom_logits, coefficient_logits, *, top_k, temperature=1.,
                           top_p=1., candidate_chunk=16384, generator=None,
                           max_candidate_atoms=None):
    if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('whole-pair temperature must be positive and finite')
    if top_p is not None and (isinstance(top_p, bool) or not math.isfinite(top_p) or not 0 < top_p <= 1):
        raise ValueError('whole-pair top-p must lie in (0, 1]')
    scores, ids, stats = joint_pair_topk(atom_logits, coefficient_logits, top_k=top_k,
        candidate_chunk=candidate_chunk, max_candidate_atoms=max_candidate_atoms)
    logits = scores / temperature
    if top_p is not None and top_p < 1:
        probabilities = logits.softmax(-1)
        remove = probabilities.cumsum(-1) - probabilities >= top_p
        logits = logits.masked_fill(remove, -torch.inf)
    selected = torch.multinomial(logits.softmax(-1), 1, generator=generator)
    pair = ids.gather(1, selected).squeeze(-1)
    bins = stats['coefficient_bins']
    return pair // bins, pair % bins, stats
