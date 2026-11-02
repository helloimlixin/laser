"""Residual-consistent whole-pair teachers and joint CE for typed token streams."""
from dataclasses import dataclass
import math

import torch


@torch.no_grad()
def best_joint_pairs(residual, dictionary, bins, top_k=64):
    """Exact global K nearest c[b]D[a] pairs without materializing A*B.

    A global top-K pair must belong to one of the K best atoms (ranked by its
    best coefficient), and be among that atom's K nearest coefficients.
    """
    if residual.ndim != 2 or dictionary.ndim != 2 or residual.shape[-1] != dictionary.shape[0]:
        raise ValueError('expected residual [rows,C] and dictionary [C,A]')
    if bins.ndim != 1 or not (bins.diff()>0).all() or top_k<1:
        raise ValueError('bins must be sorted and top_k positive')
    norms = dictionary.square().sum(0)
    if (norms<=0).any():
        raise ValueError('dictionary atoms must be nonzero')
    correlation = residual.float()@dictionary.float()
    optimum = correlation/norms
    right = torch.searchsorted(bins, optimum.contiguous()).clamp(max=len(bins)-1)
    left = (right-1).clamp(min=0)
    c0, c1 = bins[left], bins[right]
    s0 = 2*correlation*c0-norms*c0.square()
    s1 = 2*correlation*c1-norms*c1.square()
    best = torch.maximum(s0, s1)
    k = min(int(top_k), dictionary.shape[1]*len(bins))
    # Start with only the strongest atom branches, then expand until every
    # unseen atom's best possible score is below the current joint Kth score.
    # This matters for dense scalar grids: many top pairs share one atom.
    ranked_scores, ranked_atoms = best.topk(min(k+1, dictionary.shape[1]), -1)
    count = min(k, len(bins))
    limit = min(k, dictionary.shape[1])
    branches = math.ceil(k/count)
    rows = torch.arange(len(residual), device=residual.device)
    output_a = torch.empty(len(residual), k, device=residual.device, dtype=torch.long)
    output_b = torch.empty_like(output_a)
    output_scores = correlation.new_empty(len(residual), k)
    while rows.numel():
        atom_ids = ranked_atoms[rows, :branches]
        corr = correlation[rows[:, None], atom_ids]
        opt = optimum[rows[:, None], atom_ids]
        if count == len(bins):
            coefficient_ids = torch.arange(count, device=bins.device).expand(*atom_ids.shape, count)
        else:
            insertion = torch.searchsorted(bins, opt.contiguous())
            candidates = insertion[..., None]+torch.arange(-count, count, device=bins.device)
            valid = (candidates>=0)&(candidates<len(bins))
            safe = candidates.clamp(0, len(bins)-1)
            distance = (bins[safe]-opt[..., None]).square().masked_fill(~valid, torch.inf)
            coefficient_ids = safe.gather(-1, distance.topk(count, -1, largest=False).indices)
        coefficients = bins[coefficient_ids]
        scores = 2*corr[..., None]*coefficients-norms[atom_ids, None]*coefficients.square()
        top = scores.flatten(1).topk(k, -1)
        certified = (torch.ones(len(rows), device=rows.device, dtype=torch.bool) if branches == limit
                     else ranked_scores[rows, branches] <= top.values[:, -1])
        selected_a = atom_ids.gather(-1, top.indices//count)
        selected_b = coefficient_ids.flatten(1).gather(-1, top.indices)
        done = rows[certified]
        output_a[done] = selected_a[certified]
        output_b[done] = selected_b[certified]
        output_scores[done] = top.values[certified]
        rows = rows[~certified]
        branches = min(2*branches, limit)
    return output_a, output_b, output_scores


@dataclass
class TypedPairTargets:
    tokens: torch.Tensor
    candidate_atoms: torch.Tensor
    candidate_bins: torch.Tensor
    pair_weights: torch.Tensor
    coefficient_weights_given_atom: torch.Tensor
    residual_energies: torch.Tensor


@torch.no_grad()
def typed_pair_targets(codebook, latent, *, top_k=64, relative_temperature=.003,
                       stochastic=True, generator=None, chunk_size=128):
    if (not math.isfinite(relative_temperature) or relative_temperature<0 or chunk_size<1
            or latent.shape[-1] != codebook.dictionary.shape[0]):
        raise ValueError('invalid temperature, chunk size, or latent shape')
    if stochastic and relative_temperature == 0:
        raise ValueError('stochastic targets need a positive temperature')
    shape = latent.shape[:-1]
    residual = latent.reshape(-1, latent.shape[-1]).float().clone()
    selected_a, selected_b, aa, bb, ww, cc, energies = [], [], [], [], [], [], []
    with torch.autocast(device_type=latent.device.type, enabled=False):
        for d in range(codebook.depth):
            atom_parts, bin_parts, weight_parts, conditional_parts, draws_a, draws_b = [], [], [], [], [], []
            for first in range(0, len(residual), chunk_size):
                r = residual[first:first+chunk_size]
                a, b, scores = best_joint_pairs(r, codebook.dictionary, codebook.coeff_bins, top_k)
                if stochastic:
                    temperature = (relative_temperature*r.square().sum(-1, keepdim=True)).clamp_min(1e-8)
                    weights = ((scores-scores.max(-1, keepdim=True).values)/temperature).softmax(-1)
                    draw = torch.multinomial(weights, 1, generator=generator)
                else:
                    weights = torch.zeros_like(scores);weights[:, 0] = 1.
                    draw = torch.zeros(len(r), 1, device=r.device, dtype=torch.long)
                a_sample, b_sample = a.gather(-1, draw).squeeze(-1), b.gather(-1, draw).squeeze(-1)
                conditional = weights*(a==a_sample[:, None])
                conditional = conditional/conditional.sum(-1, keepdim=True)
                atom_parts.append(a);bin_parts.append(b);weight_parts.append(weights);conditional_parts.append(conditional)
                draws_a.append(a_sample);draws_b.append(b_sample)
                r.sub_(codebook.dictionary.T[a_sample]*codebook.coeff_bins[b_sample, None])
            selected_a.append(torch.cat(draws_a));selected_b.append(torch.cat(draws_b))
            aa.append(torch.cat(atom_parts));bb.append(torch.cat(bin_parts));ww.append(torch.cat(weight_parts));cc.append(torch.cat(conditional_parts))
            energies.append(residual.square().sum(-1).reshape(shape))
    atoms = torch.stack(selected_a, -1).reshape(*shape, codebook.depth)
    bins = torch.stack(selected_b, -1).reshape_as(atoms)
    def stack(values):
        result = torch.stack(values, 1)
        return result.reshape(*shape, codebook.depth, result.shape[-1])
    return TypedPairTargets(codebook.pack(atoms, bins), stack(aa), stack(bb), stack(ww), stack(cc),
                            torch.stack(energies, -1))


def typed_pair_cross_entropy(outputs, targets):
    """Joint pair CE, averaged over pairs, with no independent head weights.

    Atom targets marginalize the complete-pair teacher. Coefficient targets
    condition on the atom actually present in the sampled input stream.
    Averaging over that atom draw gives the joint soft CE exactly.
    """
    dtype = torch.float64 if outputs['atom_logits'].dtype == torch.float64 else torch.float32
    atoms = outputs['atom_logits'].to(dtype).log_softmax(-1)
    coefficients = outputs['coeff_logits'].to(dtype).log_softmax(-1)
    if atoms.shape[:-1] != targets.pair_weights.shape[:-1]:
        raise ValueError('logits and pair targets have different prefix shapes')
    gathered_atoms = atoms.gather(-1, targets.candidate_atoms)
    # A teacher candidate may have zero mass; 0*(-inf) must contribute zero.
    gathered_atoms = gathered_atoms.masked_fill(targets.pair_weights == 0, 0)
    atom_ce = -(gathered_atoms*targets.pair_weights).sum(-1)
    gathered_coefficients = coefficients.gather(-1, targets.candidate_bins)
    gathered_coefficients = gathered_coefficients.masked_fill(targets.coefficient_weights_given_atom == 0, 0)
    coefficient_ce = -(gathered_coefficients*targets.coefficient_weights_given_atom).sum(-1)
    axes = tuple(range(atom_ce.ndim-1))
    return (atom_ce+coefficient_ce).mean(), atom_ce.mean(axes), coefficient_ce.mean(axes)
