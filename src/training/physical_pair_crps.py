"""Cross entropy plus ordered coefficient distribution loss for scalar pairs.

The auxiliary loss integrates squared CDF differences over coefficient bins,
divided by their range. For soft targets this differs from expected CRPS by a
target-only constant, so it has the same optimum and gradients. Range scaling
makes the auxiliary term dimensionless and bounded by one at every depth.
"""
import math

import torch
import torch.nn.functional as F


def crps_weight_at_step(max_weight, global_step, start_step, ramp_steps):
    """Use the saved global cursor, including after resumes and best rewinds."""
    if not math.isfinite(max_weight) or max_weight < 0:
        raise ValueError('CRPS weight must be finite and nonnegative')
    if min(global_step, start_step, ramp_steps) < 0:
        raise ValueError('CRPS schedule steps must be nonnegative')
    if ramp_steps == 0:
        return max_weight if global_step >= start_step else 0.
    return max_weight * min(1., max(0., (global_step - start_step) / ramp_steps))


def validate_coefficient_bins(bins, vocabulary_size):
    if bins.ndim != 1 or bins.numel() != vocabulary_size or vocabulary_size < 2:
        raise ValueError('Coefficient bins must match the vocabulary')
    if not bool(torch.isfinite(bins).all()) or not bool((bins[1:] > bins[:-1]).all()):
        raise ValueError('Coefficient bins must be finite and strictly increasing')


def physical_pair_objective_components(atom_logits, coeff_logits, atoms,
                                       probabilities, coefficient_bins,
                                       crps_weight, accumulation, *,
                                       target_atom_ids=None, target_atom_weights=None):
    atom_log_probs = F.log_softmax(atom_logits.float(), dim=-1)
    coeff_log_probs = F.log_softmax(coeff_logits.float(), dim=-1)
    if target_atom_ids is None:
        if target_atom_weights is not None:
            raise ValueError('atom target IDs and weights must be supplied together')
        atom_nll = -atom_log_probs.gather(-1, atoms.long().unsqueeze(-1)).squeeze(-1)
    else:
        if target_atom_weights is None or target_atom_ids.shape != target_atom_weights.shape:
            raise ValueError('atom target IDs and weights must have equal shapes')
        selected = atom_log_probs.gather(-1, target_atom_ids.long())
        safe = torch.where(target_atom_weights > 0, selected, 0.)
        atom_nll = -(target_atom_weights.float() * safe).sum(-1)
    coeff_ce = -(probabilities.float() * coeff_log_probs).sum(dim=-1)
    summed_ce = (atom_nll.sum(dim=-1) + coeff_ce.sum(dim=-1)).mean()
    classification = summed_ce / (2 * atoms.shape[-1])
    bins = coefficient_bins.float()
    widths = (bins[1:] - bins[:-1]) / (bins[-1] - bins[0])
    predicted_cdf = coeff_log_probs.exp().cumsum(dim=-1)[..., :-1]
    target_cdf = probabilities.float().cumsum(dim=-1)[..., :-1]
    crps = ((predicted_cdf - target_cdf).square() * widths).sum(dim=-1).mean()
    total = summed_ce / (2 * atoms.shape[-1] * accumulation) + crps_weight * crps / accumulation
    return total, classification, crps


def physical_pair_objective(*args, **kwargs):
    return physical_pair_objective_components(*args, **kwargs)[0]
