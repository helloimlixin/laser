"""Normalized depth weighting for existing compound-token objectives."""
import torch


def reweight_depths(total, objective, *, depth_weights, atom_weight, accumulation):
    atom, coeff = objective['atom_nll'], objective['coeff_cross_entropy']
    weights = torch.as_tensor(depth_weights, device=atom.device, dtype=atom.dtype)
    if weights.ndim != 1 or len(weights) != atom.shape[-1] or not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('depth weights must be finite, positive, and match sparse depth')
    if accumulation <= 0 or atom_weight < 0:
        raise ValueError('invalid accumulation or atom weight')
    classification = (((atom_weight*atom + coeff)*weights).sum(-1) /
                      ((atom_weight+1)*weights.sum())).mean()
    updated = dict(objective, classification=classification)
    return total + (classification-objective['classification'])/accumulation, updated
