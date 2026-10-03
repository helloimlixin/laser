"""Causal soft atom labels for a finite mixture of full-refit OMP trajectories.

At each spatial site, draw one uniform trajectory V, then independent coefficient
bins from its full soft kernels. Given preceding sampled atom/coefficient pairs,
Bayes' rule determines the posterior over V. The next atom target is that
posterior aggregated by atom ID. Duplicate trajectories keep their empirical
multiplicity. Coefficient soft CE on the sampled V is an unbiased Monte Carlo
estimate of the conditional mixture coefficient objective.

No coefficient refit changes an emitted prefix during training: all continuous
coefficients were jointly refitted before the trajectory bank was constructed.
This is exact for the finite bank, not an exact full-vocabulary RQ teacher.
"""
from __future__ import annotations
import torch

@torch.no_grad()
def atom_targets(bank_atoms, bank_coeffs, bank_log_normalizers, sampled_atoms,
                 sampled_coeff_ids, coefficient_bins, temperature):
    if bank_atoms.shape != bank_coeffs.shape or bank_atoms.shape != bank_log_normalizers.shape:
        raise ValueError('Atom, coefficient and log-normalizer banks must align')
    if bank_atoms.ndim != 5 or sampled_atoms.shape != bank_atoms.shape[:3]+bank_atoms.shape[-1:]:
        raise ValueError('Expected bank[B,H,W,V,D] and sample[B,H,W,D]')
    if sampled_coeff_ids.shape != sampled_atoms.shape or temperature <= 0:
        raise ValueError('Coefficient history shape or temperature invalid')
    log_weights = torch.zeros_like(bank_coeffs[..., 0], dtype=torch.float32)
    weights=[]
    for depth in range(bank_atoms.shape[-1]):
        weights.append(log_weights.softmax(-1))
        if depth+1 == bank_atoms.shape[-1]: break
        consistent = bank_atoms[..., depth] == sampled_atoms[..., depth, None]
        value = coefficient_bins[sampled_coeff_ids[..., depth].long()][..., None]
        log_likelihood = -(value-bank_coeffs[..., depth].float()).square()/temperature-bank_log_normalizers[..., depth].float()
        log_weights = (log_weights+log_likelihood).masked_fill(~consistent, -torch.inf)
    return bank_atoms.transpose(-1,-2).long(), torch.stack(weights,dim=-2)


def sparse_atom_cross_entropy(log_probs, ids, weights):
    gathered=log_probs.gather(-1,ids.long())
    # Incompatible variants can name an already-used (masked) atom. They carry
    # exactly zero probability; mask before multiplying to avoid 0 * -infinity.
    safe=torch.where(weights>0,gathered,torch.zeros_like(gathered))
    return -(weights*safe).sum(-1)

@torch.no_grad()
def sparse_atom_entropy(ids, weights):
    mass=((ids[..., :, None]==ids[..., None, :])*weights[..., None, :]).sum(-1)
    return -(weights*mass.clamp_min(1e-30).log()).sum(-1)
