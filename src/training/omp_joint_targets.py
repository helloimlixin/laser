"""Exact conditional pair targets for a finite bank of full-refit OMP codes.

One trajectory is drawn uniformly per spatial site. Its atom IDs are fixed;
coefficient IDs are independently drawn from that trajectory's soft kernels.
Conditioning on the observed pair prefix gives a posterior over trajectories.
This module marginalizes that posterior for both heads, without changing the
bank law, final OMP coefficients, or autoregressive event order.
"""
from dataclasses import dataclass
import math

import torch


@dataclass
class OMPJointTargets:
    atom_ids: torch.Tensor
    atom_weights: torch.Tensor
    coefficient_probabilities: torch.Tensor
    trajectory_weights_given_atom: torch.Tensor


@torch.no_grad()
def omp_bank_joint_targets(bank_atoms, bank_coefficients, sampled_atoms,
                           sampled_coefficient_ids, coefficient_bins, *,
                           temperature, site_chunk_size=128):
    """Return q(a_d|pairs_<d) and q(c_d|pairs_<d, sampled a_d).

    Banks have shape [..., variants, depth], samples [..., depth], bins [bins].
    Coefficients and bins must share the same units (normalized or physical).
    Duplicate bank entries retain their empirical multiplicity. Never condition
    a target on its own sampled coefficient or any future pair.

    The coefficient target averages over every compatible trajectory, rather
    than using the kernel of just the sampled trajectory. These estimators have
    the same expected coefficient CE. At a fixed prefix and current atom, the
    averaging removes variance due to the latent trajectory draw.

    Kernel normalization is computed here, so cached normalizers from a
    different temperature cannot silently change the teacher distribution.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    if (isinstance(site_chunk_size, bool) or not isinstance(site_chunk_size, int)
            or site_chunk_size < 1):
        raise ValueError('site_chunk_size must be a positive integer')
    if (bank_atoms.ndim < 3 or bank_atoms.shape != bank_coefficients.shape
            or sampled_atoms.shape != bank_atoms.shape[:-2] + bank_atoms.shape[-1:]
            or sampled_coefficient_ids.shape != sampled_atoms.shape
            or coefficient_bins.ndim != 1
            or any(v.numel() == 0 for v in (bank_atoms, bank_coefficients, coefficient_bins))):
        raise ValueError('invalid bank, sample, or coefficient-bin shapes')
    tensors = (bank_coefficients, sampled_atoms, sampled_coefficient_ids, coefficient_bins)
    if any(v.device != bank_atoms.device for v in tensors):
        raise ValueError('bank and sample tensors must share a device')
    if (bank_coefficients.dtype not in (torch.float32, torch.float64)
            or coefficient_bins.dtype != bank_coefficients.dtype):
        raise ValueError('coefficients and bins must share an FP32/FP64 dtype')
    integer_dtypes = (torch.int16, torch.int32, torch.int64, torch.uint8)
    if any(v.dtype not in integer_dtypes for v in (bank_atoms, sampled_atoms, sampled_coefficient_ids)):
        raise ValueError('atom and coefficient IDs must be integers')
    if (not torch.isfinite(bank_coefficients).all()
            or not torch.isfinite(coefficient_bins).all()
            or (bank_atoms < 0).any() or (sampled_atoms < 0).any()
            or (sampled_coefficient_ids < 0).any()
            or (sampled_coefficient_ids >= len(coefficient_bins)).any()):
        raise ValueError('nonfinite coefficients or invalid token IDs')
    with torch.autocast(device_type=bank_atoms.device.type, enabled=False):
        leading = bank_atoms.shape[:-2]
        variants, depth = bank_atoms.shape[-2:]
        atoms = bank_atoms.reshape(-1, variants, depth)
        centers = bank_coefficients.reshape_as(atoms)
        observed_atoms = sampled_atoms.reshape(-1, depth)
        observed_coefficients = sampled_coefficient_ids.reshape(-1, depth)
        atom_weights = centers.new_empty(len(atoms), depth, variants)
        conditional_weights = torch.empty_like(atom_weights)
        coefficient_targets = centers.new_empty(len(atoms), depth, len(coefficient_bins))
        for start in range(0, len(atoms), site_chunk_size):
            stop = min(start + site_chunk_size, len(atoms))
            a, c = atoms[start:stop], centers[start:stop]
            log_posterior = c.new_zeros(len(a), variants)
            for d in range(depth):
                atom_weights[start:stop, d] = log_posterior.softmax(-1)
                consistent = a[..., d] == observed_atoms[start:stop, d, None]
                conditioned = log_posterior.masked_fill(~consistent, -torch.inf)
                if not torch.isfinite(conditioned).any(-1).all():
                    raise ValueError('sampled support prefix is absent from the bank')
                weights = conditioned.softmax(-1)
                conditional_weights[start:stop, d] = weights
                log_kernels = (-(c[..., d, None] - coefficient_bins).square()
                               / temperature).log_softmax(-1)
                coefficient_targets[start:stop, d] = (
                    weights[..., None] * log_kernels.exp()).sum(-2)
                if d + 1 < depth:
                    observed = observed_coefficients[start:stop, d, None, None]
                    likelihood = log_kernels.gather(-1, observed.expand(-1, variants, 1)).squeeze(-1)
                    log_posterior = conditioned + likelihood
                    # A common shift preserves the posterior and avoids drift.
                    log_posterior -= log_posterior.logsumexp(-1, keepdim=True)
        return OMPJointTargets(
            bank_atoms.transpose(-1, -2).long(),
            atom_weights.reshape(*leading, depth, variants),
            coefficient_targets.reshape(*leading, depth, len(coefficient_bins)),
            conditional_weights.reshape(*leading, depth, variants),
        )


def omp_joint_cross_entropy(atom_logits, coefficient_logits, targets):
    """Unweighted joint soft CE in nats per pair (mean across sites/depth).

    Coefficient logits must condition on the sampled current atom. Sparse atom
    targets may include duplicate IDs and zero-mass masked atoms. This loss has
    no extra atom multiplier and is not divided by the number of heads.
    """
    if (coefficient_logits.shape != targets.coefficient_probabilities.shape
            or atom_logits.shape[:-1] != targets.atom_ids.shape[:-1]
            or targets.atom_ids.shape != targets.atom_weights.shape):
        raise ValueError('logit and target shapes differ')
    dtype = torch.float64 if atom_logits.dtype == torch.float64 else torch.float32
    log_atoms = atom_logits.to(dtype).log_softmax(-1).gather(-1, targets.atom_ids)
    weights = targets.atom_weights.to(dtype)
    safe = torch.where(weights > 0, log_atoms, torch.zeros_like(log_atoms))
    atom_ce = -(weights * safe).sum(-1)
    coefficient_ce = -(targets.coefficient_probabilities.to(dtype)
                       * coefficient_logits.to(dtype).log_softmax(-1)).sum(-1)
    return (atom_ce + coefficient_ce).mean()
