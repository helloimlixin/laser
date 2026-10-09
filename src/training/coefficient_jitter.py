"""Nearest-bin targets after small, bounded continuous coefficient jitter."""
import math

import torch
from torch.nn import functional as F


def jitter_moments(sigma_bins=0.01, cap_bins=0.025):
    if not 0 < sigma_bins <= cap_bins < 0.5:
        raise ValueError('Require 0 < sigma <= cap < half a bin')
    a = cap_bins / sigma_bins
    mass = math.erf(a / math.sqrt(2))
    variance = 1 - 2 * a * math.exp(-a*a/2) / math.sqrt(2*math.pi) / mass
    return dict(sigma_bins=sigma_bins, cap_bins=cap_bins,
                actual_rms_bins=sigma_bins * math.sqrt(variance))


@torch.no_grad()
def jitter_probabilities(coefficients, bins, *, sigma_bins=0.01, cap_bins=0.025):
    """Integrate a truncated Gaussian over nearest-center quantization cells.

    Because the jitter is bounded below half a bin, only the two centers
    bracketing the clean coefficient can receive mass. Double precision for
    those local boundaries avoids cancellation when sigma is far below a bin.
    This is not the Gaussian-on-centers kernel and has no temperature floor.
    """
    jitter_moments(sigma_bins, cap_bins)
    c = coefficients.double()
    centers = bins.double()
    width = (centers[-1] - centers[0]) / (centers.numel() - 1)
    upper = torch.searchsorted(centers, c.contiguous()).clamp(0, len(centers)-1)
    lower = (upper-1).clamp_min(0)
    # Values above the last center quantize to the last center.
    lower = torch.where(c >= centers[-1], upper, lower)
    midpoint = (centers[lower] + centers[upper]) / 2
    a = cap_bins / sigma_bins
    z = ((midpoint-c) / (sigma_bins*width)).clamp(-a, a)
    low_cdf = 0.5 * math.erfc(a / math.sqrt(2))
    mass = math.erf(a / math.sqrt(2))
    p_lower = ((torch.special.ndtr(z)-low_cdf) / mass).clamp(0, 1)
    p_lower = torch.where(lower == upper, torch.ones_like(p_lower), p_lower)
    nearest = torch.where(c <= midpoint, lower, upper).long()
    probabilities = torch.zeros((*c.shape, len(centers)), device=c.device, dtype=torch.float32)
    probabilities.scatter_add_(-1, lower[..., None], p_lower.float()[..., None])
    probabilities.scatter_add_(-1, upper[..., None], (1-p_lower).float()[..., None])
    return nearest, probabilities


@torch.no_grad()
def quantize_with_jitter(coefficients, bins, *, stochastic=True, hard=False,
                         sigma_bins=0.01, cap_bins=0.025):
    nearest, probabilities = jitter_probabilities(
        coefficients, bins, sigma_bins=sigma_bins, cap_bins=cap_bins)
    if hard:
        probabilities = F.one_hot(nearest, len(bins)).float()
    if stochastic and not hard:
        ids = torch.multinomial(probabilities.reshape(-1, len(bins)), 1).reshape(coefficients.shape)
    else:
        ids = nearest
    return ids.long(), probabilities


@torch.no_grad()
def jittered_scalar_targets(atoms, coefficients, bins, *, num_atoms,
                            stochastic=True, hard=False, compact=True,
                            sigma_bins=0.01, cap_bins=0.025):
    """Build scalar atom/coefficient targets using bounded pre-quantization noise.

    This is the scalar prior path. Patching compound_coeff_ids alone does not
    affect scalar training. No Gaussian-on-centers temperature floor applies.
    """
    if atoms.shape != coefficients.shape or atoms.ndim != 4:
        raise ValueError('Scalar targets require matching [B, H, W, K] arrays')
    ids, probabilities = quantize_with_jitter(
        coefficients, bins, stochastic=stochastic, hard=hard,
        sigma_bins=sigma_bins, cap_bins=cap_bins)
    tokens = torch.empty((*atoms.shape[:-1], 2 * atoms.shape[-1]),
                         dtype=torch.long, device=atoms.device)
    tokens[..., 0::2] = atoms.long()
    tokens[..., 1::2] = ids + num_atoms
    if compact:
        return tokens, (atoms.long(), probabilities)
    targets = probabilities.new_zeros((*tokens.shape, num_atoms + len(bins)))
    targets[..., 0::2, :num_atoms].scatter_(-1, atoms.long()[..., None], 1.)
    targets[..., 1::2, num_atoms:] = probabilities
    return tokens, targets
