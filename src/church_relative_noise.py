"""Magnitude-aware, bounded coefficient targets for the archived Church prior."""
import math

import torch

from src.church_ffhq_archived import ChurchAux


def validate_noise(sigma_cap, relative_sigma, truncate):
    if not all(math.isfinite(x) and x > 0 for x in (sigma_cap, relative_sigma, truncate)):
        raise ValueError('Noise parameters must be finite and positive')
    if relative_sigma * truncate >= 1:
        raise ValueError('Relative noise radius must be smaller than coefficient magnitude')


@torch.no_grad()
def relative_coefficient_distribution(normalized, bins, scales, sigma_cap, relative_sigma, truncate=3.):
    """Bound deviations from the clean coefficient, with explicit bin fallback.

    sigma(c) = min(sigma_cap, relative_sigma * abs(c)). Only centers within
    truncate*sigma(c) are allowed, plus the nearest center. If quantization is
    coarser than this interval, the target becomes nearest-bin deterministic.
    The nearest-bin exception is necessary because the existing even-sized
    bin vocabulary cannot represent every value, including zero, exactly.
    """
    validate_noise(sigma_cap, relative_sigma, truncate)
    if normalized.shape[-1] != scales.numel():
        raise ValueError('Coefficient depth does not match scales')
    if not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError('Expected finite positive coefficient scales')
    physical = normalized.float() * scales
    centers = scales[:, None] * bins
    delta = centers - physical[..., None]
    squared = delta.square()
    minimum, nearest = squared.min(-1, keepdim=True)
    sigma = (relative_sigma*physical.abs()).clamp_max(sigma_cap)
    allowed = delta.abs() <= (truncate*sigma)[..., None]
    allowed.scatter_(-1, nearest, True)
    # Subtract the minimum first: the nearest logit is exactly zero even at
    # sigma=0. No artificial sigma floor broadens weak/zero coefficients.
    denominator = (2*sigma.square()).clamp_min(torch.finfo(sigma.dtype).tiny)
    logits = -(squared-minimum)/denominator[..., None]
    return logits.masked_fill(~allowed, -torch.inf).softmax(-1)


class RelativeChurchAux(ChurchAux):
    def __init__(self, *args, sigma_cap, relative_sigma, truncate=3., **kwargs):
        validate_noise(sigma_cap, relative_sigma, truncate)
        super().__init__(*args, **kwargs)
        self.sigma_cap = float(sigma_cap)
        self.relative_sigma = float(relative_sigma)
        self.truncate = float(truncate)

    @torch.no_grad()
    def compound_coeff_ids(self, coeffs, *, stochastic=True, temp=.5):
        probabilities = relative_coefficient_distribution(coeffs, self.coeff_bins,
            self.coeff_scales, self.sigma_cap, self.relative_sigma, self.truncate)
        if stochastic:
            ids = torch.multinomial(probabilities.reshape(-1, self.coeff_vocab_size), 1).reshape(coeffs.shape)
        else:
            ids = probabilities.argmax(-1)
        return ids.long(), probabilities
