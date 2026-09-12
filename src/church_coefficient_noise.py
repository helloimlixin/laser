"""Calibrate FFHQ compound coefficient targets in Church physical units."""
import math

import torch

from src.church_ffhq_archived import ChurchAux


@torch.no_grad()
def physical_coefficient_distribution(normalized, bins, scales, sigma):
    """Discrete Gaussian over the existing bins with one physical noise width."""
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError('Expected a finite positive physical coefficient sigma')
    if normalized.shape[-1] != scales.numel():
        raise ValueError('Coefficient depth does not match scales')
    if not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError('Expected finite positive coefficient scales')
    physical = normalized.float() * scales
    physical_bins = scales[:, None] * bins
    logits = -(physical[..., None] - physical_bins).square() / (2 * sigma * sigma)
    return logits.softmax(-1)


class CalibratedChurchAux(ChurchAux):
    """Same frozen tokenizer and bins; replace only the target noise width."""
    def __init__(self, *args, coefficient_sigma, **kwargs):
        if not math.isfinite(coefficient_sigma) or coefficient_sigma <= 0:
            raise ValueError('Expected a finite positive physical coefficient sigma')
        super().__init__(*args, **kwargs)
        self.coefficient_sigma = float(coefficient_sigma)

    @torch.no_grad()
    def compound_coeff_ids(self, coeffs, *, stochastic=True, temp=.5):
        # Retain the archived call signature. Its normalized temperature is
        # replaced by the calibrated physical sigma, recorded in run config.
        probabilities = physical_coefficient_distribution(
            coeffs, self.coeff_bins, self.coeff_scales, self.coefficient_sigma)
        if stochastic:
            ids = torch.multinomial(probabilities.reshape(-1, self.coeff_vocab_size), 1).reshape(coeffs.shape)
        else:
            ids = probabilities.argmax(-1)
        return ids.long(), probabilities
