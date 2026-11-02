"""Fit a shared integer-to-(atom, 512-grid-bin) map using training data only."""
import torch

from src.training.laser_hard_ar import nearest_coefficient_ids


def ordered_signed_grid_ids(centers, bins):
    levels = centers.shape[-1]
    if levels % 2 or len(bins) % 2 or levels > len(bins):
        raise ValueError('Even signed grid and catalog sizes required')
    ids = nearest_coefficient_ids(centers, bins)
    half = levels//2
    for part, offset in [(slice(0, half), 0), (slice(half, levels), len(bins)//2)]:
        index = torch.arange(half, device=ids.device)
        local = ids[:, part]-offset
        local = torch.maximum(local, index)
        local = torch.minimum(local, len(bins)//2-half+index)
        ids[:, part] = (local-index).cummax(-1).values+index+offset
    return ids


@torch.no_grad()
def fit_catalog_bins(atoms, normalized_coefficients, scales, num_atoms, bins,
                     levels, *, iterations=30, prior_weight=4.):
    if atoms.shape != normalized_coefficients.shape or atoms.shape[-1] != len(scales):
        raise ValueError('Expected matching source support and normalized coefficients')
    weights = scales.square()/scales.square().mean()
    weights = weights.expand_as(normalized_coefficients).reshape(-1)
    a, c = atoms.reshape(-1), normalized_coefficients.reshape(-1)
    positions = torch.linspace(.05, .95, levels//2, device=c.device)
    initial = torch.cat((torch.quantile(c[c < 0], positions), torch.quantile(c[c > 0], positions)))
    centers = initial.expand(num_atoms, -1).clone()
    for _ in range(iterations):
        # Normalized bins have one common ID map at all depths. Weighting by
        # scale^2 minimizes physical coefficient distortion across those depths.
        total = c.new_zeros(num_atoms*levels)
        count = torch.zeros_like(total)
        for first in range(0, len(c), 65536):
            chosen_atoms = a[first:first+65536]
            values = c[first:first+65536]
            candidates = centers[chosen_atoms]
            allowed_sign = (values[:, None] < 0) == (candidates < 0)
            chosen = (values[:, None]-candidates).square().masked_fill(~allowed_sign, torch.inf).argmin(-1)
            ids = chosen_atoms*levels+chosen
            count.index_add_(0, ids, weights[first:first+65536])
            total.index_add_(0, ids, weights[first:first+65536]*values)
        proposed = (total.view(num_atoms, levels)+prior_weight*initial)/(count.view(num_atoms, levels)+prior_weight)
        bin_ids = ordered_signed_grid_ids(proposed.sort(-1).values, bins)
        updated = bins[bin_ids]
        if torch.equal(updated, centers):
            break
        centers = updated
    return ordered_signed_grid_ids(centers, bins)
