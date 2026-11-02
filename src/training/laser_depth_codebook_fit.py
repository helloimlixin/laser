"""Training-only initialization and coordinate fitting of per-depth dictionaries."""
import torch
from torch.nn import functional as F


@torch.no_grad()
def initialize_depth_books(source_dictionary, energy_by_depth, atoms_per_book, *, iterations=8, seed=261017):
    if (energy_by_depth.ndim != 2 or energy_by_depth.shape[1] != source_dictionary.shape[1]
            or not 0 < atoms_per_book <= source_dictionary.shape[1]
            or not torch.isfinite(energy_by_depth).all() or (energy_by_depth < 0).any()):
        raise ValueError('Expected nonnegative per-depth source atom energy')
    generator = torch.Generator(device=source_dictionary.device).manual_seed(seed)
    source = source_dictionary.T
    books = []
    for weights in energy_by_depth:
        weights = weights.clamp_min(weights.max().clamp_min(1.) * 1.e-10)
        indices = torch.multinomial(weights, atoms_per_book, replacement=False, generator=generator)
        centers = source[indices].clone()
        for _ in range(iterations):
            # Signed coefficients make +v and -v equivalent dictionary rays.
            projection = source @ centers.T
            assignment = projection.abs().argmax(-1)
            selected = projection.gather(1, assignment[:, None]).squeeze(1)
            total = torch.zeros_like(centers)
            total.index_add_(0, assignment, source * (weights * selected)[:, None])
            valid = total.norm(dim=-1) > 1.e-10
            centers[valid] = F.normalize(total[valid], dim=-1)
        books.append(centers.T)
    return torch.stack(books)


@torch.no_grad()
def refit_depth_books(quantizer, signals, atoms, coefficients, *, chunk_size=4096):
    """Exact unit-norm coordinate updates at fixed atom/continuous-value targets.

    Only the offline tokenizer dictionary is fitted. Encoder, decoder and prior
    are untouched. Unused atoms retain their previous vectors.
    """
    x = signals.reshape(-1, quantizer.channels)
    atoms = atoms.reshape(-1, quantizer.depth)
    coefficients = coefficients.reshape_as(atoms)
    reconstruction = torch.empty_like(x)
    for first in range(0, len(x), chunk_size):
        stop = first + chunk_size
        reconstruction[first:stop] = quantizer.reconstruct(atoms[first:stop], coefficients[first:stop])
    before = float((x - reconstruction).square().mean())
    for depth in range(quantizer.depth):
        old = quantizer.dictionaries[depth].T.clone()
        total = torch.zeros_like(old)
        for first in range(0, len(x), chunk_size):
            stop = first + chunk_size
            selected = atoms[first:stop, depth]
            value = coefficients[first:stop, depth, None]
            partial_residual = x[first:stop] - reconstruction[first:stop] + old[selected] * value
            total.index_add_(0, selected, partial_residual * value)
        valid = total.norm(dim=-1) > 1.e-10
        updated = old.clone()
        updated[valid] = F.normalize(total[valid], dim=-1)
        quantizer.dictionaries[depth].copy_(updated.T)
        for first in range(0, len(x), chunk_size):
            stop = first + chunk_size
            selected = atoms[first:stop, depth]
            reconstruction[first:stop].add_((updated[selected] - old[selected]) * coefficients[first:stop, depth, None])
    after = float((x - reconstruction).square().mean())
    if after > before + 2.e-6 * max(1., before):
        raise RuntimeError('Fixed-code dictionary update increased MSE')
    return dict(fixed_code_mse_before=before, fixed_code_mse_after=after)
