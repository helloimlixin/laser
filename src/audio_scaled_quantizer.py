"""Small joint sparse vocabularies and an optional 40-bit residual refinement.

The encoder, dictionary and decoder stay frozen. Joint zero has one canonical
ID. Raw payloads are packed across frame boundaries, with at most seven padding
bits at the end of an utterance. The frame count and codebooks are stream metadata.
"""
import math

import numpy as np
import torch


def field_widths(vocab_sizes):
    if any(v < 2 for v in vocab_sizes):
        raise ValueError('Each vocabulary must contain at least two symbols')
    return tuple((int(v) - 1).bit_length() for v in vocab_sizes)


def pack_integer_frames(codes, vocab_sizes):
    source = np.asarray(codes)
    widths = field_widths(vocab_sizes)
    if source.ndim != 2 or source.shape[1] != len(widths) or not np.issubdtype(source.dtype, np.integer):
        raise ValueError('Expected an integer frame-by-field matrix')
    bits = []
    for d, (size, width) in enumerate(zip(vocab_sizes, widths)):
        if np.any((source[:, d] < 0) | (source[:, d] >= size)):
            raise ValueError('Integer outside the declared vocabulary')
        shifts = np.arange(width - 1, -1, -1, dtype=np.uint64)
        bits.append(((source[:, d:d+1].astype(np.uint64) >> shifts) & 1).astype(np.uint8))
    return np.packbits(np.concatenate(bits, axis=1).reshape(-1), bitorder='big').tobytes()


def unpack_integer_frames(payload, vocab_sizes, frames):
    if frames < 0:
        raise ValueError('Negative frame count')
    widths = field_widths(vocab_sizes); needed = frames * sum(widths)
    if len(payload) != (needed + 7) // 8:
        raise ValueError('Payload length disagrees with frame count and vocabulary')
    bits = np.unpackbits(np.frombuffer(payload, np.uint8), bitorder='big')
    if bits[needed:].any():
        raise ValueError('Nonzero final byte padding')
    bits = bits[:needed].reshape(frames, sum(widths)); offset = 0; columns = []
    for size, width in zip(vocab_sizes, widths):
        shifts = np.arange(width - 1, -1, -1, dtype=np.uint64)
        column = (bits[:, offset:offset+width].astype(np.uint64) << shifts).sum(1)
        if np.any(column >= size):
            raise ValueError('Reserved integer code in payload')
        columns.append(column.astype(np.int64)); offset += width
    return np.stack(columns, axis=1)


def sparse_to_joint(atoms, coefficients, levels):
    """Nearest scalar quantization on an existing OMP support, including zero."""
    all_levels = torch.cat((levels.new_zeros(1), levels))
    bins = (coefficients[..., None] - all_levels).abs().argmin(-1)
    return torch.where(bins == 0, 0, 1 + atoms * len(levels) + bins - 1)


@torch.no_grad()
def nearest_centers(values, centers, chunk_size=16384):
    ids = []
    for chunk in values.split(chunk_size):
        # The input norm cancels for nearest-center selection.
        distance = centers.square().sum(1)[None] - 2 * chunk @ centers.T
        ids.append(distance.argmin(1))
    return torch.cat(ids) if ids else torch.empty(0, dtype=torch.long, device=values.device)


@torch.no_grad()
def fit_residual_centers(residuals, count, seed=20260913, iterations=30, max_samples=65536):
    """Training-only Lloyd fit, with center zero fixed to the zero vector."""
    if residuals.ndim != 2 or len(residuals) < count or count < 2 or not torch.isfinite(residuals).all():
        raise ValueError('Need finite training residuals and at least two centers')
    generator = torch.Generator(device=residuals.device).manual_seed(seed)
    indices = torch.randperm(len(residuals), generator=generator, device=residuals.device)[:max_samples]
    values = residuals[indices]
    centers = torch.cat((values.new_zeros(1, values.shape[1]), values[:count-1].clone()))
    for _ in range(iterations):
        ids = nearest_centers(values, centers)
        sums = torch.zeros_like(centers).index_add_(0, ids, values)
        counts = torch.bincount(ids, minlength=count)
        new = torch.where(counts[:, None] > 0, sums / counts[:, None].clamp_min(1), centers)
        new[0].zero_()
        movement = (new - centers).abs().max()
        centers = new
        if movement < 1e-5: break
    return centers


def rate_spec(levels, atoms=8192, depth=2, refine=False):
    vocab = 1 + atoms * levels
    widths = field_widths([vocab] * depth)
    spare = 40 - sum(widths)
    if refine and spare <= 0:
        raise ValueError('No room for a refinement token at 6 kbps')
    sizes = [vocab] * depth + ([2**spare] if refine else [])
    widths = field_widths(sizes)
    return {'vocab_sizes': sizes, 'field_bits': list(widths), 'bits_per_frame': sum(widths),
            'nominal_kbps': sum(widths) * 150 / 1000,
            'refinement_entries': 2**spare if refine else 0,
            'refinement_is_vector_rvq': bool(refine)}
