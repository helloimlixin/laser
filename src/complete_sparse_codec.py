"""One site ID for an entire sparse support and its signed coefficient values.

Exact integer packing is separate from a finite, lossy prototype vocabulary.
Prototype codewords are complete observed sparse codes, including their atoms.
"""
from __future__ import annotations

import operator

import torch

from src.coefficient_pattern_codec import assign_coefficient_patterns, fit_coefficient_patterns


def pack_exact(atoms, coefficient_ids, *, num_atoms=16384, num_bins=2048):
    """Pack into a Python integer; a Church site needs up to 100 bits."""
    atoms, coefficient_ids = list(atoms), list(coefficient_ids)
    if not atoms or len(atoms) != len(coefficient_ids):
        raise ValueError('Expected equally sized, nonempty support and coefficient lists')
    num_atoms, num_bins = operator.index(num_atoms), operator.index(num_bins)
    if min(num_atoms, num_bins) < 1:
        raise ValueError('Vocabulary sizes must be positive')
    value = 0
    for atom, coefficient in zip(reversed(atoms), reversed(coefficient_ids)):
        atom, coefficient = operator.index(atom), operator.index(coefficient)
        if not (0 <= atom < num_atoms and 0 <= coefficient < num_bins):
            raise ValueError('Sparse-code component outside its vocabulary')
        value = value * (num_atoms * num_bins) + atom * num_bins + coefficient
    return value


def unpack_exact(value, *, depth=4, num_atoms=16384, num_bins=2048):
    value, depth = operator.index(value), operator.index(depth)
    num_atoms, num_bins = operator.index(num_atoms), operator.index(num_bins)
    if min(depth, num_atoms, num_bins) < 1 or not 0 <= value < (num_atoms * num_bins) ** depth:
        raise ValueError('Integer or code dimensions outside the representable range')
    atoms, coefficients = [], []
    for _ in range(depth):
        value, pair = divmod(value, num_atoms * num_bins)
        atom, coefficient = divmod(pair, num_bins)
        atoms.append(atom)
        coefficients.append(coefficient)
    return atoms, coefficients


def sparse_latents(atoms, coefficient_ids, dictionary_rows, bins, scales):
    """Decode the complete sparse code; coefficient-bin indices encode signs."""
    if atoms.shape != coefficient_ids.shape or atoms.shape[-1] != len(scales):
        raise ValueError('Support and coefficient shapes must match the depth scales')
    if atoms.numel() and (int(atoms.min()) < 0 or int(atoms.max()) >= len(dictionary_rows)):
        raise ValueError('Atom ID outside the dictionary')
    if coefficient_ids.numel() and (int(coefficient_ids.min()) < 0 or int(coefficient_ids.max()) >= len(bins)):
        raise ValueError('Coefficient ID outside the bins')
    physical = bins[coefficient_ids.long()] * scales
    return (dictionary_rows[atoms.long()] * physical[..., None]).sum(-2)


def decode_site_ids(site_ids, codebook):
    """A single integer selects BOTH the four atoms and the four coefficients."""
    if site_ids.numel() and (int(site_ids.min()) < 0 or int(site_ids.max()) >= len(codebook['atoms'])):
        raise ValueError('Site ID outside the codebook')
    return codebook['atoms'][site_ids.long()], codebook['coefficient_ids'][site_ids.long()]


@torch.no_grad()
def fit_complete_codebook(atoms, coefficient_ids, dictionary_rows, bins, scales,
                          *, num_codes, iterations=8, seed=9701, chunk_size=1024, progress=None):
    """Lloyd centers in decoded latent space, projected to assigned training codes.

    The projection chooses the observed member closest to its cluster center.
    Thus every returned entry remains a realizable complete sparse code. This
    is a bounded initialization/projection experiment, not global k-medoids.
    """
    latents = sparse_latents(atoms, coefficient_ids, dictionary_rows, bins, scales)
    centers = fit_coefficient_patterns(latents, num_patterns=num_codes, iterations=iterations,
        seed=seed, chunk_size=chunk_size, progress=progress)
    assignments, distances = assign_coefficient_patterns(latents, centers, chunk_size=chunk_size)
    minima = torch.full((num_codes,), float('inf'), device=latents.device)
    minima.scatter_reduce_(0, assignments, distances, reduce='amin', include_self=True)
    indices = torch.arange(len(latents), device=latents.device)
    eligible = torch.where(distances == minima[assignments], indices, len(latents))
    representatives = torch.full((num_codes,), len(latents), device=latents.device, dtype=torch.long)
    representatives.scatter_reduce_(0, assignments, eligible, reduce='amin', include_self=True)
    empty = representatives == len(latents)
    if empty.any():
        rng = torch.Generator(device=latents.device).manual_seed(seed + 1)
        representatives[empty] = torch.randperm(len(latents), device=latents.device, generator=rng)[:int(empty.sum())]
    return {'atoms': atoms[representatives].clone(),
            'coefficient_ids': coefficient_ids[representatives].clone(),
            'latents': latents[representatives].clone(), 'dense_centers': centers,
            'fit_representative_indices': representatives, 'empty_clusters': int(empty.sum())}
