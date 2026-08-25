"""Geometry-aware vector quantization for LASER coefficient tuples.

The atom support stays discrete and unchanged.  This codec replaces the
``K`` independently quantized coefficients at a spatial site with one pattern
id whose codeword is a physical ``K``-vector.  Distances are measured in the
reconstructed latent geometry induced by the selected dictionary atoms.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch


ProgressCallback = Callable[[int, float, int], None]


def selected_support_grams(
    atom_ids: torch.Tensor,
    dictionary_rows: torch.Tensor,
    *,
    chunk_size: int = 65_536,
) -> torch.Tensor:
    """Return ``D_S^T D_S`` for each support row.

    Args:
        atom_ids: Integer tensor ``[N, K]``.
        dictionary_rows: Row-major normalized atoms ``[M, C]``.
        chunk_size: Limits the temporary ``[chunk, K, C]`` gather.
    """

    if atom_ids.ndim != 2:
        raise ValueError(f"atom_ids must have shape [N,K], got {tuple(atom_ids.shape)}")
    if dictionary_rows.ndim != 2:
        raise ValueError(
            "dictionary_rows must have shape [M,C], got "
            f"{tuple(dictionary_rows.shape)}"
        )
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if atom_ids.numel() and (
        int(atom_ids.min()) < 0 or int(atom_ids.max()) >= dictionary_rows.shape[0]
    ):
        raise ValueError("atom_ids contain an index outside the dictionary")

    outputs = []
    for start in range(0, atom_ids.shape[0], int(chunk_size)):
        support = dictionary_rows[atom_ids[start : start + chunk_size].long()]
        outputs.append(support @ support.transpose(-1, -2))
    if not outputs:
        depth = atom_ids.shape[1]
        return dictionary_rows.new_empty((0, depth, depth))
    return torch.cat(outputs, dim=0)


def _validate_quantizer_inputs(
    coefficients: torch.Tensor,
    patterns: torch.Tensor,
    grams: Optional[torch.Tensor],
) -> None:
    if coefficients.ndim != 2:
        raise ValueError(
            f"coefficients must have shape [N,K], got {tuple(coefficients.shape)}"
        )
    if patterns.ndim != 2 or patterns.shape[1] != coefficients.shape[1]:
        raise ValueError(
            "patterns must have shape [V,K] matching coefficients, got "
            f"{tuple(patterns.shape)} for {tuple(coefficients.shape)}"
        )
    if patterns.shape[0] <= 0:
        raise ValueError("patterns cannot be empty")
    if grams is not None and tuple(grams.shape) != (
        coefficients.shape[0],
        coefficients.shape[1],
        coefficients.shape[1],
    ):
        raise ValueError(
            "grams must have shape [N,K,K], got "
            f"{tuple(grams.shape)} for {tuple(coefficients.shape)}"
        )
    if not torch.isfinite(coefficients).all() or not torch.isfinite(patterns).all():
        raise ValueError("coefficients and patterns must be finite")
    if grams is not None and not torch.isfinite(grams).all():
        raise ValueError("grams must be finite")


def assign_coefficient_patterns(
    coefficients: torch.Tensor,
    patterns: torch.Tensor,
    *,
    grams: Optional[torch.Tensor] = None,
    chunk_size: int = 16_384,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign coefficient rows to their nearest pattern.

    If ``grams`` is supplied, the distance for row ``n`` and pattern ``j`` is

    ``(c_n - p_j)^T G_n (c_n - p_j)``.

    Otherwise ordinary squared Euclidean distance is used.  Returned distances
    are squared distances in the selected metric.
    """

    _validate_quantizer_inputs(coefficients, patterns, grams)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    patterns = patterns.to(device=coefficients.device, dtype=coefficients.dtype)
    pattern_norms = patterns.square().sum(dim=1) if grams is None else None
    pattern_outer = (
        torch.einsum("vi,vj->vij", patterns, patterns).reshape(patterns.shape[0], -1)
        if grams is not None
        else None
    )
    assignments = []
    squared_distances = []
    for start in range(0, coefficients.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), coefficients.shape[0])
        values = coefficients[start:stop]
        if grams is None:
            distances = values.square().sum(dim=1, keepdim=True)
            distances = distances + pattern_norms.unsqueeze(0)
            distances.addmm_(values, patterns.t(), beta=1.0, alpha=-2.0)
        else:
            local_grams = grams[start:stop].to(
                device=coefficients.device, dtype=coefficients.dtype
            )
            gram_values = (local_grams @ values.unsqueeze(-1)).squeeze(-1)
            constant = (values * gram_values).sum(dim=1, keepdim=True)
            distances = local_grams.reshape(local_grams.shape[0], -1) @ pattern_outer.t()
            distances.addmm_(gram_values, patterns.t(), beta=1.0, alpha=-2.0)
            distances.add_(constant)
        nearest_distance, nearest_pattern = distances.min(dim=1)
        assignments.append(nearest_pattern)
        # Tiny negative values can occur through cancellation in the expanded
        # quadratic form.  The underlying Gram distance is non-negative.
        squared_distances.append(nearest_distance.clamp_min_(0.0))
    return torch.cat(assignments), torch.cat(squared_distances)


def _updated_pattern_centers(
    coefficients: torch.Tensor,
    assignments: torch.Tensor,
    *,
    num_patterns: int,
    grams: Optional[torch.Tensor],
    ridge: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    depth = coefficients.shape[1]
    counts = torch.bincount(assignments, minlength=num_patterns)
    if grams is None:
        sums = coefficients.new_zeros((num_patterns, depth))
        sums.index_add_(0, assignments, coefficients)
        centers = sums / counts.clamp_min(1).to(coefficients.dtype).unsqueeze(1)
        return centers, counts

    matrices = coefficients.new_zeros((num_patterns, depth, depth))
    vectors = coefficients.new_zeros((num_patterns, depth))
    matrices.index_add_(0, assignments, grams)
    vectors.index_add_(0, assignments, (grams @ coefficients.unsqueeze(-1)).squeeze(-1))
    identity = torch.eye(depth, device=coefficients.device, dtype=coefficients.dtype)
    scale = matrices.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1.0)
    matrices = matrices + float(ridge) * scale[:, None, None] * identity
    centers = torch.linalg.solve(matrices, vectors.unsqueeze(-1)).squeeze(-1)
    return centers, counts


def fit_coefficient_patterns(
    coefficients: torch.Tensor,
    *,
    num_patterns: int,
    grams: Optional[torch.Tensor] = None,
    iterations: int = 12,
    seed: int = 0,
    chunk_size: int = 16_384,
    ridge: float = 1e-6,
    initial_patterns: Optional[torch.Tensor] = None,
    progress: Optional[ProgressCallback] = None,
) -> torch.Tensor:
    """Fit coefficient prototypes by Lloyd iterations.

    With support Grams this is generalized Lloyd quantization under a distinct
    positive-definite Mahalanobis metric for every training row.  The center
    update solves ``(sum G_n) p = sum G_n c_n`` inside each cluster.
    """

    if coefficients.ndim != 2 or coefficients.shape[0] <= 0:
        raise ValueError("coefficients must be a non-empty [N,K] tensor")
    if num_patterns <= 0 or num_patterns > coefficients.shape[0]:
        raise ValueError("num_patterns must be in [1, number of coefficient rows]")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if ridge <= 0 or not math.isfinite(float(ridge)):
        raise ValueError("ridge must be finite and positive")
    if grams is not None and tuple(grams.shape) != (
        coefficients.shape[0], coefficients.shape[1], coefficients.shape[1]
    ):
        raise ValueError("grams must have shape [N,K,K]")
    if not torch.isfinite(coefficients).all():
        raise ValueError("coefficients must be finite")

    generator = torch.Generator(device=coefficients.device).manual_seed(int(seed))
    if initial_patterns is None:
        indices = torch.randperm(
            coefficients.shape[0], generator=generator, device=coefficients.device
        )[:num_patterns]
        patterns = coefficients[indices].clone()
    else:
        if tuple(initial_patterns.shape) != (num_patterns, coefficients.shape[1]):
            raise ValueError(
                "initial_patterns must have shape "
                f"{(num_patterns, coefficients.shape[1])}"
            )
        patterns = initial_patterns.to(
            device=coefficients.device, dtype=coefficients.dtype
        ).clone()

    for iteration in range(1, int(iterations) + 1):
        assignments, distances = assign_coefficient_patterns(
            coefficients, patterns, grams=grams, chunk_size=chunk_size
        )
        updated, counts = _updated_pattern_centers(
            coefficients,
            assignments,
            num_patterns=num_patterns,
            grams=grams,
            ridge=ridge,
        )
        populated = counts > 0
        patterns[populated] = updated[populated]
        empty = ~populated
        if empty.any():
            replacements = torch.randint(
                coefficients.shape[0],
                (int(empty.sum()),),
                generator=generator,
                device=coefficients.device,
            )
            patterns[empty] = coefficients[replacements]
        if progress is not None:
            progress(iteration, float(distances.mean()), int(empty.sum()))
    return patterns


def decode_coefficient_patterns(
    pattern_ids: torch.Tensor,
    patterns: torch.Tensor,
) -> torch.Tensor:
    """Look up physical coefficient vectors for an arbitrary token grid."""

    if patterns.ndim != 2 or patterns.shape[0] <= 0:
        raise ValueError("patterns must have shape [V,K]")
    if pattern_ids.numel() and (
        int(pattern_ids.min()) < 0 or int(pattern_ids.max()) >= patterns.shape[0]
    ):
        raise ValueError("pattern_ids contain an index outside the codebook")
    return patterns[pattern_ids.long()]


def reconstruct_pattern_latents(
    atom_ids: torch.Tensor,
    pattern_ids: torch.Tensor,
    patterns: torch.Tensor,
    dictionary_rows: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct dense site latents from support ids and pattern tokens."""

    coefficients = decode_coefficient_patterns(pattern_ids, patterns)
    if atom_ids.shape[:-1] != pattern_ids.shape:
        raise ValueError(
            "atom_ids leading dimensions must match pattern_ids, got "
            f"{tuple(atom_ids.shape)} and {tuple(pattern_ids.shape)}"
        )
    if atom_ids.shape[-1] != patterns.shape[-1]:
        raise ValueError("atom support depth must match coefficient pattern depth")
    atoms = dictionary_rows[atom_ids.long()]
    return (atoms * coefficients.unsqueeze(-1)).sum(dim=-2)
