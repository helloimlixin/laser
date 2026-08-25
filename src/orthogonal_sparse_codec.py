"""Causal orthogonal coordinates for ordered sparse dictionary supports.

For row-major support vectors ``A = D_S.T`` and ``A A.T = L L.T``, define
``Q.T = L^-1 A`` and ``gamma = L.T c``.  This keeps the represented latent
exact, ``D_S c = Q gamma``, while making each orthogonal basis vector depend
only on the ordered support prefix through the same depth.
"""

from __future__ import annotations

import torch


def ordered_support_basis(
    support_vectors: torch.Tensor,
    *,
    diagonal_epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return prefix-stable basis rows and the lower support Cholesky factor.

    Args:
        support_vectors: Tensor with shape ``[..., K, C]``.
        diagonal_epsilon: Minimum squared innovation used only to keep
            degenerate generated supports finite. It does not affect ordinary
            OMP supports whose innovation is larger than this threshold.
    """
    if support_vectors.ndim < 2:
        raise ValueError("support vectors must have shape [..., K, C]")
    if support_vectors.shape[-2] <= 0:
        raise ValueError("support depth must be positive")
    if diagonal_epsilon <= 0:
        raise ValueError("diagonal epsilon must be positive")

    vectors = support_vectors.float()
    basis_rows = []
    depth = int(vectors.shape[-2])
    lower = vectors.new_zeros(*vectors.shape[:-2], depth, depth)
    for index in range(depth):
        candidate = vectors[..., index, :]
        if basis_rows:
            previous = torch.stack(basis_rows, dim=-2)
            projections = torch.einsum("...kc,...c->...k", previous, candidate)
            residual = candidate - torch.einsum(
                "...k,...kc->...c", projections, previous
            )
            lower[..., index, :index] = projections
        else:
            residual = candidate
        innovation = residual.square().sum(dim=-1).clamp_min(
            float(diagonal_epsilon)
        ).sqrt()
        lower[..., index, index] = innovation
        basis_rows.append(residual / innovation.unsqueeze(-1))
    return torch.stack(basis_rows, dim=-2), lower


def dictionary_to_orthogonal_coefficients(
    support_vectors: torch.Tensor,
    dictionary_coefficients: torch.Tensor,
    *,
    diagonal_epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert physical dictionary coefficients to causal orthogonal ones."""
    if support_vectors.shape[:-1] != dictionary_coefficients.shape:
        raise ValueError(
            "support and coefficient shapes must agree through the depth axis"
        )
    basis, lower = ordered_support_basis(
        support_vectors, diagonal_epsilon=diagonal_epsilon
    )
    gamma = torch.einsum(
        "...ji,...j->...i", lower, dictionary_coefficients.float()
    )
    return gamma, basis


def orthogonal_to_dictionary_coefficients(
    support_vectors: torch.Tensor,
    orthogonal_coefficients: torch.Tensor,
    *,
    diagonal_epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert causal orthogonal coefficients back to dictionary coordinates."""
    if support_vectors.shape[:-1] != orthogonal_coefficients.shape:
        raise ValueError(
            "support and coefficient shapes must agree through the depth axis"
        )
    basis, lower = ordered_support_basis(
        support_vectors, diagonal_epsilon=diagonal_epsilon
    )
    coefficients = torch.linalg.solve_triangular(
        lower.transpose(-1, -2),
        orthogonal_coefficients.float().unsqueeze(-1),
        upper=True,
    ).squeeze(-1)
    return coefficients, basis


def reconstruct_from_orthogonal(
    support_vectors: torch.Tensor,
    orthogonal_coefficients: torch.Tensor,
    *,
    diagonal_epsilon: float = 1e-8,
) -> torch.Tensor:
    """Reconstruct latent vectors directly from orthogonal coordinates."""
    basis, _ = ordered_support_basis(
        support_vectors, diagonal_epsilon=diagonal_epsilon
    )
    return torch.einsum(
        "...k,...kc->...c", orthogonal_coefficients.float(), basis
    )
