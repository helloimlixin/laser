"""Sparse token codec used by token/cache extraction."""

import math
from typing import Optional, Sequence, Tuple

import torch

from src.models.bottleneck_utils import SparseCodes


def build_coeff_bin_values(
    *,
    coeff_vocab_size: int,
    coeff_bin_values: Optional[Sequence[float] | torch.Tensor] = None,
    coeff_max: Optional[float] = None,
    coeff_quantization: str = "uniform",
    coeff_mu: float = 0.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if coeff_bin_values is not None:
        values = torch.as_tensor(coeff_bin_values, device=device, dtype=dtype).reshape(-1)
        if values.numel() != int(coeff_vocab_size):
            raise ValueError(
                "coeff_bin_values length must match coeff_vocab_size, "
                f"got {values.numel()} vs {int(coeff_vocab_size)}"
            )
        return values

    coeff_vocab_size = int(coeff_vocab_size)
    if coeff_vocab_size <= 0:
        raise ValueError(f"coeff_vocab_size must be positive, got {coeff_vocab_size}")
    if coeff_max is None:
        raise ValueError("coeff_max is required when coeff_bin_values is not provided")

    coeff_max = float(coeff_max)
    coeff_quantization = str(coeff_quantization).strip().lower()
    if coeff_quantization == "uniform":
        return torch.linspace(-coeff_max, coeff_max, steps=coeff_vocab_size, device=device, dtype=dtype)
    if coeff_quantization != "mu_law":
        raise ValueError(
            "coeff_quantization must be 'uniform' or 'mu_law', got "
            f"{coeff_quantization!r}"
        )

    coeff_mu = float(coeff_mu)
    if coeff_mu <= 0.0:
        raise ValueError(f"coeff_mu must be > 0 for mu-law quantization, got {coeff_mu}")
    if coeff_vocab_size == 1:
        return torch.zeros(1, device=device, dtype=dtype)
    z = torch.linspace(-1.0, 1.0, steps=coeff_vocab_size, device=device, dtype=dtype)
    decoded = torch.sign(z) * (torch.expm1(z.abs() * math.log1p(coeff_mu)) / coeff_mu)
    return decoded * coeff_max


def quantize_sparse_coefficients(
    coeffs: torch.Tensor,
    *,
    coeff_vocab_size: int,
    coeff_max: float,
    coeff_quantization: str = "uniform",
    coeff_mu: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    coeff_vocab_size = int(coeff_vocab_size)
    if coeff_vocab_size <= 0:
        raise ValueError(f"coeff_vocab_size must be positive, got {coeff_vocab_size}")
    coeff_max = float(coeff_max)
    if not math.isfinite(coeff_max) or coeff_max <= 0.0:
        raise ValueError(f"coeff_max must be finite and > 0, got {coeff_max}")

    coeff_quantization = str(coeff_quantization).strip().lower()
    coeffs = coeffs.to(torch.float32)
    clamped = coeffs.clamp(-coeff_max, coeff_max)

    if coeff_vocab_size == 1:
        bin_idx = torch.zeros_like(clamped, dtype=torch.long)
    elif coeff_quantization == "uniform":
        scaled = (clamped + coeff_max) / (2.0 * coeff_max)
        bin_float = scaled * float(coeff_vocab_size - 1)
        bin_idx = torch.round(bin_float).to(torch.long).clamp_(0, coeff_vocab_size - 1)
    elif coeff_quantization == "mu_law":
        coeff_mu = float(coeff_mu)
        if coeff_mu <= 0.0:
            raise ValueError(f"coeff_mu must be > 0 for mu-law quantization, got {coeff_mu}")
        normalized = clamped / coeff_max
        encoded = torch.sign(normalized) * torch.log1p(normalized.abs() * coeff_mu) / math.log1p(coeff_mu)
        bin_float = (encoded + 1.0) * ((coeff_vocab_size - 1) / 2.0)
        bin_idx = torch.round(bin_float).to(torch.long).clamp_(0, coeff_vocab_size - 1)
    else:
        raise ValueError(
            "coeff_quantization must be 'uniform' or 'mu_law', got "
            f"{coeff_quantization!r}"
        )

    values = build_coeff_bin_values(
        coeff_vocab_size=coeff_vocab_size,
        coeff_max=coeff_max,
        coeff_quantization=coeff_quantization,
        coeff_mu=coeff_mu,
        device=coeffs.device,
        dtype=torch.float32,
    )
    return bin_idx, values[bin_idx]


def sparse_codes_to_tokens(
    sparse_codes: SparseCodes,
    *,
    num_embeddings: int,
    sparsity_level: int,
    coeff_vocab_size: int,
    coeff_max: float,
    coeff_quantization: str = "uniform",
    coeff_mu: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    support = sparse_codes.support.to(torch.long).clamp(0, int(num_embeddings) - 1)
    bin_idx, coeff_q = quantize_sparse_coefficients(
        sparse_codes.values,
        coeff_vocab_size=coeff_vocab_size,
        coeff_max=coeff_max,
        coeff_quantization=coeff_quantization,
        coeff_mu=coeff_mu,
    )
    tokens = torch.empty(
        *support.shape[:-1],
        int(sparsity_level) * 2,
        device=support.device,
        dtype=torch.long,
    )
    tokens[..., 0::2] = support
    tokens[..., 1::2] = bin_idx + int(num_embeddings)
    return tokens, coeff_q


def tokens_to_sparse_codes(
    tokens: torch.Tensor,
    *,
    num_embeddings: int,
    sparsity_level: int,
    atom_vocab_size: Optional[int] = None,
    coeff_vocab_size: Optional[int] = None,
    coeff_bin_values: Optional[Sequence[float] | torch.Tensor] = None,
    coeff_max: Optional[float] = None,
    coeff_quantization: str = "uniform",
    coeff_mu: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if tokens.dim() != 4:
        raise ValueError(f"Expected [B,H,W,D], got {tuple(tokens.shape)}")

    expected_token_depth = int(sparsity_level) * 2
    if int(tokens.shape[-1]) != expected_token_depth:
        raise ValueError(
            f"Expected token depth {expected_token_depth}, got {int(tokens.shape[-1])}"
        )

    atom_vocab_size = int(atom_vocab_size or num_embeddings)
    if atom_vocab_size <= 0:
        raise ValueError(f"atom_vocab_size must be positive, got {atom_vocab_size}")

    if coeff_vocab_size is None:
        if coeff_bin_values is None:
            raise ValueError("coeff_vocab_size or coeff_bin_values is required for token decode")
        coeff_vocab_size = int(torch.as_tensor(coeff_bin_values).numel())
    coeff_vocab_size = int(coeff_vocab_size)
    if coeff_vocab_size <= 0:
        raise ValueError(f"coeff_vocab_size must be positive, got {coeff_vocab_size}")

    support = tokens[..., 0::2].to(torch.long).clamp(0, int(num_embeddings) - 1)
    coeff_bins = tokens[..., 1::2].to(torch.long) - atom_vocab_size
    coeff_bins = coeff_bins.clamp(0, coeff_vocab_size - 1)
    bins = build_coeff_bin_values(
        coeff_vocab_size=coeff_vocab_size,
        coeff_bin_values=coeff_bin_values,
        coeff_max=coeff_max,
        coeff_quantization=coeff_quantization,
        coeff_mu=coeff_mu,
        device=tokens.device,
        dtype=torch.float32,
    )
    return support, bins[coeff_bins]
