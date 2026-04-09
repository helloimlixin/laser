"""Canonical ordering helpers for sparse token caches."""

from __future__ import annotations

import torch


def _ord(atom_ids: torch.Tensor) -> torch.Tensor:
    depth = int(atom_ids.shape[-1])
    idx = torch.arange(depth, device=atom_ids.device, dtype=torch.long)
    idx = idx.view(*([1] * (atom_ids.ndim - 1)), depth)
    return (atom_ids.to(torch.long) * depth + idx).argsort(dim=-1)


def sort_sparse_pairs(atom_ids: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if atom_ids.shape != values.shape:
        raise ValueError(f"shape mismatch: {tuple(atom_ids.shape)} vs {tuple(values.shape)}")
    order = _ord(atom_ids)
    return atom_ids.gather(-1, order), values.gather(-1, order)


def sort_token_pairs(tokens: torch.Tensor) -> torch.Tensor:
    if int(tokens.shape[-1]) % 2 != 0:
        raise ValueError(f"Expected even token depth, got {int(tokens.shape[-1])}")
    atom_ids = tokens[..., 0::2]
    pair_vals = tokens[..., 1::2]
    order = _ord(atom_ids)
    out = tokens.clone()
    out[..., 0::2] = atom_ids.gather(-1, order)
    out[..., 1::2] = pair_vals.gather(-1, order)
    return out


def canonicalize_cache(cache: dict) -> dict:
    h, w, d = (int(x) for x in cache["shape"])
    out = dict(cache)
    meta = dict(cache.get("meta", {}) or {})
    toks = cache["tokens_flat"]
    if cache.get("coeffs_flat") is not None:
        coeffs = cache["coeffs_flat"]
        toks, coeffs = sort_sparse_pairs(toks.view(-1, h, w, d), coeffs.view(-1, h, w, d))
        out["tokens_flat"] = toks.reshape(toks.shape[0], -1).contiguous()
        out["coeffs_flat"] = coeffs.reshape(coeffs.shape[0], -1).contiguous()
    else:
        toks = sort_token_pairs(toks.view(-1, h, w, d))
        out["tokens_flat"] = toks.reshape(toks.shape[0], -1).contiguous()
    meta["support_order"] = "atom_id"
    out["meta"] = meta
    return out
