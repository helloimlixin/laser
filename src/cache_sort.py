"""Canonical ordering helpers for sparse token caches.

Two within-site support orderings are supported:

* ``"atom_id"`` — atoms sorted ascending by id. Pairs with the prior's
  monotonic atom-id generation mask (enforces uniqueness + order).
* ``"magnitude"`` — atoms sorted by descending ``|coefficient|`` (matching-pursuit
  order: the dominant component first). The prior treats this like an unconstrained
  ("none") order at generation time. Far more learnable than atom_id because the
  most predictable / highest-energy atom is emitted first and early-token errors
  matter less.
"""

from __future__ import annotations

import torch

SUPPORT_ORDERS = ("atom_id", "magnitude")


def _atom_id_order(atom_ids: torch.Tensor) -> torch.Tensor:
    """argsort indices that put atoms in ascending id order (position tie-break)."""
    depth = int(atom_ids.shape[-1])
    idx = torch.arange(depth, device=atom_ids.device, dtype=torch.long)
    idx = idx.view(*([1] * (atom_ids.ndim - 1)), depth)
    return (atom_ids.to(torch.long) * depth + idx).argsort(dim=-1)


def _magnitude_order(atom_ids: torch.Tensor, magnitudes: torch.Tensor) -> torch.Tensor:
    """argsort indices for descending |coeff|, ascending atom-id as the tie-break.

    Deterministic regardless of the encoder's emission order: we first establish an
    atom-id ascending base order, then stably re-sort by descending magnitude.
    """
    base = _atom_id_order(atom_ids)                      # atom-id ascending
    mag_base = magnitudes.gather(-1, base)
    rank = torch.argsort(-mag_base, dim=-1, stable=True)  # |coeff| descending, stable
    return base.gather(-1, rank)


def _resolve_order(order: str) -> str:
    order = str(order or "atom_id").strip().lower()
    if order == "none":
        order = "atom_id"
    if order not in SUPPORT_ORDERS:
        raise ValueError(f"Unsupported support_order {order!r}; expected one of {SUPPORT_ORDERS}")
    return order


def sort_sparse_pairs(
    atom_ids: torch.Tensor,
    values: torch.Tensor,
    *,
    order: str = "atom_id",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reorder real-valued (atom_id, coeff) pairs along the last (depth) axis."""
    if atom_ids.shape != values.shape:
        raise ValueError(f"shape mismatch: {tuple(atom_ids.shape)} vs {tuple(values.shape)}")
    order = _resolve_order(order)
    if order == "magnitude":
        perm = _magnitude_order(atom_ids, values.abs())
    else:
        perm = _atom_id_order(atom_ids)
    return atom_ids.gather(-1, perm), values.gather(-1, perm)


def sort_token_pairs(
    tokens: torch.Tensor,
    *,
    order: str = "atom_id",
    coeff_bin_values: torch.Tensor | None = None,
    atom_vocab_size: int | None = None,
) -> torch.Tensor:
    """Reorder interleaved ``[atom, coeff_bin]`` token pairs along the depth axis.

    For ``order="magnitude"`` the coeff bins are dequantized via ``coeff_bin_values``
    (offset by ``atom_vocab_size``) so the ordering reflects true coefficient
    magnitude under any quantization scheme.
    """
    if int(tokens.shape[-1]) % 2 != 0:
        raise ValueError(f"Expected even token depth, got {int(tokens.shape[-1])}")
    order = _resolve_order(order)
    atom_ids = tokens[..., 0::2]
    pair_vals = tokens[..., 1::2]
    if order == "magnitude":
        if coeff_bin_values is None or atom_vocab_size is None:
            raise ValueError(
                "order='magnitude' on quantized tokens requires coeff_bin_values and atom_vocab_size"
            )
        bin_mags = coeff_bin_values.detach().reshape(-1).abs().to(pair_vals.device)
        bins = (pair_vals.to(torch.long) - int(atom_vocab_size)).clamp_(0, bin_mags.numel() - 1)
        perm = _magnitude_order(atom_ids, bin_mags[bins])
    else:
        perm = _atom_id_order(atom_ids)
    out = tokens.clone()
    out[..., 0::2] = atom_ids.gather(-1, perm)
    out[..., 1::2] = pair_vals.gather(-1, perm)
    return out


def canonicalize_cache(cache: dict, *, order: str | None = None) -> dict:
    """Re-sort an in-memory cache. ``order`` defaults to the cache's stamped order."""
    h, w, d = (int(x) for x in cache["shape"])
    out = dict(cache)
    meta = dict(cache.get("meta", {}) or {})
    order = _resolve_order(order if order is not None else meta.get("support_order", "atom_id"))
    toks = cache["tokens_flat"]
    if cache.get("coeffs_flat") is not None:
        coeffs = cache["coeffs_flat"]
        toks, coeffs = sort_sparse_pairs(
            toks.view(-1, h, w, d), coeffs.view(-1, h, w, d), order=order
        )
        out["tokens_flat"] = toks.reshape(toks.shape[0], -1).contiguous()
        out["coeffs_flat"] = coeffs.reshape(coeffs.shape[0], -1).contiguous()
    else:
        coeff_bin_values = meta.get("coeff_bin_values")
        atom_vocab_size = meta.get("num_atoms") or meta.get("num_embeddings")
        toks = sort_token_pairs(
            toks.view(-1, h, w, d),
            order=order,
            coeff_bin_values=(
                torch.as_tensor(coeff_bin_values) if coeff_bin_values is not None else None
            ),
            atom_vocab_size=atom_vocab_size,
        )
        out["tokens_flat"] = toks.reshape(toks.shape[0], -1).contiguous()
    meta["support_order"] = order
    out["meta"] = meta
    return out
