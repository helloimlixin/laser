#!/usr/bin/env python3
"""Convert final OMP coefficients into causal ordered-orthogonal coordinates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.tools.build_coefficient_pattern_cache import load_dictionary  # noqa: E402
from src.orthogonal_sparse_codec import (  # noqa: E402
    dictionary_to_orthogonal_coefficients,
)


FORMAT = "laser_orthogonal_compound_v1"
SUPPORTED_INPUTS = {
    "laser_compound_pairs_v1",
    "laser_compound_causal_prefix_v2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument(
        "--max-items", type=int, default=0,
        help="Optional deterministic leading-row slice; 0 converts all rows",
    )
    parser.add_argument(
        "--scale-percentile", type=float, default=100.0,
        help="Absolute gamma percentile mapped to coeff_max independently by depth",
    )
    parser.add_argument("--verify-sites", type=int, default=65_536)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"input does not exist: {args.input}")
    if args.output.exists() and not args.force:
        parser.error(f"output exists (pass --force to replace): {args.output}")
    if args.input.resolve() == args.output.resolve():
        parser.error("output must not overwrite input")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if args.chunk_rows <= 0 or args.verify_sites <= 0:
        parser.error("chunk rows and verify sites must be positive")
    if args.max_items < 0:
        parser.error("max items cannot be negative")
    if not 0 < args.scale_percentile <= 100:
        parser.error("scale percentile must be in (0, 100]")
    return args


def atomic_torch_save(payload: dict, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    torch.save(payload, temporary)
    os.replace(temporary, target)


def atomic_json_save(payload: dict, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, target)


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    source = torch.load(
        args.input, map_location="cpu", weights_only=True, mmap=True
    )
    for key in ("atoms", "coeffs", "labels", "meta"):
        if key not in source:
            raise ValueError(f"input cache is missing {key!r}")
    meta = dict(source["meta"])
    if meta.get("format") not in SUPPORTED_INPUTS:
        raise ValueError(
            f"unsupported input format {meta.get('format')!r}; "
            f"expected one of {sorted(SUPPORTED_INPUTS)}"
        )
    atoms = source["atoms"]
    coefficients = source["coeffs"]
    labels = source["labels"]
    if atoms.shape != coefficients.shape or atoms.ndim != 4:
        raise ValueError("atoms and coefficients must match [N,H,W,K]")
    if len(labels) != len(atoms):
        raise ValueError("input cache row counts do not match")
    items = len(atoms) if args.max_items == 0 else min(args.max_items, len(atoms))
    atoms = atoms[:items]
    coefficients = coefficients[:items]
    labels = labels[:items]
    depth = int(atoms.shape[-1])
    old_scales = torch.as_tensor(meta.get("coeff_scales"), dtype=torch.float32)
    if old_scales.shape != (depth,) or (old_scales <= 0).any():
        raise ValueError("input cache must contain positive per-depth coeff_scales")
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint_value = meta.get("stage1_checkpoint")
        if not checkpoint_value:
            raise ValueError("checkpoint is absent from arguments and cache metadata")
        checkpoint = Path(checkpoint_value)
    dictionary = load_dictionary(
        checkpoint, num_atoms=int(meta["num_atoms"])
    ).to(device)

    physical_gamma = torch.empty(coefficients.shape, dtype=torch.float32)
    max_absolute = torch.zeros(depth, dtype=torch.float32, device=device)
    sampled_absolute = []
    sample_budget = 2_000_000 if args.scale_percentile < 100 else 0
    sample_stride = max(items // max(args.chunk_rows, 1), 1)
    degeneracies = 0
    for start in range(0, items, args.chunk_rows):
        stop = min(start + args.chunk_rows, items)
        chunk_atoms = atoms[start:stop].to(device=device, dtype=torch.long)
        support = dictionary[chunk_atoms]
        physical_coefficients = (
            coefficients[start:stop].to(device=device, dtype=torch.float32)
            * old_scales.to(device).view(1, 1, 1, depth)
        )
        flat_support = support.reshape(-1, depth, support.shape[-1])
        flat_coefficients = physical_coefficients.reshape(-1, depth)
        gamma, basis = dictionary_to_orthogonal_coefficients(
            flat_support, flat_coefficients
        )
        del basis
        gamma = gamma.reshape(stop - start, *atoms.shape[1:])
        physical_gamma[start:stop].copy_(gamma.cpu())
        max_absolute = torch.maximum(
            max_absolute, gamma.abs().reshape(-1, depth).amax(dim=0)
        )
        if sample_budget and len(sampled_absolute) * args.chunk_rows < sample_budget:
            sampled_absolute.append(gamma.abs().reshape(-1, depth).cpu())
        if start == 0 or stop == items or (start // args.chunk_rows) % 100 == 0:
            print(
                f"orthogonal cache {stop:,}/{items:,} rows; "
                f"max_abs={max_absolute.cpu().tolist()}",
                flush=True,
            )
        del chunk_atoms, support, physical_coefficients, flat_support
        del flat_coefficients, gamma

    coeff_max = float(meta["coeff_max"])
    if args.scale_percentile == 100:
        scale_numerators = max_absolute.cpu()
    else:
        absolute = torch.cat(sampled_absolute)[:sample_budget]
        scale_numerators = torch.quantile(
            absolute, args.scale_percentile / 100.0, dim=0
        )
    scales = scale_numerators / coeff_max
    if not torch.isfinite(scales).all() or (scales <= 0).any():
        raise RuntimeError(f"invalid orthogonal coefficient scales: {scales}")
    normalized_gamma = (
        physical_gamma / scales.view(1, 1, 1, depth)
    ).clamp(-coeff_max, coeff_max).to(torch.float16)

    total_sites = items * int(atoms.shape[1]) * int(atoms.shape[2])
    verify_sites = min(args.verify_sites, total_sites)
    verify_indices = torch.linspace(
        0, total_sites - 1, verify_sites, dtype=torch.long
    )
    flat_atoms = atoms.reshape(-1, depth)[verify_indices].to(
        device=device, dtype=torch.long
    )
    support = dictionary[flat_atoms]
    source_physical = (
        coefficients.reshape(-1, depth)[verify_indices].to(
            device=device, dtype=torch.float32
        ) * old_scales.to(device)
    )
    gamma_physical = physical_gamma.reshape(-1, depth)[verify_indices].to(device)
    _, basis = dictionary_to_orthogonal_coefficients(support, source_physical)
    source_latents = torch.einsum("...k,...kc->...c", source_physical, support)
    orthogonal_latents = torch.einsum("...k,...kc->...c", gamma_physical, basis)
    continuous_error = orthogonal_latents - source_latents
    bins = torch.linspace(-coeff_max, coeff_max, int(meta["coeff_vocab_size"]), device=device)
    normalized_verify = normalized_gamma.reshape(-1, depth)[verify_indices].to(
        device=device, dtype=torch.float32
    )
    nearest = (normalized_verify[..., None] - bins).abs().argmin(dim=-1)
    quantized_gamma = bins[nearest] * scales.to(device)
    quantized_latents = torch.einsum("...k,...kc->...c", quantized_gamma, basis)
    quantized_error = quantized_latents - source_latents
    report = {
        "passed": bool(
            continuous_error.abs().max() < 2e-4
            and torch.isfinite(quantized_error).all()
        ),
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "items": items,
        "sites": total_sites,
        "verify_sites": verify_sites,
        "coefficient_scales": [float(value) for value in scales],
        "continuous_latent_mae": float(continuous_error.abs().mean()),
        "continuous_latent_max_error": float(continuous_error.abs().max()),
        "quantized_latent_mse": float(quantized_error.square().mean()),
        "quantized_latent_rmse": float(quantized_error.square().mean().sqrt()),
        "elapsed_seconds": time.monotonic() - started,
    }
    if not report["passed"]:
        raise RuntimeError(f"orthogonal cache validation failed: {report}")

    output_meta = {
        **meta,
        "format": FORMAT,
        "items": items,
        "coordinate_system": "ordered_cholesky_orthogonal",
        "coeff_scales": [float(value) for value in scales],
        "source_coeff_scales": [float(value) for value in old_scales],
        "orthogonal_scale_percentile": float(args.scale_percentile),
        "orthogonal_source_cache": str(args.input.resolve()),
        "causal_prefix_coeffs": True,
        "causal_prefix_coeff_units": "orthogonal_projection",
    }
    atomic_torch_save(
        {
            "atoms": atoms.contiguous(),
            "coeffs": normalized_gamma.contiguous(),
            "labels": labels.contiguous(),
            "meta": output_meta,
        },
        args.output,
    )
    report["elapsed_seconds"] = time.monotonic() - started
    atomic_json_save(report, args.output.with_suffix(".validation.json"))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
