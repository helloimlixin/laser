#!/usr/bin/env python3
"""Fit a compact nonuniform scalar quantizer to a LASER compound cache."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch


def fit_hybrid_lloyd(
    values: torch.Tensor,
    *,
    num_bins: int,
    iterations: int,
    support_min: float,
    support_max: float,
    quantile_blend: float,
) -> torch.Tensor:
    quantiles = torch.linspace(0.0, 1.0, num_bins)
    quantile_centers = torch.quantile(values, quantiles)
    uniform_centers = torch.linspace(support_min, support_max, num_bins)
    centers = (
        quantile_blend * quantile_centers
        + (1.0 - quantile_blend) * uniform_centers
    ).sort().values
    centers[0] = support_min
    centers[-1] = support_max

    for _ in range(iterations):
        boundaries = (centers[:-1] + centers[1:]) * 0.5
        assignments = torch.bucketize(values, boundaries)
        counts = torch.bincount(assignments, minlength=num_bins)
        sums = torch.bincount(assignments, weights=values, minlength=num_bins)
        updated = centers.clone()
        populated = counts > 0
        updated[populated] = sums[populated] / counts[populated]
        updated[0] = support_min
        updated[-1] = support_max
        if (updated - centers).abs().max().item() < 1e-7:
            centers = updated
            break
        centers = updated

    if not torch.all(centers[1:] > centers[:-1]):
        raise RuntimeError("fitted coefficient centers are not strictly increasing")
    return centers


def quantization_metrics(
    coeffs: torch.Tensor,
    centers: torch.Tensor,
    physical_scales: torch.Tensor,
) -> dict:
    boundaries = (centers[:-1] + centers[1:]) * 0.5
    assignments = torch.bucketize(coeffs.contiguous(), boundaries)
    reconstructed = centers[assignments]
    physical_error = (reconstructed - coeffs) * physical_scales.view(1, 2)
    flat_error = physical_error.flatten()
    counts = torch.bincount(assignments.flatten(), minlength=len(centers)).float()
    probabilities = counts[counts > 0] / counts.sum()
    return {
        "physical_mae": float(flat_error.abs().mean()),
        "physical_rmse": float(flat_error.square().mean().sqrt()),
        "physical_p99_absolute_error": float(torch.quantile(flat_error.abs(), 0.99)),
        "physical_max_absolute_error": float(flat_error.abs().max()),
        "occupied_bins": int((counts > 0).sum()),
        "token_entropy_bits": float(
            -(probabilities * probabilities.log2()).sum()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-bins", type=int, default=512)
    parser.add_argument("--fit-samples", type=int, default=2_000_000)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--quantile-blend", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    if args.num_bins < 2:
        parser.error("--num-bins must be at least two")
    if args.fit_samples <= 0 or args.iterations <= 0:
        parser.error("--fit-samples and --iterations must be positive")
    if not 0.0 <= args.quantile_blend <= 1.0:
        parser.error("--quantile-blend must be in [0, 1]")
    if args.input.resolve() == args.output.resolve():
        parser.error("--output must not overwrite the source cache")

    payload = torch.load(
        args.input, map_location="cpu", weights_only=True, mmap=True
    )
    meta = dict(payload["meta"])
    if meta.get("format") != "laser_compound_pairs_v1":
        raise ValueError(f"not a compound-pair cache: {meta.get('format')!r}")
    coeffs = payload["coeffs"].float().reshape(-1, 2)
    if coeffs.shape[-1] != 2 or not torch.isfinite(coeffs).all():
        raise ValueError("expected finite k=2 coefficient rows")
    scales = torch.tensor(meta["coeff_scales"], dtype=torch.float32)
    if scales.shape != (2,) or (scales <= 0).any():
        raise ValueError(f"invalid physical coefficient scales: {scales.tolist()}")

    flattened = coeffs.flatten()
    sample_count = min(args.fit_samples, flattened.numel())
    generator = torch.Generator().manual_seed(args.seed)
    sample_indices = torch.randperm(flattened.numel(), generator=generator)[:sample_count]
    fit_values = flattened[sample_indices].contiguous()
    support_min = float(flattened.min())
    support_max = float(flattened.max())
    centers = fit_hybrid_lloyd(
        fit_values,
        num_bins=args.num_bins,
        iterations=args.iterations,
        support_min=support_min,
        support_max=support_max,
        quantile_blend=args.quantile_blend,
    )

    fitted_metrics = quantization_metrics(coeffs, centers, scales)
    uniform_centers = torch.linspace(
        -float(meta["coeff_max"]), float(meta["coeff_max"]), args.num_bins
    )
    uniform_metrics = quantization_metrics(coeffs, uniform_centers, scales)
    report = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "items": int(len(payload["labels"])),
        "coefficients": int(coeffs.numel()),
        "num_bins": args.num_bins,
        "fit_samples": sample_count,
        "iterations": args.iterations,
        "quantile_blend": args.quantile_blend,
        "seed": args.seed,
        "support": [support_min, support_max],
        "coeff_scales": [float(value) for value in scales],
        "fitted": fitted_metrics,
        "uniform_same_size": uniform_metrics,
        "passed": bool(
            len(centers) == args.num_bins
            and torch.isfinite(centers).all()
            and torch.all(centers[1:] > centers[:-1])
            and fitted_metrics["occupied_bins"] == args.num_bins
            and math.isfinite(fitted_metrics["physical_rmse"])
        ),
    }
    if not report["passed"]:
        raise RuntimeError(f"coefficient quantizer validation failed: {report}")

    quantization = {
        "type": "shared_hybrid_lloyd_max",
        "num_bins": args.num_bins,
        "fit_samples": sample_count,
        "iterations": args.iterations,
        "quantile_blend": args.quantile_blend,
        "seed": args.seed,
        "source_coeff_vocab_size": int(meta["coeff_vocab_size"]),
    }
    meta.update({
        "coeff_vocab_size": args.num_bins,
        "coeff_bin_centers": [float(value) for value in centers],
        "coeff_quantization": quantization,
    })
    output_payload = {
        "atoms": payload["atoms"],
        "coeffs": payload["coeffs"],
        "labels": payload["labels"],
        "meta": meta,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(output_payload, temporary_output)
    os.replace(temporary_output, args.output)

    report_path = args.report or args.output.with_suffix(".validation.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_report.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary_report, report_path)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
