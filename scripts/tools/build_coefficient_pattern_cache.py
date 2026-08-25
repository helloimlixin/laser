#!/usr/bin/env python3
"""Build joint coefficient-pattern tokens from a LASER compound cache.

Atom supports are copied exactly.  The ``K`` final OMP coefficients at each
site are converted from cache-normalized units to physical units and replaced
by one vector-quantizer id.  By default, fitting and assignment minimize exact
latent error under each support's Gram matrix ``D_S^T D_S``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "rq-vae-transformer"))

from scripts.train_official_rqtransformer_laser_stage2 import load_stage1_checkpoint
from src.coefficient_pattern_codec import (
    assign_coefficient_patterns,
    fit_coefficient_patterns,
    selected_support_grams,
)


SUPPORTED_SOURCE_FORMATS = frozenset(
    {
        "laser_compound_pairs_v1",
        "laser_compound_causal_prefix_v2",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Stage-1 checkpoint; defaults to source cache metadata",
    )
    parser.add_argument("--num-patterns", type=int, default=4_096)
    parser.add_argument("--fit-samples", type=int, default=1_000_000)
    parser.add_argument(
        "--initialization-iterations",
        type=int,
        default=6,
        help="Physical-L2 Lloyd iterations before support-aware refinement",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=12,
        help="Support-aware Lloyd refinement iterations",
    )
    parser.add_argument(
        "--metric",
        choices=("support_gram", "physical_l2"),
        default="support_gram",
    )
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--assignment-chunk-size", type=int, default=16_384)
    parser.add_argument("--gram-chunk-size", type=int, default=65_536)
    parser.add_argument("--cache-chunk-size", type=int, default=65_536)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.input.resolve() == args.output.resolve():
        parser.error("--output must not overwrite --input")
    if not args.input.is_file():
        parser.error(f"input cache does not exist: {args.input}")
    if args.output.exists() and not args.force:
        parser.error(f"output exists (pass --force to replace it): {args.output}")
    if args.num_patterns <= 1:
        parser.error("--num-patterns must be greater than one")
    if args.num_patterns > 2_147_483_647:
        parser.error("--num-patterns exceeds int32 token capacity")
    if args.fit_samples <= 0:
        parser.error("--fit-samples must be positive")
    if args.initialization_iterations <= 0 or args.iterations <= 0:
        parser.error("iteration counts must be positive")
    if min(
        args.assignment_chunk_size,
        args.gram_chunk_size,
        args.cache_chunk_size,
    ) <= 0:
        parser.error("chunk sizes must be positive")
    if not math.isfinite(args.ridge) or args.ridge <= 0:
        parser.error("--ridge must be finite and positive")
    return args


def load_dictionary(checkpoint: Path, *, num_atoms: int) -> torch.Tensor:
    payload = load_stage1_checkpoint(checkpoint)
    state = payload.get("state_dict", payload)
    dictionary_key = (
        "quantizer.dictionary"
        if "quantizer.dictionary" in state
        else "bottleneck.dictionary"
    )
    if dictionary_key not in state:
        raise RuntimeError("stage-1 checkpoint does not contain a sparse dictionary")
    dictionary = state[dictionary_key].float()
    if dictionary.ndim != 2 or dictionary.shape[1] != int(num_atoms):
        raise ValueError(
            "dictionary shape does not match source cache: "
            f"{tuple(dictionary.shape)} versus num_atoms={num_atoms}"
        )
    return F.normalize(dictionary, dim=0).t().contiguous()


def coefficient_pattern_dtype(num_patterns: int) -> torch.dtype:
    return torch.int16 if int(num_patterns) <= 32_768 else torch.int32


def _atomic_torch_save(payload: dict, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    torch.save(payload, temporary)
    os.replace(temporary, target)


def _atomic_json_save(payload: dict, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
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
            raise ValueError(f"source cache is missing {key!r}")
    source_meta = dict(source["meta"])
    source_format = source_meta.get("format")
    if source_format not in SUPPORTED_SOURCE_FORMATS:
        raise ValueError(
            f"unsupported source cache format {source_format!r}; expected one of "
            f"{sorted(SUPPORTED_SOURCE_FORMATS)}"
        )

    atoms = source["atoms"]
    coefficients = source["coeffs"]
    labels = source["labels"]
    if atoms.shape != coefficients.shape or atoms.ndim != 4:
        raise ValueError(
            "source atoms and coefficients must have matching [N,H,W,K] shapes"
        )
    if len(atoms) != len(labels):
        raise ValueError("source cache tensors have inconsistent row counts")
    height, width, depth = (int(value) for value in atoms.shape[1:])
    if list(source_meta.get("shape", [])) != [height, width, depth]:
        raise ValueError("source cache shape metadata does not match its tensors")
    scales = torch.as_tensor(source_meta.get("coeff_scales"), dtype=torch.float32)
    if scales.shape != (depth,) or not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("source cache must contain positive per-depth coeff_scales")
    if not torch.isfinite(coefficients).all():
        raise ValueError("source coefficients contain non-finite values")

    total_sites = int(coefficients.numel() // depth)
    if args.num_patterns > total_sites:
        raise ValueError("--num-patterns cannot exceed the number of cached sites")
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint_value = source_meta.get("stage1_checkpoint")
        if not checkpoint_value:
            raise ValueError("--checkpoint is required when cache metadata has no checkpoint")
        checkpoint = Path(checkpoint_value)
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    print(
        f"Loading {source_meta.get('dataset')} cache with {total_sites:,} sites; "
        f"fitting {args.num_patterns:,} coefficient patterns on {device}",
        flush=True,
    )
    dictionary_rows = load_dictionary(
        checkpoint, num_atoms=int(source_meta["num_atoms"])
    ).to(device)
    flat_atoms = atoms.reshape(-1, depth)
    flat_coefficients = coefficients.reshape(-1, depth)
    fit_count = min(int(args.fit_samples), total_sites)
    generator = torch.Generator().manual_seed(int(args.seed))
    fit_indices = torch.randperm(total_sites, generator=generator)[:fit_count]
    fit_atoms = flat_atoms[fit_indices].to(device=device, dtype=torch.long)
    fit_coefficients = flat_coefficients[fit_indices].to(
        device=device, dtype=torch.float32
    )
    fit_coefficients.mul_(scales.to(device))

    print(
        f"Computing support geometry for {fit_count:,} fitting sites",
        flush=True,
    )
    fit_grams = selected_support_grams(
        fit_atoms,
        dictionary_rows,
        chunk_size=args.gram_chunk_size,
    )

    def progress(phase: str):
        def report(iteration: int, mean_squared_error: float, empty: int) -> None:
            print(
                f"{phase} iteration {iteration}: mean_squared_error="
                f"{mean_squared_error:.8f}, empty_patterns={empty}",
                flush=True,
            )

        return report

    patterns = fit_coefficient_patterns(
        fit_coefficients,
        num_patterns=args.num_patterns,
        iterations=args.initialization_iterations,
        seed=args.seed,
        chunk_size=args.assignment_chunk_size,
        ridge=args.ridge,
        progress=progress("physical-L2 initialization"),
    )
    if args.metric == "support_gram":
        patterns = fit_coefficient_patterns(
            fit_coefficients,
            num_patterns=args.num_patterns,
            grams=fit_grams,
            iterations=args.iterations,
            seed=args.seed,
            chunk_size=args.assignment_chunk_size,
            ridge=args.ridge,
            initial_patterns=patterns,
            progress=progress("support-Gram refinement"),
        )
    patterns = patterns.float()
    if not torch.isfinite(patterns).all():
        raise RuntimeError("fitted coefficient patterns contain non-finite values")

    token_dtype = coefficient_pattern_dtype(args.num_patterns)
    pattern_ids = torch.empty(total_sites, dtype=token_dtype)
    squared_latent_errors = torch.empty(total_sites, dtype=torch.float32)
    counts = torch.zeros(args.num_patterns, dtype=torch.long)
    coefficient_abs_sum = torch.zeros(depth, dtype=torch.float64)
    coefficient_squared_sum = torch.zeros(depth, dtype=torch.float64)
    latent_target_energy = 0.0

    print("Assigning the complete cache with the fitted codebook", flush=True)
    scale_device = scales.to(device)
    for start in range(0, total_sites, args.cache_chunk_size):
        stop = min(start + args.cache_chunk_size, total_sites)
        local_atoms = flat_atoms[start:stop].to(device=device, dtype=torch.long)
        local_coefficients = flat_coefficients[start:stop].to(
            device=device, dtype=torch.float32
        )
        local_coefficients.mul_(scale_device)
        local_grams = selected_support_grams(
            local_atoms,
            dictionary_rows,
            chunk_size=args.gram_chunk_size,
        )
        assignment_grams = local_grams if args.metric == "support_gram" else None
        local_ids, local_distances = assign_coefficient_patterns(
            local_coefficients,
            patterns,
            grams=assignment_grams,
            chunk_size=args.assignment_chunk_size,
        )
        reconstructed = patterns[local_ids]
        coefficient_error = reconstructed - local_coefficients
        if args.metric != "support_gram":
            latent_error = coefficient_error.unsqueeze(1) @ local_grams
            local_distances = (
                latent_error.squeeze(1) * coefficient_error
            ).sum(dim=1).clamp_min_(0.0)
        gram_coefficients = (
            local_grams @ local_coefficients.unsqueeze(-1)
        ).squeeze(-1)
        local_energy = (local_coefficients * gram_coefficients).sum(dim=1)

        cpu_ids = local_ids.cpu()
        pattern_ids[start:stop] = cpu_ids.to(token_dtype)
        squared_latent_errors[start:stop] = local_distances.float().cpu()
        counts += torch.bincount(cpu_ids, minlength=args.num_patterns)
        coefficient_abs_sum += coefficient_error.abs().sum(dim=0).double().cpu()
        coefficient_squared_sum += coefficient_error.square().sum(dim=0).double().cpu()
        latent_target_energy += float(local_energy.double().sum())
        if start == 0 or stop == total_sites or stop // 1_000_000 != start // 1_000_000:
            print(f"Assigned {stop:,}/{total_sites:,} sites", flush=True)

    probabilities = counts[counts > 0].double() / total_sites
    entropy_bits = float(-(probabilities * probabilities.log2()).sum())
    latent_squared_sum = float(squared_latent_errors.double().sum())
    coefficient_mae_by_depth = (coefficient_abs_sum / total_sites).tolist()
    coefficient_rmse_by_depth = (
        coefficient_squared_sum / total_sites
    ).sqrt().tolist()
    report = {
        "passed": bool(
            int(pattern_ids.min()) >= 0
            and int(pattern_ids.max()) < args.num_patterns
            and torch.isfinite(patterns.cpu()).all()
            and torch.isfinite(squared_latent_errors).all()
            and int(counts.sum()) == total_sites
        ),
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "checkpoint": str(checkpoint),
        "dataset": source_meta.get("dataset"),
        "items": len(atoms),
        "sites": total_sites,
        "spatial_shape": [height, width],
        "sparsity_level": depth,
        "num_patterns": args.num_patterns,
        "fit_samples": fit_count,
        "initialization_iterations": args.initialization_iterations,
        "iterations": args.iterations,
        "metric": args.metric,
        "seed": args.seed,
        "occupied_patterns": int((counts > 0).sum()),
        "pattern_entropy_bits": entropy_bits,
        "pattern_perplexity": float(2.0**entropy_bits),
        "min_pattern_count": int(counts.min()),
        "max_pattern_count": int(counts.max()),
        "physical_coefficient_mae": float(coefficient_abs_sum.sum() / (total_sites * depth)),
        "physical_coefficient_rmse": float(
            (coefficient_squared_sum.sum() / (total_sites * depth)).sqrt()
        ),
        "physical_coefficient_mae_by_depth": coefficient_mae_by_depth,
        "physical_coefficient_rmse_by_depth": coefficient_rmse_by_depth,
        "latent_vector_rmse": math.sqrt(latent_squared_sum / total_sites),
        "relative_latent_rmse": math.sqrt(
            latent_squared_sum / max(latent_target_energy, 1e-30)
        ),
        "latent_error_p50": float(squared_latent_errors.quantile(0.50).sqrt()),
        "latent_error_p95": float(squared_latent_errors.quantile(0.95).sqrt()),
        "latent_error_p99": float(squared_latent_errors.quantile(0.99).sqrt()),
        "latent_error_max": float(squared_latent_errors.max().sqrt()),
        "elapsed_seconds": time.monotonic() - started,
    }
    if not report["passed"]:
        raise RuntimeError(f"coefficient-pattern validation failed: {report}")

    metadata = dict(source_meta)
    metadata.update(
        {
            "format": "laser_coefficient_patterns_v1",
            "source_cache": str(args.input.resolve()),
            "source_cache_format": source_format,
            "stage1_checkpoint": str(checkpoint),
            "token_shape": [height, width],
            "coefficient_pattern_vocab_size": args.num_patterns,
            "coefficient_pattern_units": "physical_omp_least_squares",
            "coefficient_pattern_metric": args.metric,
            "coefficient_pattern_fit_samples": fit_count,
            "coefficient_pattern_initialization_iterations": args.initialization_iterations,
            "coefficient_pattern_iterations": args.iterations,
            "coefficient_pattern_seed": args.seed,
            "coefficient_pattern_token_dtype": str(token_dtype).removeprefix("torch."),
            "coeff_quantization": {
                "type": "joint_coefficient_patterns",
                "metric": args.metric,
                "num_patterns": args.num_patterns,
            },
        }
    )
    output_payload = {
        "atoms": atoms,
        "coefficient_pattern_ids": pattern_ids.reshape(
            len(atoms), height, width
        ).contiguous(),
        "coefficient_patterns": patterns.cpu().contiguous(),
        "labels": labels,
        "meta": metadata,
    }
    _atomic_torch_save(output_payload, args.output)
    report_path = args.output.with_suffix(".validation.json")
    _atomic_json_save(report, report_path)
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved coefficient-pattern cache to {args.output}", flush=True)


if __name__ == "__main__":
    main()
