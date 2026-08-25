#!/usr/bin/env python3
"""Build causal joint-coefficient tokens for every OMP support prefix.

At OMP depth ``d`` the cache contains the exact least-squares coefficients for
the support selected through that depth.  This tool vector-quantizes that
``d + 1`` dimensional tuple under the selected support geometry.  A Stage-2
prior can therefore generate

``atom_0, prefix_pattern_0, ..., atom_K, prefix_pattern_K``

without exposing coefficients or atoms from the future.  The last prefix
pattern is also the final sparse code used by the decoder.
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


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "rq-vae-transformer"))

from scripts.tools.build_coefficient_pattern_cache import (  # noqa: E402
    coefficient_pattern_dtype,
    load_dictionary,
)
from src.coefficient_pattern_codec import (  # noqa: E402
    assign_coefficient_patterns,
    fit_coefficient_patterns,
    selected_support_grams,
)


FORMAT = "laser_causal_prefix_patterns_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--num-patterns",
        type=int,
        nargs="+",
        default=(512, 1_024, 2_048, 4_096),
        help="One vocabulary size per OMP depth (one value repeats at all depths)",
    )
    parser.add_argument("--fit-samples", type=int, default=1_000_000)
    parser.add_argument("--initialization-iterations", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--assignment-chunk-size", type=int, default=16_384)
    parser.add_argument("--gram-chunk-size", type=int, default=65_536)
    parser.add_argument("--cache-chunk-size", type=int, default=65_536)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument(
        "--final-pattern-cache",
        type=Path,
        default=None,
        help=(
            "Reuse the final-depth ids/codebook from a validated "
            "laser_coefficient_patterns_v1 cache"
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"input cache does not exist: {args.input}")
    if args.output.exists() and not args.force:
        parser.error(f"output exists (pass --force to replace it): {args.output}")
    if args.input.resolve() == args.output.resolve():
        parser.error("--output must not overwrite --input")
    if args.final_pattern_cache is not None and not args.final_pattern_cache.is_file():
        parser.error(f"final pattern cache does not exist: {args.final_pattern_cache}")
    if not args.num_patterns or any(value <= 1 for value in args.num_patterns):
        parser.error("--num-patterns values must exceed one")
    if any(value > 2_147_483_647 for value in args.num_patterns):
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


def atomic_torch_save(payload: dict, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    torch.save(payload, temporary)
    os.replace(temporary, target)


def atomic_json_save(payload: dict, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, target)


def normalized_vocab_sizes(values: list[int], depth: int) -> list[int]:
    if len(values) == 1:
        return [int(values[0])] * int(depth)
    if len(values) != depth:
        raise ValueError(
            f"--num-patterns needs one value or {depth} values, got {len(values)}"
        )
    return [int(value) for value in values]


def validate_reused_final_cache(
    payload: dict,
    *,
    atoms: torch.Tensor,
    labels: torch.Tensor,
    expected_vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    meta = dict(payload.get("meta", {}))
    if meta.get("format") != "laser_coefficient_patterns_v1":
        raise ValueError(
            "--final-pattern-cache must use laser_coefficient_patterns_v1"
        )
    patterns = payload.get("coefficient_patterns")
    ids = payload.get("coefficient_pattern_ids")
    if not torch.is_tensor(patterns) or not torch.is_tensor(ids):
        raise ValueError("final pattern cache is missing ids or patterns")
    if tuple(ids.shape) != tuple(atoms.shape[:-1]):
        raise ValueError("final pattern ids do not align with source cache sites")
    if tuple(patterns.shape) != (expected_vocab_size, atoms.shape[-1]):
        raise ValueError(
            "final pattern codebook shape mismatch: "
            f"{tuple(patterns.shape)} != {(expected_vocab_size, atoms.shape[-1])}"
        )
    reused_atoms = payload.get("atoms")
    reused_labels = payload.get("labels")
    if not torch.is_tensor(reused_atoms) or not torch.equal(reused_atoms, atoms):
        raise ValueError("final pattern cache atom rows differ from source cache")
    if not torch.is_tensor(reused_labels) or not torch.equal(reused_labels, labels):
        raise ValueError("final pattern cache labels differ from source cache")
    return ids.reshape(-1).long(), patterns.float()


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
    for key in ("atoms", "prefix_coeffs", "labels", "meta"):
        if key not in source:
            raise ValueError(f"source cache is missing {key!r}")
    source_meta = dict(source["meta"])
    if source_meta.get("format") != "laser_compound_causal_prefix_v2":
        raise ValueError(
            "source cache must use laser_compound_causal_prefix_v2"
        )
    if source_meta.get("causal_prefix_coeff_units") != (
        "physical_omp_least_squares"
    ):
        raise ValueError("source prefix coefficients must use physical units")

    atoms = source["atoms"]
    prefix_coeffs = source["prefix_coeffs"]
    labels = source["labels"]
    if atoms.ndim != 4:
        raise ValueError("source atoms must have shape [N,H,W,K]")
    if tuple(prefix_coeffs.shape) != (*atoms.shape, atoms.shape[-1]):
        raise ValueError(
            "prefix coefficients must have shape [N,H,W,K,K], got "
            f"{tuple(prefix_coeffs.shape)} for atoms {tuple(atoms.shape)}"
        )
    if len(labels) != len(atoms):
        raise ValueError("source cache tensors have inconsistent row counts")
    if not torch.isfinite(prefix_coeffs).all():
        raise ValueError("source prefix coefficients contain non-finite values")

    height, width, depth = (int(value) for value in atoms.shape[1:])
    vocab_sizes = normalized_vocab_sizes(args.num_patterns, depth)
    total_sites = int(atoms.numel() // depth)
    if any(value > total_sites for value in vocab_sizes):
        raise ValueError("a prefix vocabulary exceeds the number of cached sites")
    max_vocab_size = max(vocab_sizes)
    token_dtype = coefficient_pattern_dtype(max_vocab_size)

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint_value = source_meta.get("stage1_checkpoint")
        if not checkpoint_value:
            raise ValueError("--checkpoint is required when metadata has no checkpoint")
        checkpoint = Path(checkpoint_value)
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    dictionary_rows = load_dictionary(
        checkpoint, num_atoms=int(source_meta["num_atoms"])
    ).to(device)
    flat_atoms = atoms.reshape(-1, depth)
    flat_prefix_coeffs = prefix_coeffs.reshape(-1, depth, depth)
    fit_count = min(int(args.fit_samples), total_sites)
    generator = torch.Generator().manual_seed(int(args.seed))
    fit_indices = torch.randperm(total_sites, generator=generator)[:fit_count]

    reused_final_ids = reused_final_patterns = None
    if args.final_pattern_cache is not None:
        final_payload = torch.load(
            args.final_pattern_cache,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        reused_final_ids, reused_final_patterns = validate_reused_final_cache(
            final_payload,
            atoms=atoms,
            labels=labels,
            expected_vocab_size=vocab_sizes[-1],
        )
        del final_payload

    pattern_ids = torch.empty((total_sites, depth), dtype=token_dtype)
    padded_patterns = torch.zeros(
        depth, max_vocab_size, depth, dtype=torch.float32
    )
    depth_reports = []

    print(
        f"Building causal prefix patterns for {total_sites:,} sites with "
        f"vocabularies {vocab_sizes} on {device}",
        flush=True,
    )
    for depth_index, num_patterns in enumerate(vocab_sizes):
        active_depth = depth_index + 1
        print(
            f"Depth {active_depth}/{depth}: fitting {num_patterns:,} patterns",
            flush=True,
        )
        if depth_index == depth - 1 and reused_final_patterns is not None:
            patterns = reused_final_patterns.to(device)
            all_ids = reused_final_ids
            print("Reusing validated final-depth pattern cache", flush=True)
        else:
            fit_atoms = flat_atoms[fit_indices, :active_depth].to(
                device=device, dtype=torch.long
            )
            fit_coefficients = flat_prefix_coeffs[
                fit_indices, depth_index, :active_depth
            ].to(device=device, dtype=torch.float32)
            fit_grams = selected_support_grams(
                fit_atoms,
                dictionary_rows,
                chunk_size=args.gram_chunk_size,
            )

            def progress(phase: str):
                def report(iteration: int, mse: float, empty: int) -> None:
                    print(
                        f"depth={active_depth} {phase} iteration={iteration} "
                        f"mean_squared_error={mse:.8f} empty_patterns={empty}",
                        flush=True,
                    )

                return report

            patterns = fit_coefficient_patterns(
                fit_coefficients,
                num_patterns=num_patterns,
                iterations=args.initialization_iterations,
                seed=args.seed + depth_index,
                chunk_size=args.assignment_chunk_size,
                ridge=args.ridge,
                progress=progress("physical-L2"),
            )
            patterns = fit_coefficient_patterns(
                fit_coefficients,
                num_patterns=num_patterns,
                grams=fit_grams,
                iterations=args.iterations,
                seed=args.seed + depth_index,
                chunk_size=args.assignment_chunk_size,
                ridge=args.ridge,
                initial_patterns=patterns,
                progress=progress("support-Gram"),
            ).float()
            all_ids = torch.empty(total_sites, dtype=torch.long)

        counts = torch.zeros(num_patterns, dtype=torch.long)
        squared_latent_errors = torch.empty(total_sites, dtype=torch.float32)
        coefficient_abs_sum = torch.zeros(active_depth, dtype=torch.float64)
        coefficient_squared_sum = torch.zeros(active_depth, dtype=torch.float64)
        latent_target_energy = 0.0
        for start in range(0, total_sites, args.cache_chunk_size):
            stop = min(start + args.cache_chunk_size, total_sites)
            local_atoms = flat_atoms[start:stop, :active_depth].to(
                device=device, dtype=torch.long
            )
            local_coefficients = flat_prefix_coeffs[
                start:stop, depth_index, :active_depth
            ].to(device=device, dtype=torch.float32)
            local_grams = selected_support_grams(
                local_atoms,
                dictionary_rows,
                chunk_size=args.gram_chunk_size,
            )
            if depth_index == depth - 1 and reused_final_patterns is not None:
                local_ids = all_ids[start:stop].to(device=device, dtype=torch.long)
                differences = patterns[local_ids] - local_coefficients
                projected = (local_grams @ differences.unsqueeze(-1)).squeeze(-1)
                local_distances = (differences * projected).sum(dim=1).clamp_min_(0)
            else:
                local_ids, local_distances = assign_coefficient_patterns(
                    local_coefficients,
                    patterns,
                    grams=local_grams,
                    chunk_size=args.assignment_chunk_size,
                )
                all_ids[start:stop] = local_ids.cpu()
                differences = patterns[local_ids] - local_coefficients
            gram_coefficients = (
                local_grams @ local_coefficients.unsqueeze(-1)
            ).squeeze(-1)
            local_energy = (local_coefficients * gram_coefficients).sum(dim=1)

            cpu_ids = local_ids.cpu()
            counts += torch.bincount(cpu_ids, minlength=num_patterns)
            squared_latent_errors[start:stop] = local_distances.float().cpu()
            coefficient_abs_sum += differences.abs().sum(dim=0).double().cpu()
            coefficient_squared_sum += differences.square().sum(dim=0).double().cpu()
            latent_target_energy += float(local_energy.double().sum())
            if (
                start == 0
                or stop == total_sites
                or stop // 1_000_000 != start // 1_000_000
            ):
                print(
                    f"Depth {active_depth}: assigned {stop:,}/{total_sites:,}",
                    flush=True,
                )

        probabilities = counts[counts > 0].double() / total_sites
        entropy_bits = float(-(probabilities * probabilities.log2()).sum())
        latent_squared_sum = float(squared_latent_errors.double().sum())
        report = {
            "depth": active_depth,
            "num_patterns": num_patterns,
            "occupied_patterns": int((counts > 0).sum()),
            "pattern_entropy_bits": entropy_bits,
            "pattern_perplexity": float(2.0**entropy_bits),
            "min_pattern_count": int(counts.min()),
            "max_pattern_count": int(counts.max()),
            "physical_coefficient_mae": float(
                coefficient_abs_sum.sum() / (total_sites * active_depth)
            ),
            "physical_coefficient_mae_by_position": (
                coefficient_abs_sum / total_sites
            ).tolist(),
            "physical_coefficient_rmse": float(
                (coefficient_squared_sum.sum() / (total_sites * active_depth)).sqrt()
            ),
            "latent_vector_rmse": math.sqrt(latent_squared_sum / total_sites),
            "relative_latent_rmse": math.sqrt(
                latent_squared_sum / max(latent_target_energy, 1e-30)
            ),
            "latent_error_p50": float(squared_latent_errors.quantile(0.50).sqrt()),
            "latent_error_p95": float(squared_latent_errors.quantile(0.95).sqrt()),
            "latent_error_p99": float(squared_latent_errors.quantile(0.99).sqrt()),
            "latent_error_max": float(squared_latent_errors.max().sqrt()),
            "reused_final_pattern_cache": bool(
                depth_index == depth - 1 and reused_final_patterns is not None
            ),
        }
        depth_reports.append(report)
        pattern_ids[:, depth_index] = all_ids.to(token_dtype)
        padded_patterns[
            depth_index, :num_patterns, :active_depth
        ] = patterns.cpu()
        print(json.dumps(report, indent=2), flush=True)

    passed = bool(
        torch.isfinite(padded_patterns).all()
        and all(
            int(pattern_ids[:, index].min()) >= 0
            and int(pattern_ids[:, index].max()) < vocab_sizes[index]
            for index in range(depth)
        )
    )
    report = {
        "passed": passed,
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "checkpoint": str(checkpoint),
        "final_pattern_cache": (
            str(args.final_pattern_cache.resolve())
            if args.final_pattern_cache is not None
            else None
        ),
        "dataset": source_meta.get("dataset"),
        "items": len(atoms),
        "sites": total_sites,
        "spatial_shape": [height, width],
        "sparsity_level": depth,
        "prefix_pattern_vocab_sizes": vocab_sizes,
        "fit_samples": fit_count,
        "initialization_iterations": args.initialization_iterations,
        "iterations": args.iterations,
        "metric": "support_gram",
        "seed": args.seed,
        "depths": depth_reports,
        "elapsed_seconds": time.monotonic() - started,
    }
    if not passed:
        raise RuntimeError(f"causal prefix-pattern validation failed: {report}")

    metadata = dict(source_meta)
    metadata.update(
        {
            "format": FORMAT,
            "source_cache": str(args.input.resolve()),
            "source_cache_format": source_meta.get("format"),
            "stage1_checkpoint": str(checkpoint),
            "token_shape": [height, width, depth],
            "prefix_pattern_vocab_sizes": vocab_sizes,
            "prefix_pattern_units": "physical_omp_least_squares",
            "prefix_pattern_metric": "support_gram",
            "prefix_pattern_fit_samples": fit_count,
            "prefix_pattern_initialization_iterations": args.initialization_iterations,
            "prefix_pattern_iterations": args.iterations,
            "prefix_pattern_seed": args.seed,
            "prefix_pattern_token_dtype": str(token_dtype).removeprefix("torch."),
            "final_pattern_cache": (
                str(args.final_pattern_cache.resolve())
                if args.final_pattern_cache is not None
                else None
            ),
            "coeff_quantization": {
                "type": "causal_prefix_coefficient_patterns",
                "metric": "support_gram",
                "vocab_sizes": vocab_sizes,
            },
        }
    )
    payload = {
        "atoms": atoms,
        "prefix_pattern_ids": pattern_ids.reshape(
            len(atoms), height, width, depth
        ).contiguous(),
        "prefix_patterns": padded_patterns.contiguous(),
        "prefix_pattern_vocab_sizes": torch.tensor(vocab_sizes, dtype=torch.long),
        "labels": labels,
        "meta": metadata,
    }
    atomic_torch_save(payload, args.output)
    report_path = args.output.with_suffix(".validation.json")
    atomic_json_save(report, report_path)
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved causal prefix-pattern cache to {args.output}", flush=True)


if __name__ == "__main__":
    main()
