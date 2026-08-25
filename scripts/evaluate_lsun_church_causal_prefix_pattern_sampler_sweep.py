#!/usr/bin/env python3
"""Matched sampler sweep for an LSUN-Church causal-prefix-pattern model."""

from __future__ import annotations

import argparse
from datetime import timedelta
import gc
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "rq-vae-transformer"))

from scripts.train_official_rqtransformer_laser_stage2 import (  # noqa: E402
    LaserAux,
    build_model,
    evaluate_generation_metrics,
)


SAMPLERS = (
    {
        "name": "baseline_at1_p092__pt1_p092",
        "atom_temperature": 1.0,
        "atom_top_k": 0,
        "atom_top_p": 0.92,
        "coeff_temperature": 1.0,
        "coeff_top_k": 0,
        "coeff_top_p": 0.92,
    },
    {
        "name": "pattern_full_at1_p092__pt1_p1",
        "atom_temperature": 1.0,
        "atom_top_k": 0,
        "atom_top_p": 0.92,
        "coeff_temperature": 1.0,
        "coeff_top_k": 0,
        "coeff_top_p": 1.0,
    },
    {
        "name": "pattern_p075_at1_p092__pt1_p075",
        "atom_temperature": 1.0,
        "atom_top_k": 0,
        "atom_top_p": 0.92,
        "coeff_temperature": 1.0,
        "coeff_top_k": 0,
        "coeff_top_p": 0.75,
    },
    {
        "name": "pattern_cool_at1_p092__pt085_p092",
        "atom_temperature": 1.0,
        "atom_top_k": 0,
        "atom_top_p": 0.92,
        "coeff_temperature": 0.85,
        "coeff_top_k": 0,
        "coeff_top_p": 0.92,
    },
    {
        "name": "atom_cool_at09_p092__pt1_p092",
        "atom_temperature": 0.9,
        "atom_top_k": 0,
        "atom_top_p": 0.92,
        "coeff_temperature": 1.0,
        "coeff_top_k": 0,
        "coeff_top_p": 0.92,
    },
    {
        "name": "atom_p085_at1_p085__pt1_p092",
        "atom_temperature": 1.0,
        "atom_top_k": 0,
        "atom_top_p": 0.85,
        "coeff_temperature": 1.0,
        "coeff_top_k": 0,
        "coeff_top_p": 0.92,
    },
    {
        "name": "joint_cool_at09_p092__pt085_p092",
        "atom_temperature": 0.9,
        "atom_top_k": 0,
        "atom_top_p": 0.92,
        "coeff_temperature": 0.85,
        "coeff_top_k": 0,
        "coeff_top_p": 0.92,
    },
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--stage2-checkpoint", type=Path, required=True)
    parser.add_argument("--token-cache", type=Path, required=True)
    parser.add_argument("--fid-reference-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument(
        "--settings",
        nargs="+",
        choices=tuple(setting["name"] for setting in SAMPLERS),
        default=None,
    )
    return parser.parse_args()


def atomic_write_json(payload, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, target)


def process_rank():
    return dist.get_rank() if dist.is_initialized() else 0


def set_matched_seed(seed: int):
    # Give ranks distinct streams, then replay those exact streams for every
    # setting so sampler comparisons are matched rather than merely similar.
    rank_seed = int(seed) + 100_003 * process_rank()
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed(rank_seed)


def validate_inputs(args):
    for path in (
        args.stage1_checkpoint,
        args.stage2_checkpoint,
        args.token_cache,
        args.fid_reference_stats,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.seed < 0:
        raise ValueError("--seed cannot be negative")


def main():
    args = parse_args()
    validate_inputs(args)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
        dist.init_process_group("nccl", timeout=timedelta(minutes=45))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    cache_payload = torch.load(
        args.token_cache, map_location="cpu", weights_only=True, mmap=True
    )
    cache_meta = dict(cache_payload["meta"])
    prefix_patterns = cache_payload["prefix_patterns"]
    prefix_pattern_vocab_sizes = [
        int(value) for value in cache_payload["prefix_pattern_vocab_sizes"]
    ]
    expected_cache = {
        "format": "laser_causal_prefix_patterns_v1",
        "dataset": "lsun_church",
        "num_atoms": 16_384,
        "coeff_vocab_size": 2_048,
        "shape": [8, 8, 4],
        "prefix_pattern_vocab_sizes": prefix_pattern_vocab_sizes,
        "prefix_pattern_units": "physical_omp_least_squares",
    }
    for key, expected in expected_cache.items():
        if cache_meta.get(key) != expected:
            raise ValueError(
                f"token cache {key} mismatch: {cache_meta.get(key)!r} != {expected!r}"
            )

    checkpoint = torch.load(
        args.stage2_checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    checkpoint_config = dict(checkpoint.get("config", {}))
    expected_checkpoint = {
        "dataset": "lsun_church",
        "model_preset": "lsun-church-350m",
        "num_atoms": 16_384,
        "sparsity_level": 4,
        "coeff_vocab_size": 2_048,
        "causal_prefix_pattern_tokens": True,
        "support_first_pattern_tokens": False,
        "compound_tokens": False,
        "levelwise_var": False,
        "compound_micro_transformer_layers": 2,
    }
    for key, expected in expected_checkpoint.items():
        if checkpoint_config.get(key) != expected:
            raise ValueError(
                f"stage-2 checkpoint {key} mismatch: "
                f"{checkpoint_config.get(key)!r} != {expected!r}"
            )

    aux = LaserAux(
        args.stage1_checkpoint,
        num_atoms=16_384,
        coeff_vocab_size=2_048,
        coeff_max=float(cache_meta["coeff_max"]),
        coeff_scale=float(cache_meta.get("coeff_scale", 6.4)),
        coeff_scales=[float(value) for value in cache_meta["coeff_scales"]],
        soft_target_physical=True,
        sparsity_level=4,
        attn_resolutions=(8,),
        prefix_patterns=prefix_patterns,
        prefix_pattern_vocab_sizes=prefix_pattern_vocab_sizes,
    ).to(device)
    del cache_payload, prefix_patterns

    model = build_model(
        16_384 + 2_048,
        16_384,
        compound=False,
        levelwise_var=False,
        coeff_vocab_size=2_048,
        compound_micro_transformer_layers=2,
        causal_prefix_patterns=True,
        prefix_pattern_vocab_sizes=prefix_pattern_vocab_sizes,
        sparsity_level=4,
        model_preset="lsun-church-350m",
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    checkpoint_metadata = {
        "epoch": int(checkpoint["epoch"]),
        "global_step": int(checkpoint["global_step"]),
        "saved_fid": float(checkpoint["fid"]),
        "saved_fid_num_samples": int(checkpoint_config.get("fid_num_samples", 0)),
    }
    del checkpoint
    model = model.to(device).eval().requires_grad_(False)

    selected = set(args.settings or ())
    settings = [
        setting for setting in SAMPLERS
        if not selected or setting["name"] in selected
    ]
    results_path = args.output / "sampler_sweep_results.json"
    results = {
        "stage1_checkpoint": str(args.stage1_checkpoint.resolve()),
        "stage2_checkpoint": str(args.stage2_checkpoint.resolve()),
        "token_cache": str(args.token_cache.resolve()),
        "fid_reference_stats": str(args.fid_reference_stats.resolve()),
        "checkpoint": checkpoint_metadata,
        "num_samples": args.num_samples,
        "batch_size_per_rank": args.batch_size,
        "world_size": world,
        "base_seed": args.seed,
        "rank_seed_stride": 100_003,
        "matched_seed": True,
        "results": [],
    }
    if process_rank() == 0:
        atomic_write_json(results, results_path)
        print(
            f"Loaded epoch {checkpoint_metadata['epoch']}; evaluating "
            f"{len(settings)} settings on {world} GPU(s)",
            flush=True,
        )

    for setting in settings:
        if dist.is_initialized():
            dist.barrier()
        set_matched_seed(args.seed)
        torch.cuda.empty_cache()
        started = time.monotonic()
        if process_rank() == 0:
            print(f"START {setting['name']}", flush=True)
        fid, _, _ = evaluate_generation_metrics(
            model,
            aux,
            val_loader=None,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            num_condition_classes=1,
            compute_inception_score=False,
            metric_backend="original-rqvae",
            fid_reference_stats=args.fid_reference_stats,
            **{key: value for key, value in setting.items() if key != "name"},
        )
        elapsed = time.monotonic() - started
        if process_rank() == 0:
            results["results"].append(
                {**setting, "fid": float(fid), "elapsed_seconds": elapsed}
            )
            results["results"].sort(key=lambda item: item["fid"])
            atomic_write_json(results, results_path)
            print(
                f"DONE {setting['name']} fid={fid:.6f} elapsed={elapsed:.1f}s",
                flush=True,
            )
        gc.collect()
        torch.cuda.empty_cache()

    if process_rank() == 0:
        winner = results["results"][0]
        results["winner"] = winner
        atomic_write_json(results, results_path)
        print(
            f"WINNER {winner['name']} fid={winner['fid']:.6f}; "
            f"results={results_path}",
            flush=True,
        )
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
