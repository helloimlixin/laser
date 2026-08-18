#!/usr/bin/env python3
"""Evaluate matched sampler settings for a frozen LSUN-Church Stage-2 model."""

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
    rank,
)


SAMPLERS = (
    {
        "name": "baseline_at1_k250_p1__ct1_k250_p1",
        "atom_temperature": 1.0,
        "atom_top_k": 250,
        "atom_top_p": 1.0,
        "coeff_temperature": 1.0,
        "coeff_top_k": 250,
        "coeff_top_p": 1.0,
    },
    {
        "name": "coeff_nucleus_at1_k250_p1__ct1_k0_p085",
        "atom_temperature": 1.0,
        "atom_top_k": 250,
        "atom_top_p": 1.0,
        "coeff_temperature": 1.0,
        "coeff_top_k": 0,
        "coeff_top_p": 0.85,
    },
    {
        "name": "coeff_nucleus_at1_k250_p1__ct1_k0_p092",
        "atom_temperature": 1.0,
        "atom_top_k": 250,
        "atom_top_p": 1.0,
        "coeff_temperature": 1.0,
        "coeff_top_k": 0,
        "coeff_top_p": 0.92,
    },
    {
        "name": "joint_nucleus_at09_k0_p092__ct1_k0_p085",
        "atom_temperature": 0.9,
        "atom_top_k": 0,
        "atom_top_p": 0.92,
        "coeff_temperature": 1.0,
        "coeff_top_k": 0,
        "coeff_top_p": 0.85,
    },
    {
        "name": "joint_tempered_at095_k250_p095__ct09_k250_p095",
        "atom_temperature": 0.95,
        "atom_top_k": 250,
        "atom_top_p": 0.95,
        "coeff_temperature": 0.9,
        "coeff_top_k": 250,
        "coeff_top_p": 0.95,
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
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument(
        "--settings",
        nargs="+",
        choices=tuple(setting["name"] for setting in SAMPLERS),
        default=None,
        help="Optional subset of the built-in matched sampler settings",
    )
    return parser.parse_args()


def atomic_write_json(payload, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, target)


def set_matched_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    del cache_payload
    expected_meta = {
        "dataset": "lsun_church",
        "num_atoms": 16_384,
        "coeff_vocab_size": 2_048,
        "shape": [8, 8, 4],
    }
    for key, expected in expected_meta.items():
        if cache_meta.get(key) != expected:
            raise ValueError(
                f"token cache {key} mismatch: {cache_meta.get(key)!r} != {expected!r}"
            )
    coeff_scales = [float(value) for value in cache_meta["coeff_scales"]]
    coeff_bin_centers = cache_meta.get("coeff_bin_centers")

    checkpoint = torch.load(
        args.stage2_checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    checkpoint_config = checkpoint.get("config", {})
    required_config = {
        "dataset": "lsun_church",
        "model_preset": "lsun-church-350m",
        "num_atoms": 16_384,
        "sparsity_level": 4,
        "coeff_vocab_size": 2_048,
        "compound_tokens": True,
        "levelwise_var": False,
        "compound_micro_transformer_layers": 2,
        "compound_depth_specific_coeff_heads": True,
    }
    for key, expected in required_config.items():
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
        coeff_scales=coeff_scales,
        soft_target_physical=True,
        coeff_bin_centers=coeff_bin_centers,
        sparsity_level=4,
        attn_resolutions=(8,),
    ).to(device)
    model = build_model(
        16_384 + 2_048,
        16_384,
        compound=True,
        levelwise_var=False,
        coeff_vocab_size=2_048,
        compound_refiner_layers=0,
        compound_geometry_head=False,
        compound_micro_transformer_layers=2,
        compound_depth_specific_coeff_heads=True,
        compound_causal_prefix_state=bool(
            checkpoint_config.get("causal_prefix_state", False)
        ),
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

    selected_names = set(args.settings or ())
    settings = [
        setting for setting in SAMPLERS
        if not selected_names or setting["name"] in selected_names
    ]
    results_path = args.output / "sampler_sweep_results.json"
    results = {
        "stage1_checkpoint": str(args.stage1_checkpoint.resolve()),
        "stage2_checkpoint": str(args.stage2_checkpoint.resolve()),
        "fid_reference_stats": str(args.fid_reference_stats.resolve()),
        "checkpoint": checkpoint_metadata,
        "num_samples": args.num_samples,
        "batch_size_per_rank": args.batch_size,
        "world_size": world,
        "seed_per_setting": args.seed,
        "matched_seed": True,
        "results": [],
    }
    if rank() == 0:
        atomic_write_json(results, results_path)
        print(
            f"Loaded epoch {checkpoint_metadata['epoch']} checkpoint; "
            f"evaluating {len(settings)} matched sampler settings on {world} GPU(s)",
            flush=True,
        )

    for setting in settings:
        if dist.is_initialized():
            dist.barrier()
        set_matched_seed(args.seed)
        torch.cuda.empty_cache()
        started = time.monotonic()
        if rank() == 0:
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
        result = {**setting, "fid": float(fid), "elapsed_seconds": elapsed}
        if rank() == 0:
            results["results"].append(result)
            results["results"].sort(key=lambda item: item["fid"])
            atomic_write_json(results, results_path)
            print(
                f"DONE {setting['name']} fid={fid:.6f} elapsed={elapsed:.1f}s",
                flush=True,
            )
        gc.collect()
        torch.cuda.empty_cache()

    if rank() == 0:
        winner = results["results"][0]
        results["winner"] = winner
        atomic_write_json(results, results_path)
        print(
            f"WINNER {winner['name']} fid={winner['fid']:.6f}; results={results_path}",
            flush=True,
        )
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
