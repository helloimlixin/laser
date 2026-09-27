#!/usr/bin/env python3
"""Evaluate aligned CC3M text FID and CLIP score for a pinned checkpoint."""

from __future__ import annotations

import argparse
from datetime import timedelta
import gc
import json
import os
from pathlib import Path
import random
import sys

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "rq-vae-transformer"))

from scripts.train_official_rqtransformer_laser_stage2 import (  # noqa: E402
    CC3MValidationDataset,
    LaserAux,
    build_model,
    evaluate_text_generation_metrics,
    val_image_transform,
)


def atomic_json(payload: object, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, target)


def setup_distributed() -> tuple[int, int, int, torch.device]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    world_rank = int(os.environ.get("RANK", "0"))
    if world_size > 1:
        os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
        dist.init_process_group("nccl", timeout=timedelta(minutes=45))
        world_rank = dist.get_rank()
        world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return world_rank, world_size, local_rank, torch.device("cuda", local_rank)


class ProgressLoader:
    def __init__(self, loader, *, rank: int, total_local: int):
        self.loader = loader
        self.rank = rank
        self.total_local = total_local

    def __iter__(self):
        seen = 0
        report_every = max(1, len(self.loader) // 8)
        for batch_index, batch in enumerate(self.loader, start=1):
            yield batch
            seen += len(batch[1])
            if self.rank == 0 and (
                batch_index % report_every == 0 or batch_index == len(self.loader)
            ):
                print(
                    f"evaluation_progress local_pairs={seen}/{self.total_local} "
                    f"batches={batch_index}/{len(self.loader)}",
                    flush=True,
                )

    def __len__(self):
        return len(self.loader)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip-cache-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=220301941)
    parser.add_argument("--expected-epoch", type=int, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--wandb-eval-run-id", default=None)
    parser.add_argument("--wandb-entity", default="helloimlixin-rutgers")
    parser.add_argument("--wandb-project", default="laser")
    args = parser.parse_args()

    for path in (args.stage1, args.stage2):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.num_samples < 2 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("invalid evaluation size, batch size, or worker count")

    world_rank, world_size, local_rank, device = setup_distributed()
    random.seed(args.seed + world_rank)
    torch.manual_seed(args.seed + world_rank)
    torch.cuda.manual_seed_all(args.seed + world_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    args.output.mkdir(parents=True, exist_ok=True)
    if world_rank == 0:
        print(
            f"evaluation_start checkpoint={args.stage2} samples={args.num_samples} "
            f"world_size={world_size} batch_per_rank={args.batch_size}",
            flush=True,
        )

    payload = torch.load(
        args.stage2, map_location="cpu", weights_only=False, mmap=True
    )
    checkpoint_epoch = int(payload.get("epoch", -1))
    checkpoint_step = int(payload.get("global_step", -1))
    if (checkpoint_epoch, checkpoint_step) != (
        args.expected_epoch,
        args.expected_step,
    ):
        raise RuntimeError(
            f"expected checkpoint epoch/step {args.expected_epoch}/{args.expected_step}, "
            f"found {checkpoint_epoch}/{checkpoint_step}"
        )
    config = dict(payload.get("config", {}))
    required_config = {
        "compound_tokens": True,
        "compound_micro_transformer_layers": 2,
        "compound_depth_specific_coeff_heads": True,
        "compound_distribution_geometry": True,
        "num_atoms": 16384,
        "coeff_vocab_size": 2048,
    }
    for key, expected in required_config.items():
        actual = config.get(key)
        if actual != expected:
            raise RuntimeError(
                f"checkpoint config {key}: expected {expected!r}, got {actual!r}"
            )
    # The frozen launcher predates the CLI sparsity-level field and hard-coded
    # k=2. Newer checkpoints record it explicitly.
    if int(config.get("sparsity_level", 2)) != 2:
        raise RuntimeError(
            f"checkpoint config sparsity_level: expected 2, got "
            f"{config.get('sparsity_level')!r}"
        )

    validation = CC3MValidationDataset(
        args.data, transform=val_image_transform()
    )
    evaluation_samples = min(len(validation), args.num_samples)
    if evaluation_samples != args.num_samples:
        raise RuntimeError(
            f"requested {args.num_samples} pairs but validation has {len(validation)}"
        )
    exact_rank_indices = range(world_rank, evaluation_samples, world_size)
    val_loader = DataLoader(
        validation,
        batch_size=args.batch_size,
        sampler=exact_rank_indices,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = ProgressLoader(
        val_loader,
        rank=world_rank,
        total_local=len(exact_rank_indices),
    )

    aux = LaserAux(
        args.stage1,
        16384,
        2048,
        20.0,
        6.4,
        attn_resolutions=(8,),
        sparsity_level=2,
    ).to(device).eval()
    model = build_model(
        16384 + 2048,
        16384,
        compound=True,
        coeff_vocab_size=2048,
        compound_micro_transformer_layers=2,
        compound_depth_specific_coeff_heads=True,
        sparsity_level=2,
        model_preset="cc3m-650m",
        num_condition_classes=16384,
        condition_length=32,
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    del payload
    gc.collect()
    model = model.to(device).eval().requires_grad_(False)
    aux.requires_grad_(False)

    fid, clip_score = evaluate_text_generation_metrics(
        model,
        aux,
        val_loader,
        evaluation_samples,
        args.clip_cache_dir,
        atom_temperature=1.0,
        atom_top_k=16384,
        atom_top_p=0.7,
        coeff_temperature=1.0,
        coeff_top_k=2048,
        coeff_top_p=0.7,
    )

    result = {
        "status": "complete",
        "source_run": args.source_run,
        "checkpoint": str(args.stage2.resolve()),
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_step": checkpoint_step,
        "fid": fid,
        "text_fid": fid,
        "clip_score": clip_score,
        "clip_model": "OpenAI CLIP ViT-B/32",
        "num_real": evaluation_samples,
        "num_fake": evaluation_samples,
        "pairing": "one generation per aligned CC3M validation caption",
        "validation_subset": f"first {evaluation_samples} aligned pairs",
        "world_size": world_size,
        "batch_size_per_rank": args.batch_size,
        "seed": args.seed,
        "sampling": {
            "atom_temperature": 1.0,
            "atom_top_k": 16384,
            "atom_top_p": 0.7,
            "coeff_temperature": 1.0,
            "coeff_top_k": 2048,
            "coeff_top_p": 0.7,
        },
    }
    if world_rank == 0:
        result_path = args.output / "metrics.json"
        atomic_json(result, result_path)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        if args.wandb_eval_run_id:
            import wandb

            eval_run = wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                id=args.wandb_eval_run_id,
                resume="allow",
                name=(
                    f"{args.source_run.rsplit('/', 1)[-1]} FID+CLIP "
                    f"step {checkpoint_step}"
                ),
                group=f"{args.source_run.rsplit('/', 1)[-1]}-evaluations",
                job_type="text_fid_clip_evaluation",
                config=result,
            )
            eval_run.log(
                {
                    "eval/fid": fid,
                    "eval/text_fid": fid,
                    "eval/clip_score": clip_score,
                    "eval/num_real": evaluation_samples,
                    "eval/num_fake": evaluation_samples,
                    "source/checkpoint_epoch": checkpoint_epoch,
                    "source/checkpoint_step": checkpoint_step,
                }
            )
            eval_run.save(str(result_path), base_path=str(args.output), policy="now")
            eval_run.finish()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
