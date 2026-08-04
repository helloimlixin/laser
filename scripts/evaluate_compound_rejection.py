#!/usr/bin/env python3
"""Evaluate compound LASER sampling with class-balanced rejection sampling."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets
from torchvision.models import ResNet101_Weights, resnet101

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_official_rqtransformer_laser_stage2 import (
    LaserAux,
    build_model,
    val_image_transform,
)


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=50_000)
    parser.add_argument("--candidate-multiplier", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-atoms", type=int, default=16_384)
    parser.add_argument("--coeff-vocab-size", type=int, default=2_048)
    parser.add_argument("--coeff-max", type=float, default=20.0)
    parser.add_argument("--coeff-scale", type=float, default=6.4)
    parser.add_argument("--atom-temperature", type=float, default=1.0)
    parser.add_argument("--atom-top-p", type=float, default=0.92)
    parser.add_argument("--coeff-temperature", type=float, default=1.0)
    parser.add_argument("--coeff-top-p", type=float, default=0.92)
    parser.add_argument("--wandb-project", default="laser")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=45))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if args.num_samples % 1000:
        raise ValueError("--num-samples must be divisible by 1000 for a balanced class prior")
    if args.candidate_multiplier < 1:
        raise ValueError("--candidate-multiplier must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    target_per_class = args.num_samples // 1000
    candidates_per_class = target_per_class * args.candidate_multiplier
    local_classes = torch.arange(rank(), 1000, world, dtype=torch.long)
    candidate_labels = local_classes.repeat_interleave(candidates_per_class)

    aux = LaserAux(
        args.stage1, args.num_atoms, args.coeff_vocab_size,
        args.coeff_max, args.coeff_scale,
    ).to(device).eval()
    model = build_model(
        args.num_atoms + args.coeff_vocab_size,
        args.num_atoms,
        compound=True,
        coeff_vocab_size=args.coeff_vocab_size,
    ).to(device).eval()
    payload = torch.load(args.stage2, map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(payload["state_dict"], strict=True)
    source_epoch = int(payload.get("epoch", -1))
    source_step = int(payload.get("global_step", -1))
    del payload

    weights = ResNet101_Weights.IMAGENET1K_V2
    classifier = resnet101(weights=weights).to(device).eval().requires_grad_(False)
    mean = torch.tensor(weights.transforms().mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(weights.transforms().std, device=device).view(1, 3, 1, 1)

    local_candidate_count = len(candidate_labels)
    candidate_images = torch.empty(
        local_candidate_count, 3, 256, 256, dtype=torch.uint8
    )
    candidate_scores = torch.empty(local_candidate_count, dtype=torch.float32)

    for start in range(0, local_candidate_count, args.batch_size):
        end = min(start + args.batch_size, local_candidate_count)
        labels = candidate_labels[start:end].to(device, non_blocking=True)
        atoms, coeff_ids = model.sample_compound(
            len(labels), aux, cond=labels,
            atom_temperature=args.atom_temperature,
            atom_top_k=args.num_atoms,
            atom_top_p=args.atom_top_p,
            coeff_temperature=args.coeff_temperature,
            coeff_top_p=args.coeff_top_p,
            amp=True,
        )
        images = ((aux.decode_compound(atoms, coeff_ids).float() + 1.0) * 0.5).clamp(0, 1)
        classifier_images = F.interpolate(
            images, size=(224, 224), mode="bilinear", align_corners=False, antialias=True
        )
        classifier_images = (classifier_images - mean) / std
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = classifier(classifier_images)
        scores = logits.float().softmax(dim=-1).gather(1, labels[:, None]).squeeze(1)
        candidate_images[start:end].copy_(
            images.mul(255).round().to(torch.uint8), non_blocking=False
        )
        candidate_scores[start:end].copy_(scores.cpu())
        if rank() == 0 and (end == local_candidate_count or end % (args.batch_size * 20) == 0):
            print(f"Generated/scored {end}/{local_candidate_count} local candidates", flush=True)

    del classifier
    torch.cuda.empty_cache()

    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore

    fid_metric = FrechetInceptionDistance(
        feature=2048, normalize=True, sync_on_compute=world > 1
    ).to(device)
    inception_metric = InceptionScore(
        normalize=True, splits=10, sync_on_compute=world > 1
    ).to(device)

    val_dataset = datasets.ImageFolder(args.data / "val", transform=val_image_transform())
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world, rank=rank(), shuffle=False, drop_last=False
    ) if world > 1 else None
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=val_sampler,
        shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True,
    )
    real_seen = 0
    local_target = target_per_class * len(local_classes)
    for images, _ in val_loader:
        keep = min(len(images), local_target - real_seen)
        images = ((images[:keep].to(device, non_blocking=True).float() + 1.0) * 0.5).clamp(0, 1)
        fid_metric.update(images, real=True)
        real_seen += keep
        if real_seen >= local_target:
            break

    selected_score_sum = torch.zeros((), device=device)
    selected_count = 0
    for class_index in range(len(local_classes)):
        start = class_index * candidates_per_class
        end = start + candidates_per_class
        selected = candidate_scores[start:end].topk(target_per_class).indices + start
        images = candidate_images[selected].to(device, non_blocking=True).float().div_(255)
        fid_metric.update(images, real=False)
        inception_metric.update(images)
        selected_score_sum += candidate_scores[selected].sum().to(device)
        selected_count += len(selected)

    fid = float(fid_metric.compute().item())
    inception_mean, inception_std = inception_metric.compute()
    count_tensor = torch.tensor(float(selected_count), device=device)
    if world > 1:
        dist.all_reduce(selected_score_sum)
        dist.all_reduce(count_tensor)
    confidence = float((selected_score_sum / count_tensor).item())

    result = {
        "source_checkpoint": str(args.stage2.resolve()),
        "source_epoch": source_epoch,
        "source_step": source_step,
        "num_samples": args.num_samples,
        "candidate_multiplier": args.candidate_multiplier,
        "fid": fid,
        "inception_score": float(inception_mean.item()),
        "inception_score_std": float(inception_std.item()),
        "selected_resnet101_confidence": confidence,
        "atom_temperature": args.atom_temperature,
        "atom_top_p": args.atom_top_p,
        "coeff_temperature": args.coeff_temperature,
        "coeff_top_p": args.coeff_top_p,
    }
    if rank() == 0:
        target = args.output / "metrics.json"
        target.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        if args.wandb_name:
            import wandb
            wb = wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                group=args.wandb_group,
                job_type="rejection-evaluation",
                config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            )
            wb.log({
                "eval/fid": fid,
                "eval/inception_score": float(inception_mean.item()),
                "eval/inception_score_std": float(inception_std.item()),
                "eval/selected_resnet101_confidence": confidence,
            })
            wb.finish()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
