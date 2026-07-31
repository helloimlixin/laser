#!/usr/bin/env python3
"""Full ImageNet reconstruction FID and generation sampling sweep."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_official_rqtransformer_laser_stage2 import (  # noqa: E402
    LaserAux, build_model, val_image_transform,
)


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def world():
    return dist.get_world_size() if dist.is_initialized() else 1


def atomic_json(payload, path: Path):
    if rank() != 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


@torch.no_grad()
def reconstruction_fid(aux, loader, device):
    metric = FrechetInceptionDistance(
        feature=2048, normalize=True, sync_on_compute=dist.is_initialized()
    ).to(device)
    seen = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        tokens, _ = aux.encode_sparse(images, temp=0.5, stochastic=False)
        recon = ((aux.decode_tokens(tokens).float() + 1.0) * 0.5).clamp(0, 1)
        real = ((images.float() + 1.0) * 0.5).clamp(0, 1)
        metric.update(real, real=True)
        metric.update(recon, real=False)
        seen += images.size(0)
    return float(metric.compute().item()), seen


@torch.no_grad()
def build_real_fid(loader, device, num_samples):
    metric = FrechetInceptionDistance(
        feature=2048, normalize=True, sync_on_compute=dist.is_initialized()
    ).to(device)
    local_samples = num_samples // world() + (rank() < num_samples % world())
    seen = 0
    for images, _ in loader:
        images = ((images.to(device, non_blocking=True).float() + 1.0) * 0.5).clamp(0, 1)
        keep = min(images.size(0), local_samples - seen)
        metric.update(images[:keep], real=True)
        seen += keep
        if seen >= local_samples:
            break
    return metric


@torch.no_grad()
def generation_fid(model, aux, real_metric, device, num_samples, batch_size, temperature, top_p):
    # Reuse the exact same full-validation real Inception statistics for every
    # sampling setting; only fake generations and their statistics change.
    metric = copy.deepcopy(real_metric)
    inception_score = InceptionScore(
        feature="logits_unbiased", normalize=True, splits=10,
        sync_on_compute=dist.is_initialized(),
    ).to(device)
    local_samples = num_samples // world() + (rank() < num_samples % world())
    generated = 0
    while generated < local_samples:
        current = min(batch_size, local_samples - generated)
        indices = torch.arange(generated, generated + current, device=device)
        labels = (indices * world() + rank()).remainder(1000)
        partial = torch.zeros(current, 8, 8, 4, device=device, dtype=torch.long)
        tokens = model.sample(
            partial, model_aux=aux, cond=labels, temperature=temperature,
            top_k=aux.num_atoms, top_p=top_p, amp=True, cached=True, is_tqdm=False,
        )
        images = ((aux.decode_tokens(tokens).float() + 1.0) * 0.5).clamp(0, 1)
        metric.update(images, real=False)
        inception_score.update(images)
        generated += current
    fid = float(metric.compute().item())
    is_mean, is_std = inception_score.compute()
    return fid, float(is_mean.item()), float(is_std.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.85, 0.95, 1.0, 1.05])
    parser.add_argument("--top-ps", type=float, nargs="+", default=[0.90, 0.92, 0.95])
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=45))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dataset = datasets.ImageFolder(args.data / "val", transform=val_image_transform())
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False) if world() > 1 else None
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler, shuffle=False,
        num_workers=8, pin_memory=True, persistent_workers=True,
    )
    aux = LaserAux(args.stage1, 16384, 2048, 20.0, 6.4).to(device).eval()
    model = build_model(16384 + 2048, 16384).to(device).eval()
    payload = torch.load(args.stage2, map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(payload["state_dict"], strict=True)

    results_path = args.output / "full_imagenet_evaluation.json"
    results = {
        "checkpoint": str(args.stage2), "checkpoint_epoch": int(payload.get("epoch", -1)),
        "checkpoint_step": int(payload.get("global_step", -1)), "num_samples": args.num_samples,
        "reconstruction_fid": None, "sampling_sweep": [],
    }
    rfid, local_seen = reconstruction_fid(aux, loader, device)
    results["reconstruction_fid"] = rfid
    results["reconstruction_images"] = int(local_seen * world())
    atomic_json(results, results_path)
    if rank() == 0:
        print(f"Full ImageNet tokenizer reconstruction FID: {rfid:.6f}", flush=True)

    real_metric = build_real_fid(loader, device, args.num_samples)
    for temperature in args.temperatures:
        for top_p in args.top_ps:
            fid, is_mean, is_std = generation_fid(
                model, aux, real_metric, device, args.num_samples, args.batch_size,
                float(temperature), float(top_p),
            )
            results["sampling_sweep"].append({
                "temperature": float(temperature), "top_p": float(top_p), "top_k": 16384,
                "fid": fid, "inception_score_mean": is_mean,
                "inception_score_std": is_std, "inception_score_splits": 10,
                "num_samples": args.num_samples,
            })
            atomic_json(results, results_path)
            if rank() == 0:
                print(
                    f"Full ImageNet temp={temperature} top_p={top_p}: "
                    f"FID={fid:.6f} IS={is_mean:.6f}+/-{is_std:.6f}", flush=True
                )
            if dist.is_initialized():
                dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
