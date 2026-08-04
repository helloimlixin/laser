#!/usr/bin/env python3
"""Calibrate atom/coeff sampling independently, then validate the winner at 50K."""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_official_rqtransformer_laser_stage2 import (  # noqa: E402
    CompoundLaserRQTransformer,
    LaserAux,
    build_model,
    val_image_transform,
)


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def setting_name(setting: dict[str, float]) -> str:
    return (
        f"at{setting['atom_temperature']:.2f}_ap{setting['atom_top_p']:.2f}_"
        f"ct{setting['coeff_temperature']:.2f}_cp{setting['coeff_top_p']:.2f}"
    )


def unique_settings(settings: list[dict[str, float]]) -> list[dict[str, float]]:
    result = []
    seen = set()
    for setting in settings:
        key = tuple(sorted(setting.items()))
        if key not in seen:
            result.append(setting)
            seen.add(key)
    return result


def reset_fake_fid_state(metric) -> None:
    metric.fake_features_sum.zero_()
    metric.fake_features_cov_sum.zero_()
    metric.fake_features_num_samples.zero_()


def all_reduce_fid_state(metric, prefix: str) -> None:
    if not dist.is_initialized():
        return
    for suffix in ("features_sum", "features_cov_sum", "features_num_samples"):
        dist.all_reduce(getattr(metric, f"{prefix}_{suffix}"), op=dist.ReduceOp.SUM)


def gather_inception_features(metric) -> None:
    if not dist.is_initialized():
        return
    local_features = torch.cat(metric.features, dim=0)
    gathered = [torch.empty_like(local_features) for _ in range(world_size())]
    dist.all_gather(gathered, local_features)
    metric.features = [torch.cat(gathered, dim=0)]


@torch.no_grad()
def evaluate_setting(
    model: CompoundLaserRQTransformer,
    aux: LaserAux,
    fid_metric,
    inception_metric,
    setting: dict[str, float],
    *,
    num_samples: int,
    batch_size: int,
    seed: int,
) -> dict[str, float]:
    if num_samples % 1000:
        raise ValueError("calibration sample counts must be divisible by 1000")
    process_rank = rank()
    world = world_size()
    local_samples = num_samples // world + (process_rank < num_samples % world)
    reset_fake_fid_state(fid_metric)
    inception_metric.reset()
    torch.manual_seed(seed + process_rank)
    torch.cuda.manual_seed_all(seed + process_rank)
    generated = 0
    while generated < local_samples:
        current = min(batch_size, local_samples - generated)
        indices = torch.arange(
            generated, generated + current, device=next(model.parameters()).device,
            dtype=torch.long,
        )
        labels = (indices * world + process_rank).remainder(1000)
        atoms, coeff_ids = model.sample_compound(
            current,
            aux,
            cond=labels,
            atom_temperature=setting["atom_temperature"],
            atom_top_k=aux.num_atoms,
            atom_top_p=setting["atom_top_p"],
            coeff_temperature=setting["coeff_temperature"],
            coeff_top_p=setting["coeff_top_p"],
            amp=True,
        )
        images = ((aux.decode_compound(atoms, coeff_ids).float() + 1.0) * 0.5).clamp(0, 1)
        fid_metric.update(images, real=False)
        inception_metric.update(images)
        generated += current
        if process_rank == 0 and (generated == local_samples or generated % (batch_size * 10) == 0):
            print(
                f"{setting_name(setting)}: generated {generated}/{local_samples} local samples",
                flush=True,
            )
    all_reduce_fid_state(fid_metric, "fake")
    gather_inception_features(inception_metric)
    fid = float(fid_metric.compute().item())
    inception_mean, inception_std = inception_metric.compute()
    return {
        **setting,
        "num_samples": int(num_samples),
        "seed": int(seed),
        "fid": fid,
        "inception_score": float(inception_mean.item()),
        "inception_score_std": float(inception_std.item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--sweep-samples", type=int, default=5_000)
    parser.add_argument("--confirm-samples", type=int, default=10_000)
    parser.add_argument("--final-samples", type=int, default=50_000)
    parser.add_argument("--finalists", type=int, default=3)
    parser.add_argument("--num-atoms", type=int, default=16_384)
    parser.add_argument("--coeff-vocab-size", type=int, default=2_048)
    parser.add_argument("--coeff-max", type=float, default=20.0)
    parser.add_argument("--coeff-scale", type=float, default=6.4)
    parser.add_argument("--wandb-project", default="laser")
    parser.add_argument("--wandb-name", default=None)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        # NVLS multicast setup is unavailable in some H100 pod allocations;
        # ordinary NVLink P2P collectives remain enabled and healthy.
        os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
        dist.init_process_group("nccl", timeout=timedelta(minutes=45))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    args.output.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.stage2, map_location="cpu", weights_only=False, mmap=True)
    config = payload.get("config", {})
    refiner_layers = int(config.get("compound_refiner_layers", 0))
    distribution_geometry = bool(config.get("compound_distribution_geometry", False))
    geometry_head = (
        float(config.get("geometry_loss_weight", 0.0)) > 0
        and not distribution_geometry
    )
    aux = LaserAux(
        args.stage1, args.num_atoms, args.coeff_vocab_size, args.coeff_max, args.coeff_scale
    ).to(device).eval()
    model = build_model(
        args.num_atoms + args.coeff_vocab_size,
        args.num_atoms,
        compound=True,
        coeff_vocab_size=args.coeff_vocab_size,
        compound_refiner_layers=refiner_layers,
        compound_geometry_head=geometry_head,
        compound_micro_transformer_layers=int(
            config.get("compound_micro_transformer_layers", 0)
        ),
        compound_depth_specific_coeff_heads=bool(
            config.get("compound_depth_specific_coeff_heads", False)
        ),
    ).to(device).eval()
    model.load_state_dict(payload["state_dict"], strict=True)
    source_epoch = int(payload.get("epoch", -1))
    source_step = int(payload.get("global_step", -1))
    del payload

    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore

    fid_metric = FrechetInceptionDistance(
        feature=2048, normalize=True, sync_on_compute=False
    ).to(device)
    inception_metric = InceptionScore(
        normalize=True, splits=10, sync_on_compute=False
    ).to(device)

    val_dataset = datasets.ImageFolder(args.data / "val", transform=val_image_transform())
    sampler = DistributedSampler(
        val_dataset, num_replicas=world_size(), rank=rank(), shuffle=False, drop_last=False
    ) if dist.is_initialized() else None
    loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    if rank() == 0:
        print("Extracting the full 50K real-image FID statistics once", flush=True)
    for images, _ in loader:
        images = ((images.to(device, non_blocking=True).float() + 1.0) * 0.5).clamp(0, 1)
        fid_metric.update(images, real=True)
    all_reduce_fid_state(fid_metric, "real")

    baseline = {
        "atom_temperature": 1.0,
        "atom_top_p": 0.92,
        "coeff_temperature": 1.0,
        "coeff_top_p": 0.92,
    }
    settings = [baseline]
    for value in (0.90, 0.95, 1.05, 1.10):
        settings.append({**baseline, "atom_temperature": value})
    for value in (0.88, 0.96):
        settings.append({**baseline, "atom_top_p": value})
    for value in (0.80, 0.90, 1.10, 1.20):
        settings.append({**baseline, "coeff_temperature": value})
    for value in (0.85, 0.97):
        settings.append({**baseline, "coeff_top_p": value})

    all_results: list[dict[str, float | str]] = []
    phase1 = []
    for setting in unique_settings(settings):
        result = evaluate_setting(
            model, aux, fid_metric, inception_metric, setting,
            num_samples=args.sweep_samples, batch_size=args.batch_size, seed=12_345,
        )
        result["phase"] = "one_factor_sweep"
        phase1.append(result)
        all_results.append(result)
        if rank() == 0:
            print(json.dumps(result, sort_keys=True), flush=True)

    best_atom = min(
        (x for x in phase1 if x["coeff_temperature"] == 1.0 and x["coeff_top_p"] == 0.92),
        key=lambda x: x["fid"],
    )
    best_coeff = min(
        (x for x in phase1 if x["atom_temperature"] == 1.0 and x["atom_top_p"] == 0.92),
        key=lambda x: x["fid"],
    )
    combined = {
        "atom_temperature": best_atom["atom_temperature"],
        "atom_top_p": best_atom["atom_top_p"],
        "coeff_temperature": best_coeff["coeff_temperature"],
        "coeff_top_p": best_coeff["coeff_top_p"],
    }
    if tuple(sorted(combined.items())) not in {
        tuple(sorted({k: x[k] for k in baseline}.items())) for x in phase1
    }:
        result = evaluate_setting(
            model, aux, fid_metric, inception_metric, combined,
            num_samples=args.sweep_samples, batch_size=args.batch_size, seed=12_345,
        )
        result["phase"] = "combined_sweep"
        phase1.append(result)
        all_results.append(result)
        if rank() == 0:
            print(json.dumps(result, sort_keys=True), flush=True)

    finalist_settings = unique_settings([
        {k: x[k] for k in baseline}
        for x in sorted(phase1, key=lambda x: x["fid"])[:args.finalists]
    ] + [baseline])
    confirmations = []
    for setting in finalist_settings:
        result = evaluate_setting(
            model, aux, fid_metric, inception_metric, setting,
            num_samples=args.confirm_samples, batch_size=args.batch_size, seed=54_321,
        )
        result["phase"] = "confirmation"
        confirmations.append(result)
        all_results.append(result)
        if rank() == 0:
            print(json.dumps(result, sort_keys=True), flush=True)

    winner = min(confirmations, key=lambda x: x["fid"])
    winner_setting = {k: winner[k] for k in baseline}
    final_result = evaluate_setting(
        model, aux, fid_metric, inception_metric, winner_setting,
        num_samples=args.final_samples, batch_size=args.batch_size, seed=98_765,
    )
    final_result["phase"] = "final_50k"
    all_results.append(final_result)

    if rank() == 0:
        report = {
            "source_checkpoint": str(args.stage2.resolve()),
            "source_epoch": source_epoch,
            "source_step": source_step,
            "refiner_layers": refiner_layers,
            "geometry_head": geometry_head,
            "winner": final_result,
            "results": all_results,
        }
        target = args.output / "sampling_calibration.json"
        target.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        if args.wandb_name:
            import wandb
            wb = wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                job_type="sampling-calibration",
                config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            )
            for index, result in enumerate(all_results):
                wb.log({
                    "eval/index": index,
                    "eval/phase": result["phase"],
                    "eval/fid": result["fid"],
                    "eval/inception_score": result["inception_score"],
                    "eval/atom_temperature": result["atom_temperature"],
                    "eval/atom_top_p": result["atom_top_p"],
                    "eval/coeff_temperature": result["coeff_temperature"],
                    "eval/coeff_top_p": result["coeff_top_p"],
                    "eval/num_samples": result["num_samples"],
                })
            wb.summary["winner/fid_50k"] = final_result["fid"]
            wb.summary["winner/inception_score_50k"] = final_result["inception_score"]
            for key in baseline:
                wb.summary[f"winner/{key}"] = final_result[key]
            wb.finish()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
