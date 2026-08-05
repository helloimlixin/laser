#!/usr/bin/env python3
"""Build an ordered ImageNet or CelebA-HQ LASER sparse-component cache."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.train_official_rqtransformer_laser_stage2 import LaserAux, val_image_transform


class WithIndex(Dataset):
    def __init__(self, dataset): self.dataset = dataset
    def __len__(self): return len(self.dataset)
    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label, index


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", choices=("imagenet", "celebahq"), default="imagenet")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--num-atoms", type=int, default=16384)
    p.add_argument("--coeff-vocab-size", type=int, default=2048)
    p.add_argument("--coeff-max", type=float, default=20.0)
    p.add_argument("--coeff-scale", type=float, default=6.4)
    p.add_argument("--coeff-scales", type=float, nargs=2)
    p.add_argument(
        "--auto-coeff-scales-percentile", type=float, default=None,
        help=(
            "Fit one physical coefficient scale per OMP depth so this absolute "
            "percentile maps to --coeff-max; for example 99.9"
        ),
    )
    p.add_argument("--verify-samples", type=int, default=256)
    p.add_argument("--compound", action="store_true",
                   help="Label cache as paired (atom, coefficient) events and validate pair decoding")
    args = p.parse_args()
    if args.coeff_scales is not None and args.auto_coeff_scales_percentile is not None:
        p.error("--coeff-scales and --auto-coeff-scales-percentile are mutually exclusive")
    if (args.auto_coeff_scales_percentile is not None and
            not 0.0 < args.auto_coeff_scales_percentile <= 100.0):
        p.error("--auto-coeff-scales-percentile must be in (0, 100]")
    local_rank, rank, world = (int(os.environ[k]) for k in ("LOCAL_RANK", "RANK", "WORLD_SIZE"))
    # Ranks only need control-plane barriers; using Gloo avoids fragile NCCL
    # peer mappings after a long, independent cache extraction workload.
    dist.init_process_group("gloo")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.set_float32_matmul_precision("high")
    base = datasets.ImageFolder(args.data / "train", transform=val_image_transform())
    indices = list(range(rank, len(base), world))
    loader = DataLoader(Subset(WithIndex(base), indices), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=args.num_workers > 0)
    calibrating = args.auto_coeff_scales_percentile is not None
    aux = LaserAux(
        args.checkpoint, args.num_atoms, args.coeff_vocab_size,
        args.coeff_max if not calibrating else 1e9,
        args.coeff_scale if not calibrating else 1.0,
        attn_resolutions=((16,) if args.dataset == "celebahq" else (8,)),
        coeff_scales=args.coeff_scales,
        clamp_coeffs=not calibrating,
    ).to(device)
    shard = args.output.with_suffix(f".rank{rank:02d}.pt")
    if shard.is_file():
        print(f"rank {rank}: reusing completed shard {shard}", flush=True)
    else:
        atoms, coeffs, labels, rows = [], [], [], []
        with torch.inference_mode():
            for step, (images, target, index) in enumerate(loader):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    a, c = aux.encode_sparse_components(images.to(device, non_blocking=True))
                atoms.append(a.to(torch.int16).cpu())
                coeffs.append(c.to(torch.float16).cpu())
                labels.append(
                    (torch.zeros_like(target) if args.dataset == "celebahq" else target).to(
                        torch.int16
                    )
                )
                rows.append(index)
                if rank == 0 and step % 100 == 0:
                    print(f"cache rank 0: {sum(x.shape[0] for x in rows):,}/{len(indices):,}", flush=True)
        shard.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"atoms": torch.cat(atoms), "coeffs": torch.cat(coeffs),
                    "labels": torch.cat(labels), "indices": torch.cat(rows)}, shard)
    dist.barrier()
    if rank == 0:
        parts = [torch.load(args.output.with_suffix(f".rank{r:02d}.pt"), weights_only=True) for r in range(world)]
        order = torch.cat([x["indices"] for x in parts]).argsort()
        merged_coeffs = torch.cat([x["coeffs"] for x in parts])[order].float()
        if calibrating:
            absolute = merged_coeffs.abs().reshape(-1, 2)
            quantile = float(args.auto_coeff_scales_percentile) / 100.0
            coeff_scales = torch.quantile(absolute, quantile, dim=0) / args.coeff_max
            if not torch.isfinite(coeff_scales).all() or (coeff_scales <= 0).any():
                raise RuntimeError(f"invalid calibrated coefficient scales: {coeff_scales}")
            merged_coeffs = (
                merged_coeffs / coeff_scales.view(1, 1, 1, 2)
            ).clamp(-args.coeff_max, args.coeff_max)
            aux.coeff_scales.copy_(coeff_scales.to(device))
            aux.coeff_max = float(args.coeff_max)
            aux.coeff_bins.copy_(
                torch.linspace(
                    -args.coeff_max, args.coeff_max, args.coeff_vocab_size,
                    device=device,
                )
            )
            aux.clamp_coeffs = True
        else:
            coeff_scales = aux.coeff_scales.detach().cpu()
        merged = {
            "atoms": torch.cat([x["atoms"] for x in parts])[order].contiguous(),
            "coeffs": merged_coeffs.to(torch.float16).contiguous(),
            "labels": torch.cat([x["labels"] for x in parts])[order].contiguous(),
            "meta": {"format": ("laser_compound_pairs_v1" if args.compound else "laser_sparse_components_v1"), "dataset": args.dataset,
                     "transform": "resize256_center_crop256", "items": len(base),
                     "shape": [8, 8, 2], "num_atoms": args.num_atoms,
                     "coeff_vocab_size": args.coeff_vocab_size, "coeff_max": args.coeff_max,
                     "coeff_scale": args.coeff_scale,
                     "coeff_scales": [float(x) for x in coeff_scales],
                     "auto_coeff_scales_percentile": args.auto_coeff_scales_percentile,
                     "stage1_checkpoint": str(args.checkpoint.resolve()),
                     "world_size": world},
        }
        torch.save(merged, args.output)
        # Structural and numerical verification against a fresh direct encoding.
        n = min(args.verify_samples, len(base))
        verify_loader = DataLoader(Subset(base, range(n)), batch_size=min(args.batch_size, n), shuffle=False)
        direct_a, direct_c, verify_images = [], [], []
        for images, _ in verify_loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                a, c = aux.encode_sparse_components(images.to(device))
            direct_a.append(a.cpu()); direct_c.append(c.cpu())
            verify_images.append(images)
        direct_a, direct_c = torch.cat(direct_a), torch.cat(direct_c)
        cached_a, cached_c = merged["atoms"][:n].long(), merged["coeffs"][:n].float()
        report = {"samples": n, "atom_exact_fraction": float((direct_a == cached_a).float().mean()),
                  "coeff_mae": float((direct_c - cached_c).abs().mean()),
                  "coeff_max_error": float((direct_c - cached_c).abs().max()),
                  "atom_min": int(merged["atoms"].min()), "atom_max": int(merged["atoms"].max()),
                  "coeff_finite": bool(torch.isfinite(merged["coeffs"]).all()),
                  "label_min": int(merged["labels"].min()), "label_max": int(merged["labels"].max()),
                  "coeff_scales": [float(x) for x in coeff_scales],
                  "auto_coeff_scales_percentile": args.auto_coeff_scales_percentile}
        if args.compound:
            with torch.inference_mode():
                coeff_ids, _ = aux.compound_coeff_ids(cached_c.to(device), stochastic=False)
                quantized = aux.coeff_bins[coeff_ids] * aux.coeff_scales.view(1, 1, 1, 2)
                physical = cached_c.to(device) * aux.coeff_scales.view(1, 1, 1, 2)
                squared_error = 0.0
                pixel_count = 0
                source = torch.cat(verify_images)
                # The original stage-1 decoder is wide enough that decoding
                # the entire verification set at once can exceed a 20 GB GPU.
                for start in range(0, n, 8):
                    stop = min(start + 8, n)
                    recon = aux.decode_compound(
                        cached_a[start:stop].to(device), coeff_ids[start:stop]
                    )
                    target = source[start:stop].to(device)
                    squared_error += float(
                        (recon.float() - target.float()).square().sum()
                    )
                    pixel_count += target.numel()
            mse = squared_error / pixel_count
            report.update({
                "compound_sequence_length": 128,
                "scalar_sequence_length_baseline": 256,
                "physical_coeff_quantization_mae": float((quantized - physical).abs().mean()),
                "coeff_bound_fraction": float((cached_c.abs() >= args.coeff_max).float().mean()),
                "quantized_reconstruction_mse": mse,
                "quantized_reconstruction_psnr": float(-10.0 * torch.log10(torch.tensor(max(mse, 1e-12)))),
                "duplicate_atom_within_pair_fraction": float((cached_a[..., 0] == cached_a[..., 1]).float().mean()),
            })
        max_label = 0 if args.dataset == "celebahq" else 999
        report["passed"] = report["atom_exact_fraction"] == 1.0 and report["coeff_max_error"] < 0.02 and report["coeff_finite"] and report["atom_min"] >= 0 and report["atom_max"] < args.num_atoms and report["label_min"] >= 0 and report["label_max"] <= max_label
        args.output.with_suffix(".validation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        if not report["passed"]: raise RuntimeError("token cache validation failed")
        for r in range(world): args.output.with_suffix(f".rank{r:02d}.pt").unlink()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__": main()
