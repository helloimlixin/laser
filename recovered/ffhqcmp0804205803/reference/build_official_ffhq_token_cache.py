#!/usr/bin/env python3
"""Build and validate an ordered FFHQ LASER compound-pair cache with torchrun."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.train_official_rqtransformer_laser_stage2 import (
    FlatImageDataset,
    LaserAux,
    val_image_transform,
)


class WithIndex(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label, index


@torch.no_grad()
def decode_continuous(aux, atoms, normalized_coeffs):
    contributions = aux.physical_contributions(atoms, normalized_coeffs)
    latent = contributions.sum(dim=-2).permute(0, 3, 1, 2).contiguous()
    return aux.decoder(aux.post_quant_conv(latent)).clamp(-1.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-atoms", type=int, default=2048)
    parser.add_argument("--coeff-vocab-size", type=int, default=2048)
    parser.add_argument("--coeff-max", type=float, default=3.0)
    parser.add_argument("--calibration-quantile", type=float, default=0.995)
    parser.add_argument("--verify-samples", type=int, default=256)
    parser.add_argument("--source-rfid", type=float, default=6.227079379843531)
    parser.add_argument("--max-items", type=int, default=0,
                        help="Testing only; 0 extracts all images")
    args = parser.parse_args()
    local_rank, process_rank, world = (
        int(os.environ[name]) for name in ("LOCAL_RANK", "RANK", "WORLD_SIZE")
    )
    dist.init_process_group("gloo")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.set_float32_matmul_precision("high")

    base = FlatImageDataset(args.data, transform=val_image_transform())
    if args.max_items:
        base = Subset(base, range(min(args.max_items, len(base))))
    indices = list(range(process_rank, len(base), world))
    loader = DataLoader(
        Subset(WithIndex(base), indices), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    # Extract physical OMP coefficients without clipping; normalize only after
    # the full-dataset per-depth calibration is known.
    aux = LaserAux(
        args.checkpoint, args.num_atoms, args.coeff_vocab_size,
        coeff_max=1.0e6, coeff_scale=1.0,
        attn_resolutions=(16,), coeff_scales=(1.0, 1.0), clamp_coeffs=False,
    ).to(device)
    shard = args.output.with_suffix(f".rank{process_rank:02d}.pt")
    if shard.is_file():
        print(f"rank {process_rank}: reusing {shard}", flush=True)
    else:
        atoms, physical_coeffs, labels, rows = [], [], [], []
        with torch.inference_mode():
            for step, (images, target, index) in enumerate(loader):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    batch_atoms, batch_coeffs = aux.encode_sparse_components(
                        images.to(device, non_blocking=True)
                    )
                atoms.append(batch_atoms.to(torch.int16).cpu())
                physical_coeffs.append(batch_coeffs.to(torch.float16).cpu())
                labels.append(target.to(torch.int16))
                rows.append(index)
                if process_rank == 0 and step % 50 == 0:
                    done = sum(item.shape[0] for item in rows)
                    print(f"cache rank 0: {done:,}/{len(indices):,}", flush=True)
        shard.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "atoms": torch.cat(atoms), "physical_coeffs": torch.cat(physical_coeffs),
            "labels": torch.cat(labels), "indices": torch.cat(rows),
        }, shard)
    dist.barrier()

    if process_rank == 0:
        parts = [
            torch.load(args.output.with_suffix(f".rank{part_rank:02d}.pt"), weights_only=True)
            for part_rank in range(world)
        ]
        order = torch.cat([part["indices"] for part in parts]).argsort()
        atoms = torch.cat([part["atoms"] for part in parts])[order].contiguous()
        physical = torch.cat([part["physical_coeffs"] for part in parts])[order].float().contiguous()
        labels = torch.cat([part["labels"] for part in parts])[order].contiguous()
        scales = [
            max(float(torch.quantile(physical[..., depth].abs().reshape(-1), args.calibration_quantile))
                / args.coeff_max, 1.0e-6)
            for depth in range(2)
        ]
        normalized = physical / torch.tensor(scales).view(1, 1, 1, 2)
        clip_fraction = float((normalized.abs() > args.coeff_max).float().mean())
        normalized = normalized.clamp(-args.coeff_max, args.coeff_max).to(torch.float16)
        meta = {
            "format": "laser_compound_pairs_v1", "dataset": "ffhq",
            "transform": "resize256_center_crop256", "items": len(base),
            "shape": [8, 8, 2], "compound_sequence_length": 128,
            "separate_atom_coeff_sequence_length": 256,
            "num_atoms": args.num_atoms, "coeff_vocab_size": args.coeff_vocab_size,
            "coeff_max": args.coeff_max, "coeff_scales": scales,
            "calibration_quantile": args.calibration_quantile,
            "stage1_checkpoint": str(args.checkpoint.resolve()),
            "stage1_rfid": args.source_rfid,
            "stage1_wandb_run": "https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhq-a2048-k2-rqvae-strict-20260720-145706",
            "world_size": world, "validation_passed": False,
        }
        merged = {"atoms": atoms, "coeffs": normalized, "labels": labels, "meta": meta}

        n = min(args.verify_samples, len(base))
        verify_loader = DataLoader(
            Subset(base, range(n)), batch_size=min(args.batch_size, n), shuffle=False,
        )
        direct_atoms, direct_physical, verify_images = [], [], []
        for images, _ in verify_loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                batch_atoms, batch_coeffs = aux.encode_sparse_components(images.to(device))
            direct_atoms.append(batch_atoms.cpu())
            direct_physical.append(batch_coeffs.cpu())
            verify_images.append(images)
        direct_atoms = torch.cat(direct_atoms)
        direct_physical = torch.cat(direct_physical)
        cached_atoms = atoms[:n].long()
        cached_coeffs = normalized[:n].float()
        direct_normalized = (
            direct_physical / torch.tensor(scales).view(1, 1, 1, 2)
        ).clamp(-args.coeff_max, args.coeff_max)

        aux.coeff_max = args.coeff_max
        aux.coeff_bins = torch.linspace(
            -args.coeff_max, args.coeff_max, args.coeff_vocab_size, device=device
        )
        aux.coeff_scales = torch.tensor(scales, device=device)
        device_atoms = cached_atoms.to(device)
        device_coeffs = cached_coeffs.to(device)
        with torch.inference_mode():
            coeff_ids, _ = aux.compound_coeff_ids(device_coeffs, stochastic=False)
            quantized_recon = aux.decode_compound(device_atoms, coeff_ids).float()
            continuous_recon = decode_continuous(aux, device_atoms, device_coeffs).float()
            source = torch.cat(verify_images).to(device).float()
        continuous_mse = float((continuous_recon - source).square().mean())
        quantized_mse = float((quantized_recon - source).square().mean())
        atom_exact = float((direct_atoms == cached_atoms).float().mean())
        coeff_error = (direct_normalized - cached_coeffs).abs()
        report = {
            "samples": n, "items": len(base), "atom_exact_fraction": atom_exact,
            "coeff_mae": float(coeff_error.mean()),
            "coeff_max_error": float(coeff_error.max()),
            "coeff_finite": bool(torch.isfinite(normalized).all()),
            "atom_min": int(atoms.min()), "atom_max": int(atoms.max()),
            "duplicate_atom_within_pair_fraction": float(
                (atoms[..., 0] == atoms[..., 1]).float().mean()
            ),
            "coeff_clip_fraction": clip_fraction,
            "coeff_scales": scales,
            "compound_sequence_length": 128,
            "separate_atom_coeff_sequence_length": 256,
            "continuous_reconstruction_mse": continuous_mse,
            "continuous_reconstruction_psnr": float(
                -10.0 * torch.log10(torch.tensor(max(continuous_mse, 1.0e-12)))
            ),
            "quantized_reconstruction_mse": quantized_mse,
            "quantized_reconstruction_psnr": float(
                -10.0 * torch.log10(torch.tensor(max(quantized_mse, 1.0e-12)))
            ),
            "quantization_mse_increase": quantized_mse - continuous_mse,
        }
        report["passed"] = (
            atom_exact == 1.0
            and report["coeff_max_error"] < 0.02
            and report["coeff_finite"]
            and report["atom_min"] >= 0 and report["atom_max"] < args.num_atoms
            and report["duplicate_atom_within_pair_fraction"] == 0.0
            and report["coeff_clip_fraction"] <= 0.02
            and report["quantization_mse_increase"] <= 0.002
        )
        merged["meta"]["validation_passed"] = report["passed"]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(merged, args.output)
        args.output.with_suffix(".validation.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(json.dumps(report, indent=2), flush=True)
        if not report["passed"]:
            raise RuntimeError("compound token cache validation failed")
        for part_rank in range(world):
            args.output.with_suffix(f".rank{part_rank:02d}.pt").unlink()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
