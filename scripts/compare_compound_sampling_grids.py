#!/usr/bin/env python3
"""Render matched-seed compound sampling grids from one stage-2 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torchvision import datasets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_official_rqtransformer_laser_stage2 import (  # noqa: E402
    LaserAux,
    build_model,
    sample_class_grid,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    payload = torch.load(args.stage2, map_location="cpu", weights_only=False, mmap=True)
    config = payload.get("config", {})
    num_atoms = int(config.get("num_atoms", 16_384))
    coeff_vocab_size = int(config.get("coeff_vocab_size", 2_048))
    coeff_max = float(config.get("coeff_max", 20.0))
    coeff_scale = float(config.get("coeff_scale", 6.4))
    distribution_geometry = bool(config.get("compound_distribution_geometry", False))

    aux = LaserAux(
        args.stage1, num_atoms, coeff_vocab_size, coeff_max, coeff_scale
    ).to(device).eval()
    model = build_model(
        num_atoms + coeff_vocab_size,
        num_atoms,
        compound=True,
        coeff_vocab_size=coeff_vocab_size,
        compound_refiner_layers=int(config.get("compound_refiner_layers", 0)),
        compound_geometry_head=(
            float(config.get("geometry_loss_weight", 0.0)) > 0
            and not distribution_geometry
        ),
        compound_micro_transformer_layers=int(
            config.get("compound_micro_transformer_layers", 0)
        ),
        compound_depth_specific_coeff_heads=bool(
            config.get("compound_depth_specific_coeff_heads", False)
        ),
    ).to(device).eval()
    model.load_state_dict(payload["state_dict"], strict=True)
    source_step = int(payload.get("global_step", -1))
    del payload

    class_names = datasets.ImageFolder(args.data / "val").classes
    settings = {
        "default_t1": (1.0, 0.92, 1.0, 0.92),
        "calibrated_mature": (0.90, 0.92, 0.80, 0.92),
    }
    for name, (atom_temperature, atom_top_p, coeff_temperature, coeff_top_p) in settings.items():
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        target = sample_class_grid(
            model,
            aux,
            class_names,
            args.output / name,
            source_step,
            atom_temperature=atom_temperature,
            atom_top_p=atom_top_p,
            coeff_temperature=coeff_temperature,
            coeff_top_p=coeff_top_p,
        )
        print(f"{name}: {target}", flush=True)


if __name__ == "__main__":
    main()
