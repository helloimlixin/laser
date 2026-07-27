#!/usr/bin/env python3
"""Render a temperature/top-k sweep from an official FFHQ RQ-Transformer checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid, save_image

from train_ffhq_official_rqtransformer_laser_stage2 import build_model
from train_official_rqtransformer_laser_stage2 import LaserAux


def csv_values(text: str, cast):
    return [cast(value.strip()) for value in text.split(",") if value.strip()]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperatures", default="0.70,0.85,1.00,1.15,1.30")
    parser.add_argument("--top-k", default="64,128,250,512,1024")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--nrow", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    temperatures = csv_values(args.temperatures, float)
    top_ks = csv_values(args.top_k, int)
    if not temperatures or not top_ks:
        parser.error("the temperature and top-k grids must be non-empty")
    if args.num_samples <= 0 or args.batch_size <= 0:
        parser.error("--num-samples and --batch-size must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    num_atoms = int(config.get("num_atoms", 1024))
    coeff_vocab_size = int(config.get("coeff_vocab_size", 1024))
    coeff_scales = config.get("coeff_scales")
    if isinstance(coeff_scales, str):
        coeff_scales = json.loads(coeff_scales)

    print(f"Loading epoch {payload.get('epoch')} checkpoint on {device}: {args.checkpoint}", flush=True)
    aux = LaserAux(
        args.stage1_checkpoint,
        num_atoms,
        coeff_vocab_size,
        float(config.get("coeff_max", 3.0)),
        float(config.get("coeff_scale", 1.0)),
        attn_resolutions=(16,),
        coeff_scales=coeff_scales,
        soft_target_physical=coeff_scales is not None,
    ).to(device).eval()
    model = build_model(num_atoms + coeff_vocab_size, num_atoms).to(device).eval()
    model.load_state_dict(payload["state_dict"], strict=True)
    torch.set_float32_matmul_precision("high")

    grids: dict[tuple[float, int], Path] = {}
    for temperature in temperatures:
        for top_k in top_ks:
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            batches = []
            remaining = args.num_samples
            while remaining:
                count = min(args.batch_size, remaining)
                partial = torch.zeros(count, 8, 8, 4, device=device, dtype=torch.long)
                cond = torch.zeros(count, device=device, dtype=torch.long)
                tokens = model.sample(
                    partial,
                    model_aux=aux,
                    cond=cond,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=1.0,
                    amp=True,
                    cached=True,
                    is_tqdm=False,
                )
                batches.append(((aux.decode_tokens(tokens).float().cpu() + 1.0) * 0.5).clamp(0, 1))
                remaining -= count
            images = torch.cat(batches)
            target = args.output / f"grid_temp{temperature:.2f}_topk{top_k:04d}.png"
            save_image(make_grid(images, nrow=args.nrow, padding=2), target)
            grids[(temperature, top_k)] = target
            print(f"saved temperature={temperature:.2f} top_k={top_k}: {target}", flush=True)

    fig, axes = plt.subplots(
        len(temperatures), len(top_ks),
        figsize=(3.2 * len(top_ks), 3.2 * len(temperatures)), squeeze=False,
    )
    for row, temperature in enumerate(temperatures):
        for column, top_k in enumerate(top_ks):
            axis = axes[row][column]
            axis.imshow(mpimg.imread(grids[(temperature, top_k)]))
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(f"top-k {top_k}")
            if column == 0:
                axis.set_ylabel(f"temp {temperature:.2f}")
    fig.suptitle(f"FFHQ sampling sweep — epoch {payload.get('epoch')} — seed {args.seed}")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    contact_sheet = args.output / "contact_sheet.png"
    fig.savefig(contact_sheet, dpi=140)
    plt.close(fig)

    manifest = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_fid": payload.get("fid"),
        "stage1_checkpoint": str(args.stage1_checkpoint.resolve()),
        "temperatures": temperatures,
        "top_k": top_ks,
        "num_samples_per_setting": args.num_samples,
        "seed": args.seed,
        "contact_sheet": str(contact_sheet.resolve()),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"SWEEP DONE: {contact_sheet}", flush=True)


if __name__ == "__main__":
    main()
