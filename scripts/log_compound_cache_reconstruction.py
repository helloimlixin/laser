#!/usr/bin/env python3
"""Render deterministic compound-cache reconstructions and matched source images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_official_rqtransformer_laser_stage2 import (
    FlatImages,
    LaserAux,
    val_image_transform,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--token-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Optional source-image root used to render a row-matched original grid",
    )
    parser.add_argument(
        "--original-output",
        type=Path,
        default=None,
        help="Output for the matched source grid; requires --data",
    )
    parser.add_argument("--num-images", type=int, default=256)
    parser.add_argument("--nrow", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--wandb-entity", default="helloimlixin-rutgers")
    parser.add_argument("--wandb-project", default="laser")
    parser.add_argument("--wandb-id", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "disabled"),
        default="online",
        help="Use disabled for local/README rendering without touching a live run",
    )
    args = parser.parse_args()

    if args.num_images <= 0 or args.nrow <= 0 or args.batch_size <= 0:
        parser.error("--num-images, --nrow, and --batch-size must be positive")
    if args.num_images != args.nrow * args.nrow:
        parser.error("--num-images must equal --nrow squared for a square grid")
    if (args.data is None) != (args.original_output is None):
        parser.error("--data and --original-output must be provided together")
    if args.wandb_mode == "online" and (not args.wandb_id or not args.wandb_name):
        parser.error("--wandb-id and --wandb-name are required in online mode")

    payload = torch.load(
        args.token_cache, map_location="cpu", weights_only=True, mmap=True
    )
    meta = dict(payload["meta"])
    if meta.get("format") != "laser_compound_pairs_v1":
        raise ValueError(f"not a compound-pair cache: {meta.get('format')!r}")
    if len(payload["atoms"]) < args.num_images:
        raise ValueError(
            f"cache has {len(payload['atoms']):,} rows, fewer than {args.num_images:,}"
        )

    coeff_scales = meta.get("coeff_scales")
    aux = LaserAux(
        args.checkpoint,
        int(meta["num_atoms"]),
        int(meta["coeff_vocab_size"]),
        float(meta["coeff_max"]),
        float(meta.get("coeff_scale", 1.0)),
        attn_resolutions=((16,) if meta.get("dataset") in {"ffhq", "celebahq"} else (8,)),
        coeff_scales=coeff_scales,
    ).to("cuda:0").eval()

    reconstruction_batches = []
    for start in range(0, args.num_images, args.batch_size):
        stop = min(start + args.batch_size, args.num_images)
        atoms = payload["atoms"][start:stop].to("cuda:0", dtype=torch.long)
        coeffs = payload["coeffs"][start:stop].to("cuda:0", dtype=torch.float32)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            coeff_ids, _ = aux.compound_coeff_ids(coeffs, stochastic=False)
            reconstructions = aux.decode_compound(atoms, coeff_ids)
        reconstruction_batches.append(reconstructions.float().cpu())

    grid_images = torch.cat(reconstruction_batches)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image(
        grid_images,
        args.output,
        nrow=args.nrow,
        normalize=True,
        value_range=(-1.0, 1.0),
        padding=2,
    )

    original_images = None
    if args.data is not None and args.original_output is not None:
        source = FlatImages(args.data, transform=val_image_transform())
        if len(source) != int(meta["items"]):
            raise ValueError(
                f"source/cache row mismatch: source has {len(source):,} images but "
                f"cache metadata records {int(meta['items']):,}"
            )
        original_images = torch.stack([source[index][0] for index in range(args.num_images)])
        args.original_output.parent.mkdir(parents=True, exist_ok=True)
        save_image(
            original_images,
            args.original_output,
            nrow=args.nrow,
            normalize=True,
            value_range=(-1.0, 1.0),
            padding=2,
        )

    grid_label = f"{args.nrow}x{args.nrow}"
    reconstruction_key = f"diagnostics/token_cache_reconstruction_{grid_label}"
    original_key = f"diagnostics/token_cache_original_{grid_label}"
    run_url = None
    if args.wandb_mode == "online":
        import wandb

        run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            id=args.wandb_id,
            resume="allow",
            name=args.wandb_name,
        )
        log_payload = {
            reconstruction_key: wandb.Image(
                str(args.output),
                caption=(
                    f"Deterministic reconstructions of cache rows 0-{args.num_images - 1}; "
                    f"coefficient scales={coeff_scales}"
                ),
            ),
            "diagnostics/token_cache_reconstruction_num_images": args.num_images,
        }
        if original_images is not None and args.original_output is not None:
            log_payload[original_key] = wandb.Image(
                str(args.original_output),
                caption=f"Matched source images for cache rows 0-{args.num_images - 1}",
            )
        run.log(log_payload)
        run_url = run.url
        run.finish()

    receipt = {
        "checkpoint": str(args.checkpoint.resolve()),
        "token_cache": str(args.token_cache.resolve()),
        "cache_items": int(meta["items"]),
        "cache_rows": [0, args.num_images - 1],
        "num_images": args.num_images,
        "grid": [args.nrow, args.nrow],
        "coeff_scales": coeff_scales,
        "output": str(args.output.resolve()),
        "original_output": (
            str(args.original_output.resolve()) if args.original_output is not None else None
        ),
        "wandb_run": run_url,
        "wandb_key": reconstruction_key if run_url is not None else None,
        "wandb_original_key": (
            original_key if run_url is not None and original_images is not None else None
        ),
    }
    receipt_path = args.output.with_suffix(".json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
