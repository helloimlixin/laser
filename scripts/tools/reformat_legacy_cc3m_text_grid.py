#!/usr/bin/env python3
"""Recompose a legacy CC3M grid and replace its referenced W&B media blob."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from scripts.train_official_rqtransformer_laser_stage2 import (
    save_paper_style_text_grid,
)

WANDB_GRID_KEY = "samples/text_grid_8x8"
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--token-cache", type=Path, required=True)
    parser.add_argument("--wandb-run", required=True, help="entity/project/run_id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--upload-wandb", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_legacy_grid(source_path: Path, prompts, step: int) -> torch.Tensor:
    """Recover the 64 image cells using the legacy matplotlib geometry."""
    source = Image.open(source_path).convert("RGB")
    figure, axes = plt.subplots(8, 8, figsize=(20, 21))
    dummy = np.zeros((256, 256, 3), dtype=np.uint8)
    for index, axis in enumerate(axes.flat):
        axis.imshow(dummy)
        if index % 8 == 0:
            axis.set_ylabel(str(prompts[index // 8])[:80], fontsize=6)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        f"CC3M text-conditioned samples — optimizer step {step}", fontsize=16
    )
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.canvas.draw()

    images = []
    for axis in axes.flat:
        box = axis.get_position()
        crop_box = (
            round(box.x0 * source.width),
            round((1.0 - box.y1) * source.height),
            round(box.x1 * source.width),
            round((1.0 - box.y0) * source.height),
        )
        crop = source.crop(crop_box).resize(
            (256, 256), Image.Resampling.LANCZOS
        )
        array = np.asarray(crop, dtype=np.float32).copy() / 255.0
        images.append(torch.from_numpy(array).permute(2, 0, 1))
    plt.close(figure)
    return torch.stack(images)


def replace_wandb_grid_media(
    local_path: Path, legacy_path: Path, run_path: str, step: int
):
    """Replace the legacy blob already referenced by the live W&B grid key."""
    import wandb

    api = wandb.Api(timeout=60)
    legacy_sha = sha256(legacy_path)
    deadline = time.monotonic() + 60.0
    run = api.run(run_path)
    summary_media = {}
    while time.monotonic() < deadline:
        candidate = run.summary.get(WANDB_GRID_KEY)
        try:
            summary_media = dict(candidate)
        except (TypeError, ValueError):
            summary_media = {}
        # Wait until the primary client's media upload for this exact source
        # step is visible; otherwise a fast watcher could replace the previous
        # sample step and miss the new one.
        if summary_media.get("sha256") == legacy_sha:
            break
        time.sleep(2.0)
        run = wandb.Api(timeout=60).run(run_path)
    else:
        raise RuntimeError(
            f"W&B summary did not expose legacy grid for step {step}: "
            f"expected sha256 {legacy_sha}, got {summary_media.get('sha256')}"
        )

    summary_path = summary_media.get("path")
    if not summary_path:
        raise RuntimeError(f"W&B summary is missing {WANDB_GRID_KEY} media path")

    with tempfile.TemporaryDirectory(prefix="wandb-figure17-summary-") as tmp:
        root = Path(tmp)
        replacement = root / summary_path
        replacement.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(local_path) as source_image:
            rgb = source_image.convert("RGB")
            if replacement.suffix.lower() == ".png":
                rgb.save(replacement, format="PNG", optimize=True)
            else:
                rgb.save(
                    replacement, format="JPEG", quality=94,
                    subsampling=0, optimize=True,
                )
        replacement_sha = sha256(replacement)
        replacement_size = replacement.stat().st_size
        run.upload_file(str(replacement), root=str(root))

    width, height = Image.open(local_path).size
    media = {
        "_type": "image-file",
        "format": Path(summary_path).suffix.lstrip(".").lower(),
        "height": height,
        "path": summary_path,
        "sha256": replacement_sha,
        "size": replacement_size,
        "width": width,
    }
    return media, run.state, summary_path


def main():
    args = parse_args()
    source = args.source.expanduser().resolve()
    token_cache = args.token_cache.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not token_cache.is_file():
        raise FileNotFoundError(token_cache)
    step = int(source.stem.rsplit("_", 1)[1])
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else source.with_name(f"text_training_prompts_figure17_step_{step:07d}.jpg")
    )

    cache = torch.load(
        token_cache, map_location="cpu", weights_only=True, mmap=True
    )
    prompts = list(cache.get("text", []))[:8]
    if len(prompts) != 8:
        raise ValueError("token cache does not contain eight preview prompts")
    images = extract_legacy_grid(source, prompts, step)
    save_paper_style_text_grid(images, prompts, output)

    new_media = run_state = summary_media_replaced = None
    if args.upload_wandb:
        new_media, run_state, summary_media_replaced = replace_wandb_grid_media(
            output, source, args.wandb_run, step
        )
    receipt = {
        "created_at": time.time(),
        "source": str(source),
        "source_size": source.stat().st_size,
        "source_sha256": sha256(source),
        "output": str(output),
        "output_size": output.stat().st_size,
        "output_sha256": sha256(output),
        "step": step,
        "wandb_run": args.wandb_run if args.upload_wandb else None,
        "wandb_new_media": new_media,
        "wandb_run_state": run_state,
        "wandb_summary_media_replaced": summary_media_replaced,
    }
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    temporary = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(receipt_path)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
