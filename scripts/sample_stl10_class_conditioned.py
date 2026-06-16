#!/usr/bin/env python3
"""Generate STL10 class-conditional image grids from a trained stage-2 prior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.utils import save_image

from src.s2 import load_run, sample, save_grid


STL10_CLASS_NAMES = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]


def _to_uint8_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().to(torch.float32)
    if tensor.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got {tuple(tensor.shape)}")
    tensor = tensor.clamp(-1.0, 1.0).add(1.0).mul(127.5).round().to(torch.uint8)
    array = tensor.permute(1, 2, 0).contiguous().numpy()
    if array.shape[-1] == 1:
        return Image.fromarray(array[..., 0], mode="L").convert("RGB")
    return Image.fromarray(array, mode="RGB")


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text.strip().lower())


def _save_labeled_grid(
    imgs: torch.Tensor,
    labels: list[int],
    names: list[str],
    out_dir: Path,
    *,
    stem: str,
    nrow: int,
) -> Path:
    panels: list[Image.Image] = []
    for image, label in zip(imgs, labels):
        label = int(label)
        name = names[label] if 0 <= label < len(names) else str(label)
        pil = _to_uint8_image(image).convert("RGB")
        width, height = pil.size
        label_height = 16
        panel = Image.new("RGB", (width, height + label_height), "white")
        draw = ImageDraw.Draw(panel)
        draw.text((2, 2), f"{label} {name}", fill="black")
        panel.paste(pil, (0, label_height))
        panels.append(panel)

    nrow = max(1, int(nrow))
    rows = (len(panels) + nrow - 1) // nrow
    cell_w, cell_h = panels[0].size
    grid = Image.new("RGB", (nrow * cell_w, rows * cell_h), "white")
    for idx, panel in enumerate(panels):
        row = idx // nrow
        col = idx % nrow
        grid.paste(panel, (col * cell_w, row * cell_h))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}_labeled.png"
    grid.save(path)
    return path


def _save_individuals(imgs: torch.Tensor, labels: list[int], names: list[str], out_dir: Path) -> None:
    class_counts: dict[int, int] = {}
    for idx, (image, label) in enumerate(zip(imgs, labels)):
        label = int(label)
        name = names[label] if 0 <= label < len(names) else str(label)
        class_dir = out_dir / f"{label:02d}_{_safe_name(name)}"
        class_dir.mkdir(parents=True, exist_ok=True)
        class_idx = class_counts.get(label, 0)
        class_counts[label] = class_idx + 1
        save_image(
            image,
            class_dir / f"sample_{class_idx:03d}.png",
            normalize=True,
            value_range=(-1.0, 1.0),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="STL10 run directory containing token_cache.pt and stage2/.")
    parser.add_argument("--ckpt", default=None, help="Stage-2 checkpoint. Defaults to latest under run-dir/stage2.")
    parser.add_argument("--token-cache", default=None, help="Token cache. Defaults to run-dir/token_cache.pt.")
    parser.add_argument("--out-dir", default=None, help="Output directory. Defaults to run-dir/stage2/class_cond_samples.")
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--coeff-temperature", type=float, default=None)
    parser.add_argument("--coeff-mode", default=None)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    token_cache = Path(args.token_cache).expanduser().resolve() if args.token_cache else run_dir / "token_cache.pt"
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else run_dir / "stage2" / "class_cond_samples"
    samples_per_class = max(1, int(args.samples_per_class))
    labels = [label for _ in range(samples_per_class) for label in range(len(STL10_CLASS_NAMES))]
    grid_nrow = len(STL10_CLASS_NAMES)

    runtime = load_run(
        ckpt=args.ckpt,
        cache_pt=token_cache,
        dev=args.device,
        out_root=run_dir,
        ar_dir=run_dir / "stage2",
    )
    label_tensor = torch.as_tensor(labels, device=runtime.dev, dtype=torch.long)
    batch = sample(
        runtime.net,
        runtime.s1,
        runtime.shape,
        n=len(labels),
        bs=max(1, int(args.batch_size)),
        temp=float(args.temperature),
        top_k=int(args.top_k),
        ctemp=args.coeff_temperature,
        cmode=args.coeff_mode,
        class_labels=label_tensor,
        dev=runtime.dev,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    unlabeled = save_grid(batch.imgs, out_dir, stem="stl10_class_cond_grid", nrow=grid_nrow)
    labeled = _save_labeled_grid(
        batch.imgs,
        labels,
        STL10_CLASS_NAMES,
        out_dir,
        stem="stl10_class_cond_grid",
        nrow=grid_nrow,
    )
    _save_individuals(batch.imgs, labels, STL10_CLASS_NAMES, out_dir / "by_class")
    (out_dir / "class_labels.json").write_text(
        json.dumps(
            {
                "labels": labels,
                "class_names": STL10_CLASS_NAMES,
                "stage2_checkpoint": str(runtime.ckpt),
                "token_cache": str(runtime.cache_pt),
                "temperature": float(args.temperature),
                "top_k": int(args.top_k),
                "samples_per_class": samples_per_class,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved unlabeled grid: {unlabeled}")
    print(f"Saved labeled grid:   {labeled}")
    print(f"Saved per-class dirs: {out_dir / 'by_class'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
