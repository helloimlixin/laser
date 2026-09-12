#!/usr/bin/env python3
"""Export a reproducible ImageNet RQ-VAE reconstruction grid for a paper."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision.datasets import ImageNet
from torchvision.utils import make_grid


ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY = ROOT / "third_party" / "rq-vae-transformer"
for source_root in (THIRD_PARTY, ROOT):
    if source_root.as_posix() not in sys.path:
        sys.path.insert(0, source_root.as_posix())

from rqvae.img_datasets.transforms import create_transforms
from rqvae.models import create_model


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_rfid(log_path: Path) -> float:
    matches = re.findall(
        r"validation-only metrics,\s*rfid:\s*([0-9]+(?:\.[0-9]+)?)",
        log_path.read_text(encoding="utf-8"),
    )
    if not matches:
        raise RuntimeError(f"no validation-only rFID was found in {log_path}")
    return float(matches[-1])


def class_name(dataset: ImageNet, target: int) -> str:
    names = dataset.classes[int(target)]
    if isinstance(names, (list, tuple)):
        return ", ".join(str(name) for name in names)
    return str(names)


def save_png(grid: torch.Tensor, path: Path, *, dpi: int) -> tuple[int, int]:
    array = (
        grid.detach()
        .cpu()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .add(0.5)
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    image = Image.fromarray(array, mode="RGB")
    image.save(path, format="PNG", compress_level=6, dpi=(dpi, dpi))
    return image.size


def save_pdf(png_path: Path, pdf_path: Path, *, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = np.asarray(Image.open(png_path).convert("RGB"))
    height, width = image.shape[:2]
    figure = plt.figure(
        figsize=(width / float(dpi), height / float(dpi)),
        dpi=dpi,
        frameon=False,
    )
    axis = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axis.imshow(image, interpolation="nearest")
    axis.set_axis_off()
    figure.savefig(
        pdf_path,
        format="pdf",
        dpi=dpi,
        facecolor="white",
        edgecolor="none",
        metadata={
            "Title": "ImageNet LASER reconstruction grid",
            "Subject": "Original images (top) and LASER reconstructions (bottom)",
            "Keywords": "ImageNet, LASER, dictionary learning, reconstruction",
        },
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rfid-log", type=Path, required=True)
    parser.add_argument(
        "--imagenet-root",
        type=Path,
        default=Path("/workspace/Projects/data/imagenet"),
    )
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--padding", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--expected-rfid", type=float)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    rfid_log_path = args.rfid_log.expanduser().resolve()
    imagenet_root = args.imagenet_root.expanduser().resolve()
    output_prefix = args.output_prefix.expanduser().resolve()
    count = int(args.count)
    padding = int(args.padding)
    dpi = int(args.dpi)
    if count <= 0:
        raise ValueError("--count must be positive")
    if padding < 0:
        raise ValueError("--padding must be non-negative")
    if dpi <= 0:
        raise ValueError("--dpi must be positive")
    for required in (checkpoint_path, config_path, rfid_log_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not (imagenet_root / "val").is_dir():
        raise FileNotFoundError(imagenet_root / "val")

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    metadata_path = output_prefix.with_suffix(".json")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    if not args.overwrite:
        existing = [path for path in (png_path, pdf_path, metadata_path) if path.exists()]
        if existing:
            raise FileExistsError(", ".join(path.as_posix() for path in existing))

    checkpoint_sha256 = sha256_file(checkpoint_path)
    if (
        args.expected_sha256 is not None
        and checkpoint_sha256 != str(args.expected_sha256).lower()
    ):
        raise RuntimeError(
            "checkpoint SHA-256 mismatch: "
            f"expected={args.expected_sha256}, actual={checkpoint_sha256}"
        )
    rfid = parse_rfid(rfid_log_path)
    if args.expected_rfid is not None and not math.isclose(
        rfid, float(args.expected_rfid), rel_tol=0.0, abs_tol=5e-7
    ):
        raise RuntimeError(
            f"rFID mismatch: expected={args.expected_rfid}, log={rfid}"
        )

    config = OmegaConf.load(config_path)
    config.dataset.root = imagenet_root.as_posix()
    if str(config.arch.hparams.get("bottleneck_type", "")).lower() != "laser":
        raise RuntimeError("the selected config is not a LASER model")
    if int(config.arch.hparams.get("sparsity_level", -1)) != 4:
        raise RuntimeError("the selected config does not use K=4")
    if str(config.experiment.get("rfid_backend", "")) != "original-rqvae":
        raise RuntimeError("the reported metric does not use the original RQ-VAE backend")

    validation_transform = create_transforms(
        config.dataset,
        split="val",
        is_eval=True,
    )
    dataset = ImageNet(
        imagenet_root.as_posix(),
        split="val",
        transform=validation_transform,
    )
    if count > len(dataset):
        raise ValueError(f"requested {count} examples from a {len(dataset)}-image split")
    generator = torch.Generator().manual_seed(int(args.seed))
    indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
    loaded = [dataset[int(index)] for index in indices]
    inputs_cpu = torch.stack([item[0] for item in loaded], dim=0).contiguous()
    targets = [int(item[1]) for item in loaded]

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    model, model_ema = create_model(config.arch, ema=False)
    if model_ema is not None:
        raise RuntimeError("unexpected EMA model")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.requires_grad_(False)
    model.eval().to(device)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    inputs = inputs_cpu.to(device, non_blocking=True)
    with torch.inference_mode():
        reconstructions, _, supports = model(inputs)
    originals = (inputs.float() * 0.5 + 0.5).clamp(0.0, 1.0)
    reconstructions = (reconstructions.float() * 0.5 + 0.5).clamp(0.0, 1.0)
    mse = (reconstructions - originals).square().mean(dim=(1, 2, 3))
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
    figure_grid = make_grid(
        torch.cat((originals, reconstructions), dim=0),
        nrow=count,
        padding=padding,
        pad_value=1.0,
    )
    width, height = save_png(figure_grid, png_path, dpi=dpi)
    save_pdf(png_path, pdf_path, dpi=dpi)
    png_sha256 = sha256_file(png_path)
    pdf_sha256 = sha256_file(pdf_path)

    items = []
    for column, (index, target) in enumerate(zip(indices, targets)):
        source_path = Path(dataset.samples[int(index)][0]).resolve()
        items.append(
            {
                "column": column,
                "validation_index": int(index),
                "target": target,
                "wnid": str(dataset.wnids[target]),
                "class_name": class_name(dataset, target),
                "source_path": source_path.as_posix(),
                "mse": float(mse[column].detach().cpu().item()),
                "psnr_db": float(psnr[column].detach().cpu().item()),
                "unique_atoms": int(torch.unique(supports[column]).numel()),
            }
        )
    metadata = {
        "figure": {
            "png": png_path.as_posix(),
            "pdf": pdf_path.as_posix(),
            "png_sha256": png_sha256,
            "pdf_sha256": pdf_sha256,
            "width_pixels": width,
            "height_pixels": height,
            "dpi": dpi,
            "rows": ["original", "LASER reconstruction"],
            "columns": count,
            "padding_pixels": padding,
        },
        "selection": {
            "method": "first columns of torch.randperm over the full validation split",
            "seed": int(args.seed),
            "quality_filtering": False,
            "validation_split_size": len(dataset),
            "transform": "Resize(256), CenterCrop(256), Resize(256x256), normalize to [-1,1]",
        },
        "checkpoint": {
            "path": checkpoint_path.as_posix(),
            "sha256": checkpoint_sha256,
            "checkpoint_id": checkpoint.get("checkpoint_id"),
            "global_step": int(checkpoint.get("global_step", -1)),
            "dictionary_update_step": int(
                checkpoint["state_dict"]["quantizer._dictionary_update_step"].item()
            ),
            "bottleneck": "LASER",
            "dictionary_size": int(config.arch.hparams.n_embed),
            "sparsity": int(config.arch.hparams.sparsity_level),
        },
        "metric": {
            "name": "rFID",
            "value": rfid,
            "backend": "original-rqvae",
            "validation_images": len(dataset),
            "log": rfid_log_path.as_posix(),
        },
        "grid_reconstruction": {
            "mse_mean": float(mse.mean().detach().cpu().item()),
            "psnr_mean_db": float(psnr.mean().detach().cpu().item()),
        },
        "items": items,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
