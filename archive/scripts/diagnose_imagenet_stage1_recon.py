#!/usr/bin/env python3
"""Save ImageNet stage-1 LASER reconstruction diagnostics from a checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.checkpoint_io import load_lightning_module
from src.codebook_visuals import render_codebook_scatter, select_codebook_vectors
from src.data.config import DataConfig
from src.data.image_folder import ImageFolderDataModule
from src.models.laser import LASER


def _device(raw: str) -> torch.device:
    text = str(raw).strip().lower()
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def _as_batch(batch: Any) -> torch.Tensor:
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch


def _unit_range(x: torch.Tensor, *, mean: tuple[float, ...], std: tuple[float, ...]) -> torch.Tensor:
    mean_t = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return (x * std_t + mean_t).clamp(0.0, 1.0)


def _heatmap(values: torch.Tensor, *, out_hw: tuple[int, int] | None = None) -> torch.Tensor:
    values = values.detach().to(torch.float32)
    if values.ndim == 3:
        values = values.unsqueeze(1)
    if out_hw is not None and tuple(values.shape[-2:]) != tuple(out_hw):
        values = F.interpolate(values, size=out_hw, mode="nearest")
    flat = values.flatten(1)
    mins = flat.min(dim=1).values.view(-1, 1, 1, 1)
    maxs = flat.max(dim=1).values.view(-1, 1, 1, 1)
    norm = ((values - mins) / (maxs - mins).clamp_min(1e-6)).clamp(0.0, 1.0)
    cmap = torch.tensor(
        [
            [0.0015, 0.0005, 0.0139],
            [0.1486, 0.0212, 0.5706],
            [0.4723, 0.1105, 0.4283],
            [0.7771, 0.2514, 0.2300],
            [0.9871, 0.5364, 0.0382],
            [0.9884, 0.9984, 0.6449],
        ],
        dtype=norm.dtype,
        device=norm.device,
    )
    scaled = norm.squeeze(1) * float(cmap.size(0) - 1)
    lo = scaled.floor().to(torch.long).clamp(0, cmap.size(0) - 1)
    hi = (lo + 1).clamp(0, cmap.size(0) - 1)
    frac = (scaled - lo.to(scaled.dtype)).unsqueeze(1)
    rgb = cmap[lo].permute(0, 3, 1, 2) * (1.0 - frac) + cmap[hi].permute(0, 3, 1, 2) * frac
    return rgb.clamp(0.0, 1.0)


def _save_grid(path: Path, images: torch.Tensor, *, nrow: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(images.detach().cpu().clamp(0.0, 1.0), path, nrow=int(nrow), normalize=False)


def _write_metrics(path: Path, *, mse: torch.Tensor, mae: torch.Tensor, sources: list[str]) -> None:
    mse_cpu = mse.detach().cpu()
    mae_cpu = mae.detach().cpu()
    payload = {
        "count": int(mse_cpu.numel()),
        "mse_mean": float(mse_cpu.mean().item()),
        "mae_mean": float(mae_cpu.mean().item()),
        "psnr_mean_db": float((-10.0 * torch.log10(mse_cpu.clamp_min(1e-12))).mean().item()),
        "items": [
            {
                "index": idx,
                "source_path": sources[idx] if idx < len(sources) else "",
                "mse": float(mse_cpu[idx].item()),
                "mae": float(mae_cpu[idx].item()),
                "psnr_db": float(-10.0 * math.log10(max(float(mse_cpu[idx].item()), 1e-12))),
            }
            for idx in range(int(mse_cpu.numel()))
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _save_codebook_scatter(model: LASER, path: Path, *, step: int, max_vectors: int) -> None:
    atoms = model.bottleneck.dictionary.detach().t().cpu()
    atoms = select_codebook_vectors(atoms, int(max_vectors))
    image = render_codebook_scatter([atoms], [int(step)], title="Dictionary Atoms (PCA)")
    if image is None:
        return
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("/workspace/Projects/data/imagenet"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--codebook-visual-max-vectors", type=int, default=512)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    count = max(1, int(args.count))
    batch_size = max(1, min(int(args.batch_size), count))
    device = _device(args.device)

    data_cfg = DataConfig(
        dataset="imagenet",
        data_dir=str(Path(args.data_dir).expanduser().resolve()),
        batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=max(0, int(args.num_workers)),
        image_size=int(args.image_size),
        train_crop_size=None,
        seed=42,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        augment=False,
    )
    dm = ImageFolderDataModule(data_cfg)
    dm.setup(args.split)
    loader = {
        "train": dm.train_dataloader,
        "val": dm.val_dataloader,
        "test": dm.test_dataloader,
    }[args.split]()

    model = load_lightning_module(
        LASER,
        checkpoint,
        map_location="cpu",
        strict=False,
        compute_fid=False,
        log_images_every_n_steps=0,
        enable_val_latent_visuals=False,
    ).eval().to(device)

    chunks: list[torch.Tensor] = []
    for batch in loader:
        chunks.append(_as_batch(batch))
        if sum(int(item.size(0)) for item in chunks) >= count:
            break
    x_cpu = torch.cat(chunks, dim=0)[:count].contiguous()
    x = x_cpu.to(device)

    with torch.inference_mode():
        recon_raw, _, sparse_codes = model(x)

    x_unit = _unit_range(x, mean=data_cfg.mean, std=data_cfg.std)
    recon_unit = _unit_range(recon_raw, mean=data_cfg.mean, std=data_cfg.std)
    error = (recon_unit - x_unit).square().mean(dim=1, keepdim=True).sqrt()
    sparse_energy = sparse_codes.values.detach().to(torch.float32).pow(2).sum(dim=-1).sqrt()

    mse = (recon_unit - x_unit).square().mean(dim=(1, 2, 3))
    mae = (recon_unit - x_unit).abs().mean(dim=(1, 2, 3))
    nrow = min(4, count)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_grid(out_dir / "imagenet_original_grid.png", x_unit, nrow=nrow)
    _save_grid(out_dir / "imagenet_recon_grid.png", recon_unit, nrow=nrow)
    _save_grid(
        out_dir / "imagenet_original_vs_recon_grid.png",
        torch.cat([x_unit, recon_unit], dim=0),
        nrow=nrow,
    )
    paired = torch.cat([x_unit, recon_unit], dim=2)
    _save_grid(out_dir / "imagenet_original_top_recon_bottom_pairs.png", paired, nrow=nrow)
    _save_grid(out_dir / "imagenet_recon_error_heatmap.png", _heatmap(error), nrow=nrow)
    _save_grid(
        out_dir / "imagenet_sparse_energy_heatmap.png",
        _heatmap(sparse_energy, out_hw=tuple(x_unit.shape[-2:])),
        nrow=nrow,
    )

    dataset = getattr(dm, f"{args.split}_dataset", None)
    sources = [str(path) for path in list(getattr(dataset, "image_paths", []) or [])[:count]]
    _write_metrics(out_dir / "imagenet_recon_metrics.json", mse=mse, mae=mae, sources=sources)

    step = 0
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        step = int(payload.get("global_step", 0) or 0)
    except Exception:
        step = 0
    _save_codebook_scatter(
        model,
        out_dir / "imagenet_dictionary_scatter.png",
        step=step,
        max_vectors=int(args.codebook_visual_max_vectors),
    )

    print(f"Saved diagnostics to: {out_dir}")
    print(f"MSE mean: {float(mse.mean().item()):.6f}")
    print(f"MAE mean: {float(mae.mean().item()):.6f}")
    print(f"PSNR mean: {float((-10.0 * torch.log10(mse.clamp_min(1e-12))).mean().item()):.2f} dB")
    print(f"Original/recon grid: {out_dir / 'imagenet_original_top_recon_bottom_pairs.png'}")
    print(f"Error heatmap:        {out_dir / 'imagenet_recon_error_heatmap.png'}")


if __name__ == "__main__":
    main()
