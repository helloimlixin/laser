#!/usr/bin/env python3
"""Decode cached CC3M LASER tokens and compare them with source images."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import tarfile
from typing import Sequence

import torch
from PIL import Image
from torchvision.utils import make_grid, save_image
import torchvision.transforms.functional as TF

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.checkpoint_io import load_lightning_module
from src.models.laser import LASER


def _parse_source_path(raw: object) -> tuple[str, str]:
    text = str(raw)
    if "::" not in text:
        raise ValueError(f"Expected shard.tar::member source path, got {text!r}")
    shard, member = text.split("::", 1)
    return str(Path(shard).expanduser().resolve()), member.lstrip("/")


def _read_image_from_tar(source_path: object, *, image_size: int) -> torch.Tensor:
    shard, member = _parse_source_path(source_path)
    with tarfile.open(shard, "r:*") as tf:
        extracted = tf.extractfile(member)
        if extracted is None:
            raise RuntimeError(f"Could not read {member!r} from {shard}")
        with Image.open(extracted) as img:
            img = img.convert("RGB")
            tensor = TF.pil_to_tensor(img).to(torch.float32) / 255.0
    return TF.resize(
        tensor,
        [int(image_size), int(image_size)],
        interpolation=TF.InterpolationMode.BILINEAR,
        antialias=True,
    )


def _caption_for(cache: dict, idx: int) -> str:
    texts = cache.get("text")
    if isinstance(texts, Sequence) and idx < len(texts):
        return str(texts[idx])
    return ""


def _save_grid(images: torch.Tensor, path: Path, *, nrow: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(images.detach().cpu().clamp(0.0, 1.0), path, nrow=int(nrow), normalize=False)


def _write_metrics(path: Path, *, captions: list[str], source_paths: list[str], mse: torch.Tensor, mae: torch.Tensor) -> None:
    payload = {
        "count": len(captions),
        "mse_mean": float(mse.mean().item()),
        "mae_mean": float(mae.mean().item()),
        "psnr_mean_db": float((-10.0 * torch.log10(mse.clamp_min(1e-12))).mean().item()),
        "items": [
            {
                "index": idx,
                "source_path": source_paths[idx],
                "caption": captions[idx],
                "mse": float(mse[idx].item()),
                "mae": float(mae[idx].item()),
                "psnr_db": float(-10.0 * math.log10(max(float(mse[idx].item()), 1e-12))),
            }
            for idx in range(len(captions))
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    cache_path = Path(args.cache).expanduser().resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    meta = dict(cache.get("meta", {}) or {})
    checkpoint = args.checkpoint or meta.get("stage1_checkpoint")
    if not checkpoint:
        raise RuntimeError("No stage1 checkpoint provided and cache metadata has no stage1_checkpoint.")
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {checkpoint}")

    shape = cache.get("shape")
    if not isinstance(shape, (tuple, list)) or len(shape) != 3:
        raise RuntimeError(f"Cache is missing shape, got {shape!r}")
    h, w, d = (int(shape[0]), int(shape[1]), int(shape[2]))
    tokens_flat = cache.get("tokens_flat")
    source_paths = list(cache.get("source_paths") or [])
    if not torch.is_tensor(tokens_flat) or tokens_flat.ndim != 2:
        raise RuntimeError("Cache is missing rank-2 tokens_flat.")
    if len(source_paths) < int(tokens_flat.size(0)):
        raise RuntimeError("Cache does not contain enough source_paths for original-image comparison.")

    offset = max(0, int(args.offset))
    count = min(max(1, int(args.count)), int(tokens_flat.size(0)) - offset)
    rows = list(range(offset, offset + count))
    tokens = tokens_flat[rows].view(count, h, w, d).to(torch.long)

    device = torch.device("cuda" if str(args.device) == "auto" and torch.cuda.is_available() else args.device)
    model = load_lightning_module(
        LASER,
        checkpoint,
        map_location="cpu",
        strict=False,
        compute_fid=False,
    ).eval().to(device)

    latent_hw = meta.get("latent_hw")
    if isinstance(latent_hw, (tuple, list)) and len(latent_hw) == 2:
        latent_hw = (int(latent_hw[0]), int(latent_hw[1]))
    else:
        latent_hw = None
    coeff_vocab_size = int(meta.get("coeff_vocab_size") or meta.get("n_bins") or 0)
    coeff_bin_values = meta.get("coeff_bin_values")
    if coeff_vocab_size <= 0 or coeff_bin_values is None:
        raise RuntimeError("Cache is missing coefficient quantization metadata.")

    image_size = int(meta.get("image_size") or 256)
    originals = torch.stack(
        [_read_image_from_tar(source_paths[idx], image_size=image_size) for idx in rows],
        dim=0,
    )
    x_norm = originals.to(device=device, dtype=torch.float32).mul(2.0).sub(1.0)

    with torch.inference_mode():
        decoded = model.decode_from_tokens(
            tokens.to(device),
            latent_hw=latent_hw,
            coeff_vocab_size=coeff_vocab_size,
            coeff_bin_values=torch.as_tensor(coeff_bin_values, device=device, dtype=torch.float32),
        )
        direct_decoded, _, _ = model(x_norm)
        atom_ids, coeffs, real_latent_hw = model.encode_to_atoms_and_coeffs(x_norm)
        real_sparse_decoded = model.decode_from_atoms_and_coeffs(
            atom_ids,
            coeffs,
            latent_hw=real_latent_hw,
        )

    recon = ((decoded.detach().cpu().to(torch.float32) + 1.0) / 2.0).clamp(0.0, 1.0)
    direct_recon = ((direct_decoded.detach().cpu().to(torch.float32) + 1.0) / 2.0).clamp(0.0, 1.0)
    real_sparse_recon = ((real_sparse_decoded.detach().cpu().to(torch.float32) + 1.0) / 2.0).clamp(0.0, 1.0)

    mse = (recon - originals).square().mean(dim=(1, 2, 3))
    mae = (recon - originals).abs().mean(dim=(1, 2, 3))
    direct_mse = (direct_recon - originals).square().mean(dim=(1, 2, 3))
    direct_mae = (direct_recon - originals).abs().mean(dim=(1, 2, 3))
    real_sparse_mse = (real_sparse_recon - originals).square().mean(dim=(1, 2, 3))
    real_sparse_mae = (real_sparse_recon - originals).abs().mean(dim=(1, 2, 3))

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    nrow = min(4, count)
    _save_grid(originals, out_dir / "cc3m_original_grid.png", nrow=nrow)
    _save_grid(recon, out_dir / "cc3m_laser_recon_grid.png", nrow=nrow)
    _save_grid(direct_recon, out_dir / "cc3m_laser_direct_forward_recon_grid.png", nrow=nrow)
    _save_grid(real_sparse_recon, out_dir / "cc3m_laser_real_sparse_recon_grid.png", nrow=nrow)
    _save_grid(torch.cat([originals, recon], dim=0), out_dir / "cc3m_original_vs_laser_recon_grid.png", nrow=nrow)
    _save_grid(
        torch.cat([originals, direct_recon, real_sparse_recon, recon], dim=0),
        out_dir / "cc3m_original_direct_real_sparse_quantized_cache_grid.png",
        nrow=nrow,
    )

    captions = [_caption_for(cache, idx) for idx in rows]
    selected_sources = [str(source_paths[idx]) for idx in rows]
    _write_metrics(
        out_dir / "cc3m_laser_recon_metrics.json",
        captions=captions,
        source_paths=selected_sources,
        mse=mse,
        mae=mae,
    )
    _write_metrics(
        out_dir / "cc3m_laser_direct_forward_recon_metrics.json",
        captions=captions,
        source_paths=selected_sources,
        mse=direct_mse,
        mae=direct_mae,
    )
    _write_metrics(
        out_dir / "cc3m_laser_real_sparse_recon_metrics.json",
        captions=captions,
        source_paths=selected_sources,
        mse=real_sparse_mse,
        mae=real_sparse_mae,
    )
    (out_dir / "cc3m_laser_recon_captions.txt").write_text(
        "\n".join(f"{idx}\t{caption}" for idx, caption in zip(rows, captions)) + "\n",
        encoding="utf-8",
    )

    print(f"Saved originals:   {out_dir / 'cc3m_original_grid.png'}")
    print(f"Saved recons:      {out_dir / 'cc3m_laser_recon_grid.png'}")
    print(f"Saved direct:      {out_dir / 'cc3m_laser_direct_forward_recon_grid.png'}")
    print(f"Saved real sparse: {out_dir / 'cc3m_laser_real_sparse_recon_grid.png'}")
    print(f"Saved comparison:  {out_dir / 'cc3m_original_vs_laser_recon_grid.png'}")
    print(f"Saved metrics:     {out_dir / 'cc3m_laser_recon_metrics.json'}")
    print(f"MSE mean:          {float(mse.mean().item()):.6f}")
    print(f"MAE mean:          {float(mae.mean().item()):.6f}")
    print(f"Direct MSE mean:   {float(direct_mse.mean().item()):.6f}")
    print(f"Direct MAE mean:   {float(direct_mae.mean().item()):.6f}")
    print(f"Real sparse MSE:   {float(real_sparse_mse.mean().item()):.6f}")
    print(f"Real sparse MAE:   {float(real_sparse_mae.mean().item()):.6f}")


if __name__ == "__main__":
    main()
