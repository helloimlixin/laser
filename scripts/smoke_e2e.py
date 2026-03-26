#!/usr/bin/env python3
"""Run a tiny end-to-end LASER smoke test on a small CelebA subset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from torchvision.io import read_image
from torchvision.transforms.functional import resize
from torchvision.utils import save_image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.laser import LASER
from src.data.token_cache import load_token_cache
from src.stage2_paths import (
    infer_latest_stage1_checkpoint,
    infer_latest_stage2_checkpoint,
    infer_latest_token_cache,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _has_images(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any(candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS for candidate in path.rglob("*"))


def _default_celeba_dir() -> Path:
    candidates = [
        os.environ.get("CELEBA_DIR", ""),
        str((REPO_ROOT / "../data/celeba").resolve()),
        str((REPO_ROOT / "data/celeba").resolve()),
        "/home/xl598/Projects/data/celeba",
        "/home/xl598/Data/celeba",
        "/home/xl598/Data/celeba/img_align_celeba",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if _has_images(path):
            return path
    raise RuntimeError(
        "Could not find a local CelebA image directory. Pass --data-dir or set CELEBA_DIR."
    )


def _collect_images(root: Path) -> list[Path]:
    images = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    images.sort()
    if not images:
        raise RuntimeError(f"No images found under {root}")
    return images


def _ensure_subset(source_dir: Path, subset_dir: Path, subset_size: int, *, refresh: bool) -> Path:
    if refresh and subset_dir.exists():
        shutil.rmtree(subset_dir)

    existing = [path for path in subset_dir.glob("*") if path.is_file()]
    if len(existing) == subset_size:
        return subset_dir

    if subset_dir.exists():
        shutil.rmtree(subset_dir)
    subset_dir.mkdir(parents=True, exist_ok=True)

    images = _collect_images(source_dir)
    if len(images) < subset_size:
        raise RuntimeError(f"Requested {subset_size} images but only found {len(images)} under {source_dir}")

    for index, source in enumerate(images[:subset_size]):
        target = subset_dir / f"{index:05d}{source.suffix.lower()}"
        try:
            target.symlink_to(source.resolve())
        except OSError:
            shutil.copy2(source, target)

    return subset_dir


def _run(command: list[str], *, env: dict[str, str]) -> None:
    print("\n>>>", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def _save_preview_variants(images: torch.Tensor, output_path: Path, *, nrow: int) -> Path:
    save_image(images, output_path, nrow=nrow, normalize=False)
    autocontrast_path = output_path.with_name(f"{output_path.stem}_autocontrast{output_path.suffix}")
    save_image(images, autocontrast_path, nrow=nrow, normalize=True, scale_each=True)
    return autocontrast_path


def _load_preview_batch(subset_dir: Path, *, image_size: int, count: int) -> torch.Tensor:
    paths = _collect_images(subset_dir)[:count]
    images = []
    for path in paths:
        image = read_image(str(path)).float() / 255.0
        image = resize(image, [int(image_size), int(image_size)], antialias=True)
        image = (image - 0.5) / 0.5
        images.append(image)
    return torch.stack(images, dim=0)


def _write_stage1_preview(stage1_root: Path, subset_dir: Path, *, image_size: int, count: int = 4) -> Path:
    checkpoint = infer_latest_stage1_checkpoint(output_root=stage1_root)
    if checkpoint is None:
        raise RuntimeError(f"Could not infer stage-1 checkpoint under {stage1_root}")
    model = LASER.load_from_checkpoint(checkpoint, map_location="cpu").eval()
    inputs = _load_preview_batch(subset_dir, image_size=image_size, count=count)
    with torch.inference_mode():
        recon, _, _ = model(inputs)
    inputs_disp = ((inputs + 1.0) / 2.0).clamp(0.0, 1.0)
    recon_disp = ((recon + 1.0) / 2.0).clamp(0.0, 1.0)
    preview = torch.cat([inputs_disp, recon_disp], dim=0)
    preview_path = stage1_root.parent / "stage1_recon_preview.png"
    _save_preview_variants(preview, preview_path, nrow=count)
    return preview_path


def _write_token_cache_preview(ar_root: Path, stage1_root: Path, *, count: int = 4) -> Path:
    checkpoint = infer_latest_stage1_checkpoint(output_root=stage1_root)
    token_cache = infer_latest_token_cache(ar_output_dir=ar_root, dataset="celeba", split="train")
    if checkpoint is None:
        raise RuntimeError(f"Could not infer stage-1 checkpoint under {stage1_root}")
    if token_cache is None:
        raise RuntimeError(f"Could not infer token cache under {ar_root}")

    model = LASER.load_from_checkpoint(checkpoint, map_location="cpu").eval()
    cache = load_token_cache(token_cache)
    shape = cache.get("shape")
    tokens_flat = cache.get("tokens_flat")
    meta = cache.get("meta", {})
    if not isinstance(shape, (tuple, list)) or len(shape) != 3:
        raise RuntimeError(f"Token cache at {token_cache} is missing a valid shape entry")
    if tokens_flat is None or int(tokens_flat.shape[0]) <= 0:
        raise RuntimeError(f"Token cache at {token_cache} is missing tokens_flat")

    height, width, depth = (int(shape[0]), int(shape[1]), int(shape[2]))
    batch = min(int(count), int(tokens_flat.shape[0]))
    tokens = tokens_flat[:batch].view(batch, height, width, depth).to(torch.long)
    latent_hw = meta.get("latent_hw")
    if isinstance(latent_hw, (tuple, list)) and len(latent_hw) == 2:
        latent_hw = (int(latent_hw[0]), int(latent_hw[1]))
    else:
        latent_hw = None

    coeff_vocab_size = int(meta.get("coeff_vocab_size") or meta.get("n_bins") or 0)
    coeff_bin_values = meta.get("coeff_bin_values")
    if coeff_vocab_size <= 0 or coeff_bin_values is None:
        raise RuntimeError(f"Token cache at {token_cache} is missing coefficient quantization metadata")

    with torch.inference_mode():
        decoded = model.decode_from_tokens(
            tokens,
            latent_hw=latent_hw,
            coeff_vocab_size=coeff_vocab_size,
            coeff_bin_values=torch.as_tensor(coeff_bin_values, dtype=torch.float32),
        )
    decoded_disp = ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)
    preview_path = ar_root.parent / "token_cache_decode_preview.png"
    _save_preview_variants(decoded_disp, preview_path, nrow=batch)
    return preview_path


def _build_stage1_command(args: argparse.Namespace, subset_dir: Path, stage1_root: Path) -> list[str]:
    num_embeddings = args.num_embeddings if args.num_embeddings is not None else (128 if args.patch_based else 32)
    command = [
        sys.executable,
        "train.py",
        f"output_dir={stage1_root}",
        f"hydra.run.dir={stage1_root / 'hydra'}",
        "model=laser",
        "data=celeba",
        f"data.data_dir={subset_dir}",
        f"data.batch_size={args.stage1_batch_size}",
        "data.num_workers=0",
        f"data.image_size={args.image_size}",
        "train.accelerator=cpu" if args.device == "cpu" else f"train.accelerator={args.train_accelerator}",
        f"train.devices={args.devices}",
        "train.strategy=auto",
        "train.precision=32",
        f"train.max_epochs={args.stage1_epochs}",
        "train.log_every_n_steps=1",
        "checkpoint.save_top_k=1",
        f"model.num_hiddens={args.stage1_num_hiddens}",
        f"model.embedding_dim={args.stage1_embedding_dim}",
        f"model.num_embeddings={num_embeddings}",
        f"model.sparsity_level={args.sparsity_level}",
        f"model.num_residual_blocks={args.stage1_num_residual_blocks}",
        f"model.num_residual_hiddens={args.stage1_num_residual_hiddens}",
        f"model.patch_based={'true' if args.patch_based else 'false'}",
        "model.perceptual_weight=0.0",
        "model.compute_fid=false",
        "model.log_images_every_n_steps=0",
        "wandb.name=smoke_e2e",
    ]
    if args.patch_based:
        command.extend(
            [
                f"model.patch_size={args.patch_size}",
                f"model.patch_stride={args.patch_stride}",
                f"model.patch_reconstruction={args.patch_reconstruction}",
            ]
        )
    return command


def _build_extract_command(args: argparse.Namespace, subset_dir: Path, stage1_root: Path, ar_root: Path) -> list[str]:
    return [
        sys.executable,
        "extract_token_cache.py",
        "--dataset",
        "celeba",
        "--data-dir",
        str(subset_dir),
        "--split",
        "train",
        "--batch-size",
        str(args.stage1_batch_size),
        "--num-workers",
        "0",
        "--image-size",
        str(args.image_size),
        "--max-items",
        str(args.token_cache_items),
        "--device",
        args.device,
        "--output-root",
        str(stage1_root),
        "--ar-output-dir",
        str(ar_root),
        "--coeff-bins",
        str(args.coeff_bins),
        "--coeff-max",
        "auto",
    ]


def _build_stage2_command(args: argparse.Namespace, subset_dir: Path, ar_root: Path) -> list[str]:
    return [
        sys.executable,
        "train_ar.py",
        f"output_dir={ar_root}",
        "token_cache_path=null",
        "data.dataset=celeba",
        f"data.data_dir={subset_dir}",
        f"data.image_size={args.image_size}",
        "data.num_workers=0",
        "ar.type=sparse_spatial_depth",
        f"ar.d_model={args.stage2_d_model}",
        f"ar.n_heads={args.stage2_n_heads}",
        f"ar.n_layers={args.stage2_n_layers}",
        f"ar.d_ff={args.stage2_d_ff}",
        "ar.dropout=0.0",
        "ar.learning_rate=1e-3",
        f"train_ar.batch_size={args.stage2_batch_size}",
        f"train_ar.max_epochs={args.stage2_epochs}",
        "train_ar.accelerator=cpu" if args.device == "cpu" else f"train_ar.accelerator={args.train_accelerator}",
        f"train_ar.devices={args.devices}",
        "train_ar.precision=32",
        "train_ar.log_every_n_steps=1",
        f"train_ar.validation_split={args.stage2_validation_split}",
        f"train_ar.test_split={args.stage2_test_split}",
        f"train_ar.sample_every_n_steps={args.stage2_sample_every_n_steps}",
        f"train_ar.sample_num_images={args.stage2_sample_num_images}",
        f"train_ar.sample_temperature={args.temperature}",
        f"train_ar.sample_top_k={args.top_k}",
        "wandb.project=laser-smoke",
        "wandb.name=smoke_e2e",
    ]


def _build_generate_command(args: argparse.Namespace, stage1_root: Path, ar_root: Path) -> list[str]:
    return [
        sys.executable,
        "generate_ar.py",
        "--output-root",
        str(stage1_root),
        "--ar-output-dir",
        str(ar_root),
        "--device",
        args.device,
        "--num-samples",
        str(args.num_samples),
        "--batch-size",
        str(args.generate_batch_size),
        "--temperature",
        str(args.temperature),
        "--top-k",
        str(args.top_k),
    ]


def _summary(
    stage1_root: Path,
    ar_root: Path,
    subset_dir: Path,
    args: argparse.Namespace,
    *,
    stage1_preview: Path,
    token_cache_preview: Path,
) -> dict[str, str | int | bool]:
    stage1_ckpt = infer_latest_stage1_checkpoint(output_root=stage1_root)
    token_cache = infer_latest_token_cache(ar_output_dir=ar_root, dataset="celeba", split="train")
    stage2_ckpt = infer_latest_stage2_checkpoint(ar_output_dir=ar_root)
    if stage1_ckpt is None:
        raise RuntimeError(f"Could not infer stage-1 checkpoint under {stage1_root}")
    if token_cache is None:
        raise RuntimeError(f"Could not infer token cache under {ar_root}")
    if stage2_ckpt is None:
        raise RuntimeError(f"Could not infer stage-2 checkpoint under {ar_root}")

    generated_dir = ar_root / "generated" / stage2_ckpt.parent.name
    samples_png = generated_dir / "samples.png"
    samples_pt = generated_dir / "samples.pt"
    if not samples_png.exists() or not samples_pt.exists():
        raise RuntimeError(f"Generation outputs missing under {generated_dir}")

    return {
        "data_dir": str(args.data_dir),
        "subset_dir": str(subset_dir.resolve()),
        "subset_size": int(args.subset_size),
        "image_size": int(args.image_size),
        "patch_based": bool(args.patch_based),
        "stage1_output_root": str(stage1_root.resolve()),
        "stage1_checkpoint": str(stage1_ckpt.resolve()),
        "stage1_recon_preview": str(stage1_preview.resolve()),
        "stage1_recon_preview_autocontrast": str(
            stage1_preview.with_name(f"{stage1_preview.stem}_autocontrast{stage1_preview.suffix}").resolve()
        ),
        "ar_output_root": str(ar_root.resolve()),
        "token_cache": str(token_cache.resolve()),
        "token_cache_decode_preview": str(token_cache_preview.resolve()),
        "token_cache_decode_preview_autocontrast": str(
            token_cache_preview.with_name(
                f"{token_cache_preview.stem}_autocontrast{token_cache_preview.suffix}"
            ).resolve()
        ),
        "stage2_checkpoint": str(stage2_ckpt.resolve()),
        "samples_png": str(samples_png.resolve()),
        "samples_autocontrast_png": str(
            samples_png.with_name(f"{samples_png.stem}_autocontrast{samples_png.suffix}").resolve()
        ),
        "samples_pt": str(samples_pt.resolve()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny end-to-end LASER smoke test on a CelebA subset.")
    parser.add_argument("--data-dir", type=Path, default=None, help="CelebA image directory. Defaults to local auto-detection.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/smoke_e2e_script"),
        help="Root directory for the generated subset, checkpoints, token cache, and samples.",
    )
    parser.add_argument("--subset-size", type=int, default=8192)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--token-cache-items", type=int, default=0)
    parser.add_argument("--coeff-bins", type=int, default=256)
    parser.add_argument("--stage1-batch-size", type=int, default=4)
    parser.add_argument("--stage2-batch-size", type=int, default=4)
    parser.add_argument("--generate-batch-size", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--stage1-epochs", type=int, default=1)
    parser.add_argument("--stage2-epochs", type=int, default=1)
    parser.add_argument("--stage1-num-hiddens", type=int, default=32)
    parser.add_argument("--stage1-embedding-dim", type=int, default=8)
    parser.add_argument("--stage1-num-residual-blocks", type=int, default=1)
    parser.add_argument("--stage1-num-residual-hiddens", type=int, default=8)
    parser.add_argument("--stage2-d-model", type=int, default=32)
    parser.add_argument("--stage2-n-heads", type=int, default=4)
    parser.add_argument("--stage2-n-layers", type=int, default=2)
    parser.add_argument("--stage2-d-ff", type=int, default=64)
    parser.add_argument("--stage2-validation-split", type=float, default=0.1)
    parser.add_argument("--stage2-test-split", type=float, default=0.1)
    parser.add_argument("--stage2-sample-every-n-steps", type=int, default=0)
    parser.add_argument("--stage2-sample-num-images", type=int, default=4)
    parser.add_argument("--sparsity-level", type=int, default=8)
    parser.add_argument("--num-embeddings", type=int, default=4096)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu", "auto"])
    parser.add_argument("--train-accelerator", type=str, default="gpu")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--patch-based", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--patch-stride", type=int, default=2)
    parser.add_argument("--patch-reconstruction", type=str, default="hann", choices=["hann", "center_crop"])
    parser.add_argument("--clean", action="store_true", help="Delete output-root before recreating generated artifacts.")
    parser.add_argument("--refresh-subset", action="store_true", help="Rebuild the symlinked subset even if it exists.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.data_dir = (args.data_dir.expanduser().resolve() if args.data_dir else _default_celeba_dir())
    output_root = args.output_root.expanduser().resolve()
    stage1_root = output_root / "stage1"
    ar_root = output_root / "ar"
    subset_dir = output_root / f"celeba_subset_{args.subset_size}"
    summary_path = output_root / "summary.json"

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    subset_dir = _ensure_subset(args.data_dir, subset_dir, args.subset_size, refresh=args.refresh_subset)
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")

    _run(_build_stage1_command(args, subset_dir, stage1_root), env=env)
    stage1_preview = _write_stage1_preview(stage1_root, subset_dir, image_size=args.image_size)
    _run(_build_extract_command(args, subset_dir, stage1_root, ar_root), env=env)
    token_cache_preview = _write_token_cache_preview(ar_root, stage1_root)
    _run(_build_stage2_command(args, subset_dir, ar_root), env=env)
    _run(_build_generate_command(args, stage1_root, ar_root), env=env)

    summary = _summary(
        stage1_root,
        ar_root,
        subset_dir,
        args,
        stage1_preview=stage1_preview,
        token_cache_preview=token_cache_preview,
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\nSmoke test complete.", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    print(f"Stage-1 preview: {summary['stage1_recon_preview']}", flush=True)
    print(f"Token-cache preview: {summary['token_cache_decode_preview']}", flush=True)
    print(f"Samples: {summary['samples_png']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
