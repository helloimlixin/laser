#!/usr/bin/env python3
"""Render fixed ImageNet reconstructions for exact available Stage-1 checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = Path("/workspace/tmp/rqvae-rfid4483-continuation-r11-20260828")
TAMING = Path("/workspace/tmp/taming-transformers")
sys.path[:0] = [str(TAMING), str(UPSTREAM), str(ROOT)]

from rqvae.models import create_model


ROWS = (
    ("VQGAN  16x16x1  K=16,384  rFID 4.90", "vqgan16"),
    ("RQ-VAE  8x8x4  K=16,384  rFID 4.73", "rq4"),
    ("LASER   8x8x2  K=16,384  rFID 7.79", "laser2"),
    ("LASER   8x8x4  K=16,384  rFID 4.21", "laser4"),
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_model(config_path: Path, checkpoint: Path, *, laser_depth: int | None):
    config = OmegaConf.load(config_path)
    if laser_depth is not None:
        config.arch.hparams.bottleneck_type = "laser"
        config.arch.hparams.code_shape = [8, 8, laser_depth]
        config.arch.hparams.sparsity_level = laser_depth
        config.arch.hparams.commitment_cost = 1.0
        config.arch.hparams.progressive_loss = True
        config.arch.hparams.use_padding_idx = False
        config.arch.hparams.patch_based = False
        config.arch.hparams.patch_size = 1
        config.arch.hparams.patch_stride = 1
        config.arch.hparams.data_init_from_first_batch = True
        config.arch.hparams.dead_atom_revival = True
        config.arch.hparams.dead_atom_revival_interval = 500
        config.arch.hparams.dead_atom_revival_max_fraction = 0.05
        config.arch.hparams.dead_atom_revival_noise = 0.05
        config.arch.hparams.dead_atom_revival_patience = 5
        config.arch.hparams.omp_compute_precision = "float32"
        config.arch.hparams.dictionary_update_mode = "alternating_residual"
        config.arch.hparams.dictionary_update_relaxation = 0.25
        config.arch.hparams.dictionary_update_max_atoms_per_step = 16384
        config.arch.hparams.dictionary_update_min_usage = 2
        config.arch.hparams.dictionary_update_max_backtracks = 6
    model, _ = create_model(config.arch, ema=False)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    state = payload.get("state_dict", payload)
    model.load_state_dict(state, strict=True)
    return model.eval()


def load_vqgan(config_path: Path, checkpoint: Path):
    # Compatibility shim for the archived 2021 release on current PyTorch.
    import types
    torch_six = types.ModuleType("torch._six")
    torch_six.string_classes = (str, bytes)
    sys.modules.setdefault("torch._six", torch_six)
    from taming.models.vqgan import VQModel

    config = OmegaConf.load(config_path)
    params = OmegaConf.to_container(config.model.params, resolve=True)
    # The loss is training-only; avoiding LPIPS construction also avoids an
    # unrelated pretrained-weight download during reconstruction rendering.
    params["lossconfig"] = {"target": "torch.nn.Identity"}
    model = VQModel(**params)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=False)
    return model.eval()


def image_tensor(paths: list[Path]) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    return torch.stack([transform(Image.open(path).convert("RGB")) for path in paths])


def strip(images: torch.Tensor) -> Image.Image:
    array = (images.mul(0.5).add(0.5).clamp(0, 1).permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)
    canvas = Image.new("RGB", (256 * len(array), 256), "white")
    for index, item in enumerate(array):
        canvas.paste(Image.fromarray(item), (256 * index, 0))
    return canvas


def labeled_rows(rows: list[tuple[str, Image.Image]], path: Path) -> None:
    label_width = 410
    canvas = Image.new("RGB", (label_width + 2048, 256 * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, (label, panel) in enumerate(rows):
        y = row * 256
        draw.text((12, y + 116), label, fill="black")
        canvas.paste(panel, (label_width, y))
    canvas.save(path)


def image_rows(rows: list[tuple[str, Image.Image]], path: Path) -> None:
    canvas = Image.new("RGB", (2048, 256 * len(rows)), "white")
    for row, (_, panel) in enumerate(rows):
        canvas.paste(panel, (0, row * 256))
    canvas.save(path)


def heatmap(coefficients: torch.Tensor) -> Image.Image:
    energy = coefficients.float().square().sum(-1).sqrt()
    lo = energy.amin((1, 2), keepdim=True)
    hi = energy.amax((1, 2), keepdim=True)
    norm = ((energy - lo) / (hi - lo).clamp_min(1e-8)).cpu().numpy()
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import colormaps
    rgb = (colormaps["magma"](norm)[..., :3] * 255).round().astype(np.uint8)
    canvas = Image.new("RGB", (2048, 256), "white")
    for index, item in enumerate(rgb):
        canvas.paste(Image.fromarray(item).resize((256, 256), Image.Resampling.NEAREST), (256 * index, 0))
    return canvas


def rgb_latent_code_heatmaps(
    projection_pairs: list[tuple[str, torch.Tensor, torch.Tensor]],
    references: torch.Tensor,
    path: Path,
) -> None:
    """Render references, latent energy, and sparse-coefficient energy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    def attention(values: torch.Tensor) -> torch.Tensor:
        energy = values.float().square().mean(-1).sqrt()
        low = energy.amin((1, 2), keepdim=True)
        high = energy.amax((1, 2), keepdim=True)
        return (energy - low) / (high - low).clamp_min(1e-8)

    projected = [
        (label, attention(latent), attention(coefficients))
        for label, latent, coefficients in projection_pairs
    ]
    reference_images = references.mul(0.5).add(0.5).clamp(0, 1).permute(0, 2, 3, 1)
    rows = [(reference_images, None, "Reference")]
    rows += [(latent, "inferno", "Latent energy") for _, latent, _ in projected]
    rows += [(coefficients, "inferno", "Sparse-code energy") for _, _, coefficients in projected]
    nrows = len(rows)
    fig = plt.figure(figsize=(16.5, 2.05 * nrows), constrained_layout=True)
    grid = fig.add_gridspec(nrows, 9, width_ratios=[1] * 8 + [0.22])
    for row_index, (values, cmap, row_label) in enumerate(rows):
        for image_index in range(8):
            ax = fig.add_subplot(grid[row_index, image_index])
            image = values[image_index].numpy()
            if cmap is None:
                ax.imshow(image)
            else:
                ax.imshow(
                    image,
                    cmap=cmap,
                    vmin=0.0,
                    vmax=1.0,
                    interpolation="nearest",
                )
            ax.set_xticks([])
            ax.set_yticks([])
            if image_index == 0:
                ax.set_ylabel(row_label, fontsize=9, labelpad=6)
            for spine in ax.spines.values():
                spine.set_linewidth(0.35)
    color_ax = fig.add_subplot(grid[1:, 8])
    fig.colorbar(
        ScalarMappable(norm=Normalize(0.0, 1.0), cmap="inferno"),
        cax=color_ax,
        label="normalized spatial energy",
    )
    fig.suptitle("LASER latent and sparse coefficient heatmaps", fontsize=15)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def dictionary_scatter(models: dict[str, torch.nn.Module], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    projected = []
    for name, model in models.items():
        atoms = model.quantizer.dictionary.detach().t().float().cpu().numpy()
        atoms -= atoms.mean(0, keepdims=True)
        _, singular, basis = np.linalg.svd(atoms, full_matrices=False)
        points = atoms @ basis[:2].T
        total = float(np.square(singular).sum())
        explained = np.square(singular[:2]) / total * 100 if total else np.zeros(2)
        projected.append((name, points, explained))
    axis_limit = max(float(np.abs(points).max()) for _, points, _ in projected) * 1.03
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for ax, (name, points, explained) in zip(axes, projected):
        ax.scatter(points[:, 0], points[:, 1], s=3, alpha=0.28, rasterized=True)
        ax.set_title(name)
        ax.set_xlabel(f"PC1 ({explained[0]:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({explained[1]:.1f}% variance)")
        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_ylim(-axis_limit, axis_limit)
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Learned LASER dictionary atoms (all 16,384 atoms; PCA per checkpoint)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoints = {
        "vqgan16": Path("/workspace/tmp/vqgan-imagenet-f16-16384/checkpoints/last.ckpt"),
        "rq4": Path("/workspace/tmp/original-rqvae-473/published-stage1/model.pt"),
        "laser2": ROOT / "outputs/imagenet-stage1-table-recon-20260915/checkpoints/laser-8x8x2-rfid7p79/best_rfid_slot1_model.pt",
        "laser4": ROOT / "outputs/imagenet-scaled-rq-stage2-20260913/assets/best_rfid_slot1_model.pt",
    }
    vqgan_config = Path("/workspace/tmp/vqgan-imagenet-f16-16384/configs/model.yaml")
    rq_config = Path("/workspace/tmp/original-rqvae-473/published-stage1/config.yaml")
    laser_config = ROOT / "outputs/imagenet-laser-8gpu-rqvae-dynamics-20260829-054801/imagenet_laser_8x8x4_rqvae_dynamics/29082026_054811/config.yaml"
    val_root = Path("/workspace/Projects/data/imagenet2012/val")
    image_paths = [sorted(class_dir.glob("*.JPEG"))[0] for class_dir in sorted(val_root.iterdir()) if class_dir.is_dir()][:8]
    inputs = image_tensor(image_paths)
    strip(inputs).save(args.output / "reference-images.png")
    rows, heatmaps, laser_models = [], [], {}
    projection_pairs = []
    metrics = {}
    for label, key in ROWS:
        depth = {"laser2": 2, "laser4": 4}.get(key)
        if key == "vqgan16":
            model = load_vqgan(vqgan_config, checkpoints[key]).to(args.device)
        else:
            model = load_model(laser_config if depth else rq_config, checkpoints[key], laser_depth=depth).to(args.device)
        x = inputs.to(args.device)
        if depth:
            z = model.encode(x)
            quantized, _, codes = model.quantizer(z.float())
            recon = model.decode(quantized.to(z.dtype))
            heatmaps.append((label, heatmap(codes.values)))
            laser_models[label] = model.cpu()
            if key == "laser4":
                projection_pairs.append((
                    label,
                    quantized.float().permute(0, 2, 3, 1).cpu(),
                    codes.values.float().cpu(),
                ))
        elif key == "vqgan16":
            z = model.quant_conv(model.encoder(x))
            quantized, _, info = model.quantize(z)
            recon = model.decode(quantized)
        else:
            z = model.encode(x)
            quantized, _, codes = model.quantizer(z)
            recon = model.decode(quantized)
        rows.append((label, strip(recon.cpu())))
        metrics[key] = {"mse": float((recon.cpu() - inputs).square().mean()), "checkpoint_sha256": sha256(checkpoints[key])}
        del model, recon
        torch.cuda.empty_cache()
    image_rows([("Reference images", strip(inputs)), *rows], args.output / "reconstruction-grid.png")
    labeled_rows(heatmaps, args.output / "laser-sparse-coefficient-heatmaps.png")
    rgb_latent_code_heatmaps(projection_pairs, inputs, args.output / "latent-and-code-heatmaps.png")
    dictionary_scatter(laser_models, args.output / "laser-dictionary-atoms-pca.png")
    manifest = {
        "image_paths": [str(path) for path in image_paths], "images_per_row": 8,
        "reconstruction_reference_row": "top",
        "rows_rendered": [label for label, _ in ROWS], "metrics": metrics,
        "unavailable_exact_rows": [
            "VQGAN-dagger 16x16x1 K=16384 rFID 4.32",
            "VQGAN 8x8x1 K=16384 rFID 17.95", "VQGAN 8x8x1 K=65536 rFID 17.66",
            "VQGAN 8x8x1 K=131072 rFID 17.09", "RQ-VAE 8x8x2 K=16384 rFID 10.77",
        ],
        "provenance": {
            "vqgan16": {
                "table_rfid": 4.90,
                "current_compvis_readme_rfid": 4.98,
                "source": "https://github.com/CompVis/taming-transformers",
                "release_name": "ImageNet f=16 (16384 indices)",
            }
        },
        "latent_code_heatmap": {
            "models": ["LASER 8x8x4 rFID 4.21"],
            "row_order": "reference images, latent energy, then sparse-code energy",
            "projection": "RMS energy across latent channels and across the four raw coefficient layers",
            "palette": "inferno attention heatmap",
            "scale": "per-image spatial min-max normalization to a shared [0, 1] scale",
            "common_colorbars": True,
        },
        "note": "Only exact available checkpoints were rendered; unavailable rows were not replaced by non-equivalent checkpoints.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
