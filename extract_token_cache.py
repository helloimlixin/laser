"""Extract a maintained quantized sparse-token cache from a trained LASER model."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Tuple

import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.celeba import CelebADataModule
from src.data.cifar10 import CIFAR10DataModule
from src.data.config import DataConfig
from src.data.imagenette2 import Imagenette2DataModule
from src.models.laser import LASER
from src.stage2_paths import default_token_cache_path, infer_latest_stage1_checkpoint


def _dataset_defaults(dataset: str) -> dict:
    dataset = str(dataset).strip().lower()
    if dataset == "cifar10":
        return {
            "mean": (0.4914, 0.4822, 0.4465),
            "std": (0.2470, 0.2435, 0.2616),
            "image_size": 32,
            "data_dir": "../data",
        }
    if dataset == "celeba":
        return {
            "mean": (0.5, 0.5, 0.5),
            "std": (0.5, 0.5, 0.5),
            "image_size": 128,
            "data_dir": "../data/celeba",
        }
    if dataset == "imagenette2":
        return {
            "mean": (0.5, 0.5, 0.5),
            "std": (0.5, 0.5, 0.5),
            "image_size": 224,
            "data_dir": "../data/imagenette2",
        }
    raise ValueError(f"Unsupported dataset: {dataset!r}")


def _build_datamodule(args: argparse.Namespace):
    defaults = _dataset_defaults(args.dataset)
    config = DataConfig(
        dataset=str(args.dataset),
        data_dir=str(args.data_dir or defaults["data_dir"]),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        image_size=int(args.image_size or defaults["image_size"]),
        seed=int(args.seed),
        mean=tuple(args.mean) if args.mean is not None else defaults["mean"],
        std=tuple(args.std) if args.std is not None else defaults["std"],
        augment=False,
    )
    if config.dataset == "cifar10":
        dm = CIFAR10DataModule(config)
    elif config.dataset == "celeba":
        dm = CelebADataModule(config)
    elif config.dataset == "imagenette2":
        dm = Imagenette2DataModule(config)
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset!r}")
    dm.prepare_data()
    dm.setup("fit")
    return dm, config


def _split_dataset(datamodule, split: str) -> Dataset:
    split = str(split).strip().lower()
    if split == "train":
        dataset = getattr(datamodule, "train_dataset", None) or getattr(datamodule, "cifar_train", None)
    elif split == "val":
        dataset = getattr(datamodule, "val_dataset", None) or getattr(datamodule, "cifar_val", None)
    elif split == "test":
        test_loader = getattr(datamodule, "test_dataloader", None)
        if callable(test_loader):
            loader = test_loader()
            dataset = getattr(loader, "dataset", None)
        else:
            dataset = None
    else:
        raise ValueError(f"Unsupported split: {split!r}")
    if dataset is None:
        raise RuntimeError(f"Could not resolve dataset split {split!r}")
    return dataset


def _stable_loader(dataset: Dataset, *, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _auto_coeff_max(
    model: LASER,
    loader: Iterable,
    *,
    device: torch.device,
    max_items: int,
) -> float:
    seen = 0
    coeff_max = 0.0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            images = images.to(device=device, non_blocking=True)
            _, _, sparse_codes = model.encode(images)
            batch_max = float(sparse_codes.values.abs().max().item())
            coeff_max = max(coeff_max, batch_max)
            seen += int(images.size(0))
            if max_items > 0 and seen >= max_items:
                break
    return max(coeff_max, 1e-6)


def _extract_cache(
    model: LASER,
    loader: Iterable,
    *,
    device: torch.device,
    max_items: int,
    coeff_vocab_size: int,
    coeff_max: float,
    coeff_quantization: str,
    coeff_mu: float,
) -> dict:
    """Extract a token cache.

    When *coeff_vocab_size* > 0 the cache contains interleaved quantized
    tokens (``tokens_flat``).  When *coeff_vocab_size* == 0 the cache
    stores raw atom ids and real-valued coefficients separately
    (``tokens_flat`` for atom ids, ``coeffs_flat`` for coefficients).
    """
    real_valued = int(coeff_vocab_size) <= 0
    all_tokens = []
    all_coeffs = [] if real_valued else None
    token_shape = None
    latent_hw = None
    seen = 0

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            if max_items > 0:
                keep = min(int(images.size(0)), max_items - seen)
                if keep <= 0:
                    break
                images = images[:keep]
            images = images.to(device=device, non_blocking=True)

            if real_valued:
                support, values, batch_latent_hw = model.encode_to_atoms_and_coeffs(images)
                # support: [B, H, W, D] atom ids, values: [B, H, W, D] coeffs
                flat_atoms = support.view(support.size(0), -1).to(torch.int32).cpu()
                flat_coeffs = values.view(values.size(0), -1).to(torch.float32).cpu()
                all_tokens.append(flat_atoms)
                all_coeffs.append(flat_coeffs)
                current_shape = (int(support.shape[1]), int(support.shape[2]), int(support.shape[3]))
            else:
                tokens, batch_latent_hw = model.encode_to_tokens(
                    images,
                    coeff_vocab_size=coeff_vocab_size,
                    coeff_max=coeff_max,
                    coeff_quantization=coeff_quantization,
                    coeff_mu=coeff_mu,
                )
                flat = tokens.view(tokens.size(0), -1).to(torch.int32).cpu()
                all_tokens.append(flat)
                current_shape = (int(tokens.shape[1]), int(tokens.shape[2]), int(tokens.shape[3]))

            if token_shape is None:
                token_shape = current_shape
                latent_hw = batch_latent_hw
            else:
                if current_shape != token_shape:
                    raise RuntimeError(f"Token grid shape changed across batches: {token_shape} vs {current_shape}")
                if batch_latent_hw != latent_hw:
                    raise RuntimeError(f"Latent shape changed across batches: {latent_hw} vs {batch_latent_hw}")

            seen += int(images.size(0))
            if max_items > 0 and seen >= max_items:
                break

    if not all_tokens or token_shape is None or latent_hw is None:
        raise RuntimeError("No items were encoded into a token cache")
    tokens_flat = torch.cat(all_tokens, dim=0)
    if max_items > 0:
        tokens_flat = tokens_flat[:max_items]
    result = {
        "tokens_flat": tokens_flat.contiguous(),
        "shape": token_shape,
        "latent_hw": latent_hw,
    }
    if all_coeffs is not None:
        coeffs_flat = torch.cat(all_coeffs, dim=0)
        if max_items > 0:
            coeffs_flat = coeffs_flat[:max_items]
        result["coeffs_flat"] = coeffs_flat.contiguous()
    return result


def _token_cache_meta(
    *,
    args: argparse.Namespace,
    model: LASER,
    config: DataConfig,
    num_items: int,
    latent_hw: Tuple[int, int],
) -> dict:
    real_valued = int(args.coeff_bins) <= 0
    if real_valued:
        coeff_bin_values = None
    else:
        coeff_bin_values = model.bottleneck._coeff_bin_values(
            coeff_vocab_size=int(args.coeff_bins),
            coeff_max=float(args.coeff_max),
            coeff_quantization=str(args.coeff_quantization),
            coeff_mu=float(args.coeff_mu),
            device=torch.device("cpu"),
            dtype=torch.float32,
        ).cpu()
    meta = {
        "version": 1,
        "dataset": str(config.dataset),
        "split": str(args.split),
        "image_size": int(config.image_size if isinstance(config.image_size, int) else config.image_size[0]),
        "seed": int(args.seed),
        "num_items": int(num_items),
        "num_atoms": int(model.bottleneck.num_embeddings),
        "sparsity_level": int(model.bottleneck.sparsity_level),
        "patch_based": bool(model.bottleneck.patch_based),
        "patch_size": int(model.bottleneck.patch_size),
        "patch_stride": int(model.bottleneck.patch_stride),
        "patch_reconstruction": str(model.bottleneck.patch_reconstruction),
        "embedding_dim": int(model.bottleneck.embedding_dim),
        "latent_hw": (int(latent_hw[0]), int(latent_hw[1])),
        "coef_max": float(args.coeff_max),
        "stage1_checkpoint": str(Path(args.stage1_checkpoint).expanduser().resolve()),
    }
    if real_valued:
        meta["quantize_sparse_coeffs"] = False
    else:
        meta["quantize_sparse_coeffs"] = True
        meta["n_bins"] = int(args.coeff_bins)
        meta["coeff_vocab_size"] = int(args.coeff_bins)
        meta["coef_quantization"] = str(args.coeff_quantization)
        meta["coef_mu"] = float(args.coeff_mu)
        meta["coeff_bin_values"] = coeff_bin_values
    return meta


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a quantized sparse-token cache from a LASER checkpoint.")
    parser.add_argument("--stage1-checkpoint", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "celeba", "imagenette2"])
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--ar-output-dir", type=Path, default=Path("outputs/ar"))
    parser.add_argument("--coeff-bins", type=int, default=256)
    parser.add_argument(
        "--coeff-max",
        type=str,
        default="auto",
        help="Coefficient quantization range. Use 'auto' to scan the chosen split for the max absolute coeff.",
    )
    parser.add_argument("--coeff-quantization", type=str, default="uniform", choices=["uniform", "mu_law"])
    parser.add_argument("--coeff-mu", type=float, default=0.0)
    parser.add_argument("--mean", type=float, nargs=3, default=None)
    parser.add_argument("--std", type=float, nargs=3, default=None)
    return parser.parse_args()


def main():
    args = _parse_args()
    pl.seed_everything(int(args.seed), workers=True)

    if args.stage1_checkpoint is None:
        inferred_stage1 = infer_latest_stage1_checkpoint(output_root=args.output_root, model_type="laser")
        if inferred_stage1 is None:
            raise FileNotFoundError(
                f"Could not infer a LASER checkpoint under {(Path(args.output_root).expanduser().resolve() / 'checkpoints')}"
            )
        args.stage1_checkpoint = inferred_stage1
        print(f"Inferred stage1 checkpoint: {args.stage1_checkpoint}")

    device = _resolve_device(str(args.device))
    datamodule, config = _build_datamodule(args)
    dataset = _split_dataset(datamodule, args.split)
    loader = _stable_loader(
        dataset,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )

    model = LASER.load_from_checkpoint(
        args.stage1_checkpoint,
        map_location="cpu",
        compute_fid=False,
    )
    model.eval().to(device)

    coeff_max_arg = str(args.coeff_max).strip().lower()
    if coeff_max_arg == "auto":
        coeff_max = _auto_coeff_max(
            model,
            loader,
            device=device,
            max_items=int(args.max_items),
        )
        print(f"Resolved coeff_max from data: {coeff_max:.6f}")
    else:
        coeff_max = float(args.coeff_max)
    args.coeff_max = coeff_max

    if args.output_path is None:
        image_size = int(config.image_size if isinstance(config.image_size, int) else config.image_size[0])
        args.output_path = default_token_cache_path(
            ar_output_dir=args.ar_output_dir,
            dataset=config.dataset,
            split=args.split,
            image_size=image_size,
            coeff_bins=int(args.coeff_bins),
            coeff_quantization=str(args.coeff_quantization),
        )
        print(f"Defaulting token cache output to: {args.output_path}")

    cache_result = _extract_cache(
        model,
        loader,
        device=device,
        max_items=int(args.max_items),
        coeff_vocab_size=int(args.coeff_bins),
        coeff_max=float(coeff_max),
        coeff_quantization=str(args.coeff_quantization),
        coeff_mu=float(args.coeff_mu),
    )
    tokens_flat = cache_result["tokens_flat"]
    shape = cache_result["shape"]
    latent_hw = cache_result["latent_hw"]

    meta = _token_cache_meta(
        args=args,
        model=model,
        config=config,
        num_items=int(tokens_flat.size(0)),
        latent_hw=latent_hw,
    )
    payload = {
        "tokens_flat": tokens_flat,
        "shape": shape,
        "meta": meta,
    }
    if "coeffs_flat" in cache_result:
        payload["coeffs_flat"] = cache_result["coeffs_flat"]
        print(f"Real-valued cache: storing atoms + coefficients (no quantization)")

    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(
        f"Saved token cache: {output_path} "
        f"(items={tokens_flat.size(0)}, shape={shape}, latent_hw={latent_hw})"
    )


if __name__ == "__main__":
    main()
