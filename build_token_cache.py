#!/usr/bin/env python3
"""Build a maintained sparse-token cache from a stage-1 LASER checkpoint."""

import argparse
from pathlib import Path

import torch

from src.data.celeba import CelebADataModule
from src.data.cifar10 import CIFAR10DataModule
from src.data.config import DataConfig
from src.data.imagenette2 import Imagenette2DataModule
from src.models.laser import LASER
from src.stage2_paths import default_token_cache_path


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _build_datamodule(args):
    data_cfg = DataConfig(
        dataset=args.dataset,
        data_dir=str(Path(args.data_dir).expanduser().resolve()),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        image_size=int(args.image_size),
        mean=tuple(float(x) for x in args.mean),
        std=tuple(float(x) for x in args.std),
    )
    if args.dataset == "cifar10":
        return CIFAR10DataModule(data_cfg)
    if args.dataset == "imagenette2":
        return Imagenette2DataModule(data_cfg)
    if args.dataset == "celeba":
        return CelebADataModule(data_cfg)
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def _resolve_loader(datamodule, split: str):
    split = str(split).strip().lower()
    datamodule.setup("fit" if split in {"train", "val"} else "test")
    if split == "train":
        return datamodule.train_dataloader()
    if split == "val":
        return datamodule.val_dataloader()
    if split == "test":
        return datamodule.test_dataloader()
    raise ValueError(f"Unsupported split: {split!r}")


def _default_output_path(args):
    cache_mode = str(args.cache_mode).strip().lower()
    if cache_mode == "quantized":
        return default_token_cache_path(
            ar_output_dir=args.ar_output_dir,
            dataset=args.dataset,
            split=args.split,
            image_size=int(args.image_size),
            coeff_bins=int(args.coeff_vocab_size),
            coeff_quantization=str(args.coeff_quantization),
        )
    return (
        Path(args.ar_output_dir).expanduser().resolve()
        / "token_cache"
        / f"{str(args.dataset).strip().lower()}__{str(args.split).strip().lower()}__img{int(args.image_size)}__real.pt"
    )


def main():
    parser = argparse.ArgumentParser(description="Build a maintained sparse-token cache from a stage-1 checkpoint.")
    parser.add_argument("--stage1_checkpoint", type=Path, required=True, help="Maintained stage-1 Lightning checkpoint.")
    parser.add_argument("--dataset", type=str, default="celeba", choices=["cifar10", "imagenette2", "celeba"])
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset root directory.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument(
        "--cache_mode",
        type=str,
        default="quantized",
        choices=["quantized", "real_valued"],
        help="Emit either interleaved quantized sparse tokens or real-valued atom+coeff caches.",
    )
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mean", type=float, nargs=3, default=(0.5, 0.5, 0.5))
    parser.add_argument("--std", type=float, nargs=3, default=(0.5, 0.5, 0.5))
    parser.add_argument("--coeff_vocab_size", type=int, default=16, help="Number of coefficient bins.")
    parser.add_argument(
        "--coeff_max",
        type=float,
        default=None,
        help="Absolute coefficient clip. Quantized caches default to 3.0 when omitted; real-valued caches infer from data when omitted.",
    )
    parser.add_argument("--coeff_quantization", type=str, default="uniform", choices=["uniform", "mu_law"])
    parser.add_argument("--coeff_mu", type=float, default=0.0, help="Mu parameter when coeff_quantization=mu_law.")
    parser.add_argument("--output", type=Path, default=None, help="Output token cache path.")
    parser.add_argument("--ar_output_dir", type=Path, default=Path("outputs/ar"), help="Default root for token caches.")
    parser.add_argument("--max_items", type=int, default=0, help="Optional cap on number of encoded items.")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    checkpoint_path = Path(args.stage1_checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {checkpoint_path}")

    device = _resolve_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("medium")

    model = LASER.load_from_checkpoint(str(checkpoint_path), map_location="cpu").eval().to(device)
    datamodule = _build_datamodule(args)
    loader = _resolve_loader(datamodule, args.split)
    cache_mode = str(args.cache_mode).strip().lower()
    quantized_cache = cache_mode == "quantized"
    coeff_max = args.coeff_max
    if coeff_max is None:
        coeff_max = 3.0 if quantized_cache else 0.0
    coeff_max = float(coeff_max)
    if quantized_cache and coeff_max <= 0.0:
        raise ValueError("Quantized token caches require coeff_max > 0.")

    coeff_bin_values = None
    if quantized_cache:
        coeff_bin_values = model.bottleneck._coeff_bin_values(
            coeff_vocab_size=int(args.coeff_vocab_size),
            coeff_max=float(coeff_max),
            coeff_quantization=str(args.coeff_quantization),
            coeff_mu=float(args.coeff_mu),
            device=torch.device("cpu"),
            dtype=torch.float32,
        ).cpu()
    latent_hw = model.infer_latent_hw((int(args.image_size), int(args.image_size)))

    all_tokens = []
    all_coeffs = []
    seen = 0
    token_shape = None
    observed_coeff_abs_max = 0.0
    with torch.no_grad():
        for batch in loader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(device, non_blocking=True)
            if quantized_cache:
                tokens, current_latent_hw = model.encode_to_tokens(
                    x,
                    coeff_vocab_size=int(args.coeff_vocab_size),
                    coeff_max=float(coeff_max),
                    coeff_quantization=str(args.coeff_quantization),
                    coeff_mu=float(args.coeff_mu),
                )
                coeffs = None
            else:
                tokens, coeffs, current_latent_hw = model.encode_to_atoms_and_coeffs(x)
            if tuple(current_latent_hw) != tuple(latent_hw):
                raise RuntimeError(
                    "latent_hw changed across batches: "
                    f"expected {tuple(latent_hw)}, got {tuple(current_latent_hw)}"
                )
            if token_shape is None:
                token_shape = tuple(int(dim) for dim in tokens.shape[1:])
            flat = tokens.view(tokens.size(0), -1).to(torch.int32).cpu()
            all_tokens.append(flat)
            if coeffs is not None:
                coeff_flat = coeffs.view(coeffs.size(0), -1).to(torch.float32).cpu()
                all_coeffs.append(coeff_flat)
                if coeff_flat.numel() > 0:
                    observed_coeff_abs_max = max(observed_coeff_abs_max, float(coeff_flat.abs().max().item()))
            seen += flat.size(0)
            if int(args.max_items) > 0 and seen >= int(args.max_items):
                break

    if not all_tokens:
        raise RuntimeError("No items were encoded into the token cache.")

    tokens_flat = torch.cat(all_tokens, dim=0)
    coeffs_flat = None
    if all_coeffs:
        coeffs_flat = torch.cat(all_coeffs, dim=0)
    if int(args.max_items) > 0:
        tokens_flat = tokens_flat[: int(args.max_items)]
        if coeffs_flat is not None:
            coeffs_flat = coeffs_flat[: int(args.max_items)]

    if (not quantized_cache) and coeff_max <= 0.0:
        coeff_max = max(1.0, float(observed_coeff_abs_max))

    token_h, token_w, token_depth = token_shape
    meta = {
        "dataset": str(args.dataset),
        "split": str(args.split),
        "image_size": int(args.image_size),
        "stage1_checkpoint": str(checkpoint_path),
        "num_atoms": int(model.bottleneck.num_embeddings),
        "sparsity_level": int(model.hparams.sparsity_level),
        "quantize_sparse_coeffs": bool(quantized_cache),
        "coeff_max": float(coeff_max),
        "coef_max": float(coeff_max),
        "latent_hw": (int(latent_hw[0]), int(latent_hw[1])),
        "patch_based": bool(model.hparams.patch_based),
        "patch_size": int(model.hparams.patch_size),
        "patch_stride": int(model.hparams.patch_stride),
        "patch_reconstruction": str(model.hparams.patch_reconstruction),
        "variational_coeffs": bool(getattr(model.hparams, "variational_coeffs", False)),
        "variational_coeff_prior_std": float(getattr(model.hparams, "variational_coeff_prior_std", 0.25)),
        "variational_coeff_min_std": float(getattr(model.hparams, "variational_coeff_min_std", 0.01)),
    }
    if quantized_cache:
        meta.update(
            {
                "coeff_vocab_size": int(args.coeff_vocab_size),
                "n_bins": int(args.coeff_vocab_size),
                "coeff_bin_values": coeff_bin_values,
                "coeff_quantization": str(args.coeff_quantization),
                "coeff_mu": float(args.coeff_mu),
            }
        )
    cache = {
        "tokens_flat": tokens_flat,
        "shape": (int(token_h), int(token_w), int(token_depth)),
        "meta": meta,
    }
    if coeffs_flat is not None:
        cache["coeffs_flat"] = coeffs_flat

    output_path = args.output
    if output_path is None:
        output_path = _default_output_path(args)
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output_path)

    print(f"Stage-1 checkpoint: {checkpoint_path}")
    print(f"Saved token cache:  {output_path}")
    print(f"Cache mode:         {cache_mode}")
    print(f"Token tensor:       {tuple(tokens_flat.shape)}")
    if coeffs_flat is not None:
        print(f"Coeff tensor:       {tuple(coeffs_flat.shape)}")
        print(f"Coeff max:          {float(coeff_max):g}")
    print(f"Token grid:         {(int(token_h), int(token_w), int(token_depth))}")
    print(f"Latent HW:          {tuple(latent_hw)}")


if __name__ == "__main__":
    main()
