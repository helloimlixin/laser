#!/usr/bin/env python3
"""Build a maintained sparse-token cache from a stage-1 LASER checkpoint."""

import argparse
import math
from pathlib import Path
import sys

import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cache_sort import sort_sparse_pairs, sort_token_pairs
from src.checkpoint_io import load_lightning_module
from src.data.imagenet_labels import class_names_for_dataset, imagenet_synsets_from_names
from src.models.laser import LASER
from src.sparse_token_codec import build_coeff_bin_values
from src.stage1_setup import build_stage1_datamodule, data_config_from_overrides
from src.stage2_paths import default_token_cache_path


IMAGE_TOKEN_CACHE_DATASETS = [
    "cifar10",
    "celeba",
    "celebahq",
    "coco",
    "ffhq",
    "imagenet",
    "imagenette2",
    "stl10",
]

MAX_EXACT_QUANTILE_VALUES = 4_000_000


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _build_datamodule(args):
    data_cfg = data_config_from_overrides(
        dataset=args.dataset,
        data_dir=str(Path(args.data_dir).expanduser().resolve()),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        image_size=int(args.image_size),
        mean=tuple(float(x) for x in args.mean),
        std=tuple(float(x) for x in args.std),
        augment=False,
    )
    return build_stage1_datamodule(data_cfg)


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


def _progress_total(loader, max_items: int) -> int | None:
    try:
        dataset_size = len(loader.dataset)
    except Exception:
        dataset_size = 0
    if max_items > 0:
        return min(max_items, dataset_size) if dataset_size > 0 else max_items
    return dataset_size if dataset_size > 0 else None


def _parse_optional_coeff_max(raw) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if text in {"", "none", "null", "auto"}:
        return None
    return float(text)


def _bounded_quantile_input(values: torch.Tensor, max_values: int = MAX_EXACT_QUANTILE_VALUES) -> torch.Tensor:
    values = values.reshape(-1)
    if values.numel() <= int(max_values):
        return values
    idx = torch.linspace(
        0,
        values.numel() - 1,
        steps=int(max_values),
        dtype=torch.long,
        device=values.device,
    )
    return values[idx]


def _auto_coeff_absmax(coeffs: torch.Tensor, percentile: float) -> float:
    coeffs = coeffs.detach().reshape(-1).to(torch.float32)
    finite = coeffs[torch.isfinite(coeffs)]
    if finite.numel() <= 0:
        return 1.0
    finite = _bounded_quantile_input(finite.abs())
    pct = min(max(float(percentile), 0.0), 100.0) / 100.0
    value = finite.max() if pct >= 1.0 else torch.quantile(finite, pct)
    out = float(value.item())
    return out if math.isfinite(out) and out > 0.0 else 1.0


def _adaptive_coeff_bin_values(
    coeffs: torch.Tensor,
    *,
    coeff_vocab_size: int,
    coeff_max: float,
    coeff_quantization: str,
    coeff_mu: float,
) -> torch.Tensor:
    coeff_vocab_size = int(coeff_vocab_size)
    coeff_max = float(coeff_max)
    coeff_quantization = str(coeff_quantization).strip().lower()
    if coeff_quantization in {"uniform", "mu_law"}:
        return build_coeff_bin_values(
            coeff_vocab_size=coeff_vocab_size,
            coeff_max=coeff_max,
            coeff_quantization=coeff_quantization,
            coeff_mu=float(coeff_mu),
            device=torch.device("cpu"),
            dtype=torch.float32,
        ).cpu()
    if coeff_quantization != "quantile":
        raise ValueError(f"Unsupported coeff_quantization: {coeff_quantization!r}")

    finite = coeffs.detach().reshape(-1).to(torch.float32)
    finite = finite[torch.isfinite(finite)]
    if finite.numel() <= 0:
        return torch.linspace(-coeff_max, coeff_max, steps=coeff_vocab_size, dtype=torch.float32)
    clipped = finite.clamp(-coeff_max, coeff_max)
    clipped = _bounded_quantile_input(clipped)
    if coeff_vocab_size == 1:
        return clipped.median().reshape(1).cpu()
    qs = torch.linspace(0.0, 1.0, steps=coeff_vocab_size, dtype=torch.float32, device=clipped.device)
    bins = torch.quantile(clipped, qs).to(torch.float32).cpu()
    eps = max(float(coeff_max), 1.0) * 1e-6
    for idx in range(1, int(bins.numel())):
        if bins[idx] <= bins[idx - 1]:
            bins[idx] = bins[idx - 1] + eps
    if bins[-1] > coeff_max:
        return torch.linspace(-coeff_max, coeff_max, steps=coeff_vocab_size, dtype=torch.float32)
    return bins


def _quantize_with_bin_values(coeffs: torch.Tensor, bin_values: torch.Tensor) -> torch.Tensor:
    bins = bin_values.to(device=coeffs.device, dtype=torch.float32).reshape(-1)
    if bins.numel() <= 0:
        raise ValueError("coeff bin values must be non-empty")
    coeffs = coeffs.to(torch.float32).clamp(float(bins.min().item()), float(bins.max().item()))
    if bins.numel() == 1:
        return torch.zeros_like(coeffs, dtype=torch.long)
    boundaries = ((bins[:-1] + bins[1:]) * 0.5).contiguous()
    return torch.bucketize(coeffs.contiguous(), boundaries).to(torch.long)


def _interleave_atom_coeff_tokens(
    atom_ids: torch.Tensor,
    coeffs: torch.Tensor,
    *,
    coeff_bin_values: torch.Tensor,
    atom_vocab_size: int,
) -> torch.Tensor:
    if tuple(atom_ids.shape) != tuple(coeffs.shape):
        raise ValueError(f"atom_ids/coeffs shape mismatch: {tuple(atom_ids.shape)} vs {tuple(coeffs.shape)}")
    bin_idx = _quantize_with_bin_values(coeffs, coeff_bin_values)
    tokens = torch.empty(*atom_ids.shape[:-1], int(atom_ids.shape[-1]) * 2, dtype=torch.long)
    tokens[..., 0::2] = atom_ids.to(torch.long).cpu()
    tokens[..., 1::2] = bin_idx.cpu() + int(atom_vocab_size)
    return tokens


def _batch_class_labels(batch, keep: int) -> torch.Tensor | None:
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        return None
    labels = batch[1]
    if torch.is_tensor(labels):
        labels = labels[:keep]
    else:
        try:
            labels = torch.as_tensor(list(labels)[:keep])
        except (TypeError, ValueError):
            return None
    labels = labels.to(torch.long).reshape(-1)
    if int(labels.numel()) < int(keep):
        return None
    return labels[:keep].cpu().contiguous()


def _finalize_class_labels(chunks: list[torch.Tensor], total_items: int) -> torch.Tensor | None:
    if not chunks:
        return None
    labels = torch.cat(chunks, dim=0).to(torch.long).reshape(-1).contiguous()
    if int(labels.numel()) < int(total_items):
        return None
    return labels[: int(total_items)].clone()


def _dataset_class_names(dataset) -> list[str]:
    class_to_idx = getattr(dataset, "class_to_idx", None)
    if isinstance(class_to_idx, dict) and class_to_idx:
        ordered = sorted(((int(idx), str(name)) for name, idx in class_to_idx.items()), key=lambda item: item[0])
        return [name for _, name in ordered]
    classes = getattr(dataset, "classes", None)
    if isinstance(classes, (list, tuple)) and classes:
        return [str(item) for item in classes]
    return []


def _attach_class_label_metadata(payload: dict, class_labels: torch.Tensor, *, dataset_name: str, dataset) -> None:
    if class_labels is None or int(class_labels.numel()) != int(payload["tokens_flat"].size(0)):
        return
    raw_names = _dataset_class_names(dataset) if dataset is not None else []
    display_names = class_names_for_dataset(dataset_name, raw_names)
    max_label = int(class_labels.max().item()) if int(class_labels.numel()) > 0 else -1
    num_classes = max(max_label + 1, len(display_names))
    if num_classes <= 1 and not display_names:
        return

    payload["class_labels"] = class_labels.to(torch.long).reshape(-1).contiguous()
    payload["meta"]["num_classes"] = int(num_classes)
    payload["meta"]["class_label_name"] = "class"
    if display_names:
        payload["meta"]["class_names"] = display_names
    synsets = imagenet_synsets_from_names(dataset_name, raw_names)
    if synsets:
        payload["meta"]["class_synsets"] = synsets


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
            coeff_mu=float(args.coeff_mu),
        )
    return (
        Path(args.ar_output_dir).expanduser().resolve()
        / "token_cache"
        / f"{str(args.dataset).strip().lower()}__{str(args.split).strip().lower()}__img{int(args.image_size)}__real.pt"
    )


def main():
    parser = argparse.ArgumentParser(description="Build a maintained sparse-token cache from a stage-1 checkpoint.")
    parser.add_argument("--stage1_checkpoint", type=Path, required=True, help="Maintained stage-1 Lightning checkpoint.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="celeba",
        choices=IMAGE_TOKEN_CACHE_DATASETS,
    )
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
        type=str,
        default=None,
        help=(
            "Absolute coefficient clip. Use 'auto' or omit with quantized caches "
            "to infer from coefficient percentiles."
        ),
    )
    parser.add_argument("--coeff_quantization", type=str, default="uniform", choices=["uniform", "mu_law", "quantile"])
    parser.add_argument(
        "--coeff_calibration_percentile",
        type=float,
        default=99.5,
        help="Absolute coefficient percentile used when coeff_max is auto/omitted.",
    )
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

    model = load_lightning_module(
        LASER,
        checkpoint_path,
        map_location="cpu",
        strict=False,
        compute_fid=False,
    ).eval().to(device)
    datamodule = _build_datamodule(args)
    loader = _resolve_loader(datamodule, args.split)
    cache_mode = str(args.cache_mode).strip().lower()
    quantized_cache = cache_mode == "quantized"
    coeff_quantization = str(args.coeff_quantization).strip().lower()
    coeff_max = _parse_optional_coeff_max(args.coeff_max)
    adaptive_quantized_cache = quantized_cache and (
        coeff_max is None or coeff_quantization == "quantile"
    )
    if not quantized_cache and coeff_max is None:
        coeff_max = 0.0
    if quantized_cache and coeff_max is not None and coeff_max <= 0.0:
        raise ValueError("Quantized token caches require coeff_max > 0.")

    coeff_bin_values = None
    if quantized_cache and not adaptive_quantized_cache:
        coeff_bin_values = build_coeff_bin_values(
            coeff_vocab_size=int(args.coeff_vocab_size),
            coeff_max=float(coeff_max),
            coeff_quantization=coeff_quantization,
            coeff_mu=float(args.coeff_mu),
            device=torch.device("cpu"),
            dtype=torch.float32,
        ).cpu()
    latent_hw = model.infer_latent_hw((int(args.image_size), int(args.image_size)))

    all_tokens = []
    all_coeffs = []
    all_class_labels = []
    seen = 0
    token_shape = None
    observed_coeff_abs_max = 0.0
    max_items = int(args.max_items)
    progress_total = _progress_total(loader, max_items)
    progress_batches = len(loader)
    if progress_total is not None and max_items > 0:
        batch_size = int(getattr(loader, "batch_size", 0) or int(args.batch_size))
        progress_batches = int(math.ceil(progress_total / max(1, batch_size)))
    with torch.no_grad():
        progress = tqdm(
            loader,
            total=progress_batches,
            desc=f"Building token cache ({args.split})",
            unit="batch",
            dynamic_ncols=True,
        )
        for batch in progress:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            remaining = max_items - seen if max_items > 0 else x.size(0)
            if max_items > 0 and remaining <= 0:
                break
            if max_items > 0 and x.size(0) > remaining:
                x = x[:remaining]
            class_labels = _batch_class_labels(batch, int(x.size(0)))
            if class_labels is not None:
                all_class_labels.append(class_labels)
            x = x.to(device, non_blocking=True)
            if quantized_cache and not adaptive_quantized_cache:
                tokens, current_latent_hw = model.encode_to_tokens(
                    x,
                    coeff_vocab_size=int(args.coeff_vocab_size),
                    coeff_max=float(coeff_max),
                    coeff_quantization=coeff_quantization,
                    coeff_mu=float(args.coeff_mu),
                )
                tokens = sort_token_pairs(tokens)
                coeffs = None
            else:
                tokens, coeffs, current_latent_hw = model.encode_to_atoms_and_coeffs(x)
                tokens, coeffs = sort_sparse_pairs(tokens, coeffs)
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
            item_suffix = f"{seen}"
            if progress_total is not None:
                item_suffix = f"{seen}/{progress_total}"
            progress.set_postfix_str(f"images={item_suffix}", refresh=False)
            if max_items > 0 and seen >= max_items:
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
    class_labels = _finalize_class_labels(all_class_labels, int(tokens_flat.size(0)))

    if quantized_cache and adaptive_quantized_cache:
        if coeffs_flat is None:
            raise RuntimeError("Adaptive quantized cache expected collected sparse coefficients.")
        if coeff_max is None:
            coeff_max = _auto_coeff_absmax(
                coeffs_flat,
                percentile=float(args.coeff_calibration_percentile),
            )
        coeff_bin_values = _adaptive_coeff_bin_values(
            coeffs_flat,
            coeff_vocab_size=int(args.coeff_vocab_size),
            coeff_max=float(coeff_max),
            coeff_quantization=coeff_quantization,
            coeff_mu=float(args.coeff_mu),
        )
        token_h, token_w, token_depth = token_shape
        atom_grid = tokens_flat.view(tokens_flat.size(0), int(token_h), int(token_w), int(token_depth))
        coeff_grid = coeffs_flat.view(coeffs_flat.size(0), int(token_h), int(token_w), int(token_depth))
        token_grid = _interleave_atom_coeff_tokens(
            atom_grid,
            coeff_grid,
            coeff_bin_values=coeff_bin_values,
            atom_vocab_size=int(model.bottleneck.num_embeddings),
        )
        tokens_flat = token_grid.view(token_grid.size(0), -1).to(torch.int32).cpu()
        coeffs_flat = None
        token_shape = tuple(int(dim) for dim in token_grid.shape[1:])
    elif (not quantized_cache) and coeff_max <= 0.0:
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
        # Renamed in May 2026 (A3); see src/models/bottleneck.py.
        "variational_coeff_target_std": float(getattr(model.hparams, "variational_coeff_target_std", 0.25)),
        "variational_coeff_min_std": float(getattr(model.hparams, "variational_coeff_min_std", 0.01)),
        "support_order": "atom_id",
    }
    if quantized_cache:
        meta.update(
            {
                "coeff_vocab_size": int(args.coeff_vocab_size),
                "n_bins": int(args.coeff_vocab_size),
                "coeff_bin_values": coeff_bin_values,
                "coeff_quantization": coeff_quantization,
                "coef_quantization": coeff_quantization,
                "coeff_mu": float(args.coeff_mu),
                "coef_mu": float(args.coeff_mu),
                "coeff_calibration_percentile": float(args.coeff_calibration_percentile),
            }
        )
    cache = {
        "tokens_flat": tokens_flat,
        "shape": (int(token_h), int(token_w), int(token_depth)),
        "meta": meta,
    }
    if coeffs_flat is not None:
        cache["coeffs_flat"] = coeffs_flat
    if class_labels is not None:
        _attach_class_label_metadata(
            cache,
            class_labels,
            dataset_name=str(args.dataset),
            dataset=getattr(loader, "dataset", None),
        )

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
