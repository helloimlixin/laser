"""Extract a maintained stage-2 token cache from a trained stage-1 model."""

import argparse
import math
from pathlib import Path
import sys
from typing import Iterable, Tuple

if sys.version_info < (3, 10):
    raise SystemExit(
        "ERROR: cache.py requires Python >= 3.10. "
        "Set PYTHON_BIN to a supported environment or run through scripts/run.sh."
    )

import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from src.cache_sort import sort_sparse_pairs, sort_token_pairs
from src.checkpoint_io import extract_hparams, extract_state_dict, load_lightning_module, load_torch_payload
from src.data.config import DataConfig
from src.models.laser import LASER
from src.models.vqvae import VQVAE
from src.stage2_paths import default_token_cache_path, infer_latest_stage1_checkpoint
from src.stage1_setup import build_stage1_datamodule, data_config_from_overrides


def _is_vqvae_model(model) -> bool:
    return isinstance(model, VQVAE) or (
        hasattr(model, "vector_quantizer") and not hasattr(model, "bottleneck")
    )


def _build_datamodule(args: argparse.Namespace):
    audio_keys = (
        "sample_rate",
        "audio_representation",
        "audio_num_samples",
        "stft_n_fft",
        "stft_hop_length",
        "stft_win_length",
        "stft_power",
        "stft_log_offset",
        "audio_dc_remove",
        "audio_peak_normalize",
        "audio_target_peak",
        "audio_rms_normalize",
        "audio_target_rms",
        "audio_max_gain",
        "audio_min_crop_rms",
        "audio_crop_attempts",
        "audio_fade_samples",
    )
    config = data_config_from_overrides(
        args.dataset,
        data_dir=(None if args.data_dir is None else str(args.data_dir)),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        image_size=args.image_size,
        seed=int(args.seed),
        mean=(None if args.mean is None else tuple(args.mean)),
        std=(None if args.std is None else tuple(args.std)),
        augment=False,
        **{key: getattr(args, key) for key in audio_keys},
    )
    dm = build_stage1_datamodule(config)
    dm.prepare_data()
    dm.setup("fit")
    return dm, config


def infer_stage1_model_type(
    *,
    payload=None,
    hparams=None,
    state_dict=None,
    explicit: str = "auto",
) -> str:
    requested = str(explicit or "auto").strip().lower()
    if requested in {"laser", "vqvae"}:
        return requested

    if hparams is None and payload is not None:
        hparams = extract_hparams(payload)
    if state_dict is None and payload is not None:
        extracted = extract_state_dict(payload)
        state_dict = extracted if isinstance(extracted, dict) else {}
    hparams = dict(hparams or {})
    state_dict = dict(state_dict or {})

    if "sparsity_level" in hparams or any(key.startswith("bottleneck.") for key in state_dict):
        return "laser"
    if "decay" in hparams or any(key.startswith("vector_quantizer.") for key in state_dict):
        return "vqvae"
    raise ValueError("Could not infer stage-1 model type from checkpoint metadata. Pass --model-type explicitly.")


def _infer_latest_stage1_checkpoint(output_root, model_type: str):
    model_type = str(model_type).strip().lower()
    if model_type in {"laser", "vqvae"}:
        return infer_latest_stage1_checkpoint(output_root=output_root, model_type=model_type)

    candidates = []
    for candidate_type in ("laser", "vqvae"):
        path = infer_latest_stage1_checkpoint(output_root=output_root, model_type=candidate_type)
        if path is not None:
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


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
    model,
    loader: Iterable,
    *,
    device: torch.device,
    max_items: int,
) -> float:
    if not isinstance(model, LASER):
        return 0.0
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


def _batch_audio_meta(batch) -> dict | None:
    if not isinstance(batch, (tuple, list)) or len(batch) < 3:
        return None
    meta = batch[-1]
    if not isinstance(meta, dict):
        return None
    required = {"path", "crop_mode", "crop_offset", "source_num_samples", "spec_min", "spec_max", "spec_shape"}
    if not required.issubset(meta.keys()):
        return None
    return meta


def _append_audio_meta(store: dict | None, meta: dict | None, keep: int) -> dict | None:
    if meta is None or keep <= 0:
        return store
    if store is None:
        store = {
            "path": [],
            "crop_mode": [],
            "crop_offset": [],
            "source_num_samples": [],
            "spec_min": [],
            "spec_max": [],
            "spec_shape": [],
        }
    store["path"].extend(str(v) for v in list(meta["path"])[:keep])
    for key in ("crop_mode", "crop_offset", "source_num_samples", "spec_min", "spec_max", "spec_shape"):
        value = meta[key]
        tensor = value.detach().cpu() if torch.is_tensor(value) else torch.as_tensor(value)
        store[key].append(tensor[:keep].clone())
    return store


def _finalize_audio_meta(store: dict | None) -> dict | None:
    if store is None:
        return None
    out = {"path": list(store["path"])}
    for key in ("crop_mode", "crop_offset", "source_num_samples", "spec_min", "spec_max", "spec_shape"):
        parts = store.get(key, [])
        if not parts:
            return None
        out[key] = torch.cat(parts, dim=0)
    return out


def _extract_cache(
    model,
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
    if _is_vqvae_model(model):
        all_tokens = []
        token_shape = None
        latent_hw = None
        audio_meta_chunks = None
        seen = 0

        model.eval()
        with torch.inference_mode():
            for batch in loader:
                images = batch[0] if isinstance(batch, (tuple, list)) else batch
                keep = int(images.size(0))
                if max_items > 0:
                    keep = min(keep, max_items - seen)
                    if keep <= 0:
                        break
                    images = images[:keep]
                audio_meta_chunks = _append_audio_meta(audio_meta_chunks, _batch_audio_meta(batch), keep)
                images = images.to(device=device, non_blocking=True)

                indices, h_z, w_z = model.encode_to_indices(images)
                tokens = indices.view(indices.size(0), h_z, w_z, 1).to(torch.int32).cpu()
                all_tokens.append(tokens.view(tokens.size(0), -1))
                current_shape = (int(h_z), int(w_z), 1)
                current_latent_hw = (int(h_z), int(w_z))

                if token_shape is None:
                    token_shape = current_shape
                    latent_hw = current_latent_hw
                else:
                    if current_shape != token_shape:
                        raise RuntimeError(
                            f"Token grid shape changed across batches: {token_shape} vs {current_shape}"
                        )
                    if current_latent_hw != latent_hw:
                        raise RuntimeError(
                            f"Latent shape changed across batches: {latent_hw} vs {current_latent_hw}"
                        )

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
        audio_meta = _finalize_audio_meta(audio_meta_chunks)
        if audio_meta is not None:
            result["audio_meta"] = audio_meta
        return result

    real_valued = int(coeff_vocab_size) <= 0
    all_tokens = []
    all_coeffs = [] if real_valued else None
    token_shape = None
    latent_hw = None
    audio_meta_chunks = None
    seen = 0

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            keep = int(images.size(0))
            if max_items > 0:
                keep = min(keep, max_items - seen)
                if keep <= 0:
                    break
                images = images[:keep]
            audio_meta_chunks = _append_audio_meta(audio_meta_chunks, _batch_audio_meta(batch), keep)
            images = images.to(device=device, non_blocking=True)

            if real_valued:
                support, values, batch_latent_hw = model.encode_to_atoms_and_coeffs(images)
                support, values = sort_sparse_pairs(support, values)
                # support: [B, H, W, D] atom ids, values: [B, H, W, D] coeffs
                flat_atoms = support.view(support.size(0), -1).to(torch.int32).cpu()
                flat_coeffs = values.view(values.size(0), -1).to(torch.float32).cpu()
                if not torch.isfinite(flat_coeffs).all():
                    raise RuntimeError("Sparse coefficient cache contains non-finite values")
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
                tokens = sort_token_pairs(tokens)
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
    audio_meta = _finalize_audio_meta(audio_meta_chunks)
    if audio_meta is not None:
        result["audio_meta"] = audio_meta
    return result


def _token_cache_meta(
    *,
    args: argparse.Namespace,
    model,
    config: DataConfig,
    num_items: int,
    latent_hw: Tuple[int, int],
) -> dict:
    common_meta = {
        "mean": tuple(float(v) for v in getattr(config, "mean", ()) or ()),
        "std": tuple(float(v) for v in getattr(config, "std", ()) or ()),
    }
    if str(config.dataset).strip().lower() in {"vctk", "maestro"}:
        common_meta.update(
            {
                "sample_rate": int(config.sample_rate),
                "audio_num_samples": int(config.audio_num_samples),
                "stft_n_fft": int(config.stft_n_fft),
                "stft_hop_length": int(config.stft_hop_length),
                "stft_win_length": int(config.stft_win_length or config.stft_n_fft),
                "stft_power": float(config.stft_power),
                "stft_log_offset": float(config.stft_log_offset),
                "audio_representation": str(getattr(config, "audio_representation", "spectrogram")),
                "audio_dc_remove": bool(getattr(config, "audio_dc_remove", False)),
                "audio_peak_normalize": bool(getattr(config, "audio_peak_normalize", False)),
                "audio_target_peak": float(getattr(config, "audio_target_peak", 0.95)),
                "audio_rms_normalize": bool(getattr(config, "audio_rms_normalize", False)),
                "audio_target_rms": float(getattr(config, "audio_target_rms", 0.12)),
                "audio_max_gain": float(getattr(config, "audio_max_gain", 8.0)),
                "audio_min_crop_rms": float(getattr(config, "audio_min_crop_rms", 0.0)),
                "audio_crop_attempts": int(getattr(config, "audio_crop_attempts", 1)),
                "audio_fade_samples": int(getattr(config, "audio_fade_samples", 0)),
            }
        )
    if _is_vqvae_model(model):
        num_embeddings = int(model.vector_quantizer.num_embeddings)
        return {
            "version": 1,
            "dataset": str(config.dataset),
            "split": str(args.split),
            "image_size": int(config.image_size if isinstance(config.image_size, int) else config.image_size[0]),
            "seed": int(args.seed),
            "num_items": int(num_items),
            "num_atoms": num_embeddings,
            "num_embeddings": num_embeddings,
            "embedding_dim": int(model.vector_quantizer.embedding_dim),
            "latent_hw": (int(latent_hw[0]), int(latent_hw[1])),
            "stage1_checkpoint": str(Path(args.stage1_checkpoint).expanduser().resolve()),
            "stage1_model_type": "vqvae",
            "support_order": "code_index",
            "quantize_sparse_coeffs": True,
            "n_bins": 1,
            "coeff_vocab_size": 1,
            "coeff_bin_values": torch.zeros(1, dtype=torch.float32),
            **common_meta,
        }

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
    backbone = str(getattr(model, "backbone", getattr(model.hparams, "backbone", "simple")))
    attn_resolutions = tuple(
        int(v) for v in getattr(model, "attn_resolutions", getattr(model.hparams, "attn_resolutions", ()))
    )
    channel_multipliers = tuple(
        int(v) for v in getattr(model, "channel_multipliers", getattr(model.hparams, "channel_multipliers", ()))
    )
    backbone_latent_channels = getattr(
        model,
        "backbone_latent_channels",
        getattr(model.hparams, "backbone_latent_channels", None),
    )
    if backbone_latent_channels in (None, "", ()):
        backbone_latent_channels = getattr(model.bottleneck, "embedding_dim", 0)
    backbone_latent_channels = int(backbone_latent_channels)
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
        "backbone": backbone,
        "num_downsamples": int(getattr(model, "num_downsamples", getattr(model.hparams, "num_downsamples", 2))),
        "attn_resolutions": attn_resolutions,
        "dropout": float(getattr(model, "dropout", getattr(model.hparams, "dropout", 0.0))),
        "channel_multipliers": channel_multipliers,
        "backbone_latent_channels": backbone_latent_channels,
        "max_ch_mult": int(getattr(model, "max_ch_mult", getattr(model.hparams, "max_ch_mult", 2))),
        "decoder_extra_residual_layers": int(
            getattr(
                model,
                "decoder_extra_residual_layers",
                getattr(model.hparams, "decoder_extra_residual_layers", 1),
            )
        ),
        "use_mid_attention": bool(
            getattr(model, "use_mid_attention", getattr(model.hparams, "use_mid_attention", False))
        ),
        "stage1_checkpoint": str(Path(args.stage1_checkpoint).expanduser().resolve()),
        "stage1_model_type": "laser",
        "support_order": "atom_id",
        **common_meta,
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
    parser = argparse.ArgumentParser(description="Extract a maintained stage-2 token cache from a stage-1 checkpoint.")
    parser.add_argument("--stage1-checkpoint", type=Path, default=None)
    parser.add_argument("--model-type", type=str, default="auto", choices=["auto", "laser", "vqvae"])
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "cifar10",
            "celeba",
            "celebahq",
            "coco",
            "ffhq",
            "imagenette2",
            "stl10",
            "vctk",
            "maestro",
        ],
    )
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
    parser.add_argument("--mean", type=float, nargs="+", default=None)
    parser.add_argument("--std", type=float, nargs="+", default=None)
    parser.add_argument("--sample-rate", type=int, default=None)
    parser.add_argument("--audio-representation", type=str, default=None, choices=["spectrogram", "waveform"])
    parser.add_argument("--audio-num-samples", type=int, default=None)
    parser.add_argument("--stft-n-fft", type=int, default=None)
    parser.add_argument("--stft-hop-length", type=int, default=None)
    parser.add_argument("--stft-win-length", type=int, default=None)
    parser.add_argument("--stft-power", type=float, default=None)
    parser.add_argument("--stft-log-offset", type=float, default=None)
    parser.add_argument("--audio-dc-remove", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--audio-peak-normalize", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--audio-target-peak", type=float, default=None)
    parser.add_argument("--audio-rms-normalize", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--audio-target-rms", type=float, default=None)
    parser.add_argument("--audio-max-gain", type=float, default=None)
    parser.add_argument("--audio-min-crop-rms", type=float, default=None)
    parser.add_argument("--audio-crop-attempts", type=int, default=None)
    parser.add_argument("--audio-fade-samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = _parse_args()
    pl.seed_everything(int(args.seed), workers=True)

    if args.stage1_checkpoint is None:
        inferred_stage1 = _infer_latest_stage1_checkpoint(args.output_root, args.model_type)
        if inferred_stage1 is None:
            raise FileNotFoundError(
                "Could not infer a stage-1 checkpoint under "
                f"{(Path(args.output_root).expanduser().resolve() / 'checkpoints')}"
            )
        args.stage1_checkpoint = inferred_stage1
        print(f"Inferred stage1 checkpoint: {args.stage1_checkpoint}")

    payload = load_torch_payload(args.stage1_checkpoint, map_location="cpu")
    state_dict = extract_state_dict(payload)
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Checkpoint at {args.stage1_checkpoint} does not contain a valid state_dict")
    model_type = infer_stage1_model_type(
        payload=payload,
        hparams=extract_hparams(payload),
        state_dict=state_dict,
        explicit=args.model_type,
    )
    print(f"Resolved stage1 model type: {model_type}")

    device = _resolve_device(str(args.device))
    datamodule, config = _build_datamodule(args)
    dataset = _split_dataset(datamodule, args.split)
    loader = _stable_loader(
        dataset,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )

    if model_type == "laser":
        model = load_lightning_module(
            LASER,
            args.stage1_checkpoint,
            map_location="cpu",
            strict=False,
            compute_fid=False,
            perceptual_weight=0.0,
        )
    else:
        if int(args.coeff_bins) != 1:
            print(f"Forcing coeff-bins to 1 for VQ-VAE token caches (got {int(args.coeff_bins)}).")
        args.coeff_bins = 1
        model = load_lightning_module(
            VQVAE,
            args.stage1_checkpoint,
            map_location="cpu",
            strict=False,
            compute_fid=False,
            perceptual_weight=0.0,
        )
    model.eval().to(device)

    coeff_max_arg = str(args.coeff_max).strip().lower()
    if model_type == "vqvae":
        coeff_max = 0.0
    elif coeff_max_arg == "auto":
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
    if "audio_meta" in cache_result:
        payload["audio_meta"] = cache_result["audio_meta"]
    if "coeffs_flat" in cache_result:
        payload["coeffs_flat"] = cache_result["coeffs_flat"]
        coeffs_flat = payload["coeffs_flat"]
        if not torch.isfinite(coeffs_flat).all():
            raise RuntimeError("Refusing to save sparse token cache with non-finite coefficients")
        coeff_abs_max = float(coeffs_flat.abs().max().item()) if coeffs_flat.numel() > 0 else 0.0
        if coeff_abs_max <= 1e-12:
            raise RuntimeError(
                "Refusing to save degenerate sparse token cache: all coefficients are zero"
            )
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
