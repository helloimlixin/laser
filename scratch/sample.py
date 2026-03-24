#!/usr/bin/env python3
"""
Sample images from existing LASER stage-1/stage-2 checkpoints without retraining.

This loader targets the current `proto.py` stage-2 checkpoint formats,
including the spatial-depth prior and the quantized minGPT prior.

Examples:
  python3 sample.py
  python3 sample.py --num_samples 16 --top_k 32
  python3 sample.py --run_dir runs/laser_celeba128_quantized
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = SCRIPT_DIR / "runs" / "laser_celeba128"


def _load_proto_module():
    module_path = SCRIPT_DIR / "proto.py"
    spec = importlib.util.spec_from_file_location("scratch_proto_checkpoint_sampler", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import proto module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _indexed_block_count(keys: Iterable[str], prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.")
    indices = {int(match.group(1)) for key in keys if (match := pattern.match(key))}
    return len(indices)


def _load_torch_payload(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _first_existing_path(*candidates: Path) -> Optional[Path]:
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    return None


def _infer_stage1_config(state_dict: Dict[str, torch.Tensor], token_cache: dict) -> dict:
    metadata = token_cache.get("meta") or token_cache.get("metadata") or {}
    dictionary_shape = tuple(state_dict["bottleneck.dictionary"].shape)
    quantize_sparse_coeffs = metadata.get("quantize_sparse_coeffs")
    if quantize_sparse_coeffs is None:
        quantize_sparse_coeffs = token_cache.get("coeffs_flat") is None

    coef_quantization = "uniform"
    coef_mu = 0.0
    coef_mu_invlog1p = state_dict.get("bottleneck.coef_mu_invlog1p")
    if coef_mu_invlog1p is not None:
        mu_invlog1p = float(coef_mu_invlog1p.item())
        if abs(mu_invlog1p - 1.0) > 1e-6:
            coef_quantization = "mu_law"
            coef_mu = float(math.expm1(1.0 / mu_invlog1p))

    num_hiddens = int(state_dict["encoder.conv_in.weight"].shape[0])
    num_residual_layers = max(1, _indexed_block_count(state_dict.keys(), "encoder.down.0.block"))
    encoder_norm_channels = int(state_dict["encoder.norm_out.weight"].shape[0])
    decoder_blocks_per_level = max(1, _indexed_block_count(state_dict.keys(), "decoder.up.0.block"))
    inferred_use_mid_attention = (
        "encoder.mid.attn_1.q.weight" in state_dict or "decoder.mid.attn_1.q.weight" in state_dict
    )

    return {
        "in_channels": int(state_dict["encoder.conv_in.weight"].shape[1]),
        "num_hiddens": num_hiddens,
        "num_downsamples": max(0, _indexed_block_count(state_dict.keys(), "encoder.down") - 1),
        "num_residual_layers": num_residual_layers,
        "resolution": int(metadata.get("image_size", 128)),
        "max_ch_mult": int(metadata.get("max_ch_mult", max(1, encoder_norm_channels // max(1, num_hiddens)))),
        "decoder_extra_residual_layers": int(
            metadata.get("decoder_extra_residual_layers", max(0, decoder_blocks_per_level - num_residual_layers))
        ),
        "use_mid_attention": bool(metadata.get("use_mid_attention", inferred_use_mid_attention)),
        "embedding_dim": int(state_dict["encoder.conv_out.weight"].shape[0]),
        "num_embeddings": int(dictionary_shape[1]),
        "sparsity_level": int(metadata.get("sparsity_level", token_cache["shape"][2])),
        "commitment_cost": 0.25,
        "n_bins": int(state_dict["bottleneck.coef_bin_centers"].shape[0]),
        "coef_max": float(state_dict["bottleneck.coef_bin_centers"].abs().max().item()),
        "coef_quantization": coef_quantization,
        "coef_mu": coef_mu,
        "out_tanh": True,
        "quantize_sparse_coeffs": bool(quantize_sparse_coeffs),
        "patch_based": bool(metadata.get("patch_based", False)),
        "patch_size": int(metadata.get("patch_size", 8)),
        "patch_stride": int(metadata.get("patch_stride", 4)),
        "patch_reconstruction": str(metadata.get("patch_reconstruction", "center_crop")),
        "variational_coeffs": bool(
            metadata.get(
                "variational_coeffs",
                "bottleneck.coeff_variational_atom_emb.weight" in state_dict,
            )
        ),
        "variational_coeff_kl_weight": float(metadata.get("variational_coeff_kl_weight", 0.0)),
        "variational_coeff_prior_std": float(metadata.get("variational_coeff_prior_std", 0.25)),
        "variational_coeff_min_std": float(metadata.get("variational_coeff_min_std", 0.01)),
    }


def _infer_stage2_arch(
    state_dict: Dict[str, torch.Tensor],
    *,
    tf_heads: int,
    tf_dropout: float,
) -> dict:
    if tf_heads <= 0:
        raise ValueError(f"tf_heads must be positive, got {tf_heads}")
    if "token_emb.weight" not in state_dict:
        raise ValueError("Stage-2 checkpoint is missing token_emb.weight.")

    n_spatial_layers = _indexed_block_count(state_dict.keys(), "spatial_blocks")
    n_depth_layers = _indexed_block_count(state_dict.keys(), "depth_blocks")
    if n_spatial_layers > 0 or n_depth_layers > 0:
        vocab_size, d_model = state_dict["token_emb.weight"].shape
        if d_model % tf_heads != 0:
            raise ValueError(f"d_model={d_model} is not divisible by tf_heads={tf_heads}")

        d_ff = None
        if n_spatial_layers > 0:
            d_ff = int(state_dict["spatial_blocks.0.ffn.0.weight"].shape[0])
        elif n_depth_layers > 0:
            d_ff = int(state_dict["depth_blocks.0.ffn.0.weight"].shape[0])
        if d_ff is None:
            raise ValueError("Could not infer transformer feed-forward width from the checkpoint.")

        autoregressive_coeffs_flag = state_dict.get("_autoregressive_coeffs_flag")
        if autoregressive_coeffs_flag is None:
            autoregressive_coeffs = not any(key.startswith("coeff_token_head.") for key in state_dict.keys())
        else:
            autoregressive_coeffs = bool(int(torch.as_tensor(autoregressive_coeffs_flag).item()))

        return {
            "stage2_arch": "spatial_depth",
            "vocab_size": int(vocab_size),
            "d_model": int(d_model),
            "n_layers": int(n_spatial_layers),
            "n_depth_layers": int(n_depth_layers),
            "d_ff": int(d_ff),
            "dropout": float(tf_dropout),
            "n_heads": int(tf_heads),
            "n_global_spatial_tokens": int(state_dict.get("global_spatial_tokens", torch.empty(1, 0, 1)).shape[1]),
            "real_valued_coeffs": any(key.startswith("coeff_head.") for key in state_dict.keys()),
            "gaussian_coeffs": any(key.startswith("coeff_logvar_head.") for key in state_dict.keys()),
            "autoregressive_coeffs": bool(autoregressive_coeffs),
        }

    n_layers = _indexed_block_count(state_dict.keys(), "blocks")
    if n_layers <= 0 or "pos_emb" not in state_dict or "token_head.weight" not in state_dict:
        raise ValueError("Unsupported stage-2 checkpoint format.")
    d_model = int(state_dict["token_emb.weight"].shape[1])
    if d_model % tf_heads != 0:
        raise ValueError(f"d_model={d_model} is not divisible by tf_heads={tf_heads}")
    d_ff = int(state_dict["blocks.0.ffn.0.weight"].shape[0])
    vocab_size = int(state_dict["token_head.weight"].shape[0])
    atom_vocab_size = int(torch.as_tensor(state_dict["_atom_vocab_size_tensor"]).item())
    coeff_vocab_size = int(torch.as_tensor(state_dict["_coeff_vocab_size_tensor"]).item())
    if atom_vocab_size + coeff_vocab_size != vocab_size:
        raise ValueError(
            "MinGPT checkpoint vocab sizes are inconsistent: "
            f"atom_vocab_size={atom_vocab_size}, coeff_vocab_size={coeff_vocab_size}, vocab_size={vocab_size}"
        )
    return {
        "stage2_arch": "mingpt",
        "vocab_size": int(vocab_size),
        "d_model": int(d_model),
        "n_layers": int(n_layers),
        "d_ff": int(d_ff),
        "dropout": float(tf_dropout),
        "n_heads": int(tf_heads),
        "n_global_spatial_tokens": 0,
        "real_valued_coeffs": False,
        "gaussian_coeffs": False,
        "autoregressive_coeffs": True,
    }


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _resolve_nrow(num_samples: int, nrow: Optional[int]) -> int:
    if nrow is not None and int(nrow) > 0:
        return int(nrow)
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    root = math.isqrt(num_samples)
    if root * root == num_samples:
        return max(1, root)
    return max(1, math.ceil(math.sqrt(num_samples)))


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _generate_samples(
    proto_mod,
    *,
    laser,
    transformer,
    H: int,
    W: int,
    D: int,
    num_samples: int,
    atom_temperature: float,
    atom_top_k: Optional[int],
    coeff_temperature: Optional[float],
    coeff_sample_mode: str,
    output_image_size: Optional[int],
    seed: int,
    prompt_tokens: Optional[torch.Tensor] = None,
    prompt_coeffs: Optional[torch.Tensor] = None,
    prompt_mask: Optional[torch.Tensor] = None,
):
    _seed_everything(seed)
    gen_out = transformer.generate(
        batch_size=num_samples,
        temperature=atom_temperature,
        top_k=atom_top_k,
        coeff_temperature=coeff_temperature,
        coeff_sample_mode=coeff_sample_mode,
        show_progress=True,
        progress_desc="[Sample] generating sparse codes",
        prompt_tokens=prompt_tokens,
        prompt_coeffs=prompt_coeffs,
        prompt_mask=prompt_mask,
    )
    laser_device = next(laser.parameters()).device
    if transformer.real_valued_coeffs:
        atom_ids, coeffs = gen_out
        atom_grid = atom_ids.view(-1, H, W, D)
        coeff_grid = coeffs.view(-1, H, W, D)
        imgs = laser.decode_from_atoms_and_coeffs(
            atom_grid.to(laser_device),
            coeff_grid.to(laser_device),
        )
        tokens_flat = atom_ids.reshape(atom_ids.size(0), -1)
        coeffs_flat = coeffs.reshape(coeffs.size(0), -1)
    else:
        tokens = gen_out.view(-1, H, W, D)
        imgs = laser.decode_from_tokens(tokens.to(laser_device))
        tokens_flat = tokens.reshape(tokens.size(0), -1)
        coeffs_flat = None

    if output_image_size is not None and int(output_image_size) > 0:
        output_size = int(output_image_size)
        if imgs.size(-2) != output_size or imgs.size(-1) != output_size:
            imgs = F.interpolate(imgs, size=(output_size, output_size), mode="bilinear", align_corners=False)
    return {
        "images": imgs.detach().cpu(),
        "tokens_flat": tokens_flat.detach().cpu(),
        "coeffs_flat": (None if coeffs_flat is None else coeffs_flat.detach().cpu()),
    }


def _index_sample_payload(sample: dict, selected_idx: torch.Tensor) -> dict:
    selected_idx = selected_idx.to(device="cpu", dtype=torch.long)
    result = {
        "images": sample["images"].index_select(0, selected_idx),
        "tokens_flat": sample["tokens_flat"].index_select(0, selected_idx),
        "coeffs_flat": None,
    }
    if sample["coeffs_flat"] is not None:
        result["coeffs_flat"] = sample["coeffs_flat"].index_select(0, selected_idx)
    return result


def _decode_sparse_code_payload(
    laser,
    *,
    tokens_flat: torch.Tensor,
    coeffs_flat: Optional[torch.Tensor],
    H: int,
    W: int,
    D: int,
    output_image_size: Optional[int],
) -> dict:
    laser_device = next(laser.parameters()).device
    token_grid = tokens_flat.view(-1, H, W, D).to(laser_device, dtype=torch.long)
    if coeffs_flat is not None:
        coeff_grid = coeffs_flat.view(-1, H, W, D).to(laser_device, dtype=torch.float32)
        images = laser.decode_from_atoms_and_coeffs(token_grid, coeff_grid)
    else:
        images = laser.decode_from_tokens(token_grid)
    if output_image_size is not None and int(output_image_size) > 0:
        output_size = int(output_image_size)
        if images.size(-2) != output_size or images.size(-1) != output_size:
            images = F.interpolate(images, size=(output_size, output_size), mode="bilinear", align_corners=False)
    return {
        "images": images.detach().cpu(),
        "tokens_flat": tokens_flat.detach().cpu(),
        "coeffs_flat": (None if coeffs_flat is None else coeffs_flat.detach().cpu()),
    }


def _build_prompt_payload_from_cache(
    token_cache: dict,
    *,
    batch_size: int,
    H: int,
    W: int,
    D: int,
    prompt_spatial_steps: int,
    prompt_offset: int,
) -> Optional[dict]:
    prompt_spatial_steps = int(prompt_spatial_steps)
    if prompt_spatial_steps <= 0:
        return None
    if prompt_offset < 0:
        raise ValueError("prompt_offset must be >= 0.")

    T = int(H) * int(W)
    if prompt_spatial_steps > T:
        raise ValueError(f"prompt_spatial_steps must be <= H*W ({T}), got {prompt_spatial_steps}.")

    tokens_flat = token_cache.get("tokens_flat")
    if tokens_flat is None:
        raise ValueError("Token cache is missing tokens_flat, so prompt-conditioned sampling is unavailable.")
    if tokens_flat.ndim != 2:
        raise ValueError(f"Expected token_cache['tokens_flat'] to have rank 2, got {tuple(tokens_flat.shape)}")
    expected_width = T * int(D)
    if int(tokens_flat.shape[1]) != expected_width:
        raise ValueError(
            f"Expected token_cache['tokens_flat'] width {expected_width} for shape {(H, W, D)}, "
            f"got {int(tokens_flat.shape[1])}."
        )
    total_items = int(tokens_flat.shape[0])
    if total_items <= 0:
        raise ValueError("Token cache has no stored sparse codes.")

    source_idx = (torch.arange(int(batch_size), dtype=torch.long) + int(prompt_offset)) % total_items
    prompt_tokens_flat = tokens_flat.index_select(0, source_idx)
    prompt_tokens = prompt_tokens_flat.view(int(batch_size), T, int(D))

    coeffs_flat = token_cache.get("coeffs_flat")
    prompt_coeffs_flat = None
    prompt_coeffs = None
    if coeffs_flat is not None:
        prompt_coeffs_flat = coeffs_flat.index_select(0, source_idx)
        prompt_coeffs = prompt_coeffs_flat.view(int(batch_size), T, int(D))

    prompt_mask = torch.zeros(int(batch_size), T, int(D), dtype=torch.bool)
    prompt_mask[:, :prompt_spatial_steps, :] = True
    return {
        "source_indices": [int(x) for x in source_idx.tolist()],
        "prompt_spatial_steps": int(prompt_spatial_steps),
        "tokens_flat": prompt_tokens_flat,
        "coeffs_flat": prompt_coeffs_flat,
        "prompt_tokens": prompt_tokens,
        "prompt_coeffs": prompt_coeffs,
        "prompt_mask": prompt_mask,
    }


def _compute_candidate_pool_records(
    proto_mod,
    *,
    images: torch.Tensor,
    reference_stats: Optional[dict],
    brightness_weight: float,
    overbright_weight: float,
    reject_dark_z: float,
    reject_bright_z: float,
    selected_idx: Optional[torch.Tensor],
) -> list[dict]:
    feats = proto_mod._sample_quality_features(images)
    quality = torch.zeros(images.size(0), dtype=feats.dtype, device=feats.device)
    brightness_penalty = torch.zeros_like(quality)
    overbright_penalty = torch.zeros_like(quality)
    passes_dark_reject = torch.ones(images.size(0), dtype=torch.bool, device=feats.device)
    passes_bright_reject = torch.ones(images.size(0), dtype=torch.bool, device=feats.device)

    if reference_stats is not None:
        ref_mean = reference_stats["mean"].to(device=feats.device, dtype=feats.dtype)
        ref_std = reference_stats["std"].to(device=feats.device, dtype=feats.dtype)
        quality = (((feats - ref_mean) / ref_std) ** 2).mean(dim=1)
        brightness_penalty = proto_mod._sample_feature_low_brightness_penalty(
            feats,
            ref_mean=ref_mean,
            ref_std=ref_std,
        )
        overbright_penalty = proto_mod._sample_feature_high_brightness_penalty(
            feats,
            ref_mean=ref_mean,
            ref_std=ref_std,
        )
        if float(reject_dark_z) > 0.0:
            passes_dark_reject = proto_mod._sample_feature_dark_rejection_mask(
                feats,
                ref_mean=ref_mean,
                ref_std=ref_std,
                reject_dark_z=float(reject_dark_z),
            )
        if float(reject_bright_z) > 0.0:
            passes_bright_reject = proto_mod._sample_feature_bright_rejection_mask(
                feats,
                ref_mean=ref_mean,
                ref_std=ref_std,
                reject_bright_z=float(reject_bright_z),
            )

    brightness = feats[:, proto_mod._SAMPLE_FEAT_BRIGHTNESS_SLICE].squeeze(1)
    center_brightness = proto_mod._sample_feature_center_brightness(feats).squeeze(1)
    luma_p10 = feats[:, proto_mod._SAMPLE_FEAT_LUMA_P10_SLICE].squeeze(1)
    luma_p25 = feats[:, proto_mod._SAMPLE_FEAT_LUMA_P25_SLICE].squeeze(1)
    dark_frac20 = feats[:, proto_mod._SAMPLE_FEAT_DARK_FRAC20_SLICE].squeeze(1)
    dark_frac30 = feats[:, proto_mod._SAMPLE_FEAT_DARK_FRAC30_SLICE].squeeze(1)
    center_luma_p10 = feats[:, proto_mod._SAMPLE_FEAT_CENTER_LUMA_P10_SLICE].squeeze(1)
    center_dark_frac20 = feats[:, proto_mod._SAMPLE_FEAT_CENTER_DARK_FRAC20_SLICE].squeeze(1)
    center_dark_frac30 = feats[:, proto_mod._SAMPLE_FEAT_CENTER_DARK_FRAC30_SLICE].squeeze(1)
    combined_score = (
        quality
        + float(brightness_weight) * brightness_penalty
        + float(overbright_weight) * overbright_penalty
    )

    selected_rank = {}
    if selected_idx is not None:
        for rank, idx in enumerate(selected_idx.to(device="cpu", dtype=torch.long).tolist()):
            selected_rank[int(idx)] = int(rank)

    records = []
    for idx in range(int(images.size(0))):
        records.append(
            {
                "grid_index": int(idx),
                "candidate_index": int(idx),
                "selected": int(idx) in selected_rank,
                "selected_rank": selected_rank.get(int(idx)),
                "quality_score": float(quality[idx].item()),
                "brightness_penalty": float(brightness_penalty[idx].item()),
                "low_brightness_penalty": float(brightness_penalty[idx].item()),
                "overbright_penalty": float(overbright_penalty[idx].item()),
                "selection_score": float(combined_score[idx].item()),
                "passes_dark_reject": bool(passes_dark_reject[idx].item()),
                "passes_bright_reject": bool(passes_bright_reject[idx].item()),
                "brightness": float(brightness[idx].item()),
                "center_brightness": float(center_brightness[idx].item()),
                "luma_p10": float(luma_p10[idx].item()),
                "luma_p25": float(luma_p25[idx].item()),
                "dark_frac20": float(dark_frac20[idx].item()),
                "dark_frac30": float(dark_frac30[idx].item()),
                "center_luma_p10": float(center_luma_p10[idx].item()),
                "center_dark_frac20": float(center_dark_frac20[idx].item()),
                "center_dark_frac30": float(center_dark_frac30[idx].item()),
            }
        )
    return records


def _save_indexed_image_grid(
    proto_mod,
    images: torch.Tensor,
    path: Path,
    *,
    nrow: int,
    index_offset: int = 0,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    grid = proto_mod._make_image_grid(images, nrow=nrow)
    image = grid.permute(1, 2, 0).mul(255).clamp(0, 255).byte().numpy()
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    font = ImageFont.load_default()

    count = int(images.size(0))
    if count <= 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        pil.save(path)
        return

    tile_h = int(images.size(-2))
    tile_w = int(images.size(-1))
    padding = 2
    xmaps = min(int(nrow), count)
    label_fill = (255, 255, 0)
    label_bg = (0, 0, 0)

    for idx in range(count):
        row = idx // xmaps
        col = idx % xmaps
        x0 = padding + col * (tile_w + padding)
        y0 = padding + row * (tile_h + padding)
        label = str(int(index_offset) + idx)
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = int(bbox[2] - bbox[0])
        text_h = int(bbox[3] - bbox[1])
        rect = (x0 + 1, y0 + 1, x0 + text_w + 7, y0 + text_h + 5)
        draw.rectangle(rect, fill=label_bg)
        draw.text((x0 + 4, y0 + 2), label, fill=label_fill, font=font)

    path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(path)


def _log_wandb_file_image(
    proto_mod,
    run: Optional[object],
    key: str,
    path: Path,
    *,
    caption: Optional[str] = None,
) -> None:
    wandb_mod = getattr(proto_mod, "wandb", None)
    disable_reason = getattr(proto_mod, "_WANDB_DISABLE_REASON", None)
    if run is None or wandb_mod is None or disable_reason is not None:
        return
    payload = {key: wandb_mod.Image(str(path), caption=caption)}
    try:
        run.log(payload, step=proto_mod._next_wandb_log_step())
    except Exception as exc:
        proto_mod._disable_wandb_logging(run, f"image log {key}", exc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Run directory containing stage1/ and stage2/ checkpoints.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Where to save generated sample grids. Defaults to <run_dir>/stage2_checkpoint_samples.",
    )
    parser.add_argument("--stage1_checkpoint", type=Path, default=None, help="Override stage-1 checkpoint path.")
    parser.add_argument("--stage2_checkpoint", type=Path, default=None, help="Override stage-2 checkpoint path.")
    parser.add_argument("--token_cache", type=Path, default=None, help="Override token cache path.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument(
        "--prompt_spatial_steps",
        type=int,
        default=0,
        help="Clamp the first N spatial positions from tokens_cache.pt before sampling the remaining positions. 0 disables prompt-conditioned generation.",
    )
    parser.add_argument(
        "--prompt_offset",
        type=int,
        default=0,
        help="Starting index into tokens_cache.pt for prompt-conditioned generation. Prompts wrap around when the request exceeds cache length.",
    )
    parser.add_argument(
        "--candidate_factor",
        type=int,
        default=1,
        help="Generate num_samples * candidate_factor samples and rerank them against stage-1 reference stats.",
    )
    parser.add_argument("--nrow", type=int, default=None)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Atom temperature, and the default coefficient temperature when --coeff_temperature is omitted.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="Top-k truncation for atom sampling. <= 0 disables top-k, matching proto.py's default sampler.",
    )
    parser.add_argument(
        "--atom_temperature",
        type=float,
        default=None,
        help="Optional alias that overrides --temperature for atom sampling only.",
    )
    parser.add_argument(
        "--atom_top_k",
        type=int,
        default=None,
        help="Optional alias that overrides --top_k for atom sampling only.",
    )
    parser.add_argument(
        "--coeff_temperature",
        type=float,
        default=None,
        help="Optional separate coefficient temperature for real-valued Gaussian coefficient sampling.",
    )
    parser.add_argument(
        "--coeff_sample_mode",
        choices=["gaussian", "mean"],
        default="gaussian",
        help="How to sample real-valued coefficients. 'mean' disables coefficient noise.",
    )
    parser.add_argument(
        "--coeff_top_k",
        type=int,
        default=None,
        help="Legacy option from the removed flattened-token transformer sampler; ignored.",
    )
    parser.add_argument(
        "--greedy_atom_prefix_steps",
        type=int,
        default=None,
        help="Legacy option from the removed flattened-token transformer sampler; ignored.",
    )
    parser.add_argument(
        "--compare_legacy_greedy0",
        action="store_true",
        help="Legacy option from the removed flattened-token transformer sampler; ignored.",
    )
    parser.add_argument(
        "--output_image_size",
        type=int,
        default=None,
        help="Optional output image size for saved grids. Defaults to the decoder output resolution.",
    )
    parser.add_argument(
        "--selection_quality_weight",
        type=float,
        default=1.0,
        help="Quality penalty used during candidate reranking when candidate_factor > 1. Higher values prefer safer samples.",
    )
    parser.add_argument(
        "--selection_brightness_weight",
        type=float,
        default=1.0,
        help="Extra penalty for candidates that are darker than the stage-1 reference distribution.",
    )
    parser.add_argument(
        "--selection_overbright_weight",
        type=float,
        default=1.0,
        help="Extra penalty for candidates that are brighter than the stage-1 reference distribution.",
    )
    parser.add_argument(
        "--selection_reject_dark_z",
        type=float,
        default=1.5,
        help="Hard rejection threshold in reference-standard-deviation units for abnormally dark candidates. <= 0 disables rejection.",
    )
    parser.add_argument(
        "--selection_reject_bright_z",
        type=float,
        default=1.5,
        help="Hard rejection threshold in reference-standard-deviation units for abnormally bright candidates. <= 0 disables rejection.",
    )
    parser.add_argument(
        "--selection_mode",
        choices=["diverse", "quality_only"],
        default="diverse",
        help="How to choose the final subset from the candidate pool.",
    )
    parser.add_argument(
        "--selection_sort_by_quality",
        dest="selection_sort_by_quality",
        action="store_true",
        default=True,
        help="Sort the final selected batch by reference quality so earlier tile indices are better samples.",
    )
    parser.add_argument(
        "--no_selection_sort_by_quality",
        dest="selection_sort_by_quality",
        action="store_false",
        help="Keep the diversity-selection order in the final grid.",
    )
    parser.add_argument(
        "--selection_reference_max_items",
        type=int,
        default=256,
        help="Maximum number of stage-1 reference codes used to estimate sample-quality statistics.",
    )
    parser.add_argument(
        "--log_candidate_pool",
        dest="log_candidate_pool",
        action="store_true",
        default=True,
        help="Save and optionally W&B-log the full pre-rerank candidate pool when candidate_factor > 1.",
    )
    parser.add_argument(
        "--no_log_candidate_pool",
        dest="log_candidate_pool",
        action="store_false",
        help="Do not save a separate candidate-pool grid/payload.",
    )
    parser.add_argument(
        "--candidate_pool_nrow",
        type=int,
        default=None,
        help="Optional grid width for the saved candidate-pool sheet. Defaults to an auto square-ish layout.",
    )
    parser.add_argument(
        "--tf_heads",
        type=int,
        default=8,
        help="Transformer head count. This is not recoverable from the checkpoint and defaults to the training default.",
    )
    parser.add_argument(
        "--tf_dropout",
        type=float,
        default=0.1,
        help="Transformer dropout used when reconstructing the module for loading. Eval mode disables it at runtime.",
    )
    parser.add_argument("--wandb", dest="wandb", action="store_true", default=False, help="Enable Weights & Biases logging.")
    parser.add_argument("--no_wandb", dest="wandb", action="store_false", help="Disable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="laser-samples")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_dir", type=str, default="./wandb")
    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if args.prompt_spatial_steps < 0:
        raise ValueError("prompt_spatial_steps must be >= 0.")
    if args.prompt_offset < 0:
        raise ValueError("prompt_offset must be >= 0.")
    if args.candidate_factor <= 0:
        raise ValueError("candidate_factor must be positive.")
    if args.selection_brightness_weight < 0.0:
        raise ValueError("selection_brightness_weight must be >= 0.")
    if args.selection_overbright_weight < 0.0:
        raise ValueError("selection_overbright_weight must be >= 0.")
    if args.selection_reject_dark_z < 0.0:
        raise ValueError("selection_reject_dark_z must be >= 0.")
    if args.selection_reject_bright_z < 0.0:
        raise ValueError("selection_reject_bright_z must be >= 0.")
    if args.candidate_pool_nrow is not None and int(args.candidate_pool_nrow) <= 0:
        raise ValueError("candidate_pool_nrow must be positive when set.")
    resolved_atom_temperature = float(args.temperature if args.atom_temperature is None else args.atom_temperature)
    if resolved_atom_temperature <= 0.0:
        raise ValueError("temperature must be > 0.")
    resolved_atom_top_k = args.top_k if args.atom_top_k is None else args.atom_top_k
    resolved_atom_top_k = None if resolved_atom_top_k is None or int(resolved_atom_top_k) <= 0 else int(resolved_atom_top_k)
    resolved_coeff_temperature = None if args.coeff_temperature is None else float(args.coeff_temperature)
    if resolved_coeff_temperature is not None and resolved_coeff_temperature <= 0.0:
        raise ValueError("coeff_temperature must be > 0 when set.")
    if args.coeff_top_k is not None:
        print("[Sample] ignoring legacy coeff_top_k; real-valued coefficients are sampled from a scalar head.")
    if args.greedy_atom_prefix_steps is not None or args.compare_legacy_greedy0:
        print("[Sample] ignoring legacy greedy atom prefix options; the current spatial-depth prior sampler does not use them.")

    proto_mod = _load_proto_module()
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or (run_dir / "stage2_checkpoint_samples")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.wandb_name is None:
        args.wandb_name = output_dir.name
    if args.wandb_dir == "./wandb":
        args.wandb_dir = str(output_dir / "wandb")

    stage1_checkpoint = args.stage1_checkpoint or _first_existing_path(
        run_dir / "stage1" / "ae_best.pt",
        run_dir / "stage1" / "ae_last.pt",
    )
    stage2_checkpoint = args.stage2_checkpoint or _first_existing_path(
        run_dir / "stage2" / "transformer_last.pt",
    )
    token_cache_path = args.token_cache or (run_dir / "stage2" / "tokens_cache.pt")

    if stage1_checkpoint is None or not stage1_checkpoint.exists():
        raise FileNotFoundError(f"Stage-1 checkpoint not found under {run_dir / 'stage1'}")
    if stage2_checkpoint is None or not stage2_checkpoint.exists():
        raise FileNotFoundError(f"Stage-2 checkpoint not found under {run_dir / 'stage2'}")
    if not token_cache_path.exists():
        raise FileNotFoundError(f"Token cache not found: {token_cache_path}")

    stage1_state = _load_torch_payload(stage1_checkpoint)
    stage2_state = _load_torch_payload(stage2_checkpoint)
    token_cache = _load_torch_payload(token_cache_path)
    if "shape" not in token_cache:
        raise ValueError(f"Token cache {token_cache_path} is missing the stored token grid shape.")

    stage1_cfg = _infer_stage1_config(stage1_state, token_cache)
    H, W, D = (int(token_cache["shape"][0]), int(token_cache["shape"][1]), int(token_cache["shape"][2]))
    transformer_arch = _infer_stage2_arch(
        stage2_state,
        tf_heads=args.tf_heads,
        tf_dropout=args.tf_dropout,
    )

    if transformer_arch["real_valued_coeffs"] == bool(stage1_cfg["quantize_sparse_coeffs"]):
        raise ValueError(
            "Stage-1 sparse-code mode and stage-2 checkpoint disagree: "
            f"stage1 quantize_sparse_coeffs={stage1_cfg['quantize_sparse_coeffs']}, "
            f"stage2 real_valued_coeffs={transformer_arch['real_valued_coeffs']}"
        )

    device = _resolve_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("medium")

    laser = proto_mod.LASER(**stage1_cfg).to(device)
    proto_mod._load_module_checkpoint(laser, stage1_checkpoint)
    laser.eval()

    transformer = proto_mod.build_stage2_model(
        laser.bottleneck,
        stage2_arch=str(transformer_arch["stage2_arch"]),
        H=H,
        W=W,
        D=D,
        d_model=int(transformer_arch["d_model"]),
        n_heads=int(transformer_arch["n_heads"]),
        n_layers=int(transformer_arch["n_layers"]),
        d_ff=int(transformer_arch["d_ff"]),
        dropout=float(transformer_arch["dropout"]),
        n_global_spatial_tokens=int(transformer_arch["n_global_spatial_tokens"]),
        real_valued_coeffs=bool(transformer_arch["real_valued_coeffs"]),
        coeff_max_fallback=float(stage1_cfg["coef_max"]),
        autoregressive_coeffs=bool(transformer_arch["autoregressive_coeffs"]),
    ).to(device)
    checkpoint_gaussian_coeffs = bool(transformer_arch["gaussian_coeffs"])
    if bool(getattr(transformer.cfg, "gaussian_coeffs", False)) != checkpoint_gaussian_coeffs:
        print(
            "[Sample] stage-1 variational coefficient metadata and stage-2 checkpoint topology disagree; "
            f"reconstructing stage-2 with gaussian_coeffs={checkpoint_gaussian_coeffs} from the checkpoint."
        )
    if hasattr(transformer.cfg, "gaussian_coeffs"):
        transformer.cfg.gaussian_coeffs = checkpoint_gaussian_coeffs
    proto_mod._load_module_checkpoint(transformer, stage2_checkpoint)
    transformer.eval()

    wandb_run = None
    if args.wandb:
        wandb_run = proto_mod._init_wandb(args)

    candidate_count = int(args.num_samples) * max(1, int(args.candidate_factor))
    prompt_payload = _build_prompt_payload_from_cache(
        token_cache,
        batch_size=int(candidate_count),
        H=H,
        W=W,
        D=D,
        prompt_spatial_steps=int(args.prompt_spatial_steps),
        prompt_offset=int(args.prompt_offset),
    )
    prompt_source_sample = None
    if prompt_payload is not None:
        prompt_source_sample = _decode_sparse_code_payload(
            laser,
            tokens_flat=prompt_payload["tokens_flat"],
            coeffs_flat=prompt_payload["coeffs_flat"],
            H=H,
            W=W,
            D=D,
            output_image_size=args.output_image_size,
        )
    sample = _generate_samples(
        proto_mod,
        laser=laser,
        transformer=transformer,
        H=H,
        W=W,
        D=D,
        num_samples=int(candidate_count),
        atom_temperature=resolved_atom_temperature,
        atom_top_k=resolved_atom_top_k,
        coeff_temperature=resolved_coeff_temperature,
        coeff_sample_mode=str(args.coeff_sample_mode),
        output_image_size=args.output_image_size,
        seed=int(args.seed),
        prompt_tokens=(None if prompt_payload is None else prompt_payload["prompt_tokens"]),
        prompt_coeffs=(None if prompt_payload is None else prompt_payload["prompt_coeffs"]),
        prompt_mask=(None if prompt_payload is None else prompt_payload["prompt_mask"]),
    )
    candidate_sample = sample
    selected_idx = None
    reference_stats = None
    if int(args.candidate_factor) > 1 and candidate_count > int(args.num_samples):
        reference_stats = proto_mod._compute_stage2_sample_reference_stats(
            laser,
            token_cache["tokens_flat"],
            token_cache.get("coeffs_flat"),
            H=H,
            W=W,
            D=D,
            device=device,
            max_items=max(1, int(args.selection_reference_max_items)),
        )
        selected_idx = proto_mod._select_best_stage2_sample_indices(
            sample["images"],
            keep=int(args.num_samples),
            reference_stats=reference_stats,
            quality_weight=float(args.selection_quality_weight),
            brightness_weight=float(args.selection_brightness_weight),
            overbright_weight=float(args.selection_overbright_weight),
            reject_dark_z=float(args.selection_reject_dark_z),
            reject_bright_z=float(args.selection_reject_bright_z),
            selection_mode=str(args.selection_mode),
            sort_by_quality=bool(args.selection_sort_by_quality),
        )
        sample = _index_sample_payload(sample, selected_idx)

    candidate_pool_path = None
    candidate_pool_payload_path = None
    candidate_pool_records_path = None
    if bool(args.log_candidate_pool) and candidate_count > int(args.num_samples):
        candidate_pool_path = output_dir / "candidate_pool.png"
        candidate_pool_nrow = _resolve_nrow(int(candidate_count), args.candidate_pool_nrow)
        _save_indexed_image_grid(
            proto_mod,
            candidate_sample["images"],
            candidate_pool_path,
            nrow=candidate_pool_nrow,
            index_offset=0,
        )
        candidate_pool_payload_path = output_dir / "candidate_pool.pt"
        torch.save(
            {
                "images": candidate_sample["images"].to(torch.float16),
                "tokens_flat": candidate_sample["tokens_flat"],
                "coeffs_flat": candidate_sample["coeffs_flat"],
                "sampling": {
                    "candidate_count": int(candidate_count),
                    "candidate_factor": int(args.candidate_factor),
                    "selection_mode": str(args.selection_mode),
                    "indexing": "zero_based_left_to_right_top_to_bottom",
                    "selected_candidate_indices": (
                        None if selected_idx is None else [int(x) for x in selected_idx.tolist()]
                    ),
                },
            },
            candidate_pool_payload_path,
        )
        candidate_pool_records = _compute_candidate_pool_records(
            proto_mod,
            images=candidate_sample["images"],
            reference_stats=reference_stats,
            brightness_weight=float(args.selection_brightness_weight),
            overbright_weight=float(args.selection_overbright_weight),
            reject_dark_z=float(args.selection_reject_dark_z),
            reject_bright_z=float(args.selection_reject_bright_z),
            selected_idx=selected_idx,
        )
        candidate_pool_records_path = output_dir / "candidate_pool.json"
        with candidate_pool_records_path.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "indexing": "zero_based_left_to_right_top_to_bottom",
                    "candidate_count": int(candidate_count),
                    "candidate_factor": int(args.candidate_factor),
                    "selection_mode": str(args.selection_mode),
                    "nrow": int(candidate_pool_nrow),
                    "selected_candidate_indices": (
                        None if selected_idx is None else [int(x) for x in selected_idx.tolist()]
                    ),
                    "records": candidate_pool_records,
                },
                fh,
                indent=2,
            )

    prompt_source_path = None
    prompt_source_payload_path = None
    if prompt_source_sample is not None:
        prompt_source_path = output_dir / "prompt_source.png"
        proto_mod.save_image_grid(
            prompt_source_sample["images"],
            str(prompt_source_path),
            nrow=_resolve_nrow(int(prompt_source_sample["images"].size(0)), args.nrow),
        )
        prompt_source_payload_path = output_dir / "prompt_source.pt"
        torch.save(
            {
                "images": prompt_source_sample["images"].to(torch.float16),
                "tokens_flat": prompt_source_sample["tokens_flat"],
                "coeffs_flat": prompt_source_sample["coeffs_flat"],
                "prompt": {
                    "source_indices": prompt_payload["source_indices"],
                    "prompt_spatial_steps": int(prompt_payload["prompt_spatial_steps"]),
                    "prompt_fraction": float(prompt_payload["prompt_spatial_steps"]) / float(H * W),
                },
            },
            prompt_source_payload_path,
        )

    output_path = output_dir / "samples.png"
    proto_mod.save_image_grid(
        sample["images"],
        str(output_path),
        nrow=_resolve_nrow(int(args.num_samples), args.nrow),
    )
    output_payload_path = output_dir / "samples.pt"
    torch.save(
        {
            "images": sample["images"].to(torch.float16),
            "tokens_flat": sample["tokens_flat"],
            "coeffs_flat": sample["coeffs_flat"],
            "sampling": {
                "seed": int(args.seed),
                "num_samples": int(args.num_samples),
                "candidate_factor": int(args.candidate_factor),
                "candidate_count": int(candidate_count),
                "prompt_spatial_steps": int(args.prompt_spatial_steps),
                "prompt_offset": int(args.prompt_offset),
                "atom_temperature": float(resolved_atom_temperature),
                "atom_top_k": (None if resolved_atom_top_k is None else int(resolved_atom_top_k)),
                "coeff_temperature": (None if resolved_coeff_temperature is None else float(resolved_coeff_temperature)),
                "coeff_sample_mode": str(args.coeff_sample_mode),
                "selection_quality_weight": float(args.selection_quality_weight),
                "selection_brightness_weight": float(args.selection_brightness_weight),
                "selection_overbright_weight": float(args.selection_overbright_weight),
                "selection_reject_dark_z": float(args.selection_reject_dark_z),
                "selection_reject_bright_z": float(args.selection_reject_bright_z),
                "selection_mode": str(args.selection_mode),
                "selection_sort_by_quality": bool(args.selection_sort_by_quality),
                "selected_candidate_indices": (
                    None if selected_idx is None else [int(x) for x in selected_idx.tolist()]
                ),
                "log_candidate_pool": bool(args.log_candidate_pool),
                "candidate_pool_nrow": (None if args.candidate_pool_nrow is None else int(args.candidate_pool_nrow)),
                "output_image_size": (None if args.output_image_size is None else int(args.output_image_size)),
            },
        },
        output_payload_path,
    )

    transformer_cfg = transformer.cfg
    manifest = {
        "run_dir": str(run_dir),
        "stage1_checkpoint": str(stage1_checkpoint),
        "stage2_checkpoint": str(stage2_checkpoint),
        "token_cache": str(token_cache_path),
        "device": str(device),
        "token_grid_shape": {"H": H, "W": W, "D": D},
        "stage1_config": stage1_cfg,
        "transformer_config": {
            "stage2_arch": str(transformer_arch["stage2_arch"]),
            "vocab_size": int(transformer_cfg.vocab_size),
            "atom_vocab_size": (None if getattr(transformer_cfg, "atom_vocab_size", None) is None else int(transformer_cfg.atom_vocab_size)),
            "coeff_vocab_size": (None if getattr(transformer_cfg, "coeff_vocab_size", None) is None else int(transformer_cfg.coeff_vocab_size)),
            "real_valued_coeffs": bool(transformer_arch["real_valued_coeffs"]),
            "gaussian_coeffs": bool(getattr(transformer_cfg, "gaussian_coeffs", False)),
            "autoregressive_coeffs": bool(transformer_arch["autoregressive_coeffs"]),
            "d_model": int(transformer_cfg.d_model),
            "n_heads": int(transformer_arch["n_heads"]),
            "n_layers": int(transformer_arch["n_layers"]),
            "n_spatial_layers": int(transformer_arch.get("n_layers", 0)) if str(transformer_arch["stage2_arch"]) == "spatial_depth" else 0,
            "n_depth_layers": int(transformer_arch.get("n_depth_layers", 0)) if str(transformer_arch["stage2_arch"]) == "spatial_depth" else 0,
            "n_global_spatial_tokens": int(transformer_arch["n_global_spatial_tokens"]),
            "d_ff": int(transformer_cfg.d_ff),
            "dropout": float(transformer_cfg.dropout),
            "coeff_max": (float(transformer_cfg.coeff_max) if hasattr(transformer_cfg, "coeff_max") else None),
        },
        "sampling": {
            "num_samples": int(args.num_samples),
            "candidate_factor": int(args.candidate_factor),
            "candidate_count": int(candidate_count),
            "nrow": int(_resolve_nrow(args.num_samples, args.nrow)),
            "prompt_spatial_steps": int(args.prompt_spatial_steps),
            "prompt_offset": int(args.prompt_offset),
            "atom_temperature": float(resolved_atom_temperature),
            "atom_top_k": (None if resolved_atom_top_k is None else int(resolved_atom_top_k)),
            "coeff_temperature": (None if resolved_coeff_temperature is None else float(resolved_coeff_temperature)),
            "coeff_sample_mode": str(args.coeff_sample_mode),
            "selection_quality_weight": float(args.selection_quality_weight),
            "selection_brightness_weight": float(args.selection_brightness_weight),
            "selection_overbright_weight": float(args.selection_overbright_weight),
            "selection_reject_dark_z": float(args.selection_reject_dark_z),
            "selection_reject_bright_z": float(args.selection_reject_bright_z),
            "selection_mode": str(args.selection_mode),
            "selection_sort_by_quality": bool(args.selection_sort_by_quality),
            "selection_reference_max_items": int(args.selection_reference_max_items),
            "selected_candidate_indices": (
                None if selected_idx is None else [int(x) for x in selected_idx.tolist()]
            ),
            "log_candidate_pool": bool(args.log_candidate_pool),
            "candidate_pool_nrow": (None if args.candidate_pool_nrow is None else int(args.candidate_pool_nrow)),
            "legacy_atom_temperature": args.atom_temperature,
            "legacy_atom_top_k": args.atom_top_k,
            "legacy_coeff_temperature": args.coeff_temperature,
            "legacy_coeff_top_k": args.coeff_top_k,
            "legacy_greedy_atom_prefix_steps": args.greedy_atom_prefix_steps,
            "legacy_compare_greedy0": bool(args.compare_legacy_greedy0),
            "output_image_size": (None if args.output_image_size is None else int(args.output_image_size)),
            "seed": int(args.seed),
        },
        "outputs": [
            {
                "grid": str(output_path),
                "payload": str(output_payload_path),
            }
        ],
    }
    if prompt_payload is not None:
        manifest["prompt"] = {
            "enabled": True,
            "prompt_spatial_steps": int(prompt_payload["prompt_spatial_steps"]),
            "prompt_fraction": float(prompt_payload["prompt_spatial_steps"]) / float(H * W),
            "prompt_offset": int(args.prompt_offset),
            "source_indices": prompt_payload["source_indices"],
            "source_count": int(len(prompt_payload["source_indices"])),
        }
    else:
        manifest["prompt"] = {
            "enabled": False,
            "prompt_spatial_steps": 0,
            "prompt_fraction": 0.0,
            "prompt_offset": int(args.prompt_offset),
            "source_indices": [],
            "source_count": 0,
        }
    if candidate_pool_path is not None:
        manifest["outputs"].append(
            {
                "candidate_pool_grid": str(candidate_pool_path),
                "candidate_pool_payload": str(candidate_pool_payload_path),
                "candidate_pool_records": str(candidate_pool_records_path),
            }
        )
    if prompt_source_path is not None:
        manifest["outputs"].append(
            {
                "prompt_source_grid": str(prompt_source_path),
                "prompt_source_payload": str(prompt_source_payload_path),
            }
        )
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    if wandb_run is not None:
        try:
            proto_mod._log_wandb(
                wandb_run,
                {
                    "sample/num_samples": int(args.num_samples),
                    "sample/candidate_factor": int(args.candidate_factor),
                    "sample/candidate_count": int(candidate_count),
                    "sample/prompt_enabled": (1 if prompt_payload is not None else 0),
                    "sample/prompt_spatial_steps": int(args.prompt_spatial_steps),
                    "sample/prompt_fraction": float(args.prompt_spatial_steps) / float(H * W),
                    "sample/atom_temperature": float(resolved_atom_temperature),
                    "sample/atom_top_k": (-1 if resolved_atom_top_k is None else int(resolved_atom_top_k)),
                    "sample/coeff_temperature": (
                        float(resolved_coeff_temperature)
                        if resolved_coeff_temperature is not None else float(resolved_atom_temperature)
                    ),
                    "sample/coeff_mode_is_mean": (1 if str(args.coeff_sample_mode) == "mean" else 0),
                    "sample/selection_quality_weight": float(args.selection_quality_weight),
                    "sample/selection_brightness_weight": float(args.selection_brightness_weight),
                    "sample/selection_overbright_weight": float(args.selection_overbright_weight),
                    "sample/selection_reject_dark_z": float(args.selection_reject_dark_z),
                    "sample/selection_reject_bright_z": float(args.selection_reject_bright_z),
                    "sample/selection_mode_is_quality_only": (
                        1 if str(args.selection_mode) == "quality_only" else 0
                    ),
                    "sample/selection_sort_by_quality": (1 if bool(args.selection_sort_by_quality) else 0),
                    "sample/log_candidate_pool": (1 if bool(args.log_candidate_pool) else 0),
                },
            )
            proto_mod._log_wandb_image(
                wandb_run,
                "sample/grid",
                sample["images"],
                caption=(
                    f"run={run_dir.name} atom_temp={resolved_atom_temperature} "
                    f"atom_top_k={resolved_atom_top_k} coeff_mode={args.coeff_sample_mode} "
                    f"coeff_temp={resolved_coeff_temperature} prompt_steps={args.prompt_spatial_steps}"
                ),
            )
            if prompt_source_sample is not None:
                proto_mod._log_wandb_image(
                    wandb_run,
                    "sample/prompt_source",
                    prompt_source_sample["images"],
                    caption=(
                        f"run={run_dir.name} prompt_steps={args.prompt_spatial_steps} "
                        f"offset={args.prompt_offset}"
                    ),
                )
            if candidate_pool_path is not None:
                _log_wandb_file_image(
                    proto_mod,
                    wandb_run,
                    "sample/candidate_pool_grid",
                    candidate_pool_path,
                    caption=(
                        f"run={run_dir.name} candidate_count={candidate_count} "
                        "indexing=zero_based_left_to_right_top_to_bottom"
                    ),
                )
            wandb_run.summary["sample_coeff_sample_mode"] = str(args.coeff_sample_mode)
            wandb_run.summary["sample_output_dir"] = str(output_dir)
            wandb_run.summary["sample_run_dir"] = str(run_dir)
            wandb_run.summary["sample_grid_path"] = str(output_path)
            wandb_run.summary["sample_manifest_path"] = str(manifest_path)
            wandb_run.summary["sample_payload_path"] = str(output_payload_path)
            wandb_run.summary["sample_prompt_enabled"] = bool(prompt_payload is not None)
            wandb_run.summary["sample_prompt_spatial_steps"] = int(args.prompt_spatial_steps)
            wandb_run.summary["sample_prompt_offset"] = int(args.prompt_offset)
            wandb_run.summary["sample_prompt_fraction"] = float(args.prompt_spatial_steps) / float(H * W)
            if prompt_source_path is not None:
                wandb_run.summary["sample_prompt_source_grid_path"] = str(prompt_source_path)
                wandb_run.summary["sample_prompt_source_payload_path"] = str(prompt_source_payload_path)
            if candidate_pool_path is not None:
                wandb_run.summary["sample_candidate_pool_grid_path"] = str(candidate_pool_path)
                wandb_run.summary["sample_candidate_pool_payload_path"] = str(candidate_pool_payload_path)
                wandb_run.summary["sample_candidate_pool_records_path"] = str(candidate_pool_records_path)
        finally:
            proto_mod._finish_wandb(wandb_run)

    print(f"[Sample] wrote {output_path}")
    print(f"[Sample] wrote {output_payload_path}")
    print(f"[Sample] wrote {manifest_path}")
    if prompt_source_path is not None:
        print(f"[Sample] wrote {prompt_source_path}")
        print(f"[Sample] wrote {prompt_source_payload_path}")
    if candidate_pool_path is not None:
        print(f"[Sample] wrote {candidate_pool_path}")
        print(f"[Sample] wrote {candidate_pool_payload_path}")
        print(f"[Sample] wrote {candidate_pool_records_path}")


if __name__ == "__main__":
    main()
