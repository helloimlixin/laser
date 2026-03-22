#!/usr/bin/env python3

import argparse
import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = SCRIPT_DIR / "runs" / "laser_celeba128"


def _load_proto_module():
    module_path = SCRIPT_DIR / "proto.py"
    spec = importlib.util.spec_from_file_location("scratch_proto_checkpoint_eval", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to import proto module from {}".format(module_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _indexed_block_count(keys: Iterable[str], prefix: str) -> int:
    pattern = re.compile(r"^{}\.(\d+)\.".format(re.escape(prefix)))
    indices = set()
    for key in keys:
        match = pattern.match(key)
        if match is not None:
            indices.add(int(match.group(1)))
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


def _infer_spatial_prior_arch(
    state_dict: Dict[str, torch.Tensor],
    *,
    tf_heads: int,
    tf_dropout: float,
) -> dict:
    if tf_heads <= 0:
        raise ValueError("tf_heads must be positive, got {}".format(tf_heads))
    if "token_emb.weight" not in state_dict:
        raise ValueError("Stage-2 checkpoint is missing token_emb.weight.")

    n_spatial_layers = _indexed_block_count(state_dict.keys(), "spatial_blocks")
    n_depth_layers = _indexed_block_count(state_dict.keys(), "depth_blocks")
    if n_spatial_layers <= 0 and n_depth_layers <= 0:
        raise ValueError("This evaluator only supports the current spatial-depth prior checkpoint format.")

    vocab_size, d_model = state_dict["token_emb.weight"].shape
    if d_model % tf_heads != 0:
        raise ValueError("d_model={} is not divisible by tf_heads={}".format(d_model, tf_heads))

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
        "vocab_size": int(vocab_size),
        "d_model": int(d_model),
        "n_spatial_layers": int(n_spatial_layers),
        "n_depth_layers": int(n_depth_layers),
        "d_ff": int(d_ff),
        "dropout": float(tf_dropout),
        "n_heads": int(tf_heads),
        "n_global_spatial_tokens": int(state_dict.get("global_spatial_tokens", torch.empty(1, 0, 1)).shape[1]),
        "real_valued_coeffs": any(key.startswith("coeff_head.") for key in state_dict.keys()),
        "gaussian_coeffs": any(key.startswith("coeff_logvar_head.") for key in state_dict.keys()),
        "autoregressive_coeffs": bool(autoregressive_coeffs),
    }


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_prompt_steps(spec: str, total_steps: int) -> List[int]:
    values = []
    for chunk in str(spec).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = int(chunk)
        if value < 0 or value > total_steps:
            raise ValueError("prompt step {} is outside [0, {}]".format(value, total_steps))
        values.append(value)
    if not values:
        return [0]
    deduped = sorted(set(values))
    return deduped


def _to_float_list(tensor: torch.Tensor) -> List[float]:
    return [float(x) for x in tensor.detach().cpu().tolist()]


def _quantized_teacher_forced_metrics(
    proto_mod,
    transformer,
    tokens_flat: torch.Tensor,
    *,
    batch_size: int,
    H: int,
    W: int,
    D: int,
    device: torch.device,
) -> dict:
    T = int(H) * int(W)
    token_match_sum = 0.0
    atom_match_sum = 0.0
    coeff_match_sum = 0.0
    step_match_sum = torch.zeros(T, dtype=torch.float64)
    atom_step_match_sum = torch.zeros(T, dtype=torch.float64)
    coeff_step_match_sum = torch.zeros(T, dtype=torch.float64)
    spatial_token_match_sum = torch.zeros(T, dtype=torch.float64)
    spatial_atom_match_sum = torch.zeros(T, dtype=torch.float64)
    spatial_coeff_match_sum = torch.zeros(T, dtype=torch.float64)
    spatial_token_ce_sum = torch.zeros(T, dtype=torch.float64)
    spatial_atom_ce_sum = torch.zeros(T, dtype=torch.float64)
    spatial_coeff_ce_sum = torch.zeros(T, dtype=torch.float64)
    token_ce_sum = 0.0
    atom_ce_sum = 0.0
    coeff_ce_sum = 0.0
    n_items = 0

    vocab = int(transformer.cfg.vocab_size)
    with torch.no_grad():
        for start in range(0, int(tokens_flat.size(0)), int(batch_size)):
            end = min(int(tokens_flat.size(0)), start + int(batch_size))
            tok_grid = tokens_flat[start:end].view(-1, T, D).to(device=device, dtype=torch.long)
            logits = transformer(tok_grid)
            per_token_ce = F.cross_entropy(
                logits.reshape(-1, vocab),
                tok_grid.reshape(-1),
                reduction="none",
            ).view(-1, T, D)
            ce_loss, atom_ce_loss, coeff_ce_loss, _ = proto_mod._compute_quantized_rq_losses(
                per_token_ce,
                atom_loss_weight=1.0,
                coeff_loss_weight=1.0,
                coeff_depth_weighting="none",
                coeff_focal_gamma=0.0,
                coeff_logits=None,
            )
            pred = logits.argmax(dim=-1)
            exact = pred.eq(tok_grid)
            atom_exact = exact[..., 0::2]
            coeff_exact = exact[..., 1::2]
            step_exact = exact.all(dim=-1)
            atom_step_exact = atom_exact.all(dim=-1)
            coeff_step_exact = coeff_exact.all(dim=-1)

            batch_n = int(tok_grid.size(0))
            n_items += batch_n
            token_match_sum += float(exact.sum().item())
            atom_match_sum += float(atom_exact.sum().item())
            coeff_match_sum += float(coeff_exact.sum().item())
            token_ce_sum += float(per_token_ce.sum().item())
            atom_ce_sum += float(per_token_ce[..., 0::2].sum().item())
            coeff_ce_sum += float(per_token_ce[..., 1::2].sum().item())

            step_match_sum += step_exact.to(torch.float64).sum(dim=0).cpu()
            atom_step_match_sum += atom_step_exact.to(torch.float64).sum(dim=0).cpu()
            coeff_step_match_sum += coeff_step_exact.to(torch.float64).sum(dim=0).cpu()
            spatial_token_match_sum += exact.to(torch.float64).sum(dim=(0, 2)).cpu()
            spatial_atom_match_sum += atom_exact.to(torch.float64).sum(dim=(0, 2)).cpu()
            spatial_coeff_match_sum += coeff_exact.to(torch.float64).sum(dim=(0, 2)).cpu()
            spatial_token_ce_sum += per_token_ce.to(torch.float64).sum(dim=(0, 2)).cpu()
            spatial_atom_ce_sum += per_token_ce[..., 0::2].to(torch.float64).sum(dim=(0, 2)).cpu()
            spatial_coeff_ce_sum += per_token_ce[..., 1::2].to(torch.float64).sum(dim=(0, 2)).cpu()

    token_count = float(n_items * T * D)
    atom_depth = float(D // 2)
    atom_count = float(n_items * T * atom_depth)
    coeff_count = float(n_items * T * atom_depth)
    per_step_token_count = float(n_items * D)
    per_step_half_count = float(n_items * atom_depth)

    return {
        "num_items": int(n_items),
        "token_ce": float(token_ce_sum / max(token_count, 1.0)),
        "atom_ce": float(atom_ce_sum / max(atom_count, 1.0)),
        "coeff_ce": float(coeff_ce_sum / max(coeff_count, 1.0)),
        "token_exact": float(token_match_sum / max(token_count, 1.0)),
        "atom_exact": float(atom_match_sum / max(atom_count, 1.0)),
        "coeff_exact": float(coeff_match_sum / max(coeff_count, 1.0)),
        "spatial_token_exact_by_index": _to_float_list(spatial_token_match_sum / max(per_step_token_count, 1.0)),
        "spatial_atom_exact_by_index": _to_float_list(spatial_atom_match_sum / max(per_step_half_count, 1.0)),
        "spatial_coeff_exact_by_index": _to_float_list(spatial_coeff_match_sum / max(per_step_half_count, 1.0)),
        "step_exact_by_index": _to_float_list(step_match_sum / max(float(n_items), 1.0)),
        "atom_step_exact_by_index": _to_float_list(atom_step_match_sum / max(float(n_items), 1.0)),
        "coeff_step_exact_by_index": _to_float_list(coeff_step_match_sum / max(float(n_items), 1.0)),
        "spatial_token_ce_by_index": _to_float_list(spatial_token_ce_sum / max(per_step_token_count, 1.0)),
        "spatial_atom_ce_by_index": _to_float_list(spatial_atom_ce_sum / max(per_step_half_count, 1.0)),
        "spatial_coeff_ce_by_index": _to_float_list(spatial_coeff_ce_sum / max(per_step_half_count, 1.0)),
    }


def _real_valued_teacher_forced_metrics(
    transformer,
    laser,
    tokens_flat: torch.Tensor,
    coeffs_flat: torch.Tensor,
    *,
    batch_size: int,
    H: int,
    W: int,
    D: int,
    device: torch.device,
) -> dict:
    T = int(H) * int(W)
    token_match_sum = 0.0
    token_ce_sum = 0.0
    coeff_mse_sum = 0.0
    coeff_mae_sum = 0.0
    step_match_sum = torch.zeros(T, dtype=torch.float64)
    spatial_token_match_sum = torch.zeros(T, dtype=torch.float64)
    spatial_token_ce_sum = torch.zeros(T, dtype=torch.float64)
    spatial_coeff_mse_sum = torch.zeros(T, dtype=torch.float64)
    spatial_coeff_mae_sum = torch.zeros(T, dtype=torch.float64)
    n_items = 0

    vocab = int(transformer.cfg.vocab_size)
    with torch.no_grad():
        for start in range(0, int(tokens_flat.size(0)), int(batch_size)):
            end = min(int(tokens_flat.size(0)), start + int(batch_size))
            tok_grid = tokens_flat[start:end].view(-1, T, D).to(device=device, dtype=torch.long)
            coeff_grid = coeffs_flat[start:end].view(-1, T, D).to(device=device, dtype=torch.float32)
            out = transformer(tok_grid, coeff_grid)
            if len(out) == 3:
                atom_logits, coeff_pred, _ = out
            else:
                atom_logits, coeff_pred = out
            coeff_pred = laser.clamp_sparse_coeffs(coeff_pred)
            coeff_target = laser.clamp_sparse_coeffs(coeff_grid)

            per_token_ce = F.cross_entropy(
                atom_logits.reshape(-1, vocab),
                tok_grid.reshape(-1),
                reduction="none",
            ).view(-1, T, D)
            pred = atom_logits.argmax(dim=-1)
            exact = pred.eq(tok_grid)
            step_exact = exact.all(dim=-1)
            coeff_diff = coeff_pred - coeff_target
            coeff_sq = coeff_diff.square()
            coeff_abs = coeff_diff.abs()

            batch_n = int(tok_grid.size(0))
            n_items += batch_n
            token_match_sum += float(exact.sum().item())
            token_ce_sum += float(per_token_ce.sum().item())
            coeff_mse_sum += float(coeff_sq.sum().item())
            coeff_mae_sum += float(coeff_abs.sum().item())
            step_match_sum += step_exact.to(torch.float64).sum(dim=0).cpu()
            spatial_token_match_sum += exact.to(torch.float64).sum(dim=(0, 2)).cpu()
            spatial_token_ce_sum += per_token_ce.to(torch.float64).sum(dim=(0, 2)).cpu()
            spatial_coeff_mse_sum += coeff_sq.to(torch.float64).sum(dim=(0, 2)).cpu()
            spatial_coeff_mae_sum += coeff_abs.to(torch.float64).sum(dim=(0, 2)).cpu()

    token_count = float(n_items * T * D)
    per_step_token_count = float(n_items * D)
    return {
        "num_items": int(n_items),
        "token_ce": float(token_ce_sum / max(token_count, 1.0)),
        "token_exact": float(token_match_sum / max(token_count, 1.0)),
        "coeff_mse": float(coeff_mse_sum / max(token_count, 1.0)),
        "coeff_mae": float(coeff_mae_sum / max(token_count, 1.0)),
        "spatial_token_exact_by_index": _to_float_list(spatial_token_match_sum / max(per_step_token_count, 1.0)),
        "step_exact_by_index": _to_float_list(step_match_sum / max(float(n_items), 1.0)),
        "spatial_token_ce_by_index": _to_float_list(spatial_token_ce_sum / max(per_step_token_count, 1.0)),
        "spatial_coeff_mse_by_index": _to_float_list(spatial_coeff_mse_sum / max(per_step_token_count, 1.0)),
        "spatial_coeff_mae_by_index": _to_float_list(spatial_coeff_mae_sum / max(per_step_token_count, 1.0)),
    }


def _build_prompt_mask(batch_size: int, T: int, D: int, prompt_steps: int) -> Optional[torch.Tensor]:
    if int(prompt_steps) <= 0:
        return None
    mask = torch.zeros(batch_size, T, D, dtype=torch.bool)
    mask[:, : int(prompt_steps), :] = True
    return mask


def _quantized_greedy_prompt_metrics(
    transformer,
    tokens_flat: torch.Tensor,
    *,
    batch_size: int,
    H: int,
    W: int,
    D: int,
    device: torch.device,
    prompt_steps: int,
    top_k: int,
    temperature: float,
) -> dict:
    T = int(H) * int(W)
    token_match_sum = 0.0
    atom_match_sum = 0.0
    coeff_match_sum = 0.0
    step_match_sum = torch.zeros(T, dtype=torch.float64)
    spatial_token_match_sum = torch.zeros(T, dtype=torch.float64)
    spatial_atom_match_sum = torch.zeros(T, dtype=torch.float64)
    spatial_coeff_match_sum = torch.zeros(T, dtype=torch.float64)
    n_items = 0

    with torch.no_grad():
        for start in range(0, int(tokens_flat.size(0)), int(batch_size)):
            end = min(int(tokens_flat.size(0)), start + int(batch_size))
            target = tokens_flat[start:end].view(-1, T, D)
            prompt_mask = _build_prompt_mask(int(target.size(0)), T, D, prompt_steps)
            if prompt_mask is None:
                generated = transformer.generate(
                    batch_size=int(target.size(0)),
                    temperature=float(temperature),
                    top_k=int(top_k),
                    coeff_temperature=float(temperature),
                    coeff_sample_mode="mean",
                    show_progress=False,
                )
            else:
                generated = transformer.generate(
                    batch_size=int(target.size(0)),
                    temperature=float(temperature),
                    top_k=int(top_k),
                    coeff_temperature=float(temperature),
                    coeff_sample_mode="mean",
                    show_progress=False,
                    prompt_tokens=target.to(device=device, dtype=torch.long),
                    prompt_mask=prompt_mask.to(device=device),
                )
            gen_tokens = generated.to(device="cpu", dtype=torch.long)
            exact = gen_tokens.eq(target.to(torch.long))
            atom_exact = exact[..., 0::2]
            coeff_exact = exact[..., 1::2]
            step_exact = exact.all(dim=-1)

            batch_n = int(target.size(0))
            n_items += batch_n
            token_match_sum += float(exact.sum().item())
            atom_match_sum += float(atom_exact.sum().item())
            coeff_match_sum += float(coeff_exact.sum().item())
            step_match_sum += step_exact.to(torch.float64).sum(dim=0)
            spatial_token_match_sum += exact.to(torch.float64).sum(dim=(0, 2))
            spatial_atom_match_sum += atom_exact.to(torch.float64).sum(dim=(0, 2))
            spatial_coeff_match_sum += coeff_exact.to(torch.float64).sum(dim=(0, 2))

    token_count = float(n_items * T * D)
    atom_depth = float(D // 2)
    atom_count = float(n_items * T * atom_depth)
    coeff_count = float(n_items * T * atom_depth)
    per_step_token_count = float(n_items * D)
    per_step_half_count = float(n_items * atom_depth)

    step_exact = step_match_sum / max(float(n_items), 1.0)
    spatial_token_exact = spatial_token_match_sum / max(per_step_token_count, 1.0)
    spatial_atom_exact = spatial_atom_match_sum / max(per_step_half_count, 1.0)
    spatial_coeff_exact = spatial_coeff_match_sum / max(per_step_half_count, 1.0)
    prefix_slice = slice(0, int(prompt_steps))
    suffix_slice = slice(int(prompt_steps), T)

    result = {
        "prompt_steps": int(prompt_steps),
        "token_exact": float(token_match_sum / max(token_count, 1.0)),
        "atom_exact": float(atom_match_sum / max(atom_count, 1.0)),
        "coeff_exact": float(coeff_match_sum / max(coeff_count, 1.0)),
        "step_exact_by_index": _to_float_list(step_exact),
        "spatial_token_exact_by_index": _to_float_list(spatial_token_exact),
        "spatial_atom_exact_by_index": _to_float_list(spatial_atom_exact),
        "spatial_coeff_exact_by_index": _to_float_list(spatial_coeff_exact),
    }
    if int(prompt_steps) > 0:
        result["prefix_step_exact"] = float(step_exact[prefix_slice].mean().item())
        result["prefix_token_exact"] = float(spatial_token_exact[prefix_slice].mean().item())
        result["prefix_atom_exact"] = float(spatial_atom_exact[prefix_slice].mean().item())
        result["prefix_coeff_exact"] = float(spatial_coeff_exact[prefix_slice].mean().item())
    else:
        result["prefix_step_exact"] = None
        result["prefix_token_exact"] = None
        result["prefix_atom_exact"] = None
        result["prefix_coeff_exact"] = None
    if int(prompt_steps) < T:
        result["suffix_step_exact"] = float(step_exact[suffix_slice].mean().item())
        result["suffix_token_exact"] = float(spatial_token_exact[suffix_slice].mean().item())
        result["suffix_atom_exact"] = float(spatial_atom_exact[suffix_slice].mean().item())
        result["suffix_coeff_exact"] = float(spatial_coeff_exact[suffix_slice].mean().item())
    else:
        result["suffix_step_exact"] = None
        result["suffix_token_exact"] = None
        result["suffix_atom_exact"] = None
        result["suffix_coeff_exact"] = None
    return result


def _real_valued_greedy_prompt_metrics(
    transformer,
    tokens_flat: torch.Tensor,
    coeffs_flat: torch.Tensor,
    *,
    batch_size: int,
    H: int,
    W: int,
    D: int,
    prompt_steps: int,
    temperature: float,
) -> dict:
    T = int(H) * int(W)
    atom_match_sum = 0.0
    coeff_mse_sum = 0.0
    coeff_mae_sum = 0.0
    step_match_sum = torch.zeros(T, dtype=torch.float64)
    spatial_atom_match_sum = torch.zeros(T, dtype=torch.float64)
    spatial_coeff_mse_sum = torch.zeros(T, dtype=torch.float64)
    spatial_coeff_mae_sum = torch.zeros(T, dtype=torch.float64)
    n_items = 0

    with torch.no_grad():
        for start in range(0, int(tokens_flat.size(0)), int(batch_size)):
            end = min(int(tokens_flat.size(0)), start + int(batch_size))
            target_tokens = tokens_flat[start:end].view(-1, T, D)
            target_coeffs = coeffs_flat[start:end].view(-1, T, D)
            prompt_mask = _build_prompt_mask(int(target_tokens.size(0)), T, D, prompt_steps)
            if prompt_mask is None:
                gen_atoms, gen_coeffs = transformer.generate(
                    batch_size=int(target_tokens.size(0)),
                    temperature=float(temperature),
                    top_k=1,
                    coeff_temperature=float(temperature),
                    coeff_sample_mode="mean",
                    show_progress=False,
                )
            else:
                gen_atoms, gen_coeffs = transformer.generate(
                    batch_size=int(target_tokens.size(0)),
                    temperature=float(temperature),
                    top_k=1,
                    coeff_temperature=float(temperature),
                    coeff_sample_mode="mean",
                    show_progress=False,
                    prompt_tokens=target_tokens.to(device=next(transformer.parameters()).device, dtype=torch.long),
                    prompt_coeffs=target_coeffs.to(
                        device=next(transformer.parameters()).device,
                        dtype=next(transformer.parameters()).dtype,
                    ),
                    prompt_mask=prompt_mask.to(device=next(transformer.parameters()).device),
                )
            gen_atoms = gen_atoms.to(device="cpu", dtype=torch.long)
            gen_coeffs = gen_coeffs.to(device="cpu", dtype=torch.float32)
            atom_exact = gen_atoms.eq(target_tokens.to(torch.long))
            step_exact = atom_exact.all(dim=-1)
            coeff_diff = gen_coeffs - target_coeffs.to(torch.float32)
            coeff_sq = coeff_diff.square()
            coeff_abs = coeff_diff.abs()

            batch_n = int(target_tokens.size(0))
            n_items += batch_n
            atom_match_sum += float(atom_exact.sum().item())
            coeff_mse_sum += float(coeff_sq.sum().item())
            coeff_mae_sum += float(coeff_abs.sum().item())
            step_match_sum += step_exact.to(torch.float64).sum(dim=0)
            spatial_atom_match_sum += atom_exact.to(torch.float64).sum(dim=(0, 2))
            spatial_coeff_mse_sum += coeff_sq.to(torch.float64).sum(dim=(0, 2))
            spatial_coeff_mae_sum += coeff_abs.to(torch.float64).sum(dim=(0, 2))

    token_count = float(n_items * T * D)
    per_step_token_count = float(n_items * D)
    step_exact = step_match_sum / max(float(n_items), 1.0)
    spatial_atom_exact = spatial_atom_match_sum / max(per_step_token_count, 1.0)
    spatial_coeff_mse = spatial_coeff_mse_sum / max(per_step_token_count, 1.0)
    spatial_coeff_mae = spatial_coeff_mae_sum / max(per_step_token_count, 1.0)
    prefix_slice = slice(0, int(prompt_steps))
    suffix_slice = slice(int(prompt_steps), T)

    result = {
        "prompt_steps": int(prompt_steps),
        "atom_exact": float(atom_match_sum / max(token_count, 1.0)),
        "coeff_mse": float(coeff_mse_sum / max(token_count, 1.0)),
        "coeff_mae": float(coeff_mae_sum / max(token_count, 1.0)),
        "step_exact_by_index": _to_float_list(step_exact),
        "spatial_atom_exact_by_index": _to_float_list(spatial_atom_exact),
        "spatial_coeff_mse_by_index": _to_float_list(spatial_coeff_mse),
        "spatial_coeff_mae_by_index": _to_float_list(spatial_coeff_mae),
    }
    if int(prompt_steps) > 0:
        result["prefix_step_exact"] = float(step_exact[prefix_slice].mean().item())
        result["prefix_atom_exact"] = float(spatial_atom_exact[prefix_slice].mean().item())
    else:
        result["prefix_step_exact"] = None
        result["prefix_atom_exact"] = None
    if int(prompt_steps) < T:
        result["suffix_step_exact"] = float(step_exact[suffix_slice].mean().item())
        result["suffix_atom_exact"] = float(spatial_atom_exact[suffix_slice].mean().item())
        result["suffix_coeff_mse"] = float(spatial_coeff_mse[suffix_slice].mean().item())
        result["suffix_coeff_mae"] = float(spatial_coeff_mae[suffix_slice].mean().item())
    else:
        result["suffix_step_exact"] = None
        result["suffix_atom_exact"] = None
        result["suffix_coeff_mse"] = None
        result["suffix_coeff_mae"] = None
    return result


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--stage1_checkpoint", type=Path, default=None)
    parser.add_argument("--stage2_checkpoint", type=Path, default=None)
    parser.add_argument("--token_cache", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_items", type=int, default=512)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--prompt_steps", type=str, default="0,16,32,48,56,60,63")
    parser.add_argument("--greedy_temperature", type=float, default=1.0)
    parser.add_argument("--greedy_top_k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tf_heads", type=int, default=8)
    parser.add_argument("--tf_dropout", type=float, default=0.1)
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print("[Eval] ignoring unrecognized args: {}".format(" ".join(str(x) for x in unknown_args)))

    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.max_items == 0:
        raise ValueError("max_items cannot be zero")
    if args.greedy_temperature <= 0.0:
        raise ValueError("greedy_temperature must be > 0")

    run_dir = args.run_dir.expanduser().resolve()
    stage1_checkpoint = args.stage1_checkpoint or _first_existing_path(
        run_dir / "stage1" / "ae_best.pt",
    )
    stage2_checkpoint = args.stage2_checkpoint or _first_existing_path(
        run_dir / "stage2" / "transformer_last.pt",
        run_dir / "stage2" / "transformer_final.pt",
    )
    token_cache_path = args.token_cache or (run_dir / "stage2" / "tokens_cache.pt")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (run_dir / "stage2_checkpoint_eval" / stage2_checkpoint.stem).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if stage1_checkpoint is None or not stage1_checkpoint.exists():
        raise FileNotFoundError("Stage-1 checkpoint not found under {}".format(run_dir / "stage1"))
    if stage2_checkpoint is None or not stage2_checkpoint.exists():
        raise FileNotFoundError("Stage-2 checkpoint not found under {}".format(run_dir / "stage2"))
    if not token_cache_path.exists():
        raise FileNotFoundError("Token cache not found: {}".format(token_cache_path))

    proto_mod = _load_proto_module()
    _seed_everything(int(args.seed))

    stage1_state = _load_torch_payload(stage1_checkpoint)
    stage2_state = _load_torch_payload(stage2_checkpoint)
    token_cache = _load_torch_payload(token_cache_path)
    if "shape" not in token_cache:
        raise ValueError("Token cache {} is missing stored token grid shape.".format(token_cache_path))

    H, W, D = (
        int(token_cache["shape"][0]),
        int(token_cache["shape"][1]),
        int(token_cache["shape"][2]),
    )
    T = int(H) * int(W)
    prompt_steps = _parse_prompt_steps(args.prompt_steps, T)
    stage1_cfg = _infer_stage1_config(stage1_state, token_cache)
    transformer_arch = _infer_spatial_prior_arch(
        stage2_state,
        tf_heads=int(args.tf_heads),
        tf_dropout=float(args.tf_dropout),
    )
    if transformer_arch["real_valued_coeffs"] == bool(stage1_cfg["quantize_sparse_coeffs"]):
        raise ValueError(
            "Stage-1 sparse-code mode and stage-2 checkpoint disagree: stage1 quantize_sparse_coeffs={} "
            "stage2 real_valued_coeffs={}".format(
                stage1_cfg["quantize_sparse_coeffs"], transformer_arch["real_valued_coeffs"]
            )
        )

    device = _resolve_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("medium")

    laser = proto_mod.LASER(**stage1_cfg).to(device)
    proto_mod._load_module_checkpoint(laser, stage1_checkpoint)
    laser.eval()

    transformer_cfg = proto_mod.build_spatial_depth_prior_config(
        laser.bottleneck,
        H=H,
        W=W,
        D=D,
        d_model=int(transformer_arch["d_model"]),
        n_heads=int(transformer_arch["n_heads"]),
        n_spatial_layers=int(transformer_arch["n_spatial_layers"]),
        n_depth_layers=int(transformer_arch["n_depth_layers"]),
        d_ff=int(transformer_arch["d_ff"]),
        dropout=float(transformer_arch["dropout"]),
        n_global_spatial_tokens=int(transformer_arch["n_global_spatial_tokens"]),
        real_valued_coeffs=bool(transformer_arch["real_valued_coeffs"]),
        coeff_max_fallback=float(stage1_cfg["coef_max"]),
        autoregressive_coeffs=bool(transformer_arch["autoregressive_coeffs"]),
    )
    transformer_cfg.gaussian_coeffs = bool(transformer_arch["gaussian_coeffs"])
    transformer = proto_mod.SpatialDepthPrior(transformer_cfg).to(device)
    proto_mod._load_module_checkpoint(transformer, stage2_checkpoint)
    transformer.eval()

    tokens_flat = token_cache["tokens_flat"]
    coeffs_flat = token_cache.get("coeffs_flat")
    total_items = int(tokens_flat.size(0))
    start = max(0, int(args.offset))
    if start >= total_items:
        raise ValueError("offset {} is >= token cache size {}".format(start, total_items))
    end = total_items if int(args.max_items) < 0 else min(total_items, start + int(args.max_items))
    tokens_flat = tokens_flat[start:end].contiguous()
    if coeffs_flat is not None:
        coeffs_flat = coeffs_flat[start:end].contiguous()

    summary = {
        "run_dir": str(run_dir),
        "stage1_checkpoint": str(stage1_checkpoint),
        "stage2_checkpoint": str(stage2_checkpoint),
        "token_cache": str(token_cache_path),
        "output_dir": str(output_dir),
        "device": str(device),
        "eval_items": int(tokens_flat.size(0)),
        "offset": int(start),
        "token_grid_shape": {"H": H, "W": W, "D": D, "T": T},
        "stage1_config": stage1_cfg,
        "transformer_config": transformer_arch,
        "eval_config": {
            "batch_size": int(args.batch_size),
            "prompt_steps": prompt_steps,
            "greedy_temperature": float(args.greedy_temperature),
            "greedy_top_k": int(args.greedy_top_k),
            "seed": int(args.seed),
        },
    }

    print("[Eval] teacher-forced metrics on {} items".format(int(tokens_flat.size(0))))
    if coeffs_flat is None:
        teacher_metrics = _quantized_teacher_forced_metrics(
            proto_mod,
            transformer,
            tokens_flat,
            batch_size=int(args.batch_size),
            H=H,
            W=W,
            D=D,
            device=device,
        )
    else:
        teacher_metrics = _real_valued_teacher_forced_metrics(
            transformer,
            laser,
            tokens_flat,
            coeffs_flat,
            batch_size=int(args.batch_size),
            H=H,
            W=W,
            D=D,
            device=device,
        )
    summary["teacher_forced"] = teacher_metrics

    greedy_results = []
    print("[Eval] greedy prefix continuation for prompt steps {}".format(prompt_steps))
    for prompt_step in prompt_steps:
        if coeffs_flat is None:
            result = _quantized_greedy_prompt_metrics(
                transformer,
                tokens_flat,
                batch_size=int(args.batch_size),
                H=H,
                W=W,
                D=D,
                device=device,
                prompt_steps=int(prompt_step),
                top_k=int(args.greedy_top_k),
                temperature=float(args.greedy_temperature),
            )
            print(
                "[Eval] prompt_steps={} token_exact={:.6f} suffix_token_exact={} suffix_step_exact={}".format(
                    int(prompt_step),
                    float(result["token_exact"]),
                    "None" if result["suffix_token_exact"] is None else "{:.6f}".format(float(result["suffix_token_exact"])),
                    "None" if result["suffix_step_exact"] is None else "{:.6f}".format(float(result["suffix_step_exact"])),
                )
            )
        else:
            result = _real_valued_greedy_prompt_metrics(
                transformer,
                tokens_flat,
                coeffs_flat,
                batch_size=int(args.batch_size),
                H=H,
                W=W,
                D=D,
                prompt_steps=int(prompt_step),
                temperature=float(args.greedy_temperature),
            )
            print(
                "[Eval] prompt_steps={} atom_exact={:.6f} suffix_atom_exact={} suffix_step_exact={}".format(
                    int(prompt_step),
                    float(result["atom_exact"]),
                    "None" if result["suffix_atom_exact"] is None else "{:.6f}".format(float(result["suffix_atom_exact"])),
                    "None" if result["suffix_step_exact"] is None else "{:.6f}".format(float(result["suffix_step_exact"])),
                )
            )
        greedy_results.append(result)
    summary["greedy_prefix_eval"] = greedy_results

    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    print("[Eval] wrote {}".format(summary_path))


if __name__ == "__main__":
    main()
