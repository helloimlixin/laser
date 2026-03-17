#!/usr/bin/env python3
"""
Sample images from existing LASER stage-1/stage-2 checkpoints without retraining.

This loader targets the current `proto.py` + `spatial_prior.py` checkpoint format.
Legacy flattened-token transformer checkpoints are no longer supported here.

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

    return {
        "in_channels": int(state_dict["encoder.conv_in.weight"].shape[1]),
        "num_hiddens": int(state_dict["encoder.conv_in.weight"].shape[0]),
        "num_downsamples": max(0, _indexed_block_count(state_dict.keys(), "encoder.down") - 1),
        "num_residual_layers": max(1, _indexed_block_count(state_dict.keys(), "encoder.down.0.block")),
        "resolution": int(metadata.get("image_size", 128)),
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
    }


def _infer_spatial_prior_arch(
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
    if n_spatial_layers <= 0 and n_depth_layers <= 0:
        raise ValueError(
            "This sample script only supports the current spatial-depth prior checkpoint format."
        )

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
    temperature: float,
    top_k: Optional[int],
    output_image_size: Optional[int],
    seed: int,
):
    _seed_everything(seed)
    gen_out = transformer.generate(
        batch_size=num_samples,
        temperature=temperature,
        top_k=top_k,
        show_progress=True,
        progress_desc="[Sample] generating sparse codes",
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
    parser.add_argument("--nrow", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=16)
    parser.add_argument(
        "--atom_temperature",
        type=float,
        default=None,
        help="Legacy option from the removed flattened-token transformer sampler; ignored.",
    )
    parser.add_argument(
        "--atom_top_k",
        type=int,
        default=None,
        help="Legacy option from the removed flattened-token transformer sampler; ignored.",
    )
    parser.add_argument(
        "--coeff_temperature",
        type=float,
        default=None,
        help="Legacy option from the removed flattened-token transformer sampler; ignored.",
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
    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if args.temperature <= 0.0:
        raise ValueError("temperature must be > 0.")
    if args.atom_temperature is not None or args.atom_top_k is not None or args.coeff_temperature is not None or args.coeff_top_k is not None:
        print("[Sample] ignoring legacy atom/coeff sampling overrides; the spatial-depth prior uses a single temperature/top_k.")
    if args.greedy_atom_prefix_steps is not None or args.compare_legacy_greedy0:
        print("[Sample] ignoring legacy greedy atom prefix options; the current spatial-depth prior sampler does not use them.")

    proto_mod = _load_proto_module()
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or (run_dir / "stage2_checkpoint_samples")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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
    transformer_arch = _infer_spatial_prior_arch(
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
    )
    transformer = proto_mod.SpatialDepthPrior(transformer_cfg).to(device)
    proto_mod._load_module_checkpoint(transformer, stage2_checkpoint)
    transformer.eval()

    resolved_top_k = None if args.top_k is None or int(args.top_k) <= 0 else int(args.top_k)
    sample = _generate_samples(
        proto_mod,
        laser=laser,
        transformer=transformer,
        H=H,
        W=W,
        D=D,
        num_samples=int(args.num_samples),
        temperature=float(args.temperature),
        top_k=resolved_top_k,
        output_image_size=args.output_image_size,
        seed=int(args.seed),
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
            "tokens_flat": sample["tokens_flat"],
            "coeffs_flat": sample["coeffs_flat"],
            "sampling": {
                "seed": int(args.seed),
                "num_samples": int(args.num_samples),
                "temperature": float(args.temperature),
                "top_k": (None if resolved_top_k is None else int(resolved_top_k)),
                "output_image_size": (None if args.output_image_size is None else int(args.output_image_size)),
            },
        },
        output_payload_path,
    )

    manifest = {
        "run_dir": str(run_dir),
        "stage1_checkpoint": str(stage1_checkpoint),
        "stage2_checkpoint": str(stage2_checkpoint),
        "token_cache": str(token_cache_path),
        "device": str(device),
        "token_grid_shape": {"H": H, "W": W, "D": D},
        "stage1_config": stage1_cfg,
        "transformer_config": {
            "vocab_size": int(transformer_cfg.vocab_size),
            "atom_vocab_size": (None if transformer_cfg.atom_vocab_size is None else int(transformer_cfg.atom_vocab_size)),
            "coeff_vocab_size": (None if transformer_cfg.coeff_vocab_size is None else int(transformer_cfg.coeff_vocab_size)),
            "real_valued_coeffs": bool(transformer_cfg.real_valued_coeffs),
            "d_model": int(transformer_cfg.d_model),
            "n_heads": int(transformer_cfg.n_heads),
            "n_spatial_layers": int(transformer_cfg.n_spatial_layers),
            "n_depth_layers": int(transformer_cfg.n_depth_layers),
            "n_global_spatial_tokens": int(transformer_cfg.n_global_spatial_tokens),
            "d_ff": int(transformer_cfg.d_ff),
            "dropout": float(transformer_cfg.dropout),
            "coeff_max": float(transformer_cfg.coeff_max),
        },
        "sampling": {
            "num_samples": int(args.num_samples),
            "nrow": int(_resolve_nrow(args.num_samples, args.nrow)),
            "temperature": float(args.temperature),
            "top_k": (None if resolved_top_k is None else int(resolved_top_k)),
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
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[Sample] wrote {output_path}")
    print(f"[Sample] wrote {output_payload_path}")
    print(f"[Sample] wrote {manifest_path}")


if __name__ == "__main__":
    main()
