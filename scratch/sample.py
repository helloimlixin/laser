#!/usr/bin/env python3
"""
Sample images from existing LASER stage-1/stage-2 checkpoints without retraining.

This is intended for quick validation of sampling-only changes against an existing
run directory. By default it uses the current non-quantized CelebA checkpoint and
the updated sampler behavior from `scratch/laser.py`.

Examples:
  python3 sample.py
  python3 sample.py --num_samples 16 --compare_legacy_greedy0
  python3 sample.py --run_dir runs/laser_celeba128_quantized --tf_heads 8
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
DEFAULT_RUN_DIR = SCRIPT_DIR / "runs" / "laser_celeba128_no_quantized"


def _load_scratch_laser_module():
    module_path = SCRIPT_DIR / "laser.py"
    spec = importlib.util.spec_from_file_location("scratch_laser_checkpoint_sampler", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import scratch laser module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _indexed_block_count(keys: Iterable[str], prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.")
    indices = {int(match.group(1)) for key in keys if (match := pattern.match(key))}
    return len(indices)


def _infer_stage1_config(state_dict: Dict[str, torch.Tensor], token_cache: dict) -> dict:
    metadata = token_cache.get("metadata") or {}
    dictionary_shape = tuple(state_dict["bottleneck.dictionary"].shape)
    embedding_dim = int(state_dict["pre_bottleneck.weight"].shape[0])
    patch_dim = int(dictionary_shape[0])
    num_embeddings = int(dictionary_shape[1])
    patch_area = patch_dim // max(embedding_dim, 1)
    patch_size = int(round(math.sqrt(patch_area)))
    if patch_size * patch_size * embedding_dim != patch_dim:
        raise ValueError(
            f"Could not infer latent patch size from dictionary shape {dictionary_shape} "
            f"and embedding_dim={embedding_dim}"
        )

    quantize_sparse_coeffs = metadata.get("quantize_sparse_coeffs")
    if quantize_sparse_coeffs is None:
        quantize_sparse_coeffs = token_cache.get("coeffs_flat") is None

    if "latent_patch_stride" not in metadata:
        raise ValueError(
            "Token cache metadata is missing latent_patch_stride; pass a cache built by the current training code."
        )

    return {
        "in_channels": int(state_dict["encoder.down_convs.0.weight"].shape[1]),
        "num_hiddens": int(state_dict["encoder.conv3.weight"].shape[0]),
        "num_downsamples": _indexed_block_count(state_dict.keys(), "encoder.down_convs"),
        "num_residual_layers": _indexed_block_count(state_dict.keys(), "encoder.res.layers"),
        "num_residual_hiddens": int(state_dict["encoder.res.layers.0.conv1.weight"].shape[0]),
        "embedding_dim": embedding_dim,
        "num_embeddings": num_embeddings,
        "sparsity_level": int(metadata.get("sparsity_level", token_cache["shape"][2] // (2 if quantize_sparse_coeffs else 1))),
        "latent_patch_size": int(metadata.get("latent_patch_size", patch_size)),
        "latent_patch_stride": int(metadata["latent_patch_stride"]),
        "n_bins": int(state_dict["bottleneck.coef_bin_centers"].shape[0]),
        "coef_max": float(state_dict["bottleneck.coef_bin_centers"].abs().max().item()),
        "commitment_cost": 0.25,
        "quantize_sparse_coeffs": bool(quantize_sparse_coeffs),
    }


def _infer_transformer_config(
    laser_mod,
    state_dict: Dict[str, torch.Tensor],
    token_cache: dict,
    stage1_cfg: dict,
    *,
    tf_heads: int,
    tf_dropout: float,
):
    if tf_heads <= 0:
        raise ValueError(f"tf_heads must be positive, got {tf_heads}")
    vocab_size, d_model = state_dict["token_emb.weight"].shape
    d_ff = int(state_dict["blocks.0.ffn.0.weight"].shape[0])
    n_layers = _indexed_block_count(state_dict.keys(), "blocks")
    H, W, D = token_cache["shape"]
    if d_model % tf_heads != 0:
        raise ValueError(f"d_model={d_model} is not divisible by tf_heads={tf_heads}")
    predict_coefficients = any(key.startswith("coeff_head.") for key in state_dict.keys())
    quantized_tokens = bool(stage1_cfg["quantize_sparse_coeffs"])
    return laser_mod.TransformerConfig(
        vocab_size=int(vocab_size),
        H=int(H),
        W=int(W),
        D=int(D),
        atom_vocab_size=(int(stage1_cfg["num_embeddings"]) if quantized_tokens else None),
        coeff_vocab_size=(int(stage1_cfg["n_bins"]) if quantized_tokens else None),
        predict_coefficients=bool(predict_coefficients),
        d_model=int(d_model),
        n_heads=int(tf_heads),
        n_layers=int(n_layers),
        d_ff=int(d_ff),
        dropout=float(tf_dropout),
        coeff_max=float(stage1_cfg["coef_max"]),
    )


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
    laser_mod,
    *,
    laser,
    transformer,
    H: int,
    W: int,
    D: int,
    num_samples: int,
    temperature: float,
    top_k: Optional[int],
    atom_temperature: Optional[float],
    atom_top_k: Optional[int],
    coeff_temperature: Optional[float],
    coeff_top_k: Optional[int],
    greedy_atom_prefix_steps: Optional[int],
    output_image_size: Optional[int],
    seed: int,
):
    _seed_everything(seed)
    gen_out = transformer.generate(
        batch_size=num_samples,
        temperature=temperature,
        top_k=top_k,
        atom_temperature=atom_temperature,
        atom_top_k=atom_top_k,
        coeff_temperature=coeff_temperature,
        coeff_top_k=coeff_top_k,
        greedy_atom_prefix_steps=greedy_atom_prefix_steps,
        show_progress=True,
        progress_desc="[Sample] generating tokens",
    )
    if transformer.predict_coefficients:
        tokens_flat, coeffs_flat = gen_out
        tokens = tokens_flat.view(-1, H, W, D)
        coeffs = coeffs_flat.view(-1, H, W, D)
        imgs = laser.decode(tokens.to(next(laser.parameters()).device), coeffs.to(next(laser.parameters()).device))
    else:
        tokens_flat = gen_out
        coeffs_flat = None
        tokens = tokens_flat.view(-1, H, W, D)
        imgs = laser.decode(tokens.to(next(laser.parameters()).device))

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
    parser.add_argument("--atom_temperature", type=float, default=None)
    parser.add_argument("--atom_top_k", type=int, default=None)
    parser.add_argument("--coeff_temperature", type=float, default=None)
    parser.add_argument("--coeff_top_k", type=int, default=None)
    parser.add_argument(
        "--greedy_atom_prefix_steps",
        type=int,
        default=None,
        help="Override the sampler's greedy atom prefix length. Omit to use the current auto default.",
    )
    parser.add_argument(
        "--compare_legacy_greedy0",
        action="store_true",
        help="Also render a second grid with greedy_atom_prefix_steps forced to 0.",
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
    if args.atom_temperature is not None and args.atom_temperature <= 0.0:
        raise ValueError("atom_temperature must be > 0 when provided.")
    if args.coeff_temperature is not None and args.coeff_temperature <= 0.0:
        raise ValueError("coeff_temperature must be > 0 when provided.")
    if args.greedy_atom_prefix_steps is not None and args.greedy_atom_prefix_steps < 0:
        raise ValueError("greedy_atom_prefix_steps must be >= 0 when provided.")

    laser_mod = _load_scratch_laser_module()
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or (run_dir / "stage2_checkpoint_samples")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stage1_checkpoint = args.stage1_checkpoint or Path(
        laser_mod._first_existing_path(
            str(run_dir / "stage1" / "ae_best.pt"),
            str(run_dir / "stage1" / "ae_last.pt"),
        )
    )
    stage2_checkpoint = args.stage2_checkpoint or Path(
        laser_mod._first_existing_path(str(run_dir / "stage2" / "transformer_last.pt"))
    )
    token_cache_path = args.token_cache or (run_dir / "stage2" / "tokens_cache.pt")

    if stage1_checkpoint is None or not stage1_checkpoint.exists():
        raise FileNotFoundError(f"Stage-1 checkpoint not found under {run_dir / 'stage1'}")
    if stage2_checkpoint is None or not stage2_checkpoint.exists():
        raise FileNotFoundError(f"Stage-2 checkpoint not found under {run_dir / 'stage2'}")
    if not token_cache_path.exists():
        raise FileNotFoundError(f"Token cache not found: {token_cache_path}")

    stage1_state = laser_mod._load_torch_payload(str(stage1_checkpoint))
    stage2_state = laser_mod._load_torch_payload(str(stage2_checkpoint))
    token_cache = laser_mod._load_torch_payload(str(token_cache_path))
    if "shape" not in token_cache:
        raise ValueError(f"Token cache {token_cache_path} is missing the stored token grid shape.")

    stage1_cfg = _infer_stage1_config(stage1_state, token_cache)
    transformer_cfg = _infer_transformer_config(
        laser_mod,
        stage2_state,
        token_cache,
        stage1_cfg,
        tf_heads=args.tf_heads,
        tf_dropout=args.tf_dropout,
    )

    device = _resolve_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("medium")

    laser = laser_mod.LASER(**stage1_cfg).to(device)
    laser_mod._load_module_checkpoint(laser, str(stage1_checkpoint))
    laser.eval()

    transformer = laser_mod.TransformerPrior(
        transformer_cfg,
        bos_token_id=laser.bottleneck.bos_token_id,
        pad_token_id=laser.bottleneck.pad_token_id,
    ).to(device)
    laser_mod._load_module_checkpoint(transformer, str(stage2_checkpoint))
    transformer.eval()

    H, W, D = token_cache["shape"]
    resolved_top_k = None if args.top_k is None or int(args.top_k) <= 0 else int(args.top_k)
    resolved_atom_top_k = None if args.atom_top_k is None or int(args.atom_top_k) <= 0 else int(args.atom_top_k)
    resolved_coeff_top_k = None if args.coeff_top_k is None or int(args.coeff_top_k) <= 0 else int(args.coeff_top_k)
    resolved_greedy = laser_mod._resolve_stage2_sample_greedy_atom_prefix_steps(
        args.greedy_atom_prefix_steps,
        quantized_tokens=(transformer.content_vocab_size is not None),
        token_depth=int(D),
        grid_width=int(W),
    )

    runs = [
        {
            "label": "current",
            "greedy_atom_prefix_steps": args.greedy_atom_prefix_steps,
            "resolved_greedy_atom_prefix_steps": resolved_greedy,
            "filename": "samples.png",
            "seed": args.seed,
        }
    ]
    if args.compare_legacy_greedy0:
        runs.append(
            {
                "label": "legacy_greedy0",
                "greedy_atom_prefix_steps": 0,
                "resolved_greedy_atom_prefix_steps": 0,
                "filename": "samples_legacy.png",
                "seed": args.seed,
            }
        )

    manifest = {
        "run_dir": str(run_dir),
        "stage1_checkpoint": str(stage1_checkpoint),
        "stage2_checkpoint": str(stage2_checkpoint),
        "token_cache": str(token_cache_path),
        "device": str(device),
        "token_grid_shape": {"H": int(H), "W": int(W), "D": int(D)},
        "stage1_config": stage1_cfg,
        "transformer_config": {
            "vocab_size": int(transformer_cfg.vocab_size),
            "H": int(transformer_cfg.H),
            "W": int(transformer_cfg.W),
            "D": int(transformer_cfg.D),
            "atom_vocab_size": (
                None if transformer_cfg.atom_vocab_size is None else int(transformer_cfg.atom_vocab_size)
            ),
            "coeff_vocab_size": (
                None if transformer_cfg.coeff_vocab_size is None else int(transformer_cfg.coeff_vocab_size)
            ),
            "predict_coefficients": bool(transformer_cfg.predict_coefficients),
            "d_model": int(transformer_cfg.d_model),
            "n_heads": int(transformer_cfg.n_heads),
            "n_layers": int(transformer_cfg.n_layers),
            "d_ff": int(transformer_cfg.d_ff),
            "dropout": float(transformer_cfg.dropout),
            "coeff_max": float(transformer_cfg.coeff_max),
        },
        "sampling": {
            "num_samples": int(args.num_samples),
            "nrow": int(_resolve_nrow(args.num_samples, args.nrow)),
            "temperature": float(args.temperature),
            "top_k": (None if resolved_top_k is None else int(resolved_top_k)),
            "atom_temperature": (None if args.atom_temperature is None else float(args.atom_temperature)),
            "atom_top_k": (None if resolved_atom_top_k is None else int(resolved_atom_top_k)),
            "coeff_temperature": (None if args.coeff_temperature is None else float(args.coeff_temperature)),
            "coeff_top_k": (None if resolved_coeff_top_k is None else int(resolved_coeff_top_k)),
            "requested_greedy_atom_prefix_steps": (
                None if args.greedy_atom_prefix_steps is None else int(args.greedy_atom_prefix_steps)
            ),
            "resolved_greedy_atom_prefix_steps": int(resolved_greedy),
            "output_image_size": (None if args.output_image_size is None else int(args.output_image_size)),
            "seed": int(args.seed),
        },
        "outputs": [],
    }

    for run_info in runs:
        print(
            f"[Sample] {run_info['label']}: seed={run_info['seed']} "
            f"requested_greedy={run_info['greedy_atom_prefix_steps']} "
            f"resolved_greedy={run_info['resolved_greedy_atom_prefix_steps']}"
        )
        sample = _generate_samples(
            laser_mod,
            laser=laser,
            transformer=transformer,
            H=int(H),
            W=int(W),
            D=int(D),
            num_samples=int(args.num_samples),
            temperature=float(args.temperature),
            top_k=resolved_top_k,
            atom_temperature=args.atom_temperature,
            atom_top_k=resolved_atom_top_k,
            coeff_temperature=args.coeff_temperature,
            coeff_top_k=resolved_coeff_top_k,
            greedy_atom_prefix_steps=run_info["greedy_atom_prefix_steps"],
            output_image_size=args.output_image_size,
            seed=int(run_info["seed"]),
        )
        output_path = output_dir / run_info["filename"]
        laser_mod.save_image_grid(
            sample["images"],
            str(output_path),
            nrow=_resolve_nrow(int(args.num_samples), args.nrow),
        )
        output_payload_path = output_dir / f"{output_path.stem}.pt"
        torch.save(
            {
                "tokens_flat": sample["tokens_flat"],
                "coeffs_flat": sample["coeffs_flat"],
                "sampling": run_info,
            },
            output_payload_path,
        )
        manifest["outputs"].append(
            {
                "label": run_info["label"],
                "grid": str(output_path),
                "payload": str(output_payload_path),
                "requested_greedy_atom_prefix_steps": run_info["greedy_atom_prefix_steps"],
                "resolved_greedy_atom_prefix_steps": int(run_info["resolved_greedy_atom_prefix_steps"]),
            }
        )
        print(f"[Sample] wrote {output_path}")

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[Sample] wrote {manifest_path}")


if __name__ == "__main__":
    main()
