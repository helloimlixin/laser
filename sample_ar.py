#!/usr/bin/env python3
"""Sample images from maintained stage-2 Lightning checkpoints."""

import argparse
import math
from pathlib import Path
from typing import Optional

import torch
from torchvision.utils import save_image

from src.data.token_cache import load_token_cache
from src.models.sparse_token_prior import (
    SparseTokenPriorModule,
    build_sparse_prior_from_hparams,
    token_cache_grid_shape,
)
from src.stage2_compat import (
    decode_stage2_outputs,
    ensure_stage2_cache_metadata,
    load_stage1_decoder_bundle,
    load_torch_payload,
)
from src.stage2_paths import infer_latest_stage2_checkpoint, infer_latest_token_cache


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _resolve_nrow(num_samples: int, nrow: Optional[int]) -> int:
    if nrow is not None and int(nrow) > 0:
        return int(nrow)
    root = math.isqrt(num_samples)
    if root * root == num_samples:
        return max(1, root)
    return max(1, math.ceil(math.sqrt(num_samples)))


def _build_module_from_checkpoint(checkpoint_path: Path, cache: dict):
    payload = load_torch_payload(checkpoint_path)
    hparams = dict(payload.get("hyper_parameters", {}) or {})
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Checkpoint {checkpoint_path} is missing a Lightning state_dict")

    prior = build_sparse_prior_from_hparams(cache, hparams=hparams)
    module = SparseTokenPriorModule(
        prior=prior,
        learning_rate=float(hparams.get("learning_rate", 3e-4)),
        weight_decay=float(hparams.get("weight_decay", 0.01)),
        warmup_steps=int(hparams.get("warmup_steps", 1000)),
        min_lr_ratio=float(hparams.get("min_lr_ratio", 0.01)),
        atom_loss_weight=float(hparams.get("atom_loss_weight", 1.0)),
        coeff_loss_weight=float(hparams.get("coeff_loss_weight", 1.0)),
        coeff_depth_weighting=str(hparams.get("coeff_depth_weighting", "none")),
        coeff_focal_gamma=float(hparams.get("coeff_focal_gamma", 0.0)),
        coeff_loss_type=hparams.get("coeff_loss_type", "auto"),
        coeff_huber_delta=float(hparams.get("coeff_huber_delta", 0.5)),
        sample_coeff_temperature=hparams.get("sample_coeff_temperature"),
        sample_coeff_mode=str(hparams.get("sample_coeff_mode") or "gaussian"),
    )
    module.load_state_dict(state_dict)
    module.save_hyperparameters(hparams)
    return module, hparams


def main():
    parser = argparse.ArgumentParser(description="Sample from maintained sparse-token stage-2 checkpoints.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Stage-2 Lightning checkpoint (.ckpt).")
    parser.add_argument("--token_cache", type=Path, default=None, help="Token cache used for training.")
    parser.add_argument("--output_dir", type=Path, default=None, help="Where to save sample grids.")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of images to generate.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling cutoff; <=0 disables.")
    parser.add_argument("--coeff_temperature", type=float, default=None, help="Optional coefficient sampling temperature for real-valued priors.")
    parser.add_argument("--coeff_sample_mode", type=str, default=None, choices=["gaussian", "mean"], help="Coefficient sampling rule for real-valued priors.")
    parser.add_argument("--device", type=str, default="auto", help="Device string or 'auto'.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--nrow", type=int, default=None, help="Images per row in the output grid.")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or infer_latest_stage2_checkpoint()
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        raise FileNotFoundError("Could not infer a maintained stage-2 checkpoint. Pass --checkpoint explicitly.")
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()

    payload = load_torch_payload(checkpoint_path)
    hparams = dict(payload.get("hyper_parameters", {}) or {})
    token_cache_path = args.token_cache or hparams.get("token_cache_path") or infer_latest_token_cache()
    if token_cache_path is None or not Path(token_cache_path).exists():
        raise FileNotFoundError("Could not infer a token cache. Pass --token_cache explicitly.")
    token_cache_path = Path(token_cache_path).expanduser().resolve()

    cache = ensure_stage2_cache_metadata(
        load_token_cache(token_cache_path),
        token_cache_path=token_cache_path,
        output_root=(Path.cwd() / "outputs"),
    )
    module, _ = _build_module_from_checkpoint(checkpoint_path, cache)

    device = _resolve_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("medium")
        torch.cuda.manual_seed_all(int(args.seed))
    torch.manual_seed(int(args.seed))

    module = module.eval().to(device)
    stage1_bundle = load_stage1_decoder_bundle(
        cache,
        token_cache_path=token_cache_path,
        device=device,
        output_root=(Path.cwd() / "outputs"),
    )
    token_h, token_w, token_depth = token_cache_grid_shape(cache)
    top_k = None if int(args.top_k) <= 0 else int(args.top_k)

    with torch.no_grad():
        generated = module.generate_sparse_codes(
            int(args.num_samples),
            temperature=float(args.temperature),
            top_k=top_k,
            coeff_temperature=args.coeff_temperature,
            coeff_sample_mode=args.coeff_sample_mode,
        )
        if getattr(module.prior, "real_valued_coeffs", False):
            atom_ids, coeffs = generated
            images = decode_stage2_outputs(
                stage1_bundle,
                atom_ids.view(int(args.num_samples), token_h, token_w, token_depth),
                coeffs.view(int(args.num_samples), token_h, token_w, token_depth),
                device=device,
            ).detach().cpu()
        else:
            token_grid = generated.view(int(args.num_samples), token_h, token_w, token_depth)
            images = decode_stage2_outputs(
                stage1_bundle,
                token_grid,
                device=device,
            ).detach().cpu()

    output_dir = args.output_dir or (checkpoint_path.parent / f"{checkpoint_path.stem}_samples")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    nrow = _resolve_nrow(int(args.num_samples), args.nrow)
    raw_path = output_dir / "samples.png"
    auto_path = output_dir / "samples_autocontrast.png"
    save_image(images, raw_path, nrow=nrow, normalize=True, value_range=(-1.0, 1.0))
    save_image(images, auto_path, nrow=nrow, normalize=True, scale_each=True)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Token cache: {token_cache_path}")
    print(f"Saved raw grid: {raw_path}")
    print(f"Saved autocontrast grid: {auto_path}")


if __name__ == "__main__":
    main()
