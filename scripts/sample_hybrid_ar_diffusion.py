#!/usr/bin/env python3
"""Sample a hybrid sparse prior: AR atom ids + diffusion real coefficients."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.sparse_diffusion_prior import (
    SparseCoeffDiffusionConfig,
    SparseCoeffDiffusionModule,
    SparseCoeffDenoiserPrior,
)
from src.models.sparse_token_prior import SparseTokenPriorModule, build_sparse_prior_from_hparams
from src.stage2_compat import (
    decode_stage2_outputs,
    ensure_stage2_cache_metadata,
    load_stage1_decoder_bundle,
)


def _load_torch(path: Path, *, map_location="cpu") -> dict:
    payload = torch.load(str(path), map_location=map_location)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected checkpoint/cache dict at {path}, got {type(payload)!r}")
    return payload


def _checkpoint_hparams(payload: dict, path: Path) -> dict:
    hparams = payload.get("hyper_parameters")
    if not isinstance(hparams, dict):
        raise RuntimeError(f"Checkpoint {path} is missing Lightning hyper_parameters")
    return hparams


def _load_ar_module(cache: dict, checkpoint_path: Path, device: torch.device) -> SparseTokenPriorModule:
    payload = _load_torch(checkpoint_path, map_location="cpu")
    hparams = _checkpoint_hparams(payload, checkpoint_path)
    prior = build_sparse_prior_from_hparams(cache, hparams=hparams)
    module = SparseTokenPriorModule(
        prior=prior,
        learning_rate=float(hparams.get("learning_rate", 3.0e-4)),
        weight_decay=float(hparams.get("weight_decay", 0.01)),
        warmup_steps=int(hparams.get("warmup_steps", 1000)),
        min_lr_ratio=float(hparams.get("min_lr_ratio", 0.01)),
        atom_loss_weight=float(hparams.get("atom_loss_weight", 1.0)),
        coeff_loss_weight=float(hparams.get("coeff_loss_weight", 0.0)),
        coeff_depth_weighting=str(hparams.get("coeff_depth_weighting", "none")),
        coeff_focal_gamma=float(hparams.get("coeff_focal_gamma", 0.0)),
        coeff_loss_type=hparams.get("coeff_loss_type", "auto"),
        coeff_huber_delta=float(hparams.get("coeff_huber_delta", 0.5)),
        sample_coeff_temperature=hparams.get("sample_coeff_temperature"),
        sample_coeff_mode=str(hparams.get("sample_coeff_mode", "mean")),
    )
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Checkpoint {checkpoint_path} is missing state_dict")
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"AR checkpoint missing keys: {len(missing)}")
    if unexpected:
        print(f"AR checkpoint unexpected keys: {len(unexpected)}")
    return module.eval().to(device)


def _load_diffusion_module(checkpoint_path: Path, device: torch.device) -> SparseCoeffDiffusionModule:
    payload = _load_torch(checkpoint_path, map_location="cpu")
    hparams = _checkpoint_hparams(payload, checkpoint_path)
    cfg = SparseCoeffDiffusionConfig(
        H=int(hparams["prior_H"]),
        W=int(hparams["prior_W"]),
        D=int(hparams["prior_D"]),
        atom_vocab_size=int(hparams["prior_atom_vocab_size"]),
        hidden_channels=int(hparams["prior_hidden_channels"]),
        atom_embed_dim=int(hparams["prior_atom_embed_dim"]),
        time_embed_dim=int(hparams["prior_time_embed_dim"]),
        n_res_blocks=int(hparams["prior_n_res_blocks"]),
        dropout=float(hparams["prior_dropout"]),
        num_timesteps=int(hparams["prior_num_timesteps"]),
        beta_start=float(hparams["prior_beta_start"]),
        beta_end=float(hparams["prior_beta_end"]),
        coeff_mean=float(hparams["prior_coeff_mean"]),
        coeff_std=float(hparams["prior_coeff_std"]),
    )
    prior = SparseCoeffDenoiserPrior(cfg)
    module = SparseCoeffDiffusionModule(
        prior=prior,
        learning_rate=float(hparams.get("learning_rate", 2.0e-4)),
        weight_decay=float(hparams.get("weight_decay", 1.0e-2)),
    )
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Checkpoint {checkpoint_path} is missing state_dict")
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Diffusion checkpoint missing keys: {len(missing)}")
    if unexpected:
        print(f"Diffusion checkpoint unexpected keys: {len(unexpected)}")
    return module.eval().to(device)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-cache-path", type=Path, required=True)
    parser.add_argument("--ar-checkpoint", type=Path, required=True)
    parser.add_argument("--diffusion-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--num-images", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--coeff-temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    torch.manual_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    token_cache_path = args.token_cache_path.expanduser().resolve()
    cache = _load_torch(token_cache_path, map_location="cpu")
    cache = ensure_stage2_cache_metadata(
        cache,
        token_cache_path=token_cache_path,
        output_root=args.output_root.expanduser().resolve(),
    )
    shape = cache.get("shape")
    if not isinstance(shape, (tuple, list)) or len(shape) != 3:
        raise RuntimeError(f"Token cache has invalid shape metadata: {shape!r}")
    H, W, D = (int(shape[0]), int(shape[1]), int(shape[2]))

    ar = _load_ar_module(cache, args.ar_checkpoint.expanduser().resolve(), device)
    diffusion = _load_diffusion_module(args.diffusion_checkpoint.expanduser().resolve(), device)
    stage1 = load_stage1_decoder_bundle(
        cache,
        token_cache_path=token_cache_path,
        device=device,
        output_root=args.output_root.expanduser().resolve(),
    )

    top_k = None if int(args.top_k) <= 0 else int(args.top_k)
    atoms, _ = ar.generate_sparse_codes(
        int(args.num_images),
        temperature=float(args.temperature),
        top_k=top_k,
        coeff_sample_mode="mean",
    )
    atoms = atoms.to(device=device, dtype=torch.long)
    _, coeffs = diffusion.generate_sparse_codes(
        int(args.num_images),
        atom_ids=atoms,
        temperature=float(args.coeff_temperature),
        coeff_temperature=float(args.coeff_temperature),
        steps=int(args.diffusion_steps),
    )
    atom_grid = atoms.view(int(args.num_images), H, W, D)
    coeff_grid = coeffs.to(device=device, dtype=torch.float32).view(int(args.num_images), H, W, D)
    images = decode_stage2_outputs(stage1, atom_grid, coeff_grid, device=device).detach().cpu()

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    nrow = max(1, int(math.ceil(math.sqrt(int(args.num_images)))))
    png_path = out_dir / "hybrid_ar_atoms_diffusion_coeffs.png"
    pt_path = out_dir / "hybrid_ar_atoms_diffusion_coeffs.pt"
    save_image(images, png_path, nrow=nrow, normalize=True, value_range=(-1.0, 1.0))
    torch.save(
        {
            "atom_ids": atom_grid.detach().cpu(),
            "coeffs": coeff_grid.detach().cpu(),
            "images": images,
            "shape": (H, W, D),
            "token_cache": str(token_cache_path),
            "ar_checkpoint": str(args.ar_checkpoint.expanduser().resolve()),
            "diffusion_checkpoint": str(args.diffusion_checkpoint.expanduser().resolve()),
        },
        pt_path,
    )
    print(f"Saved hybrid sample image: {png_path}")
    print(f"Saved hybrid sample tensor payload: {pt_path}")


if __name__ == "__main__":
    main()
