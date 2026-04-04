"""Sample images from a maintained sparse-token prior and decode through LASER."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import torch
from torchvision.utils import save_image

from src.data.token_cache import load_token_cache
from src.models.laser import LASER
from src.models.mingpt_prior import MinGPTQuantizedPrior, MinGPTQuantizedPriorConfig
from src.models.sparse_token_prior import (
    SparseTokenPriorModule,
    build_sparse_prior_from_hparams,
    token_cache_grid_shape,
)
from src.models.spatial_prior import SpatialDepthPrior, SpatialDepthPriorConfig
from src.stage2_paths import infer_latest_stage1_checkpoint, infer_latest_stage2_checkpoint, infer_latest_token_cache


def _safe_torch_load(path: str | Path):
    resolved = Path(path).expanduser().resolve()
    try:
        return torch.load(resolved, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(resolved, map_location="cpu")


def _checkpoint_state_dict(payload: dict) -> dict:
    state_dict = payload.get("state_dict") if isinstance(payload, dict) else None
    if not isinstance(state_dict, dict):
        raise ValueError("Expected a Lightning checkpoint with a state_dict")
    return state_dict


def _checkpoint_hparams(payload: dict) -> dict:
    hparams = payload.get("hyper_parameters") if isinstance(payload, dict) else None
    return hparams if isinstance(hparams, dict) else {}


def _count_indexed_modules(state_dict: dict, prefix: str) -> int:
    indices = set()
    for key in state_dict:
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix) :]
        head = suffix.split(".", 1)[0]
        if head.isdigit():
            indices.add(int(head))
    return len(indices)


def _first_matching_key(state_dict: dict, prefix: str, suffix: str) -> str:
    for key in state_dict:
        if key.startswith(prefix) and key.endswith(suffix):
            return key
    raise KeyError(f"Could not find a state_dict key matching prefix={prefix!r} suffix={suffix!r}")


def _resolve_prior_architecture(payload: dict, override: str) -> str:
    if override != "auto":
        return str(override)
    hparams = _checkpoint_hparams(payload)
    arch = hparams.get("prior_architecture")
    if arch in {"spatial_depth", "mingpt"}:
        return str(arch)
    state_dict = _checkpoint_state_dict(payload)
    if any(key.startswith("prior.spatial_blocks.") for key in state_dict):
        return "spatial_depth"
    if "prior.pos_emb" in state_dict:
        return "mingpt"
    raise ValueError(
        "Could not infer prior architecture from checkpoint. "
        "Pass --architecture=spatial_depth or --architecture=mingpt."
    )


def _resolve_coeff_bin_values(
    *,
    coeff_vocab_size: int,
    cache: dict,
    stage2_hparams: dict,
    coeff_max_override: Optional[float],
    coeff_quantization_override: Optional[str],
    coeff_mu_override: Optional[float],
) -> torch.Tensor:
    meta = cache.get("meta", {}) if isinstance(cache, dict) else {}

    for container in (meta, cache, stage2_hparams):
        raw_values = container.get("coeff_bin_values") if isinstance(container, dict) else None
        if raw_values is not None:
            values = torch.as_tensor(raw_values, dtype=torch.float32).reshape(-1)
            if values.numel() != int(coeff_vocab_size):
                raise ValueError(
                    "coeff_bin_values length must match coeff_vocab_size, "
                    f"got {values.numel()} vs {int(coeff_vocab_size)}"
                )
            return values

    coeff_max = (
        float(coeff_max_override)
        if coeff_max_override is not None
        else meta.get("coef_max", stage2_hparams.get("prior_coeff_max"))
    )
    coeff_quantization = (
        str(coeff_quantization_override).strip().lower()
        if coeff_quantization_override is not None
        else str(meta.get("coef_quantization", "uniform")).strip().lower()
    )
    coeff_mu = (
        float(coeff_mu_override)
        if coeff_mu_override is not None
        else float(meta.get("coef_mu", 0.0))
    )

    if coeff_max is None:
        raise ValueError(
            "Could not resolve coefficient bin values from the token cache or stage-2 checkpoint. "
            "Pass --coeff-max, and --coeff-quantization/--coeff-mu if needed."
        )

    if coeff_quantization == "uniform":
        return torch.linspace(-float(coeff_max), float(coeff_max), steps=int(coeff_vocab_size), dtype=torch.float32)
    if coeff_quantization != "mu_law":
        raise ValueError(
            "coeff_quantization must be 'uniform' or 'mu_law', got "
            f"{coeff_quantization!r}"
        )
    if coeff_mu <= 0.0:
        raise ValueError(f"coeff_mu must be > 0 for mu-law quantization, got {coeff_mu}")
    if int(coeff_vocab_size) == 1:
        return torch.zeros(1, dtype=torch.float32)
    z = torch.linspace(-1.0, 1.0, steps=int(coeff_vocab_size), dtype=torch.float32)
    decoded = torch.sign(z) * (torch.expm1(z.abs() * math.log1p(float(coeff_mu))) / float(coeff_mu))
    return decoded * float(coeff_max)


def _resolve_latent_hw(stage1: LASER, cache: dict, image_size_override: Optional[int]) -> Optional[tuple[int, int]]:
    token_h, token_w, _ = token_cache_grid_shape(cache)
    meta = cache.get("meta", {}) if isinstance(cache, dict) else {}
    if not getattr(stage1.bottleneck, "patch_based", False):
        return int(token_h), int(token_w)
    latent_hw = meta.get("latent_hw")
    if isinstance(latent_hw, (tuple, list)) and len(latent_hw) == 2:
        return int(latent_hw[0]), int(latent_hw[1])
    image_size = int(image_size_override or meta.get("image_size") or 0)
    if image_size <= 0:
        raise ValueError(
            "Patch-based decode requires the original image size. "
            "Pass --image-size or use a token cache whose meta includes image_size."
        )
    return stage1.infer_latent_hw((image_size, image_size))


def _build_prior(
    *,
    payload: dict,
    cache: dict,
    architecture: str,
    atom_vocab_size: int,
    coeff_vocab_size: int,
    coeff_bin_values: torch.Tensor,
    n_heads_override: Optional[int],
    dropout_override: Optional[float],
) -> torch.nn.Module:
    state_dict = _checkpoint_state_dict(payload)
    hparams = _checkpoint_hparams(payload)
    H, W, D = token_cache_grid_shape(cache)
    d_model = int(state_dict["prior.token_emb.weight"].shape[1])
    d_ff_key = (
        _first_matching_key(state_dict, "prior.blocks.", ".ffn.0.weight")
        if architecture == "mingpt"
        else _first_matching_key(state_dict, "prior.spatial_blocks.", ".ffn.0.weight")
    )
    d_ff = int(
        hparams.get("prior_d_ff")
        or state_dict[d_ff_key].shape[0]
    )
    n_heads = int(n_heads_override or hparams.get("prior_n_heads") or 8)
    dropout = float(dropout_override if dropout_override is not None else hparams.get("prior_dropout", 0.0))
    total_vocab_size = int(state_dict["prior.token_head.weight"].shape[0])

    if architecture == "spatial_depth":
        n_spatial_layers = int(
            hparams.get("prior_n_spatial_layers") or _count_indexed_modules(state_dict, "prior.spatial_blocks.")
        )
        n_depth_layers = int(
            hparams.get("prior_n_depth_layers") or _count_indexed_modules(state_dict, "prior.depth_blocks.")
        )
        n_global_spatial_tokens = int(
            hparams.get("prior_n_global_spatial_tokens")
            or (
                state_dict["prior.global_spatial_tokens"].shape[1]
                if "prior.global_spatial_tokens" in state_dict
                else 0
            )
        )
        prior = SpatialDepthPrior(
            SpatialDepthPriorConfig(
                vocab_size=total_vocab_size,
                atom_vocab_size=int(atom_vocab_size),
                coeff_vocab_size=int(coeff_vocab_size),
                coeff_bin_values=coeff_bin_values,
                H=int(H),
                W=int(W),
                D=int(D),
                real_valued_coeffs=False,
                d_model=d_model,
                n_heads=n_heads,
                n_spatial_layers=n_spatial_layers,
                n_depth_layers=n_depth_layers,
                n_global_spatial_tokens=n_global_spatial_tokens,
                d_ff=d_ff,
                dropout=dropout,
                coeff_max=float(hparams.get("prior_coeff_max", coeff_bin_values.abs().max().item())),
            )
        )
    elif architecture == "mingpt":
        n_layers = int(hparams.get("prior_n_layers") or _count_indexed_modules(state_dict, "prior.blocks."))
        prior = MinGPTQuantizedPrior(
            MinGPTQuantizedPriorConfig(
                vocab_size=total_vocab_size,
                H=int(H),
                W=int(W),
                D=int(D),
                atom_vocab_size=int(atom_vocab_size),
                coeff_vocab_size=int(coeff_vocab_size),
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                d_ff=d_ff,
                dropout=dropout,
            )
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture!r}")
    return prior


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample images from a trained sparse-token prior.")
    parser.add_argument("--stage1-checkpoint", type=Path, default=None, help="Path to the LASER Lightning checkpoint.")
    parser.add_argument("--stage2-checkpoint", type=Path, default=None, help="Path to the sparse prior Lightning checkpoint.")
    parser.add_argument("--token-cache", type=Path, default=None, help="Path to the stage-2 token cache used for training.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to write samples into.")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0, help="0 disables top-k truncation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'cpu', or a CUDA device like 'cuda:0'.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--ar-output-dir", type=Path, default=Path("outputs/ar"))
    parser.add_argument(
        "--architecture",
        type=str,
        default="auto",
        choices=["auto", "spatial_depth", "mingpt"],
        help="Prior architecture. 'auto' infers it from the checkpoint when possible.",
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=None,
        help="Override the stage-2 attention head count when the checkpoint does not store it.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Optional prior dropout override. Only affects module construction in eval mode.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Original input image size. Needed for patch-based decode when the token cache meta does not store it.",
    )
    parser.add_argument(
        "--coeff-max",
        type=float,
        default=None,
        help="Coefficient dequantization range. Required when the token cache omits quantization metadata.",
    )
    parser.add_argument(
        "--coeff-quantization",
        type=str,
        default=None,
        help="Coefficient quantization mode: 'uniform' or 'mu_law'.",
    )
    parser.add_argument(
        "--coeff-mu",
        type=float,
        default=None,
        help="Mu parameter for mu-law coefficient dequantization.",
    )
    parser.add_argument(
        "--nrow",
        type=int,
        default=0,
        help="Grid columns for the saved sample sheet. 0 uses sqrt(num_samples).",
    )
    parser.add_argument(
        "--coeff-temperature",
        type=float,
        default=None,
        help="Optional override for real-valued coefficient sampling temperature.",
    )
    parser.add_argument(
        "--coeff-sample-mode",
        type=str,
        default=None,
        choices=["gaussian", "mean"],
        help="Optional override for real-valued coefficient sampling mode.",
    )
    return parser.parse_args()


@torch.no_grad()
def main():
    args = _parse_args()
    torch.manual_seed(int(args.seed))

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu"
    )

    if args.stage2_checkpoint is None:
        inferred_stage2 = infer_latest_stage2_checkpoint(ar_output_dir=args.ar_output_dir)
        if inferred_stage2 is None:
            raise FileNotFoundError(
                f"Could not infer a stage-2 checkpoint under {(Path(args.ar_output_dir).expanduser().resolve() / 'checkpoints')}"
            )
        args.stage2_checkpoint = inferred_stage2
        print(f"Inferred stage2 checkpoint: {args.stage2_checkpoint}")

    stage2_payload = _safe_torch_load(args.stage2_checkpoint)
    stage2_hparams = _checkpoint_hparams(stage2_payload)

    if args.token_cache is None:
        hparams_token_cache = stage2_hparams.get("token_cache_path")
        if hparams_token_cache:
            candidate = Path(str(hparams_token_cache)).expanduser().resolve()
            if candidate.exists():
                args.token_cache = candidate
        if args.token_cache is None:
            inferred_cache = infer_latest_token_cache(ar_output_dir=args.ar_output_dir)
            if inferred_cache is None:
                raise FileNotFoundError(
                    f"Could not infer a token cache under {(Path(args.ar_output_dir).expanduser().resolve() / 'token_cache')}"
                )
            args.token_cache = inferred_cache
        print(f"Inferred token cache: {args.token_cache}")

    cache = load_token_cache(args.token_cache)
    real_valued_cache = cache.get("coeffs_flat") is not None
    token_h, token_w, token_depth = token_cache_grid_shape(cache)

    if args.stage1_checkpoint is None:
        stage1_from_cache = cache.get("meta", {}).get("stage1_checkpoint")
        if stage1_from_cache:
            candidate = Path(str(stage1_from_cache)).expanduser().resolve()
            if candidate.exists():
                args.stage1_checkpoint = candidate
        if args.stage1_checkpoint is None:
            inferred_stage1 = infer_latest_stage1_checkpoint(output_root=args.output_root, model_type="laser")
            if inferred_stage1 is None:
                raise FileNotFoundError(
                    f"Could not infer a LASER checkpoint under {(Path(args.output_root).expanduser().resolve() / 'checkpoints')}"
                )
            args.stage1_checkpoint = inferred_stage1
        print(f"Inferred stage1 checkpoint: {args.stage1_checkpoint}")

    if args.output_dir is None:
        run_tag = Path(args.stage2_checkpoint).resolve().parent.name
        args.output_dir = Path(args.ar_output_dir).expanduser().resolve() / "generated" / run_tag
        print(f"Defaulting output_dir to: {args.output_dir}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stage1 = LASER.load_from_checkpoint(args.stage1_checkpoint, map_location="cpu")
    stage1.eval().to(device)

    build_hparams = dict(stage2_hparams)
    if args.architecture != "auto":
        build_hparams["prior_architecture"] = str(args.architecture)
    if args.n_heads is not None:
        build_hparams["prior_n_heads"] = int(args.n_heads)
    if args.dropout is not None:
        build_hparams["prior_dropout"] = float(args.dropout)
    if args.coeff_max is not None:
        build_hparams["prior_coeff_max"] = float(args.coeff_max)

    prior = build_sparse_prior_from_hparams(cache, hparams=build_hparams)
    stage2 = SparseTokenPriorModule.load_from_checkpoint(
        args.stage2_checkpoint,
        map_location="cpu",
        prior=prior,
    )
    stage2.eval().to(device)

    latent_hw = _resolve_latent_hw(stage1, cache, args.image_size)
    top_k = None if int(args.top_k) <= 0 else int(args.top_k)

    # Pre-resolve coeff_bin_values once for the quantized path.
    if not real_valued_cache:
        coeff_bin_values = getattr(prior.cfg, "coeff_bin_values", None)
        if coeff_bin_values is None:
            coeff_vocab_size = getattr(prior.cfg, "coeff_vocab_size", None)
            coeff_bin_values = _resolve_coeff_bin_values(
                coeff_vocab_size=int(coeff_vocab_size or 0),
                cache=cache,
                stage2_hparams=build_hparams,
                coeff_max_override=args.coeff_max,
                coeff_quantization_override=args.coeff_quantization,
                coeff_mu_override=args.coeff_mu,
            )

    generated_images = []
    generated_tokens = []
    generated_coeffs = []
    remaining = int(args.num_samples)
    while remaining > 0:
        cur_batch = min(int(args.batch_size), remaining)
        if real_valued_cache:
            atom_ids, coeffs = stage2.generate_sparse_codes(
                cur_batch,
                temperature=float(args.temperature),
                top_k=top_k,
                coeff_temperature=args.coeff_temperature,
                coeff_sample_mode=args.coeff_sample_mode,
            )
            atom_grid = atom_ids.view(cur_batch, token_h, token_w, token_depth)
            coeff_grid = coeffs.view(cur_batch, token_h, token_w, token_depth)
            images = stage1.decode_from_atoms_and_coeffs(
                atom_grid.to(device=device, dtype=torch.long),
                coeff_grid.to(device=device, dtype=torch.float32),
                latent_hw=latent_hw,
            )
            generated_tokens.append(atom_grid.cpu())
            generated_coeffs.append(coeff_grid.cpu())
        else:
            token_grid = stage2.generate_tokens(
                cur_batch,
                temperature=float(args.temperature),
                top_k=top_k,
            ).view(cur_batch, token_h, token_w, token_depth)
            images = stage1.decode_from_tokens(
                token_grid.to(device=device, dtype=torch.long),
                latent_hw=latent_hw,
                atom_vocab_size=getattr(prior.cfg, "atom_vocab_size", None),
                coeff_vocab_size=getattr(prior.cfg, "coeff_vocab_size", None),
                coeff_bin_values=torch.as_tensor(coeff_bin_values, dtype=torch.float32, device=device),
            )
            generated_tokens.append(token_grid.cpu())
        generated_images.append(images.cpu())
        remaining -= cur_batch

    tokens = torch.cat(generated_tokens, dim=0)
    images = torch.cat(generated_images, dim=0)
    nrow = int(args.nrow) if int(args.nrow) > 0 else max(1, int(math.sqrt(images.size(0))))
    save_image(
        images,
        output_dir / "samples.png",
        nrow=nrow,
        normalize=True,
        value_range=(-1.0, 1.0),
    )
    save_image(
        images,
        output_dir / "samples_autocontrast.png",
        nrow=nrow,
        normalize=True,
        scale_each=True,
    )
    payload = {
        "shape": (token_h, token_w, token_depth),
        "latent_hw": latent_hw,
        "stage1_checkpoint": str(args.stage1_checkpoint.expanduser().resolve()),
        "stage2_checkpoint": str(args.stage2_checkpoint.expanduser().resolve()),
        "token_cache": str(args.token_cache.expanduser().resolve()),
    }
    if real_valued_cache:
        payload["atom_ids"] = tokens
        payload["coeffs"] = torch.cat(generated_coeffs, dim=0)
    else:
        payload["tokens"] = tokens
    torch.save(payload, output_dir / "samples.pt")
    print(f"Saved {images.size(0)} samples to {output_dir}")


if __name__ == "__main__":
    main()
