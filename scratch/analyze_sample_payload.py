#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


def _entropy_from_counts(counts: torch.Tensor) -> float:
    counts = counts.to(torch.float64)
    total = counts.sum()
    if not torch.isfinite(total) or total.item() <= 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float((-(probs * probs.log())).sum().item())


def _js_divergence_from_counts(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    lhs = lhs.to(torch.float64)
    rhs = rhs.to(torch.float64)
    lhs_total = lhs.sum()
    rhs_total = rhs.sum()
    if lhs_total.item() <= 0 or rhs_total.item() <= 0:
        return 0.0
    p = lhs / lhs_total
    q = rhs / rhs_total
    m = 0.5 * (p + q)
    mask_p = p > 0
    mask_q = q > 0
    kl_pm = (p[mask_p] * (p[mask_p] / m[mask_p]).log()).sum()
    kl_qm = (q[mask_q] * (q[mask_q] / m[mask_q]).log()).sum()
    return float((0.5 * (kl_pm + kl_qm)).item())


def _distinct_count_stats(atom_ids: torch.Tensor) -> Dict[str, float]:
    sorted_atoms, _ = atom_ids.to(torch.long).sort(dim=-1)
    distinct = 1 + (sorted_atoms[..., 1:] != sorted_atoms[..., :-1]).sum(dim=-1)
    distinct = distinct.to(torch.float32)
    return {
        "mean": float(distinct.mean().item()),
        "std": float(distinct.std(unbiased=False).item()),
        "min": int(distinct.min().item()),
        "max": int(distinct.max().item()),
    }


def _topk_from_counts(counts: torch.Tensor, k: int = 16, *, offset: int = 0) -> Dict[str, Any]:
    counts = counts.to(torch.float64)
    total = counts.sum()
    if total.item() <= 0:
        return {"indices": [], "counts": [], "probs": []}
    k = max(1, min(int(k), int(counts.numel())))
    vals, idx = torch.topk(counts, k=k)
    probs = vals / total
    return {
        "indices": [int(offset + x) for x in idx.tolist()],
        "counts": [float(x) for x in vals.tolist()],
        "probs": [float(x) for x in probs.tolist()],
    }


def _tensor_shape(tensor: Optional[torch.Tensor]) -> Optional[List[int]]:
    if tensor is None:
        return None
    return [int(x) for x in tensor.shape]


def _payload_token_stats(
    *,
    name: str,
    tokens_flat: torch.Tensor,
    coeffs_flat: Optional[torch.Tensor],
    T: int,
    D: int,
    atom_vocab_size: int,
    coeff_vocab_size: Optional[int],
    real_valued_coeffs: bool,
) -> Dict[str, Any]:
    tokens = tokens_flat.view(tokens_flat.size(0), T, D).to(torch.long)
    result: Dict[str, Any] = {
        "name": name,
        "batch_size": int(tokens.size(0)),
        "tokens_shape": _tensor_shape(tokens),
        "tokens_unique": int(tokens.unique().numel()),
        "tokens_min": int(tokens.min().item()),
        "tokens_max": int(tokens.max().item()),
    }

    if not real_valued_coeffs:
        atom_ids = tokens[..., 0::2]
        coeff_tokens = tokens[..., 1::2]
        coeff_bins = (coeff_tokens - int(atom_vocab_size)).to(torch.long)
        atom_counts = torch.bincount(atom_ids.reshape(-1), minlength=int(atom_vocab_size))
        coeff_minlength = int(coeff_vocab_size) if coeff_vocab_size is not None else int(coeff_bins.max().item()) + 1
        coeff_counts = torch.bincount(coeff_bins.reshape(-1), minlength=coeff_minlength)
        result["atom"] = {
            "shape": _tensor_shape(atom_ids),
            "unique": int(atom_ids.unique().numel()),
            "entropy_nats": _entropy_from_counts(atom_counts),
            "distinct_per_site": _distinct_count_stats(atom_ids),
            "top_bins": _topk_from_counts(atom_counts),
        }
        result["coeff"] = {
            "shape": _tensor_shape(coeff_bins),
            "unique": int(coeff_bins.unique().numel()),
            "entropy_nats": _entropy_from_counts(coeff_counts),
            "min": int(coeff_bins.min().item()),
            "max": int(coeff_bins.max().item()),
            "top_bins": _topk_from_counts(coeff_counts),
        }
    else:
        atom_counts = torch.bincount(tokens.reshape(-1), minlength=int(atom_vocab_size))
        result["atom"] = {
            "shape": _tensor_shape(tokens),
            "unique": int(tokens.unique().numel()),
            "entropy_nats": _entropy_from_counts(atom_counts),
            "distinct_per_site": _distinct_count_stats(tokens),
            "top_bins": _topk_from_counts(atom_counts),
        }
        if coeffs_flat is not None:
            coeffs = coeffs_flat.view(coeffs_flat.size(0), T, D).to(torch.float32)
            result["coeff"] = {
                "shape": _tensor_shape(coeffs),
                "mean": float(coeffs.mean().item()),
                "std": float(coeffs.std(unbiased=False).item()),
                "min": float(coeffs.min().item()),
                "max": float(coeffs.max().item()),
            }
    result["images"] = None
    return result


def _image_stats(images: Optional[torch.Tensor]) -> Optional[Dict[str, float]]:
    if images is None:
        return None
    x = images.to(torch.float32)
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def _prompt_agreement(
    *,
    sample_payload: Dict[str, Any],
    prompt_payload: Dict[str, Any],
    T: int,
    D: int,
    prompt_spatial_steps: int,
    atom_vocab_size: int,
    real_valued_coeffs: bool,
) -> Dict[str, Any]:
    sample_tokens = sample_payload["tokens_flat"].view(-1, T, D).to(torch.long)
    prompt_tokens = prompt_payload["tokens_flat"].view(-1, T, D).to(torch.long)
    if sample_tokens.shape != prompt_tokens.shape:
        raise ValueError(
            f"Sample/prompt token shape mismatch: {tuple(sample_tokens.shape)} vs {tuple(prompt_tokens.shape)}"
        )

    full_token_match = (sample_tokens == prompt_tokens).to(torch.float32)
    full_step_match = full_token_match.all(dim=-1).to(torch.float32)
    result: Dict[str, Any] = {
        "prompt_spatial_steps": int(prompt_spatial_steps),
        "full_token_match_mean": float(full_token_match.mean().item()),
        "full_step_match_mean": float(full_step_match.mean().item()),
        "full_step_match_by_spatial_index": [float(x) for x in full_step_match.mean(dim=0).tolist()],
    }
    if prompt_spatial_steps > 0:
        result["prefix_step_match_mean"] = float(full_step_match[:, :prompt_spatial_steps].mean().item())
    if prompt_spatial_steps < T:
        result["suffix_step_match_mean"] = float(full_step_match[:, prompt_spatial_steps:].mean().item())

    if not real_valued_coeffs:
        sample_atoms = sample_tokens[..., 0::2]
        prompt_atoms = prompt_tokens[..., 0::2]
        sample_coeff = sample_tokens[..., 1::2]
        prompt_coeff = prompt_tokens[..., 1::2]
        atom_step_match = (sample_atoms == prompt_atoms).all(dim=-1).to(torch.float32)
        coeff_step_match = (sample_coeff == prompt_coeff).all(dim=-1).to(torch.float32)
        result["atom_step_match_mean"] = float(atom_step_match.mean().item())
        result["coeff_step_match_mean"] = float(coeff_step_match.mean().item())
        result["atom_step_match_by_spatial_index"] = [float(x) for x in atom_step_match.mean(dim=0).tolist()]
        result["coeff_step_match_by_spatial_index"] = [float(x) for x in coeff_step_match.mean(dim=0).tolist()]

    sample_images = sample_payload.get("images")
    prompt_images = prompt_payload.get("images")
    if sample_images is not None and prompt_images is not None and tuple(sample_images.shape) == tuple(prompt_images.shape):
        diff = sample_images.to(torch.float32) - prompt_images.to(torch.float32)
        mse = (diff * diff).mean(dim=(1, 2, 3))
        result["image_mse_mean"] = float(mse.mean().item())
        result["image_mse_min"] = float(mse.min().item())
        result["image_mse_max"] = float(mse.max().item())

    return result


def _distribution_comparison(
    *,
    cache_tokens_flat: torch.Tensor,
    sample_tokens_flat: torch.Tensor,
    T: int,
    D: int,
    atom_vocab_size: int,
    coeff_vocab_size: Optional[int],
    real_valued_coeffs: bool,
) -> Dict[str, Any]:
    cache_tokens = cache_tokens_flat.view(cache_tokens_flat.size(0), T, D).to(torch.long)
    sample_tokens = sample_tokens_flat.view(sample_tokens_flat.size(0), T, D).to(torch.long)
    result: Dict[str, Any] = {}
    if not real_valued_coeffs:
        cache_atoms = cache_tokens[..., 0::2]
        sample_atoms = sample_tokens[..., 0::2]
        cache_coeff = (cache_tokens[..., 1::2] - int(atom_vocab_size)).to(torch.long)
        sample_coeff = (sample_tokens[..., 1::2] - int(atom_vocab_size)).to(torch.long)
        cache_atom_counts = torch.bincount(cache_atoms.reshape(-1), minlength=int(atom_vocab_size))
        sample_atom_counts = torch.bincount(sample_atoms.reshape(-1), minlength=int(atom_vocab_size))
        coeff_minlength = int(coeff_vocab_size) if coeff_vocab_size is not None else int(
            max(cache_coeff.max().item(), sample_coeff.max().item()) + 1
        )
        cache_coeff_counts = torch.bincount(cache_coeff.reshape(-1), minlength=coeff_minlength)
        sample_coeff_counts = torch.bincount(sample_coeff.reshape(-1), minlength=coeff_minlength)
        result["atom_js_divergence_nats"] = _js_divergence_from_counts(cache_atom_counts, sample_atom_counts)
        result["coeff_js_divergence_nats"] = _js_divergence_from_counts(cache_coeff_counts, sample_coeff_counts)
        result["cache_atom_entropy_nats"] = _entropy_from_counts(cache_atom_counts)
        result["sample_atom_entropy_nats"] = _entropy_from_counts(sample_atom_counts)
        result["cache_coeff_entropy_nats"] = _entropy_from_counts(cache_coeff_counts)
        result["sample_coeff_entropy_nats"] = _entropy_from_counts(sample_coeff_counts)
        result["cache_distinct_atoms_per_site"] = _distinct_count_stats(cache_atoms)
        result["sample_distinct_atoms_per_site"] = _distinct_count_stats(sample_atoms)
    else:
        cache_counts = torch.bincount(cache_tokens.reshape(-1), minlength=int(atom_vocab_size))
        sample_counts = torch.bincount(sample_tokens.reshape(-1), minlength=int(atom_vocab_size))
        result["atom_js_divergence_nats"] = _js_divergence_from_counts(cache_counts, sample_counts)
        result["cache_atom_entropy_nats"] = _entropy_from_counts(cache_counts)
        result["sample_atom_entropy_nats"] = _entropy_from_counts(sample_counts)
        result["cache_distinct_atoms_per_site"] = _distinct_count_stats(cache_tokens)
        result["sample_distinct_atoms_per_site"] = _distinct_count_stats(sample_tokens)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze saved sample payloads against a run's token cache.")
    parser.add_argument("--sample_dir", type=Path, required=True, help="Directory containing manifest.json and samples.pt.")
    parser.add_argument("--token_cache", type=Path, default=None, help="Optional override for tokens_cache.pt.")
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Optional output path. Defaults to <sample_dir>/analysis.json.",
    )
    args = parser.parse_args()

    sample_dir = args.sample_dir.expanduser().resolve()
    manifest_path = sample_dir / "manifest.json"
    sample_payload_path = sample_dir / "samples.pt"
    prompt_payload_path = sample_dir / "prompt_source.pt"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not sample_payload_path.is_file():
        raise FileNotFoundError(f"Missing sample payload: {sample_payload_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    token_cache_path = (
        args.token_cache.expanduser().resolve()
        if args.token_cache is not None
        else Path(manifest["token_cache"]).expanduser().resolve()
    )
    if not token_cache_path.is_file():
        raise FileNotFoundError(f"Missing token cache: {token_cache_path}")

    H = int(manifest["token_grid_shape"]["H"])
    W = int(manifest["token_grid_shape"]["W"])
    D = int(manifest["token_grid_shape"]["D"])
    T = H * W
    transformer_cfg = manifest["transformer_config"]
    atom_vocab_size = int(transformer_cfg["atom_vocab_size"])
    coeff_vocab_size = transformer_cfg.get("coeff_vocab_size")
    coeff_vocab_size = None if coeff_vocab_size is None else int(coeff_vocab_size)
    real_valued_coeffs = bool(transformer_cfg["real_valued_coeffs"])

    sample_payload = torch.load(sample_payload_path, map_location="cpu")
    token_cache = torch.load(token_cache_path, map_location="cpu")
    prompt_payload = torch.load(prompt_payload_path, map_location="cpu") if prompt_payload_path.is_file() else None

    analysis: Dict[str, Any] = {
        "sample_dir": str(sample_dir),
        "manifest_path": str(manifest_path),
        "sample_payload_path": str(sample_payload_path),
        "prompt_payload_path": (str(prompt_payload_path) if prompt_payload is not None else None),
        "token_cache_path": str(token_cache_path),
        "token_grid_shape": {"H": H, "W": W, "D": D, "T": T},
        "transformer_config": {
            "atom_vocab_size": atom_vocab_size,
            "coeff_vocab_size": coeff_vocab_size,
            "real_valued_coeffs": real_valued_coeffs,
            "autoregressive_coeffs": bool(transformer_cfg["autoregressive_coeffs"]),
        },
        "sampling": manifest.get("sampling", {}),
    }

    analysis["sample"] = _payload_token_stats(
        name="sample",
        tokens_flat=sample_payload["tokens_flat"],
        coeffs_flat=sample_payload.get("coeffs_flat"),
        T=T,
        D=D,
        atom_vocab_size=atom_vocab_size,
        coeff_vocab_size=coeff_vocab_size,
        real_valued_coeffs=real_valued_coeffs,
    )
    analysis["sample"]["images"] = _image_stats(sample_payload.get("images"))

    analysis["cache"] = _payload_token_stats(
        name="cache",
        tokens_flat=token_cache["tokens_flat"],
        coeffs_flat=token_cache.get("coeffs_flat"),
        T=T,
        D=D,
        atom_vocab_size=atom_vocab_size,
        coeff_vocab_size=coeff_vocab_size,
        real_valued_coeffs=real_valued_coeffs,
    )

    analysis["distribution"] = _distribution_comparison(
        cache_tokens_flat=token_cache["tokens_flat"],
        sample_tokens_flat=sample_payload["tokens_flat"],
        T=T,
        D=D,
        atom_vocab_size=atom_vocab_size,
        coeff_vocab_size=coeff_vocab_size,
        real_valued_coeffs=real_valued_coeffs,
    )

    if prompt_payload is not None:
        analysis["prompt_source"] = _payload_token_stats(
            name="prompt_source",
            tokens_flat=prompt_payload["tokens_flat"],
            coeffs_flat=prompt_payload.get("coeffs_flat"),
            T=T,
            D=D,
            atom_vocab_size=atom_vocab_size,
            coeff_vocab_size=coeff_vocab_size,
            real_valued_coeffs=real_valued_coeffs,
        )
        analysis["prompt_source"]["images"] = _image_stats(prompt_payload.get("images"))
        prompt_steps = int(manifest.get("prompt", {}).get("prompt_spatial_steps", 0))
        analysis["prompt_agreement"] = _prompt_agreement(
            sample_payload=sample_payload,
            prompt_payload=prompt_payload,
            T=T,
            D=D,
            prompt_spatial_steps=prompt_steps,
            atom_vocab_size=atom_vocab_size,
            real_valued_coeffs=real_valued_coeffs,
        )

    output_json = (
        args.output_json.expanduser().resolve()
        if args.output_json is not None
        else sample_dir / "analysis.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    print(f"[Analyze] wrote {output_json}")
    if "distribution" in analysis:
        dist = analysis["distribution"]
        print(
            "[Analyze] atom_js_divergence_nats="
            f"{dist.get('atom_js_divergence_nats', 0.0):.6f} "
            "sample_atom_entropy_nats="
            f"{dist.get('sample_atom_entropy_nats', 0.0):.6f} "
            "cache_atom_entropy_nats="
            f"{dist.get('cache_atom_entropy_nats', 0.0):.6f}"
        )
    if "prompt_agreement" in analysis:
        pa = analysis["prompt_agreement"]
        print(
            "[Analyze] prompt prefix_step_match_mean="
            f"{pa.get('prefix_step_match_mean', 0.0):.6f} "
            "suffix_step_match_mean="
            f"{pa.get('suffix_step_match_mean', 0.0):.6f} "
            "image_mse_mean="
            f"{pa.get('image_mse_mean', 0.0):.6f}"
        )


if __name__ == "__main__":
    main()
