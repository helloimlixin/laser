"""Compatibility helpers for maintained stage-2 training/sampling."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Set, Tuple, Union

import torch

from .checkpoint_io import build_lightning_module, extract_state_dict, load_torch_payload
from .models.rq_ae import PatchDictionaryLearningTokenized, RQAE
from .models.vqvae import VQVAE
from .stage2_paths import infer_latest_stage1_checkpoint


@dataclass
class Stage1DecodeBundle:
    kind: str
    model: torch.nn.Module
    checkpoint_path: Path
    latent_hw: Optional[Tuple[int, int]]
    coeff_vocab_size: int
    coeff_bin_values: Optional[torch.Tensor]


def _bundle_patch_layout(bundle: Stage1DecodeBundle) -> tuple[bool, int]:
    model = bundle.model
    bottleneck = getattr(model, "bottleneck", None)
    if bottleneck is not None:
        patch_based = getattr(bottleneck, "patch_based", None)
        if patch_based is None:
            # When the patch flag is not exposed, infer it the same way the
            # reconstruct path does: a patch bundle carries a ``patch_size``
            # (RQ patch bottlenecks set it; non-patch ones leave it unset).
            patch_based = (
                isinstance(bottleneck, PatchDictionaryLearningTokenized)
                or getattr(bottleneck, "patch_size", None) is not None
            )
        patch_stride = int(getattr(bottleneck, "patch_stride", 1) or 1)
        return bool(patch_based), max(1, patch_stride)

    hparams = getattr(model, "hparams", None)
    if hparams is not None:
        patch_based = bool(getattr(hparams, "patch_based", False))
        patch_stride = int(getattr(hparams, "patch_stride", 1) or 1)
        return patch_based, max(1, patch_stride)

    return False, 1


def _resolve_decode_latent_hw(
    bundle: Stage1DecodeBundle,
    *,
    token_hw: Tuple[int, int],
    latent_hw: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int]:
    token_h, token_w = int(token_hw[0]), int(token_hw[1])
    patch_based, patch_stride = _bundle_patch_layout(bundle)
    requested_hw = None if latent_hw is None else (int(latent_hw[0]), int(latent_hw[1]))

    if not patch_based:
        if requested_hw is not None:
            return requested_hw
        return (token_h, token_w)

    if requested_hw is not None and requested_hw != (token_h, token_w):
        return requested_hw

    bundle_hw = bundle.latent_hw
    if bundle_hw is not None:
        bundle_hw = (int(bundle_hw[0]), int(bundle_hw[1]))
        train_token_h = int(math.ceil(bundle_hw[0] / float(patch_stride)))
        train_token_w = int(math.ceil(bundle_hw[1] / float(patch_stride)))
        if (token_h, token_w) == (train_token_h, train_token_w):
            return bundle_hw

    return (token_h * patch_stride, token_w * patch_stride)

def _indexed_block_count(keys, prefix: str) -> int:
    import re

    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.")
    indices = set()
    for key in keys:
        match = pattern.match(key)
        if match is not None:
            indices.add(int(match.group(1)))
    return len(indices)


def _cache_sparse_depth(token_cache: dict, *, quantized_sparse_coeffs: bool) -> int:
    shape = token_cache.get("shape")
    if not isinstance(shape, (tuple, list)) or len(shape) != 3:
        return 0
    depth = int(shape[2])
    if not quantized_sparse_coeffs:
        return depth
    return max(1, depth // 2)


def _infer_rq_patch_layout(
    state_dict: dict,
    token_cache: dict,
    metadata: dict,
    *,
    embedding_dim: int,
    num_downsamples: int,
) -> tuple[bool, int, int, str]:
    patch_based = metadata.get("patch_based")
    dictionary_rows = int(state_dict["bottleneck.dictionary"].shape[0])
    inferred_patch_size = None
    if embedding_dim > 0 and dictionary_rows % embedding_dim == 0:
        patch_area = int(dictionary_rows // embedding_dim)
        if patch_area > 1:
            patch_size = int(round(math.sqrt(patch_area)))
            if patch_size * patch_size == patch_area:
                inferred_patch_size = patch_size
                if patch_based is None:
                    patch_based = True
        elif patch_based is None:
            patch_based = False
    if patch_based is None:
        patch_based = False
    patch_based = bool(patch_based)

    if patch_based:
        patch_size = int(metadata.get("patch_size", inferred_patch_size or 8))
        patch_stride = int(metadata.get("patch_stride", patch_size))
    else:
        patch_size = int(metadata.get("patch_size", 1))
        patch_stride = int(metadata.get("patch_stride", 1))

    patch_reconstruction = str(metadata.get("patch_reconstruction", "tile"))
    return patch_based, patch_size, patch_stride, patch_reconstruction


def _infer_channel_multipliers(
    state_dict: dict,
    *,
    num_hiddens: int,
    num_downsamples: int,
) -> Optional[Tuple[int, ...]]:
    mults = []
    for level in range(max(0, int(num_downsamples)) + 1):
        weight = state_dict.get(f"encoder.down.{level}.block.0.conv1.weight")
        if weight is None:
            return None
        out_channels = int(weight.shape[0])
        if num_hiddens <= 0 or out_channels % num_hiddens != 0:
            return None
        mults.append(out_channels // num_hiddens)
    return tuple(mults)


def _infer_rq_stage1_config(state_dict: dict, token_cache: dict) -> dict:
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
    num_downsamples = max(0, _indexed_block_count(state_dict.keys(), "encoder.down") - 1)
    num_residual_layers = max(1, _indexed_block_count(state_dict.keys(), "encoder.down.0.block"))
    encoder_norm_channels = int(state_dict["encoder.norm_out.weight"].shape[0])
    decoder_blocks_per_level = max(1, _indexed_block_count(state_dict.keys(), "decoder.up.0.block"))
    inferred_use_mid_attention = (
        "encoder.mid.attn_1.q.weight" in state_dict or "decoder.mid.attn_1.q.weight" in state_dict
    )
    pre_bottleneck_weight = state_dict.get("pre_bottleneck.weight")
    post_bottleneck_weight = state_dict.get("post_bottleneck.weight")
    meta_backbone_latent_channels = metadata.get("backbone_latent_channels")
    if pre_bottleneck_weight is not None:
        embedding_dim = int(pre_bottleneck_weight.shape[0])
        inferred_backbone_latent_channels = int(pre_bottleneck_weight.shape[1])
    elif post_bottleneck_weight is not None:
        embedding_dim = int(post_bottleneck_weight.shape[1])
        inferred_backbone_latent_channels = int(post_bottleneck_weight.shape[0])
    else:
        embedding_dim = int(state_dict["encoder.conv_out.weight"].shape[0])
        inferred_backbone_latent_channels = embedding_dim
    patch_based, patch_size, patch_stride, patch_reconstruction = _infer_rq_patch_layout(
        state_dict,
        token_cache,
        metadata,
        embedding_dim=embedding_dim,
        num_downsamples=num_downsamples,
    )
    inferred_sparsity_level = _cache_sparse_depth(
        token_cache,
        quantized_sparse_coeffs=bool(quantize_sparse_coeffs),
    )
    meta_channel_multipliers = metadata.get("channel_multipliers")
    if isinstance(meta_channel_multipliers, (tuple, list)):
        channel_multipliers = tuple(int(v) for v in meta_channel_multipliers)
    else:
        channel_multipliers = _infer_channel_multipliers(
            state_dict,
            num_hiddens=num_hiddens,
            num_downsamples=num_downsamples,
        )

    coeff_bin_centers = state_dict.get("bottleneck.coef_bin_centers")
    n_bins = int(coeff_bin_centers.shape[0]) if torch.is_tensor(coeff_bin_centers) else int(
        metadata.get("coeff_vocab_size") or metadata.get("n_bins") or 0
    )
    coef_max = float(coeff_bin_centers.abs().max().item()) if torch.is_tensor(coeff_bin_centers) else float(
        metadata.get("coef_max", 1.0)
    )

    return {
        "in_channels": int(state_dict["encoder.conv_in.weight"].shape[1]),
        "num_hiddens": num_hiddens,
        "num_downsamples": num_downsamples,
        "channel_multipliers": channel_multipliers,
        "num_residual_layers": num_residual_layers,
        "resolution": int(metadata.get("image_size", 128)),
        "backbone_latent_channels": int(
            meta_backbone_latent_channels
            if meta_backbone_latent_channels is not None
            else inferred_backbone_latent_channels
        ),
        "decoder_extra_residual_layers": int(
            metadata.get("decoder_extra_residual_layers", max(0, decoder_blocks_per_level - num_residual_layers))
        ),
        "use_mid_attention": bool(metadata.get("use_mid_attention", inferred_use_mid_attention)),
        "embedding_dim": embedding_dim,
        "num_embeddings": int(dictionary_shape[1]),
        "sparsity_level": int(metadata.get("sparsity_level", inferred_sparsity_level)),
        "commitment_cost": 0.25,
        "n_bins": max(1, n_bins),
        "coef_max": max(1e-6, coef_max),
        "coef_quantization": coef_quantization,
        "coef_mu": coef_mu,
        "quantize_sparse_coeffs": bool(quantize_sparse_coeffs),
        "patch_based": patch_based,
        "patch_size": patch_size,
        "patch_stride": patch_stride,
        "patch_reconstruction": patch_reconstruction,
        "variational_coeffs": bool(
            metadata.get(
                "variational_coeffs",
                "bottleneck.coeff_variational_atom_emb.weight" in state_dict,
            )
        ),
        # Renamed in May 2026 (A3). Caches written before the rename used
        # variational_coeff_kl_weight / variational_coeff_prior_std — those will
        # silently fall back to the defaults below; re-extract such caches against
        # the new stage-1 code if you need accurate values.
        "variational_coeff_refine_weight": float(metadata.get("variational_coeff_refine_weight", 0.0)),
        "variational_coeff_target_std": float(metadata.get("variational_coeff_target_std", 0.25)),
        "variational_coeff_min_std": float(metadata.get("variational_coeff_min_std", 0.01)),
    }


def _first_existing_path(candidates: Sequence[Union[Path, str, None]]) -> Optional[Path]:
    seen = set()  # type: Set[Path]
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = Path(candidate).expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return None


def resolve_stage1_checkpoint_from_token_cache(
    token_cache_path,
    *,
    meta: Optional[dict] = None,
    output_root="outputs",
) -> Optional[Path]:
    token_cache_path = Path(token_cache_path).expanduser().resolve()
    meta = meta or {}
    explicit = meta.get("stage1_checkpoint")

    candidates = [explicit]
    if token_cache_path.parent.name == "stage2":
        run_root = token_cache_path.parent.parent
        candidates.extend(
            [
                run_root / "stage1" / "last.ckpt",
                run_root / "stage1" / "ae_best.pt",
                run_root / "stage1" / "ae_last.pt",
            ]
        )

    for parent in token_cache_path.parents[:4]:
        candidates.extend(
            [
                parent / "stage1" / "last.ckpt",
                parent / "stage1" / "ae_best.pt",
                parent / "stage1" / "ae_last.pt",
            ]
        )

    candidates.append(infer_latest_stage1_checkpoint(output_root=output_root))
    return _first_existing_path(candidates)


def _coeff_codec_from_state_dict(state_dict: dict):
    coeff_bin_centers = state_dict.get("bottleneck.coef_bin_centers")
    if coeff_bin_centers is None:
        return 0, None
    coeff_bin_centers = torch.as_tensor(coeff_bin_centers, dtype=torch.float32).reshape(-1).cpu()
    return int(coeff_bin_centers.numel()), coeff_bin_centers


def _infer_lightning_stage1_type(payload, state_dict: dict) -> str:
    hparams = dict(payload.get("hyper_parameters", {}) or {}) if isinstance(payload, dict) else {}
    if "sparsity_level" in hparams or any(key.startswith("bottleneck.") for key in state_dict):
        return "laser"
    if "decay" in hparams or any(key.startswith("vector_quantizer.") for key in state_dict):
        return "vqvae"
    raise RuntimeError("Could not infer stage-1 Lightning module type from checkpoint metadata.")


def _infer_stage1_metadata_defaults(payload, cache: dict) -> dict:
    state_dict = extract_state_dict(payload)
    if not isinstance(state_dict, dict):
        return {}

    inferred = {}
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        hparams = dict(payload.get("hyper_parameters", {}) or {})
        inferred["stage1_model_type"] = _infer_lightning_stage1_type(payload, state_dict)
        for key in (
            "patch_based",
            "patch_size",
            "patch_stride",
            "patch_reconstruction",
            "variational_coeffs",
            "variational_coeff_target_std",
            "variational_coeff_min_std",
            "image_size",
        ):
            if hparams.get(key) is not None:
                inferred[key] = hparams.get(key)
        coeff_max = hparams.get("coeff_max", hparams.get("coef_max"))
        if coeff_max is not None:
            inferred["coeff_max"] = float(coeff_max)
            inferred["coef_max"] = float(coeff_max)
        return inferred

    rq_cfg = _infer_rq_stage1_config(state_dict, cache)
    inferred.update(
        {
            "patch_based": bool(rq_cfg.get("patch_based", False)),
            "patch_size": int(rq_cfg.get("patch_size", 8)),
            "patch_stride": int(rq_cfg.get("patch_stride", rq_cfg.get("patch_size", 8))),
            "patch_reconstruction": str(rq_cfg.get("patch_reconstruction", "tile")),
            "variational_coeffs": bool(rq_cfg.get("variational_coeffs", False)),
            "variational_coeff_target_std": float(rq_cfg.get("variational_coeff_target_std", 0.25)),
            "variational_coeff_min_std": float(rq_cfg.get("variational_coeff_min_std", 0.01)),
            "coef_max": float(rq_cfg.get("coef_max", 24.0)),
            "coeff_max": float(rq_cfg.get("coef_max", 24.0)),
            "quantize_sparse_coeffs": bool(rq_cfg.get("quantize_sparse_coeffs", False)),
        }
    )
    return inferred


def ensure_stage2_cache_metadata(
    cache: dict,
    *,
    token_cache_path,
    output_root="outputs",
) -> dict:
    if not isinstance(cache, dict):
        raise ValueError("cache must be a dict")

    enriched = dict(cache)
    meta = dict(cache.get("meta", {}) or {})
    is_real_valued = cache.get("coeffs_flat") is not None
    meta["quantize_sparse_coeffs"] = not bool(is_real_valued)
    stage1_checkpoint = resolve_stage1_checkpoint_from_token_cache(
        token_cache_path,
        meta=meta,
        output_root=output_root,
    )
    payload = None
    if stage1_checkpoint is not None:
        meta.setdefault("stage1_checkpoint", str(stage1_checkpoint))
        try:
            payload = load_torch_payload(stage1_checkpoint)
        except Exception:
            payload = None
    if payload is not None:
        for key, value in _infer_stage1_metadata_defaults(payload, cache).items():
            meta.setdefault(key, value)

    coeff_bin_values = meta.get("coeff_bin_values")
    if coeff_bin_values is not None:
        coeff_bin_values = torch.as_tensor(coeff_bin_values, dtype=torch.float32).reshape(-1).cpu()
        meta["coeff_bin_values"] = coeff_bin_values
        meta.setdefault("coeff_vocab_size", int(coeff_bin_values.numel()))
        meta.setdefault("n_bins", int(coeff_bin_values.numel()))

    if meta.get("coeff_max") is not None and meta.get("coef_max") is None:
        meta["coef_max"] = float(meta["coeff_max"])
    if meta.get("coef_max") is not None and meta.get("coeff_max") is None:
        meta["coeff_max"] = float(meta["coef_max"])

    needs_coeff_codec = int(meta.get("coeff_vocab_size") or meta.get("n_bins") or 0) <= 0 or meta.get("coeff_bin_values") is None
    if needs_coeff_codec and stage1_checkpoint is not None:
        if payload is None:
            payload = load_torch_payload(stage1_checkpoint)
        state_dict = extract_state_dict(payload)
        if isinstance(state_dict, dict):
            inferred_vocab_size, inferred_bin_values = _coeff_codec_from_state_dict(state_dict)
            if inferred_vocab_size > 0:
                meta.setdefault("coeff_vocab_size", inferred_vocab_size)
                meta.setdefault("n_bins", inferred_vocab_size)
            if inferred_bin_values is not None:
                meta.setdefault("coeff_bin_values", inferred_bin_values)

    enriched["meta"] = meta
    return enriched


def ensure_quantized_cache_metadata(
    cache: dict,
    *,
    token_cache_path,
    output_root="outputs",
) -> dict:
    return ensure_stage2_cache_metadata(
        cache,
        token_cache_path=token_cache_path,
        output_root=output_root,
    )


def _infer_input_channels(model: torch.nn.Module) -> int:
    hparams = getattr(model, "hparams", None)
    if hparams is not None and hasattr(hparams, "in_channels"):
        return int(hparams.in_channels)
    weight = getattr(getattr(model, "encoder", None), "conv_in", None)
    if weight is not None and hasattr(weight, "weight"):
        return int(weight.weight.shape[1])
    raise RuntimeError(f"Could not infer input channels for stage-1 model {type(model)!r}")


def _infer_latent_hw_from_model(model: torch.nn.Module, image_size: Optional[int]) -> Optional[Tuple[int, int]]:
    if image_size is None:
        return None
    image_size = int(image_size)
    if image_size <= 0:
        return None

    if hasattr(model, "infer_latent_hw"):
        latent_hw = model.infer_latent_hw((image_size, image_size))
        return int(latent_hw[0]), int(latent_hw[1])

    in_channels = _infer_input_channels(model)
    try:
        ref_param = next(model.parameters())
        device = ref_param.device
    except StopIteration:
        device = torch.device("cpu")
    dummy = torch.zeros(1, in_channels, image_size, image_size, device=device, dtype=torch.float32)
    with torch.no_grad():
        if hasattr(model, "pre_bottleneck"):
            z = model.pre_bottleneck(model.encoder(dummy))
        else:
            z = model.encoder(dummy)
    return int(z.shape[-2]), int(z.shape[-1])


def load_stage1_decoder_bundle(
    cache: dict,
    *,
    token_cache_path,
    device="cpu",
    output_root="outputs",
) -> Stage1DecodeBundle:
    cache = ensure_stage2_cache_metadata(
        cache,
        token_cache_path=token_cache_path,
        output_root=output_root,
    )
    meta = cache.get("meta", {}) if isinstance(cache, dict) else {}
    checkpoint_path = meta.get("stage1_checkpoint")
    if not checkpoint_path:
        raise RuntimeError(
            f"Could not resolve a stage-1 checkpoint for token cache {Path(token_cache_path).expanduser().resolve()}"
        )
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    payload = load_torch_payload(checkpoint_path)
    state_dict = extract_state_dict(payload)
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Checkpoint at {checkpoint_path} does not contain a state_dict")

    coeff_vocab_size = int(meta.get("coeff_vocab_size") or meta.get("n_bins") or 0)
    coeff_bin_values = meta.get("coeff_bin_values")
    if coeff_bin_values is not None:
        coeff_bin_values = torch.as_tensor(coeff_bin_values, dtype=torch.float32).reshape(-1).cpu()
    if coeff_vocab_size <= 0 or coeff_bin_values is None:
        inferred_vocab_size, inferred_bin_values = _coeff_codec_from_state_dict(state_dict)
        if coeff_vocab_size <= 0:
            coeff_vocab_size = inferred_vocab_size
        if coeff_bin_values is None:
            coeff_bin_values = inferred_bin_values

    device = torch.device(device)
    image_size = meta.get("image_size")
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        stage1_model_type = str(meta.get("stage1_model_type") or _infer_lightning_stage1_type(payload, state_dict))
        if stage1_model_type == "laser":
            from .models.laser import LASER

            model = build_lightning_module(
                LASER,
                payload,
                strict=False,
                compute_fid=False,
                perceptual_weight=0.0,
            ).eval().to(device)
            bundle_kind = "lightning"
        elif stage1_model_type == "vqvae":
            model = build_lightning_module(
                VQVAE,
                payload,
                strict=False,
                compute_fid=False,
                perceptual_weight=0.0,
            ).eval().to(device)
            bundle_kind = "vqvae"
        else:
            raise RuntimeError(f"Unsupported stage-1 Lightning module type: {stage1_model_type!r}")
        model.requires_grad_(False)
        latent_hw = meta.get("latent_hw")
        if isinstance(latent_hw, (tuple, list)) and len(latent_hw) == 2:
            latent_hw = (int(latent_hw[0]), int(latent_hw[1]))
        else:
            latent_hw = _infer_latent_hw_from_model(model, image_size)
        return Stage1DecodeBundle(
            kind=bundle_kind,
            model=model,
            checkpoint_path=checkpoint_path,
            latent_hw=latent_hw,
            coeff_vocab_size=int(coeff_vocab_size),
            coeff_bin_values=coeff_bin_values,
        )

    rq_cfg = _infer_rq_stage1_config(state_dict, cache)
    model = RQAE(**rq_cfg)
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    model.requires_grad_(False)
    latent_hw = meta.get("latent_hw")
    if isinstance(latent_hw, (tuple, list)) and len(latent_hw) == 2:
        latent_hw = (int(latent_hw[0]), int(latent_hw[1]))
    else:
        latent_hw = _infer_latent_hw_from_model(model, image_size or rq_cfg.get("resolution"))
    return Stage1DecodeBundle(
        kind="rq",
        model=model,
        checkpoint_path=checkpoint_path,
        latent_hw=latent_hw,
        coeff_vocab_size=int(coeff_vocab_size),
        coeff_bin_values=coeff_bin_values,
    )


@torch.no_grad()
def decode_stage2_tokens(
    bundle: Stage1DecodeBundle,
    token_grid: torch.Tensor,
    *,
    device=None,
    latent_hw: Optional[Tuple[int, int]] = None,
    class_labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if device is None:
        try:
            ref_param = next(bundle.model.parameters())
            device = ref_param.device
        except StopIteration:
            device = torch.device("cpu")
    else:
        device = torch.device(device)
        bundle.model = bundle.model.to(device)

    tokens = token_grid.to(device=device, dtype=torch.long)
    if bundle.kind == "vqvae":
        if tokens.ndim == 4:
            if int(tokens.shape[-1]) != 1:
                raise ValueError(
                    "VQ-VAE stage-2 decoding expects token grids with depth 1, "
                    f"got shape {tuple(tokens.shape)}"
                )
            indices = tokens[..., 0]
        elif tokens.ndim == 3:
            indices = tokens
        else:
            raise ValueError(f"Expected VQ-VAE tokens with rank 3 or 4, got shape {tuple(tokens.shape)}")
        h_z = int(indices.shape[1])
        w_z = int(indices.shape[2])
        speaker_indices = None
        if class_labels is not None:
            speaker_indices = torch.as_tensor(class_labels, device=device, dtype=torch.long).reshape(-1)
        return bundle.model.decode_from_indices(
            indices.reshape(indices.size(0), -1),
            h_z,
            w_z,
            speaker_indices=speaker_indices,
        )

    target_hw = _resolve_decode_latent_hw(
        bundle,
        token_hw=(int(tokens.shape[1]), int(tokens.shape[2])),
        latent_hw=latent_hw,
    )
    if bundle.kind == "rq":
        return bundle.model.decode_from_tokens(tokens, latent_hw=target_hw)

    if bundle.coeff_vocab_size <= 0 or bundle.coeff_bin_values is None:
        raise RuntimeError(
            f"Stage-1 checkpoint {bundle.checkpoint_path} is missing coefficient bin metadata required for token decode."
        )
    return bundle.model.decode_from_tokens(
        tokens,
        latent_hw=target_hw,
        coeff_vocab_size=int(bundle.coeff_vocab_size),
        coeff_bin_values=bundle.coeff_bin_values.to(device=device, dtype=torch.float32),
    )


def reconstruct_stage2_sparse_latent(
    bundle: Stage1DecodeBundle,
    atom_ids: torch.Tensor,
    coeffs: torch.Tensor,
    *,
    device=None,
    latent_hw: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    if bundle.kind == "vqvae":
        raise RuntimeError("VQ-VAE stage-2 checkpoints do not support sparse atom+coefficient latent reconstruction.")
    if device is None:
        try:
            ref_param = next(bundle.model.parameters())
            device = ref_param.device
        except StopIteration:
            device = torch.device("cpu")
    else:
        device = torch.device(device)
        bundle.model = bundle.model.to(device)

    atoms = atom_ids.to(device=device, dtype=torch.long)
    values = coeffs.to(device=device, dtype=torch.float32)
    target_hw = _resolve_decode_latent_hw(
        bundle,
        token_hw=(int(atoms.shape[1]), int(atoms.shape[2])),
        latent_hw=latent_hw,
    )
    if bundle.kind == "rq":
        bundle.model.eval()
        if hasattr(bundle.model, "clamp_sparse_coeffs"):
            values = bundle.model.clamp_sparse_coeffs(values)
        if getattr(bundle.model.bottleneck, "patch_size", None) is not None:
            if target_hw is None:
                raise ValueError("latent_hw is required for patch-based sparse latent reconstruction")
            return bundle.model.bottleneck._reconstruct_sparse(
                atoms,
                values,
                int(target_hw[0]),
                int(target_hw[1]),
            )
        return bundle.model.bottleneck._reconstruct_sparse(atoms, values)

    return bundle.model.reconstruct_latent_from_atoms_and_coeffs(
        atoms,
        values,
        latent_hw=target_hw,
    )


@torch.no_grad()
def decode_stage2_outputs(
    bundle: Stage1DecodeBundle,
    atoms_or_tokens: torch.Tensor,
    coeffs: Optional[torch.Tensor] = None,
    *,
    device=None,
    latent_hw: Optional[Tuple[int, int]] = None,
    class_labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if coeffs is None:
        return decode_stage2_tokens(
            bundle,
            atoms_or_tokens,
            device=device,
            latent_hw=latent_hw,
            class_labels=class_labels,
        )
    if bundle.kind == "vqvae":
        raise RuntimeError("VQ-VAE stage-2 decoding does not use sparse atom+coefficient outputs.")

    if device is None:
        try:
            ref_param = next(bundle.model.parameters())
            device = ref_param.device
        except StopIteration:
            device = torch.device("cpu")
    else:
        device = torch.device(device)
        bundle.model = bundle.model.to(device)

    atoms = atoms_or_tokens.to(device=device, dtype=torch.long)
    values = coeffs.to(device=device, dtype=torch.float32)
    target_hw = _resolve_decode_latent_hw(
        bundle,
        token_hw=(int(atoms.shape[1]), int(atoms.shape[2])),
        latent_hw=latent_hw,
    )
    if bundle.kind == "rq":
        return bundle.model.decode_from_atoms_and_coeffs(atoms, values, latent_hw=target_hw)
    return bundle.model.decode_from_atoms_and_coeffs(atoms, values, latent_hw=target_hw)
