"""Lightning wrapper and factory helpers for sparse-token stage-2 priors."""

import math
import warnings
from typing import Optional, Tuple

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gpt_prior import GPTPrior, GPTPriorConfig
from .spatial_prior import SpatialDepthPrior, SpatialDepthPriorConfig


def _unpack_cached_batch(
    batch,
) -> tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Normalize tensor- and tuple-based cache batches into tokens and optional metadata.

    CachedTokenDataset emits fields in this order:
    tokens, optional real-valued coeffs, optional class label, optional text tokens,
    optional text mask, optional non-tensor audio metadata.  Keep the parser dtype
    aware so text tokens with the same 2-D shape as tokens are not mistaken for
    real-valued coefficients.
    """
    if torch.is_tensor(batch):
        return batch, None, None, None, None
    if not isinstance(batch, (tuple, list)) or len(batch) == 0:
        raise ValueError("Sparse-token batches must be a tensor or a non-empty sequence.")
    tokens = batch[0]
    if not torch.is_tensor(tokens):
        raise ValueError("Sparse-token batches must start with a token tensor.")
    coeffs = None
    class_labels = None
    text_tokens = None
    text_mask = None
    batch_size = int(tokens.shape[0])
    for item in batch[1:]:
        if not torch.is_tensor(item):
            continue
        if (
            coeffs is None
            and tuple(item.shape) == tuple(tokens.shape)
            and torch.is_floating_point(item)
        ):
            coeffs = item
        elif (
            class_labels is None
            and item.ndim == 1
            and int(item.shape[0]) == batch_size
            and item.dtype != torch.bool
        ):
            class_labels = item
        elif item.ndim == 2 and int(item.shape[0]) == batch_size:
            if item.dtype == torch.bool:
                text_mask = item
            elif text_tokens is None:
                text_tokens = item
    return tokens, coeffs, class_labels, text_tokens, text_mask


def _prior_checkpoint_hparams(prior: nn.Module) -> dict:
    cfg = getattr(prior, "cfg", None)
    if isinstance(prior, SpatialDepthPrior) and isinstance(cfg, SpatialDepthPriorConfig):
        return {
            "prior_architecture": "spatial_depth",
            "prior_H": int(cfg.H),
            "prior_W": int(cfg.W),
            "prior_D": int(cfg.D),
            "prior_d_model": int(cfg.d_model),
            "prior_n_heads": int(cfg.n_heads),
            "prior_n_spatial_layers": int(cfg.n_spatial_layers),
            "prior_n_depth_layers": int(cfg.n_depth_layers),
            "prior_n_global_spatial_tokens": int(cfg.n_global_spatial_tokens),
            "prior_d_ff": int(cfg.d_ff),
            "prior_dropout": float(cfg.dropout),
            "prior_atom_vocab_size": int(
                cfg.atom_vocab_size if cfg.atom_vocab_size is not None else cfg.vocab_size
            ),
            "prior_coeff_vocab_size": int(cfg.coeff_vocab_size or 0),
            "prior_coeff_max": float(cfg.coeff_max),
            "prior_real_valued_coeffs": bool(cfg.real_valued_coeffs),
            "prior_gaussian_coeffs": bool(cfg.gaussian_coeffs),
            "prior_autoregressive_coeffs": bool(cfg.autoregressive_coeffs),
            "prior_coeff_prior_std": float(cfg.coeff_prior_std),
            "prior_coeff_min_std": float(cfg.coeff_min_std),
            "prior_support_order": str(getattr(cfg, "support_order", "none") or "none"),
            "prior_class_conditional": bool(getattr(cfg, "class_conditional", False)),
            "prior_num_classes": int(getattr(cfg, "num_classes", 0) or 0),
            "prior_text_conditional": bool(getattr(cfg, "text_conditional", False)),
            "prior_text_vocab_size": int(getattr(cfg, "text_vocab_size", 0) or 0),
            "prior_text_max_length": int(getattr(cfg, "text_max_length", 0) or 0),
            "prior_text_pad_id": int(getattr(cfg, "text_pad_id", 0) or 0),
            "prior_text_prefix_length": int(getattr(cfg, "text_prefix_length", 0) or 0),
        }
    if isinstance(prior, GPTPrior) and isinstance(cfg, GPTPriorConfig):
        return {
            "prior_architecture": "gpt",
            "prior_H": int(cfg.H),
            "prior_W": int(cfg.W),
            "prior_D": int(cfg.D),
            "prior_d_model": int(cfg.d_model),
            "prior_n_heads": int(cfg.n_heads),
            "prior_n_layers": int(cfg.n_layers),
            "prior_d_ff": int(cfg.d_ff),
            "prior_dropout": float(cfg.dropout),
            "prior_atom_vocab_size": int(cfg.atom_vocab_size),
            "prior_coeff_vocab_size": int(cfg.coeff_vocab_size),
            "prior_window_sites": int(cfg.window_sites),
            "prior_n_global_spatial_tokens": int(cfg.n_global_spatial_tokens),
        }
    return {"prior_architecture": prior.__class__.__name__.lower()}


def compute_quantized_rq_losses(
    per_token_ce: torch.Tensor,
    atom_loss_weight: float,
    coeff_loss_weight: float,
    coeff_depth_weighting: str = "none",
    coeff_focal_gamma: float = 0.0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
    """Split shared-vocab CE into atom/coeff views while keeping one total loss."""
    if per_token_ce.numel() == 0:
        raise ValueError("Expected non-empty per-token CE tensor")

    def _depth_weights(depth_steps: int, mode: str, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if depth_steps <= 0:
            raise ValueError("depth_steps must be positive")
        mode = str(mode).strip().lower()
        if mode == "none":
            return torch.ones(depth_steps, device=device, dtype=dtype)
        if mode == "linear":
            weights = torch.arange(depth_steps, 0, -1, device=device, dtype=dtype)
        elif mode == "inverse_rank":
            weights = 1.0 / torch.arange(1, depth_steps + 1, device=device, dtype=dtype)
        else:
            raise ValueError(f"Unsupported coeff_depth_weighting: {mode!r}")
        return weights / weights.mean().clamp_min(1e-8)

    token_ce_loss = per_token_ce.mean()
    atom_terms = per_token_ce[..., 0::2]
    coeff_terms = per_token_ce[..., 1::2]

    atom_ce_loss = atom_terms.mean() if atom_terms.numel() > 0 else None

    coeff_ce_loss = None
    coeff_weighted_terms = None
    if coeff_terms.numel() > 0:
        coeff_weighted_terms = coeff_terms
        depth_weights = _depth_weights(
            coeff_terms.size(-1),
            coeff_depth_weighting,
            device=coeff_terms.device,
            dtype=coeff_terms.dtype,
        )
        coeff_weighted_terms = coeff_weighted_terms * depth_weights.view(
            *([1] * (coeff_terms.ndim - 1)),
            coeff_terms.size(-1),
        )
        coeff_focal_gamma = float(max(0.0, coeff_focal_gamma))
        if coeff_focal_gamma > 0.0:
            pt = torch.exp(-coeff_terms.clamp_min(0.0))
            coeff_weighted_terms = coeff_weighted_terms * (1.0 - pt).pow(coeff_focal_gamma)
        coeff_ce_loss = coeff_weighted_terms.mean()

    total_numerator = per_token_ce.new_tensor(0.0)
    total_denominator = per_token_ce.new_tensor(0.0)
    atom_loss_weight = float(atom_loss_weight)
    coeff_loss_weight = float(coeff_loss_weight)
    if atom_ce_loss is not None:
        total_numerator = total_numerator + atom_loss_weight * atom_terms.sum()
        total_denominator = total_denominator + atom_loss_weight * atom_terms.numel()
    if coeff_ce_loss is not None and coeff_weighted_terms is not None:
        total_numerator = total_numerator + coeff_loss_weight * coeff_weighted_terms.sum()
        total_denominator = total_denominator + coeff_loss_weight * coeff_weighted_terms.numel()
    if total_denominator.item() <= 0.0:
        raise ValueError("Expected positive quantized loss denominator")
    total_loss = total_numerator / total_denominator
    return token_ce_loss, atom_ce_loss, coeff_ce_loss, total_loss


def infer_sparse_vocab_sizes(
    cache: dict,
    *,
    total_vocab_size: Optional[int],
    atom_vocab_size: Optional[int] = None,
    coeff_vocab_size: Optional[int] = None,
) -> Tuple[int, int, int]:
    """Resolve total/atom/coeff vocab sizes from config plus cache metadata."""
    tokens_flat = cache.get("tokens_flat")
    meta = cache.get("meta", {}) if isinstance(cache, dict) else {}
    if not torch.is_tensor(tokens_flat) or tokens_flat.ndim != 2:
        raise ValueError("cache['tokens_flat'] must be a rank-2 tensor")

    resolved_atom_vocab = int(atom_vocab_size or meta.get("num_atoms") or 0)
    if resolved_atom_vocab <= 0:
        atom_tokens = tokens_flat[:, 0::2].to(torch.long)
        resolved_atom_vocab = int(atom_tokens.max().item()) + 1

    coeff_bin_values = meta.get("coeff_bin_values")
    inferred_coeff_bins = 0
    if coeff_bin_values is not None:
        inferred_coeff_bins = int(torch.as_tensor(coeff_bin_values).numel())

    resolved_coeff_vocab = int(
        coeff_vocab_size or meta.get("n_bins") or meta.get("coeff_vocab_size") or inferred_coeff_bins or 0
    )
    resolved_total_vocab = int(total_vocab_size or 0)
    if resolved_coeff_vocab <= 0:
        if resolved_total_vocab <= resolved_atom_vocab:
            raise ValueError(
                "Sparse-token prior needs either coeff_vocab_size or a total vocab_size larger than atom_vocab_size."
            )
        resolved_coeff_vocab = resolved_total_vocab - resolved_atom_vocab

    if resolved_total_vocab <= 0:
        resolved_total_vocab = resolved_atom_vocab + resolved_coeff_vocab

    if resolved_total_vocab != resolved_atom_vocab + resolved_coeff_vocab:
        raise ValueError(
            "Expected total vocab size to equal atom_vocab_size + coeff_vocab_size, "
            f"got total={resolved_total_vocab}, atom={resolved_atom_vocab}, coeff={resolved_coeff_vocab}"
        )
    return resolved_total_vocab, resolved_atom_vocab, resolved_coeff_vocab


def token_cache_grid_shape(cache: dict) -> Tuple[int, int, int]:
    shape = cache.get("shape")
    if not isinstance(shape, (tuple, list)) or len(shape) != 3:
        raise ValueError("cache['shape'] must be a length-3 tuple/list")
    return int(shape[0]), int(shape[1]), int(shape[2])


def infer_sparse_atom_vocab_size(
    cache: dict,
    *,
    atom_vocab_size: Optional[int] = None,
) -> int:
    tokens_flat = cache.get("tokens_flat")
    meta = cache.get("meta", {}) if isinstance(cache, dict) else {}
    if not torch.is_tensor(tokens_flat) or tokens_flat.ndim != 2:
        raise ValueError("cache['tokens_flat'] must be a rank-2 tensor")

    resolved_atom_vocab = int(atom_vocab_size or meta.get("num_atoms") or 0)
    if resolved_atom_vocab <= 0:
        resolved_atom_vocab = int(tokens_flat.to(torch.long).max().item()) + 1
    return resolved_atom_vocab


def build_sparse_prior_from_cache(
    cache: dict,
    *,
    architecture: str,
    total_vocab_size: Optional[int],
    atom_vocab_size: Optional[int],
    coeff_vocab_size: Optional[int],
    grid_shape: Optional[Tuple[int, int, int]] = None,
    window_sites: int = 0,
    d_model: int,
    n_heads: int,
    n_layers: int,
    d_ff: int,
    dropout: float,
    n_global_spatial_tokens: int = 0,
    autoregressive_coeffs: bool = True,
    class_conditional: bool = False,
    num_classes: int = 0,
    text_conditional: bool = False,
    text_vocab_size: int = 0,
    text_max_length: int = 0,
    text_pad_id: int = 0,
    text_prefix_length: int = 16,
) -> nn.Module:
    """Build a maintained sparse-token prior from a cached token grid."""
    if grid_shape is None:
        H, W, D = token_cache_grid_shape(cache)
    else:
        H, W, D = (int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2]))
    meta = cache.get("meta", {}) if isinstance(cache, dict) else {}
    support_order = str(meta.get("support_order", "none") or "none")
    real_valued_coeffs = cache.get("coeffs_flat") is not None
    architecture = str(architecture).strip().lower()
    if architecture == "mingpt":
        architecture = "gpt"

    if real_valued_coeffs:
        if architecture != "spatial_depth":
            raise ValueError("Real-valued sparse-token caches are only supported with architecture='spatial_depth'.")
        atom_vocab_size = infer_sparse_atom_vocab_size(cache, atom_vocab_size=atom_vocab_size)
        total_vocab_size = int(total_vocab_size or atom_vocab_size)
        if total_vocab_size != atom_vocab_size:
            raise ValueError(
                "Real-valued sparse-token priors expect total vocab size to equal atom vocab size, "
                f"got total={total_vocab_size}, atom={atom_vocab_size}"
            )
        coeff_max = float(meta.get("coeff_max", meta.get("coef_max", 24.0)))
        return SpatialDepthPrior(
            SpatialDepthPriorConfig(
                vocab_size=total_vocab_size,
                atom_vocab_size=atom_vocab_size,
                coeff_vocab_size=None,
                coeff_bin_values=None,
                H=H,
                W=W,
                D=D,
                real_valued_coeffs=True,
                d_model=int(d_model),
                n_heads=int(n_heads),
                n_spatial_layers=int(n_layers),
                n_depth_layers=max(1, int(n_layers) // 2),
                n_global_spatial_tokens=int(n_global_spatial_tokens),
                d_ff=int(d_ff),
                dropout=float(dropout),
                coeff_max=coeff_max,
                gaussian_coeffs=bool(meta.get("variational_coeffs", False)),
                coeff_prior_std=float(meta.get("variational_coeff_target_std", 0.25)),
                coeff_min_std=float(meta.get("variational_coeff_min_std", 0.01)),
                autoregressive_coeffs=bool(autoregressive_coeffs),
                support_order=support_order,
                class_conditional=bool(class_conditional),
                num_classes=int(num_classes),
                text_conditional=bool(text_conditional),
                text_vocab_size=int(text_vocab_size),
                text_max_length=int(text_max_length),
                text_pad_id=int(text_pad_id),
                text_prefix_length=int(text_prefix_length),
            )
        )

    total_vocab_size, atom_vocab_size, coeff_vocab_size = infer_sparse_vocab_sizes(
        cache,
        total_vocab_size=total_vocab_size,
        atom_vocab_size=atom_vocab_size,
        coeff_vocab_size=coeff_vocab_size,
    )

    if architecture == "spatial_depth":
        coeff_bin_values = meta.get("coeff_bin_values")
        coeff_max = float(meta.get("coef_max", 24.0))
        return SpatialDepthPrior(
            SpatialDepthPriorConfig(
                vocab_size=total_vocab_size,
                atom_vocab_size=atom_vocab_size,
                coeff_vocab_size=coeff_vocab_size,
                coeff_bin_values=coeff_bin_values,
                H=H,
                W=W,
                D=D,
                real_valued_coeffs=False,
                d_model=int(d_model),
                n_heads=int(n_heads),
                n_spatial_layers=int(n_layers),
                n_depth_layers=max(1, int(n_layers) // 2),
                n_global_spatial_tokens=int(n_global_spatial_tokens),
                d_ff=int(d_ff),
                dropout=float(dropout),
                coeff_max=coeff_max,
                autoregressive_coeffs=bool(autoregressive_coeffs),
                support_order=support_order,
                class_conditional=bool(class_conditional),
                num_classes=int(num_classes),
                text_conditional=bool(text_conditional),
                text_vocab_size=int(text_vocab_size),
                text_max_length=int(text_max_length),
                text_pad_id=int(text_pad_id),
                text_prefix_length=int(text_prefix_length),
            )
        )
    if architecture == "gpt":
        return GPTPrior(
            GPTPriorConfig(
                vocab_size=total_vocab_size,
                H=H,
                W=W,
                D=D,
                atom_vocab_size=atom_vocab_size,
                coeff_vocab_size=coeff_vocab_size,
                window_sites=int(window_sites),
                n_global_spatial_tokens=int(n_global_spatial_tokens),
                d_model=int(d_model),
                n_heads=int(n_heads),
                n_layers=int(n_layers),
                d_ff=int(d_ff),
                dropout=float(dropout),
            )
        )
    raise ValueError(f"Unsupported sparse prior architecture: {architecture!r}")


def build_sparse_prior_from_hparams(
    cache: dict,
    *,
    hparams: dict,
) -> nn.Module:
    """Rebuild a maintained sparse-token prior from checkpoint hparams plus cache metadata."""
    architecture = str(hparams.get("prior_architecture", "")).strip().lower()
    if not architecture:
        raise ValueError("Checkpoint hparams are missing prior_architecture")
    if architecture == "mingpt":
        architecture = "gpt"
    real_valued_coeffs = bool(hparams.get("prior_real_valued_coeffs", False))
    gaussian_coeffs = bool(hparams.get("prior_gaussian_coeffs", False))
    autoregressive_coeffs = bool(hparams.get("prior_autoregressive_coeffs", True))

    total_vocab_size = hparams.get("resolved_total_vocab_size")
    if total_vocab_size in (None, 0):
        atom_part = int(hparams.get("resolved_atom_vocab_size") or hparams.get("prior_atom_vocab_size") or 0)
        coeff_part = int(hparams.get("resolved_coeff_vocab_size") or hparams.get("prior_coeff_vocab_size") or 0)
        total_vocab_size = atom_part + coeff_part if atom_part > 0 and coeff_part > 0 else None

    atom_vocab_size = hparams.get("resolved_atom_vocab_size")
    if atom_vocab_size in (None, 0):
        atom_vocab_size = hparams.get("prior_atom_vocab_size")

    coeff_vocab_size = hparams.get("resolved_coeff_vocab_size")
    if coeff_vocab_size in (None, 0):
        coeff_vocab_size = hparams.get("prior_coeff_vocab_size")

    saved_shape = (
        hparams.get("prior_H"),
        hparams.get("prior_W"),
        hparams.get("prior_D"),
    )
    if all(v not in (None, 0) for v in saved_shape):
        H, W, D = (int(saved_shape[0]), int(saved_shape[1]), int(saved_shape[2]))
    else:
        H, W, D = token_cache_grid_shape(cache)
    meta = cache.get("meta", {}) if isinstance(cache, dict) else {}
    support_order = str(hparams.get("prior_support_order", meta.get("support_order", "none")) or "none")
    class_conditional = bool(hparams.get("prior_class_conditional", False))
    num_classes = int(hparams.get("prior_num_classes", meta.get("num_classes", 0)) or 0)
    text_conditional = bool(hparams.get("prior_text_conditional", False))
    text_vocab_size = int(hparams.get("prior_text_vocab_size", meta.get("text_vocab_size", 0)) or 0)
    text_max_length = int(hparams.get("prior_text_max_length", meta.get("text_max_length", 0)) or 0)
    text_pad_id = int(hparams.get("prior_text_pad_id", meta.get("text_pad_id", 0)) or 0)
    text_prefix_length = int(hparams.get("prior_text_prefix_length", meta.get("text_prefix_length", 0)) or 0)
    if real_valued_coeffs:
        atom_vocab_size = infer_sparse_atom_vocab_size(cache, atom_vocab_size=atom_vocab_size)
        total_vocab_size = int(total_vocab_size or atom_vocab_size)
        coeff_vocab_size = None
    else:
        total_vocab_size, atom_vocab_size, coeff_vocab_size = infer_sparse_vocab_sizes(
            cache,
            total_vocab_size=total_vocab_size,
            atom_vocab_size=atom_vocab_size,
            coeff_vocab_size=coeff_vocab_size,
        )

    if architecture == "spatial_depth":
        coeff_max = float(hparams.get("prior_coeff_max", meta.get("coef_max", 24.0)))
        coeff_bin_values = None if real_valued_coeffs else meta.get("coeff_bin_values")
        return SpatialDepthPrior(
            SpatialDepthPriorConfig(
                vocab_size=total_vocab_size,
                atom_vocab_size=atom_vocab_size,
                coeff_vocab_size=coeff_vocab_size,
                coeff_bin_values=coeff_bin_values,
                H=H,
                W=W,
                D=D,
                real_valued_coeffs=bool(real_valued_coeffs),
                d_model=int(hparams["prior_d_model"]),
                n_heads=int(hparams["prior_n_heads"]),
                n_spatial_layers=int(hparams["prior_n_spatial_layers"]),
                n_depth_layers=int(hparams.get("prior_n_depth_layers", max(1, int(hparams["prior_n_spatial_layers"]) // 2))),
                n_global_spatial_tokens=int(hparams.get("prior_n_global_spatial_tokens", 0)),
                d_ff=int(hparams["prior_d_ff"]),
                dropout=float(hparams["prior_dropout"]),
                coeff_max=coeff_max,
                gaussian_coeffs=bool(gaussian_coeffs),
                coeff_prior_std=float(hparams.get("prior_coeff_prior_std", 0.25)),
                coeff_min_std=float(hparams.get("prior_coeff_min_std", 0.01)),
                autoregressive_coeffs=bool(autoregressive_coeffs),
                support_order=support_order,
                class_conditional=class_conditional,
                num_classes=num_classes,
                text_conditional=text_conditional,
                text_vocab_size=text_vocab_size,
                text_max_length=text_max_length,
                text_pad_id=text_pad_id,
                text_prefix_length=text_prefix_length,
            )
        )

    if architecture == "gpt":
        if real_valued_coeffs:
            raise ValueError("gpt checkpoints do not support real-valued sparse coefficients.")
        return GPTPrior(
            GPTPriorConfig(
                vocab_size=total_vocab_size,
                H=H,
                W=W,
                D=D,
                atom_vocab_size=atom_vocab_size,
                coeff_vocab_size=coeff_vocab_size,
                window_sites=int(hparams.get("prior_window_sites", 0)),
                n_global_spatial_tokens=int(hparams.get("prior_n_global_spatial_tokens", 0)),
                d_model=int(hparams["prior_d_model"]),
                n_heads=int(hparams["prior_n_heads"]),
                n_layers=int(hparams["prior_n_layers"]),
                d_ff=int(hparams["prior_d_ff"]),
                dropout=float(hparams["prior_dropout"]),
            )
        )

    raise ValueError(f"Unsupported sparse prior architecture in checkpoint hparams: {architecture!r}")


class SparseTokenPriorModule(pl.LightningModule):
    """Lightning wrapper for maintained sparse-token stage-2 priors."""

    def __init__(
        self,
        prior: nn.Module,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        min_lr_ratio: float = 0.01,
        atom_loss_weight: float = 1.0,
        coeff_loss_weight: float = 1.0,
        coeff_depth_weighting: str = "none",
        coeff_focal_gamma: float = 0.0,
        coeff_loss_type: Optional[str] = "auto",
        coeff_huber_delta: float = 0.5,
        sample_coeff_temperature: Optional[float] = None,
        sample_coeff_mode: str = "gaussian",
        atom_label_smoothing: float = 0.0,
        atom_coverage_weight: float = 0.0,
    ):
        super().__init__()
        self.prior = prior
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(max(0.0, weight_decay))
        self.warmup_steps = max(0, int(warmup_steps))
        self.min_lr_ratio = float(max(0.0, min(float(min_lr_ratio), 1.0)))
        self.atom_loss_weight = float(atom_loss_weight)
        # Anti-collapse levers for the atom-id head (both default off -> exact
        # legacy behavior). Label smoothing softens the CE targets so the prior
        # cannot over-peak on the few dominant atoms; the coverage weight rewards a
        # high-entropy batch-marginal atom distribution so generation keeps using a
        # broad atom support instead of collapsing to tens of atoms.
        self.atom_label_smoothing = float(min(max(atom_label_smoothing, 0.0), 1.0))
        self.atom_coverage_weight = float(max(0.0, atom_coverage_weight))
        self.coeff_loss_weight = float(coeff_loss_weight)
        self.coeff_depth_weighting = str(coeff_depth_weighting).strip().lower()
        self.coeff_focal_gamma = float(max(0.0, coeff_focal_gamma))
        self.coeff_huber_delta = float(max(1e-6, coeff_huber_delta))
        self.sample_coeff_temperature = (
            None if sample_coeff_temperature is None else float(sample_coeff_temperature)
        )
        self.sample_coeff_mode = str(sample_coeff_mode).strip().lower()
        if self.sample_coeff_mode not in {"gaussian", "mean"}:
            raise ValueError(
                f"sample_coeff_mode must be 'gaussian' or 'mean', got {self.sample_coeff_mode!r}"
            )
        if (
            self.sample_coeff_mode == "gaussian"
            and bool(getattr(self.prior, "real_valued_coeffs", False))
            and not bool(getattr(self.prior, "gaussian_coeffs", False))
        ):
            warnings.warn(
                "sample_coeff_mode='gaussian' requires a variational/Gaussian coefficient head; "
                "falling back to deterministic mean coefficient sampling.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.sample_coeff_mode = "mean"
        self.coeff_loss_type = self._resolve_coeff_loss_type(coeff_loss_type)
        self._lr_base_lrs = ()
        self._lr_total_steps = 1
        self.save_hyperparameters(ignore=["prior"])
        self.save_hyperparameters(_prior_checkpoint_hparams(prior))

    def _wandb_step(self, experiment=None, *, requested_step: Optional[int] = None) -> int:
        step = int(self.global_step if requested_step is None else requested_step)
        exp = experiment
        if exp is None:
            logger = getattr(self, "logger", None)
            exp = getattr(logger, "experiment", None) if logger is not None else None
        if exp is not None:
            for attr in ("step", "_step"):
                raw = getattr(exp, attr, None)
                if raw is None:
                    continue
                try:
                    step = max(step, int(raw))
                except (TypeError, ValueError):
                    continue
        return step

    def _wandb_epoch_end_step(self, experiment=None, *, requested_step: Optional[int] = None) -> int:
        base = int(self.global_step if requested_step is None else requested_step)
        return self._wandb_step(experiment, requested_step=base + 1)

    def _lr_multiplier_for_step(self, step: int) -> float:
        if self.warmup_steps <= 0 and self.min_lr_ratio >= 1.0:
            return 1.0
        total_steps = max(1, int(getattr(self, "_lr_total_steps", 1)))
        warmup_steps = min(self.warmup_steps, max(0, total_steps - 1))
        cur_step = max(0, int(step)) + 1
        if warmup_steps > 0 and cur_step <= warmup_steps:
            return max(0.01, cur_step / float(max(1, warmup_steps)))
        progress = (cur_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def _apply_scheduled_lrs(self, optimizer, step: int) -> None:
        if not self._lr_base_lrs:
            self._lr_base_lrs = tuple(float(group["lr"]) for group in optimizer.param_groups)
        scale = self._lr_multiplier_for_step(step)
        for group, base_lr in zip(optimizer.param_groups, self._lr_base_lrs):
            group["lr"] = float(base_lr) * float(scale)

    def _resolve_coeff_loss_type(self, requested: Optional[str]) -> Optional[str]:
        text = "" if requested is None else str(requested).strip().lower()
        if not getattr(self.prior, "real_valued_coeffs", False):
            return None if text in {"", "auto"} else text
        if text in {"", "auto"}:
            return "gaussian_nll" if bool(getattr(self.prior, "gaussian_coeffs", False)) else "mse"
        if text not in {"mse", "huber", "gaussian_nll"}:
            raise ValueError(f"Unsupported coeff_loss_type: {requested!r}")
        if text == "gaussian_nll" and not bool(getattr(self.prior, "gaussian_coeffs", False)):
            raise ValueError("coeff_loss_type='gaussian_nll' requires a Gaussian coefficient head.")
        return text

    def _clamp_coeffs(self, coeffs: torch.Tensor) -> torch.Tensor:
        coeff_max = getattr(getattr(self.prior, "cfg", None), "coeff_max", None)
        try:
            coeff_max = float(coeff_max)
        except (TypeError, ValueError):
            coeff_max = None
        if coeff_max is None or not math.isfinite(coeff_max) or coeff_max <= 0.0:
            return coeffs
        return coeffs.clamp(-coeff_max, coeff_max)

    def _step_log_kwargs(self, prefix: str, batch_size: int) -> dict:
        return dict(
            on_step=(prefix == "train"),
            on_epoch=(prefix != "train"),
            sync_dist=True,
            batch_size=int(batch_size),
            prog_bar=(prefix != "test"),
        )

    def _atom_ce_loss(self, logits_flat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Atom-id cross-entropy that stays finite under -inf-masked logits.

        The spatial_depth prior masks invalid / already-selected atoms to -inf
        (spatial_prior.py). Plain ``F.cross_entropy(..., label_smoothing>0)`` would
        spread the smoothing mass onto those -inf classes and return +inf, so when
        smoothing is on we distribute it uniformly over only the finite-logit
        (valid) classes. With smoothing off this is exactly the legacy CE.
        """
        eps = self.atom_label_smoothing
        if eps <= 0.0:
            return F.cross_entropy(logits_flat, target)
        logp = F.log_softmax(logits_flat, dim=-1)
        finite = torch.isfinite(logp)
        nll = -logp.gather(1, target.unsqueeze(1)).squeeze(1)
        # mean of -logp over valid classes only; zero out -inf so 0*(-inf) is not NaN
        safe_logp = torch.where(finite, logp, torch.zeros_like(logp))
        n_valid = finite.sum(dim=1).clamp_min(1)
        uniform = -(safe_logp.sum(dim=1) / n_valid)
        return ((1.0 - eps) * nll + eps * uniform).mean()

    def _real_valued_shared_step(self, batch, prefix: str) -> torch.Tensor:
        tok_flat, coeff_flat, class_labels, text_tokens, text_mask = _unpack_cached_batch(batch)
        if coeff_flat is None:
            raise ValueError("Real-valued sparse-token training expects batches of (tokens_flat, coeffs_flat).")

        tok_flat = tok_flat.to(self.device, dtype=torch.long, non_blocking=True)
        coeff_flat = coeff_flat.to(self.device, dtype=torch.float32, non_blocking=True)
        if class_labels is not None:
            class_labels = class_labels.to(self.device, dtype=torch.long, non_blocking=True)
        if bool(getattr(self.prior, "text_conditional", False)):
            if text_tokens is None:
                raise ValueError("ar.text_conditional=true requires text_tokens in the token cache.")
            text_tokens = text_tokens.to(self.device, dtype=torch.long, non_blocking=True)
            text_mask = None if text_mask is None else text_mask.to(self.device, dtype=torch.bool, non_blocking=True)
        else:
            text_tokens = None
            text_mask = None
        bsz = tok_flat.size(0)
        cfg = self.prior.cfg
        # Token indices are integer tensors with no gradient; a single contiguous
        # copy is sufficient for all read-only uses (CE targets, forward input,
        # mask tokens). Coefficient grid is float but only read, never written.
        tok_grid = tok_flat.view(bsz, int(cfg.H) * int(cfg.W), int(cfg.D)).clone()
        coeff_grid = coeff_flat.view(bsz, int(cfg.H) * int(cfg.W), int(cfg.D))

        loss_type = str(self.coeff_loss_type or "mse")
        forward_out = self.prior(
            tok_grid,
            coeff_grid,
            class_labels=class_labels,
            text_tokens=text_tokens,
            text_mask=text_mask,
        )
        if bool(getattr(self.prior, "gaussian_coeffs", False)):
            atom_logits, coeff_pred, coeff_logvar_pred = forward_out
        else:
            atom_logits, coeff_pred = forward_out
            coeff_logvar_pred = None

        atom_logits_flat = atom_logits.reshape(-1, int(cfg.vocab_size))
        ce_loss = self._atom_ce_loss(atom_logits_flat, tok_grid.reshape(-1))
        pred_coeff = self._clamp_coeffs(coeff_pred)
        target_coeff = self._clamp_coeffs(coeff_grid)
        if loss_type == "mse":
            coeff_reg_loss = F.mse_loss(pred_coeff, target_coeff)
        elif loss_type == "huber":
            coeff_reg_loss = F.huber_loss(
                pred_coeff,
                target_coeff,
                delta=self.coeff_huber_delta,
            )
        elif loss_type == "gaussian_nll":
            if coeff_logvar_pred is None:
                raise RuntimeError("gaussian_nll requested but the prior did not return coefficient log-variance.")
            pred_var = coeff_logvar_pred.exp().clamp_min(1e-6)
            coeff_reg_loss = 0.5 * (
                coeff_logvar_pred + (pred_coeff - target_coeff).square() / pred_var
            )
            coeff_reg_loss = coeff_reg_loss.mean()
        else:
            raise RuntimeError(f"Unexpected coeff_loss_type: {loss_type!r}")

        loss = self.atom_loss_weight * ce_loss + self.coeff_loss_weight * coeff_reg_loss

        # Collapse diagnostics: the batch-marginal predicted atom distribution and
        # its entropy. Logged regardless of the regularizer weight so the collapse
        # (low coverage_frac -> few atoms ever predicted) is visible in baseline
        # runs too. When atom_coverage_weight > 0 we also reward this entropy.
        atom_marginal = F.softmax(atom_logits_flat, dim=-1).mean(dim=0).clamp_min(1e-12)
        atom_marginal_entropy = -(atom_marginal * atom_marginal.log()).sum()
        atom_coverage_frac = atom_marginal_entropy / math.log(float(cfg.vocab_size))
        if self.atom_coverage_weight > 0.0:
            loss = loss - self.atom_coverage_weight * atom_marginal_entropy

        atom_preds = atom_logits.argmax(dim=-1)
        atom_accuracy = (atom_preds == tok_grid).float().mean()
        atom_pred_unique_frac = (
            atom_preds.reshape(-1).unique().numel() / float(cfg.vocab_size)
        )
        coeff_mae = (pred_coeff - target_coeff).abs().mean()

        log_kwargs = self._step_log_kwargs(prefix, bsz)
        self.log(f"{prefix}/loss", loss, **log_kwargs)
        self.log(f"{prefix}/ce_loss", ce_loss, **log_kwargs)
        self.log(f"{prefix}/atom_accuracy", atom_accuracy, **log_kwargs)
        self.log(f"{prefix}/atom_marginal_entropy", atom_marginal_entropy, **log_kwargs)
        self.log(f"{prefix}/atom_coverage_frac", atom_coverage_frac, **log_kwargs)
        self.log(f"{prefix}/atom_pred_unique_frac", atom_pred_unique_frac, **log_kwargs)
        self.log(f"{prefix}/coeff_reg_loss", coeff_reg_loss, **log_kwargs)
        self.log(f"{prefix}/coeff_mae", coeff_mae, **log_kwargs)
        if loss_type == "mse":
            self.log(f"{prefix}/coeff_mse_loss", coeff_reg_loss, **log_kwargs)
        elif loss_type == "gaussian_nll":
            self.log(f"{prefix}/coeff_gaussian_nll", coeff_reg_loss, **log_kwargs)
        elif loss_type == "huber":
            self.log(f"{prefix}/coeff_huber_loss", coeff_reg_loss, **log_kwargs)
        else:
            raise RuntimeError(f"Unexpected coeff_loss_type: {loss_type!r}")
        return loss

    def configure_optimizers(self):
        optimizer_cls = torch.optim.AdamW if self.weight_decay > 0 else torch.optim.Adam
        optimizer = optimizer_cls(self.prior.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        trainer = getattr(self, "_trainer", None)
        total_steps = None
        if trainer is not None:
            for value in (getattr(trainer, "max_steps", None),):
                try:
                    if value is not None and int(value) > 0:
                        total_steps = int(value)
                        break
                except (TypeError, ValueError):
                    pass
            if total_steps is None:
                try:
                    max_epochs_raw = float(getattr(trainer, "max_epochs", 0) or 0)
                    num_batches_raw = float(getattr(trainer, "num_training_batches", 0) or 0)
                    if (
                        math.isfinite(max_epochs_raw)
                        and math.isfinite(num_batches_raw)
                        and max_epochs_raw > 0
                        and num_batches_raw > 0
                    ):
                        total_steps = int(max_epochs_raw) * int(num_batches_raw)
                except (TypeError, ValueError):
                    total_steps = None
            if total_steps is None:
                descriptor = getattr(type(trainer), "estimated_stepping_batches", None)
                if isinstance(descriptor, property):
                    trainer_state = getattr(trainer, "__dict__", {})
                    estimated_steps = (
                        trainer_state.get("estimated_stepping_batches") if isinstance(trainer_state, dict) else None
                    )
                else:
                    estimated_steps = getattr(trainer, "estimated_stepping_batches", None)
                try:
                    total_steps = int(estimated_steps)
                except (TypeError, ValueError):
                    total_steps = None
        self._lr_total_steps = max(1, int(total_steps or 1))
        self._lr_base_lrs = tuple(float(group["lr"]) for group in optimizer.param_groups)
        return optimizer

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        self._apply_scheduled_lrs(optimizer, step=int(getattr(self, "global_step", 0)))
        optimizer.step(closure=optimizer_closure)

    def _shared_step(self, batch, prefix: str) -> torch.Tensor:
        if getattr(self.prior, "real_valued_coeffs", False):
            return self._real_valued_shared_step(batch, prefix)

        tok_flat, _, class_labels, text_tokens, text_mask = _unpack_cached_batch(batch)
        tok_flat = tok_flat.to(self.device, dtype=torch.long, non_blocking=True)
        if class_labels is not None:
            class_labels = class_labels.to(self.device, dtype=torch.long, non_blocking=True)
        if bool(getattr(self.prior, "text_conditional", False)):
            if text_tokens is None:
                raise ValueError("ar.text_conditional=true requires text_tokens in the token cache.")
            text_tokens = text_tokens.to(self.device, dtype=torch.long, non_blocking=True)
            text_mask = None if text_mask is None else text_mask.to(self.device, dtype=torch.bool, non_blocking=True)
        else:
            text_tokens = None
            text_mask = None
        bsz = tok_flat.size(0)
        cfg = self.prior.cfg
        tok_grid = tok_flat.view(bsz, int(cfg.H) * int(cfg.W), int(cfg.D))

        logits = self.prior(
            tok_grid,
            class_labels=class_labels,
            text_tokens=text_tokens,
            text_mask=text_mask,
        )
        per_token_ce = F.cross_entropy(
            logits.reshape(-1, int(cfg.vocab_size)),
            tok_grid.reshape(-1),
            reduction="none",
        ).view(bsz, int(cfg.H) * int(cfg.W), int(cfg.D))
        ce_loss, atom_ce_loss, coeff_ce_loss, loss = compute_quantized_rq_losses(
            per_token_ce,
            atom_loss_weight=self.atom_loss_weight,
            coeff_loss_weight=self.coeff_loss_weight,
            coeff_depth_weighting=self.coeff_depth_weighting,
            coeff_focal_gamma=self.coeff_focal_gamma,
        )

        preds = logits.argmax(dim=-1)
        accuracy = (preds == tok_grid).float().mean()
        atom_targets = tok_grid[..., 0::2]
        atom_accuracy = (
            (preds[..., 0::2] == atom_targets).float().mean()
            if atom_targets.numel() > 0
            else None
        )
        coeff_targets = tok_grid[..., 1::2]
        coeff_accuracy = (
            (preds[..., 1::2] == coeff_targets).float().mean()
            if coeff_targets.numel() > 0
            else None
        )

        log_kwargs = self._step_log_kwargs(prefix, bsz)
        self.log(f"{prefix}/loss", loss, **log_kwargs)
        self.log(f"{prefix}/ce_loss", ce_loss, **log_kwargs)
        self.log(f"{prefix}/accuracy", accuracy, **log_kwargs)
        if atom_accuracy is not None:
            self.log(f"{prefix}/atom_accuracy", atom_accuracy, **log_kwargs)
        if coeff_accuracy is not None:
            self.log(f"{prefix}/coeff_accuracy", coeff_accuracy, **log_kwargs)
        if atom_ce_loss is not None:
            self.log(f"{prefix}/atom_ce_loss", atom_ce_loss, **log_kwargs)
        if coeff_ce_loss is not None:
            self.log(f"{prefix}/coeff_ce_loss", coeff_ce_loss, **log_kwargs)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    @torch.no_grad()
    def generate_tokens(
        self,
        batch_size: int,
        *,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        class_labels: Optional[torch.Tensor] = None,
        text_tokens: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        generated = self.generate_sparse_codes(
            batch_size,
            temperature=temperature,
            top_k=top_k,
            class_labels=class_labels,
            text_tokens=text_tokens,
            text_mask=text_mask,
        )
        if getattr(self.prior, "real_valued_coeffs", False):
            return generated[0]
        return generated

    @torch.no_grad()
    def generate_sparse_codes(
        self,
        batch_size: int,
        *,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        coeff_temperature: Optional[float] = None,
        coeff_sample_mode: Optional[str] = None,
        class_labels: Optional[torch.Tensor] = None,
        text_tokens: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ):
        if coeff_temperature is None:
            coeff_temperature = self.sample_coeff_temperature
        if coeff_sample_mode is None:
            coeff_sample_mode = self.sample_coeff_mode
        return self.prior.generate(
            batch_size=batch_size,
            temperature=temperature,
            top_k=top_k,
            coeff_temperature=coeff_temperature,
            coeff_sample_mode=coeff_sample_mode,
            class_labels=class_labels,
            text_tokens=text_tokens,
            text_mask=text_mask,
            show_progress=False,
        )
