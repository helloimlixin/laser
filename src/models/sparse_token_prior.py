"""Lightning wrapper and factory helpers for sparse-token stage-2 priors."""

import math
from typing import Optional, Tuple

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils

from src.stage2_compat import decode_stage2_outputs, reconstruct_stage2_sparse_latent

from .mingpt_prior import MinGPTQuantizedPrior, MinGPTQuantizedPriorConfig
from .spatial_prior import SpatialDepthPrior, SpatialDepthPriorConfig


def _prior_checkpoint_hparams(prior: nn.Module) -> dict:
    cfg = getattr(prior, "cfg", None)
    if isinstance(prior, SpatialDepthPrior) and isinstance(cfg, SpatialDepthPriorConfig):
        return {
            "prior_architecture": "spatial_depth",
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
        }
    if isinstance(prior, MinGPTQuantizedPrior) and isinstance(cfg, MinGPTQuantizedPriorConfig):
        return {
            "prior_architecture": "mingpt",
            "prior_d_model": int(cfg.d_model),
            "prior_n_heads": int(cfg.n_heads),
            "prior_n_layers": int(cfg.n_layers),
            "prior_d_ff": int(cfg.d_ff),
            "prior_dropout": float(cfg.dropout),
            "prior_atom_vocab_size": int(cfg.atom_vocab_size),
            "prior_coeff_vocab_size": int(cfg.coeff_vocab_size),
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
    d_model: int,
    n_heads: int,
    n_layers: int,
    d_ff: int,
    dropout: float,
    n_global_spatial_tokens: int = 0,
    autoregressive_coeffs: bool = True,
) -> nn.Module:
    """Build a maintained sparse-token prior from a cached token grid."""
    H, W, D = token_cache_grid_shape(cache)
    meta = cache.get("meta", {}) if isinstance(cache, dict) else {}
    real_valued_coeffs = cache.get("coeffs_flat") is not None
    architecture = str(architecture).strip().lower()

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
                coeff_prior_std=float(meta.get("variational_coeff_prior_std", 0.25)),
                coeff_min_std=float(meta.get("variational_coeff_min_std", 0.01)),
                autoregressive_coeffs=bool(autoregressive_coeffs),
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
            )
        )
    if architecture == "mingpt":
        if int(n_global_spatial_tokens) != 0:
            raise ValueError("mingpt sparse prior does not support global spatial tokens.")
        return MinGPTQuantizedPrior(
            MinGPTQuantizedPriorConfig(
                vocab_size=total_vocab_size,
                H=H,
                W=W,
                D=D,
                atom_vocab_size=atom_vocab_size,
                coeff_vocab_size=coeff_vocab_size,
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

    H, W, D = token_cache_grid_shape(cache)
    meta = cache.get("meta", {}) if isinstance(cache, dict) else {}
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
            )
        )

    if architecture == "mingpt":
        if real_valued_coeffs:
            raise ValueError("mingpt checkpoints do not support real-valued sparse coefficients.")
        return MinGPTQuantizedPrior(
            MinGPTQuantizedPriorConfig(
                vocab_size=total_vocab_size,
                H=H,
                W=W,
                D=D,
                atom_vocab_size=atom_vocab_size,
                coeff_vocab_size=coeff_vocab_size,
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
        stage1_decoder_bundle=None,
        log_recon_every_n_steps: int = 500,
    ):
        super().__init__()
        self.prior = prior
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(max(0.0, weight_decay))
        self.warmup_steps = max(0, int(warmup_steps))
        self.min_lr_ratio = float(max(0.0, min(float(min_lr_ratio), 1.0)))
        self.atom_loss_weight = float(atom_loss_weight)
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
        self.coeff_loss_type = self._resolve_coeff_loss_type(coeff_loss_type)
        self._stage1_decoder_bundle = stage1_decoder_bundle
        self.log_recon_every_n_steps = max(0, int(log_recon_every_n_steps))
        self._last_recon_logged_step = -1
        self._cached_val_batch = None
        self.save_hyperparameters(ignore=["prior", "stage1_decoder_bundle"])
        self.save_hyperparameters(_prior_checkpoint_hparams(prior))

    def _resolve_coeff_loss_type(self, requested: Optional[str]) -> Optional[str]:
        text = "" if requested is None else str(requested).strip().lower()
        if not getattr(self.prior, "real_valued_coeffs", False):
            return None if text in {"", "auto"} else text
        if text in {"", "auto"}:
            return "gaussian_nll" if bool(getattr(self.prior, "gaussian_coeffs", False)) else "mse"
        if text not in {"mse", "huber", "recon_mse", "gt_atom_recon_mse", "gaussian_nll"}:
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

    def _real_valued_shared_step(self, batch, prefix: str) -> torch.Tensor:
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise ValueError("Real-valued sparse-token training expects batches of (tokens_flat, coeffs_flat).")

        tok_flat = batch[0].to(self.device, dtype=torch.long, non_blocking=True)
        coeff_flat = batch[1].to(self.device, dtype=torch.float32, non_blocking=True)
        bsz = tok_flat.size(0)
        cfg = self.prior.cfg
        # Token indices are integer tensors with no gradient; a single contiguous
        # copy is sufficient for all read-only uses (CE targets, forward input,
        # mask tokens).  Coefficient grid is float but only read, never written.
        tok_grid = tok_flat.view(bsz, int(cfg.H) * int(cfg.W), int(cfg.D)).clone()
        coeff_grid = coeff_flat.view(bsz, int(cfg.H) * int(cfg.W), int(cfg.D))

        loss_type = str(self.coeff_loss_type or "mse")
        needs_rollout_features = loss_type in {"recon_mse", "gt_atom_recon_mse"}
        if needs_rollout_features:
            forward_out = self.prior(
                tok_grid,
                coeff_grid,
                mask_tokens=tok_grid,
                return_features=True,
            )
            if bool(getattr(self.prior, "gaussian_coeffs", False)):
                atom_logits, coeff_pred, coeff_logvar_pred, depth_h = forward_out
            else:
                atom_logits, coeff_pred, depth_h = forward_out
                coeff_logvar_pred = None
        else:
            forward_out = self.prior(tok_grid, coeff_grid)
            if bool(getattr(self.prior, "gaussian_coeffs", False)):
                atom_logits, coeff_pred, coeff_logvar_pred = forward_out
            else:
                atom_logits, coeff_pred = forward_out
                coeff_logvar_pred = None
            depth_h = None

        ce_loss = F.cross_entropy(
            atom_logits.reshape(-1, int(cfg.vocab_size)),
            tok_grid.reshape(-1),
        )
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
            if self._stage1_decoder_bundle is None:
                raise RuntimeError(
                    f"coeff_loss_type={loss_type!r} requires a stage-1 decoder bundle for sparse latent reconstruction."
                )
            if loss_type == "recon_mse":
                rollout_out = self.prior(
                    tok_grid,
                    coeff_grid,
                    mask_tokens=tok_grid,
                    return_features=True,
                )
                if bool(getattr(self.prior, "gaussian_coeffs", False)):
                    rollout_atom_logits, _, _, rollout_depth_h = rollout_out
                else:
                    rollout_atom_logits, _, rollout_depth_h = rollout_out
                pred_atoms = rollout_atom_logits.argmax(dim=-1)
                pred_coeff = self._clamp_coeffs(
                    self.prior.predict_coeffs_for_atoms(rollout_depth_h, pred_atoms)
                )
                atoms_for_latent = pred_atoms.view(bsz, int(cfg.H), int(cfg.W), int(cfg.D))
            else:
                atoms_for_latent = tok_grid.view(
                    bsz, int(cfg.H), int(cfg.W), int(cfg.D)
                )
            pred_latent = reconstruct_stage2_sparse_latent(
                self._stage1_decoder_bundle,
                atoms_for_latent,
                pred_coeff.view(bsz, int(cfg.H), int(cfg.W), int(cfg.D)),
                device=self.device,
            )
            with torch.no_grad():
                target_latent = reconstruct_stage2_sparse_latent(
                    self._stage1_decoder_bundle,
                    tok_grid.view(bsz, int(cfg.H), int(cfg.W), int(cfg.D)),
                    target_coeff.view(bsz, int(cfg.H), int(cfg.W), int(cfg.D)),
                    device=self.device,
                )
            coeff_reg_loss = F.mse_loss(pred_latent, target_latent)

        loss = ce_loss + self.coeff_loss_weight * coeff_reg_loss
        atom_preds = atom_logits.argmax(dim=-1)
        atom_accuracy = (atom_preds == tok_grid).float().mean()
        coeff_mae = (pred_coeff - target_coeff).abs().mean()

        log_kwargs = dict(
            on_step=(prefix == "train"),
            on_epoch=True,
            sync_dist=True,
            batch_size=bsz,
            prog_bar=(prefix != "test"),
        )
        self.log(f"{prefix}/loss", loss, **log_kwargs)
        self.log(f"{prefix}/ce_loss", ce_loss, **log_kwargs)
        self.log(f"{prefix}/atom_accuracy", atom_accuracy, **log_kwargs)
        self.log(f"{prefix}/coeff_reg_loss", coeff_reg_loss, **log_kwargs)
        self.log(f"{prefix}/coeff_mae", coeff_mae, **log_kwargs)
        if loss_type == "mse":
            self.log(f"{prefix}/coeff_mse_loss", coeff_reg_loss, **log_kwargs)
        elif loss_type == "gaussian_nll":
            self.log(f"{prefix}/coeff_gaussian_nll", coeff_reg_loss, **log_kwargs)
        elif loss_type in {"recon_mse", "gt_atom_recon_mse"}:
            self.log(f"{prefix}/recon_mse_loss", coeff_reg_loss, **log_kwargs)
        else:
            self.log(f"{prefix}/coeff_huber_loss", coeff_reg_loss, **log_kwargs)
        return loss

    def configure_optimizers(self):
        optimizer_cls = torch.optim.AdamW if self.weight_decay > 0 else torch.optim.Adam
        optimizer = optimizer_cls(self.prior.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.warmup_steps <= 0 and self.min_lr_ratio >= 1.0:
            return optimizer

        estimated_steps = getattr(self.trainer, "estimated_stepping_batches", None)
        total_steps = max(1, int(estimated_steps or 1))
        warmup_steps = min(self.warmup_steps, max(0, total_steps - 1))
        min_lr_ratio = self.min_lr_ratio

        def lr_lambda(step: int) -> float:
            cur_step = int(step) + 1
            if warmup_steps > 0 and cur_step <= warmup_steps:
                return max(0.01, cur_step / float(max(1, warmup_steps)))
            progress = (cur_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def _shared_step(self, batch, prefix: str) -> torch.Tensor:
        if getattr(self.prior, "real_valued_coeffs", False):
            return self._real_valued_shared_step(batch, prefix)

        tok_flat = batch[0] if isinstance(batch, (tuple, list)) else batch
        tok_flat = tok_flat.to(self.device, dtype=torch.long, non_blocking=True)
        bsz = tok_flat.size(0)
        cfg = self.prior.cfg
        tok_grid = tok_flat.view(bsz, int(cfg.H) * int(cfg.W), int(cfg.D))

        logits = self.prior(tok_grid)
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
        atom_accuracy = (preds[..., 0::2] == tok_grid[..., 0::2]).float().mean()
        coeff_accuracy = (preds[..., 1::2] == tok_grid[..., 1::2]).float().mean()

        log_kwargs = dict(
            on_step=(prefix == "train"),
            on_epoch=True,
            sync_dist=True,
            batch_size=bsz,
            prog_bar=(prefix != "test"),
        )
        self.log(f"{prefix}/loss", loss, **log_kwargs)
        self.log(f"{prefix}/ce_loss", ce_loss, **log_kwargs)
        self.log(f"{prefix}/accuracy", accuracy, **log_kwargs)
        self.log(f"{prefix}/atom_accuracy", atom_accuracy, **log_kwargs)
        self.log(f"{prefix}/coeff_accuracy", coeff_accuracy, **log_kwargs)
        if atom_ce_loss is not None:
            self.log(f"{prefix}/atom_ce_loss", atom_ce_loss, **log_kwargs)
        if coeff_ce_loss is not None:
            self.log(f"{prefix}/coeff_ce_loss", coeff_ce_loss, **log_kwargs)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "val")
        if batch_idx == 0:
            self._cached_val_batch = batch
        return loss

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        step = self.global_step
        if (
            self.log_recon_every_n_steps > 0
            and self._cached_val_batch is not None
            and step > 0
            and step != self._last_recon_logged_step
            and step % self.log_recon_every_n_steps == 0
        ):
            self._log_recon_images(self._cached_val_batch)
            self._last_recon_logged_step = step
        self._cached_val_batch = None

    @torch.no_grad()
    def _log_recon_images(self, batch, max_images: int = 8):
        """Log ground-truth vs predicted reconstruction images to W&B."""
        if not getattr(self.trainer, "is_global_zero", False):
            return
        logger = getattr(self, "logger", None)
        experiment = getattr(logger, "experiment", None) if logger else None
        if experiment is None or not hasattr(experiment, "log"):
            return
        if self._stage1_decoder_bundle is None:
            return

        cfg = self.prior.cfg
        real_valued = getattr(self.prior, "real_valued_coeffs", False)
        H, W, D = int(cfg.H), int(cfg.W), int(cfg.D)

        tok_flat = batch[0].to(self.device, dtype=torch.long, non_blocking=True)
        bsz = min(tok_flat.size(0), max_images)
        tok_flat = tok_flat[:bsz]
        tok_grid = tok_flat.view(bsz, H * W, D)

        if real_valued:
            coeff_flat = batch[1].to(self.device, dtype=torch.float32, non_blocking=True)[:bsz]
            coeff_grid = coeff_flat.view(bsz, H * W, D)
            # Ground-truth reconstruction
            gt_images = decode_stage2_outputs(
                self._stage1_decoder_bundle,
                tok_grid.view(bsz, H, W, D),
                coeff_grid.view(bsz, H, W, D),
                device=self.device,
            )
            # Predicted reconstruction: run through prior, take argmax atoms + predicted coeffs
            out = self.prior(tok_grid, coeff_grid, mask_tokens=tok_grid, return_features=True)
            if bool(getattr(self.prior, "gaussian_coeffs", False)):
                atom_logits, _, _, depth_h = out
            else:
                atom_logits, _, depth_h = out
            pred_atoms = atom_logits.argmax(dim=-1)
            pred_coeffs = self._clamp_coeffs(
                self.prior.predict_coeffs_for_atoms(depth_h, pred_atoms)
            )
            pred_images = decode_stage2_outputs(
                self._stage1_decoder_bundle,
                pred_atoms.view(bsz, H, W, D),
                pred_coeffs.view(bsz, H, W, D),
                device=self.device,
            )
        else:
            # Quantized path: decode ground-truth tokens
            gt_images = decode_stage2_outputs(
                self._stage1_decoder_bundle,
                tok_grid.view(bsz, H, W, D),
                device=self.device,
            )
            # Predicted tokens via argmax
            logits = self.prior(tok_grid)
            pred_tokens = logits.argmax(dim=-1)
            pred_images = decode_stage2_outputs(
                self._stage1_decoder_bundle,
                pred_tokens.view(bsz, H, W, D),
                device=self.device,
            )

        gt_images = gt_images.detach().cpu().float()
        pred_images = pred_images.detach().cpu().float()

        # Normalize from [-1,1] to [0,1] for display
        gt_disp = ((gt_images + 1.0) / 2.0).clamp(0.0, 1.0)
        pred_disp = ((pred_images + 1.0) / 2.0).clamp(0.0, 1.0)

        nrow = min(8, bsz)
        gt_grid = torchvision.utils.make_grid(gt_disp, nrow=nrow)
        pred_grid = torchvision.utils.make_grid(pred_disp, nrow=nrow)

        gt_grid = torch.nan_to_num(gt_grid, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        pred_grid = torch.nan_to_num(pred_grid, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)

        import wandb as _wandb
        experiment.log(
            {
                "val/recon_images": [
                    _wandb.Image(gt_grid.permute(1, 2, 0).numpy(), caption="Ground Truth"),
                    _wandb.Image(pred_grid.permute(1, 2, 0).numpy(), caption="Predicted"),
                ],
                "global_step": self.global_step,
            },
        )

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    @torch.no_grad()
    def generate_tokens(
        self,
        batch_size: int,
        *,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        generated = self.generate_sparse_codes(
            batch_size,
            temperature=temperature,
            top_k=top_k,
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
            show_progress=False,
        )
