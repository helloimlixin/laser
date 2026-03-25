"""Lightning wrapper and factory helpers for sparse-token stage-2 priors."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mingpt_prior import MinGPTQuantizedPrior, MinGPTQuantizedPriorConfig
from .spatial_prior import SpatialDepthPrior, SpatialDepthPriorConfig


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

    resolved_coeff_vocab = int(coeff_vocab_size or meta.get("n_bins") or meta.get("coeff_vocab_size") or 0)
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
) -> nn.Module:
    """Build a maintained sparse-token prior from a cached token grid."""
    H, W, D = token_cache_grid_shape(cache)
    if cache.get("coeffs_flat") is not None:
        raise NotImplementedError("Maintained sparse-token prior training currently supports quantized caches only.")

    total_vocab_size, atom_vocab_size, coeff_vocab_size = infer_sparse_vocab_sizes(
        cache,
        total_vocab_size=total_vocab_size,
        atom_vocab_size=atom_vocab_size,
        coeff_vocab_size=coeff_vocab_size,
    )

    architecture = str(architecture).strip().lower()
    if architecture == "spatial_depth":
        return SpatialDepthPrior(
            SpatialDepthPriorConfig(
                vocab_size=total_vocab_size,
                atom_vocab_size=atom_vocab_size,
                coeff_vocab_size=coeff_vocab_size,
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


class SparseTokenPriorModule(pl.LightningModule):
    """Lightning wrapper for quantized sparse-token stage-2 priors."""

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
        self.save_hyperparameters(ignore=["prior"])

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
            raise NotImplementedError("SparseTokenPriorModule currently supports quantized sparse-token caches only.")

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
    ) -> torch.Tensor:
        return self.prior.generate(batch_size=batch_size, temperature=temperature, top_k=top_k, show_progress=False)
