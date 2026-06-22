"""Experimental diffusion prior for real-valued sparse coefficients.

This module intentionally keeps the stage-1 sparse autoencoder fixed and trains
only a continuous DDPM-style prior over cached sparse coefficient values. Atom
ids are treated as conditioning; unconditional generation uses an empirical
support bank sampled from the token cache.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SparseCoeffDiffusionConfig:
    H: int
    W: int
    D: int
    atom_vocab_size: int
    hidden_channels: int = 128
    atom_embed_dim: int = 16
    time_embed_dim: int = 128
    n_res_blocks: int = 6
    dropout: float = 0.0
    num_timesteps: int = 1000
    beta_start: float = 1.0e-4
    beta_end: float = 2.0e-2
    coeff_mean: float = 0.0
    coeff_std: float = 1.0


def _valid_group_count(channels: int, preferred: int = 8) -> int:
    channels = int(channels)
    for groups in range(min(int(preferred), channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.to(torch.float32).view(-1)
        half = self.dim // 2
        if half <= 0:
            return timesteps[:, None]
        scale = math.log(10000.0) / float(max(1, half - 1))
        freqs = torch.exp(
            -torch.arange(half, device=timesteps.device, dtype=torch.float32) * scale
        )
        args = timesteps[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.size(-1) < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.size(-1)))
        return emb


class CoeffResBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int, dropout: float):
        super().__init__()
        groups = _valid_group_count(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(int(time_dim), channels)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.dropout = nn.Dropout(float(dropout))
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb).view(t_emb.size(0), -1, 1, 1)
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return x + h


class SparseCoeffDenoiserPrior(nn.Module):
    """Convolutional epsilon-predictor for sparse coefficient grids.

    Inputs are real-valued sparse coefficient tensors with shape ``[B,H,W,D]``
    or ``[B,H*W,D]`` and matching atom-id tensors. The denoiser predicts the
    additive Gaussian noise used by the DDPM forward process.
    """

    real_valued_coeffs = True
    gaussian_coeffs = False
    autoregressive_coeffs = False

    def __init__(
        self,
        cfg: SparseCoeffDiffusionConfig,
        *,
        support_bank: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.atom_vocab_size = int(cfg.atom_vocab_size)
        self.hidden_channels = int(cfg.hidden_channels)

        if int(cfg.H) <= 0 or int(cfg.W) <= 0 or int(cfg.D) <= 0:
            raise ValueError(f"Invalid sparse coefficient grid shape {(cfg.H, cfg.W, cfg.D)}")
        if int(cfg.atom_vocab_size) <= 0:
            raise ValueError("atom_vocab_size must be positive")
        if int(cfg.num_timesteps) <= 1:
            raise ValueError("num_timesteps must be > 1")

        self.atom_emb = nn.Embedding(int(cfg.atom_vocab_size), int(cfg.atom_embed_dim))
        atom_cond_channels = int(cfg.D) * int(cfg.atom_embed_dim)
        self.atom_in = nn.Conv2d(atom_cond_channels, int(cfg.hidden_channels), kernel_size=1)
        self.coeff_in = nn.Conv2d(int(cfg.D), int(cfg.hidden_channels), kernel_size=3, padding=1)
        self.row_emb = nn.Parameter(torch.zeros(1, int(cfg.hidden_channels), int(cfg.H), 1))
        self.col_emb = nn.Parameter(torch.zeros(1, int(cfg.hidden_channels), 1, int(cfg.W)))

        time_dim = int(cfg.time_embed_dim)
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.blocks = nn.ModuleList(
            [CoeffResBlock(int(cfg.hidden_channels), time_dim, float(cfg.dropout)) for _ in range(int(cfg.n_res_blocks))]
        )
        self.out = nn.Sequential(
            nn.GroupNorm(_valid_group_count(int(cfg.hidden_channels)), int(cfg.hidden_channels)),
            nn.SiLU(),
            nn.Conv2d(int(cfg.hidden_channels), int(cfg.D), kernel_size=3, padding=1),
        )

        betas = torch.linspace(float(cfg.beta_start), float(cfg.beta_end), int(cfg.num_timesteps), dtype=torch.float32)
        if torch.any(betas <= 0) or torch.any(betas >= 1):
            raise ValueError("Diffusion betas must be in (0, 1)")
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1, dtype=torch.float32), alpha_bars[:-1]], dim=0)
        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars).clamp_min(1.0e-12)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("posterior_variance", posterior_variance.clamp_min(0.0))
        self.register_buffer("coeff_mean", torch.tensor(float(cfg.coeff_mean), dtype=torch.float32))
        self.register_buffer("coeff_std", torch.tensor(max(float(cfg.coeff_std), 1.0e-6), dtype=torch.float32))

        if support_bank is None:
            support_bank = torch.empty(0, int(cfg.H) * int(cfg.W) * int(cfg.D), dtype=torch.long)
        self.set_support_bank(support_bank)

    @property
    def num_timesteps(self) -> int:
        return int(self.betas.numel())

    def set_support_bank(self, support_bank: torch.Tensor) -> None:
        if not torch.is_tensor(support_bank):
            raise ValueError("support_bank must be a tensor")
        support_bank = support_bank.to(torch.long).contiguous()
        expected_flat = int(self.cfg.H) * int(self.cfg.W) * int(self.cfg.D)
        if support_bank.numel() == 0:
            support_bank = support_bank.reshape(0, expected_flat)
        elif support_bank.ndim == 4:
            if tuple(support_bank.shape[1:]) != (int(self.cfg.H), int(self.cfg.W), int(self.cfg.D)):
                raise ValueError(
                    f"Expected support_bank grid shape (*,{self.cfg.H},{self.cfg.W},{self.cfg.D}), "
                    f"got {tuple(support_bank.shape)}"
                )
            support_bank = support_bank.reshape(support_bank.size(0), expected_flat)
        elif support_bank.ndim != 2 or int(support_bank.size(1)) != expected_flat:
            raise ValueError(f"support_bank must have shape [N,{expected_flat}] or [N,H,W,D]")
        if support_bank.numel() > 0:
            if int(support_bank.min().item()) < 0 or int(support_bank.max().item()) >= int(self.cfg.atom_vocab_size):
                raise ValueError("support_bank contains atom ids outside the configured vocabulary")
        # Keep the empirical support bank out of the state dict and off GPU.
        # Large sparse grids can make this tensor much bigger than the denoiser.
        self.support_bank = support_bank.cpu()

    def _extract(self, values: torch.Tensor, timesteps: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        out = values.gather(0, timesteps.to(torch.long).view(-1))
        return out.view(timesteps.size(0), *([1] * (target.ndim - 1)))

    def _grid(self, tensor: torch.Tensor, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if tensor.ndim == 4:
            expected = (int(self.cfg.H), int(self.cfg.W), int(self.cfg.D))
            if tuple(tensor.shape[1:]) != expected:
                raise ValueError(f"Expected tensor grid shape [B,{expected}], got {tuple(tensor.shape)}")
            out = tensor
        elif tensor.ndim == 3:
            if int(tensor.size(1)) != int(self.cfg.H) * int(self.cfg.W) or int(tensor.size(2)) != int(self.cfg.D):
                raise ValueError(
                    f"Expected tensor shape [B,{self.cfg.H * self.cfg.W},{self.cfg.D}], got {tuple(tensor.shape)}"
                )
            out = tensor.view(tensor.size(0), int(self.cfg.H), int(self.cfg.W), int(self.cfg.D))
        elif tensor.ndim == 2:
            expected_flat = int(self.cfg.H) * int(self.cfg.W) * int(self.cfg.D)
            if int(tensor.size(1)) != expected_flat:
                raise ValueError(f"Expected flat tensor shape [B,{expected_flat}], got {tuple(tensor.shape)}")
            out = tensor.view(tensor.size(0), int(self.cfg.H), int(self.cfg.W), int(self.cfg.D))
        else:
            raise ValueError(f"Expected rank-2, rank-3, or rank-4 tensor, got shape {tuple(tensor.shape)}")
        return out if dtype is None else out.to(dtype=dtype)

    def normalize_coeffs(self, coeffs: torch.Tensor) -> torch.Tensor:
        return (coeffs - self.coeff_mean.to(coeffs.device, coeffs.dtype)) / self.coeff_std.to(coeffs.device, coeffs.dtype)

    def denormalize_coeffs(self, coeffs: torch.Tensor) -> torch.Tensor:
        return coeffs * self.coeff_std.to(coeffs.device, coeffs.dtype) + self.coeff_mean.to(coeffs.device, coeffs.dtype)

    def q_sample(self, x0: torch.Tensor, timesteps: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        return (
            self._extract(self.sqrt_alpha_bars, timesteps, x0) * x0
            + self._extract(self.sqrt_one_minus_alpha_bars, timesteps, x0) * noise
        )

    def predict_x0_from_eps(self, x_t: torch.Tensor, timesteps: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        sqrt_alpha_bar = self._extract(self.sqrt_alpha_bars, timesteps, x_t)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alpha_bars, timesteps, x_t)
        return (x_t - sqrt_one_minus * eps) / sqrt_alpha_bar.clamp_min(1.0e-8)

    def forward(self, atom_ids: torch.Tensor, noisy_coeffs: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        atoms = self._grid(atom_ids).to(device=noisy_coeffs.device, dtype=torch.long)
        coeffs = self._grid(noisy_coeffs, dtype=noisy_coeffs.dtype)
        if coeffs.device != atoms.device:
            coeffs = coeffs.to(atoms.device)
        if int(atoms.min().item()) < 0 or int(atoms.max().item()) >= int(self.cfg.atom_vocab_size):
            raise ValueError("atom_ids contains values outside the configured vocabulary")

        bsz = coeffs.size(0)
        coeff_ch = coeffs.permute(0, 3, 1, 2).contiguous()
        atom_emb = self.atom_emb(atoms)
        atom_ch = atom_emb.permute(0, 3, 4, 1, 2).reshape(
            bsz,
            int(self.cfg.D) * int(self.cfg.atom_embed_dim),
            int(self.cfg.H),
            int(self.cfg.W),
        )
        h = self.coeff_in(coeff_ch) + self.atom_in(atom_ch) + self.row_emb + self.col_emb
        t_emb = self.time_emb(timesteps.to(device=coeffs.device))
        for block in self.blocks:
            h = block(h, t_emb)
        eps = self.out(h)
        return eps.permute(0, 2, 3, 1).contiguous()

    def sample_support(self, batch_size: int) -> torch.Tensor:
        if self.support_bank.numel() <= 0:
            raise RuntimeError("Cannot sample sparse supports: support_bank is empty.")
        idx = torch.randint(
            0,
            int(self.support_bank.size(0)),
            (int(batch_size),),
            device=self.support_bank.device,
        )
        return self.support_bank.index_select(0, idx).view(
            int(batch_size),
            int(self.cfg.H),
            int(self.cfg.W),
            int(self.cfg.D),
        )

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        *,
        atom_ids: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        steps: Optional[int] = None,
        top_k: Optional[int] = None,
        coeff_temperature: Optional[float] = None,
        coeff_sample_mode: Optional[str] = None,
        show_progress: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del top_k, coeff_sample_mode, show_progress
        batch_size = int(batch_size)
        if atom_ids is None:
            atoms = self.sample_support(batch_size).to(device=self.betas.device)
        else:
            atoms = self._grid(atom_ids).to(device=self.betas.device, dtype=torch.long)
            if int(atoms.size(0)) != batch_size:
                raise ValueError(f"atom_ids batch size {int(atoms.size(0))} does not match {batch_size}")

        steps = self.num_timesteps if steps is None else int(steps)
        if steps <= 0 or steps > self.num_timesteps:
            raise ValueError(f"steps must be in [1,{self.num_timesteps}], got {steps}")
        schedule = torch.linspace(self.num_timesteps - 1, 0, steps, device=self.betas.device).round().to(torch.long)

        temp = float(coeff_temperature if coeff_temperature is not None else temperature)
        x = torch.randn(
            batch_size,
            int(self.cfg.H),
            int(self.cfg.W),
            int(self.cfg.D),
            device=self.betas.device,
            dtype=self.coeff_mean.dtype,
        ) * max(temp, 1.0e-6)

        for t_scalar in schedule:
            t = torch.full((batch_size,), int(t_scalar.item()), device=x.device, dtype=torch.long)
            eps = self(atoms, x, t)
            beta_t = self._extract(self.betas, t, x)
            sqrt_recip_alpha_t = self._extract(self.sqrt_recip_alphas, t, x)
            sqrt_one_minus_alpha_bar_t = self._extract(self.sqrt_one_minus_alpha_bars, t, x)
            model_mean = sqrt_recip_alpha_t * (x - beta_t * eps / sqrt_one_minus_alpha_bar_t.clamp_min(1.0e-8))
            if int(t_scalar.item()) > 0:
                noise = torch.randn_like(x)
                var = self._extract(self.posterior_variance, t, x)
                x = model_mean + torch.sqrt(var).to(x.dtype) * noise
            else:
                x = model_mean

        coeffs = self.denormalize_coeffs(x)
        return atoms.view(batch_size, int(self.cfg.H) * int(self.cfg.W), int(self.cfg.D)), coeffs.view(
            batch_size,
            int(self.cfg.H) * int(self.cfg.W),
            int(self.cfg.D),
        )


def infer_sparse_coeff_diffusion_cache_metadata(cache: dict) -> tuple[tuple[int, int, int], int]:
    if not isinstance(cache, dict):
        raise ValueError("cache must be a dictionary")
    tokens_flat = cache.get("tokens_flat")
    coeffs_flat = cache.get("coeffs_flat")
    shape = cache.get("shape")
    if coeffs_flat is None:
        raise ValueError("Sparse coefficient diffusion requires a real-valued cache with coeffs_flat.")
    if not torch.is_tensor(tokens_flat) or tokens_flat.ndim != 2:
        raise ValueError("cache['tokens_flat'] must be a rank-2 tensor")
    if not torch.is_tensor(coeffs_flat) or coeffs_flat.shape != tokens_flat.shape:
        raise ValueError("cache['coeffs_flat'] must be a tensor with the same shape as tokens_flat")
    if not isinstance(shape, (tuple, list)) or len(shape) != 3:
        raise ValueError("cache['shape'] must be a length-3 tuple/list")
    grid_shape = (int(shape[0]), int(shape[1]), int(shape[2]))
    expected_flat = grid_shape[0] * grid_shape[1] * grid_shape[2]
    if int(tokens_flat.size(1)) != expected_flat:
        raise ValueError(f"Cache flat width {int(tokens_flat.size(1))} does not match shape {grid_shape}")

    meta = cache.get("meta", {}) if isinstance(cache.get("meta", {}), dict) else {}
    atom_vocab = int(meta.get("num_atoms") or meta.get("atom_vocab_size") or 0)
    if atom_vocab <= 0:
        atom_vocab = int(tokens_flat.max().item()) + 1
    return grid_shape, atom_vocab


def build_sparse_coeff_diffusion_prior_from_cache(
    cache: dict,
    *,
    hidden_channels: int = 128,
    atom_embed_dim: int = 16,
    time_embed_dim: int = 128,
    n_res_blocks: int = 6,
    dropout: float = 0.0,
    num_timesteps: int = 1000,
    beta_start: float = 1.0e-4,
    beta_end: float = 2.0e-2,
    coeff_mean: Optional[float] = None,
    coeff_std: Optional[float] = None,
    support_bank: Optional[torch.Tensor] = None,
) -> SparseCoeffDenoiserPrior:
    grid_shape, atom_vocab = infer_sparse_coeff_diffusion_cache_metadata(cache)
    coeffs_flat = cache["coeffs_flat"].to(torch.float32)
    if coeff_mean is None:
        coeff_mean = float(coeffs_flat.mean().item())
    if coeff_std is None:
        coeff_std = float(coeffs_flat.std(unbiased=False).clamp_min(1.0e-6).item())
    cfg = SparseCoeffDiffusionConfig(
        H=grid_shape[0],
        W=grid_shape[1],
        D=grid_shape[2],
        atom_vocab_size=atom_vocab,
        hidden_channels=int(hidden_channels),
        atom_embed_dim=int(atom_embed_dim),
        time_embed_dim=int(time_embed_dim),
        n_res_blocks=int(n_res_blocks),
        dropout=float(dropout),
        num_timesteps=int(num_timesteps),
        beta_start=float(beta_start),
        beta_end=float(beta_end),
        coeff_mean=float(coeff_mean),
        coeff_std=float(coeff_std),
    )
    return SparseCoeffDenoiserPrior(cfg, support_bank=support_bank)


class SparseCoeffDiffusionModule(pl.LightningModule):
    """Lightning wrapper for the experimental sparse coefficient diffusion prior."""

    def __init__(
        self,
        prior: SparseCoeffDenoiserPrior,
        *,
        learning_rate: float = 2.0e-4,
        weight_decay: float = 1.0e-2,
    ):
        super().__init__()
        self.prior = prior
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(max(0.0, weight_decay))
        cfg = prior.cfg
        self.save_hyperparameters(
            {
                "prior_architecture": "sparse_coeff_diffusion",
                "prior_H": int(cfg.H),
                "prior_W": int(cfg.W),
                "prior_D": int(cfg.D),
                "prior_atom_vocab_size": int(cfg.atom_vocab_size),
                "prior_hidden_channels": int(cfg.hidden_channels),
                "prior_atom_embed_dim": int(cfg.atom_embed_dim),
                "prior_time_embed_dim": int(cfg.time_embed_dim),
                "prior_n_res_blocks": int(cfg.n_res_blocks),
                "prior_dropout": float(cfg.dropout),
                "prior_num_timesteps": int(cfg.num_timesteps),
                "prior_beta_start": float(cfg.beta_start),
                "prior_beta_end": float(cfg.beta_end),
                "prior_coeff_mean": float(cfg.coeff_mean),
                "prior_coeff_std": float(cfg.coeff_std),
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
            }
        )

    def _step(self, batch, prefix: str) -> torch.Tensor:
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise ValueError("Sparse coefficient diffusion expects batches of (tokens_flat, coeffs_flat).")
        tok_flat = batch[0].to(self.device, dtype=torch.long, non_blocking=True)
        coeff_flat = batch[1].to(self.device, dtype=torch.float32, non_blocking=True)
        bsz = int(tok_flat.size(0))
        cfg = self.prior.cfg
        atoms = tok_flat.view(bsz, int(cfg.H), int(cfg.W), int(cfg.D))
        x0 = coeff_flat.view(bsz, int(cfg.H), int(cfg.W), int(cfg.D))
        x0 = self.prior.normalize_coeffs(x0)
        noise = torch.randn_like(x0)
        t = torch.randint(0, self.prior.num_timesteps, (bsz,), device=self.device, dtype=torch.long)
        x_t = self.prior.q_sample(x0, t, noise)
        pred_noise = self.prior(atoms, x_t, t)
        loss = F.mse_loss(pred_noise, noise)
        with torch.no_grad():
            x0_pred = self.prior.predict_x0_from_eps(x_t, t, pred_noise)
            x0_mse = F.mse_loss(x0_pred, x0)
            coeff_mae = (self.prior.denormalize_coeffs(x0_pred) - coeff_flat.view_as(x0)).abs().mean()
        self.log(f"{prefix}/loss", loss, on_step=(prefix == "train"), on_epoch=True, batch_size=bsz, prog_bar=True)
        self.log(f"{prefix}/eps_mse", loss, on_step=False, on_epoch=True, batch_size=bsz)
        self.log(f"{prefix}/x0_mse", x0_mse, on_step=False, on_epoch=True, batch_size=bsz)
        self.log(f"{prefix}/coeff_mae", coeff_mae, on_step=False, on_epoch=True, batch_size=bsz)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        optimizer_cls = torch.optim.AdamW if self.weight_decay > 0 else torch.optim.Adam
        return optimizer_cls(self.prior.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    @torch.no_grad()
    def generate_sparse_codes(
        self,
        batch_size: int,
        *,
        atom_ids: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        steps: Optional[int] = None,
        top_k: Optional[int] = None,
        coeff_temperature: Optional[float] = None,
        coeff_sample_mode: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.prior.generate(
            int(batch_size),
            atom_ids=atom_ids,
            temperature=temperature,
            steps=steps,
            top_k=top_k,
            coeff_temperature=coeff_temperature,
            coeff_sample_mode=coeff_sample_mode,
        )
