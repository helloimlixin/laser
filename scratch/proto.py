<<<<<<< HEAD
"""
Sparse dictionary autoencoder + transformer prior.

Stage 1: Encoder -> DictionaryLearning bottleneck (batch OMP) -> Decoder
Stage 2: Autoregressive transformer prior over (atom_id, coefficient) stacks

Run:
  python proto.py --stage1_epochs 5 --stage2_epochs 10
"""
import argparse
import math
import os
import socket
import subprocess
import sys
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
torch.set_float32_matmul_precision("high")
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from PIL import Image
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets, transforms, utils
from tqdm import tqdm

try:
    import wandb
except Exception:
    wandb = None


def _pick_free_tcp_port(excluded: Optional[set[int]] = None) -> int:
    excluded = excluded or set()
    for _ in range(32):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("0.0.0.0", 0))
            port = int(sock.getsockname()[1])
            if port not in excluded:
                return port
        except OSError:
            continue
        finally:
            sock.close()
    raise RuntimeError("Unable to allocate a free TCP port.")


def _ensure_free_master_port(tag: str) -> None:
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        try:
            if int(local_rank) != 0:
                return
        except ValueError:
            pass
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    raw = os.environ.get("MASTER_PORT")
    excluded: set[int] = set()
    if raw:
        try:
            excluded.add(int(raw))
        except ValueError:
            pass
    port = _pick_free_tcp_port(excluded)
    os.environ["MASTER_PORT"] = str(port)
    print(f"[{tag}] MASTER_PORT={port}")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FlatImageDataset(Dataset):
    """Recursively loads images from a directory tree."""

    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root}")
        self.image_paths = sorted(
            p for p in self.root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
        )
        if not self.image_paths:
            raise RuntimeError(f"No images found under: {self.root}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        with Image.open(self.image_paths[idx]) as img:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0


# ---------------------------------------------------------------------------
# VQ-VAE style building blocks
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, num_hiddens: int, num_residual_hiddens: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, num_residual_hiddens, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_residual_hiddens)
        self.conv2 = nn.Conv2d(num_residual_hiddens, num_hiddens, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_hiddens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels: int, num_hiddens: int,
                 num_residual_layers: int, num_residual_hiddens: int):
        super().__init__()
        self.layers = nn.ModuleList([
            ResidualBlock(in_channels, num_hiddens, num_residual_hiddens)
            for _ in range(num_residual_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels: int, num_hiddens: int,
                 num_residual_layers: int, num_residual_hiddens: int,
                 num_downsamples: int = 2):
        super().__init__()
        down = []
        ch = in_channels
        for i in range(num_downsamples):
            out_ch = max(1, num_hiddens // 2) if i == 0 else num_hiddens
            down.append(nn.Conv2d(ch, out_ch, 4, stride=2, padding=1))
            ch = out_ch
        self.down = nn.ModuleList(down)
        self.conv = nn.Conv2d(ch, num_hiddens, 3, padding=1)
        self.res = ResidualStack(num_hiddens, num_hiddens,
                                 num_residual_layers, num_residual_hiddens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.down:
            x = F.relu(conv(x))
        return self.res(self.conv(x))


class Decoder(nn.Module):
    def __init__(self, in_channels: int, num_hiddens: int,
                 num_residual_layers: int, num_residual_hiddens: int,
                 out_channels: int = 3, num_upsamples: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_hiddens, 3, padding=1)
        self.res = ResidualStack(num_hiddens, num_hiddens,
                                 num_residual_layers, num_residual_hiddens)
        up = []
        ch = num_hiddens
        for i in range(num_upsamples):
            if i == num_upsamples - 1:
                out_ch = out_channels
            elif i == num_upsamples - 2:
                out_ch = max(1, num_hiddens // 2)
            else:
                out_ch = num_hiddens
            up.append(nn.ConvTranspose2d(ch, out_ch, 4, stride=2, padding=1))
            ch = out_ch
        self.up = nn.ModuleList(up)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res(self.conv(x))
        for i, deconv in enumerate(self.up):
            x = deconv(x)
            if i != len(self.up) - 1:
                x = F.relu(x)
        return x


# ---------------------------------------------------------------------------
# Dictionary Learning bottleneck (batch OMP)
# ---------------------------------------------------------------------------

class DictionaryLearning(nn.Module):
    """Patch-based dictionary learning bottleneck with batch OMP sparse coding.

    When ``patch_size > 1``, non-overlapping ``patch_size x patch_size``
    spatial patches are flattened and encoded jointly, yielding a coarser
    latent grid (H/P x W/P) with higher-dimensional dictionary atoms.
    """

    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        sparsity_level: int = 5,
        commitment_cost: float = 0.25,
        epsilon: float = 1e-10,
        usage_ema_decay: float = 0.99,
        usage_balance_alpha: float = 0.3,
        dead_atom_threshold: float = 5e-4,
        dead_atom_interval: int = 200,
        dead_atom_max_reinit: int = 16,
        patch_size: int = 1,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.sparsity_level = sparsity_level
        self.commitment_cost = commitment_cost
        self.epsilon = epsilon
        self.usage_ema_decay = float(min(max(usage_ema_decay, 0.0), 0.9999))
        self.usage_balance_alpha = float(max(usage_balance_alpha, 0.0))
        self.dead_atom_threshold = float(max(dead_atom_threshold, 0.0))
        self.dead_atom_interval = int(max(dead_atom_interval, 1))
        self.dead_atom_max_reinit = int(max(dead_atom_max_reinit, 0))

        self.patch_size = patch_size
        self.atom_dim = embedding_dim * patch_size * patch_size

        self.dictionary = nn.Parameter(
            torch.randn(self.atom_dim, num_embeddings)
        )

        self.vocab_size = num_embeddings
        self.register_buffer(
            "usage_ema",
            torch.full((num_embeddings,), 1.0 / max(1, num_embeddings)),
            persistent=False,
        )
        self.register_buffer(
            "_steps_since_reinit",
            torch.zeros((), dtype=torch.long),
            persistent=False,
        )
        self._residual_prototypes: Optional[torch.Tensor] = None

    # ---- batch OMP (adapted from bottleneck.py / sparse-vqvae) ----

    def _selection_boost(
        self, device: torch.device, dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if self.usage_balance_alpha <= 0:
            return None
        usage = self.usage_ema.to(device=device, dtype=dtype)
        usage = usage / usage.sum().clamp(min=self.epsilon)
        uniform = 1.0 / max(1.0, float(self.num_embeddings))
        boost = (uniform / usage.clamp(min=self.epsilon)).pow(self.usage_balance_alpha)
        return boost.clamp(max=8.0)

    @torch.no_grad()
    def _update_usage_ema(self, support: torch.Tensor) -> None:
        flat = support.reshape(-1)
        if flat.numel() == 0:
            return
        counts = torch.bincount(flat, minlength=self.num_embeddings).to(
            device=self.usage_ema.device, dtype=self.usage_ema.dtype,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
        probs = counts / counts.sum().clamp(min=self.epsilon)
        self.usage_ema.mul_(self.usage_ema_decay).add_(
            probs * (1.0 - self.usage_ema_decay)
        )

    @torch.no_grad()
    def _cache_residual_prototypes(
        self, signals_bt: torch.Tensor, recon_bt: torch.Tensor,
    ) -> None:
        if self.dead_atom_threshold <= 0 or self.dead_atom_max_reinit <= 0:
            return
        residual = (signals_bt - recon_bt).detach()
        if residual.numel() == 0:
            return
        bank_size = min(
            residual.size(0),
            max(1, 4 * self.dead_atom_max_reinit),
        )
        energy = residual.pow(2).sum(dim=1)
        top_idx = torch.topk(energy, k=bank_size, largest=True).indices
        self._residual_prototypes = residual[top_idx].t().contiguous()

    @torch.no_grad()
    def usage_stats(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        usage = self.usage_ema / self.usage_ema.sum().clamp(min=self.epsilon)
        top1 = usage.max()
        entropy = -(usage * usage.clamp(min=self.epsilon).log()).sum()
        perplexity = entropy.exp()
        threshold = (
            self.dead_atom_threshold
            if self.dead_atom_threshold > 0 else (0.5 / max(1, self.num_embeddings))
        )
        active_frac = (usage > threshold).float().mean()
        return top1, perplexity, active_frac

    @torch.no_grad()
    def revive_dead_atoms(self) -> int:
        if self.dead_atom_threshold <= 0 or self.dead_atom_max_reinit <= 0:
            return 0
        if self._residual_prototypes is None or self._residual_prototypes.numel() == 0:
            return 0

        usage = self.usage_ema / self.usage_ema.sum().clamp(min=self.epsilon)
        dead = torch.nonzero(
            usage < self.dead_atom_threshold, as_tuple=False,
        ).squeeze(1)
        if dead.numel() == 0:
            return 0

        k = min(
            int(dead.numel()),
            int(self.dead_atom_max_reinit),
            int(self._residual_prototypes.size(1)),
        )
        if k <= 0:
            return 0

        order = torch.argsort(usage[dead], descending=False)
        dead = dead[order[:k]]
        pick = torch.randperm(
            self._residual_prototypes.size(1),
            device=self._residual_prototypes.device,
        )[:k]
        repl = self._residual_prototypes[:, pick]
        repl = F.normalize(
            repl + 0.01 * torch.randn_like(repl),
            p=2, dim=0, eps=self.epsilon,
        )
        self.dictionary[:, dead] = repl.to(
            device=self.dictionary.device, dtype=self.dictionary.dtype,
        )
        self.usage_ema[dead] = usage.mean().to(self.usage_ema.dtype)
        self._steps_since_reinit.zero_()
        self._residual_prototypes = None
        return int(k)

    def batch_omp(
        self, X: torch.Tensor, D: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batched Orthogonal Matching Pursuit.

        Args:
            X: [M, B] input signals.
            D: [M, N] dictionary with L2-normalised columns.

        Returns:
            support:      [B, K] atom indices in selection order.
            coefficients: [B, K] corresponding coefficients.
        """
        _, B = X.size()
        Dt = D.t()
        G = Dt @ D
        h_bar = (Dt @ X).t()                       # [B, N]
        h = h_bar.clone()
        L = torch.ones(B, 1, 1, device=X.device, dtype=X.dtype)
        I = torch.zeros(B, 0, dtype=torch.long, device=X.device)
        mask = torch.ones_like(h_bar, dtype=torch.bool)
        gamma = torch.zeros_like(h_bar)
        batch_idx = torch.arange(B, device=X.device)
        selection_boost = self._selection_boost(device=X.device, dtype=X.dtype)

        for k in range(1, self.sparsity_level + 1):
            scores = h.abs() * mask.float()
            if selection_boost is not None:
                scores = scores * selection_boost.unsqueeze(0)
            idx = scores.argmax(dim=1)
            mask[batch_idx, idx] = False
            expanded = batch_idx.unsqueeze(0).expand(k, B).t()

            if k > 1:
                G_col = G[
                    I[batch_idx, :],
                    idx[expanded[..., :-1]],
                ].view(B, k - 1, 1)
                w = torch.linalg.solve_triangular(
                    L, G_col, upper=False,
                ).view(-1, 1, k - 1)
                w_corner = torch.sqrt(
                    torch.clamp(1 - (w ** 2).sum(dim=2, keepdim=True), min=1e-12)
                )
                zeros = torch.zeros(B, k - 1, 1, device=X.device, dtype=X.dtype)
                L = torch.cat([
                    torch.cat([L, zeros], dim=2),
                    torch.cat([w, w_corner], dim=2),
                ], dim=1)

            I = torch.cat([I, idx.unsqueeze(1)], dim=1)
            h_stack = h_bar[expanded, I[batch_idx, :]].view(B, k, 1)
            gamma_stack = torch.cholesky_solve(h_stack, L)
            gamma[batch_idx.unsqueeze(1), I[batch_idx]] = (
                gamma_stack[batch_idx].squeeze(-1)
            )

            beta = (
                gamma[batch_idx.unsqueeze(1), I[batch_idx]]
                .unsqueeze(1)
                .bmm(G[I[batch_idx], :])
                .squeeze(1)
            )
            h = h_bar - beta

        coeffs_ordered = gamma[batch_idx[:, None], I]
        return I, coeffs_ordered

    # ---- patch helpers ----

    def _patchify(self, z: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """[B, C, H, W] → [B, Hp, Wp, atom_dim] and return (patches, Hp, Wp)."""
        B, C, H, W = z.shape
        P = self.patch_size
        if P <= 1:
            return z.permute(0, 2, 3, 1).contiguous(), H, W
        Hp, Wp = H // P, W // P
        return (
            z.reshape(B, C, Hp, P, Wp, P)
            .permute(0, 2, 4, 1, 3, 5)
            .contiguous()
            .reshape(B, Hp, Wp, self.atom_dim),
            Hp, Wp,
        )

    def _unpatchify(
        self, flat: torch.Tensor, B: int, Hp: int, Wp: int,
    ) -> torch.Tensor:
        """[B*Hp*Wp, atom_dim] → [B, C, H, W]."""
        P = self.patch_size
        C = self.embedding_dim
        if P <= 1:
            return flat.view(B, Hp, Wp, C).permute(0, 3, 1, 2).contiguous()
        return (
            flat.view(B, Hp, Wp, C, P, P)
            .permute(0, 3, 1, 4, 2, 5)
            .contiguous()
            .reshape(B, C, Hp * P, Wp * P)
        )

    # ---- reconstruction helper ----

    def _reconstruct(
        self, support: torch.Tensor, coeffs: torch.Tensor,
        dictionary: torch.Tensor,
    ) -> torch.Tensor:
        dict_t = dictionary.t()                     # [N, atom_dim]
        atoms = dict_t[support.long()]              # [..., K, atom_dim]
        return (atoms * coeffs.unsqueeze(-1)).sum(dim=-2)

    # ---- forward (matches bottleneck.py style) ----

    def forward(
        self, z_e: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_e: [B, C, H, W] encoder output.

        Returns:
            z_dl:    [B, C, H, W]    STE-quantised latent (full resolution).
            loss:    scalar           bottleneck loss.
            support: [B, Hp, Wp, K]  ordered atom indices (patch grid).
            coeffs:  [B, Hp, Wp, K]  ordered coefficients (patch grid).
        """
        if z_e.dim() != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(z_e.shape)}")
        B, C, H, W = z_e.shape
        P = self.patch_size
        if P > 1 and (H % P != 0 or W % P != 0):
            raise ValueError(
                f"Spatial dims ({H},{W}) not divisible by patch_size={P}"
            )

        patches, Hp, Wp = self._patchify(z_e)
        signals = patches.reshape(-1, self.atom_dim).t()
        dictionary = F.normalize(self.dictionary, p=2, dim=0, eps=self.epsilon)

        with torch.no_grad():
            support, coeffs = self.batch_omp(signals, dictionary)

        recon = self._reconstruct(support, coeffs, dictionary)
        if self.training:
            with torch.no_grad():
                self._update_usage_ema(support)
                self._cache_residual_prototypes(signals.t(), recon)
                self._steps_since_reinit.add_(1)
        z_dl = self._unpatchify(recon, B, Hp, Wp)

        dl_latent_loss = F.mse_loss(z_dl, z_e.detach())
        e_latent_loss = F.mse_loss(z_dl.detach(), z_e)
        loss = dl_latent_loss + self.commitment_cost * e_latent_loss

        z_dl = z_e + (z_dl - z_e).detach()

        K = self.sparsity_level
        return z_dl, loss, support.view(B, Hp, Wp, K), coeffs.view(B, Hp, Wp, K)

    @torch.no_grad()
    def decode_sparse_codes(
        self, support: torch.Tensor, coeffs: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct [B, C, H, W] from atom ids + coefficients."""
        B, Hp, Wp, K = support.shape
        dictionary = F.normalize(self.dictionary, p=2, dim=0, eps=self.epsilon)
        recon = self._reconstruct(
            support.reshape(-1, K),
            coeffs.reshape(-1, K).to(dictionary.dtype),
            dictionary,
        )
        return self._unpatchify(recon, B, Hp, Wp)


# ---------------------------------------------------------------------------
# Stage-1 model: Encoder + DictionaryLearning + Decoder
# ---------------------------------------------------------------------------

class SparseDictAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_hiddens: int = 128,
        num_downsamples: int = 2,
        num_residual_layers: int = 2,
        num_residual_hiddens: int = 32,
        embedding_dim: int = 64,
        num_embeddings: int = 256,
        sparsity_level: int = 4,
        commitment_cost: float = 0.25,
        usage_ema_decay: float = 0.99,
        usage_balance_alpha: float = 0.3,
        dead_atom_threshold: float = 5e-4,
        dead_atom_interval: int = 200,
        dead_atom_max_reinit: int = 16,
        bottleneck_patch_size: int = 1,
        n_coeff_bins: int = 1024,
        coeff_mu: float = 255.0,
        coeff_max_val: float = 24.0,
        discretize_sparse_coeffs: bool = True,
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels, num_hiddens,
            num_residual_layers, num_residual_hiddens,
            num_downsamples,
        )
        self.pre = nn.Conv2d(num_hiddens, embedding_dim, 1)
        self.bottleneck = DictionaryLearning(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            sparsity_level=sparsity_level,
            commitment_cost=commitment_cost,
            usage_ema_decay=usage_ema_decay,
            usage_balance_alpha=usage_balance_alpha,
            dead_atom_threshold=dead_atom_threshold,
            dead_atom_interval=dead_atom_interval,
            dead_atom_max_reinit=dead_atom_max_reinit,
            patch_size=bottleneck_patch_size,
        )
        self.post = nn.Conv2d(embedding_dim, num_hiddens, 3, padding=1)
        self.decoder = Decoder(
            num_hiddens, num_hiddens,
            num_residual_layers, num_residual_hiddens,
            in_channels, num_downsamples,
        )
        self.discretize_sparse_coeffs = bool(discretize_sparse_coeffs)
        self.coeff_quantizer = CoefficientQuantizer(
            n_coeff_bins, coeff_max_val, coeff_mu,
        )

    def _discretize_coeffs(self, coeffs: torch.Tensor) -> torch.Tensor:
        if not self.discretize_sparse_coeffs:
            return coeffs
        bins = self.coeff_quantizer.encode(coeffs)
        return self.coeff_quantizer.decode(bins)

    def forward(self, x: torch.Tensor):
        z = self.pre(self.encoder(x))
        z_q, loss, support, coeffs = self.bottleneck(z)
        coeffs = self._discretize_coeffs(coeffs)
        if self.discretize_sparse_coeffs:
            with torch.no_grad():
                z_q_disc = self.bottleneck.decode_sparse_codes(support, coeffs)
            # Keep STE gradients from z_q while using discretized values forward.
            z_q = z_q + (z_q_disc - z_q).detach()
        recon = torch.tanh(self.decoder(self.post(z_q)))
        return recon, loss, support, coeffs

    @torch.no_grad()
    def encode(self, x: torch.Tensor):
        """Returns (support [B,H,W,K], coeffs [B,H,W,K])."""
        z = self.pre(self.encoder(x))
        _, _, support, coeffs = self.bottleneck(z)
        coeffs = self._discretize_coeffs(coeffs)
        return support, coeffs

    @torch.no_grad()
    def decode_from_codes(
        self, support: torch.Tensor, coeffs: torch.Tensor,
    ) -> torch.Tensor:
        z_q = self.bottleneck.decode_sparse_codes(support, coeffs)
        return torch.tanh(self.decoder(self.post(z_q)))


# ---------------------------------------------------------------------------
# Coefficient quantizer
# ---------------------------------------------------------------------------

class CoefficientQuantizer:
    """Sparse coefficient quantizer with optional mu-law companding."""

    def __init__(
        self, n_bins: int = 1024, max_val: float = 24.0, mu: float = 255.0,
    ):
        self.n_bins = max(int(n_bins), 2)
        self.max_val = max(float(max_val), 1e-6)
        self.mu = max(float(mu), 0.0)
        self.use_mu_law = self.mu > 0.0
        self._log_mu1 = math.log1p(self.mu) if self.use_mu_law else None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = x.float().clamp(-self.max_val, self.max_val) / self.max_val
        if self.use_mu_law:
            x_norm = (
                torch.sign(x_norm)
                * torch.log1p(self.mu * x_norm.abs())
                / self._log_mu1
            )
        return (
            ((x_norm + 1.0) / 2.0 * (self.n_bins - 1))
            .round()
            .long()
            .clamp(0, self.n_bins - 1)
        )

    def decode(self, bins: torch.Tensor) -> torch.Tensor:
        x_norm = (
            bins.float().clamp(0, self.n_bins - 1) / (self.n_bins - 1) * 2.0 - 1.0
        )
        if self.use_mu_law:
            x_norm = (
                torch.sign(x_norm)
                * (torch.pow(1.0 + self.mu, x_norm.abs()) - 1.0)
                / self.mu
            )
        return x_norm * self.max_val


# ---------------------------------------------------------------------------
# RQ Transformer prior
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=(self.dropout if self.training else 0.0),
            is_causal=(kv_cache is None and T > 1),
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out)), (k, v)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attn_out, new_cache = self.attn(self.ln1(x), kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_cache


@dataclass
class RQTransformerConfig:
    vocab_size: int
    H: int
    W: int
    D: int
    num_classes: int = 0
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    n_depth_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    use_pred_coeff_feedback: bool = True
    use_pred_atom_feedback: bool = True
    stochastic_feedback_sampling: bool = False
    feedback_temperature: float = 1.0
    atom_label_smoothing: float = 0.0
    coeff_adjacent_soft_target: float = 0.0
    n_coeff_bins: int = 1024
    coeff_mu: float = 255.0
    coeff_max_val: float = 24.0


class RQTransformerPrior(nn.Module):
    """Two-stage prior with atom logits + coefficient-bin logits."""

    def __init__(self, cfg: RQTransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.positions_per_image = cfg.H * cfg.W

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.row_emb = nn.Embedding(cfg.H, cfg.d_model)
        self.col_emb = nn.Embedding(cfg.W, cfg.d_model)
        self.depth_emb = nn.Embedding(cfg.D, cfg.d_model)
        self.depth_type_emb = nn.Embedding(2, cfg.d_model)
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.row_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.col_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.depth_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.depth_type_emb.weight, mean=0.0, std=0.02)
        self.start_emb = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        spatial_pos = torch.arange(self.positions_per_image, dtype=torch.long)
        self.register_buffer(
            "_spatial_rows",
            torch.div(spatial_pos, cfg.W, rounding_mode="floor"),
            persistent=False,
        )
        self.register_buffer(
            "_spatial_cols",
            torch.remainder(spatial_pos, cfg.W),
            persistent=False,
        )
        self.class_emb = (
            nn.Embedding(cfg.num_classes, cfg.d_model)
            if cfg.num_classes > 0 else None
        )

        self.drop = nn.Dropout(cfg.dropout)
        self.spatial_blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.depth_blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_depth_layers)
        ])
        self.spatial_ln = nn.LayerNorm(cfg.d_model)
        self.depth_ln = nn.LayerNorm(cfg.d_model)
        self.atom_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.spatial_depth_fuse = nn.Sequential(
            nn.LayerNorm(cfg.D * cfg.d_model),
            nn.Linear(cfg.D * cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.num_atoms = cfg.vocab_size
        self.quantizer = CoefficientQuantizer(
            cfg.n_coeff_bins, cfg.coeff_max_val, cfg.coeff_mu,
        )
        self.coeff_quantizer = self.quantizer
        coeff_in_dim = (2 + 2 * cfg.D) * cfg.d_model
        self.coeff_head = nn.Sequential(
            nn.Linear(coeff_in_dim, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.n_coeff_bins),
        )
        nn.init.zeros_(self.coeff_head[-1].weight)
        nn.init.zeros_(self.coeff_head[-1].bias)

    def _class_bias(
        self, class_ids: Optional[torch.Tensor], batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if self.class_emb is None:
            return None
        if class_ids is None:
            raise ValueError("class_ids required when num_classes > 0")
        if class_ids.numel() != batch_size:
            raise ValueError(
                f"class_ids shape mismatch: expected {batch_size} elements, "
                f"got {tuple(class_ids.shape)}"
            )
        return self.class_emb(class_ids.long().to(device)).unsqueeze(1)

    def _spatial_pos(self, length: int, device: torch.device) -> torch.Tensor:
        """Return learned 2D positional embeddings for flattened raster order."""
        if length > self.positions_per_image:
            raise ValueError(
                f"Requested {length} positions, but max is {self.positions_per_image}."
            )
        rows = self._spatial_rows[:length].to(device)
        cols = self._spatial_cols[:length].to(device)
        return self.row_emb(rows) + self.col_emb(cols)

    def _flat_to_btd_depth_major(
        self, x_flat: torch.Tensor, name: str,
    ) -> torch.Tensor:
        """[B, H*W*D] -> [B, H*W, D], depth-major inside each location."""
        expected = self.positions_per_image * self.cfg.D
        if x_flat.dim() != 2 or x_flat.size(1) != expected:
            raise ValueError(
                f"{name} must have shape [B, {expected}], got {tuple(x_flat.shape)}"
            )
        return x_flat.reshape(-1, self.positions_per_image, self.cfg.D)

    def _btd_to_flat_depth_major(self, x_btd: torch.Tensor) -> torch.Tensor:
        """[B, H*W, D] -> [B, H*W*D], inverse of _flat_to_btd_depth_major."""
        return x_btd.reshape(x_btd.size(0), self.positions_per_image * self.cfg.D)

    def _coeff_bins_and_values(
        self, coeffs_flat: torch.Tensor, name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.is_floating_point(coeffs_flat):
            coeff_vals = self._flat_to_btd_depth_major(coeffs_flat.float(), name)
            coeff_bins = self.quantizer.encode(coeff_vals)
        else:
            coeff_bins = self._flat_to_btd_depth_major(coeffs_flat.long(), name)
            coeff_bins = coeff_bins.clamp(0, self.cfg.n_coeff_bins - 1)
        coeff_vals = self.quantizer.decode(coeff_bins)
        return coeff_bins, coeff_vals

    def _coeff_pair_to_embed(
        self, atom_emb: torch.Tensor, coeff_val: torch.Tensor,
    ) -> torch.Tensor:
        """Encode (atom id embedding, scalar coeff) as coeff feedback feature."""
        return torch.tanh(atom_emb * coeff_val.unsqueeze(-1))

    def _sample_or_argmax(
        self, logits: torch.Tensor, temperature: float = 1.0,
    ) -> torch.Tensor:
        """Sample token ids (or argmax) for stochastic teacher roll-in."""
        if not self.cfg.stochastic_feedback_sampling:
            return logits.argmax(dim=-1)
        orig_shape = logits.shape[:-1]
        flat = logits.reshape(-1, logits.size(-1)) / max(float(temperature), 1e-8)
        ids = torch.multinomial(F.softmax(flat, dim=-1), 1).squeeze(-1)
        return ids.reshape(orig_shape)

    def _fuse_spatial_context(
        self, tok_emb: torch.Tensor, coeff_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse [B, T, D, C] token/coeff embeddings to [B, T, C]."""
        fused = (tok_emb + coeff_emb).reshape(tok_emb.size(0), tok_emb.size(1), -1)
        return self.spatial_depth_fuse(fused)

    def _run_no_cache_blocks(
        self, x: torch.Tensor, blocks: nn.ModuleList,
    ) -> torch.Tensor:
        """Run transformer blocks without KV cache, with checkpointing in train."""
        for blk in blocks:
            if self.training:
                x = checkpoint(
                    lambda inp, m=blk: m(inp)[0],
                    x,
                    use_reentrant=False,
                )
            else:
                x, _ = blk(x)
        return x

    def forward_tokens(
        self,
        tokens_flat: torch.Tensor,
        coeffs_flat: torch.Tensor,
        class_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced logits over flattened [H*W*D] tokens.

        Returns:
            atom_logits:       [B, H*W, D, vocab_size]
            coeff_logits:      [B, H*W, D, n_coeff_bins]
        """
        device = tokens_flat.device
        D = self.cfg.D
        tok = self._flat_to_btd_depth_major(tokens_flat.long(), "tokens_flat")
        B, T, D = tok.shape
        d_model = self.cfg.d_model
        _, coeffs = self._coeff_bins_and_values(coeffs_flat, "coeffs_flat")

        tok_emb_spatial = self.token_emb(tok)
        coeff_teacher_spatial = torch.tanh(tok_emb_spatial * coeffs.unsqueeze(-1))
        spatial_in = torch.zeros(B, T, d_model, device=device)
        if T > 1:
            prev_tok_emb = tok_emb_spatial[:, :-1, :]
            prev_coeff_emb = coeff_teacher_spatial[:, :-1, :]
            spatial_in[:, 1:, :] = self._fuse_spatial_context(
                prev_tok_emb,
                prev_coeff_emb,
            )
        spatial_in = spatial_in + self._spatial_pos(T, device).unsqueeze(0)
        spatial_in[:, :1, :] = spatial_in[:, :1, :] + self.start_emb

        class_bias = self._class_bias(class_ids, B, device)
        if class_bias is not None:
            spatial_in = spatial_in + class_bias

        spatial_h = self.drop(spatial_in)
        spatial_h = self._run_no_cache_blocks(spatial_h, self.spatial_blocks)
        spatial_h = self.spatial_ln(spatial_h)

        bt = B * T
        h_t = spatial_h.reshape(bt, d_model)
        tok_bt = tok.view(bt, D)
        tok_emb = self.token_emb(tok_bt)
        coeff_bt = coeffs.reshape(bt, D)
        depth_pos = self.depth_emb(
            torch.arange(D, device=device, dtype=torch.long),
        )
        depth_pos_steps = (
            depth_pos.unsqueeze(1).expand(D, 2, d_model).reshape(2 * D, d_model)
        )
        depth_type = self.depth_type_emb(
            torch.tensor([0, 1], dtype=torch.long, device=device),
        )

        token_slots = torch.zeros(bt, D, d_model, device=device)
        coeff_slots = torch.zeros(bt, D, d_model, device=device)
        atom_logits_list: list[torch.Tensor] = []
        coeff_logits_list: list[torch.Tensor] = []
        depth_kv: list[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [
            None
        ] * len(self.depth_blocks)
        fb_temp = max(float(self.cfg.feedback_temperature), 1e-8)

        # Roll through depth steps autoregressively, mirroring generate().
        for step in range(2 * D):
            d = step // 2
            is_atom_step = (step % 2 == 0)
            if is_atom_step:
                depth_x = h_t if d == 0 else coeff_slots[:, d - 1, :]
            else:
                tok_teacher = tok_emb[:, d, :]
                if self.cfg.use_pred_atom_feedback:
                    tok_pred_ids = self._sample_or_argmax(
                        atom_logits_list[d],
                        temperature=fb_temp,
                    )
                    tok_cur = self.token_emb(tok_pred_ids)
                else:
                    tok_cur = tok_teacher
                token_slots[:, d, :] = tok_cur
                depth_x = tok_cur

            depth_h = (
                depth_x
                + depth_pos_steps[step]
                + depth_type[0 if is_atom_step else 1]
            ).unsqueeze(1)
            depth_h = self.drop(depth_h)
            for i, blk in enumerate(self.depth_blocks):
                depth_h, depth_kv[i] = blk(depth_h, kv_cache=depth_kv[i])
            z_last = self.depth_ln(depth_h)[:, 0, :]

            if is_atom_step:
                atom_logits_list.append(self.atom_head(z_last))
                continue

            combined_h = torch.cat([
                h_t,
                z_last,
                token_slots.reshape(bt, D * d_model),
                coeff_slots.reshape(bt, D * d_model),
            ], dim=-1)
            coeff_logits_d = self.coeff_head(combined_h)
            coeff_logits_list.append(coeff_logits_d)
            coeff_feedback = (
                self.quantizer.decode(
                    self._sample_or_argmax(coeff_logits_d, temperature=fb_temp),
                )
                if self.cfg.use_pred_coeff_feedback
                else coeff_bt[:, d]
            )
            coeff_slots[:, d, :] = self._coeff_pair_to_embed(
                token_slots[:, d, :], coeff_feedback,
            )

        atom_logits_bt = torch.stack(atom_logits_list, dim=1)
        atom_logits = atom_logits_bt.view(B, T, D, self.cfg.vocab_size)
        coeff_logits = torch.stack(coeff_logits_list, dim=1).view(
            B, T, D, self.cfg.n_coeff_bins,
        )
        return atom_logits, coeff_logits

    def forward_loss(
        self,
        tokens_flat: torch.Tensor,
        coeff_bins_flat: torch.Tensor,
        class_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single CE objective over atom and coefficient-bin predictions."""
        atom_logits, coeff_logits = self.forward_tokens(
            tokens_flat, coeff_bins_flat, class_ids=class_ids,
        )
        atom_target = self._flat_to_btd_depth_major(tokens_flat.long(), "tokens_flat")
        coeff_target, _ = self._coeff_bins_and_values(
            coeff_bins_flat, "coeffs_flat",
        )
        atom_smooth = float(min(max(self.cfg.atom_label_smoothing, 0.0), 0.999))
        atom_loss = F.cross_entropy(
            atom_logits.reshape(-1, self.cfg.vocab_size),
            atom_target.reshape(-1),
            label_smoothing=atom_smooth,
        )
        coeff_logits_flat = coeff_logits.reshape(-1, self.cfg.n_coeff_bins)
        coeff_target_flat = coeff_target.reshape(-1)
        coeff_alpha = float(
            min(max(self.cfg.coeff_adjacent_soft_target, 0.0), 0.999),
        )
        if coeff_alpha <= 0.0:
            coeff_loss = F.cross_entropy(coeff_logits_flat, coeff_target_flat)
        else:
            # Soft bins: center gets most mass, immediate neighbors share coeff_alpha.
            log_probs = F.log_softmax(coeff_logits_flat, dim=-1)
            center_idx = coeff_target_flat
            left_idx = (center_idx - 1).clamp_min(0)
            right_idx = (center_idx + 1).clamp_max(self.cfg.n_coeff_bins - 1)
            has_left = (center_idx > 0).to(log_probs.dtype)
            has_right = (center_idx < (self.cfg.n_coeff_bins - 1)).to(log_probs.dtype)
            left_w = 0.5 * coeff_alpha * has_left
            right_w = 0.5 * coeff_alpha * has_right
            center_w = 1.0 - left_w - right_w
            center_lp = log_probs.gather(1, center_idx.unsqueeze(1)).squeeze(1)
            left_lp = log_probs.gather(1, left_idx.unsqueeze(1)).squeeze(1)
            right_lp = log_probs.gather(1, right_idx.unsqueeze(1)).squeeze(1)
            coeff_loss = -(
                center_w * center_lp + left_w * left_lp + right_w * right_lp
            ).mean()
        return 0.5 * (atom_loss + coeff_loss)

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        temperature: float = 1.0,
        class_ids: Optional[torch.Tensor] = None,
        show_progress: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        T = self.positions_per_image
        D = self.cfg.D
        d_model = self.cfg.d_model

        class_bias = self._class_bias(class_ids, batch_size, device)
        spatial_pos = self._spatial_pos(T, device)
        depth_pos = self.depth_emb(
            torch.arange(D, device=device, dtype=torch.long),
        )
        depth_pos_steps = (
            depth_pos.unsqueeze(1).expand(D, 2, d_model).reshape(2 * D, d_model)
        )
        depth_type = self.depth_type_emb(
            torch.tensor([0, 1], device=device, dtype=torch.long),
        )

        tokens = torch.zeros(batch_size, T, D, dtype=torch.long, device=device)
        coeffs = torch.zeros(batch_size, T, D, dtype=torch.float32, device=device)
        prev_coeff_slots = torch.zeros(batch_size, D, d_model, device=device)
        spatial_kv: list[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [
            None
        ] * len(self.spatial_blocks)
        steps = tqdm(
            range(T), desc="sampling", leave=False, disable=(not show_progress),
        )

        for t in steps:
            if t == 0:
                x_new = self.start_emb.expand(batch_size, -1, -1)
            else:
                prev_tok = self.token_emb(tokens[:, t - 1 : t, :])
                x_new = self._fuse_spatial_context(
                    prev_tok, prev_coeff_slots.unsqueeze(1),
                )
            x_new = x_new + spatial_pos[t : t + 1].unsqueeze(0)
            if class_bias is not None:
                x_new = x_new + class_bias

            spatial_h = x_new
            for i, blk in enumerate(self.spatial_blocks):
                spatial_h, spatial_kv[i] = blk(
                    spatial_h, kv_cache=spatial_kv[i],
                )
            h_t = self.spatial_ln(spatial_h).squeeze(1)

            token_slots = torch.zeros(batch_size, D, d_model, device=device)
            coeff_slots = torch.zeros(batch_size, D, d_model, device=device)
            depth_kv: list[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [
                None
            ] * len(self.depth_blocks)
            for step in range(2 * D):
                d = step // 2
                is_atom_step = (step % 2 == 0)
                if is_atom_step:
                    depth_x = h_t if d == 0 else coeff_slots[:, d - 1, :]
                else:
                    depth_x = token_slots[:, d, :]
                depth_h = (
                    depth_x
                    + depth_pos_steps[step]
                    + depth_type[0 if is_atom_step else 1]
                ).unsqueeze(1)
                for i, blk in enumerate(self.depth_blocks):
                    depth_h, depth_kv[i] = blk(depth_h, kv_cache=depth_kv[i])
                z_last = self.depth_ln(depth_h)[:, 0, :]

                if is_atom_step:
                    logits = self.atom_head(z_last) / max(temperature, 1e-8)
                    tok_ids = torch.multinomial(
                        F.softmax(logits, dim=-1), 1,
                    ).squeeze(-1)
                    tokens[:, t, d] = tok_ids
                    token_slots[:, d, :] = self.token_emb(tok_ids)
                    continue

                tok_cur = token_slots[:, d, :]
                combined_h = torch.cat([
                    h_t,
                    z_last,
                    token_slots.reshape(batch_size, D * d_model),
                    coeff_slots.reshape(batch_size, D * d_model),
                ], dim=-1)
                coeff_logits = self.coeff_head(combined_h) / max(temperature, 1e-8)
                coeff_bins = torch.multinomial(
                    F.softmax(coeff_logits, dim=-1), 1,
                ).squeeze(-1)
                pred_coeff = self.quantizer.decode(coeff_bins)
                coeffs[:, t, d] = pred_coeff
                coeff_slots[:, d, :] = self._coeff_pair_to_embed(
                    tok_cur, pred_coeff,
                )
            prev_coeff_slots = coeff_slots

        return (
            self._btd_to_flat_depth_major(tokens),
            self._btd_to_flat_depth_major(coeffs),
        )


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _make_grid(x: torch.Tensor, nrow: int = 8) -> torch.Tensor:
    x = x.detach().cpu().clamp(-1, 1)
    return utils.make_grid((x + 1.0) / 2.0, nrow=nrow)


def save_grid(x: torch.Tensor, path: str, nrow: int = 8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    utils.save_image(_make_grid(x, nrow), path)


def _get_wandb_logger(logger_obj):
    if isinstance(logger_obj, WandbLogger):
        return logger_obj
    for lg in getattr(logger_obj, "loggers", []):
        if isinstance(lg, WandbLogger):
            return lg
    return None


def log_images_wandb(logger_obj, key, x, step, caption=None) -> bool:
    wb = _get_wandb_logger(logger_obj)
    if wb is None or wandb is None:
        return False
    grid = _make_grid(x).permute(1, 2, 0).mul(255).clamp(0, 255).byte().numpy()
    img = wandb.Image(grid, caption=caption) if caption else wandb.Image(grid)
    wb.experiment.log({key: img}, step=int(step))
    return True


def log_scalar_wandb(logger_obj, key: str, value: float, step: int) -> bool:
    wb = _get_wandb_logger(logger_obj)
    if wb is None or wandb is None:
        return False
    wb.experiment.log({key: float(value)}, step=int(step))
    return True


# ---------------------------------------------------------------------------
# Stage-1 Lightning module
# ---------------------------------------------------------------------------

class Stage1Module(pl.LightningModule):
    def __init__(
        self,
        ae: SparseDictAE,
        lr: float,
        bottleneck_weight: float,
        out_dir: str,
        val_vis_images: Optional[torch.Tensor] = None,
        lr_schedule: str = "cosine",
        warmup_epochs: int = 1,
        min_lr_ratio: float = 0.1,
    ):
        super().__init__()
        self.ae = ae
        self.lr = lr
        self.bottleneck_weight = bottleneck_weight
        self.out_dir = out_dir
        self.best_val = float("inf")
        self.val_vis_images = val_vis_images
        self.lr_schedule = lr_schedule
        self.warmup_epochs = warmup_epochs
        self.min_lr_ratio = min_lr_ratio

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.ae.parameters(), lr=self.lr)
        if self.lr_schedule != "cosine":
            return opt
        max_ep = max(1, int(getattr(self.trainer, "max_epochs", 1) or 1))
        warmup = min(self.warmup_epochs, max_ep - 1)
        min_r = self.min_lr_ratio

        def lr_lambda(epoch: int) -> float:
            s = epoch + 1
            if warmup > 0 and s <= warmup:
                return 0.1 + 0.9 * (s / max(1, warmup))
            decay = max(1, max_ep - warmup)
            t = min(max(s - warmup, 0), decay) / decay
            return min_r + (1 - min_r) * 0.5 * (1 + math.cos(math.pi * t))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
        }

    def training_step(self, batch, batch_idx):
        x, _ = batch
        recon, b_loss, _, _ = self.ae(x)
        recon_loss = F.mse_loss(recon, x)
        loss = recon_loss + self.bottleneck_weight * b_loss
        bs = x.size(0)

        self.log("stage1/train_loss", loss,
                 on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)
        self.log("stage1/recon_loss", recon_loss,
                 on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)
        self.log("stage1/b_loss", b_loss,
                 on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        bs = batch[0].size(0) if isinstance(batch, (tuple, list)) else 1
        revived = 0
        with torch.no_grad():
            bn = self.ae.bottleneck
            bn.dictionary.copy_(
                F.normalize(bn.dictionary, p=2, dim=0, eps=bn.epsilon)
            )
            if (
                self.trainer.is_global_zero
                and bn.dead_atom_threshold > 0
                and bn.dead_atom_max_reinit > 0
                and int(bn._steps_since_reinit.item()) >= bn.dead_atom_interval
            ):
                revived = bn.revive_dead_atoms()

            revived_value = torch.tensor(
                float(revived),
                device=bn.dictionary.device,
                dtype=bn.dictionary.dtype,
            )
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.broadcast(bn.dictionary.data, src=0)
                torch.distributed.broadcast(bn.usage_ema, src=0)
                torch.distributed.broadcast(revived_value, src=0)

            usage_top1, usage_perplexity, usage_active = bn.usage_stats()

        self.log(
            "stage1/usage_top1", usage_top1,
            on_step=True, on_epoch=True, sync_dist=True, batch_size=bs,
        )
        self.log(
            "stage1/usage_perplexity", usage_perplexity,
            on_step=True, on_epoch=True, sync_dist=True, batch_size=bs,
        )
        self.log(
            "stage1/usage_active_frac", usage_active,
            on_step=True, on_epoch=True, sync_dist=True, batch_size=bs,
        )
        self.log(
            "stage1/revived_atoms", revived_value,
            on_step=True, on_epoch=True, sync_dist=True, batch_size=bs,
        )

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        recon, b_loss, _, _ = self.ae(x)
        recon_loss = F.mse_loss(recon, x)
        loss = recon_loss + self.bottleneck_weight * b_loss
        psnr = 10.0 * torch.log10(4.0 / recon_loss.detach().clamp(min=1e-8))
        bs = x.size(0)

        self.log("stage1/val_loss", loss,
                 on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)
        self.log("stage1/val_psnr", psnr,
                 on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)
        return loss

    def on_validation_epoch_end(self):
        if not self.trainer.is_global_zero or self.trainer.sanity_checking:
            return
        os.makedirs(self.out_dir, exist_ok=True)
        cur = self.trainer.callback_metrics.get("stage1/val_loss")
        if cur is not None:
            torch.save(self.ae.state_dict(),
                       os.path.join(self.out_dir, "ae_last.pt"))
            if float(cur) < self.best_val:
                self.best_val = float(cur)
                torch.save(self.ae.state_dict(),
                           os.path.join(self.out_dir, "ae_best.pt"))

        if self.val_vis_images is not None:
            x_vis = self.val_vis_images.to(self.device)
            with torch.no_grad():
                recon_vis, _, _, _ = self.ae(x_vis)
            epoch = self.current_epoch + 1
            step = self.global_step
            r = log_images_wandb(self.logger, "stage1/real", x_vis, step,
                                 f"epoch={epoch}")
            f = log_images_wandb(self.logger, "stage1/recon", recon_vis, step,
                                 f"epoch={epoch}")
            if not (r and f):
                save_grid(x_vis,
                          os.path.join(self.out_dir, f"epoch{epoch:03d}_real.png"))
                save_grid(recon_vis,
                          os.path.join(self.out_dir, f"epoch{epoch:03d}_recon.png"))


# ---------------------------------------------------------------------------
# Sparse-code ordering helpers
# ---------------------------------------------------------------------------

def _flatten_depth_major(x_bhwd: torch.Tensor) -> torch.Tensor:
    """[B,H,W,D] -> [B,H*W*D], depth-major inside each spatial location."""
    bsz, h, w, d = x_bhwd.shape
    return x_bhwd.reshape(bsz, h * w, d).reshape(bsz, h * w * d)


def _flat_to_btd_depth_major(
    x_flat: torch.Tensor, H: int, W: int, D: int,
) -> torch.Tensor:
    """[B,H*W*D] -> [B,H*W,D] inverse of _flatten_depth_major."""
    return x_flat.reshape(-1, H * W, D)


def _flat_to_bhwd_depth_major(
    x_flat: torch.Tensor, H: int, W: int, D: int,
) -> torch.Tensor:
    """[B,H*W*D] -> [B,H,W,D] inverse of _flatten_depth_major."""
    return _flat_to_btd_depth_major(x_flat, H, W, D).reshape(-1, H, W, D)


# ---------------------------------------------------------------------------
# Stage-2 Lightning module
# ---------------------------------------------------------------------------

class Stage2Module(pl.LightningModule):
    def __init__(
        self,
        transformer: nn.Module,
        ae: SparseDictAE,
        lr: float,
        out_dir: str,
        H: int,
        W: int,
        D: int,
        sample_every_steps: int = 200,
        sample_batch_size: int = 8,
        sample_temperature: float = 0.8,
        direct_coeff_loss_weight: float = 0.25,
        lr_schedule: str = "cosine",
        warmup_epochs: int = 1,
        min_lr_ratio: float = 0.1,
        sample_class_id: int = -1,
        sample_image_size: Optional[int] = None,
    ):
        super().__init__()
        self.transformer = transformer
        self.ae = ae
        self.lr = lr
        self.out_dir = out_dir
        self.H, self.W, self.D = H, W, D
        self.sample_every_steps = sample_every_steps
        self.sample_batch_size = sample_batch_size
        self.sample_temperature = max(sample_temperature, 1e-8)
        self.direct_coeff_loss_weight = direct_coeff_loss_weight
        self.lr_schedule = lr_schedule
        self.warmup_epochs = warmup_epochs
        self.min_lr_ratio = min_lr_ratio
        self.num_classes = int(getattr(transformer.cfg, "num_classes", 0))
        self.sample_class_id = None if sample_class_id < 0 else sample_class_id
        self.sample_image_size = sample_image_size
        self._last_sample_step = -1

        self.ae.eval()
        for p in self.ae.parameters():
            p.requires_grad_(False)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.transformer.parameters(),
            lr=self.lr,
            betas=(0.9, 0.95),
            weight_decay=1e-4,
        )
        if self.lr_schedule != "cosine":
            return opt
        max_ep = max(1, int(getattr(self.trainer, "max_epochs", 1) or 1))
        warmup = min(self.warmup_epochs, max_ep - 1)
        min_r = self.min_lr_ratio

        def lr_lambda(epoch: int) -> float:
            s = epoch + 1
            if warmup > 0 and s <= warmup:
                return 0.1 + 0.9 * (s / max(1, warmup))
            decay = max(1, max_ep - warmup)
            t = min(max(s - warmup, 0), decay) / decay
            return min_r + (1 - min_r) * 0.5 * (1 + math.cos(math.pi * t))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
        }

    def _unpack_stage2_batch(
        self,
        batch,
        limit: Optional[int] = None,
        move_to_device: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if not isinstance(batch, (tuple, list)):
            batch = (batch,)
        items = list(batch)

        class_ids = None
        if self.num_classes > 0:
            class_ids = items.pop(-1).long()

        tok_flat, coeff_flat = items
        tok_flat = tok_flat.long()
        coeff_flat = coeff_flat.float()

        if limit is not None and limit > 0 and tok_flat.size(0) > limit:
            tok_flat = tok_flat[:limit]
            coeff_flat = coeff_flat[:limit]
            if class_ids is not None:
                class_ids = class_ids[:limit]

        if move_to_device:
            tok_flat = tok_flat.to(self.device, non_blocking=True)
            coeff_flat = coeff_flat.to(self.device, non_blocking=True)
            if class_ids is not None:
                class_ids = class_ids.to(self.device, non_blocking=True)

        return tok_flat, coeff_flat, class_ids

    def _resize_for_logging(self, x: torch.Tensor) -> torch.Tensor:
        if self.sample_image_size and self.sample_image_size > 0:
            return F.interpolate(
                x,
                size=(self.sample_image_size, self.sample_image_size),
                mode="bilinear",
                align_corners=False,
            )
        return x

    _COEFF_CLAMP = 24.0

    @property
    def _uses_model_loss(self) -> bool:
        return hasattr(self.transformer, "forward_loss")

    @property
    def _uses_unified_loss(self) -> bool:
        return self._uses_model_loss and self._uses_coeff_bins

    @property
    def _uses_coeff_bins(self) -> bool:
        return hasattr(self.transformer, "quantizer")

    def training_step(self, batch, batch_idx):
        tok_flat, coeff_flat, class_ids = self._unpack_stage2_batch(batch)
        B = tok_flat.size(0)

        if self._uses_model_loss:
            coeff_input = coeff_flat
            kwargs = {}
            if self._uses_coeff_bins:
                coeff_input = self.transformer.quantizer.encode(coeff_flat)
            else:
                kwargs["coeff_loss_weight"] = self.direct_coeff_loss_weight
            loss = self.transformer.forward_loss(
                tok_flat, coeff_input, class_ids=class_ids, **kwargs,
            )
            self.log("train/loss", loss,
                     prog_bar=True, on_step=True, on_epoch=True,
                     sync_dist=True, batch_size=B)
            return loss

        y = _flat_to_btd_depth_major(tok_flat, self.H, self.W, self.D)
        if self._uses_coeff_bins:
            coeff_bins_flat = self.transformer.quantizer.encode(coeff_flat)
            coeff_bins_target = _flat_to_btd_depth_major(
                coeff_bins_flat, self.H, self.W, self.D,
            )
            atom_logits, coeff_logits = self.transformer.forward_tokens(
                tok_flat, coeff_bins_flat, class_ids=class_ids,
            )
        else:
            coeff_target = _flat_to_btd_depth_major(
                coeff_flat, self.H, self.W, self.D,
            )
            coeff_target = coeff_target.clamp(-self._COEFF_CLAMP, self._COEFF_CLAMP)
            atom_logits, direct_coeff_pred = self.transformer.forward_tokens(
                tok_flat, coeff_flat, class_ids=class_ids,
            )
        atom_loss = F.cross_entropy(
            atom_logits.reshape(-1, self.transformer.cfg.vocab_size),
            y.reshape(-1),
        )
        if self._uses_coeff_bins:
            direct_coeff_loss = F.cross_entropy(
                coeff_logits.reshape(-1, self.transformer.cfg.n_coeff_bins),
                coeff_bins_target.reshape(-1),
            )
        else:
            direct_coeff_loss = F.smooth_l1_loss(
                direct_coeff_pred, coeff_target, beta=4.0,
            )
        loss = atom_loss + self.direct_coeff_loss_weight * direct_coeff_loss

        self.log("train/atom_loss", atom_loss,
                 prog_bar=True, on_step=True, on_epoch=True,
                 sync_dist=True, batch_size=B)
        self.log("train/direct_coeff_loss", direct_coeff_loss,
                 prog_bar=True, on_step=True, on_epoch=True,
                 sync_dist=True, batch_size=B)
        self.log("train/loss", loss,
                 prog_bar=True, on_step=True, on_epoch=True,
                 sync_dist=True, batch_size=B)
        return loss

    def on_fit_start(self):
        self.ae.to(self.device)
        self.ae.eval()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.sample_every_steps <= 0:
            return
        if self.global_step <= 0:
            return
        if (self.global_step % self.sample_every_steps) != 0:
            return
        if self.global_step == self._last_sample_step:
            return
        self._last_sample_step = self.global_step
        if self.trainer.is_global_zero:
            self._sample_and_save(batch=batch)

    def on_train_epoch_end(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        if self.trainer.is_global_zero:
            os.makedirs(self.out_dir, exist_ok=True)
            torch.save(
                self.transformer.state_dict(),
                os.path.join(self.out_dir, "transformer_last.pt"),
            )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _build_sample_class_ids(self, n: int) -> Optional[torch.Tensor]:
        if self.num_classes <= 0:
            return None
        if self.sample_class_id is not None:
            return torch.full(
                (n,), self.sample_class_id,
                dtype=torch.long, device=self.device,
            )
        return torch.arange(n, dtype=torch.long, device=self.device) % self.num_classes

    @torch.no_grad()
    def _sample_and_save(self, batch=None):
        self.transformer.eval()
        self.ae.eval()
        step = self.global_step
        class_ids = self._build_sample_class_ids(self.sample_batch_size)

        tokens, coeffs = self.transformer.generate(
            self.sample_batch_size,
            temperature=self.sample_temperature,
            class_ids=class_ids,
            show_progress=True,
        )
        support = _flat_to_bhwd_depth_major(tokens, self.H, self.W, self.D)
        coeffs = _flat_to_bhwd_depth_major(coeffs, self.H, self.W, self.D)
        raw_imgs = self.ae.decode_from_codes(
            support.to(self.device), coeffs.to(self.device),
        )
        imgs = self._resize_for_logging(raw_imgs)

        if not log_images_wandb(
            self.logger, "stage2/samples", imgs, step, f"step={step}",
        ):
            save_grid(
                imgs, os.path.join(self.out_dir, f"step{step:06d}_samples.png"),
            )

        if batch is not None and not self._uses_unified_loss:
            try:
                tf_tok, tf_coeff, tf_class_ids = self._unpack_stage2_batch(
                    batch,
                    limit=self.sample_batch_size,
                    move_to_device=True,
                )
                if tf_tok.numel() > 0:
                    bsz = tf_tok.size(0)
                    tf_support = _flat_to_bhwd_depth_major(
                        tf_tok, self.H, self.W, self.D,
                    )
                    tf_coeff_target = _flat_to_bhwd_depth_major(
                        tf_coeff, self.H, self.W, self.D,
                    )
                    if self._uses_coeff_bins:
                        tf_coeff_bins = self.transformer.quantizer.encode(tf_coeff)
                        _, tf_coeff_logits = self.transformer.forward_tokens(
                            tf_tok, tf_coeff_bins, class_ids=tf_class_ids,
                        )
                        tf_coeff_pred_btd = self.transformer.quantizer.decode(
                            tf_coeff_logits.argmax(dim=-1),
                        )
                        tf_coeff_pred = _flat_to_bhwd_depth_major(
                            tf_coeff_pred_btd.reshape(bsz, -1),
                            self.H, self.W, self.D,
                        )
                    else:
                        _, tf_direct_coeff = self.transformer.forward_tokens(
                            tf_tok, tf_coeff, class_ids=tf_class_ids,
                        )
                        tf_coeff_pred = tf_direct_coeff.reshape(
                            bsz, self.H, self.W, self.D,
                        )
                    tf_gt = self._resize_for_logging(
                        self.ae.decode_from_codes(tf_support, tf_coeff_target),
                    )
                    tf_pred = self._resize_for_logging(
                        self.ae.decode_from_codes(tf_support, tf_coeff_pred),
                    )
                    gt_logged = log_images_wandb(
                        self.logger,
                        "stage2/teacher_forced_gt",
                        tf_gt, step, f"step={step}",
                    )
                    pred_logged = log_images_wandb(
                        self.logger,
                        "stage2/teacher_forced_pred",
                        tf_pred, step, f"step={step}",
                    )
                    if not (gt_logged and pred_logged):
                        save_grid(
                            tf_gt,
                            os.path.join(
                                self.out_dir,
                                f"step{step:06d}_teacher_forced_gt.png",
                            ),
                        )
                        save_grid(
                            tf_pred,
                            os.path.join(
                                self.out_dir,
                                f"step{step:06d}_teacher_forced_pred.png",
                            ),
                        )
            except Exception as exc:
                print(f"[Stage2] teacher-forced preview failed at step {step}: {exc}")

        self.transformer.train()


# ---------------------------------------------------------------------------
# Token precomputation
# ---------------------------------------------------------------------------

@torch.no_grad()
def precompute_tokens(
    ae: SparseDictAE,
    loader: DataLoader,
    device: torch.device,
    max_items: Optional[int] = None,
    collect_labels: bool = False,
    coeff_quantizer: Optional[CoefficientQuantizer] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], int, int, int]:
    """Encode dataset into atom-id / coefficient sequences.

    Returns:
        tokens_flat:    [N, H*W*K] int32, depth-major per location
        coeffs_flat:    [N, H*W*K] float32, depth-major per location
                        (discretized when coeff_quantizer is provided)
        class_ids_flat: [N] int64  (None when collect_labels=False)
        H, W, K
    """
    ae.eval()
    all_tokens: list[torch.Tensor] = []
    all_coeffs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    seen = 0
    H = W = K = None

    for x, y in tqdm(loader, desc="precompute tokens"):
        x = x.to(device)
        support, coeffs = ae.encode(x)
        if H is None:
            H, W, K = support.shape[1], support.shape[2], support.shape[3]
        all_tokens.append(
            _flatten_depth_major(support).to(torch.int32).cpu()
        )
        coeffs_flat = _flatten_depth_major(coeffs).to(torch.float32)
        if coeff_quantizer is not None:
            coeffs_flat = coeff_quantizer.decode(coeff_quantizer.encode(coeffs_flat))
        all_coeffs.append(coeffs_flat.cpu())
        if collect_labels:
            y_t = torch.as_tensor(y, dtype=torch.long)
            if y_t.dim() == 0:
                y_t = y_t.view(1)
            all_labels.append(y_t[: support.size(0)].cpu())
        seen += support.size(0)
        if max_items is not None and seen >= max_items:
            break

    tokens = torch.cat(all_tokens)
    coeffs = torch.cat(all_coeffs)
    labels = (
        torch.cat(all_labels) if (collect_labels and all_labels) else None
    )
    if max_items is not None:
        tokens = tokens[:max_items]
        coeffs = coeffs[:max_items]
        if labels is not None:
            labels = labels[:max_items]
    return tokens, coeffs, labels, H, W, K


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _add_typed_args(add, specs):
    for flag, arg_type, default in specs:
        add(flag, type=arg_type, default=default)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    add = parser.add_argument
    add(
        "--dataset", type=str, default="cifar10",
        choices=["cifar10", "celeba", "stl10", "imagenette"],
    )
    _add_typed_args(add, (
        ("--data_dir", str, None),
        ("--image_size", int, 32),
        ("--out_dir", str, None),
        ("--seed", int, 0),
    ))


def _add_stage1_args(parser: argparse.ArgumentParser) -> None:
    add = parser.add_argument
    _add_typed_args(add, (
        ("--stage1_epochs", int, 5),
        ("--stage1_lr", float, 2e-4),
        ("--stage1_warmup_epochs", int, 1),
        ("--stage1_min_lr_ratio", float, 0.1),
        ("--bottleneck_weight", float, 1.0),
        ("--batch_size", int, 128),
        ("--grad_clip", float, 1.0),
        ("--devices", int, 2),
        ("--stage1_init_ckpt", str, None),
    ))
    add(
        "--stage1_lr_schedule", type=str, default="cosine",
        choices=["constant", "cosine"],
    )


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    add = parser.add_argument
    _add_typed_args(add, (
        ("--num_hiddens", int, 128),
        ("--ae_num_downsamples", int, 2),
        ("--num_res_layers", int, 2),
        ("--num_res_hiddens", int, 32),
        ("--embedding_dim", int, 64),
        ("--num_atoms", int, 128),
        ("--sparsity_level", int, 3),
        ("--commitment_cost", float, 0.25),
    ))
    add(
        "--usage_ema_decay", type=float, default=0.99,
        help="EMA decay for tracking dictionary atom usage.",
    )
    add(
        "--usage_balance_alpha", type=float, default=0.3,
        help="Inverse-usage reweighting strength for OMP atom selection.",
    )
    add(
        "--dead_atom_threshold", type=float, default=5e-4,
        help="Usage threshold for reviving dead atoms; <=0 disables revival.",
    )
    add(
        "--dead_atom_interval", type=int, default=200,
        help="Train batches between dead-atom revival checks.",
    )
    add(
        "--dead_atom_max_reinit", type=int, default=16,
        help="Maximum number of atoms to reinitialize per revival check.",
    )
    add(
        "--bottleneck_patch_size", type=int, default=1,
        help="Patch size for dictionary learning; >1 groups "
             "PxP spatial patches into single atoms.",
    )
    add(
        "--stage1_discretize_sparse_coeffs",
        action="store_true",
        default=True,
        help="Discretize sparse coefficients in stage-1 forward/encode paths.",
    )
    add(
        "--no_stage1_discretize_sparse_coeffs",
        action="store_false",
        dest="stage1_discretize_sparse_coeffs",
    )


def _add_stage2_args(parser: argparse.ArgumentParser) -> None:
    add = parser.add_argument
    _add_typed_args(add, (
        ("--stage2_epochs", int, 10),
        ("--stage2_lr", float, 3e-4),
        ("--stage2_warmup_epochs", int, 1),
        ("--stage2_min_lr_ratio", float, 0.1),
        ("--stage2_batch_size", int, 16),
        ("--stage2_num_classes", int, None),
        ("--stage2_sample_class_id", int, -1),
        ("--stage2_sample_every_steps", int, 200),
        ("--stage2_sample_batch_size", int, 64),
        ("--stage2_sample_temperature", float, 0.8),
        ("--stage2_sample_image_size", int, 256),
    ))
    add(
        "--stage2_token_cache",
        type=str,
        default="tokens_cache.pt",
        help="Stage-2 token cache filename (inside out_dir/stage2) or absolute path.",
    )
    add(
        "--stage2_lr_schedule", type=str, default="cosine",
        choices=["constant", "cosine"],
    )
    add(
        "--stage2_arch",
        type=str,
        default="rq_hier",
        choices=["rq_hier", "decoder_dual", "mingpt"],
        help=(
            "Stage-2 prior architecture: "
            "'decoder_dual' (simple decoder-only dual-head prior), "
            "'mingpt' (plain GPT over interleaved [atom_id, coeff_bin] tokens), or "
            "'rq_hier' (spatial+depth RQ transformer)."
        ),
    )
    add("--stage2_conditional", action="store_true", default=False)
    add("--stage2_resume_from_last", action="store_true", default=True)
    add(
        "--no_stage2_resume_from_last",
        action="store_false",
        dest="stage2_resume_from_last",
    )
    add(
        "--stage2_direct_coeff_loss_weight",
        type=float,
        default=0.25,
        help="Weight for coefficient loss (CE over bins or smooth-L1 regression).",
    )
    _add_typed_args(add, (
        ("--dual_coeff_pred_atom_mix", float, 0.5),
        ("--dual_coeff_atom_coherence_weight", float, 0.1),
    ))
    add(
        "--rq_use_pred_coeff_feedback",
        action="store_true",
        default=True,
        help="Use predicted coeff feedback (detached) inside rq_hier depth loop.",
    )
    add(
        "--no_rq_use_pred_coeff_feedback",
        action="store_false",
        dest="rq_use_pred_coeff_feedback",
    )
    add(
        "--rq_use_pred_atom_feedback",
        action="store_true",
        default=True,
        help="Condition rq_hier coeff prediction on predicted atom embeddings.",
    )
    add(
        "--no_rq_use_pred_atom_feedback",
        action="store_false",
        dest="rq_use_pred_atom_feedback",
    )
    add(
        "--rq_atom_label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing for rq_hier atom-id CE targets.",
    )
    add(
        "--rq_coeff_adjacent_soft_target",
        type=float,
        default=0.0,
        help=(
            "Soft target mass for immediate neighbor coeff bins in rq_hier "
            "loss (center gets remaining mass)."
        ),
    )
    add(
        "--rq_stochastic_feedback_sampling",
        action="store_true",
        default=False,
        help=(
            "Use multinomial sampling (instead of argmax) for rq_hier "
            "training-time predicted atom/coeff feedback."
        ),
    )
    add(
        "--rq_feedback_temperature",
        type=float,
        default=1.0,
        help="Temperature for rq_hier stochastic feedback sampling.",
    )
    _add_typed_args(add, (
        ("--n_coeff_bins", int, 1024),
        ("--coeff_mu", float, 255.0),
        ("--coeff_max_val", float, 24.0),
        ("--tf_d_model", int, 256),
        ("--tf_heads", int, 4),
        ("--tf_layers", int, 4),
        ("--tf_depth_layers", int, 4),
        ("--tf_ff", int, 1024),
        ("--tf_dropout", float, 0.1),
        ("--token_subset", int, 0),
        ("--wandb_project", str, "laser-scratch"),
        ("--wandb_entity", str, None),
        ("--wandb_name", str, None),
        ("--wandb_group", str, None),
        ("--wandb_dir", str, "./wandb"),
    ))
    add(
        "--wandb_mode", type=str, default="online",
        choices=["online", "offline", "disabled"],
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    _add_common_args(parser)
    _add_stage1_args(parser)
    _add_model_args(parser)
    _add_stage2_args(parser)
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    # ---- defaults ----
    if args.data_dir is None:
        args.data_dir = {
            "celeba": "../../data/celeba",
            "imagenette": "./data/imagenette2-160",
        }.get(args.dataset, "./data")
    if args.out_dir is None:
        args.out_dir = f"./runs/sparse_dict_{args.dataset}_{args.image_size}"

    hint = {"cifar10": 10, "stl10": 10, "imagenette": 10}.get(args.dataset, 0)
    stage2_num_classes = 0
    if args.stage2_conditional:
        stage2_num_classes = args.stage2_num_classes or hint
        if stage2_num_classes <= 1:
            raise ValueError(
                "Conditional mode requires a labelled dataset or "
                "--stage2_num_classes > 1."
            )

    P = args.bottleneck_patch_size
    if P > 1:
        signal_dim = P * P * args.embedding_dim
        min_atoms = 2 * signal_dim
        if args.num_atoms < signal_dim:
            print(
                f"[Overcomplete] atom_dim={signal_dim} "
                f"(patch {P}x{P} x embed {args.embedding_dim}), "
                f"num_atoms={args.num_atoms} < atom_dim; "
                f"increasing to {min_atoms} for 2x overcomplete."
            )
            args.num_atoms = min_atoms
        else:
            ratio = args.num_atoms / signal_dim
            print(
                f"[Dictionary] atom_dim={signal_dim}, "
                f"num_atoms={args.num_atoms} ({ratio:.1f}x overcomplete)"
            )

    pl.seed_everything(args.seed, workers=True)
    torch.backends.cudnn.benchmark = True

    os.makedirs(args.out_dir, exist_ok=True)
    stage1_dir = os.path.join(args.out_dir, "stage1")
    stage2_dir = os.path.join(args.out_dir, "stage2")
    os.makedirs(stage1_dir, exist_ok=True)
    os.makedirs(stage2_dir, exist_ok=True)
    token_cache_path = (
        args.stage2_token_cache
        if os.path.isabs(args.stage2_token_cache)
        else os.path.join(stage2_dir, args.stage2_token_cache)
    )
    os.makedirs(os.path.dirname(token_cache_path), exist_ok=True)

    run_name = args.wandb_name or f"sparse_dict_{args.dataset}_{args.image_size}"
    if "LASER_WANDB_GROUP" in os.environ:
        wandb_group = os.environ["LASER_WANDB_GROUP"]
    else:
        wandb_group = (
            args.wandb_group
            or f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.environ["LASER_WANDB_GROUP"] = wandb_group

    # ---- helpers (closures over args) ----

    def _wandb_logger(tag: str):
        if args.wandb_mode == "disabled":
            return False
        if os.environ.get("LOCAL_RANK") not in (None, "0"):
            return False
        try:
            return WandbLogger(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=f"{run_name}_{tag}",
                group=wandb_group,
                save_dir=args.wandb_dir,
                offline=(args.wandb_mode == "offline"),
                log_model=False,
            )
        except Exception as exc:
            print(f"[WandB] init failed ({exc}); continuing without W&B.")
            return False

    def _build_ae() -> SparseDictAE:
        return SparseDictAE(
            in_channels=3,
            num_hiddens=args.num_hiddens,
            num_downsamples=args.ae_num_downsamples,
            num_residual_layers=args.num_res_layers,
            num_residual_hiddens=args.num_res_hiddens,
            embedding_dim=args.embedding_dim,
            num_embeddings=args.num_atoms,
            sparsity_level=args.sparsity_level,
            commitment_cost=args.commitment_cost,
            usage_ema_decay=args.usage_ema_decay,
            usage_balance_alpha=args.usage_balance_alpha,
            dead_atom_threshold=args.dead_atom_threshold,
            dead_atom_interval=args.dead_atom_interval,
            dead_atom_max_reinit=args.dead_atom_max_reinit,
            bottleneck_patch_size=args.bottleneck_patch_size,
            n_coeff_bins=args.n_coeff_bins,
            coeff_mu=args.coeff_mu,
            coeff_max_val=args.coeff_max_val,
            discretize_sparse_coeffs=args.stage1_discretize_sparse_coeffs,
        )

    def _load_ae(ae: SparseDictAE, path: str, tag: str = "Stage1"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{tag} checkpoint not found at {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        ae.load_state_dict(state)
        print(f"[{tag}] loaded AE from {path}")

    def _build_datasets(
        transform,
    ) -> Tuple[Dataset, Dataset]:
        if args.dataset == "cifar10":
            train_set = datasets.CIFAR10(
                root=args.data_dir, train=True, download=True, transform=transform,
            )
            val_set = datasets.CIFAR10(
                root=args.data_dir, train=False, download=True, transform=transform,
            )
            return train_set, val_set
        if args.dataset == "stl10":
            train_set = datasets.STL10(
                root=args.data_dir, split="train", download=True,
                transform=transform,
            )
            val_set = datasets.STL10(
                root=args.data_dir, split="test", download=True, transform=transform,
            )
            return train_set, val_set
        if args.dataset == "celeba":
            full = FlatImageDataset(root=args.data_dir, transform=transform)
            n_val = max(1, int(0.05 * len(full)))
            idx = torch.randperm(
                len(full), generator=torch.Generator().manual_seed(args.seed),
            )
            train_set = Subset(full, idx[: len(full) - n_val].tolist())
            val_set = Subset(full, idx[len(full) - n_val:].tolist())
            return train_set, val_set
        if args.dataset == "imagenette":
            train_set = datasets.ImageFolder(
                root=os.path.join(args.data_dir, "train"), transform=transform,
            )
            val_set = datasets.ImageFolder(
                root=os.path.join(args.data_dir, "val"), transform=transform,
            )
            return train_set, val_set
        raise ValueError(f"Unknown dataset: {args.dataset}")

    def _build_ddp_strategy() -> object:
        if args.devices <= 1:
            return "auto"
        from lightning.pytorch.strategies import DDPStrategy
        return DDPStrategy(broadcast_buffers=False, find_unused_parameters=False)

    def _load_or_precompute_tokens(
        ae: SparseDictAE,
        train_set: Dataset,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], int, int, int,
    ]:
        coeff_quantizer = CoefficientQuantizer(
            args.n_coeff_bins, args.coeff_max_val, args.coeff_mu,
        )
        use_cache = args.stage1_epochs <= 0 and os.path.exists(token_cache_path)
        if use_cache:
            print(f"Loading cached tokens from {token_cache_path}")
            cache = torch.load(
                token_cache_path, map_location="cpu", weights_only=False,
            )
            tokens_flat = cache["tokens"]
            if tokens_flat.size(0) == 0:
                print("[WARNING] Cached tokens are empty, recomputing...")
                os.remove(token_cache_path)
                use_cache = False
        if use_cache:
            coeffs_flat = cache["coeffs"]
            coeffs_flat = coeff_quantizer.decode(
                coeff_quantizer.encode(coeffs_flat.float()),
            )
            cache_meta = cache.get("coeff_quantizer") or {}
            cache_matches = (
                bool(cache.get("coeff_discretized", False))
                and int(cache_meta.get("n_coeff_bins", -1)) == int(args.n_coeff_bins)
                and float(cache_meta.get("coeff_max_val", float("nan")))
                == float(args.coeff_max_val)
                and float(cache_meta.get("coeff_mu", float("nan")))
                == float(args.coeff_mu)
            )
            if not cache_matches:
                cache["coeffs"] = coeffs_flat
                cache["coeff_discretized"] = True
                cache["coeff_quantizer"] = {
                    "n_coeff_bins": args.n_coeff_bins,
                    "coeff_max_val": args.coeff_max_val,
                    "coeff_mu": args.coeff_mu,
                }
                torch.save(cache, token_cache_path)
                print("[Stage2] updated cached coeffs to discretized values.")
            class_ids_flat = cache.get("class_ids")
            H, W, D = cache["shape"]
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            ae = ae.to(device)
            tok_loader = DataLoader(
                train_set, batch_size=args.batch_size,
                shuffle=False, num_workers=0, pin_memory=True,
            )
            tokens_flat, coeffs_flat, class_ids_flat, H, W, D = precompute_tokens(
                ae, tok_loader, device,
                max_items=(
                    min(args.token_subset, len(train_set))
                    if args.token_subset > 0 else None
                ),
                collect_labels=(stage2_num_classes > 0),
                coeff_quantizer=coeff_quantizer,
            )
            ae.cpu()
            torch.save({
                "tokens": tokens_flat,
                "coeffs": coeffs_flat,
                "class_ids": class_ids_flat,
                "shape": (H, W, D),
                "stage2_num_classes": stage2_num_classes,
                "coeff_discretized": True,
                "coeff_quantizer": {
                    "n_coeff_bins": args.n_coeff_bins,
                    "coeff_max_val": args.coeff_max_val,
                    "coeff_mu": args.coeff_mu,
                },
            }, token_cache_path)
        return tokens_flat, coeffs_flat, class_ids_flat, H, W, D

    def _clear_ddp_env_for_stage2():
        for k in (
            "LOCAL_RANK", "RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
            "GROUP_RANK", "ROLE_RANK", "NODE_RANK",
            "MASTER_ADDR", "MASTER_PORT",
        ):
            os.environ.pop(k, None)

    def _run_stage2(
        tokens_flat: torch.Tensor,
        coeffs_flat: torch.Tensor,
        class_ids_flat: Optional[torch.Tensor],
        H: int, W: int, D: int,
        ae: SparseDictAE,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("Stage-2 requires CUDA.")

        tensors = [tokens_flat, coeffs_flat]
        if class_ids_flat is not None:
            tensors.append(class_ids_flat)
        ds = TensorDataset(*tensors)
        dl = DataLoader(
            ds,
            batch_size=args.stage2_batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )

        if args.stage2_arch == "decoder_dual":
            from dual_head_decoder_transformer import (
                DualHeadDecoderConfig,
                DualHeadDecoderPrior,
            )

            cfg = DualHeadDecoderConfig(
                vocab_size=ae.bottleneck.vocab_size,
                H=H,
                W=W,
                D=D,
                num_classes=stage2_num_classes,
                d_model=args.tf_d_model,
                n_heads=args.tf_heads,
                n_layers=args.tf_layers,
                d_ff=args.tf_ff,
                dropout=args.tf_dropout,
                coeff_pred_atom_mix=args.dual_coeff_pred_atom_mix,
                coeff_atom_coherence_weight=args.dual_coeff_atom_coherence_weight,
                n_coeff_bins=args.n_coeff_bins,
                coeff_mu=args.coeff_mu,
                coeff_max_val=args.coeff_max_val,
            )
            transformer = DualHeadDecoderPrior(cfg)
        elif args.stage2_arch == "mingpt":
            from mingpt import MinGPTSparse, MinGPTSparseConfig

            cfg = MinGPTSparseConfig(
                vocab_size=ae.bottleneck.vocab_size,
                H=H,
                W=W,
                D=D,
                num_classes=stage2_num_classes,
                d_model=args.tf_d_model,
                n_heads=args.tf_heads,
                n_layers=args.tf_layers,
                dropout=args.tf_dropout,
                n_coeff_bins=args.n_coeff_bins,
                coeff_mu=args.coeff_mu,
                coeff_max_val=args.coeff_max_val,
            )
            transformer = MinGPTSparse(cfg)
        else:
            cfg = RQTransformerConfig(
                vocab_size=ae.bottleneck.vocab_size,
                H=H, W=W, D=D,
                num_classes=stage2_num_classes,
                d_model=args.tf_d_model,
                n_heads=args.tf_heads,
                n_layers=args.tf_layers,
                n_depth_layers=args.tf_depth_layers,
                d_ff=args.tf_ff,
                dropout=args.tf_dropout,
                use_pred_coeff_feedback=args.rq_use_pred_coeff_feedback,
                use_pred_atom_feedback=args.rq_use_pred_atom_feedback,
                stochastic_feedback_sampling=args.rq_stochastic_feedback_sampling,
                feedback_temperature=args.rq_feedback_temperature,
                atom_label_smoothing=args.rq_atom_label_smoothing,
                coeff_adjacent_soft_target=args.rq_coeff_adjacent_soft_target,
                n_coeff_bins=args.n_coeff_bins,
                coeff_mu=args.coeff_mu,
                coeff_max_val=args.coeff_max_val,
            )
            transformer = RQTransformerPrior(cfg)
        print(f"[Stage2] architecture: {args.stage2_arch}")

        if args.stage2_resume_from_last:
            ckpt = os.path.join(stage2_dir, "transformer_last.pt")
            if os.path.exists(ckpt):
                state = torch.load(ckpt, map_location="cpu", weights_only=True)
                model_state = transformer.state_dict()
                filtered = {
                    k: v for k, v in state.items()
                    if k in model_state and model_state[k].shape == v.shape
                }
                transformer.load_state_dict(filtered, strict=False)
                print(f"[Stage2] resumed from {ckpt}")

        module = Stage2Module(
            transformer=transformer,
            ae=ae,
            lr=args.stage2_lr,
            out_dir=stage2_dir,
            H=H, W=W, D=D,
            sample_every_steps=args.stage2_sample_every_steps,
            sample_batch_size=args.stage2_sample_batch_size,
            sample_temperature=args.stage2_sample_temperature,
            direct_coeff_loss_weight=args.stage2_direct_coeff_loss_weight,
            lr_schedule=args.stage2_lr_schedule,
            warmup_epochs=args.stage2_warmup_epochs,
            min_lr_ratio=args.stage2_min_lr_ratio,
            sample_class_id=args.stage2_sample_class_id,
            sample_image_size=(
                args.stage2_sample_image_size
                if args.stage2_sample_image_size > 0 else None
            ),
        )

        s2_strategy = _build_ddp_strategy()
        if args.devices > 1:
            os.environ["LASER_DDP_PHASE"] = "stage2"
            _ensure_free_master_port("Stage2")

        trainer = pl.Trainer(
            accelerator="gpu",
            devices=args.devices,
            strategy=s2_strategy,
            max_epochs=args.stage2_epochs,
            logger=_wandb_logger("stage2"),
            enable_checkpointing=False,
            gradient_clip_val=1.0,
            log_every_n_steps=10,
        )
        trainer.fit(module, train_dataloaders=dl)

    # ---- DDP re-entry for stage-2 workers ----

    if os.environ.get("LASER_DDP_PHASE") == "stage2":
        if not os.path.exists(token_cache_path):
            raise FileNotFoundError(f"Missing token cache: {token_cache_path}")
        cache = torch.load(token_cache_path, map_location="cpu",
                           weights_only=False)
        coeff_quantizer = CoefficientQuantizer(
            args.n_coeff_bins, args.coeff_max_val, args.coeff_mu,
        )
        cache_coeffs = coeff_quantizer.decode(
            coeff_quantizer.encode(cache["coeffs"].float()),
        )
        ae = _build_ae()
        _load_ae(ae, os.path.join(stage1_dir, "ae_best.pt"))
        H, W, D = cache["shape"]
        _run_stage2(
            cache["tokens"], cache_coeffs, cache.get("class_ids"),
            H, W, D, ae,
        )
        return

    # ---- transforms ----

    tfm = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # ---- datasets ----

    train_set, val_set = _build_datasets(tfm)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=64,
        shuffle=False, num_workers=2, pin_memory=True,
    )

    vis_loader = DataLoader(
        val_set, batch_size=min(64, len(val_set)),
        shuffle=False, num_workers=0,
    )
    val_vis_images, _ = next(iter(vis_loader))

    # ---- stage-1 ----

    ae = _build_ae()
    if args.stage1_init_ckpt:
        _load_ae(ae, args.stage1_init_ckpt, tag="Stage1 init")

    if args.stage1_epochs > 0:
        if not torch.cuda.is_available():
            raise RuntimeError("Stage-1 requires CUDA.")

        stage1_mod = Stage1Module(
            ae=ae, lr=args.stage1_lr,
            bottleneck_weight=args.bottleneck_weight,
            out_dir=stage1_dir,
            val_vis_images=val_vis_images,
            lr_schedule=args.stage1_lr_schedule,
            warmup_epochs=args.stage1_warmup_epochs,
            min_lr_ratio=args.stage1_min_lr_ratio,
        )

        if args.devices > 1:
            os.environ["LASER_DDP_PHASE"] = "stage1"
            _ensure_free_master_port("Stage1")
        s1_strategy = _build_ddp_strategy()
        pl.Trainer(
            accelerator="gpu",
            devices=args.devices,
            strategy=s1_strategy,
            max_epochs=args.stage1_epochs,
            logger=_wandb_logger("stage1"),
            enable_checkpointing=False,
            gradient_clip_val=args.grad_clip,
            log_every_n_steps=10,
        ).fit(stage1_mod, train_dataloaders=train_loader,
              val_dataloaders=val_loader)

        if os.environ.get("LOCAL_RANK") not in (None, "0"):
            return

    _load_ae(ae, os.path.join(stage1_dir, "ae_best.pt"))

    # ---- precompute tokens ----

    tokens_flat, coeffs_flat, class_ids_flat, H, W, D = (
        _load_or_precompute_tokens(ae, train_set)
    )

    print(f"[Stage2] tokens: {tokens_flat.shape}  (H={H}, W={W}, D={D})")

    # ---- stage-2 ----

    if args.stage1_epochs <= 0:
        _run_stage2(
            tokens_flat, coeffs_flat, class_ids_flat, H, W, D, ae,
        )
        return

    # Re-exec into a clean process so CUDA / NCCL state from stage-1 DDP
    # doesn't conflict with stage-2 DDP.
    os.environ["LASER_DDP_PHASE"] = "stage2"
    _clear_ddp_env_for_stage2()
    print("[Stage2] restarting for clean DDP launch...")
    ret = subprocess.call(
        [sys.executable, __file__, *sys.argv[1:]],
        env=os.environ.copy(), close_fds=True,
    )
    if ret != 0:
        raise RuntimeError(f"Stage-2 restart failed with exit code {ret}.")


if __name__ == "__main__":
    main()
=======
"""
cifar10_sparse_dict_rqtransformer.py

A minimal, end-to-end "RQ-VAE-ish" pipeline using:
  - VQ-VAE-style Encoder/Decoder (conv + residual stack) (from LASER's VQ-VAE baseline)
  - Dictionary-learning bottleneck with batched OMP sparse coding (LASER-style)
  - Option A tokenization: token = atom_id * n_bins + coef_bin
  - A simple "RQTransformer prior" (GPT-style causal transformer) over (H,W,D) stacks
  - CIFAR-10 / CelebA quick test

NEW: High-res friendly random cropping during data loading
  - Decode high-res images on CPU
  - Random crop to a fixed crop_size BEFORE sending to GPU
  - Keeps GPU memory bounded by crop_size (not original image resolution)

Run:
  python cifar10_sparse_dict_rqtransformer.py --stage1_epochs 5 --stage2_epochs 10

This is intentionally compact and hackable, not "best possible" training.
"""
import argparse
import math
import os
import sys
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torchvision import datasets, transforms, utils
from tqdm import tqdm
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except Exception:
    FrechetInceptionDistance = None
try:
    import wandb
except Exception:
    wandb = None


# -----------------------------
# VQ-VAE style building blocks
# -----------------------------

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FlatImageDataset(Dataset):
    """
    Recursively loads images from a directory tree.
    Returns (image_tensor, dummy_label) to match torchvision dataset API.
    """

    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root}")

        image_paths = [
            p for p in self.root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
        ]
        image_paths.sort()
        if not image_paths:
            raise RuntimeError(f"No images found under: {self.root}")
        self.image_paths = image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, num_hiddens: int, num_residual_hiddens: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, num_residual_hiddens, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_residual_hiddens)
        self.conv2 = nn.Conv2d(num_residual_hiddens, num_hiddens, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_hiddens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=False)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = F.relu(out, inplace=False)
        return out


class ResidualStack(nn.Module):
    def __init__(self, in_channels: int, num_hiddens: int, num_residual_layers: int, num_residual_hiddens: int):
        super().__init__()
        self.layers = nn.ModuleList([
            ResidualBlock(in_channels=in_channels, num_hiddens=num_hiddens, num_residual_hiddens=num_residual_hiddens)
            for _ in range(num_residual_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return F.relu(x, inplace=False)


class Encoder(nn.Module):
    """
    VQ-VAE style encoder:
      32x32 -> 16x16 -> 8x8 (two strided convs), then residual stack.
    """
    def __init__(self, in_channels: int, num_hiddens: int, num_residual_layers: int, num_residual_hiddens: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, num_hiddens // 2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(num_hiddens // 2, num_hiddens, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(num_hiddens, num_hiddens, kernel_size=3, stride=1, padding=1)
        self.res = ResidualStack(num_hiddens, num_hiddens, num_residual_layers, num_residual_hiddens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.res(x)
        return x


class Decoder(nn.Module):
    """
    VQ-VAE style decoder:
      8x8 -> 16x16 -> 32x32 (two transposed convs), with residual stack.
    """
    def __init__(self, in_channels: int, num_hiddens: int, num_residual_layers: int,
                 num_residual_hiddens: int, out_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, num_hiddens, kernel_size=3, stride=1, padding=1)
        self.res = ResidualStack(num_hiddens, num_hiddens, num_residual_layers, num_residual_hiddens)
        self.deconv1 = nn.ConvTranspose2d(num_hiddens, num_hiddens // 2, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(num_hiddens // 2, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.res(x)
        x = F.relu(self.deconv1(x))
        x = self.deconv2(x)
        return x


# -----------------------------
# Dictionary learning bottleneck (batch OMP) + Option-A tokenization
# -----------------------------

class DictionaryLearningTokenized(nn.Module):
    """
    Dictionary-learning bottleneck with batched Orthogonal Matching Pursuit (OMP) sparse coding.
    Option A tokenization: token = atom_id * n_bins + coefficient_bin.

    Outputs, per latent pixel, a *stack* of length D=sparsity_level.

    Important simplifications (good for a quick test):
      - OMP runs under torch.no_grad() like in LASER: we do NOT backprop through sparse coding.
      - We reconstruct the latent using quantized coefficients, then apply STE so the encoder
        still receives gradients (VQ-VAE style).
    """
    def __init__(
        self,
        num_embeddings: int = 256,
        embedding_dim: int = 64,
        sparsity_level: int = 4,
        n_bins: int = 129,
        coef_max: float = 3.0,
        commitment_cost: float = 0.25,
        epsilon: float = 1e-10,
    ):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.sparsity_level = int(sparsity_level)
        self.n_bins = int(n_bins)
        self.coef_max = float(coef_max)
        self.commitment_cost = float(commitment_cost)
        self.epsilon = float(epsilon)

        # Dictionary shape [C, K] (matches LASER)
        self.dictionary = nn.Parameter(torch.randn(self.embedding_dim, self.num_embeddings) * 0.02)

        # Coefficient bin centers (uniform)
        centers = torch.linspace(-self.coef_max, self.coef_max, steps=self.n_bins)
        self.register_buffer("coef_bin_centers", centers)

        # Special tokens (for the transformer)
        self.pad_token_id = self.num_embeddings * self.n_bins
        self.bos_token_id = self.pad_token_id + 1
        self.vocab_size = self.num_embeddings * self.n_bins + 2

    def _normalize_dict(self) -> torch.Tensor:
        return F.normalize(self.dictionary, p=2, dim=0, eps=self.epsilon)

    def _quantize_coeff(self, coeff: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize coefficients to bins; return (bin_idx, bin_center_value)."""
        c = coeff.clamp(-self.coef_max, self.coef_max)
        scaled = (c + self.coef_max) / (2 * self.coef_max)  # [0,1]
        bin_f = scaled * (self.n_bins - 1)
        bin_idx = torch.round(bin_f).to(torch.long).clamp(0, self.n_bins - 1)
        coeff_q = self.coef_bin_centers[bin_idx]
        return bin_idx, coeff_q

    def batch_omp_with_support(self, X: torch.Tensor, D: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched OMP adapted from LASER's DictionaryLearning.batch_omp.
        Runs exactly sparsity_level steps (no early-stop) so stack depth is fixed.

        Args:
            X: [M, B] signals
            D: [M, N] normalized dictionary
        Returns:
            support: [B, K] indices in selection order (K = sparsity_level)
            coeffs:  [B, K] coefficients aligned with support (same order)
        """
        _, batch_size = X.size()
        Dt = D.t()                   # [N, M]
        G = Dt.mm(D)                 # [N, N]
        eps = torch.norm(X, dim=0)   # [B]
        h_bar = Dt.mm(X).t()         # [B, N]
        h = h_bar
        x = torch.zeros_like(h_bar)  # [B, N]
        L = torch.ones(batch_size, 1, 1, device=h.device, dtype=h.dtype)
        I = torch.ones(batch_size, 0, device=h.device, dtype=torch.long)
        I_logic = torch.zeros_like(h_bar, dtype=torch.bool)
        delta = torch.zeros(batch_size, device=h.device, dtype=h.dtype)

        def _update_logical(logical: torch.Tensor, to_add: torch.Tensor):
            running_idx = torch.arange(to_add.shape[0], device=to_add.device)
            logical[running_idx, to_add] = True

        k = 0
        tol = -1.0  # force exactly K steps (eps.max() is always > -1)

        while k < self.sparsity_level and eps.max() > tol:
            k += 1
            index = (h * (~I_logic).float()).abs().argmax(dim=1)  # [B]
            _update_logical(I_logic, index)

            batch_idx = torch.arange(batch_size, device=G.device)
            expanded_batch_idx = batch_idx.unsqueeze(0).expand(k, batch_size).t()  # [B,k]

            if k > 1:
                G_stack = G[I[batch_idx, :], index[expanded_batch_idx[..., :-1]]].view(batch_size, k - 1, 1)
                w = torch.linalg.solve_triangular(L, G_stack, upper=False)
                w = w.view(-1, 1, k - 1)
                w_corner = torch.sqrt(torch.clamp(1 - (w ** 2).sum(dim=2, keepdim=True), min=1e-12))
                k_zeros = torch.zeros(batch_size, k - 1, 1, device=h.device, dtype=h.dtype)
                L = torch.cat(
                    (
                        torch.cat((L, k_zeros), dim=2),
                        torch.cat((w, w_corner), dim=2),
                    ),
                    dim=1,
                )

            I = torch.cat([I, index.unsqueeze(1)], dim=1)  # [B,k]

            h_stack = h_bar[expanded_batch_idx, I[batch_idx, :]].view(batch_size, k, 1)
            x_stack = torch.cholesky_solve(h_stack, L)
            x[batch_idx.unsqueeze(1), I[batch_idx]] = x_stack[batch_idx].squeeze(-1)

            beta = (
                x[batch_idx.unsqueeze(1), I[batch_idx]]
                .unsqueeze(1)
                .bmm(G[I[batch_idx], :])
                .squeeze(1)
            )
            h = h_bar - beta

            new_delta = (x * beta).sum(dim=1)
            eps = eps + delta - new_delta
            delta = new_delta

        # Ordered coefficients: x is dense over atoms; gather along support I
        batch_idx = torch.arange(batch_size, device=x.device)[:, None]
        coeffs_ordered = x[batch_idx, I].squeeze(1)  # [B, K]

        return I, coeffs_ordered

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_e: [B, C, H, W]
        Returns:
            z_q_ste: [B, C, H, W]
            loss: scalar bottleneck loss
            tokens: [B, H, W, D] (D=sparsity_level)
        """
        if z_e.dim() != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(z_e.shape)}")
        B, C, H, W = z_e.shape
        if C != self.embedding_dim:
            raise ValueError(f"Expected channel dim {self.embedding_dim}, got {C}")

        # [B,C,H,W] -> [C, B*H*W]
        signals = z_e.permute(0, 2, 3, 1).contiguous().view(-1, C).t()

        dictionary = self._normalize_dict()

        with torch.no_grad():
            support, coeffs = self.batch_omp_with_support(signals, dictionary)
            # support: [Nsig, D], coeffs: [Nsig, D]

        # Quantize coefficients (Option A)
        bin_idx, coeff_q = self._quantize_coeff(coeffs)  # both [Nsig, D]

        # Tokens: [Nsig, D] -> [B,H,W,D]
        tokens = (support * self.n_bins + bin_idx).view(B, H, W, self.sparsity_level)

        # Reconstruct latent using (support, coeff_q)
        dict_t = dictionary.t()  # [K, C]
        atoms = dict_t[support]  # [Nsig, D, C]
        recon_flat = (atoms * coeff_q.unsqueeze(-1)).sum(dim=1)  # [Nsig, C]
        z_q = recon_flat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # Bottleneck loss (LASER-style)
        dl_latent_loss = F.mse_loss(z_q, z_e.detach())
        e_latent_loss = F.mse_loss(z_q.detach(), z_e)
        loss = dl_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator to encoder
        z_q_ste = z_e + (z_q - z_e).detach()
        return z_q_ste, loss, tokens

    @torch.no_grad()
    def tokens_to_latent(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Decode tokens back to a latent map.
        Args:
            tokens: [B, H, W, D]
        Returns:
            z_q: [B, C, H, W]
        """
        if tokens.dim() != 4:
            raise ValueError(f"Expected [B,H,W,D], got {tuple(tokens.shape)}")
        B, H, W, D = tokens.shape
        if D != self.sparsity_level:
            raise ValueError(f"Expected D={self.sparsity_level}, got {D}")

        dictionary = self._normalize_dict()
        dict_t = dictionary.t()  # [K, C]

        tok = tokens.to(torch.long)
        special = tok >= self.pad_token_id  # pad or bos
        tok_clamped = tok.clamp_max(self.pad_token_id - 1)

        atom = tok_clamped // self.n_bins
        bin_idx = tok_clamped % self.n_bins

        coeff = self.coef_bin_centers[bin_idx].to(dict_t.dtype)
        coeff = coeff * (~special).float()
        atom = atom * (~special).long()

        Nsig = B * H * W
        atom_flat = atom.view(Nsig, D)
        coeff_flat = coeff.view(Nsig, D)

        atoms = dict_t[atom_flat]  # [Nsig, D, C]
        recon_flat = (atoms * coeff_flat.unsqueeze(-1)).sum(dim=1)  # [Nsig, C]
        z_q = recon_flat.view(B, H, W, self.embedding_dim).permute(0, 3, 1, 2).contiguous()
        return z_q


# -----------------------------
# Stage-1 model: Encoder + Dictionary bottleneck + Decoder
# -----------------------------

class SparseDictAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_hiddens: int = 128,
        num_residual_layers: int = 2,
        num_residual_hiddens: int = 32,
        embedding_dim: int = 64,
        num_embeddings: int = 256,
        sparsity_level: int = 4,
        commitment_cost: float = 0.25,
        n_bins: int = 129,
        coef_max: float = 3.0,
        out_tanh: bool = True,
    ):
        super().__init__()
        self.out_tanh = bool(out_tanh)

        self.encoder = Encoder(in_channels, num_hiddens, num_residual_layers, num_residual_hiddens)
        self.pre = nn.Conv2d(num_hiddens, embedding_dim, kernel_size=1)
        self.bottleneck = DictionaryLearningTokenized(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            sparsity_level=sparsity_level,
            n_bins=n_bins,
            coef_max=coef_max,
            commitment_cost=commitment_cost,
        )
        self.post = nn.Conv2d(embedding_dim, num_hiddens, kernel_size=3, padding=1)
        self.decoder = Decoder(num_hiddens, num_hiddens, num_residual_layers, num_residual_hiddens, out_channels=in_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        z = self.pre(z)
        z_q, b_loss, tokens = self.bottleneck(z)
        z_q = self.post(z_q)
        recon = self.decoder(z_q)
        if self.out_tanh:
            recon = torch.tanh(recon)
        return recon, b_loss, tokens

    @torch.no_grad()
    def encode_to_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        z = self.encoder(x)
        z = self.pre(z)
        _, _, tokens = self.bottleneck(z)
        return tokens, tokens.shape[1], tokens.shape[2]

    @torch.no_grad()
    def decode_from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        z_q = self.bottleneck.tokens_to_latent(tokens)
        z_q = self.post(z_q)
        recon = self.decoder(z_q)
        if self.out_tanh:
            recon = torch.tanh(recon)
        return recon


# -----------------------------
# Stage-2: RQTransformer prior (GPT-style causal transformer over stacks)
# -----------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, max_seq_len: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("causal_mask", mask.view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, max_seq_len: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


@dataclass
class RQTransformerConfig:
    vocab_size: int
    H: int
    W: int
    D: int
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1


class RQTransformerPrior(nn.Module):
    """
    A simple RQ-style prior:
      - The full sequence is: [BOS] + raster_scan(H*W) each with depth D tokens.
      - Embedding = token + spatial_pos + depth_pos + type(BOS vs normal)
      - GPT-style causal blocks.
    """
    def __init__(self, cfg: RQTransformerConfig, bos_token_id: int, pad_token_id: int):
        super().__init__()
        self.cfg = cfg
        self.bos_token_id = int(bos_token_id)
        self.pad_token_id = int(pad_token_id)

        self.max_len = 1 + cfg.H * cfg.W * cfg.D

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.spatial_emb = nn.Embedding(cfg.H * cfg.W, cfg.d_model)
        self.depth_emb = nn.Embedding(cfg.D, cfg.d_model)
        self.type_emb = nn.Embedding(2, cfg.d_model)

        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout, self.max_len)
            for _ in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Precompute position ids for [BOS] + H*W*D tokens
        spatial_ids = torch.zeros(self.max_len, dtype=torch.long)
        depth_ids = torch.zeros(self.max_len, dtype=torch.long)
        type_ids = torch.zeros(self.max_len, dtype=torch.long)  # 0 for BOS, 1 for normal
        if self.max_len > 1:
            idx = torch.arange(self.max_len - 1)
            spatial_ids[1:] = idx // cfg.D
            depth_ids[1:] = idx % cfg.D
            type_ids[1:] = 1
        self.register_buffer("_spatial_ids", spatial_ids)
        self.register_buffer("_depth_ids", depth_ids)
        self.register_buffer("_type_ids", type_ids)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L] tokens, L <= max_len
        Returns:
            logits: [B, L, vocab]
        """
        B, L = x.shape
        if L > self.max_len:
            raise ValueError(f"Got L={L}, but max_len={self.max_len}")

        tok = self.token_emb(x)
        sp = self.spatial_emb(self._spatial_ids[:L])
        dp = self.depth_emb(self._depth_ids[:L])
        tp = self.type_emb(self._type_ids[:L])

        h = tok + sp.unsqueeze(0) + dp.unsqueeze(0) + tp.unsqueeze(0)
        h = self.drop(h)

        for block in self.blocks:
            h = block(h)

        h = self.ln_f(h)
        logits = self.head(h)
        return logits

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Unconditional generation.
        Returns:
            flat_tokens: [B, H*W*D] (without BOS)
        """
        device = next(self.parameters()).device
        T = self.cfg.H * self.cfg.W * self.cfg.D

        seq = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=device)
        steps = tqdm(
            range(T),
            desc=(progress_desc or "[Stage2] sampling tokens"),
            leave=False,
            dynamic_ncols=True,
            disable=(not show_progress),
        )
        for _ in steps:
            logits = self(seq)[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None and top_k > 0:
                v, ix = torch.topk(logits, top_k, dim=-1)
                mask = torch.full_like(logits, float("-inf"))
                mask.scatter_(1, ix, v)
                logits = mask
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            seq = torch.cat([seq, nxt], dim=1)

        return seq[:, 1:]


# -----------------------------
# Training helpers
# -----------------------------

def _make_image_grid(x: torch.Tensor, nrow: int = 8) -> torch.Tensor:
    """Build image grid tensor from a batch in [-1, 1]."""
    x = x.detach().cpu().clamp(-1, 1)
    x = (x + 1.0) / 2.0
    return utils.make_grid(x, nrow=nrow)


def save_image_grid(x: torch.Tensor, path: str, nrow: int = 8):
    """
    Save a grid of images. Expects x in [-1,1] (we'll map to [0,1]).
    """
    grid = _make_image_grid(x, nrow=nrow)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    utils.save_image(grid, path)


def _resolve_wandb_logger(logger_obj):
    if isinstance(logger_obj, WandbLogger):
        return logger_obj
    for lg in getattr(logger_obj, "loggers", []):
        if isinstance(lg, WandbLogger):
            return lg
    return None


def log_image_grid_wandb(
    logger_obj,
    key: str,
    x: torch.Tensor,
    step: int,
    nrow: int = 8,
    caption: Optional[str] = None,
) -> bool:
    wb_logger = _resolve_wandb_logger(logger_obj)
    if wb_logger is None or wandb is None:
        return False

    grid = _make_image_grid(x, nrow=nrow)
    grid_np = grid.permute(1, 2, 0).mul(255).clamp(0, 255).byte().numpy()
    wb_image = wandb.Image(grid_np, caption=caption) if caption else wandb.Image(grid_np)
    wb_logger.experiment.log({key: wb_image}, step=int(step))
    return True


def log_scalar_wandb(
    logger_obj,
    key: str,
    value: float,
    step: int,
) -> bool:
    wb_logger = _resolve_wandb_logger(logger_obj)
    if wb_logger is None or wandb is None:
        return False
    wb_logger.experiment.log({key: float(value)}, step=int(step))
    return True


class Stage2TokenDataModule(pl.LightningDataModule):
    def __init__(self, tokens_flat: torch.Tensor, batch_size: int, num_workers: int = 2):
        super().__init__()
        self.tokens_flat = tokens_flat
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)

    def train_dataloader(self):
        tok_ds = TensorDataset(self.tokens_flat)
        return DataLoader(
            tok_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(self.num_workers > 0),
        )


class Stage1LightningModule(pl.LightningModule):
    def __init__(
        self,
        ae: SparseDictAE,
        lr: float,
        bottleneck_weight: float,
        out_dir: str,
        fid_num_samples: int = 1024,
        fid_feature: int = 2048,
        fid_compute_batch_size: int = 32,
        lr_schedule: str = "cosine",
        warmup_epochs: int = 1,
        min_lr_ratio: float = 0.1,
    ):
        super().__init__()
        self.ae = ae
        self.lr = float(lr)
        self.bottleneck_weight = float(bottleneck_weight)
        self.out_dir = out_dir
        self.best_val = float("inf")
        self._val_vis = None
        self.fid_num_samples = max(0, int(fid_num_samples))
        self.fid_feature = int(fid_feature)
        self.fid_compute_batch_size = max(1, int(fid_compute_batch_size))
        self.lr_schedule = str(lr_schedule)
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.min_lr_ratio = float(max(0.0, min(min_lr_ratio, 1.0)))
        self._fid_real = []
        self._fid_fake = []
        self._fid_seen = 0
        self._fid_warned_unavailable = False
        self._fid_metric = None
        self._fid_metric_feature = None

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.ae.parameters(), lr=self.lr)
        if self.lr_schedule != "cosine":
            return opt

        max_epochs = 1
        if getattr(self, "_trainer", None) is not None:
            max_epochs = int(getattr(self.trainer, "max_epochs", 1) or 1)
        max_epochs = max(1, max_epochs)
        warmup_epochs = min(self.warmup_epochs, max_epochs - 1)
        min_ratio = self.min_lr_ratio

        def lr_lambda(epoch: int) -> float:
            # Lightning steps epoch schedulers once per epoch; use 1-indexed step count.
            step_idx = int(epoch) + 1
            if warmup_epochs > 0 and step_idx <= warmup_epochs:
                return 0.1 + 0.9 * (step_idx / float(max(1, warmup_epochs)))

            decay_steps = max(1, max_epochs - warmup_epochs)
            decay_idx = min(max(step_idx - warmup_epochs, 0), decay_steps)
            t = decay_idx / float(decay_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * t))
            return min_ratio + (1.0 - min_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _get_or_create_fid_metric(self, feature: int):
        feat = int(feature)
        if self._fid_metric is None or self._fid_metric_feature != feat:
            self._fid_metric = FrechetInceptionDistance(
                feature=feat,
                sync_on_compute=False,
            ).to(self.device)
            self._fid_metric_feature = feat
        else:
            self._fid_metric = self._fid_metric.to(self.device)
        self._fid_metric.reset()
        return self._fid_metric

    def on_validation_epoch_start(self):
        self._fid_real = []
        self._fid_fake = []
        self._fid_seen = 0

    def training_step(self, batch, batch_idx):
        x, _ = batch
        recon, b_loss, _ = self.ae(x)
        recon_loss = F.mse_loss(recon, x)
        loss = recon_loss + self.bottleneck_weight * b_loss
        self.log("stage1/train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=x.size(0))
        self.log("stage1/recon_loss", recon_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=x.size(0))
        self.log("stage1/b_loss", b_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=x.size(0))
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Normalize dictionary only after a full optimization step finishes.
        # Doing this inside training_step can invalidate autograd versioning.
        with torch.no_grad():
            self.ae.bottleneck.dictionary.copy_(
                F.normalize(self.ae.bottleneck.dictionary, p=2, dim=0, eps=self.ae.bottleneck.epsilon)
            )

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        recon, b_loss, _ = self.ae(x)
        recon_loss = F.mse_loss(recon, x)
        loss = recon_loss + self.bottleneck_weight * b_loss
        psnr = -10.0 * torch.log10(torch.clamp(recon_loss.detach(), min=1e-8))
        self.log("stage1/val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=x.size(0))
        self.log("stage1/val_psnr", psnr, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=x.size(0))

        if batch_idx == 0 and self.trainer.is_global_zero:
            self._val_vis = (x[:64].detach(), recon[:64].detach())

        if (not self.trainer.sanity_checking) and self.fid_num_samples > 0:
            if FrechetInceptionDistance is None:
                if self.trainer.is_global_zero and not self._fid_warned_unavailable:
                    print("[Stage1] FID unavailable: torchmetrics.image.fid not installed.")
                    self._fid_warned_unavailable = True
            elif self._fid_seen < self.fid_num_samples:
                keep = min(x.size(0), self.fid_num_samples - self._fid_seen)
                if keep > 0:
                    real_u8 = ((x[:keep].detach().cpu().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
                    fake_u8 = ((recon[:keep].detach().cpu().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
                    self._fid_real.append(real_u8)
                    self._fid_fake.append(fake_u8)
                    self._fid_seen += keep
        return loss

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero and not self.trainer.sanity_checking:
            os.makedirs(self.out_dir, exist_ok=True)
            cur = self.trainer.callback_metrics.get("stage1/val_loss")
            if cur is not None:
                cur_val = float(cur.detach().cpu().item())

                torch.save(self.ae.state_dict(), os.path.join(self.out_dir, "ae_last.pt"))
                if cur_val < self.best_val:
                    self.best_val = cur_val
                    torch.save(self.ae.state_dict(), os.path.join(self.out_dir, "ae_best.pt"))

            if self._val_vis is not None:
                x_vis, recon_vis = self._val_vis
                epoch = int(self.current_epoch + 1)
                log_step = int(self.global_step)
                logged_real = log_image_grid_wandb(
                    self.logger,
                    key="stage1/real",
                    x=x_vis,
                    step=log_step,
                    caption=f"epoch={epoch} real",
                )
                logged_recon = log_image_grid_wandb(
                    self.logger,
                    key="stage1/recon",
                    x=recon_vis,
                    step=log_step,
                    caption=f"epoch={epoch} recon",
                )
                if not (logged_real and logged_recon):
                    save_image_grid(x_vis, os.path.join(self.out_dir, f"stage1_epoch{epoch:03d}_real.png"))
                    save_image_grid(recon_vis, os.path.join(self.out_dir, f"stage1_epoch{epoch:03d}_recon.png"))
                self._val_vis = None

        if (
            not self.trainer.sanity_checking
            and
            self.fid_num_samples > 0
            and FrechetInceptionDistance is not None
            and self._fid_seen > 0
            and self._fid_real
            and self._fid_fake
        ):
            with torch.no_grad():
                fid_metric = self._get_or_create_fid_metric(self.fid_feature)
                for real_cpu, fake_cpu in zip(self._fid_real, self._fid_fake):
                    bs = int(self.fid_compute_batch_size)
                    for start in range(0, real_cpu.size(0), bs):
                        end = start + bs
                        real_chunk = real_cpu[start:end].to(self.device)
                        fake_chunk = fake_cpu[start:end].to(self.device)
                        fid_metric.update(real_chunk, real=True)
                        fid_metric.update(fake_chunk, real=False)
                fid_value = fid_metric.compute().detach()
                fid_metric.reset()
            self.log("stage1/fid", fid_value, on_epoch=True, prog_bar=True, sync_dist=True)
            self._fid_real = []
            self._fid_fake = []
            self._fid_seen = 0


class Stage2LightningModule(pl.LightningModule):
    def __init__(
        self,
        transformer: RQTransformerPrior,
        lr: float,
        pad_token_id: int,
        out_dir: str,
        ae_for_decode: SparseDictAE,
        H: int,
        W: int,
        D: int,
        sample_every_steps: int = 200,
        sample_batch_size: int = 8,
        fid_real_images: Optional[torch.Tensor] = None,
        fid_num_samples: int = 64,
        fid_feature: int = 2048,
        fid_every_n_epochs: int = 1,
        lr_schedule: str = "cosine",
        warmup_epochs: int = 1,
        min_lr_ratio: float = 0.1,
    ):
        super().__init__()
        self.transformer = transformer
        self.lr = float(lr)
        self.pad_token_id = int(pad_token_id)
        self.out_dir = out_dir
        self.ae_for_decode = ae_for_decode
        self.H, self.W, self.D = int(H), int(W), int(D)
        self.sample_every_steps = int(sample_every_steps)
        self.sample_batch_size = int(sample_batch_size)
        self.fid_real_images = fid_real_images
        self.fid_num_samples = max(0, int(fid_num_samples))
        self.fid_feature = int(fid_feature)
        self.fid_every_n_epochs = max(1, int(fid_every_n_epochs))
        self.lr_schedule = str(lr_schedule)
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.min_lr_ratio = float(max(0.0, min(min_lr_ratio, 1.0)))
        self._fid_warned_unavailable = False
        self._fid_warned_adjusted_feature = False
        self._fid_warned_compute_failed = False
        self._fid_metric = None
        self._fid_metric_feature = None

        self.ae_for_decode.eval()
        for p in self.ae_for_decode.parameters():
            p.requires_grad_(False)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.transformer.parameters(), lr=self.lr)
        if self.lr_schedule != "cosine":
            return opt

        max_epochs = 1
        if getattr(self, "_trainer", None) is not None:
            max_epochs = int(getattr(self.trainer, "max_epochs", 1) or 1)
        max_epochs = max(1, max_epochs)
        warmup_epochs = min(self.warmup_epochs, max_epochs - 1)
        min_ratio = self.min_lr_ratio

        def lr_lambda(epoch: int) -> float:
            # Lightning steps epoch schedulers once per epoch; use 1-indexed step count.
            step_idx = int(epoch) + 1
            if warmup_epochs > 0 and step_idx <= warmup_epochs:
                return 0.1 + 0.9 * (step_idx / float(max(1, warmup_epochs)))

            decay_steps = max(1, max_epochs - warmup_epochs)
            decay_idx = min(max(step_idx - warmup_epochs, 0), decay_steps)
            t = decay_idx / float(decay_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * t))
            return min_ratio + (1.0 - min_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _get_or_create_fid_metric(self, feature: int):
        feat = int(feature)
        if self._fid_metric is None or self._fid_metric_feature != feat:
            self._fid_metric = FrechetInceptionDistance(
                feature=feat,
                sync_on_compute=False,
            ).to(self.device)
            self._fid_metric_feature = feat
        else:
            self._fid_metric = self._fid_metric.to(self.device)
        self._fid_metric.reset()
        return self._fid_metric

    def training_step(self, batch, batch_idx):
        (tok_flat,) = batch
        tok_flat = tok_flat.long()
        B = tok_flat.size(0)
        bos = self.transformer.bos_token_id

        seq = torch.cat([torch.full((B, 1), bos, device=tok_flat.device, dtype=torch.long), tok_flat], dim=1)
        x_in = seq[:, :-1]
        y = seq[:, 1:]

        logits = self.transformer(x_in)
        loss = F.cross_entropy(
            logits.reshape(-1, self.transformer.cfg.vocab_size),
            y.reshape(-1),
            ignore_index=self.pad_token_id,
        )
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=B)
        return loss

    def on_fit_start(self):
        self.ae_for_decode.to(self.device)
        self.ae_for_decode.eval()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.sample_every_steps <= 0:
            return
        if self.global_step <= 0 or (self.global_step % self.sample_every_steps) != 0:
            return
        world_size = int(getattr(self.trainer, "world_size", 1))
        if world_size > 1 and getattr(self.trainer, "strategy", None) and hasattr(self.trainer.strategy, "barrier"):
            # Keep ranks synchronized while rank0 generates/saves.
            self.trainer.strategy.barrier()
            if self.trainer.is_global_zero:
                self._sample_and_save(step=self.global_step)
            self.trainer.strategy.barrier()
            return
        if self.trainer.is_global_zero:
            self._sample_and_save(step=self.global_step)

    def on_train_epoch_end(self):
        world_size = int(getattr(self.trainer, "world_size", 1))
        if world_size > 1 and getattr(self.trainer, "strategy", None) and hasattr(self.trainer.strategy, "barrier"):
            self.trainer.strategy.barrier()

        if self.trainer.is_global_zero:
            os.makedirs(self.out_dir, exist_ok=True)
            torch.save(self.transformer.state_dict(), os.path.join(self.out_dir, "transformer_last.pt"))

        if world_size > 1 and getattr(self.trainer, "strategy", None) and hasattr(self.trainer.strategy, "barrier"):
            self.trainer.strategy.barrier()

    @torch.no_grad()
    def _sample_and_save(self, step: int):
        self.transformer.eval()
        self.ae_for_decode.eval()
        print(f"[Stage2] sampling at step {step} (batch_size={self.sample_batch_size})...")
        flat_gen = self.transformer.generate(
            batch_size=self.sample_batch_size,
            temperature=1.0,
            top_k=256,
            show_progress=True,
            progress_desc=f"[Stage2] sample step {step}",
        )
        tokens_gen = flat_gen.view(-1, self.H, self.W, self.D)
        imgs = self.ae_for_decode.decode_from_tokens(tokens_gen.to(self.device))
        logged = log_image_grid_wandb(
            self.logger,
            key="stage2/samples",
            x=imgs,
            step=step,
            caption=f"step={step}",
        )
        if not logged:
            save_image_grid(imgs, os.path.join(self.out_dir, f"stage2_step{step:06d}_samples.png"))
        self._compute_and_log_fid(step=step, initial_fake_imgs=imgs)
        print(f"[Stage2] sampling done at step {step}")
        self.transformer.train()

    @torch.no_grad()
    def _compute_and_log_fid(self, step: int, initial_fake_imgs: Optional[torch.Tensor] = None):
        if self.fid_num_samples <= 0:
            return
        epoch = int(self.current_epoch + 1)
        if (epoch % self.fid_every_n_epochs) != 0:
            return
        if FrechetInceptionDistance is None:
            if not self._fid_warned_unavailable:
                print("[Stage2] FID unavailable: torchmetrics.image.fid not installed.")
                self._fid_warned_unavailable = True
            return
        if self.fid_real_images is None or self.fid_real_images.numel() == 0:
            return
        if initial_fake_imgs is None or initial_fake_imgs.numel() == 0:
            return

        seeded_u8 = ((initial_fake_imgs.detach().cpu().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
        n = min(
            self.fid_num_samples,
            int(self.fid_real_images.shape[0]),
            int(seeded_u8.size(0)),
        )
        if n <= 0:
            return

        effective_feature = self.fid_feature
        valid_features = [64, 192, 768, 2048]
        if effective_feature not in valid_features:
            effective_feature = 64
        if n < effective_feature:
            adjusted_feature = 64
            if adjusted_feature != effective_feature and not self._fid_warned_adjusted_feature:
                print(
                    f"[Stage2] adjusting FID feature from {effective_feature} "
                    f"to {adjusted_feature} for n={n}."
                )
                self._fid_warned_adjusted_feature = True
            effective_feature = adjusted_feature

        self.transformer.eval()
        self.ae_for_decode.eval()

        real_u8 = self.fid_real_images[:n].to(self.device)
        fake_u8 = seeded_u8[:n].to(self.device)

        try:
            fid_metric = self._get_or_create_fid_metric(effective_feature)
            fid_metric.update(real_u8, real=True)
            fid_metric.update(fake_u8, real=False)
            fid_value = float(fid_metric.compute().detach().cpu().item())
            fid_metric.reset()
        except Exception as exc:
            if not self._fid_warned_compute_failed:
                print(f"[Stage2] FID compute failed at step {step}: {exc}")
                self._fid_warned_compute_failed = True
            fid_value = -1.0
        logged = log_scalar_wandb(
            self.logger,
            key="stage2/fid",
            value=fid_value,
            step=step,
        )
        if not logged:
            self.log("stage2/fid", fid_value, on_step=True, on_epoch=False, prog_bar=True, sync_dist=False, rank_zero_only=True)
        self.transformer.train()


@torch.no_grad()
def precompute_tokens(
    ae: SparseDictAE,
    loader: DataLoader,
    device: torch.device,
    max_items: Optional[int] = None,
) -> Tuple[torch.Tensor, int, int, int]:
    """
    Encode dataset to tokens for stage-2 training.
    Returns:
      tokens_flat: [N, H*W*D] int32
      H, W, D
    """
    ae.eval()
    all_tokens = []
    seen = 0
    H = W = D = None

    for x, _ in tqdm(loader, desc="[Stage2] precompute tokens"):
        x = x.to(device)
        tokens, h, w = ae.encode_to_tokens(x)
        if H is None:
            H, W = h, w
            D = tokens.shape[-1]
        flat = tokens.view(tokens.size(0), -1).to(torch.int32).cpu()
        all_tokens.append(flat)
        seen += flat.size(0)
        if max_items is not None and seen >= max_items:
            break

    tokens_flat = torch.cat(all_tokens, dim=0)
    if max_items is not None:
        tokens_flat = tokens_flat[:max_items]
    return tokens_flat, H, W, D


@torch.no_grad()
def collect_real_images_uint8(
    loader: DataLoader,
    max_items: int,
) -> Optional[torch.Tensor]:
    """Collect real images as uint8 tensors for FID reference."""
    if max_items <= 0:
        return None

    images = []
    seen = 0
    for x, _ in tqdm(loader, desc="[Stage2] collect FID real images"):
        keep = min(x.size(0), max_items - seen)
        if keep <= 0:
            break
        real_u8 = ((x[:keep].detach().cpu().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
        images.append(real_u8)
        seen += keep
        if seen >= max_items:
            break

    if not images:
        return None
    return torch.cat(images, dim=0)


# -----------------------------
# Main
# -----------------------------

def main():
    ORIG_CVD = os.environ.get("CUDA_VISIBLE_DEVICES")
    ORIG_NVD = os.environ.get("NVIDIA_VISIBLE_DEVICES")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "celeba"])
    parser.add_argument("--data_dir", type=str, default=None, help="Root directory for dataset files.")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)

    # High-res friendly cropping controls
    parser.add_argument("--crop_size", type=int, default=None,
                        help="Final training crop size fed to the model (controls GPU memory).")
    parser.add_argument("--load_size", type=int, default=None,
                        help="Optional resize before cropping (CPU-side). Common: 256/320 for crop 128/160.")
    parser.add_argument("--crop_mode", type=str, default="rrc", choices=["rrc", "rcrop"],
                        help="rrc=RandomResizedCrop, rcrop=Resize+RandomCrop.")

    # Stage-1 (AE)
    parser.add_argument("--stage1_epochs", type=int, default=5)
    parser.add_argument("--stage1_lr", type=float, default=2e-4)
    parser.add_argument("--stage1_lr_schedule", type=str, default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--stage1_warmup_epochs", type=int, default=1)
    parser.add_argument("--stage1_min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--bottleneck_weight", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--stage1_devices", type=int, default=2, help="Number of GPUs for Lightning stage-1 training.")
    parser.add_argument("--stage1_precision", type=str, default="32-true", help="Lightning precision for stage-1.")
    parser.add_argument("--stage1_strategy", type=str, default="ddp", choices=["ddp", "auto"])

    # Model sizes
    parser.add_argument("--num_hiddens", type=int, default=128)
    parser.add_argument("--num_res_layers", type=int, default=2)
    parser.add_argument("--num_res_hiddens", type=int, default=32)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_atoms", type=int, default=128)      # dictionary size K
    parser.add_argument("--sparsity_level", type=int, default=3)   # stack depth D
    parser.add_argument("--n_bins", type=int, default=129, help="Coefficient quantization bins (higher = lower quantization error, larger vocab).")
    parser.add_argument("--coef_max", type=float, default=3.0, help="Coefficient clipping range for quantization in [-coef_max, coef_max].")
    parser.add_argument("--commitment_cost", type=float, default=0.25)

    # Stage-2 (Transformer)
    parser.add_argument("--stage2_epochs", type=int, default=10)
    parser.add_argument("--stage2_lr", type=float, default=3e-4)
    parser.add_argument("--stage2_lr_schedule", type=str, default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--stage2_warmup_epochs", type=int, default=1)
    parser.add_argument("--stage2_min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--stage2_batch_size", type=int, default=16)
    parser.add_argument("--stage2_sample_every_steps", type=int, default=200)
    parser.add_argument("--stage2_sample_batch_size", type=int, default=8)
    parser.add_argument(
        "--stage2_fid_num_samples",
        type=int,
        default=None,
        help="Number of images used for stage-2 FID (default: stage2_sample_batch_size).",
    )
    parser.add_argument("--stage2_fid_feature", type=int, default=64, help="Inception feature dims for stage-2 FID.")
    parser.add_argument("--stage2_fid_every_n_epochs", type=int, default=1, help="Log stage-2 FID every N epochs.")
    parser.add_argument("--stage2_devices", type=int, default=2, help="Number of GPUs for Lightning stage-2 training.")
    parser.add_argument("--stage2_precision", type=str, default="32-true", help="Lightning precision, e.g. 32-true or 16-mixed.")
    parser.add_argument("--stage2_strategy", type=str, default="ddp_fork", choices=["ddp", "ddp_fork", "auto"])
    parser.add_argument("--tf_d_model", type=int, default=256)
    parser.add_argument("--tf_heads", type=int, default=8)
    parser.add_argument("--tf_layers", type=int, default=6)
    parser.add_argument("--tf_ff", type=int, default=1024)
    parser.add_argument("--tf_dropout", type=float, default=0.1)
    parser.add_argument("--token_subset", type=int, default=50000, help="Use only first N tokens/images for speed (<=50000).")
    parser.add_argument("--token_num_workers", type=int, default=0, help="Workers for stage-2 token precompute loader.")
    parser.add_argument("--fid_num_samples", type=int, default=1024, help="Number of validation images for stage-1 FID.")
    parser.add_argument("--fid_feature", type=int, default=192, help="Inception feature dims for stage-1 FID.")
    parser.add_argument("--fid_compute_batch_size", type=int, default=32, help="Mini-batch size for stage-1 FID feature extraction.")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_project", type=str, default="laser-scratch")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None, help="Base run name (stage suffix added automatically).")
    parser.add_argument("--wandb_group", type=str, default=None, help="Group runs for stage1/stage2 in W&B.")
    parser.add_argument("--wandb_dir", type=str, default="./wandb")

    args = parser.parse_args()

    if args.stage2_fid_num_samples is None:
        args.stage2_fid_num_samples = int(args.stage2_sample_batch_size)

    # Defaults for crop/load sizes
    if args.crop_size is None:
        args.crop_size = 64 if args.dataset == "celeba" else 32
    if args.load_size is None:
        args.load_size = 256 if args.dataset == "celeba" else args.crop_size

    # The model only ever sees crop_size x crop_size (controls GPU memory).
    args.image_size = int(args.crop_size)

    if args.data_dir is None:
        args.data_dir = "../../data/celeba" if args.dataset == "celeba" else "./data"
    if args.out_dir is None:
        args.out_dir = f"./runs/sparse_dict_rq_{args.dataset}_crop{args.crop_size}"

    pl.seed_everything(args.seed, workers=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[Data] dataset={args.dataset} data_dir={args.data_dir}")
    print(f"[Data] crop_mode={args.crop_mode} load_size={args.load_size} crop_size={args.crop_size} (GPU-bounded)")

    stage1_dir = os.path.join(args.out_dir, "stage1")
    stage2_dir = os.path.join(args.out_dir, "stage2")
    os.makedirs(stage1_dir, exist_ok=True)
    os.makedirs(stage2_dir, exist_ok=True)
    token_cache_path = os.path.join(stage2_dir, "tokens_cache.pt")
    run_base_name = args.wandb_name or f"sparse_dict_rq_{args.dataset}_crop{args.crop_size}"
    if "LASER_WANDB_GROUP" in os.environ:
        wandb_group = os.environ["LASER_WANDB_GROUP"]
    else:
        wandb_group = args.wandb_group or f"{run_base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.environ["LASER_WANDB_GROUP"] = wandb_group

    def _build_wandb_logger(stage_tag: str):
        if args.wandb_mode == "disabled":
            return False
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank not in (None, "0"):
            return False
        try:
            return WandbLogger(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=f"{run_base_name}_{stage_tag}",
                group=wandb_group,
                save_dir=args.wandb_dir,
                offline=(args.wandb_mode == "offline"),
                log_model=False,
            )
        except Exception as exc:
            print(f"[WandB] logger init failed ({exc}); continuing without W&B.")
            return False

    def _build_ae() -> SparseDictAE:
        return SparseDictAE(
            in_channels=3,
            num_hiddens=args.num_hiddens,
            num_residual_layers=args.num_res_layers,
            num_residual_hiddens=args.num_res_hiddens,
            embedding_dim=args.embedding_dim,
            num_embeddings=args.num_atoms,
            sparsity_level=args.sparsity_level,
            commitment_cost=args.commitment_cost,
            n_bins=args.n_bins,
            coef_max=args.coef_max,
            out_tanh=True,
        )

    def _load_best_ae_weights(ae_model: SparseDictAE):
        best_path = os.path.join(stage1_dir, "ae_best.pt")
        if os.path.exists(best_path):
            try:
                state_dict = torch.load(best_path, map_location="cpu", weights_only=True)
            except TypeError:
                state_dict = torch.load(best_path, map_location="cpu")
            ae_model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"Stage-1 checkpoint not found at {best_path}")

    def _run_stage2_lightning(
        tokens_flat: torch.Tensor,
        H: int,
        W: int,
        D: int,
        ae_model: SparseDictAE,
        fid_real_images: Optional[torch.Tensor],
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("Stage-2 Lightning multi-GPU training requires CUDA.")
        if torch.cuda.device_count() < args.stage2_devices:
            raise RuntimeError(
                f"Requested {args.stage2_devices} GPUs, but only {torch.cuda.device_count()} detected."
            )

        stage2_dm = Stage2TokenDataModule(
            tokens_flat=tokens_flat,
            batch_size=args.stage2_batch_size,
            num_workers=2,
        )

        cfg = RQTransformerConfig(
            vocab_size=ae_model.bottleneck.vocab_size,
            H=H,
            W=W,
            D=D,
            d_model=args.tf_d_model,
            n_heads=args.tf_heads,
            n_layers=args.tf_layers,
            d_ff=args.tf_ff,
            dropout=args.tf_dropout,
        )
        transformer = RQTransformerPrior(
            cfg,
            bos_token_id=ae_model.bottleneck.bos_token_id,
            pad_token_id=ae_model.bottleneck.pad_token_id,
        )
        stage2_module = Stage2LightningModule(
            transformer=transformer,
            lr=args.stage2_lr,
            pad_token_id=ae_model.bottleneck.pad_token_id,
            out_dir=stage2_dir,
            ae_for_decode=ae_model,
            H=H,
            W=W,
            D=D,
            sample_every_steps=args.stage2_sample_every_steps,
            sample_batch_size=args.stage2_sample_batch_size,
            fid_real_images=fid_real_images,
            fid_num_samples=args.stage2_fid_num_samples,
            fid_feature=args.stage2_fid_feature,
            fid_every_n_epochs=args.stage2_fid_every_n_epochs,
            lr_schedule=args.stage2_lr_schedule,
            warmup_epochs=args.stage2_warmup_epochs,
            min_lr_ratio=args.stage2_min_lr_ratio,
        )

        effective_strategy = (args.stage2_strategy if args.stage2_devices > 1 else "auto")
        if effective_strategy == "ddp_fork" and torch.cuda.is_initialized():
            print("[Stage2] CUDA already initialized; falling back from ddp_fork to ddp.")
            effective_strategy = "ddp"

        trainer = pl.Trainer(
            accelerator="gpu",
            devices=args.stage2_devices,
            strategy=effective_strategy,
            max_epochs=args.stage2_epochs,
            logger=_build_wandb_logger("stage2"),
            enable_checkpointing=False,
            gradient_clip_val=1.0,
            precision=args.stage2_precision,
            log_every_n_steps=10,
            deterministic=False,
        )
        trainer.fit(stage2_module, datamodule=stage2_dm)

    # During stage-2 DDP script re-entry, skip stage-1 and tokenization work.
    if os.environ.get("LASER_DDP_PHASE") == "stage2":
        if not os.path.exists(token_cache_path):
            raise FileNotFoundError(f"Missing token cache: {token_cache_path}")
        try:
            cache = torch.load(token_cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            cache = torch.load(token_cache_path, map_location="cpu")
        tokens_flat = cache["tokens_flat"]
        H, W, D = cache["shape"]
        fid_real_images = cache.get("fid_real_images")
        ae = _build_ae()
        _load_best_ae_weights(ae)
        _run_stage2_lightning(
            tokens_flat=tokens_flat,
            H=H,
            W=W,
            D=D,
            ae_model=ae,
            fid_real_images=fid_real_images,
        )
        return

    # -----------------------------
    # Transforms: CPU-side crop to bounded size BEFORE GPU
    # -----------------------------
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    if args.crop_mode == "rrc":
        # Good default: bounded output size, with scale/aspect jitter
        tfm = transforms.Compose([
            transforms.RandomResizedCrop(
                size=args.crop_size,
                scale=(0.5, 1.0),
                ratio=(0.75, 1.33),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        # Literal: resize then random crop
        tfm = transforms.Compose([
            transforms.Resize(
                size=args.load_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.RandomCrop(args.crop_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize,
        ])

    if args.dataset == "cifar10":
        train_set = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=tfm)
        val_set = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=tfm)
    elif args.dataset == "celeba":
        full_set = FlatImageDataset(root=args.data_dir, transform=tfm)
        if len(full_set) < 2:
            raise RuntimeError("CelebA dataset needs at least 2 images for train/val split.")
        val_size = max(1, int(0.05 * len(full_set)))
        train_size = len(full_set) - val_size
        train_set, val_set = random_split(
            full_set,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    ae = _build_ae()
    if args.stage1_epochs > 0:
        if not torch.cuda.is_available():
            raise RuntimeError("Stage-1 Lightning multi-GPU training requires CUDA.")
        if torch.cuda.device_count() < args.stage1_devices:
            raise RuntimeError(
                f"Requested {args.stage1_devices} GPUs for stage-1, but only {torch.cuda.device_count()} detected."
            )
        stage1_module = Stage1LightningModule(
            ae=ae,
            lr=args.stage1_lr,
            bottleneck_weight=args.bottleneck_weight,
            out_dir=stage1_dir,
            fid_num_samples=args.fid_num_samples,
            fid_feature=args.fid_feature,
            fid_compute_batch_size=args.fid_compute_batch_size,
            lr_schedule=args.stage1_lr_schedule,
            warmup_epochs=args.stage1_warmup_epochs,
            min_lr_ratio=args.stage1_min_lr_ratio,
        )
        stage1_trainer = pl.Trainer(
            accelerator="gpu",
            devices=args.stage1_devices,
            strategy=(args.stage1_strategy if args.stage1_devices > 1 else "auto"),
            max_epochs=args.stage1_epochs,
            logger=_build_wandb_logger("stage1"),
            enable_checkpointing=False,
            gradient_clip_val=args.grad_clip,
            precision=args.stage1_precision,
            log_every_n_steps=10,
            deterministic=False,
        )
        stage1_trainer.fit(stage1_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank not in (None, "0"):
            # In stage-1 DDP non-zero ranks, stop here so only rank 0
            # proceeds to tokenization and stage-2 launch.
            return

    _load_best_ae_weights(ae)
    encode_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ae = ae.to(encode_device)
    token_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.token_num_workers,
        pin_memory=True,
        persistent_workers=(args.token_num_workers > 0),
    )
    tokens_flat, H, W, D = precompute_tokens(ae, token_loader, encode_device, max_items=min(args.token_subset, len(train_set)))
    fid_real_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    fid_real_images = collect_real_images_uint8(
        fid_real_loader,
        max_items=args.stage2_fid_num_samples,
    )
    ae = ae.cpu()
    print(f"[Stage2] token dataset: {tokens_flat.shape}   (H={H}, W={W}, D={D})")

    torch.save(
        {
            "tokens_flat": tokens_flat,
            "shape": (H, W, D),
            "fid_real_images": fid_real_images,
        },
        token_cache_path,
    )
    # Re-exec into a clean process before launching stage-2 DDP.
    os.environ["LASER_DDP_PHASE"] = "stage2"
    for k in (
        "LOCAL_RANK",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "NODE_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        os.environ.pop(k, None)

    print("[Stage2] restarting process for clean DDP launch...")

    # Restore GPU visibility env vars (stage1 DDP may have altered them)
    if ORIG_CVD is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ORIG_CVD
    if ORIG_NVD is not None:
        os.environ["NVIDIA_VISIBLE_DEVICES"] = ORIG_NVD

    os.execv(sys.executable, [sys.executable, __file__, *sys.argv[1:]])

    print("Done.")
    print(f"Outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
>>>>>>> parent of 4eca305 (wip)
