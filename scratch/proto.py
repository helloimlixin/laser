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

    def forward(self, x: torch.Tensor):
        z = self.pre(self.encoder(x))
        z_q, loss, support, coeffs = self.bottleneck(z)
        recon = torch.tanh(self.decoder(self.post(z_q)))
        return recon, loss, support, coeffs

    @torch.no_grad()
    def encode(self, x: torch.Tensor):
        """Returns (support [B,H,W,K], coeffs [B,H,W,K])."""
        z = self.pre(self.encoder(x))
        _, _, support, coeffs = self.bottleneck(z)
        return support, coeffs

    @torch.no_grad()
    def decode_from_codes(
        self, support: torch.Tensor, coeffs: torch.Tensor,
    ) -> torch.Tensor:
        z_q = self.bottleneck.decode_sparse_codes(support, coeffs)
        return torch.tanh(self.decoder(self.post(z_q)))


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


class RQTransformerPrior(nn.Module):
    """Two-stage transformer prior with continuous coefficient modeling."""

    def __init__(self, cfg: RQTransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.positions_per_image = cfg.H * cfg.W

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.row_emb = nn.Embedding(cfg.H, cfg.d_model)
        self.col_emb = nn.Embedding(cfg.W, cfg.d_model)
        self.depth_emb = nn.Embedding(cfg.D, cfg.d_model)
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.row_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.col_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.depth_emb.weight, mean=0.0, std=0.02)
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
        coeff_in_dim = (2 + 2 * cfg.D) * cfg.d_model
        self.coeff_query_head = nn.Sequential(
            nn.Linear(coeff_in_dim, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        nn.init.zeros_(self.coeff_query_head[-1].weight)
        nn.init.zeros_(self.coeff_query_head[-1].bias)

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

    def _coeff_pair_to_embed(
        self, atom_emb: torch.Tensor, coeff_val: torch.Tensor,
    ) -> torch.Tensor:
        """Encode (atom id embedding, scalar coeff) as coeff feedback feature."""
        return torch.tanh(atom_emb * coeff_val.unsqueeze(-1))

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
            direct_coeff_pred: [B, H*W, D]
        """
        device = tokens_flat.device
        D = self.cfg.D
        tok = tokens_flat.long().view(-1, self.positions_per_image, D)
        B, T, D = tok.shape
        d_model = self.cfg.d_model
        coeffs = coeffs_flat.float().view(B, T, D)

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
        coeff_teacher_depth = self._coeff_pair_to_embed(tok_emb, coeff_bt)
        depth_pos = self.depth_emb(
            torch.arange(D, device=device, dtype=torch.long),
        ).unsqueeze(0)

        depth_in = torch.zeros(bt, D, d_model, device=device)
        depth_in[:, 0, :] = h_t
        if D > 1:
            depth_in[:, 1:, :] = tok_emb[:, :-1, :] + coeff_teacher_depth[:, :-1, :]
        depth_in = depth_in + depth_pos

        depth_h = self.drop(depth_in)
        depth_h = self._run_no_cache_blocks(depth_h, self.depth_blocks)
        depth_h = self.depth_ln(depth_h)
        atom_logits_bt = self.atom_head(depth_h)

        # Detached GT atoms for coeff head: coeff gradients don't corrupt the
        # atom embedding table, keeping both heads training independently.
        tok_emb_detached = tok_emb.detach()

        token_slots = torch.zeros(bt, D, d_model, device=device)
        coeff_slots = torch.zeros(bt, D, d_model, device=device)
        direct_coeff_list: list[torch.Tensor] = []

        for d in range(D):
            z_last = depth_h[:, d, :]
            tok_cur = tok_emb_detached[:, d, :]
            token_slots[:, d, :] = tok_cur
            combined_h = torch.cat([
                h_t,
                z_last,
                token_slots.reshape(bt, D * d_model),
                coeff_slots.reshape(bt, D * d_model),
            ], dim=-1)
            coeff_query_d = self.coeff_query_head(combined_h)
            direct_coeff_d = (coeff_query_d * tok_cur).sum(dim=-1)
            direct_coeff_list.append(direct_coeff_d)
            coeff_slots[:, d, :] = self._coeff_pair_to_embed(
                tok_cur, coeff_bt[:, d],
            )

        atom_logits = atom_logits_bt.view(B, T, D, self.cfg.vocab_size)
        direct_coeff_pred = torch.stack(direct_coeff_list, dim=1).view(B, T, D)
        return atom_logits, direct_coeff_pred

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
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
            for d in range(D):
                if d == 0:
                    depth_x = h_t
                else:
                    depth_x = token_slots[:, d - 1, :] + coeff_slots[:, d - 1, :]
                depth_h = (depth_x + depth_pos[d]).unsqueeze(1)
                for i, blk in enumerate(self.depth_blocks):
                    depth_h, depth_kv[i] = blk(depth_h, kv_cache=depth_kv[i])
                z_last = self.depth_ln(depth_h)[:, 0, :]

                logits = self.atom_head(z_last) / max(temperature, 1e-8)
                if top_k is not None and top_k > 0:
                    v, ix = torch.topk(
                        logits, min(top_k, logits.size(-1)), dim=-1,
                    )
                    mask = torch.full_like(logits, float("-inf"))
                    mask.scatter_(1, ix, v)
                    logits = mask
                tokens[:, t, d] = torch.multinomial(
                    F.softmax(logits, dim=-1), 1,
                ).squeeze(-1)
                tok_cur = self.token_emb(tokens[:, t, d])
                token_slots[:, d, :] = tok_cur
                combined_h = torch.cat([
                    h_t,
                    z_last,
                    token_slots.reshape(batch_size, D * d_model),
                    coeff_slots.reshape(batch_size, D * d_model),
                ], dim=-1)
                coeff_query = self.coeff_query_head(combined_h)
                pred_coeff = (coeff_query * tok_cur).sum(dim=-1)
                pred_coeff = pred_coeff.clamp(-24.0, 24.0)
                coeffs[:, t, d] = pred_coeff
                coeff_slots[:, d, :] = self._coeff_pair_to_embed(tok_cur, pred_coeff)
            prev_coeff_slots = coeff_slots

        return (
            tokens.view(batch_size, T * D),
            coeffs.view(batch_size, T * D),
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
        sample_top_k: int = 64,
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
        self.sample_top_k = sample_top_k if sample_top_k > 0 else None
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

    def training_step(self, batch, batch_idx):
        tok_flat, coeff_flat, class_ids = self._unpack_stage2_batch(batch)
        B = tok_flat.size(0)
        y = tok_flat.view(B, self.H * self.W, self.D)
        coeff_target = coeff_flat.view(B, self.H * self.W, self.D)
        coeff_target = coeff_target.clamp(-self._COEFF_CLAMP, self._COEFF_CLAMP)

        atom_logits, direct_coeff_pred = self.transformer.forward_tokens(
            tok_flat, coeff_flat, class_ids=class_ids,
        )
        atom_loss = F.cross_entropy(
            atom_logits.reshape(-1, self.transformer.cfg.vocab_size),
            y.reshape(-1),
        )
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
            top_k=self.sample_top_k,
            class_ids=class_ids,
            show_progress=True,
        )
        support = tokens.view(-1, self.H, self.W, self.D)
        coeffs = coeffs.view(-1, self.H, self.W, self.D)
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

        if batch is not None:
            try:
                tf_tok, tf_coeff, tf_class_ids = self._unpack_stage2_batch(
                    batch,
                    limit=self.sample_batch_size,
                    move_to_device=True,
                )
                if tf_tok.numel() > 0:
                    bsz = tf_tok.size(0)
                    tf_support = tf_tok.view(bsz, self.H, self.W, self.D)
                    tf_coeff_target = tf_coeff.view(bsz, self.H, self.W, self.D)
                    _, tf_direct_coeff = self.transformer.forward_tokens(
                        tf_tok, tf_coeff, class_ids=tf_class_ids,
                    )
                    tf_coeff_pred = tf_direct_coeff.view(bsz, self.H, self.W, self.D)
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
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], int, int, int]:
    """Encode dataset into atom-id / coefficient sequences.

    Returns:
        tokens_flat:    [N, H*W*K] int32
        coeffs_flat:    [N, H*W*K] float32
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
            support.view(support.size(0), -1).to(torch.int32).cpu()
        )
        all_coeffs.append(
            coeffs.view(coeffs.size(0), -1).to(torch.float32).cpu()
        )
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
        ("--stage2_sample_top_k", int, 64),
        ("--stage2_sample_image_size", int, 256),
    ))
    add(
        "--stage2_lr_schedule", type=str, default="cosine",
        choices=["constant", "cosine"],
    )
    add(
        "--stage2_arch",
        type=str,
        default="rq_hier",
        choices=["rq_hier", "decoder_dual"],
        help=(
            "Stage-2 prior architecture: "
            "'decoder_dual' (simple decoder-only dual-head prior) or "
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
        help="Weight for direct coefficient MSE loss on sampled head output.",
    )
    _add_typed_args(add, (
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
    token_cache_path = os.path.join(stage2_dir, "tokens_cache.pt")

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
            )
            ae.cpu()
            torch.save({
                "tokens": tokens_flat,
                "coeffs": coeffs_flat,
                "class_ids": class_ids_flat,
                "shape": (H, W, D),
                "stage2_num_classes": stage2_num_classes,
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
            )
            transformer = DualHeadDecoderPrior(cfg)
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
            sample_top_k=args.stage2_sample_top_k,
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
        ae = _build_ae()
        _load_ae(ae, os.path.join(stage1_dir, "ae_best.pt"))
        H, W, D = cache["shape"]
        _run_stage2(
            cache["tokens"], cache["coeffs"], cache.get("class_ids"),
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
