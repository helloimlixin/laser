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

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except Exception:
    FrechetInceptionDistance = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "models"))
from lpips import LPIPS



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
        # Fuse per-depth token + coefficient context before spatial AR.
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

    @staticmethod
    def _apply_coeff_clamp(
        x: torch.Tensor, coeff_clamp: Optional[Tuple[float, float]],
    ) -> torch.Tensor:
        if coeff_clamp is None:
            return x
        lo, hi = coeff_clamp
        lo_t = x.new_tensor(lo)
        hi_t = x.new_tensor(hi)
        mid = 0.5 * (lo_t + hi_t)
        half = torch.clamp(0.5 * (hi_t - lo_t), min=1e-6)
        return mid + half * torch.tanh((x - mid) / half)

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
        compute_sparse_reg: bool = False,
        coeff_neg_samples: int = 0,
        coeff_clamp: Optional[Tuple[float, float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Teacher-forced logits over flattened [H*W*D] tokens.

        Returns:
            atom_logits:       [B, H*W, D, vocab_size]
            direct_coeff_pred: [B, H*W, D]
            sparse_reg:        [B, H*W, D] or None
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

        # Teacher-forced depth AR can run in one causal pass:
        # depth position d predicts atom/coeff for sparse layer d.
        depth_in = torch.zeros(bt, D, d_model, device=device)
        depth_in[:, 0, :] = h_t
        if D > 1:
            depth_in[:, 1:, :] = tok_emb[:, :-1, :] + coeff_teacher_depth[:, :-1, :]
        depth_in = depth_in + depth_pos

        depth_h = self.drop(depth_in)
        depth_h = self._run_no_cache_blocks(depth_h, self.depth_blocks)
        depth_h = self.depth_ln(depth_h)
        atom_logits_bt = self.atom_head(depth_h)

        token_slots = torch.zeros(bt, D, d_model, device=device)
        coeff_slots = torch.zeros(bt, D, d_model, device=device)
        direct_coeff_list: list[torch.Tensor] = []
        sparse_reg_list: list[torch.Tensor] = []
        neg_k = int(max(0, coeff_neg_samples))
        use_sparse_reg = bool(compute_sparse_reg and neg_k > 0 and self.num_atoms > 1)

        for d in range(D):
            z_last = depth_h[:, d, :]
            tok_cur = tok_emb[:, d, :]
            token_slots[:, d, :] = tok_cur
            combined_h = torch.cat([
                h_t,
                z_last,
                token_slots.reshape(bt, D * d_model),
                coeff_slots.reshape(bt, D * d_model),
            ], dim=-1)
            coeff_query_d = self.coeff_query_head(combined_h)
            direct_coeff_d = (coeff_query_d * tok_cur).sum(dim=-1)
            direct_coeff_d = self._apply_coeff_clamp(direct_coeff_d, coeff_clamp)
            direct_coeff_list.append(direct_coeff_d)
            if use_sparse_reg:
                k = min(neg_k, self.num_atoms - 1)
                neg_idx = torch.randint(
                    0, self.num_atoms, (bt, k), device=device, dtype=torch.long,
                )
                pos_idx = tok_bt[:, d: d + 1]
                collision = neg_idx.eq(pos_idx)
                if collision.any():
                    neg_idx = torch.where(collision, (neg_idx + 1) % self.num_atoms, neg_idx)
                neg_emb = self.token_emb(neg_idx)
                neg_coeff = torch.einsum("bd,bkd->bk", coeff_query_d, neg_emb)
                neg_coeff = self._apply_coeff_clamp(neg_coeff, coeff_clamp)
                sparse_reg_list.append(neg_coeff.pow(2).mean(dim=-1))

            coeff_slots[:, d, :] = coeff_teacher_depth[:, d, :]

        atom_logits = atom_logits_bt.view(B, T, D, self.cfg.vocab_size)
        direct_coeff_pred = torch.stack(direct_coeff_list, dim=1).view(B, T, D)
        sparse_reg: Optional[torch.Tensor] = None
        if use_sparse_reg:
            sparse_reg = torch.stack(sparse_reg_list, dim=1).view(B, T, D)
        return (
            atom_logits,
            direct_coeff_pred,
            sparse_reg,
        )

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        class_ids: Optional[torch.Tensor] = None,
        show_progress: bool = False,
        coeff_clamp: Optional[Tuple[float, float]] = None,
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
                pred_coeff = self._apply_coeff_clamp(pred_coeff, coeff_clamp)
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


@torch.no_grad()
def collect_real_images_uint8(
    loader: DataLoader, max_items: int,
) -> Optional[torch.Tensor]:
    """Collect real images as uint8 tensors for FID reference."""
    if max_items <= 0:
        return None
    images: list[torch.Tensor] = []
    seen = 0
    for x, _ in tqdm(loader, desc="collect FID real images"):
        keep = min(x.size(0), max_items - seen)
        if keep <= 0:
            break
        u8 = ((x[:keep].detach().cpu().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
        images.append(u8)
        seen += keep
        if seen >= max_items:
            break
    return torch.cat(images) if images else None


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
        lpips_weight: float = 0.0,
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
        self.lpips_weight = lpips_weight
        self._lpips = None

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

    def _get_lpips(self):
        if self._lpips is None and self.lpips_weight > 0:
            self._lpips = LPIPS().to(self.device)
            self._lpips.eval()
        return self._lpips

    def training_step(self, batch, batch_idx):
        x, _ = batch
        recon, b_loss, _, _ = self.ae(x)
        recon_loss = F.mse_loss(recon, x)
        loss = recon_loss + self.bottleneck_weight * b_loss
        bs = x.size(0)

        if self.lpips_weight > 0:
            lpips_fn = self._get_lpips()
            lpips_val = lpips_fn(x, recon).mean()
            loss = loss + self.lpips_weight * lpips_val
            self.log("stage1/lpips", lpips_val,
                     on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)

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

        if self.lpips_weight > 0:
            lpips_fn = self._get_lpips()
            lpips_val = lpips_fn(x, recon).mean()
            self.log("stage1/val_lpips", lpips_val,
                     on_epoch=True, sync_dist=True, batch_size=bs)

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
        transformer: RQTransformerPrior,
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
        coeff_loss_weight: float = 1.0,
        coeff_neg_samples: int = 32,
        direct_coeff_loss_weight: float = 0.25,
        recon_loss_weight: float = 1.0,
        lr_schedule: str = "cosine",
        warmup_epochs: int = 1,
        min_lr_ratio: float = 0.1,
        sample_class_id: int = -1,
        sample_image_size: Optional[int] = None,
        fid_real_images: Optional[torch.Tensor] = None,
        fid_num_samples: int = 64,
        fid_feature: int = 64,
        coeff_clamp: Optional[Tuple[float, float]] = None,
        atom_focus_ratio: float = 0.2,
        coeff_ramp_ratio: float = 0.2,
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
        self.coeff_loss_weight = coeff_loss_weight
        self.coeff_neg_samples = max(0, int(coeff_neg_samples))
        self.direct_coeff_loss_weight = direct_coeff_loss_weight
        self.recon_loss_weight = recon_loss_weight
        self.lr_schedule = lr_schedule
        self.warmup_epochs = warmup_epochs
        self.min_lr_ratio = min_lr_ratio
        self.num_classes = int(getattr(transformer.cfg, "num_classes", 0))
        self.sample_class_id = None if sample_class_id < 0 else sample_class_id
        self.sample_image_size = sample_image_size
        self.fid_real_images = fid_real_images
        self.fid_num_samples = max(0, fid_num_samples)
        self.fid_feature = fid_feature
        self.coeff_clamp = coeff_clamp
        self._fid_metric = None
        self._last_sample_step = -1
        self.atom_focus_ratio = float(max(0.0, atom_focus_ratio))
        self.coeff_ramp_ratio = float(max(0.0, coeff_ramp_ratio))

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

    def _coefficulum_scale(self) -> float:
        max_ep = int(getattr(self.trainer, "max_epochs", 0) or 0)
        if max_ep <= 1:
            return 1.0

        focus_epochs = max(1, int(round(max_ep * self.atom_focus_ratio)))
        ramp_epochs = max(1, int(round(max_ep * self.coeff_ramp_ratio)))
        epoch = int(self.current_epoch)

        if epoch < focus_epochs:
            return 0.0
        if epoch >= (focus_epochs + ramp_epochs):
            return 1.0

        return float(epoch - focus_epochs + 1) / float(ramp_epochs)

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

    def training_step(self, batch, batch_idx):
        tok_flat, coeff_flat, class_ids = self._unpack_stage2_batch(batch)
        B = tok_flat.size(0)
        y = tok_flat.view(B, self.H * self.W, self.D)
        coeff_target = coeff_flat.view(B, self.H * self.W, self.D)

        coeff_scale = self._coefficulum_scale()
        coeff_weight = self.coeff_loss_weight * coeff_scale
        direct_coeff_weight = self.direct_coeff_loss_weight * coeff_scale
        recon_weight = self.recon_loss_weight * coeff_scale

        atom_logits, direct_coeff_pred, sparse_reg = (
            self.transformer.forward_tokens(
                tok_flat,
                coeff_flat,
                class_ids=class_ids,
                compute_sparse_reg=(coeff_weight > 0),
                coeff_neg_samples=self.coeff_neg_samples,
                coeff_clamp=self.coeff_clamp,
            )
        )
        atom_hard_loss = F.cross_entropy(
            atom_logits.reshape(-1, self.transformer.cfg.vocab_size),
            y.reshape(-1),
        )

        D = self.D
        if sparse_reg is None:
            coeff_reg_loss = coeff_target.new_zeros(())
        else:
            coeff_reg_loss = sparse_reg.mean()
        direct_coeff_loss = F.mse_loss(direct_coeff_pred, coeff_target)

        if recon_weight > 0:
            bn = self.ae.bottleneck
            dictionary = F.normalize(bn.dictionary, p=2, dim=0, eps=bn.epsilon)

            z_pred = bn._unpatchify(
                bn._reconstruct(
                    y.reshape(-1, D), direct_coeff_pred.reshape(-1, D), dictionary,
                ), B, self.H, self.W,
            )
            feat_pred = self.ae.decoder(self.ae.post(z_pred))

            with torch.no_grad():
                z_gt = bn._unpatchify(
                    bn._reconstruct(
                        y.reshape(-1, D), coeff_target.reshape(-1, D), dictionary,
                    ), B, self.H, self.W,
                )
                feat_gt = self.ae.decoder(self.ae.post(z_gt))
            recon_loss = F.mse_loss(feat_pred, feat_gt)
        else:
            recon_loss = coeff_target.new_zeros(())

        loss = (
            atom_hard_loss
            + coeff_weight * coeff_reg_loss
            + direct_coeff_weight * direct_coeff_loss
            + recon_weight * recon_loss
        )

        self.log("train/atom_loss", atom_hard_loss,
                 prog_bar=True, on_step=True, on_epoch=True,
                 sync_dist=True, batch_size=B)
        self.log("train/coeff_reg_loss", coeff_reg_loss,
                 prog_bar=True, on_step=True, on_epoch=True,
                 sync_dist=True, batch_size=B)
        self.log("train/direct_coeff_loss", direct_coeff_loss,
                 prog_bar=True, on_step=True, on_epoch=True,
                 sync_dist=True, batch_size=B)
        self.log("train/recon_loss", recon_loss,
                 prog_bar=True, on_step=True, on_epoch=True,
                 sync_dist=True, batch_size=B)
        self.log("train/loss", loss,
                 prog_bar=True, on_step=True, on_epoch=True,
                 sync_dist=True, batch_size=B)
        self.log("train/coefficulum_scale", coeff_scale,
             prog_bar=True, on_step=True, on_epoch=True,
             sync_dist=True, batch_size=B)
        self.log("train/coeff_w_eff", coeff_weight,
             on_step=True, on_epoch=True,
             sync_dist=True, batch_size=B)
        self.log("train/direct_coeff_w_eff", direct_coeff_weight,
             on_step=True, on_epoch=True,
             sync_dist=True, batch_size=B)
        self.log("train/recon_w_eff", recon_weight,
             on_step=True, on_epoch=True,
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
            coeff_clamp=self.coeff_clamp,
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
                    _, tf_direct_coeff, _ = self.transformer.forward_tokens(
                        tf_tok,
                        tf_coeff,
                        class_ids=tf_class_ids,
                        compute_sparse_reg=False,
                        coeff_clamp=self.coeff_clamp,
                    )
                    tf_coeff_pred = tf_direct_coeff.view(
                        bsz, self.H, self.W, self.D,
                    )
                    tf_gt_raw = self.ae.decode_from_codes(
                        tf_support, tf_coeff_target,
                    )
                    tf_pred_raw = self.ae.decode_from_codes(
                        tf_support, tf_coeff_pred,
                    )
                    tf_gt = self._resize_for_logging(tf_gt_raw)
                    tf_pred = self._resize_for_logging(tf_pred_raw)

                    gt_logged = log_images_wandb(
                        self.logger,
                        "stage2/teacher_forced_gt",
                        tf_gt,
                        step,
                        f"step={step}",
                    )
                    pred_logged = log_images_wandb(
                        self.logger,
                        "stage2/teacher_forced_pred",
                        tf_pred,
                        step,
                        f"step={step}",
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

        self._compute_and_log_fid(step, raw_imgs)
        self.transformer.train()

    @torch.no_grad()
    def _compute_and_log_fid(
        self, step: int, fake_imgs: torch.Tensor,
    ):
        if self.fid_num_samples <= 0:
            return
        if FrechetInceptionDistance is None:
            return
        if self.fid_real_images is None or self.fid_real_images.numel() == 0:
            return

        fake_u8 = (
            (fake_imgs.detach().cpu().clamp(-1, 1) + 1.0) * 127.5
        ).to(torch.uint8)
        n = min(self.fid_num_samples, self.fid_real_images.size(0), fake_u8.size(0))
        if n < 2:
            return

        feat = self.fid_feature
        if n < feat:
            feat = 64

        if self._fid_metric is None:
            self._fid_metric = FrechetInceptionDistance(
                feature=feat, sync_on_compute=False,
            )
        fid = self._fid_metric.to(self.device)
        fid.reset()

        bs = 64
        real_u8 = self.fid_real_images[:n].to(self.device)
        fake_u8 = fake_u8[:n].to(self.device)
        for i in range(0, n, bs):
            fid.update(real_u8[i: i + bs], real=True)
            fid.update(fake_u8[i: i + bs], real=False)

        try:
            fid_val = float(fid.compute().detach().cpu().item())
            print(f"[Stage2] FID @ step {step}: {fid_val:.2f}")
            if not log_scalar_wandb(self.logger, "stage2/fid", fid_val, step):
                self.log("stage2/fid", fid_val, on_step=True, prog_bar=True,
                         sync_dist=False, rank_zero_only=True)
        except Exception as exc:
            print(f"[Stage2] FID failed at step {step}: {exc}")
        fid.reset()
        fid.cpu()


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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "celeba", "stl10", "imagenette"])
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)

    # Stage-1
    parser.add_argument("--stage1_epochs", type=int, default=5)
    parser.add_argument("--stage1_lr", type=float, default=2e-4)
    parser.add_argument("--stage1_lr_schedule", type=str, default="cosine",
                        choices=["constant", "cosine"])
    parser.add_argument("--stage1_warmup_epochs", type=int, default=1)
    parser.add_argument("--stage1_min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--bottleneck_weight", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--devices", type=int, default=2)
    parser.add_argument("--stage1_init_ckpt", type=str, default=None)
    # Model
    parser.add_argument("--num_hiddens", type=int, default=128)
    parser.add_argument("--ae_num_downsamples", type=int, default=2)
    parser.add_argument("--num_res_layers", type=int, default=2)
    parser.add_argument("--num_res_hiddens", type=int, default=32)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_atoms", type=int, default=128)
    parser.add_argument("--sparsity_level", type=int, default=3)
    parser.add_argument("--commitment_cost", type=float, default=0.25)
    parser.add_argument(
        "--usage_ema_decay", type=float, default=0.99,
        help="EMA decay for tracking dictionary atom usage.",
    )
    parser.add_argument(
        "--usage_balance_alpha", type=float, default=0.3,
        help="Inverse-usage reweighting strength for OMP atom selection.",
    )
    parser.add_argument(
        "--dead_atom_threshold", type=float, default=5e-4,
        help="Usage threshold for reviving dead atoms; <=0 disables revival.",
    )
    parser.add_argument(
        "--dead_atom_interval", type=int, default=200,
        help="Train batches between dead-atom revival checks.",
    )
    parser.add_argument(
        "--dead_atom_max_reinit", type=int, default=16,
        help="Maximum number of atoms to reinitialize per revival check.",
    )
    parser.add_argument("--lpips_weight", type=float, default=0.1,
                        help="Weight for LPIPS perceptual loss in stage-1.")
    parser.add_argument("--bottleneck_patch_size", type=int, default=1,
                        help="Patch size for dictionary learning; >1 groups "
                             "PxP spatial patches into single atoms.")

    # Stage-2
    parser.add_argument("--stage2_epochs", type=int, default=10)
    parser.add_argument("--stage2_lr", type=float, default=3e-4)
    parser.add_argument("--stage2_lr_schedule", type=str, default="cosine",
                        choices=["constant", "cosine"])
    parser.add_argument("--stage2_warmup_epochs", type=int, default=1)
    parser.add_argument("--stage2_min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--stage2_batch_size", type=int, default=16)
    parser.add_argument("--stage2_conditional", action="store_true",
                        default=False)
    parser.add_argument("--stage2_num_classes", type=int, default=None)
    parser.add_argument("--stage2_sample_class_id", type=int, default=-1)
    parser.add_argument("--stage2_sample_every_steps", type=int, default=200)
    parser.add_argument("--stage2_sample_batch_size", type=int, default=64)
    parser.add_argument("--stage2_sample_temperature", type=float, default=0.8)
    parser.add_argument("--stage2_sample_top_k", type=int, default=64)
    parser.add_argument("--stage2_sample_image_size", type=int, default=256)
    parser.add_argument("--stage2_resume_from_last", action="store_true",
                        default=True)
    parser.add_argument("--no_stage2_resume_from_last", action="store_false",
                        dest="stage2_resume_from_last")
    parser.add_argument(
        "--stage2_coeff_loss_weight",
        type=float,
        default=0.1,
        help="Weight for sampled negative-atom coefficient energy in stage-2.",
    )
    parser.add_argument(
        "--stage2_coeff_neg_samples",
        type=int,
        default=32,
        help="Number of negative atoms sampled per sparse coeff for coeff regularization.",
    )
    parser.add_argument(
        "--stage2_direct_coeff_loss_weight",
        type=float,
        default=0.25,
        help="Weight for direct coefficient MSE loss on sampled head output.",
    )
    parser.add_argument(
        "--stage2_recon_loss_weight",
        type=float,
        default=1.0,
        help="Weight for latent reconstruction loss (z_pred vs z_gt) in stage-2.",
    )
    parser.add_argument(
        "--stage2_atom_focus_ratio",
        type=float,
        default=0.2,
        help="Fraction of stage-2 epochs focused on atom ID loss only.",
    )
    parser.add_argument(
        "--stage2_coeff_ramp_ratio",
        type=float,
        default=0.2,
        help="Fraction of stage-2 epochs to ramp coeff/recon losses from 0 to full weight.",
    )
    parser.add_argument("--stage2_fid_num_samples", type=int, default=64)
    parser.add_argument("--stage2_fid_feature", type=int, default=64)
    parser.add_argument("--tf_d_model", type=int, default=256)
    parser.add_argument("--tf_heads", type=int, default=4)
    parser.add_argument("--tf_layers", type=int, default=4)
    parser.add_argument("--tf_depth_layers", type=int, default=4)
    parser.add_argument("--tf_ff", type=int, default=1024)
    parser.add_argument("--tf_dropout", type=float, default=0.1)
    parser.add_argument("--token_subset", type=int, default=0)
    parser.add_argument("--wandb_mode", type=str, default="online",
                        choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_project", type=str, default="laser-scratch")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_dir", type=str, default="./wandb")

    args = parser.parse_args()
    # ---- defaults ----
    if args.data_dir is None:
        if args.dataset == "celeba":
            args.data_dir = "../../data/celeba"
        elif args.dataset == "imagenette":
            args.data_dir = "./data/imagenette2-160"
        else:
            args.data_dir = "./data"
    if args.out_dir is None:
        args.out_dir = f"./runs/sparse_dict_{args.dataset}_{args.image_size}"

    hint = 10 if args.dataset in ("cifar10", "stl10", "imagenette") else 0
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

    def _run_stage2(
        tokens_flat: torch.Tensor,
        coeffs_flat: torch.Tensor,
        class_ids_flat: Optional[torch.Tensor],
        H: int, W: int, D: int,
        ae: SparseDictAE,
        fid_real_images: Optional[torch.Tensor] = None,
        coeff_clamp: Optional[Tuple[float, float]] = None,
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
        transformer = RQTransformerPrior(
            cfg,
        )

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
            coeff_loss_weight=args.stage2_coeff_loss_weight,
            coeff_neg_samples=args.stage2_coeff_neg_samples,
            direct_coeff_loss_weight=args.stage2_direct_coeff_loss_weight,
            recon_loss_weight=args.stage2_recon_loss_weight,
            lr_schedule=args.stage2_lr_schedule,
            warmup_epochs=args.stage2_warmup_epochs,
            min_lr_ratio=args.stage2_min_lr_ratio,
            sample_class_id=args.stage2_sample_class_id,
            sample_image_size=(
                args.stage2_sample_image_size
                if args.stage2_sample_image_size > 0 else None
            ),
            fid_real_images=fid_real_images,
            fid_num_samples=args.stage2_fid_num_samples,
            fid_feature=args.stage2_fid_feature,
            coeff_clamp=coeff_clamp,
            atom_focus_ratio=args.stage2_atom_focus_ratio,
            coeff_ramp_ratio=args.stage2_coeff_ramp_ratio,
        )

        from lightning.pytorch.strategies import DDPStrategy
        s2_strategy: object = (
            DDPStrategy(broadcast_buffers=False, find_unused_parameters=False)
            if args.devices > 1 else "auto"
        )
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
            fid_real_images=cache.get("fid_real_images"),
            coeff_clamp=cache.get("coeff_clamp"),
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

    if args.dataset == "cifar10":
        train_set = datasets.CIFAR10(
            root=args.data_dir, train=True, download=True, transform=tfm,
        )
        val_set = datasets.CIFAR10(
            root=args.data_dir, train=False, download=True, transform=tfm,
        )
    elif args.dataset == "stl10":
        train_set = datasets.STL10(
            root=args.data_dir, split="train", download=True, transform=tfm,
        )
        val_set = datasets.STL10(
            root=args.data_dir, split="test", download=True, transform=tfm,
        )
    elif args.dataset == "celeba":
        full = FlatImageDataset(root=args.data_dir, transform=tfm)
        n_val = max(1, int(0.05 * len(full)))
        idx = torch.randperm(
            len(full), generator=torch.Generator().manual_seed(args.seed),
        )
        train_set = Subset(full, idx[: len(full) - n_val].tolist())
        val_set = Subset(full, idx[len(full) - n_val:].tolist())
    elif args.dataset == "imagenette":
        train_set = datasets.ImageFolder(
            root=os.path.join(args.data_dir, "train"), transform=tfm,
        )
        val_set = datasets.ImageFolder(
            root=os.path.join(args.data_dir, "val"), transform=tfm,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

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
            lpips_weight=args.lpips_weight,
        )

        if args.devices > 1:
            os.environ["LASER_DDP_PHASE"] = "stage1"
            _ensure_free_master_port("Stage1")
        from lightning.pytorch.strategies import DDPStrategy
        s1_strategy: object = (
            DDPStrategy(broadcast_buffers=False, find_unused_parameters=False)
            if args.devices > 1 else "auto"
        )
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
        fid_real_images = cache.get("fid_real_images")
        H, W, D = cache["shape"]
        coeff_clamp = cache.get("coeff_clamp")
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
        fid_loader = DataLoader(
            train_set, batch_size=args.batch_size,
            shuffle=False, num_workers=0,
        )
        fid_real_images = collect_real_images_uint8(
            fid_loader, max_items=args.stage2_fid_num_samples,
        )
        ae = ae.cpu()
        cf = coeffs_flat.reshape(-1)
        if cf.numel() > 1_000_000:
            idx = torch.randint(0, cf.numel(), (1_000_000,))
            cf = cf[idx]
        coeff_lo = float(cf.quantile(0.01).item())
        coeff_hi = float(cf.quantile(0.99).item())
        coeff_clamp = (coeff_lo, coeff_hi)
        print(f"[Coeffs] range 1%-99%: [{coeff_lo:.3f}, {coeff_hi:.3f}]")
        torch.save({
            "tokens": tokens_flat,
            "coeffs": coeffs_flat,
            "class_ids": class_ids_flat,
            "shape": (H, W, D),
            "stage2_num_classes": stage2_num_classes,
            "fid_real_images": fid_real_images,
            "coeff_clamp": (coeff_lo, coeff_hi),
        }, token_cache_path)

    print(f"[Stage2] tokens: {tokens_flat.shape}  (H={H}, W={W}, D={D})")
    if coeff_clamp is not None:
        print(f"[Stage2] coeff_clamp: [{coeff_clamp[0]:.3f}, {coeff_clamp[1]:.3f}]")

    # ---- stage-2 ----

    if args.stage1_epochs <= 0:
        _run_stage2(
            tokens_flat, coeffs_flat, class_ids_flat, H, W, D, ae,
            fid_real_images=fid_real_images,
            coeff_clamp=coeff_clamp,
        )
        return

    # Re-exec into a clean process so CUDA / NCCL state from stage-1 DDP
    # doesn't conflict with stage-2 DDP.
    os.environ["LASER_DDP_PHASE"] = "stage2"
    for k in (
        "LOCAL_RANK", "RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
        "GROUP_RANK", "ROLE_RANK", "NODE_RANK",
        "MASTER_ADDR", "MASTER_PORT",
    ):
        os.environ.pop(k, None)
    print("[Stage2] restarting for clean DDP launch...")
    ret = subprocess.call(
        [sys.executable, __file__, *sys.argv[1:]],
        env=os.environ.copy(), close_fds=True,
    )
    if ret != 0:
        raise RuntimeError(f"Stage-2 restart failed with exit code {ret}.")


if __name__ == "__main__":
    main()
