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
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from PIL import Image
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


def _disable_lightning_cuda_matmul_capability_probe():
    try:
        import lightning.pytorch.accelerators.cuda as pl_cuda
        import lightning.fabric.accelerators.cuda as fabric_cuda
    except Exception:
        return

    def _noop(device):
        return

    pl_cuda._check_cuda_matmul_precision = _noop
    fabric_cuda._check_cuda_matmul_precision = _noop


_disable_lightning_cuda_matmul_capability_probe()


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
    """Per-pixel dictionary learning bottleneck with batch OMP sparse coding.

    Closely follows src/models/bottleneck.py but returns ordered support
    indices + coefficients so the transformer prior can tokenise them.
    """

    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        sparsity_level: int = 5,
        commitment_cost: float = 0.25,
        epsilon: float = 1e-10,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.sparsity_level = sparsity_level
        self.commitment_cost = commitment_cost
        self.epsilon = epsilon

        self.dictionary = nn.Parameter(
            torch.randn(embedding_dim, num_embeddings)
        )

        self.pad_token_id = num_embeddings
        self.bos_token_id = num_embeddings + 1
        self.vocab_size = num_embeddings + 2

    # ---- batch OMP (adapted from bottleneck.py / sparse-vqvae) ----

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

        for k in range(1, self.sparsity_level + 1):
            idx = (h.abs() * mask.float()).argmax(dim=1)
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

    # ---- reconstruction helper ----

    def _reconstruct(
        self, support: torch.Tensor, coeffs: torch.Tensor,
        dictionary: torch.Tensor,
    ) -> torch.Tensor:
        dict_t = dictionary.t()                     # [N, C]
        atoms = dict_t[support.long()]              # [..., K, C]
        return (atoms * coeffs.unsqueeze(-1)).sum(dim=-2)

    # ---- forward (matches bottleneck.py style) ----

    def forward(
        self, z_e: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_e: [B, C, H, W] encoder output.

        Returns:
            z_dl:    [B, C, H, W]  STE-quantised latent.
            loss:    scalar        bottleneck loss.
            support: [B, H, W, K]  ordered atom indices.
            coeffs:  [B, H, W, K]  ordered coefficients.
        """
        if z_e.dim() != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(z_e.shape)}")
        B, C, H, W = z_e.shape

        signals = z_e.permute(0, 2, 3, 1).contiguous().view(-1, C).t()
        dictionary = F.normalize(self.dictionary, p=2, dim=0, eps=self.epsilon)

        with torch.no_grad():
            support, coeffs = self.batch_omp(signals, dictionary)

        recon = self._reconstruct(support, coeffs, dictionary)
        z_dl = recon.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        dl_latent_loss = F.mse_loss(z_dl, z_e.detach())
        e_latent_loss = F.mse_loss(z_dl.detach(), z_e)
        loss = dl_latent_loss + self.commitment_cost * e_latent_loss

        z_dl = z_e + (z_dl - z_e).detach()

        K = self.sparsity_level
        return z_dl, loss, support.view(B, H, W, K), coeffs.view(B, H, W, K)

    @torch.no_grad()
    def decode_sparse_codes(
        self, support: torch.Tensor, coeffs: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct [B, C, H, W] from atom ids + coefficients."""
        B, H, W, K = support.shape
        dictionary = F.normalize(self.dictionary, p=2, dim=0, eps=self.epsilon)
        recon = self._reconstruct(
            support.reshape(-1, K),
            coeffs.reshape(-1, K).to(dictionary.dtype),
            dictionary,
        )
        return recon.view(B, H, W, self.embedding_dim).permute(0, 3, 1, 2).contiguous()


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
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
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
        new_kv = (k, v)

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=(kv_cache is None and T > 1),
            dropout_p=(self.dropout if self.training else 0.0),
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out)), new_kv


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
        attn_out, new_kv = self.attn(self.ln1(x), kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_kv


@dataclass
class RQTransformerConfig:
    vocab_size: int
    H: int
    W: int
    D: int
    num_classes: int = 0
    coeff_max: Optional[float] = 12.0
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1


class RQTransformerPrior(nn.Module):
    """GPT-style causal transformer over (H, W, D) token stacks.

    Always predicts atom IDs (cross-entropy) and regresses continuous
    coefficients (MSE, per-atom).
    """

    def __init__(self, cfg: RQTransformerConfig,
                 bos_token_id: int, pad_token_id: int):
        super().__init__()
        self.cfg = cfg
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id

        self.tokens_per_image = cfg.H * cfg.W * cfg.D
        self.max_len = 1 + self.tokens_per_image

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.spatial_emb = nn.Embedding(cfg.H * cfg.W, cfg.d_model)
        self.depth_emb = nn.Embedding(cfg.D, cfg.d_model)
        self.type_emb = nn.Embedding(2, cfg.d_model)
        self.class_emb = (
            nn.Embedding(cfg.num_classes, cfg.d_model)
            if cfg.num_classes > 0 else None
        )

        spatial_ids = torch.zeros(self.max_len, dtype=torch.long)
        depth_ids = torch.zeros(self.max_len, dtype=torch.long)
        type_ids = torch.zeros(self.max_len, dtype=torch.long)
        if self.max_len > 1:
            idx = torch.arange(self.max_len - 1)
            spatial_ids[1:] = idx // cfg.D
            depth_ids[1:] = idx % cfg.D
            type_ids[1:] = 1
        self.register_buffer("_spatial_ids", spatial_ids)
        self.register_buffer("_depth_ids", depth_ids)
        self.register_buffer("_type_ids", type_ids)

        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.atom_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.coeff_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, cfg.vocab_size),
        )

    def forward(
        self,
        x: torch.Tensor,
        class_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[list] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, list]:
        B, L = x.shape
        pos_end = start_pos + L

        h = (
            self.token_emb(x)
            + self.spatial_emb(self._spatial_ids[start_pos:pos_end]).unsqueeze(0)
            + self.depth_emb(self._depth_ids[start_pos:pos_end]).unsqueeze(0)
            + self.type_emb(self._type_ids[start_pos:pos_end]).unsqueeze(0)
        )
        if self.class_emb is not None:
            if class_ids is None:
                raise ValueError("class_ids required when num_classes > 0")
            h = h + self.class_emb(class_ids.long()).unsqueeze(1)
        h = self.drop(h)

        new_kv: list = []
        for i, block in enumerate(self.blocks):
            cache = kv_cache[i] if kv_cache is not None else None
            h, kv = block(h, kv_cache=cache)
            new_kv.append(kv)

        h = self.ln_f(h)
        logits = self.atom_head(h)
        coeff_pred = self.coeff_head(h)            # [B, L, vocab_size]
        return logits, coeff_pred, new_kv

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
        T = self.tokens_per_image

        seq = torch.full(
            (batch_size, 1), self.bos_token_id,
            dtype=torch.long, device=device,
        )
        special = torch.tensor(
            [self.bos_token_id, self.pad_token_id], device=device,
        )
        gen_coeffs: list[torch.Tensor] = []
        kv_cache = None
        steps = tqdm(
            range(T), desc="sampling", leave=False, disable=(not show_progress),
        )

        for _ in steps:
            if kv_cache is None:
                inp, start = seq, 0
            else:
                inp, start = seq[:, -1:], seq.size(1) - 1

            logits, coeff_pred, kv_cache = self(
                inp, class_ids=class_ids, kv_cache=kv_cache, start_pos=start,
            )
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            logits[:, special] = float("-inf")
            if top_k is not None and top_k > 0:
                v, ix = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                mask = torch.full_like(logits, float("-inf"))
                mask.scatter_(1, ix, v)
                logits = mask
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            seq = torch.cat([seq, nxt], dim=1)
            c = coeff_pred[:, -1, :].gather(-1, nxt).squeeze(-1)
            if self.cfg.coeff_max is None or float(self.cfg.coeff_max) <= 0.0:
                gen_coeffs.append(c)
            else:
                # Soft clamp avoids hard clipping edges while still bounding
                # outlier coefficients.
                s = float(self.cfg.coeff_max)
                gen_coeffs.append(s * torch.tanh(c / s))

        tokens = seq[:, 1:]
        coeffs = torch.stack(gen_coeffs, dim=1)
        return tokens, coeffs


# ---------------------------------------------------------------------------
# Spatial coefficient refinement (post-transformer)
# ---------------------------------------------------------------------------

class CoeffRefiner(nn.Module):
    """Spatial CNN that refines per-token coefficient predictions from the
    transformer by enforcing coherence across the (H, W) spatial grid.

    The transformer predicts coefficients independently per token position.
    This module takes the raw predictions, embeds the selected atom IDs for
    context, and applies spatial convolutions so that neighboring positions
    produce mutually consistent coefficients.
    """

    def __init__(self, K: int, vocab_size: int, atom_embed_dim: int = 16,
                 hidden_dim: int = 64, num_layers: int = 4):
        super().__init__()
        self.K = K
        self.atom_emb = nn.Embedding(vocab_size, atom_embed_dim)
        in_ch = K * (atom_embed_dim + 1)
        layers: list[nn.Module] = []
        ch = in_ch
        for i in range(num_layers):
            out_ch = hidden_dim if i < num_layers - 1 else K
            layers.append(nn.Conv2d(ch, out_ch, 3, padding=1))
            if i < num_layers - 1:
                layers.append(nn.GroupNorm(min(8, out_ch), out_ch))
                layers.append(nn.GELU())
            ch = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, atom_ids: torch.Tensor,
                coeffs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            atom_ids: [B, H, W, K] long — selected atom indices.
            coeffs:   [B, H, W, K] float — raw coefficient predictions.

        Returns:
            [B, H, W, K] refined coefficients (residual correction).
        """
        B, H, W, K = atom_ids.shape
        emb = self.atom_emb(atom_ids.long())               # [B, H, W, K, E]
        feat = torch.cat([emb, coeffs.unsqueeze(-1)], dim=-1)
        feat = feat.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        delta = self.net(feat)                              # [B, K, H, W]
        return coeffs + delta.permute(0, 2, 3, 1)


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
        with torch.no_grad():
            bn = self.ae.bottleneck
            bn.dictionary.copy_(
                F.normalize(bn.dictionary, p=2, dim=0, eps=bn.epsilon)
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
        lr_schedule: str = "cosine",
        warmup_epochs: int = 1,
        min_lr_ratio: float = 0.1,
        sample_class_id: int = -1,
        sample_image_size: Optional[int] = None,
        fid_real_images: Optional[torch.Tensor] = None,
        fid_num_samples: int = 64,
        fid_feature: int = 64,
        coeff_refiner: Optional["CoeffRefiner"] = None,
        refine_weight: float = 1.0,
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
        self.lr_schedule = lr_schedule
        self.warmup_epochs = warmup_epochs
        self.min_lr_ratio = min_lr_ratio
        self.num_classes = int(getattr(transformer.cfg, "num_classes", 0))
        self.sample_class_id = None if sample_class_id < 0 else sample_class_id
        self.sample_image_size = sample_image_size
        self.fid_real_images = fid_real_images
        self.fid_num_samples = max(0, fid_num_samples)
        self.fid_feature = fid_feature
        self._fid_metric = None
        self._last_sample_step = -1
        self.coeff_refiner = coeff_refiner
        self.refine_weight = refine_weight

        self.ae.eval()
        for p in self.ae.parameters():
            p.requires_grad_(False)

    def configure_optimizers(self):
        params = list(self.transformer.parameters())
        if self.coeff_refiner is not None:
            params += list(self.coeff_refiner.parameters())
        opt = torch.optim.Adam(params, lr=self.lr)
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
        if not isinstance(batch, (tuple, list)):
            batch = (batch,)
        items = list(batch)

        class_ids = None
        if self.num_classes > 0:
            class_ids = items.pop(-1).long()

        tok_flat, coeff_flat = items
        tok_flat = tok_flat.long()
        coeff_flat = coeff_flat.float()
        B = tok_flat.size(0)
        bos = self.transformer.bos_token_id
        pad = self.transformer.pad_token_id

        seq = torch.cat([
            torch.full((B, 1), bos, device=tok_flat.device, dtype=torch.long),
            tok_flat,
        ], dim=1)
        x_in = seq[:, :-1]
        y = seq[:, 1:]

        logits, coeff_pred, _ = self.transformer(x_in, class_ids=class_ids)
        atom_loss = F.cross_entropy(
            logits.reshape(-1, self.transformer.cfg.vocab_size),
            y.reshape(-1),
            ignore_index=pad,
        )
        coeff_for_target = coeff_pred.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        coeff_loss = F.mse_loss(coeff_for_target, coeff_flat)
        loss = atom_loss + self.coeff_loss_weight * coeff_loss

        if self.coeff_refiner is not None:
            H, W, K = self.H, self.W, self.D
            atom_ids_2d = y.reshape(B, H, W, K)
            raw_coeffs_2d = coeff_for_target.detach().reshape(B, H, W, K)
            gt_coeffs_2d = coeff_flat.reshape(B, H, W, K)
            refined = self.coeff_refiner(atom_ids_2d, raw_coeffs_2d)
            refine_loss = F.mse_loss(refined, gt_coeffs_2d)
            loss = loss + self.refine_weight * refine_loss
            self.log("train/refine_loss", refine_loss,
                     prog_bar=True, on_step=True, on_epoch=True,
                     sync_dist=True, batch_size=B)

        self.log("train/atom_loss", atom_loss,
                 prog_bar=True, on_step=True, on_epoch=True,
                 sync_dist=True, batch_size=B)
        self.log("train/coeff_loss", coeff_loss,
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
            self._sample_and_save()

    def on_train_epoch_end(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        if self.trainer.is_global_zero:
            os.makedirs(self.out_dir, exist_ok=True)
            torch.save(
                self.transformer.state_dict(),
                os.path.join(self.out_dir, "transformer_last.pt"),
            )
            if self.coeff_refiner is not None:
                torch.save(
                    self.coeff_refiner.state_dict(),
                    os.path.join(self.out_dir, "refiner_last.pt"),
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
    def _sample_and_save(self):
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
        if self.coeff_refiner is not None:
            self.coeff_refiner.eval()
            coeffs = self.coeff_refiner(
                support.to(self.device), coeffs.to(self.device),
            )
        raw_imgs = self.ae.decode_from_codes(
            support.to(self.device), coeffs.to(self.device),
        )
        if self.sample_image_size and self.sample_image_size > 0:
            imgs = F.interpolate(
                raw_imgs,
                size=(self.sample_image_size, self.sample_image_size),
                mode="bilinear",
                align_corners=False,
            )
        else:
            imgs = raw_imgs

        if not log_images_wandb(
            self.logger, "stage2/samples", imgs, step, f"step={step}",
        ):
            save_grid(
                imgs, os.path.join(self.out_dir, f"step{step:06d}_samples.png"),
            )

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
    parser.add_argument("--stage1_devices", type=int, default=2)
    parser.add_argument("--stage1_precision", type=str, default="32-true")
    parser.add_argument("--stage1_strategy", type=str, default="ddp",
                        choices=["ddp", "auto"])
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
    parser.add_argument("--lpips_weight", type=float, default=0.1,
                        help="Weight for LPIPS perceptual loss in stage-1.")

    # Stage-2
    parser.add_argument("--stage2_epochs", type=int, default=10)
    parser.add_argument("--stage2_lr", type=float, default=3e-4)
    parser.add_argument("--stage2_lr_schedule", type=str, default="cosine",
                        choices=["constant", "cosine"])
    parser.add_argument("--stage2_warmup_epochs", type=int, default=1)
    parser.add_argument("--stage2_min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--stage2_batch_size", type=int, default=16)
    parser.add_argument("--stage2_grad_accum", type=int, default=4)
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
    parser.add_argument("--stage2_devices", type=int, default=2)
    parser.add_argument("--stage2_precision", type=str, default="32-true")
    parser.add_argument("--stage2_strategy", type=str, default="ddp_fork",
                        choices=["ddp", "ddp_fork", "auto"])
    parser.add_argument("--stage2_coeff_loss_weight", type=float, default=2.0)
    parser.add_argument(
        "--stage2_coeff_max",
        type=float,
        default=12.0,
        help=(
            "Soft-clamp scale for sampled coefficients in stage-2 "
            "(scale * tanh(x / scale)). "
            "Set <= 0 to disable clamping."
        ),
    )
    parser.add_argument("--stage2_fid_num_samples", type=int, default=64)
    parser.add_argument("--stage2_fid_feature", type=int, default=64)

    # Coefficient refiner
    parser.add_argument(
        "--stage2_refine_weight", type=float, default=0.0,
        help=(
            "Weight for the CoeffRefiner loss.  When > 0 a small spatial CNN "
            "is trained alongside the transformer to smooth per-token "
            "coefficient predictions across the (H,W) grid, eliminating "
            "blocky artifacts at generation time.  0 = disabled."
        ),
    )
    parser.add_argument("--refiner_hidden", type=int, default=64)
    parser.add_argument("--refiner_layers", type=int, default=4)
    parser.add_argument("--refiner_embed_dim", type=int, default=16)
    parser.add_argument("--tf_d_model", type=int, default=256)
    parser.add_argument("--tf_heads", type=int, default=4)
    parser.add_argument("--tf_layers", type=int, default=4)
    parser.add_argument("--tf_ff", type=int, default=1024)
    parser.add_argument("--tf_dropout", type=float, default=0.1)
    parser.add_argument("--token_subset", type=int, default=50000)
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

    pl.seed_everything(args.seed, workers=True)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")

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
            coeff_max=(
                None
                if float(args.stage2_coeff_max) <= 0.0
                else float(args.stage2_coeff_max)
            ),
            d_model=args.tf_d_model,
            n_heads=args.tf_heads,
            n_layers=args.tf_layers,
            d_ff=args.tf_ff,
            dropout=args.tf_dropout,
        )
        transformer = RQTransformerPrior(
            cfg,
            bos_token_id=ae.bottleneck.bos_token_id,
            pad_token_id=ae.bottleneck.pad_token_id,
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

        coeff_refiner = None
        if args.stage2_refine_weight > 0:
            coeff_refiner = CoeffRefiner(
                K=D,
                vocab_size=ae.bottleneck.vocab_size,
                atom_embed_dim=args.refiner_embed_dim,
                hidden_dim=args.refiner_hidden,
                num_layers=args.refiner_layers,
            )
            if args.stage2_resume_from_last:
                ref_ckpt = os.path.join(stage2_dir, "refiner_last.pt")
                if os.path.exists(ref_ckpt):
                    coeff_refiner.load_state_dict(
                        torch.load(ref_ckpt, map_location="cpu",
                                   weights_only=True),
                    )
                    print(f"[Stage2] resumed refiner from {ref_ckpt}")

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
            coeff_refiner=coeff_refiner,
            refine_weight=args.stage2_refine_weight,
        )

        strategy: object = (
            args.stage2_strategy if args.stage2_devices > 1 else "auto"
        )
        if strategy == "ddp_fork" and torch.cuda.is_initialized():
            strategy = "ddp"
        if strategy in ("ddp", "ddp_fork"):
            from lightning.pytorch.strategies import DDPStrategy
            strategy = DDPStrategy(
                broadcast_buffers=False, find_unused_parameters=False,
            )
        if args.stage2_devices > 1:
            os.environ["LASER_DDP_PHASE"] = "stage2"
            _ensure_free_master_port("Stage2")

        trainer = pl.Trainer(
            accelerator="gpu",
            devices=args.stage2_devices,
            strategy=strategy,
            max_epochs=args.stage2_epochs,
            logger=_wandb_logger("stage2"),
            enable_checkpointing=False,
            gradient_clip_val=1.0,
            precision=args.stage2_precision,
            log_every_n_steps=10,
            accumulate_grad_batches=args.stage2_grad_accum,
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

        if args.stage1_devices > 1:
            os.environ["LASER_DDP_PHASE"] = "stage1"
            _ensure_free_master_port("Stage1")
        s1_strategy: object = (
            args.stage1_strategy if args.stage1_devices > 1 else "auto"
        )
        if s1_strategy in ("ddp", "ddp_fork"):
            from lightning.pytorch.strategies import DDPStrategy
            s1_strategy = DDPStrategy(
                broadcast_buffers=False, find_unused_parameters=False,
            )
        pl.Trainer(
            accelerator="gpu",
            devices=args.stage1_devices,
            strategy=s1_strategy,
            max_epochs=args.stage1_epochs,
            logger=_wandb_logger("stage1"),
            enable_checkpointing=False,
            gradient_clip_val=args.grad_clip,
            precision=args.stage1_precision,
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
        coeffs_flat = cache["coeffs"]
        class_ids_flat = cache.get("class_ids")
        fid_real_images = cache.get("fid_real_images")
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
            max_items=min(args.token_subset, len(train_set)),
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
        torch.save({
            "tokens": tokens_flat,
            "coeffs": coeffs_flat,
            "class_ids": class_ids_flat,
            "shape": (H, W, D),
            "stage2_num_classes": stage2_num_classes,
            "fid_real_images": fid_real_images,
        }, token_cache_path)

    print(f"[Stage2] tokens: {tokens_flat.shape}  (H={H}, W={W}, D={D})")

    # ---- stage-2 ----

    if args.stage1_epochs <= 0:
        _run_stage2(
            tokens_flat, coeffs_flat, class_ids_flat, H, W, D, ae,
            fid_real_images=fid_real_images,
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
