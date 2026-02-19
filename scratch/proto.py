
"""
cifar10_sparse_dict_rqtransformer.py

A minimal, end-to-end "RQ-VAE-ish" pipeline using:
  - VQ-VAE-style Encoder/Decoder (conv + residual stack) (from LASER's VQ-VAE baseline)
  - Dictionary-learning bottleneck with batched OMP sparse coding (LASER-style)
  - Option A tokenization: token = atom_id * n_bins + coef_bin
  - A simple "RQTransformer prior" (GPT-style causal transformer) over (H,W,D) stacks
  - CIFAR-10 quick test

Run:
  python cifar10_sparse_dict_rqtransformer.py --stage1_epochs 5 --stage2_epochs 10

This is intentionally compact and hackable, not "best possible" training.
"""
import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torchvision import datasets, transforms, utils
from tqdm import tqdm


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
        n_bins: int = 33,
        coef_max: float = 2.0,
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
        n_bins: int = 33,
        coef_max: float = 2.0,
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

def save_image_grid(x: torch.Tensor, path: str, nrow: int = 8):
    """
    Save a grid of images. Expects x in [-1,1] (we'll map to [0,1]).
    """
    x = x.detach().cpu().clamp(-1, 1)
    x = (x + 1.0) / 2.0
    grid = utils.make_grid(x, nrow=nrow)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    utils.save_image(grid, path)


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
    ):
        super().__init__()
        self.ae = ae
        self.lr = float(lr)
        self.bottleneck_weight = float(bottleneck_weight)
        self.out_dir = out_dir
        self.best_val = float("inf")
        self._val_vis = None

    def configure_optimizers(self):
        return torch.optim.Adam(self.ae.parameters(), lr=self.lr)

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
        self.log("stage1/val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=x.size(0))

        if batch_idx == 0 and self.trainer.is_global_zero:
            self._val_vis = (x[:64].detach(), recon[:64].detach())
        return loss

    def on_validation_epoch_end(self):
        if not self.trainer.is_global_zero:
            return
        os.makedirs(self.out_dir, exist_ok=True)
        cur = self.trainer.callback_metrics.get("stage1/val_loss")
        if cur is None:
            return
        cur_val = float(cur.detach().cpu().item())

        torch.save(self.ae.state_dict(), os.path.join(self.out_dir, "ae_last.pt"))
        if cur_val < self.best_val:
            self.best_val = cur_val
            torch.save(self.ae.state_dict(), os.path.join(self.out_dir, "ae_best.pt"))

        if self._val_vis is not None:
            x_vis, recon_vis = self._val_vis
            epoch = int(self.current_epoch + 1)
            save_image_grid(x_vis, os.path.join(self.out_dir, f"stage1_epoch{epoch:03d}_real.png"))
            save_image_grid(recon_vis, os.path.join(self.out_dir, f"stage1_epoch{epoch:03d}_recon.png"))
            self._val_vis = None


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

        self.ae_for_decode.eval()
        for p in self.ae_for_decode.parameters():
            p.requires_grad_(False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.transformer.parameters(), lr=self.lr)

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
        if not self.trainer.is_global_zero:
            return
        os.makedirs(self.out_dir, exist_ok=True)
        torch.save(self.transformer.state_dict(), os.path.join(self.out_dir, "transformer_last.pt"))

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
        save_image_grid(imgs, os.path.join(self.out_dir, f"stage2_step{step:06d}_samples.png"))
        print(f"[Stage2] sampling done at step {step}")
        self.transformer.train()


def train_stage1_ae(
    ae: SparseDictAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    bottleneck_weight: float,
    grad_clip: float,
    out_dir: str,
):
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        ae.train()
        pbar = tqdm(train_loader, desc=f"[Stage1] epoch {epoch}/{epochs}")
        running = 0.0
        for x, _ in pbar:
            x = x.to(device)
            recon, b_loss, _ = ae(x)
            recon_loss = F.mse_loss(recon, x)
            loss = recon_loss + bottleneck_weight * b_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(ae.parameters(), grad_clip)
            opt.step()

            # Optional: keep dictionary columns bounded (helps stability)
            with torch.no_grad():
                ae.bottleneck.dictionary.copy_(F.normalize(ae.bottleneck.dictionary, p=2, dim=0, eps=ae.bottleneck.epsilon))

            running += loss.item()
            pbar.set_postfix(loss=loss.item(), recon=recon_loss.item(), b=b_loss.item())

        # Validation
        ae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                recon, b_loss, _ = ae(x)
                recon_loss = F.mse_loss(recon, x)
                loss = recon_loss + bottleneck_weight * b_loss
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"[Stage1] epoch {epoch} val_loss={val_loss:.6f}")

        # Save recon sample
        x_vis, _ = next(iter(val_loader))
        x_vis = x_vis.to(device)[:64]
        with torch.no_grad():
            recon_vis, _, _ = ae(x_vis)
        save_image_grid(x_vis, os.path.join(out_dir, f"stage1_epoch{epoch:03d}_real.png"))
        save_image_grid(recon_vis, os.path.join(out_dir, f"stage1_epoch{epoch:03d}_recon.png"))

        # Save best ckpt
        os.makedirs(out_dir, exist_ok=True)
        ckpt_path = os.path.join(out_dir, "ae_last.pt")
        torch.save(ae.state_dict(), ckpt_path)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ae.state_dict(), os.path.join(out_dir, "ae_best.pt"))


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


def train_stage2_transformer(
    transformer: RQTransformerPrior,
    token_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    pad_token_id: int,
    out_dir: str,
    ae_for_decode: SparseDictAE,
    H: int,
    W: int,
    D: int,
    sample_every_steps: int = 200,
    sample_batch_size: int = 8,
):
    opt = torch.optim.Adam(transformer.parameters(), lr=lr)
    vocab = transformer.cfg.vocab_size
    bos = transformer.bos_token_id
    global_step = 0

    for epoch in range(1, epochs + 1):
        transformer.train()
        pbar = tqdm(token_loader, desc=f"[Stage2] epoch {epoch}/{epochs}")
        running = 0.0

        for (tok_flat,) in pbar:
            tok_flat = tok_flat.to(device).long()  # [B, T]
            B = tok_flat.size(0)

            # Prepend BOS: [B, 1+T]
            seq = torch.cat([torch.full((B, 1), bos, device=device, dtype=torch.long), tok_flat], dim=1)
            x_in = seq[:, :-1]
            y = seq[:, 1:]

            logits = transformer(x_in)  # [B, L, vocab]
            loss = F.cross_entropy(
                logits.reshape(-1, vocab),
                y.reshape(-1),
                ignore_index=pad_token_id
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
            opt.step()
            global_step += 1

            running += loss.item()
            pbar.set_postfix(loss=loss.item())

            if sample_every_steps > 0 and (global_step % sample_every_steps == 0):
                transformer.eval()
                ae_for_decode.eval()
                print(f"[Stage2] sampling at step {global_step} (batch_size={sample_batch_size})...")
                with torch.no_grad():
                    flat_gen = transformer.generate(
                        batch_size=sample_batch_size,
                        temperature=1.0,
                        top_k=256,
                        show_progress=True,
                        progress_desc=f"[Stage2] sample step {global_step}",
                    )  # [B, T]
                    tokens_gen = flat_gen.view(-1, H, W, D)
                    imgs = ae_for_decode.decode_from_tokens(tokens_gen.to(device))
                save_image_grid(imgs, os.path.join(out_dir, f"stage2_step{global_step:06d}_samples.png"))
                print(f"[Stage2] sampling done at step {global_step}")
                transformer.train()

        print(f"[Stage2] epoch {epoch} train_loss={running/len(token_loader):.6f}")

        os.makedirs(out_dir, exist_ok=True)
        torch.save(transformer.state_dict(), os.path.join(out_dir, "transformer_last.pt"))


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "celeba"])
    parser.add_argument("--data_dir", type=str, default=None, help="Root directory for dataset files.")
    parser.add_argument("--image_size", type=int, default=None, help="Training image size (HxW).")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)

    # Stage-1 (AE)
    parser.add_argument("--stage1_epochs", type=int, default=5)
    parser.add_argument("--stage1_lr", type=float, default=2e-4)
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
    parser.add_argument("--n_bins", type=int, default=33)
    parser.add_argument("--coef_max", type=float, default=2.0)
    parser.add_argument("--commitment_cost", type=float, default=0.25)

    # Stage-2 (Transformer)
    parser.add_argument("--stage2_epochs", type=int, default=10)
    parser.add_argument("--stage2_lr", type=float, default=3e-4)
    parser.add_argument("--stage2_batch_size", type=int, default=16)
    parser.add_argument("--stage2_sample_every_steps", type=int, default=200)
    parser.add_argument("--stage2_sample_batch_size", type=int, default=8)
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

    args = parser.parse_args()
    if args.image_size is None:
        args.image_size = 64 if args.dataset == "celeba" else 32
    if args.data_dir is None:
        args.data_dir = "../../data/celeba" if args.dataset == "celeba" else "./data"
    if args.out_dir is None:
        args.out_dir = f"./runs/sparse_dict_rq_{args.dataset}_{args.image_size}"

    pl.seed_everything(args.seed, workers=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[Data] dataset={args.dataset} data_dir={args.data_dir} image_size={args.image_size}")

    stage1_dir = os.path.join(args.out_dir, "stage1")
    stage2_dir = os.path.join(args.out_dir, "stage2")
    os.makedirs(stage1_dir, exist_ok=True)
    os.makedirs(stage2_dir, exist_ok=True)
    token_cache_path = os.path.join(stage2_dir, "tokens_cache.pt")

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

    def _run_stage2_lightning(tokens_flat: torch.Tensor, H: int, W: int, D: int, ae_model: SparseDictAE):
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
            logger=False,
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
        ae = _build_ae()
        _load_best_ae_weights(ae)
        _run_stage2_lightning(tokens_flat=tokens_flat, H=H, W=W, D=D, ae_model=ae)
        return

    # Normalize to [-1, 1]
    tfm = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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
        )
        stage1_trainer = pl.Trainer(
            accelerator="gpu",
            devices=args.stage1_devices,
            strategy=(args.stage1_strategy if args.stage1_devices > 1 else "auto"),
            max_epochs=args.stage1_epochs,
            logger=False,
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
    ae = ae.cpu()
    print(f"[Stage2] token dataset: {tokens_flat.shape}   (H={H}, W={W}, D={D})")

    torch.save({"tokens_flat": tokens_flat, "shape": (H, W, D)}, token_cache_path)
    # Re-exec into a clean process before launching stage-2 DDP.
    # This avoids hangs from initializing two different DDP trainers in one process.
    # Also clear stale DDP env from stage-1 so stage-2 can launch all ranks.
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
    os.execv(sys.executable, [sys.executable, __file__, *sys.argv[1:]])

    print("Done.")
    print(f"Outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
