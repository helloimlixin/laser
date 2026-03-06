
"""
laser.py

A minimal, end-to-end "RQ-VAE-ish" pipeline using:
  - VQ-VAE-style Encoder/Decoder (conv + residual stack) (from LASER's VQ-VAE baseline)
  - Dictionary-learning bottleneck with batched OMP sparse coding (LASER-style)
  - Option A tokenization: token = code_id * n_bins + coef_bin
  - A simple "RQTransformer prior" (GPT-style causal transformer) over (H,W,D) stacks

Run:
  python laser.py --dataset cifar10 --stage1_epochs 5 --stage2_epochs 10

This is intentionally compact and hackable, not "best possible" training.
"""
import argparse
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

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
    from torchmetrics.image.fid import FrechetInceptionDistance
except Exception:
    FrechetInceptionDistance = None
try:
    import wandb
except Exception:
    wandb = None


def _disable_cuda_matmul_capability_probe():
    """
    Lightning probes device capability to suggest matmul precision settings.
    On some systems this probe can raise cudaGetDeviceCount error 304 even when
    CUDA training itself works. Disable the probe to avoid false startup crashes.
    """
    try:
        import lightning.pytorch.accelerators.cuda as pl_cuda_accel
        import lightning.fabric.accelerators.cuda as fabric_cuda_accel
    except Exception:
        return

    def _noop_check_cuda_matmul_precision(device):
        return

    pl_cuda_accel._check_cuda_matmul_precision = _noop_check_cuda_matmul_precision
    fabric_cuda_accel._check_cuda_matmul_precision = _noop_check_cuda_matmul_precision


_disable_cuda_matmul_capability_probe()


def soft_clamp(x: torch.Tensor, max_val: float) -> torch.Tensor:
    """Tanh-based soft clamp: approximately linear near zero, smoothly
    saturates towards ±max_val instead of the hard discontinuity of clamp."""
    return max_val * torch.tanh(x / max_val)


def spatial_smooth_coeffs(
    coeffs: torch.Tensor, kernel_size: int = 3, sigma: float = 0.8
) -> torch.Tensor:
    """Apply mild spatial Gaussian smoothing to a coefficient grid.

    Args:
        coeffs: [B, H, W, D] coefficient grid.
        kernel_size: size of the square Gaussian kernel (odd).
        sigma: standard deviation of the Gaussian.

    Returns:
        Smoothed coefficients, same shape.
    """
    B, H, W, D = coeffs.shape
    if H < kernel_size or W < kernel_size:
        return coeffs
    x = coeffs.permute(0, 3, 1, 2).reshape(B * D, 1, H, W)
    pad = kernel_size // 2
    ax = torch.arange(kernel_size, dtype=coeffs.dtype, device=coeffs.device) - pad
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    x = F.pad(x, [pad] * 4, mode="reflect")
    x = F.conv2d(x, kernel)
    return x.reshape(B, D, H, W).permute(0, 2, 3, 1).contiguous()


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

        scan_root = self.root / "img_align_celeba"
        if not scan_root.is_dir():
            scan_root = self.root
        t0 = time.time()
        print(f"[Data] scanning image tree: {scan_root}")
        image_paths = []
        for dirpath, _, filenames in os.walk(scan_root, followlinks=False):
            for name in filenames:
                ext = os.path.splitext(name)[1].lower()
                if ext in IMG_EXTENSIONS:
                    image_paths.append(Path(dirpath) / name)
        image_paths.sort()
        dt = time.time() - t0
        print(f"[Data] indexed {len(image_paths)} images from {scan_root} in {dt:.1f}s")
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


def _bits_needed(n: int) -> int:
    n = max(2, int(n))
    return int(math.ceil(math.log2(n)))


def quantize_coeff_values(
    coeffs: torch.Tensor,
    coef_max: float,
    num_bins: int,
    quantization: str = "mu_law",
    coef_mu: float = 255.0,
) -> torch.Tensor:
    if num_bins <= 1:
        return torch.zeros_like(coeffs, dtype=torch.long)
    coeffs = coeffs.to(torch.float32).clamp(-coef_max, coef_max)
    if quantization == "uniform":
        scaled = (coeffs + coef_max) / (2.0 * coef_max)
    else:
        coeffs_norm = coeffs / coef_max
        mu = max(float(coef_mu), 1e-6)
        scaled = torch.sign(coeffs_norm) * torch.log1p(mu * coeffs_norm.abs()) / math.log1p(mu)
        scaled = (scaled + 1.0) * 0.5
    return torch.round(scaled * (num_bins - 1)).to(torch.long).clamp_(0, num_bins - 1)


def dequantize_coeff_values(
    bin_idx: torch.Tensor,
    coef_max: float,
    num_bins: int,
    quantization: str = "mu_law",
    coef_mu: float = 255.0,
) -> torch.Tensor:
    if num_bins <= 1:
        return torch.zeros_like(bin_idx, dtype=torch.float32)
    z = bin_idx.to(torch.float32) / float(num_bins - 1)
    if quantization == "uniform":
        return (z * 2.0 - 1.0) * coef_max
    z = z * 2.0 - 1.0
    mu = max(float(coef_mu), 1e-6)
    z_abs = z.abs()
    decoded_norm = torch.sign(z) * (torch.expm1(z_abs * math.log1p(mu)) / mu)
    return decoded_norm * coef_max


def pack_sparse_site_keys(
    atom_ids: torch.Tensor,
    coeff_bins: torch.Tensor,
    atom_bits: int,
    bin_bits: int,
) -> torch.Tensor:
    atom_ids = atom_ids.to(torch.int64)
    coeff_bins = coeff_bins.to(torch.int64)
    if atom_ids.shape != coeff_bins.shape:
        raise ValueError(f"shape mismatch: {atom_ids.shape} vs {coeff_bins.shape}")
    key = torch.zeros(atom_ids.size(0), dtype=torch.int64)
    for d in range(atom_ids.size(1)):
        key = (key << atom_bits) | atom_ids[:, d]
        key = (key << bin_bits) | coeff_bins[:, d]
    return key


def unpack_sparse_site_keys(
    keys: torch.Tensor,
    depth: int,
    atom_bits: int,
    bin_bits: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    keys = keys.to(torch.int64)
    atom_mask = (1 << atom_bits) - 1
    bin_mask = (1 << bin_bits) - 1
    atom_ids = torch.zeros(keys.size(0), depth, dtype=torch.long)
    coeff_bins = torch.zeros(keys.size(0), depth, dtype=torch.long)
    work = keys.clone()
    for d in range(depth - 1, -1, -1):
        coeff_bins[:, d] = (work & bin_mask).to(torch.long)
        work = work >> bin_bits
        atom_ids[:, d] = (work & atom_mask).to(torch.long)
        work = work >> atom_bits
    return atom_ids, coeff_bins


@dataclass
class SparseSiteTokenizer:
    vocab_atom_ids: torch.Tensor
    vocab_coeffs: torch.Tensor
    coeff_bins: int
    coeff_quantization: str
    coeff_max: float
    coeff_mu: float

    @property
    def code_depth(self) -> int:
        return int(self.vocab_atom_ids.size(1))

    @property
    def num_site_tokens(self) -> int:
        return int(self.vocab_atom_ids.size(0))

    @property
    def bos_token_id(self) -> int:
        return self.num_site_tokens

    @property
    def pad_token_id(self) -> int:
        return self.num_site_tokens + 1

    @property
    def vocab_size(self) -> int:
        return self.num_site_tokens + 2

    def state_dict(self) -> Dict[str, object]:
        return {
            "vocab_atom_ids": self.vocab_atom_ids.cpu(),
            "vocab_coeffs": self.vocab_coeffs.cpu(),
            "coeff_bins": int(self.coeff_bins),
            "coeff_quantization": str(self.coeff_quantization),
            "coeff_max": float(self.coeff_max),
            "coeff_mu": float(self.coeff_mu),
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, object]) -> "SparseSiteTokenizer":
        return cls(
            vocab_atom_ids=state["vocab_atom_ids"].to(torch.long),
            vocab_coeffs=state["vocab_coeffs"].to(torch.float32),
            coeff_bins=int(state["coeff_bins"]),
            coeff_quantization=str(state["coeff_quantization"]),
            coeff_max=float(state["coeff_max"]),
            coeff_mu=float(state["coeff_mu"]),
        )

    def decode_tokens(
        self,
        tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if tokens.dim() == 4 and tokens.size(-1) == 1:
            tokens = tokens.squeeze(-1)
        flat = tokens.to(torch.long).reshape(-1)
        atom_ids = torch.zeros(
            flat.numel(),
            self.code_depth,
            dtype=torch.long,
            device=flat.device,
        )
        coeffs = torch.zeros(
            flat.numel(),
            self.code_depth,
            dtype=torch.float32,
            device=flat.device,
        )
        valid = (flat >= 0) & (flat < self.num_site_tokens)
        if valid.any():
            atom_vocab = self.vocab_atom_ids.to(flat.device)
            coeff_vocab = self.vocab_coeffs.to(flat.device)
            atom_ids[valid] = atom_vocab[flat[valid]]
            coeffs[valid] = coeff_vocab[flat[valid]]
        out_shape = tokens.shape + (self.code_depth,)
        return atom_ids.view(out_shape), coeffs.view(out_shape)


@torch.no_grad()
def build_sparse_site_tokenizer(
    tokens_flat: torch.Tensor,
    coeffs_flat: torch.Tensor,
    H: int,
    W: int,
    D: int,
    num_atoms: int,
    coeff_bins: int,
    coeff_max: float,
    coeff_quantization: str,
    coeff_mu: float,
    vocab_size: int,
    chunk_images: int = 512,
) -> Tuple[torch.Tensor, SparseSiteTokenizer, float]:
    if coeffs_flat is None:
        raise ValueError("coeffs_flat is required for site tokenization.")
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    atom_bits = _bits_needed(num_atoms)
    bin_bits = _bits_needed(coeff_bins)
    total_bits = D * (atom_bits + bin_bits)
    if total_bits > 62:
        raise ValueError(
            f"Packed site token needs {total_bits} bits; exceeds int64 safety budget."
        )

    counts: Dict[int, int] = {}
    num_images = int(tokens_flat.size(0))
    sites_per_image = int(H * W)
    for start in tqdm(range(0, num_images, chunk_images), desc="[Stage2] build site vocab"):
        end = min(start + chunk_images, num_images)
        atom_chunk = tokens_flat[start:end].view(-1, D).to(torch.long)
        coeff_chunk = coeffs_flat[start:end].view(-1, D).to(torch.float32)
        coeff_bin_chunk = quantize_coeff_values(
            coeff_chunk,
            coef_max=coeff_max,
            num_bins=coeff_bins,
            quantization=coeff_quantization,
            coef_mu=coeff_mu,
        )
        keys = pack_sparse_site_keys(atom_chunk, coeff_bin_chunk, atom_bits, bin_bits).cpu()
        uniq, cnt = torch.unique(keys, return_counts=True)
        for key, count in zip(uniq.tolist(), cnt.tolist()):
            counts[int(key)] = counts.get(int(key), 0) + int(count)

    if not counts:
        raise RuntimeError("Empty site-token vocabulary.")

    top_items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:vocab_size]
    top_keys = torch.tensor([item[0] for item in top_items], dtype=torch.int64)
    vocab_atom_ids, vocab_coeff_bins = unpack_sparse_site_keys(top_keys, D, atom_bits, bin_bits)
    vocab_coeffs = dequantize_coeff_values(
        vocab_coeff_bins,
        coef_max=coeff_max,
        num_bins=coeff_bins,
        quantization=coeff_quantization,
        coef_mu=coeff_mu,
    )
    tokenizer = SparseSiteTokenizer(
        vocab_atom_ids=vocab_atom_ids,
        vocab_coeffs=vocab_coeffs,
        coeff_bins=coeff_bins,
        coeff_quantization=coeff_quantization,
        coeff_max=coeff_max,
        coeff_mu=coeff_mu,
    )

    sort_idx = torch.argsort(top_keys)
    sorted_keys = top_keys[sort_idx]
    sorted_token_ids = torch.arange(top_keys.numel(), dtype=torch.int64)[sort_idx]

    site_tokens_flat = torch.empty((num_images, sites_per_image), dtype=torch.int32)
    oov_sites = 0
    total_sites = int(num_images * sites_per_image)
    for start in tqdm(range(0, num_images, chunk_images), desc="[Stage2] encode site tokens"):
        end = min(start + chunk_images, num_images)
        atom_chunk = tokens_flat[start:end].view(-1, D).to(torch.long)
        coeff_chunk = coeffs_flat[start:end].view(-1, D).to(torch.float32)
        coeff_bin_chunk = quantize_coeff_values(
            coeff_chunk,
            coef_max=coeff_max,
            num_bins=coeff_bins,
            quantization=coeff_quantization,
            coef_mu=coeff_mu,
        )
        keys = pack_sparse_site_keys(atom_chunk, coeff_bin_chunk, atom_bits, bin_bits).cpu()
        idx = torch.searchsorted(sorted_keys, keys)
        idx_clamped = idx.clamp(max=max(int(sorted_keys.numel()) - 1, 0))
        valid = (idx < sorted_keys.numel()) & (sorted_keys[idx_clamped] == keys)
        token_ids = torch.zeros(keys.size(0), dtype=torch.int64)
        token_ids[valid] = sorted_token_ids[idx_clamped[valid]]
        oov_sites += int((~valid).sum().item())
        site_tokens_flat[start:end] = token_ids.view(end - start, sites_per_image).to(torch.int32)

    oov_rate = float(oov_sites) / float(max(total_sites, 1))
    return site_tokens_flat, tokenizer, oov_rate

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
      Applies configurable stride-2 downsampling stages, then residual stack.
    """
    def __init__(
        self,
        in_channels: int,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
        num_downsamples: int = 2,
    ):
        super().__init__()
        self.num_downsamples = int(num_downsamples)
        if self.num_downsamples <= 0:
            raise ValueError(f"num_downsamples must be positive, got {self.num_downsamples}")

        down_convs = []
        cur_in = int(in_channels)
        for i in range(self.num_downsamples):
            if i == 0:
                out_ch = max(1, int(num_hiddens // 2))
            else:
                out_ch = int(num_hiddens)
            down_convs.append(nn.Conv2d(cur_in, out_ch, kernel_size=4, stride=2, padding=1))
            cur_in = out_ch
        self.down_convs = nn.ModuleList(down_convs)
        self.conv3 = nn.Conv2d(cur_in, num_hiddens, kernel_size=3, stride=1, padding=1)
        self.res = ResidualStack(num_hiddens, num_hiddens, num_residual_layers, num_residual_hiddens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.down_convs:
            x = F.relu(conv(x))
        x = self.conv3(x)
        x = self.res(x)
        return x


class Decoder(nn.Module):
    """
    VQ-VAE style decoder:
      Mirrors encoder with configurable transposed-conv upsampling stages.
    """
    def __init__(
        self,
        in_channels: int,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
        out_channels: int = 3,
        num_upsamples: int = 2,
    ):
        super().__init__()
        self.num_upsamples = int(num_upsamples)
        if self.num_upsamples <= 0:
            raise ValueError(f"num_upsamples must be positive, got {self.num_upsamples}")

        self.conv1 = nn.Conv2d(in_channels, num_hiddens, kernel_size=3, stride=1, padding=1)
        self.res = ResidualStack(num_hiddens, num_hiddens, num_residual_layers, num_residual_hiddens)
        up_convs = []
        cur_in = int(num_hiddens)
        for i in range(self.num_upsamples):
            if i == self.num_upsamples - 1:
                out_ch = int(out_channels)
            elif i == self.num_upsamples - 2:
                out_ch = max(1, int(num_hiddens // 2))
            else:
                out_ch = int(num_hiddens)
            up_convs.append(nn.ConvTranspose2d(cur_in, out_ch, kernel_size=4, stride=2, padding=1))
            cur_in = out_ch
        self.up_convs = nn.ModuleList(up_convs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.res(x)
        for i, deconv in enumerate(self.up_convs):
            x = deconv(x)
            if i != (len(self.up_convs) - 1):
                x = F.relu(x)
        return x


# -----------------------------
# Dictionary learning bottleneck (batch OMP) + Option-A tokenization
# -----------------------------

class SparseBottleneck(nn.Module):
    """
    Dictionary-learning bottleneck with batched Orthogonal Matching Pursuit (OMP) sparse coding.
    Tokenization modes:
    - Quantized-mode: token = code_id * n_bins + coefficient_bin (backward compatible).
    - Regressor-mode: token = code_id only, coefficients are modeled with a separate head.

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
        quantize_sparse_coeffs: bool = True,
        coef_quantization: str = "mu_law",
        coef_mu: float = 50.0,
        commitment_cost: float = 0.25,
        epsilon: float = 1e-10,
        latent_patch_size: int = 1,
    ):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.sparsity_level = int(sparsity_level)
        self.latent_patch_size = int(latent_patch_size)
        self.signal_dim = self.embedding_dim * self.latent_patch_size ** 2
        self.n_bins = int(n_bins)
        self.coef_max = float(coef_max)
        self.quantize_sparse_coeffs = bool(quantize_sparse_coeffs)
        self.coef_quantization = str(coef_quantization)
        self.coef_mu = float(coef_mu)
        if self.coef_quantization not in ("uniform", "mu_law"):
            raise ValueError(
                "coef_quantization must be one of {'uniform', 'mu_law'}"
            )
        if self.coef_mu <= 0.0:
            raise ValueError(f"coef_mu must be > 0, got {self.coef_mu}")
        if self.latent_patch_size < 1:
            raise ValueError(f"latent_patch_size must be >= 1, got {self.latent_patch_size}")
        self.commitment_cost = float(commitment_cost)
        self.epsilon = float(epsilon)

        # Dictionary shape [signal_dim, K].  signal_dim = C * P * P.
        self.dictionary = nn.Parameter(torch.randn(self.signal_dim, self.num_embeddings) * 0.02)

        # Coefficient bin centers (uniform)
        centers = torch.linspace(-self.coef_max, self.coef_max, steps=self.n_bins)
        self.register_buffer("coef_bin_centers", centers)
        self.register_buffer(
            "coef_mu_invlog1p",
            torch.tensor(1.0 / (math.log1p(self.coef_mu))),
        )

        # Special tokens (for the transformer)
        if self.quantize_sparse_coeffs:
            self.pad_token_id = self.num_embeddings * self.n_bins
            self.bos_token_id = self.pad_token_id + 1
            self.vocab_size = self.num_embeddings * self.n_bins + 2
        else:
            self.pad_token_id = self.num_embeddings
            self.bos_token_id = self.num_embeddings + 1
            self.vocab_size = self.num_embeddings + 2

    def _norm_dict(self) -> torch.Tensor:
        return F.normalize(self.dictionary, p=2, dim=0, eps=self.epsilon)

    def _quantize_coeff(self, coeff: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize coefficients to bins; return (bin_idx, bin_center_value)."""
        if self.coef_quantization == "uniform":
            c = coeff.clamp(-self.coef_max, self.coef_max)
            scaled = (c + self.coef_max) / (2 * self.coef_max)  # [0,1]
            bin_f = scaled * (self.n_bins - 1)
            bin_idx = torch.round(bin_f).to(torch.long).clamp(0, self.n_bins - 1)
            coeff_q = self.coef_bin_centers[bin_idx]
            return bin_idx, coeff_q

        # μ-law companding: finer resolution near zero for sparse code magnitudes.
        c = coeff.clamp(-self.coef_max, self.coef_max) / self.coef_max
        c_abs = c.abs()
        encoded = torch.sign(c) * torch.log1p(c_abs * self.coef_mu) * self.coef_mu_invlog1p
        scaled = (encoded + 1.0) * ((self.n_bins - 1) / 2.0)
        bin_idx = torch.round(scaled).to(torch.long).clamp(0, self.n_bins - 1)
        decoded = self._dequantize_coeff(bin_idx)
        return bin_idx, decoded

    def _dequantize_coeff(self, bin_idx: torch.Tensor) -> torch.Tensor:
        """Decode bin indices back to quantized coefficients."""
        if self.coef_quantization == "uniform":
            return self.coef_bin_centers[bin_idx]

        # Inverse μ-law companding.
        z = bin_idx.float() * (2.0 / (self.n_bins - 1)) - 1.0
        z_abs = z.abs()
        decoded_norm = torch.sign(z) * (torch.expm1(z_abs / self.coef_mu_invlog1p) / self.coef_mu)
        return decoded_norm * self.coef_max

    def _to_patches(self, z: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """[B, C, H, W] -> [B*pH*pW, C*P*P] and return (signals, pH, pW)."""
        B, C, H, W = z.shape
        P = self.latent_patch_size
        if P == 1:
            return z.permute(0, 2, 3, 1).reshape(-1, C), H, W
        pH, pW = H // P, W // P
        z = z.reshape(B, C, pH, P, pW, P)
        z = z.permute(0, 2, 4, 1, 3, 5).contiguous()
        return z.reshape(B * pH * pW, C * P * P), pH, pW

    def _from_patches(self, flat: torch.Tensor, B: int, pH: int, pW: int) -> torch.Tensor:
        """[B*pH*pW, C*P*P] -> [B, C, H, W]."""
        P = self.latent_patch_size
        C = self.embedding_dim
        if P == 1:
            return flat.view(B, pH, pW, C).permute(0, 3, 1, 2).contiguous()
        z = flat.view(B, pH, pW, C, P, P)
        z = z.permute(0, 3, 1, 4, 2, 5).contiguous()
        return z.reshape(B, C, pH * P, pW * P)

    def _encode(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run OMP and return support code ids and continuous coefficients."""
        B, C, H, W = z_e.shape
        signals, pH, pW = self._to_patches(z_e)
        n_signals = signals.size(0)
        dictionary = self._norm_dict()
        with torch.no_grad():
            support, coeffs = self.omp(signals.t(), dictionary)
        if support.ndim != 2 or coeffs.ndim != 2:
            raise RuntimeError(
                f"OMP returned invalid rank: support={tuple(support.shape)} coeffs={tuple(coeffs.shape)}"
            )
        if support.size(0) != n_signals or coeffs.size(0) != n_signals:
            raise RuntimeError(
                f"OMP returned invalid batch size: expected {n_signals}, "
                f"got support={support.size(0)} coeffs={coeffs.size(0)}"
            )
        if support.size(1) != self.sparsity_level or coeffs.size(1) != self.sparsity_level:
            cur_d = min(support.size(1), coeffs.size(1))
            if cur_d > 0:
                support = support[:, :cur_d]
                coeffs = coeffs[:, :cur_d]
            else:
                support = torch.zeros((n_signals, 0), device=support.device, dtype=support.dtype)
                coeffs = torch.zeros((n_signals, 0), device=coeffs.device, dtype=coeffs.dtype)
            if cur_d < self.sparsity_level:
                pad = self.sparsity_level - cur_d
                support_pad = torch.zeros((n_signals, pad), device=support.device, dtype=support.dtype)
                coeffs_pad = torch.zeros((n_signals, pad), device=coeffs.device, dtype=coeffs.dtype)
                support = torch.cat([support, support_pad], dim=1)
                coeffs = torch.cat([coeffs, coeffs_pad], dim=1)
        return (
            support.view(B, pH, pW, self.sparsity_level),
            coeffs.view(B, pH, pW, self.sparsity_level),
        )

    def _decode(
        self, support: torch.Tensor, coeffs: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct latent map from code ids + coefficients.
        support/coeffs: [B, pH, pW, D] where pH, pW are patch-grid dims."""
        if support.shape != coeffs.shape:
            raise ValueError(
                f"support and coeffs shape mismatch: {support.shape} vs {coeffs.shape}"
            )

        B, pH, pW, D = support.shape
        if D != self.sparsity_level:
            raise ValueError(f"Expected D={self.sparsity_level}, got {D}")

        dictionary = self._norm_dict().t()  # [num_embeddings, signal_dim]
        support = support.to(torch.long)
        coeffs = coeffs.to(dictionary.dtype)
        support_flat = support.reshape(-1, D)
        coeffs_flat = coeffs.reshape(-1, D)
        code_vectors = dictionary[support_flat]  # [N, D, signal_dim]
        recon_flat = (code_vectors * coeffs_flat.unsqueeze(-1)).sum(dim=1)  # [N, signal_dim]
        return self._from_patches(recon_flat, B, pH, pW)

    def omp(self, X: torch.Tensor, D: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched OMP matching the earlier DictLearn.update_gamma implementation.
        Runs exactly sparsity_level steps (no early-stop) so stack depth is fixed.

        Args:
            X: [M, B] signals
            D: [M, N] normalized dictionary
        Returns:
            support: [B, K] indices in selection order (K = sparsity_level)
            coeffs:  [B, K] coefficients aligned with support (same order)
        """
        if self.sparsity_level > int(D.size(1)):
            raise ValueError(
                f"sparsity_level ({self.sparsity_level}) must be <= num_atoms ({int(D.size(1))})"
            )
        _, num_signals = X.size()
        dictionary_t = D.t()                  # [N, M]
        gram_matrix = dictionary_t.mm(D)      # [N, N]
        corr_init = dictionary_t.mm(X).t()    # [B, N]
        gamma = torch.zeros_like(corr_init)   # [B, N]
        corr = corr_init
        L = torch.ones(num_signals, 1, 1, device=X.device, dtype=X.dtype)
        I = torch.zeros(num_signals, 0, dtype=torch.long, device=X.device)
        omega = torch.ones_like(corr_init, dtype=torch.bool)
        signal_idx = torch.arange(num_signals, device=X.device)

        k = 0
        while k < self.sparsity_level:
            k += 1
            # Select max-correlation code while forbidding already-selected atoms.
            scores = torch.abs(corr).masked_fill(~omega, float("-inf"))
            k_hats = torch.argmax(scores, dim=1)  # [B]
            omega[signal_idx, k_hats] = False
            expanded_signal_idx = signal_idx.unsqueeze(0).expand(k, num_signals).t()  # [B, k]

            if k > 1:
                G_ = gram_matrix[
                    I[signal_idx, :],
                    k_hats[expanded_signal_idx[..., :-1]],
                ].view(num_signals, k - 1, 1)
                w = torch.linalg.solve_triangular(L, G_, upper=False).view(-1, 1, k - 1)
                schur = 1.0 - (w ** 2).sum(dim=2, keepdim=True)
                w_br = torch.sqrt(schur.clamp_min(self.epsilon))
                k_zeros = torch.zeros(num_signals, k - 1, 1, device=X.device, dtype=X.dtype)
                L = torch.cat(
                    (
                        torch.cat((L, k_zeros), dim=2),
                        torch.cat((w, w_br), dim=2),
                    ),
                    dim=1,
                )

            I = torch.cat([I, k_hats.unsqueeze(1)], dim=1)  # [B, k]
            corr_ = corr_init[expanded_signal_idx, I[signal_idx, :]].view(num_signals, k, 1)
            gamma_ = torch.cholesky_solve(corr_, L)
            gamma[signal_idx.unsqueeze(1), I[signal_idx]] = gamma_[signal_idx].squeeze(-1)

            beta = (
                gamma[signal_idx.unsqueeze(1), I[signal_idx]]
                .unsqueeze(1)
                .bmm(gram_matrix[I[signal_idx], :])
                .squeeze(1)
            )
            corr = corr_init - beta

        batch_idx = torch.arange(num_signals, device=gamma.device)[:, None]
        coeffs_ordered = gamma[batch_idx, I]  # [B, K]

        return I, coeffs_ordered

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_e: [B, C, H, W]
        Returns:
            z_q_ste: [B, C, H, W]
            loss: scalar bottleneck loss
            tokens: [B, pH, pW, D] where pH=H/P, pW=W/P
        """
        if z_e.dim() != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(z_e.shape)}")
        B, C, H, W = z_e.shape
        if C != self.embedding_dim:
            raise ValueError(f"Expected channel dim {self.embedding_dim}, got {C}")

        support, coeffs = self._encode(z_e)
        pH, pW = support.shape[1], support.shape[2]
        support_flat = support.view(-1, self.sparsity_level)
        coeffs_flat = coeffs.view(-1, self.sparsity_level)

        if self.quantize_sparse_coeffs:
            bin_idx, coeff_q = self._quantize_coeff(coeffs_flat)
            tokens = (support_flat * self.n_bins + bin_idx).view(B, pH, pW, self.sparsity_level)
            coeffs_for_recon = coeff_q
        else:
            tokens = support.view(B, pH, pW, self.sparsity_level).long()
            coeffs_for_recon = coeffs_flat

        coeffs_for_recon = coeffs_for_recon.reshape(B, pH, pW, self.sparsity_level)
        z_q = self._decode(support, coeffs_for_recon)

        dl_latent_loss = F.mse_loss(z_q, z_e.detach())
        e_latent_loss = F.mse_loss(z_q.detach(), z_e)
        loss = dl_latent_loss + self.commitment_cost * e_latent_loss

        z_q_ste = z_e + (z_q - z_e).detach()
        return z_q_ste, loss, tokens

    @torch.no_grad()
    def decode_tokens(self, tokens: torch.Tensor, coeffs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode tokens back to a latent map.
        Args:
            tokens: [B, pH, pW, D]
            coeffs: [B, pH, pW, D] (only used in non-quantized mode)
        Returns:
            z_q: [B, C, H, W]  (H = pH * P, W = pW * P)
        """
        if tokens.dim() != 4:
            raise ValueError(f"Expected [B,pH,pW,D], got {tuple(tokens.shape)}")
        D = tokens.shape[3]
        if D != self.sparsity_level:
            raise ValueError(f"Expected D={self.sparsity_level}, got {D}")

        if self.quantize_sparse_coeffs:
            tok = tokens.to(torch.long)
            special = tok >= self.pad_token_id
            tok_clamped = tok.clamp_max(self.pad_token_id - 1)

            code_ids = tok_clamped // self.n_bins
            bin_idx = tok_clamped % self.n_bins

            coeff = self._dequantize_coeff(bin_idx).to(self._norm_dict().dtype)
            coeff = coeff * (~special).float()
            code_ids = code_ids * (~special).long()
            return self._decode(code_ids, coeff)

        if coeffs is None:
            raise ValueError("coeffs must be provided when quantize_sparse_coeffs=False")

        return self._decode(tokens.to(torch.long), coeffs.to(self._norm_dict().dtype))

    @torch.no_grad()
    def project_codes(
        self,
        support: torch.Tensor,
        coeffs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project arbitrary sparse codes back onto the valid OMP code manifold.
        This is useful for stage-2 generated codes, which can otherwise contain
        duplicate atoms or unstable coefficient combinations.
        """
        z_q = self._decode(support, coeffs)
        return self._encode(z_q)

# -----------------------------
# Stage-1 model: LASER (Encoder + Dictionary bottleneck + Decoder)
# -----------------------------

class LASER(nn.Module):
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
        n_bins: int = 129,
        coef_max: float = 3.0,
        coef_quantization: str = "mu_law",
        coef_mu: float = 50.0,
        out_tanh: bool = True,
        quantize_sparse_coeffs: bool = True,
        latent_patch_size: int = 1,
    ):
        super().__init__()
        self.out_tanh = bool(out_tanh)

        self.encoder = Encoder(
            in_channels,
            num_hiddens,
            num_residual_layers,
            num_residual_hiddens,
            num_downsamples=num_downsamples,
        )
        self.pre_bottleneck = nn.Conv2d(num_hiddens, embedding_dim, kernel_size=1)
        self.bottleneck = SparseBottleneck(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            sparsity_level=sparsity_level,
            n_bins=n_bins,
            coef_max=coef_max,
            quantize_sparse_coeffs=quantize_sparse_coeffs,
            coef_quantization=coef_quantization,
            coef_mu=coef_mu,
            commitment_cost=commitment_cost,
            latent_patch_size=latent_patch_size,
        )
        self.post_bottleneck = nn.Conv2d(embedding_dim, num_hiddens, kernel_size=3, padding=1)
        self.decoder = Decoder(
            num_hiddens,
            num_hiddens,
            num_residual_layers,
            num_residual_hiddens,
            out_channels=in_channels,
            num_upsamples=num_downsamples,
        )

    def _enc(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        LASER-style encoder API.
        Returns:
            z_q: quantized latent [B, C, H, W]
            b_loss: bottleneck loss scalar
            tokens: [B, H, W, D]
        """
        z = self.encoder(x)
        z = self.pre_bottleneck(z)
        z_q, b_loss, tokens = self.bottleneck(z)
        return z_q, b_loss, tokens

    def _dec(self, z_q: torch.Tensor) -> torch.Tensor:
        """LASER-style decoder API from latent to image space."""
        z_q = self.post_bottleneck(z_q)
        recon = self.decoder(z_q)
        if self.out_tanh:
            recon = torch.tanh(recon)
        return recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_q, b_loss, tokens = self._enc(x)
        recon = self._dec(z_q)
        return recon, b_loss, tokens

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience API matching LASER-style reconstruction usage."""
        recon, _, _ = self(x)
        return recon

    def compute_metrics(self, x: torch.Tensor, bottleneck_weight: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Compute stage-1 metrics in model space (mirrors src/models/laser.py style).
        Returns recon/loss tensors used by the Lightning wrapper.
        """
        recon, b_loss, tokens = self(x)
        recon_loss = F.mse_loss(recon, x)
        loss = recon_loss + float(bottleneck_weight) * b_loss
        # Inputs/recon are normalized to [-1, 1], so PSNR peak value is 2.0.
        psnr = 10.0 * torch.log10(4.0 / torch.clamp(recon_loss.detach(), min=1e-8))
        return {
            "recon": recon,
            "tokens": tokens,
            "b_loss": b_loss,
            "recon_loss": recon_loss,
            "loss": loss,
            "psnr": psnr,
        }

    @torch.no_grad()
    def encode_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        _, _, tokens = self._enc(x)
        return tokens, tokens.shape[1], tokens.shape[2]

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        z = self.encoder(x)
        z = self.pre_bottleneck(z)
        atom_ids, coeffs = self.bottleneck._encode(z)
        return atom_ids, coeffs, atom_ids.shape[1], atom_ids.shape[2]

    @torch.no_grad()
    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        z_q = self.bottleneck.decode_tokens(tokens)
        return self._dec(z_q)

    @torch.no_grad()
    def decode(self, atom_ids: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        z_q = self.bottleneck._decode(atom_ids, coeffs)
        return self._dec(z_q)

    @torch.no_grad()
    def project_codes(
        self,
        atom_ids: torch.Tensor,
        coeffs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.bottleneck.project_codes(atom_ids, coeffs)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        dropout_p = self.attn_dropout.p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=dropout_p,
        )
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
class PriorConfig:
    vocab_size: int
    H: int
    W: int
    D: int
    predict_coefficients: bool = False
    coeff_loss_weight: float = 1.0
    coeff_max: float = 3.0
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1


class Prior(nn.Module):
    """
    A simple RQ-style prior:
      - The full sequence is: [BOS] + raster_scan(H*W) each with depth D tokens.
      - Embedding = token + spatial_pos + depth_pos + type(BOS vs normal)
      - GPT-style causal blocks.
    """
    def __init__(self, cfg: PriorConfig, bos_token_id: int, pad_token_id: int):
        super().__init__()
        self.cfg = cfg
        self.predict_coefficients = bool(cfg.predict_coefficients)
        self.bos_token_id = int(bos_token_id)
        self.pad_token_id = int(pad_token_id)

        self.tokens_per_patch = cfg.H * cfg.W * cfg.D
        self.max_len = 1 + self.tokens_per_patch

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
        self.atom_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if self.predict_coefficients:
            self.atom_coeff_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.coeff_head = nn.Sequential(
                nn.Linear(cfg.d_model * 2, cfg.d_model),
                nn.GELU(),
                nn.Linear(cfg.d_model, 1),
            )

        spatial_ids = torch.zeros(self.max_len, dtype=torch.long)
        depth_ids = torch.zeros(self.max_len, dtype=torch.long)
        type_ids = torch.zeros(self.max_len, dtype=torch.long)
        if self.max_len > 1:
            idx = torch.arange(self.max_len - 1)
            local_idx = idx % self.tokens_per_patch
            spatial_ids[1:] = local_idx // cfg.D
            depth_ids[1:] = local_idx % cfg.D
            type_ids[1:] = 1
        self.register_buffer("_spatial_ids", spatial_ids)
        self.register_buffer("_depth_ids", depth_ids)
        self.register_buffer("_type_ids", type_ids)

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Run transformer backbone, return hidden states [B, L, d_model]."""
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

        return self.ln_f(h)

    def _predict_coeff(self, h: torch.Tensor, atom_ids: torch.Tensor) -> torch.Tensor:
        """Predict coefficients given hidden states and atom IDs. Both [B, L]."""
        atom_ids = atom_ids.clamp(0, self.cfg.vocab_size - 1)
        atom_emb = self.atom_coeff_emb(atom_ids)
        coeff_input = torch.cat([h, atom_emb], dim=-1)
        return self.coeff_head(coeff_input).squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        atom_targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L] tokens, L <= max_len
            atom_targets: [B, L] ground-truth atom IDs for coefficient prediction
                          (training only; ignored when predict_coefficients=False)
        Returns:
            If predict_coefficients=False:
                logits: [B, L, vocab]
            If predict_coefficients=True:
                atom_logits: [B, L, vocab], coeff: [B, L]
        """
        h = self._backbone(x)
        atom_logits = self.atom_head(h)
        if not self.predict_coefficients:
            return atom_logits

        if atom_targets is not None:
            atom_ids = atom_targets
        else:
            atom_ids = atom_logits.argmax(dim=-1)
        coeff = self._predict_coeff(h, atom_ids)
        return atom_logits, coeff

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Unconditional generation.
        Returns:
            If predict_coefficients=False:
                flat_tokens: [B, H*W*D] (without BOS)
            If predict_coefficients=True:
                flat_tokens, flat_coeffs
        """
        device = next(self.parameters()).device
        T = self.cfg.H * self.cfg.W * self.cfg.D

        seq = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=device)
        coeffs: list[torch.Tensor] = []
        steps = tqdm(
            range(T),
            desc=(progress_desc or "[Stage2] sampling tokens"),
            leave=False,
            dynamic_ncols=True,
            disable=(not show_progress),
        )
        for _ in steps:
            h = self._backbone(seq)
            atom_logits = self.atom_head(h)
            special_ids = torch.tensor(
                [self.bos_token_id, self.pad_token_id],
                device=atom_logits.device,
            )
            atom_logits[:, :, special_ids] = float("-inf")
            logits = atom_logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            seq = torch.cat([seq, nxt], dim=1)
            if self.predict_coefficients:
                coeff_step = self._predict_coeff(h[:, -1:, :], nxt)
                coeffs.append(coeff_step.squeeze(-1))

        if self.predict_coefficients:
            coeff_flat = torch.stack(coeffs, dim=1)
            coeff_flat = soft_clamp(coeff_flat, self.cfg.coeff_max)
            return seq[:, 1:], coeff_flat
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


class TokenDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokens_flat: torch.Tensor,
        batch_size: int,
        num_workers: int = 2,
        coeffs_flat: Optional[torch.Tensor] = None,
        sliding_window_shape: Optional[Tuple[int, int]] = None,
        full_latent_shape: Optional[Tuple[int, int]] = None,
        latent_depth: Optional[int] = None,
        sliding_window_stride_latent: Optional[int] = None,
    ):
        super().__init__()
        self.tokens_flat = tokens_flat
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.coeffs_flat = coeffs_flat
        self.sliding_window_shape = (
            None
            if sliding_window_shape is None
            else (int(sliding_window_shape[0]), int(sliding_window_shape[1]))
        )
        self.full_latent_shape = (
            None
            if full_latent_shape is None
            else (int(full_latent_shape[0]), int(full_latent_shape[1]))
        )
        self.latent_depth = (None if latent_depth is None else int(latent_depth))
        self.sliding_window_stride_latent = (
            None
            if sliding_window_stride_latent is None
            else max(1, int(sliding_window_stride_latent))
        )

    def train_dataloader(self):
        if self.sliding_window_shape is not None:
            if self.full_latent_shape is None or self.latent_depth is None:
                raise ValueError("full_latent_shape and latent_depth are required for sliding-window mode.")
            tok_ds = WindowTokenDataset(
                tokens_flat=self.tokens_flat,
                full_latent_h=self.full_latent_shape[0],
                full_latent_w=self.full_latent_shape[1],
                latent_depth=self.latent_depth,
                window_latent_h=self.sliding_window_shape[0],
                window_latent_w=self.sliding_window_shape[1],
                stride_latent=(
                    self.sliding_window_stride_latent
                    if self.sliding_window_stride_latent is not None
                    else self.sliding_window_shape[0]
                ),
                coeffs_flat=self.coeffs_flat,
            )
        else:
            tensors = [self.tokens_flat]
            if self.coeffs_flat is not None:
                tensors.append(self.coeffs_flat)
            tok_ds = TensorDataset(*tensors)
        return DataLoader(
            tok_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(self.num_workers > 0),
        )


class WindowTokenDataset(Dataset):
    """
    Builds fixed-length autoregressive training windows from full latent-token sequences.
    Each sample yields:
      tok_flat: [window_tokens]
      coeff_flat: [window_tokens] (optional)
    """

    def __init__(
        self,
        tokens_flat: torch.Tensor,
        full_latent_h: int,
        full_latent_w: int,
        latent_depth: int,
        window_latent_h: int,
        window_latent_w: int,
        stride_latent: int,
        coeffs_flat: Optional[torch.Tensor] = None,
    ):
        if tokens_flat.dim() != 2:
            raise ValueError(f"tokens_flat must be [N, T], got {tuple(tokens_flat.shape)}")
        self.tokens_flat = tokens_flat
        self.coeffs_flat = coeffs_flat
        self.full_latent_h = int(full_latent_h)
        self.full_latent_w = int(full_latent_w)
        self.latent_depth = int(latent_depth)
        self.window_latent_h = int(window_latent_h)
        self.window_latent_w = int(window_latent_w)
        self.stride_latent = max(1, int(stride_latent))
        self.full_tokens = int(tokens_flat.shape[1])
        expected_tokens = self.full_latent_h * self.full_latent_w * self.latent_depth
        if self.full_tokens != expected_tokens:
            raise ValueError(
                f"tokens_flat has T={self.full_tokens}, but expected full_latent_h*full_latent_w*latent_depth="
                f"{self.full_latent_h}*{self.full_latent_w}*{self.latent_depth}={expected_tokens}"
            )
        if self.window_latent_h <= 0 or self.window_latent_w <= 0:
            raise ValueError("window_latent_h and window_latent_w must be positive.")
        if self.window_latent_h > self.full_latent_h or self.window_latent_w > self.full_latent_w:
            raise ValueError(
                f"Window latent shape {(self.window_latent_h, self.window_latent_w)} exceeds "
                f"full latent shape {(self.full_latent_h, self.full_latent_w)}."
            )
        if coeffs_flat is not None and coeffs_flat.shape != tokens_flat.shape:
            raise ValueError(
                f"coeffs_flat shape mismatch: expected {tuple(tokens_flat.shape)}, got {tuple(coeffs_flat.shape)}"
            )
        if ((self.full_latent_h - self.window_latent_h) % self.stride_latent) != 0:
            raise ValueError(
                f"Incompatible latent height window/stride: full_h={self.full_latent_h}, "
                f"window_h={self.window_latent_h}, stride={self.stride_latent}"
            )
        if ((self.full_latent_w - self.window_latent_w) % self.stride_latent) != 0:
            raise ValueError(
                f"Incompatible latent width window/stride: full_w={self.full_latent_w}, "
                f"window_w={self.window_latent_w}, stride={self.stride_latent}"
            )
        self.grid_h = ((self.full_latent_h - self.window_latent_h) // self.stride_latent) + 1
        self.grid_w = ((self.full_latent_w - self.window_latent_w) // self.stride_latent) + 1
        self.windows_per_image = int(self.grid_h * self.grid_w)
        if self.windows_per_image <= 0:
            raise ValueError("No valid sliding windows produced for the requested configuration.")
        self.window_tokens = int(self.window_latent_h * self.window_latent_w * self.latent_depth)

    def __len__(self) -> int:
        return int(self.tokens_flat.shape[0]) * int(self.windows_per_image)

    def __getitem__(self, idx: int):
        image_idx = int(idx) // self.windows_per_image
        window_idx = int(idx) % self.windows_per_image
        wy = window_idx // self.grid_w
        wx = window_idx % self.grid_w
        h0 = int(wy * self.stride_latent)
        w0 = int(wx * self.stride_latent)
        h1 = h0 + self.window_latent_h
        w1 = w0 + self.window_latent_w

        tok_img = self.tokens_flat[image_idx].view(
            self.full_latent_h, self.full_latent_w, self.latent_depth
        )
        tok = tok_img[h0:h1, w0:w1, :].contiguous().view(self.window_tokens)
        if self.coeffs_flat is None:
            return (tok,)
        coeff_img = self.coeffs_flat[image_idx].view(
            self.full_latent_h, self.full_latent_w, self.latent_depth
        )
        coeff = coeff_img[h0:h1, w0:w1, :].contiguous().view(self.window_tokens)
        return tok, coeff


class Stage1Module(pl.LightningModule):
    def __init__(
        self,
        ae: LASER,
        lr: float,
        bottleneck_weight: float,
        out_dir: str,
        val_vis_images: Optional[torch.Tensor] = None,
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
        self.val_vis_images = val_vis_images
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
        self._fid_warned_adjusted_feature = False
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
                sync_on_compute=True,
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
        metrics = self.ae.compute_metrics(x, bottleneck_weight=self.bottleneck_weight)
        loss = metrics["loss"]
        self.log("stage1/train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=x.size(0))
        self.log("stage1/recon_loss", metrics["recon_loss"], on_step=True, on_epoch=True, sync_dist=True, batch_size=x.size(0))
        self.log("stage1/b_loss", metrics["b_loss"], on_step=True, on_epoch=True, sync_dist=True, batch_size=x.size(0))
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        with torch.no_grad():
            bn = self.ae.bottleneck
            bn.dictionary.copy_(F.normalize(bn.dictionary, p=2, dim=0, eps=bn.epsilon))

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.broadcast(bn.dictionary.data, src=0)

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        metrics = self.ae.compute_metrics(x, bottleneck_weight=self.bottleneck_weight)
        recon = metrics["recon"]
        loss = metrics["loss"]
        self.log("stage1/val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=x.size(0))
        self.log("stage1/val_psnr", metrics["psnr"], on_epoch=True, prog_bar=True, sync_dist=True, batch_size=x.size(0))

        if (not self.trainer.sanity_checking) and self.fid_num_samples > 0:
            if FrechetInceptionDistance is None:
                if self.trainer.is_global_zero and not self._fid_warned_unavailable:
                    print("[Stage1] FID unavailable: torchmetrics.image.fid not installed.")
                    self._fid_warned_unavailable = True
            else:
                world_size = max(1, int(getattr(self.trainer, "world_size", 1)))
                local_cap = int(math.ceil(float(self.fid_num_samples) / float(world_size)))
                if self._fid_seen >= local_cap:
                    return loss
                keep = min(x.size(0), local_cap - self._fid_seen)
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

            if self.val_vis_images is not None and self.val_vis_images.numel() > 0:
                x_vis = self.val_vis_images.to(self.device)
                with torch.no_grad():
                    recon_vis = self.ae.reconstruct(x_vis)
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
                global_seen = int(self._fid_seen)
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    seen_tensor = torch.tensor([global_seen], device=self.device, dtype=torch.long)
                    torch.distributed.all_reduce(seen_tensor, op=torch.distributed.ReduceOp.SUM)
                    global_seen = int(seen_tensor.item())

                effective_feature = self.fid_feature
                valid_features = [64, 192, 768, 2048]
                if effective_feature not in valid_features:
                    effective_feature = 64
                if global_seen < effective_feature:
                    adjusted_feature = 64
                    if (
                        self.trainer.is_global_zero
                        and adjusted_feature != effective_feature
                        and not self._fid_warned_adjusted_feature
                    ):
                        print(
                            f"[Stage1] adjusting FID feature from {effective_feature} "
                            f"to {adjusted_feature} for n={global_seen}."
                        )
                        self._fid_warned_adjusted_feature = True
                    effective_feature = adjusted_feature

                fid_metric = self._get_or_create_fid_metric(effective_feature)
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


class Stage2Module(pl.LightningModule):
    def __init__(
        self,
        transformer,  # Prior or SpatialDepthPrior
        lr: float,
        pad_token_id: int,
        out_dir: str,
        laser: LASER,
        H: int,
        W: int,
        D: int,
        image_size: int,
        sample_every_steps: int = 200,
        sample_batch_size: int = 8,
        sample_latent_shape: Optional[Tuple[int, int]] = None,
        sample_window_stride_latent: int = 1,
        fid_real_images: Optional[torch.Tensor] = None,
        fid_num_samples: int = 64,
        fid_feature: int = 2048,
        fid_every_n_epochs: int = 1,
        coeff_loss_weight: float = 1.0,
        coeff_mean: float = 0.0,
        coeff_std: float = 1.0,
        lr_schedule: str = "cosine",
        warmup_epochs: int = 1,
        min_lr_ratio: float = 0.1,
        coeff_norm_clip: float = 3.0,
        latent_loss_weight: float = 0.25,
        coeff_energy_loss_weight: float = 0.25,
        projection_consistency_weight: float = 0.5,
        projection_consistency_sites: int = 256,
        site_tokenizer: Optional[SparseSiteTokenizer] = None,
        arch_tag: Optional[str] = None,
        dump_sparse_debug: bool = False,
        sparse_debug_topk: int = 16,
    ):
        super().__init__()
        self.transformer = transformer
        self.lr = float(lr)
        self.pad_token_id = int(pad_token_id)
        from laser_transformer import SpatialDepthPrior
        from laser_diffusion_prior import DiffusionPrior
        self.is_spatial_depth = isinstance(transformer, SpatialDepthPrior)
        self.is_diffusion = isinstance(transformer, DiffusionPrior)
        self.predict_coefficients = (
            True if (self.is_spatial_depth or self.is_diffusion)
            else bool(self.transformer.cfg.predict_coefficients)
        )
        self.coeff_loss_weight = float(coeff_loss_weight)
        self.latent_loss_weight = float(max(0.0, latent_loss_weight))
        self.coeff_energy_loss_weight = float(max(0.0, coeff_energy_loss_weight))
        self.projection_consistency_weight = float(max(0.0, projection_consistency_weight))
        self.projection_consistency_sites = max(1, int(projection_consistency_sites))
        self.coeff_mean = float(coeff_mean)
        self.coeff_std = max(float(coeff_std), 1e-8)
        self.coeff_norm_clip = max(float(coeff_norm_clip), 1e-6)
        self.site_tokenizer = site_tokenizer
        self.dump_sparse_debug = bool(dump_sparse_debug)
        self.sparse_debug_topk = max(1, int(sparse_debug_topk))
        if arch_tag is None:
            if self.is_spatial_depth:
                arch_tag = "spatial_depth"
            elif self.is_diffusion:
                arch_tag = "diffusion"
            elif self.site_tokenizer is not None:
                arch_tag = "site_flat"
            else:
                arch_tag = "flat"
        self.arch_tag = str(arch_tag)
        self.out_dir = out_dir
        self.laser = laser
        self.H, self.W, self.D = int(H), int(W), int(D)
        self.image_size = int(image_size)
        self.sample_every_steps = int(sample_every_steps)
        self.sample_batch_size = int(sample_batch_size)
        if sample_latent_shape is None:
            self.sample_latent_h, self.sample_latent_w = self.H, self.W
        else:
            self.sample_latent_h = int(sample_latent_shape[0])
            self.sample_latent_w = int(sample_latent_shape[1])
        self.sample_window_stride_latent = max(1, int(sample_window_stride_latent))
        self.use_sliding_window_sampling = bool(sample_latent_shape is not None)
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

        if (
            self.use_sliding_window_sampling
            and (self.is_spatial_depth or self.is_diffusion)
            and self.sample_every_steps > 0
        ):
            print(
                "[Stage2] disabling periodic sampling: sliding-window sampling is only "
                "implemented for stage2_arch=flat."
            )
            self.sample_every_steps = 0

        self.laser.eval()
        for p in self.laser.parameters():
            p.requires_grad_(False)

    def _denormalize_coeffs(
        self,
        coeff_gen: torch.Tensor,
        apply_spatial_smoothing: bool = False,
    ) -> torch.Tensor:
        """Map normalized coefficients back to AE coefficient space safely."""
        coeff_gen = coeff_gen.to(self.device, dtype=torch.float32)
        coeff_gen = torch.nan_to_num(coeff_gen, nan=0.0, posinf=0.0, neginf=0.0)
        coeff_gen = soft_clamp(coeff_gen, self.coeff_norm_clip)
        coeff_gen = coeff_gen * self.coeff_std + self.coeff_mean
        raw_clip = float(getattr(self.laser.bottleneck, "coef_max", 0.0))
        if raw_clip > 0.0:
            coeff_gen = coeff_gen.clamp(-raw_clip, raw_clip)
        if apply_spatial_smoothing:
            coeff_gen = spatial_smooth_coeffs(coeff_gen)
        return coeff_gen

    def _postprocess_generated_coeffs(self, coeff_gen: torch.Tensor) -> torch.Tensor:
        return self._denormalize_coeffs(coeff_gen, apply_spatial_smoothing=True)

    def _reshape_coeff_grid(self, coeffs: torch.Tensor) -> torch.Tensor:
        if coeffs.dim() == 2:
            return coeffs.view(-1, self.H, self.W, self.D)
        if coeffs.dim() == 3:
            return coeffs.view(-1, self.H, self.W, self.D)
        if coeffs.dim() == 4:
            return coeffs
        raise ValueError(f"Unexpected coeff shape: {tuple(coeffs.shape)}")

    def _reshape_atom_logits_grid(self, atom_logits: torch.Tensor) -> torch.Tensor:
        if atom_logits.dim() == 4:
            return atom_logits.view(-1, self.H, self.W, self.D, atom_logits.size(-1))
        if atom_logits.dim() == 3:
            return atom_logits.view(-1, self.H, self.W, self.D, atom_logits.size(-1))
        raise ValueError(f"Unexpected atom_logits shape: {tuple(atom_logits.shape)}")

    def _site_coeff_energy(self, coeffs_raw: torch.Tensor) -> torch.Tensor:
        coeffs_grid = self._reshape_coeff_grid(coeffs_raw).to(torch.float32)
        return coeffs_grid.abs().sum(dim=-1)

    def _coefficient_energy_loss(
        self,
        coeff_pred_norm: torch.Tensor,
        coeff_target_raw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_coeff_raw = self._denormalize_coeffs(
            coeff_pred_norm,
            apply_spatial_smoothing=False,
        )
        pred_energy = self._site_coeff_energy(pred_coeff_raw)
        target_energy = self._site_coeff_energy(coeff_target_raw.to(pred_energy.device))
        loss = F.smooth_l1_loss(pred_energy, target_energy)
        return loss, pred_energy.mean(), target_energy.mean()

    def _projection_consistency_loss(
        self,
        atom_logits: torch.Tensor,
        coeff_pred_norm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        atom_logits_grid = self._reshape_atom_logits_grid(atom_logits)
        coeff_pred_grid = self._reshape_coeff_grid(coeff_pred_norm)

        B, H, W, D, V = atom_logits_grid.shape
        site_count = B * H * W
        logits_sites = atom_logits_grid.view(site_count, D, V)
        coeff_sites = coeff_pred_grid.view(site_count, D)
        pred_atom_sites = logits_sites.argmax(dim=-1)

        if site_count > self.projection_consistency_sites:
            keep = torch.randperm(site_count, device=logits_sites.device)[: self.projection_consistency_sites]
            logits_sites = logits_sites[keep]
            coeff_sites = coeff_sites[keep]
            pred_atom_sites = pred_atom_sites[keep]

        pred_atom_codes = pred_atom_sites.view(-1, 1, 1, D).detach()
        pred_coeff_raw = self._denormalize_coeffs(
            coeff_sites.view(-1, 1, 1, D).detach(),
            apply_spatial_smoothing=False,
        )
        proj_atom_ids, proj_coeffs_raw = self.laser.project_codes(pred_atom_codes, pred_coeff_raw)
        proj_atom_ids = proj_atom_ids.view(-1, D)
        proj_coeffs_norm = (proj_coeffs_raw.view(-1, D) - self.coeff_mean) / self.coeff_std

        atom_loss = F.cross_entropy(
            logits_sites.reshape(-1, V),
            proj_atom_ids.reshape(-1),
        )
        coeff_loss = F.mse_loss(
            coeff_sites.reshape(-1),
            proj_coeffs_norm.reshape(-1),
        )
        change_rate = (pred_atom_sites != proj_atom_ids).any(dim=-1).float().mean()
        return atom_loss + coeff_loss, change_rate

    def _soft_code_to_latent(
        self,
        atom_logits: torch.Tensor,
        coeff_pred_norm: torch.Tensor,
    ) -> torch.Tensor:
        num_atoms = int(self.laser.bottleneck.num_embeddings)
        if atom_logits.dim() == 3:
            B, _, V = atom_logits.shape
            atom_logits = atom_logits.view(B, self.H, self.W, self.D, V)
        elif atom_logits.dim() == 4:
            B, _, _, V = atom_logits.shape
            atom_logits = atom_logits.view(B, self.H, self.W, self.D, V)
        else:
            raise ValueError(f"Unexpected atom_logits shape: {tuple(atom_logits.shape)}")
        if atom_logits.size(-1) < num_atoms:
            raise ValueError(
                f"atom_logits vocab dim {atom_logits.size(-1)} is smaller than num_atoms {num_atoms}"
            )
        atom_logits = atom_logits[..., :num_atoms]

        if coeff_pred_norm.dim() == 2:
            coeff_pred_norm = coeff_pred_norm.view(-1, self.H, self.W, self.D)
        elif coeff_pred_norm.dim() == 3:
            coeff_pred_norm = coeff_pred_norm.view(-1, self.H, self.W, self.D)
        else:
            raise ValueError(f"Unexpected coeff_pred_norm shape: {tuple(coeff_pred_norm.shape)}")

        probs = F.softmax(atom_logits.float(), dim=-1)
        coeff_pred = self._denormalize_coeffs(coeff_pred_norm, apply_spatial_smoothing=False)
        dictionary = self.laser.bottleneck._norm_dict().t().to(probs.dtype)
        soft_atoms = torch.einsum("bhwdv,vc->bhwdc", probs, dictionary)
        recon_flat = (soft_atoms * coeff_pred.unsqueeze(-1)).sum(dim=3)
        B = recon_flat.size(0)
        recon_flat = recon_flat.reshape(-1, self.laser.bottleneck.signal_dim)
        return self.laser.bottleneck._from_patches(recon_flat, B, self.H, self.W)

    def _target_code_to_latent(
        self,
        atom_ids: torch.Tensor,
        coeffs_raw: torch.Tensor,
    ) -> torch.Tensor:
        if atom_ids.dim() == 3:
            atom_ids = atom_ids.view(-1, self.H, self.W, self.D)
        if coeffs_raw.dim() == 3:
            coeffs_raw = coeffs_raw.view(-1, self.H, self.W, self.D)
        return self.laser.bottleneck._decode(atom_ids, coeffs_raw)

    def _reshape_generated_codes(self, gen) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if not self.predict_coefficients:
            return None
        if self.is_spatial_depth or self.is_diffusion:
            atom_ids, coeff_gen = gen
        else:
            atom_ids, coeff_gen = gen
        atom_ids = atom_ids.view(-1, self.H, self.W, self.D).to(self.device)
        coeff_gen = coeff_gen.view(-1, self.H, self.W, self.D).to(self.device)
        return atom_ids, coeff_gen

    def _build_sparse_debug_snapshot(
        self,
        gen,
        raw_imgs: torch.Tensor,
    ) -> Optional[dict]:
        reshaped = self._reshape_generated_codes(gen)
        if reshaped is None:
            return None
        raw_atom_ids, raw_coeff_norm = reshaped
        proc_coeffs = self._postprocess_generated_coeffs(raw_coeff_norm)
        proj_atom_ids, proj_coeffs = self.laser.project_codes(raw_atom_ids, proc_coeffs)
        proj_imgs = self.laser.decode(proj_atom_ids, proj_coeffs)

        duplicate_sites = torch.zeros(raw_atom_ids.shape[:-1], device=self.device, dtype=torch.bool)
        for i in range(self.D):
            for j in range(i + 1, self.D):
                duplicate_sites |= (raw_atom_ids[..., i] == raw_atom_ids[..., j])
        atom_changed_sites = (raw_atom_ids != proj_atom_ids).any(dim=-1)
        coeff_change = (proc_coeffs - proj_coeffs).abs().sum(dim=-1)
        invalid_score = coeff_change + duplicate_sites.float() + atom_changed_sites.float()
        image_l1 = (raw_imgs - proj_imgs).abs()

        flat_scores = invalid_score.reshape(-1)
        topk = min(self.sparse_debug_topk, int(flat_scores.numel()))
        top_vals, top_idx = torch.topk(flat_scores, k=topk, largest=True, sorted=True)
        top_sites = []
        for score, flat_idx in zip(top_vals.tolist(), top_idx.tolist()):
            b = flat_idx // (self.H * self.W)
            rem = flat_idx % (self.H * self.W)
            y = rem // self.W
            x = rem % self.W
            top_sites.append(
                {
                    "batch": int(b),
                    "y": int(y),
                    "x": int(x),
                    "score": float(score),
                    "duplicate": bool(duplicate_sites[b, y, x].item()),
                    "atom_changed": bool(atom_changed_sites[b, y, x].item()),
                    "raw_atoms": raw_atom_ids[b, y, x].detach().cpu().tolist(),
                    "proj_atoms": proj_atom_ids[b, y, x].detach().cpu().tolist(),
                    "raw_coeff_norm": raw_coeff_norm[b, y, x].detach().cpu().tolist(),
                    "proc_coeffs": proc_coeffs[b, y, x].detach().cpu().tolist(),
                    "proj_coeffs": proj_coeffs[b, y, x].detach().cpu().tolist(),
                }
            )

        return {
            "summary": {
                "duplicate_rate": float(duplicate_sites.float().mean().item()),
                "atom_changed_rate": float(atom_changed_sites.float().mean().item()),
                "coeff_change_mean": float(coeff_change.mean().item()),
                "coeff_change_max": float(coeff_change.max().item()),
                "invalid_score_mean": float(invalid_score.mean().item()),
                "invalid_score_max": float(invalid_score.max().item()),
                "projection_image_l1_mean": float(image_l1.mean().item()),
                "projection_image_l1_max": float(image_l1.max().item()),
            },
            "top_sites": top_sites,
            "raw_atom_ids": raw_atom_ids.detach().cpu(),
            "raw_coeff_norm": raw_coeff_norm.detach().cpu(),
            "processed_coeffs": proc_coeffs.detach().cpu(),
            "projected_atom_ids": proj_atom_ids.detach().cpu(),
            "projected_coeffs": proj_coeffs.detach().cpu(),
            "raw_images": raw_imgs.detach().cpu(),
            "projected_images": proj_imgs.detach().cpu(),
            "abs_diff_images": image_l1.detach().cpu(),
        }

    def _save_sparse_debug_visuals(self, prefix: str, snapshot: dict):
        raw_imgs = snapshot["raw_images"].to(torch.float32)
        proj_imgs = snapshot["projected_images"].to(torch.float32)
        diff_imgs = snapshot["abs_diff_images"].to(torch.float32).clamp(0.0, 1.0)
        diff_vis = diff_imgs * 2.0 - 1.0
        nrow = max(1, min(8, int(raw_imgs.size(0))))
        save_image_grid(raw_imgs, f"{prefix}_raw.png", nrow=nrow)
        save_image_grid(proj_imgs, f"{prefix}_projected.png", nrow=nrow)
        save_image_grid(diff_vis, f"{prefix}_absdiff.png", nrow=nrow)

        compare = torch.stack([raw_imgs, proj_imgs, diff_vis], dim=1)
        compare = compare.reshape(-1, *raw_imgs.shape[1:])
        save_image_grid(compare, f"{prefix}_compare.png", nrow=3)

    def _dump_sparse_debug(self, step: int, gen, raw_imgs: torch.Tensor):
        snapshot = self._build_sparse_debug_snapshot(gen, raw_imgs)
        if snapshot is None:
            return
        os.makedirs(self.out_dir, exist_ok=True)
        prefix = os.path.join(self.out_dir, f"stage2_step{step:06d}_sparse_debug")
        torch.save(snapshot, f"{prefix}.pt")
        self._save_sparse_debug_visuals(prefix, snapshot)
        lines = [
            f"duplicate_rate={snapshot['summary']['duplicate_rate']:.6f}",
            f"atom_changed_rate={snapshot['summary']['atom_changed_rate']:.6f}",
            f"coeff_change_mean={snapshot['summary']['coeff_change_mean']:.6f}",
            f"coeff_change_max={snapshot['summary']['coeff_change_max']:.6f}",
            f"invalid_score_mean={snapshot['summary']['invalid_score_mean']:.6f}",
            f"invalid_score_max={snapshot['summary']['invalid_score_max']:.6f}",
            f"projection_image_l1_mean={snapshot['summary']['projection_image_l1_mean']:.6f}",
            f"projection_image_l1_max={snapshot['summary']['projection_image_l1_max']:.6f}",
            "",
        ]
        for idx, site in enumerate(snapshot["top_sites"]):
            lines.append(
                f"[{idx}] b={site['batch']} y={site['y']} x={site['x']} "
                f"score={site['score']:.6f} duplicate={site['duplicate']} atom_changed={site['atom_changed']}"
            )
            lines.append(f"    raw_atoms={site['raw_atoms']} proj_atoms={site['proj_atoms']}")
            lines.append(f"    raw_coeff_norm={site['raw_coeff_norm']}")
            lines.append(f"    proc_coeffs={site['proc_coeffs']}")
            lines.append(f"    proj_coeffs={site['proj_coeffs']}")
        with open(f"{prefix}.txt", "w", encoding="ascii") as f:
            f.write("\n".join(lines) + "\n")
        print(
            "[Stage2] sparse debug: "
            f"duplicate_rate={snapshot['summary']['duplicate_rate']:.4f} "
            f"atom_changed_rate={snapshot['summary']['atom_changed_rate']:.4f} "
            f"saved={prefix}.pt"
        )

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
        if self.predict_coefficients:
            tok_flat, coeff_flat = batch
            coeff_flat = coeff_flat.to(torch.float32)
        else:
            (tok_flat,) = batch
            coeff_flat = None
        tok_flat = tok_flat.long()
        B = tok_flat.size(0)
        latent_loss = torch.tensor(0.0, device=tok_flat.device, dtype=torch.float32)
        coeff_energy_loss = torch.tensor(0.0, device=tok_flat.device, dtype=torch.float32)
        pred_coeff_energy = torch.tensor(0.0, device=tok_flat.device, dtype=torch.float32)
        target_coeff_energy = torch.tensor(0.0, device=tok_flat.device, dtype=torch.float32)
        projection_consistency_loss = torch.tensor(0.0, device=tok_flat.device, dtype=torch.float32)
        projection_change_rate = torch.tensor(0.0, device=tok_flat.device, dtype=torch.float32)

        if self.is_diffusion:
            atom_ids = tok_flat.view(B, self.H * self.W, self.D)
            coeffs_3d = (coeff_flat.view(B, self.H * self.W, self.D).to(tok_flat.device) - self.coeff_mean) / self.coeff_std
            loss = self.transformer(atom_ids, coeffs_3d)
            atom_loss = loss
            self.log("train/diffusion_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=B)
        elif self.is_spatial_depth:
            atom_ids = tok_flat.view(B, self.H * self.W, self.D)
            coeffs_3d = coeff_flat.view(B, self.H * self.W, self.D).to(tok_flat.device)
            coeff_target = (coeffs_3d - self.coeff_mean) / self.coeff_std
            atom_logits, coeff_pred = self.transformer(atom_ids, coeff_target)
            vocab = self.transformer.cfg.vocab_size
            atom_loss = F.cross_entropy(
                atom_logits.reshape(-1, vocab),
                atom_ids.reshape(-1),
            )
            coeff_loss = F.mse_loss(coeff_pred.reshape(-1), coeff_target.reshape(-1))
            if self.coeff_energy_loss_weight > 0.0:
                coeff_energy_loss, pred_coeff_energy, target_coeff_energy = self._coefficient_energy_loss(
                    coeff_pred,
                    coeffs_3d,
                )
            if self.projection_consistency_weight > 0.0:
                projection_consistency_loss, projection_change_rate = self._projection_consistency_loss(
                    atom_logits,
                    coeff_pred,
                )
            if self.latent_loss_weight > 0.0:
                target_latent = self._target_code_to_latent(atom_ids, coeffs_3d).detach()
                pred_latent = self._soft_code_to_latent(atom_logits, coeff_pred)
                latent_loss = F.mse_loss(pred_latent, target_latent)
            loss = (
                atom_loss
                + self.coeff_loss_weight * coeff_loss
                + self.latent_loss_weight * latent_loss
                + self.coeff_energy_loss_weight * coeff_energy_loss
                + self.projection_consistency_weight * projection_consistency_loss
            )
            self.log("train/coeff_loss", coeff_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=B)
        else:
            bos = self.transformer.bos_token_id
            seq = torch.cat([torch.full((B, 1), bos, device=tok_flat.device, dtype=torch.long), tok_flat], dim=1)
            y = seq[:, 1:]
            x_in = seq[:, :-1]

            out = self.transformer(x_in, atom_targets=y if self.predict_coefficients else None)
            if self.predict_coefficients:
                logits, coeff_pred = out
                atom_loss = F.cross_entropy(
                    logits.reshape(-1, self.transformer.cfg.vocab_size),
                    y.reshape(-1),
                    ignore_index=self.pad_token_id,
                )
                coeff_target = (coeff_flat.to(logits.device) - self.coeff_mean) / self.coeff_std
                coeff_loss = F.mse_loss(coeff_pred.reshape(-1), coeff_target.reshape(-1))
                coeffs_3d = coeff_flat.to(logits.device).view(B, self.H * self.W, self.D)
                if self.coeff_energy_loss_weight > 0.0:
                    coeff_energy_loss, pred_coeff_energy, target_coeff_energy = self._coefficient_energy_loss(
                        coeff_pred,
                        coeffs_3d,
                    )
                if self.latent_loss_weight > 0.0:
                    atom_ids = y.view(B, self.H * self.W, self.D)
                    target_latent = self._target_code_to_latent(atom_ids, coeffs_3d).detach()
                    pred_latent = self._soft_code_to_latent(logits, coeff_pred)
                    latent_loss = F.mse_loss(pred_latent, target_latent)
                loss = (
                    atom_loss
                    + self.coeff_loss_weight * coeff_loss
                    + self.latent_loss_weight * latent_loss
                    + self.coeff_energy_loss_weight * coeff_energy_loss
                )
                self.log("train/coeff_loss", coeff_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=B)
            else:
                atom_loss = F.cross_entropy(
                    out.reshape(-1, self.transformer.cfg.vocab_size),
                    y.reshape(-1),
                    ignore_index=self.pad_token_id,
                )
                coeff_loss = torch.tensor(0.0, device=out.device, dtype=out.dtype)
                loss = atom_loss
        if self.predict_coefficients and not self.is_diffusion:
            self.log("train/latent_loss", latent_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=B)
            self.log("train/coeff_energy_loss", coeff_energy_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=B)
            self.log("train/pred_coeff_energy", pred_coeff_energy, on_step=True, on_epoch=True, sync_dist=True, batch_size=B)
            self.log("train/target_coeff_energy", target_coeff_energy, on_step=True, on_epoch=True, sync_dist=True, batch_size=B)
            self.log("train/proj_consistency_loss", projection_consistency_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=B)
            self.log("train/proj_change_rate", projection_change_rate, on_step=True, on_epoch=True, sync_dist=True, batch_size=B)
        self.log(
            "train/atom_loss",
            atom_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=B,
        )
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=B)
        return loss

    def on_fit_start(self):
        self.laser.to(self.device)
        self.laser.eval()

    @staticmethod
    def _dist_initialized() -> bool:
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.sample_every_steps <= 0:
            return
        if self.global_step <= 0 or (self.global_step % self.sample_every_steps) != 0:
            return
        if self._dist_initialized():
            torch.distributed.barrier()
            sample_error = torch.zeros(1, dtype=torch.int32, device=self.device)
            if self.trainer.is_global_zero:
                try:
                    self._log_gt_recons(batch, step=self.global_step)
                    self._sample_and_save(step=self.global_step)
                except Exception as exc:
                    sample_error.fill_(1)
                    print(f"[Stage2] sampling failed at step {self.global_step}: {exc}")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            torch.distributed.all_reduce(sample_error, op=torch.distributed.ReduceOp.MAX)
            torch.distributed.barrier()
            if int(sample_error.item()) != 0:
                raise RuntimeError(
                    f"Stage-2 sampling failed at step {self.global_step} on at least one rank."
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return
        if self.trainer.is_global_zero:
            self._log_gt_recons(batch, step=self.global_step)
            self._sample_and_save(step=self.global_step)

    @torch.no_grad()
    def _log_gt_recons(self, batch, step: int, max_imgs: int = 8):
        """Decode a training batch back to images and log as ground-truth."""
        self.laser.eval()
        if self.predict_coefficients:
            tok_flat, coeff_flat = batch
            coeff_flat = coeff_flat.to(torch.float32)
        else:
            (tok_flat,) = batch
            coeff_flat = None
        tok_flat = tok_flat[:max_imgs].long().to(self.device)

        if self.site_tokenizer is not None:
            gt_imgs = self._decode_site_token_batch(tok_flat)
        elif self.predict_coefficients:
            atom_ids = tok_flat.view(-1, self.H, self.W, self.D)
            coeffs = coeff_flat[:max_imgs].to(self.device).view(-1, self.H, self.W, self.D)
            gt_imgs = self.laser.decode(atom_ids, coeffs)
        else:
            tokens = tok_flat.view(-1, self.H, self.W, self.D)
            gt_imgs = self.laser.decode_tokens(tokens)

        if gt_imgs.size(-2) != self.image_size or gt_imgs.size(-1) != self.image_size:
            gt_imgs = F.interpolate(
                gt_imgs, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )
        logged = log_image_grid_wandb(
            self.logger, key="stage2/gt_recons", x=gt_imgs,
            step=step, caption=f"gt_recons step={step}",
        )
        if not logged:
            save_image_grid(gt_imgs, os.path.join(self.out_dir, f"stage2_step{step:06d}_gt_recons.png"))

    def on_train_epoch_end(self):
        if self._dist_initialized():
            torch.distributed.barrier()
        if self.trainer.is_global_zero:
            os.makedirs(self.out_dir, exist_ok=True)
            torch.save(self.transformer.state_dict(), os.path.join(self.out_dir, f"transformer_{self.arch_tag}_last.pt"))
        if self._dist_initialized():
            torch.distributed.barrier()

    def _decode_site_token_batch(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.site_tokenizer is None:
            raise RuntimeError("site_tokenizer is required for site-token decoding.")
        if tokens.dim() == 2:
            tokens = tokens.view(-1, self.H, self.W, self.D)
        elif tokens.dim() == 3:
            tokens = tokens.view(-1, self.H, self.W, self.D)
        atom_ids, coeffs = self.site_tokenizer.decode_tokens(tokens)
        return self.laser.decode(atom_ids.to(self.device), coeffs.to(self.device))

    def _decode_generated_batch(self, gen) -> torch.Tensor:
        if self.site_tokenizer is not None:
            return self._decode_site_token_batch(gen)
        if self.is_spatial_depth or self.is_diffusion:
            atom_ids, coeff_gen = gen
            atom_ids = atom_ids.view(-1, self.H, self.W, self.D).to(self.device)
            coeff_gen = coeff_gen.view(-1, self.H, self.W, self.D)
            coeff_gen = self._postprocess_generated_coeffs(coeff_gen)
            return self.laser.decode(atom_ids, coeff_gen)
        if self.predict_coefficients:
            flat_gen, coeff_gen = gen
            atom_ids = flat_gen.view(-1, self.H, self.W, self.D).to(self.device)
            coeff_gen = coeff_gen.view(-1, self.H, self.W, self.D)
            coeff_gen = self._postprocess_generated_coeffs(coeff_gen)
            return self.laser.decode(atom_ids, coeff_gen)
        tokens_gen = gen.view(-1, self.H, self.W, self.D)
        return self.laser.decode_tokens(tokens_gen.to(self.device))

    @torch.no_grad()
    def _sample_sliding_window_batch(self, step: int) -> torch.Tensor:
        B = self.sample_batch_size
        bos_id = int(self.transformer.bos_token_id)
        pad_id = int(self.pad_token_id)
        win_tokens = int(self.H * self.W * self.D)
        full_h = int(self.sample_latent_h)
        full_w = int(self.sample_latent_w)
        if self.H > full_h or self.W > full_w:
            raise ValueError(
                f"Sliding window {(self.H, self.W)} must fit within full latent {(full_h, full_w)}"
            )
        full_tokens = int(self.sample_latent_h * self.sample_latent_w * self.D)
        generated = torch.zeros((B, full_h, full_w, self.D), dtype=torch.long, device=self.device)
        bos = torch.full((B, 1), bos_id, dtype=torch.long, device=self.device)
        special_ids = torch.tensor([bos_id, pad_id], dtype=torch.long, device=self.device)

        coeffs_gen = None
        if self.predict_coefficients:
            coeffs_gen = torch.zeros((B, full_h, full_w, self.D), device=self.device)

        steps = tqdm(
            range(full_tokens),
            desc=f"[Stage2] sliding-window sample step {step}",
            leave=False,
            dynamic_ncols=True,
        )

        for t in steps:
            spatial_idx = t // self.D
            depth_idx = t % self.D
            y = spatial_idx // full_w
            x = spatial_idx % full_w

            h0 = max(0, min(y - self.H + 1, full_h - self.H))
            w0 = max(0, min(x - self.W + 1, full_w - self.W))
            local_y = y - h0
            local_x = x - w0
            local_pos = (local_y * self.W) + local_x
            local_t = (local_pos * self.D) + depth_idx

            window_flat = generated[:, h0 : h0 + self.H, w0 : w0 + self.W, :].contiguous().view(B, win_tokens)
            ctx = window_flat[:, :local_t]
            seq = torch.cat([bos, ctx], dim=1)

            if self.predict_coefficients:
                h = self.transformer._backbone(seq)
                atom_logits = self.transformer.atom_head(h)
                logits = atom_logits[:, -1, :]
                logits[:, special_ids] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
                generated[:, y, x, depth_idx] = nxt.squeeze(1)

                coeff_val = self.transformer._predict_coeff(h[:, -1:, :], nxt)
                coeff_val = soft_clamp(coeff_val.squeeze(-1), self.transformer.cfg.coeff_max)
                coeffs_gen[:, y, x, depth_idx] = coeff_val
            else:
                logits = self.transformer(seq)[:, -1, :]
                logits[:, special_ids] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1).squeeze(1)
                generated[:, y, x, depth_idx] = nxt

        if self.predict_coefficients:
            atom_ids = generated.to(self.device)
            coeffs_gen = self._postprocess_generated_coeffs(coeffs_gen)
            return self.laser.decode(atom_ids, coeffs_gen)
        return self.laser.decode_tokens(generated.to(self.device))

    @torch.no_grad()
    def _project_generated_site(
        self,
        atom_ids_site: torch.Tensor,
        coeffs_norm_site: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = int(atom_ids_site.size(0))
        atom_ids_site = atom_ids_site.view(B, 1, 1, self.D).to(self.device)
        coeffs_norm_site = coeffs_norm_site.view(B, 1, 1, self.D).to(self.device)
        coeffs_raw_site = self._denormalize_coeffs(
            coeffs_norm_site,
            apply_spatial_smoothing=False,
        )
        proj_atom_ids, proj_coeffs_raw = self.laser.project_codes(atom_ids_site, coeffs_raw_site)
        proj_atom_ids = proj_atom_ids.view(B, 1, self.D)
        proj_coeffs_norm = (proj_coeffs_raw.view(B, 1, self.D) - self.coeff_mean) / self.coeff_std
        return proj_atom_ids, proj_coeffs_norm

    @torch.no_grad()
    def _sample_spatial_depth_batch(
        self,
        step: int,
        capture_raw: bool = False,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size = int(self.sample_batch_size)
        device = self.device
        cfg = self.transformer.cfg
        T = int(cfg.H * cfg.W)
        D = int(cfg.D)

        atom_ids = torch.zeros(batch_size, T, D, dtype=torch.long, device=device)
        coeffs_norm = torch.zeros(batch_size, T, D, device=device)
        raw_atom_ids = torch.zeros_like(atom_ids) if capture_raw else None
        raw_coeffs_norm = torch.zeros_like(coeffs_norm) if capture_raw else None

        spatial_kv: list[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * len(self.transformer.spatial_blocks)
        steps = tqdm(
            range(T),
            desc=f"[Stage2] sample step {step}",
            leave=False,
            dynamic_ncols=True,
        )

        for t in steps:
            if t == 0:
                x_new = self.transformer.start_emb.expand(batch_size, -1, -1)
            else:
                prev_emb = (
                    self.transformer.token_emb(atom_ids[:, t - 1])
                    + self.transformer.coeff_proj(coeffs_norm[:, t - 1].unsqueeze(-1))
                )
                x_new = self.transformer.spatial_fuse(prev_emb.reshape(batch_size, -1)).unsqueeze(1)

            x_new = x_new + self.transformer.row_emb(self.transformer._rows[t]) + self.transformer.col_emb(self.transformer._cols[t])

            spatial_h = x_new
            for i, blk in enumerate(self.transformer.spatial_blocks):
                spatial_h, spatial_kv[i] = blk(spatial_h, kv_cache=spatial_kv[i])
            h_t = self.transformer.spatial_ln(spatial_h).squeeze(1)

            depth_seq: list[torch.Tensor] = []
            for d in range(D):
                if d == 0:
                    step_in = h_t.unsqueeze(1)
                else:
                    step_in = (
                        self.transformer.token_emb(atom_ids[:, t, d - 1]).unsqueeze(1)
                        + self.transformer.coeff_proj(coeffs_norm[:, t, d - 1].view(batch_size, 1, 1))
                    )
                step_in = step_in + self.transformer.depth_emb.weight[d]
                depth_seq.append(step_in)

                depth_h = torch.cat(depth_seq, dim=1)
                for blk in self.transformer.depth_blocks:
                    depth_h, _ = blk(depth_h)
                depth_h = self.transformer.depth_ln(depth_h)
                last_h = depth_h[:, -1]

                logits = self.transformer.atom_head(last_h)
                if d > 0:
                    prev_atoms = atom_ids[:, t, :d]
                    logits.scatter_(1, prev_atoms, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                sampled = torch.multinomial(probs, 1).squeeze(-1)
                atom_ids[:, t, d] = sampled

                coeff_emb = self.transformer.atom_coeff_emb(sampled)
                coeff_in = torch.cat([last_h, coeff_emb], dim=-1)
                coeff_val = self.transformer.coeff_head(coeff_in).squeeze(-1)
                coeffs_norm[:, t, d] = soft_clamp(coeff_val, cfg.coeff_max)

            if capture_raw:
                raw_atom_ids[:, t] = atom_ids[:, t]
                raw_coeffs_norm[:, t] = coeffs_norm[:, t]

            proj_atom_ids, proj_coeffs_norm = self._project_generated_site(
                atom_ids[:, t:t + 1],
                coeffs_norm[:, t:t + 1],
            )
            atom_ids[:, t:t + 1] = proj_atom_ids
            coeffs_norm[:, t:t + 1] = proj_coeffs_norm

        gen = (atom_ids, coeffs_norm)
        raw_gen = (raw_atom_ids, raw_coeffs_norm) if capture_raw else None
        return gen, raw_gen

    @torch.no_grad()
    def _sample_and_save(self, step: int):
        self.transformer.eval()
        self.laser.eval()
        gen = None
        print(
            f"[Stage2] sampling at step {step} "
            f"(batch_size={self.sample_batch_size}, output_size={self.image_size}x{self.image_size})..."
        )
        if self.use_sliding_window_sampling:
            if self.is_spatial_depth or self.is_diffusion:
                raise RuntimeError(
                    "Sliding-window sampling currently supports only stage2_arch=flat. "
                    "Use --stage2_arch flat or disable --stage2_sliding_window."
                )
            print(
                f"[Stage2] latent sliding-window sampling enabled "
                f"(window={self.H}x{self.W}, full_latent={self.sample_latent_h}x{self.sample_latent_w})"
            )
            raw_imgs = self._sample_sliding_window_batch(step=step)
        else:
            debug_gen = None
            if self.is_spatial_depth:
                gen, debug_gen = self._sample_spatial_depth_batch(
                    step=step,
                    capture_raw=self.dump_sparse_debug,
                )
            else:
                gen_kwargs = dict(batch_size=self.sample_batch_size, show_progress=True)
                if not self.is_diffusion:
                    gen_kwargs["progress_desc"] = f"[Stage2] sample step {step}"
                gen = self.transformer.generate(**gen_kwargs)
            raw_imgs = self._decode_generated_batch(gen)
            if self.dump_sparse_debug and self.predict_coefficients:
                if debug_gen is not None:
                    debug_raw_imgs = self._decode_generated_batch(debug_gen)
                    self._dump_sparse_debug(step=step, gen=debug_gen, raw_imgs=debug_raw_imgs)
                else:
                    self._dump_sparse_debug(step=step, gen=gen, raw_imgs=raw_imgs)
        sample_imgs = raw_imgs
        if sample_imgs.size(-2) != self.image_size or sample_imgs.size(-1) != self.image_size:
            sample_imgs = F.interpolate(
                sample_imgs,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        print(f"[Stage2] sample tensor shape: {tuple(sample_imgs.shape)}")
        logged = log_image_grid_wandb(
            self.logger,
            key="stage2/samples",
            x=sample_imgs,
            step=step,
            caption=f"step={step}",
        )
        if not logged:
            save_image_grid(sample_imgs, os.path.join(self.out_dir, f"stage2_step{step:06d}_samples.png"))
        self._compute_and_log_fid(step=step, initial_fake_imgs=raw_imgs)
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
        if n < 2:
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
        self.laser.eval()

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
    ae: LASER,
    loader: DataLoader,
    device: torch.device,
    max_items: Optional[int] = None,
)-> Tuple[torch.Tensor, Optional[torch.Tensor], int, int, int]:
    """
    Encode dataset to tokens for stage-2 training.
    Returns:
      tokens_flat: [N, H*W*D] int32
      coeffs_flat: [N, H*W*D] float32 (None if quantized)
      H, W, D
    """
    ae.eval()
    all_tokens = []
    all_coeffs = []
    seen = 0
    H = W = D = None

    for x, _ in tqdm(loader, desc="[Stage2] precompute tokens"):
        x = x.to(device)
        if ae.bottleneck.quantize_sparse_coeffs:
            tokens, h, w = ae.encode_tokens(x)
            coeffs = None
        else:
            tokens, coeffs, h, w = ae.encode(x)
        if H is None:
            H, W = h, w
            D = tokens.shape[-1]
        flat = tokens.view(tokens.size(0), -1).to(torch.int32).cpu()
        all_tokens.append(flat)
        if coeffs is not None:
            all_coeffs.append(coeffs.view(coeffs.size(0), -1).to(torch.float32).cpu())
        seen += flat.size(0)
        if max_items is not None and seen >= max_items:
            break

    tokens_flat = torch.cat(all_tokens, dim=0)
    if len(all_coeffs) > 0:
        coeffs_flat = torch.cat(all_coeffs, dim=0)
    else:
        coeffs_flat = None
    if max_items is not None:
        tokens_flat = tokens_flat[:max_items]
        if coeffs_flat is not None:
            coeffs_flat = coeffs_flat[:max_items]
    return tokens_flat, coeffs_flat, H, W, D


@torch.no_grad()
def maybe_build_site_tokenization(
    cache: Dict[str, object],
    ae: LASER,
    coeff_bins: int,
    vocab_size: int,
) -> Tuple[torch.Tensor, SparseSiteTokenizer, int, int, int, Dict[str, object]]:
    cached_bins = cache.get("site_token_bins")
    cached_vocab = cache.get("site_vocab_size")
    cached_state = cache.get("site_tokenizer_state")
    cached_tokens = cache.get("site_tokens_flat")
    shape = cache["shape"]
    run_H, run_W, D = int(shape[0]), int(shape[1]), int(shape[2])
    full_latent_shape = cache.get("full_latent_shape") or cache.get("sample_latent_shape")
    if full_latent_shape is None:
        full_H, full_W = run_H, run_W
    else:
        full_H, full_W = int(full_latent_shape[0]), int(full_latent_shape[1])
    if (
        cached_tokens is not None
        and cached_state is not None
        and int(cached_bins) == int(coeff_bins)
        and int(cached_vocab) == int(vocab_size)
    ):
        tokenizer = SparseSiteTokenizer.from_state_dict(cached_state)
        return cached_tokens, tokenizer, run_H, run_W, 1, cache

    slot_tokens_flat = cache["tokens_flat"]
    coeffs_flat = cache.get("coeffs_flat")
    site_tokens_flat, tokenizer, oov_rate = build_sparse_site_tokenizer(
        tokens_flat=slot_tokens_flat,
        coeffs_flat=coeffs_flat,
        H=full_H,
        W=full_W,
        D=D,
        num_atoms=int(ae.bottleneck.num_embeddings),
        coeff_bins=coeff_bins,
        coeff_max=float(ae.bottleneck.coef_max),
        coeff_quantization=str(ae.bottleneck.coef_quantization),
        coeff_mu=float(ae.bottleneck.coef_mu),
        vocab_size=vocab_size,
    )
    print(
        f"[Stage2] site tokenization built: vocab={tokenizer.num_site_tokens} "
        f"coeff_bins={coeff_bins} oov_rate={oov_rate:.4f}"
    )
    cache = dict(cache)
    cache["site_tokens_flat"] = site_tokens_flat
    cache["site_tokenizer_state"] = tokenizer.state_dict()
    cache["site_token_bins"] = int(coeff_bins)
    cache["site_vocab_size"] = int(vocab_size)
    cache["site_shape"] = (full_H, full_W, 1)
    cache["site_oov_rate"] = float(oov_rate)
    return site_tokens_flat, tokenizer, run_H, run_W, 1, cache


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "celeba"])
    parser.add_argument("--data_dir", type=str, default=None, help="Root directory for dataset files.")
    parser.add_argument("--image_size", type=int, default=32, help="Resize images to this square size.")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)

    # Stage-1 (AE)
    parser.add_argument("--stage1_epochs", type=int, default=5)
    parser.add_argument("--stage1_lr", type=float, default=2e-4)
    parser.add_argument("--stage1_lr_schedule", type=str, default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--stage1_warmup_epochs", type=int, default=1)
    parser.add_argument("--stage1_min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--bottleneck_weight", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=128)  # stage 1 batch size
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--stage1_devices", type=int, default=2, help="Number of GPUs for Lightning stage-1 training.")
    parser.add_argument("--stage1_precision", type=str, default="32-true", help="Lightning precision for stage-1.")
    parser.add_argument("--stage1_strategy", type=str, default="ddp", choices=["ddp", "auto"])
    parser.add_argument(
        "--stage1_val_vis_batch_size",
        type=int,
        default=32,
        help="Number of full-resolution validation images logged each stage-1 epoch.",
    )

    # Model sizes
    parser.add_argument("--num_hiddens", type=int, default=128)
    parser.add_argument(
        "--ae_num_downsamples",
        type=int,
        default=2,
        help="Number of stride-2 downsampling stages in stage-1 AE.",
    )
    parser.add_argument("--num_res_layers", type=int, default=2)
    parser.add_argument("--num_res_hiddens", type=int, default=32)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_atoms", type=int, default=128)      # dictionary size K
    parser.add_argument("--sparsity_level", type=int, default=3)   # stack depth D
    parser.add_argument("--latent_patch_size", type=int, default=1, help="Patch size for dictionary learning in latent space (e.g. 4 for 4x4 patches).")
    parser.add_argument("--n_bins", type=int, default=1024, help="Coefficient quantization bins (higher = lower quantization error, larger vocab).")
    parser.add_argument("--coef_max", type=float, default=24.0, help="Coefficient clipping range for quantization in [-coef_max, coef_max].")
    parser.add_argument("--coef_quantization", type=str, default="mu_law", choices=["uniform", "mu_law"])
    parser.add_argument("--coef_mu", type=float, default=255.0, help="Mu for mu-law quantization (only used when coef_quantization=mu_law).")
    parser.add_argument("--commitment_cost", type=float, default=0.25)
    parser.add_argument(
        "--quantize_sparse_coeffs",
        action="store_true",
        default=False,
        help="Quantize sparse coefficients into token IDs.",
    )
    parser.add_argument(
        "--no_quantize_sparse_coeffs",
        action="store_false",
        dest="quantize_sparse_coeffs",
        help="Disable quantized coefficients and use a coefficient regressor head.",
    )

    # Stage-2 (Transformer)
    parser.add_argument("--stage2_epochs", type=int, default=10)
    parser.add_argument("--stage2_lr", type=float, default=1e-4)
    parser.add_argument("--stage2_lr_schedule", type=str, default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--stage2_warmup_epochs", type=int, default=1)
    parser.add_argument("--stage2_min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--stage2_batch_size", type=int, default=64)
    parser.add_argument("--stage2_sample_every_steps", type=int, default=200)
    parser.add_argument("--stage2_sample_batch_size", type=int, default=8)
    parser.add_argument(
        "--stage2_sliding_window",
        action="store_true",
        default=False,
        help="Train stage-2 on full-image latent sequences with a fixed sliding attention window.",
    )
    parser.add_argument(
        "--stage2_window_latent_h",
        type=int,
        default=16,
        help="Latent window height for stage-2 sliding-window mode.",
    )
    parser.add_argument(
        "--stage2_window_latent_w",
        type=int,
        default=16,
        help="Latent window width for stage-2 sliding-window mode.",
    )
    parser.add_argument(
        "--stage2_window_stride_latent",
        type=int,
        default=16,
        help="Stride in latent spatial positions between training windows in sliding-window mode.",
    )
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
    parser.add_argument("--stage2_arch", type=str, default="flat", choices=["flat", "spatial_depth", "diffusion", "site_flat"],
                        help="Prior architecture: flat, spatial_depth, diffusion, or site_flat.")
    parser.add_argument("--tf_d_model", type=int, default=256)
    parser.add_argument("--tf_heads", type=int, default=8)
    parser.add_argument("--tf_layers", type=int, default=6)
    parser.add_argument("--tf_depth_layers", type=int, default=4, help="Depth-stage layers (spatial_depth arch only).")
    parser.add_argument("--tf_ff", type=int, default=1024)
    parser.add_argument("--tf_dropout", type=float, default=0.1)
    parser.add_argument(
        "--stage2_coeff_loss_weight",
        type=float,
        default=1.0,
        help="Coefficient regression loss weight when using continuous coefficients.",
    )
    parser.add_argument(
        "--stage2_latent_loss_weight",
        type=float,
        default=0.25,
        help="Latent reconstruction loss weight for continuous sparse-code priors.",
    )
    parser.add_argument(
        "--stage2_coeff_energy_loss_weight",
        type=float,
        default=0.25,
        help="Per-site coefficient energy loss weight for continuous sparse-code priors.",
    )
    parser.add_argument(
        "--stage2_projection_consistency_weight",
        type=float,
        default=0.5,
        help="Self-projection consistency loss weight for stage2 sparse tuple predictions.",
    )
    parser.add_argument(
        "--stage2_projection_consistency_sites",
        type=int,
        default=256,
        help="Number of spatial sites per batch used for stage2 self-projection consistency.",
    )
    parser.add_argument(
        "--stage2_site_token_bins",
        type=int,
        default=16,
        help="Coefficient bins per sparse slot when packing a whole latent site into one stage-2 token.",
    )
    parser.add_argument(
        "--stage2_site_vocab_size",
        type=int,
        default=8192,
        help="Vocabulary size for whole-site sparse tuple tokens.",
    )
    parser.add_argument(
        "--stage2_site_max_oov_rate",
        type=float,
        default=0.2,
        help="Maximum tolerated OOV rate for whole-site tokenization before aborting stage-2.",
    )
    parser.add_argument(
        "--stage2_coeff_norm_clip",
        type=float,
        default=3.0,
        help="Clamp generated normalized coefficients to [-clip, clip] before unnormalizing.",
    )
    parser.add_argument(
        "--stage2_dump_sparse_debug",
        action="store_true",
        default=False,
        help="Dump generated sparse atom/coeff diagnostics during stage-2 sampling.",
    )
    parser.add_argument(
        "--stage2_sparse_debug_topk",
        type=int,
        default=16,
        help="Number of worst sparse-code spatial sites to include in stage-2 debug dumps.",
    )
    parser.add_argument("--token_subset", type=int, default=0, help="Use only first N tokens/images (0 = use all).")
    parser.add_argument("--token_num_workers", type=int, default=0, help="Workers for stage-2 token precompute loader.")
    parser.add_argument("--force_retokenize", action="store_true", help="Recompute tokens even if cache exists.")
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
    if args.ae_num_downsamples <= 0:
        raise ValueError(f"ae_num_downsamples must be positive, got {args.ae_num_downsamples}")
    args.image_size = int(args.image_size)
    if args.image_size <= 0:
        raise ValueError(f"image_size must be positive, got {args.image_size}")
    if args.stage2_sliding_window:
        if args.stage2_window_latent_h <= 0 or args.stage2_window_latent_w <= 0:
            raise ValueError("stage2_window_latent_h and stage2_window_latent_w must be positive.")
        if args.stage2_window_stride_latent <= 0:
            raise ValueError("stage2_window_stride_latent must be positive.")
    if args.stage2_arch in {"spatial_depth", "diffusion", "site_flat"} and args.quantize_sparse_coeffs:
        raise ValueError(
            f"stage2_arch={args.stage2_arch} requires continuous atom/coeff pairs. "
            "Use --no_quantize_sparse_coeffs."
        )
    if args.stage2_coeff_norm_clip <= 0:
        raise ValueError(f"stage2_coeff_norm_clip must be positive, got {args.stage2_coeff_norm_clip}")
    if args.stage2_coeff_energy_loss_weight < 0:
        raise ValueError(
            "stage2_coeff_energy_loss_weight must be non-negative, "
            f"got {args.stage2_coeff_energy_loss_weight}"
        )
    if args.stage2_projection_consistency_weight < 0:
        raise ValueError(
            "stage2_projection_consistency_weight must be non-negative, "
            f"got {args.stage2_projection_consistency_weight}"
        )
    if args.stage2_projection_consistency_sites <= 0:
        raise ValueError(
            "stage2_projection_consistency_sites must be positive, "
            f"got {args.stage2_projection_consistency_sites}"
        )
    if args.stage2_site_token_bins <= 1:
        raise ValueError(
            f"stage2_site_token_bins must be > 1, got {args.stage2_site_token_bins}"
        )
    if args.stage2_site_vocab_size <= 0:
        raise ValueError(
            f"stage2_site_vocab_size must be positive, got {args.stage2_site_vocab_size}"
        )
    if not (0.0 <= args.stage2_site_max_oov_rate < 1.0):
        raise ValueError(
            f"stage2_site_max_oov_rate must be in [0, 1), got {args.stage2_site_max_oov_rate}"
        )
    if args.stage2_sparse_debug_topk <= 0:
        raise ValueError(
            f"stage2_sparse_debug_topk must be positive, got {args.stage2_sparse_debug_topk}"
        )
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
    print(
        f"[Data] dataset={args.dataset} data_dir={args.data_dir} "
        f"image_size={args.image_size} "
        f"stage2_sliding_window={args.stage2_sliding_window}"
    )

    stage1_dir = os.path.join(args.out_dir, "stage1")
    stage2_dir = os.path.join(args.out_dir, "stage2")
    os.makedirs(stage1_dir, exist_ok=True)
    os.makedirs(stage2_dir, exist_ok=True)
    token_cache_path = os.path.join(stage2_dir, "tokens_cache.pt")
    run_base_name = args.wandb_name or f"sparse_dict_rq_{args.dataset}_{args.image_size}"
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

    def _build_trainer_plugins():
        """
        In single-task SLURM jobs, Lightning may auto-detect SLURM and lock world size
        to 1. Force the default Lightning environment so `devices>1` can spawn local
        child ranks inside this single task.
        """
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        slurm_ntasks = os.environ.get("SLURM_NTASKS")
        try:
            slurm_ntasks_int = int(slurm_ntasks) if slurm_ntasks is not None else None
        except ValueError:
            slurm_ntasks_int = None
        if slurm_job_id and slurm_ntasks_int == 1:
            try:
                from lightning.pytorch.plugins.environments import LightningEnvironment
                print(
                    "[DDP] Detected SLURM with ntasks=1; forcing LightningEnvironment "
                    "for local multi-GPU spawn."
                )
                return [LightningEnvironment()]
            except Exception as exc:
                print(f"[DDP] failed to configure LightningEnvironment plugin ({exc}); using default plugins.")
        return None

    trainer_plugins = _build_trainer_plugins()

    def _build_ae(quantize_sparse_coeffs: bool = args.quantize_sparse_coeffs) -> LASER:
        return LASER(
            in_channels=3,
            num_hiddens=args.num_hiddens,
            num_downsamples=args.ae_num_downsamples,
            num_residual_layers=args.num_res_layers,
            num_residual_hiddens=args.num_res_hiddens,
            embedding_dim=args.embedding_dim,
            num_embeddings=args.num_atoms,
            sparsity_level=args.sparsity_level,
            commitment_cost=args.commitment_cost,
            n_bins=args.n_bins,
            coef_max=args.coef_max,
            coef_quantization=args.coef_quantization,
            coef_mu=args.coef_mu,
            quantize_sparse_coeffs=quantize_sparse_coeffs,
            out_tanh=True,
            latent_patch_size=args.latent_patch_size,
        )

    def _load_checkpoint(
        model: nn.Module,
        candidates: list[str],
        label: str,
        required: bool = False,
    ) -> Optional[str]:
        last_error = None
        for ckpt_path in candidates:
            if not os.path.exists(ckpt_path):
                continue
            try:
                try:
                    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                except TypeError:
                    state_dict = torch.load(ckpt_path, map_location="cpu")
                model.load_state_dict(state_dict)
                print(f"[{label}] loaded checkpoint: {ckpt_path}")
                return ckpt_path
            except Exception as exc:
                last_error = exc
                print(f"[{label}] failed to load checkpoint {ckpt_path}: {exc}")
                continue

        if required:
            tried = ", ".join(candidates)
            if last_error is not None:
                raise RuntimeError(
                    f"{label} checkpoint not loadable. Tried: {tried}. Last error: {last_error}"
                )
            raise FileNotFoundError(f"{label} checkpoint not found. Tried: {tried}")
        return None

    def _load_laser_weights(
        laser_model: LASER,
        required: bool = False,
        prefer_best: bool = False,
    ) -> Optional[str]:
        best_path = os.path.join(stage1_dir, "ae_best.pt")
        last_path = os.path.join(stage1_dir, "ae_last.pt")
        candidates = [best_path, last_path] if prefer_best else [last_path, best_path]
        return _load_checkpoint(
            model=laser_model,
            candidates=candidates,
            label="Stage1",
            required=required,
        )

    def _load_transformer_weights(transformer_model, arch: str = "flat") -> Optional[str]:
        last_path = os.path.join(stage2_dir, f"transformer_{arch}_last.pt")
        best_path = os.path.join(stage2_dir, f"transformer_{arch}_best.pt")
        return _load_checkpoint(
            model=transformer_model,
            candidates=[last_path, best_path],
            label="Stage2",
            required=False,
        )

    def _prepare_stage2_token_data(
        cache: Dict[str, object],
        ae_model: LASER,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, int, int, Optional[SparseSiteTokenizer], Dict[str, object]]:
        tokens_flat = cache["tokens_flat"]
        coeffs_flat = cache.get("coeffs_flat")
        H, W, D = cache["shape"]
        if args.stage2_arch != "site_flat":
            return tokens_flat, coeffs_flat, int(H), int(W), int(D), None, cache
        site_tokens_flat, site_tokenizer, site_H, site_W, site_D, cache = maybe_build_site_tokenization(
            cache=cache,
            ae=ae_model,
            coeff_bins=args.stage2_site_token_bins,
            vocab_size=args.stage2_site_vocab_size,
        )
        site_oov_rate = float(cache.get("site_oov_rate", 0.0))
        if site_oov_rate > args.stage2_site_max_oov_rate:
            raise RuntimeError(
                "site_flat tokenization is invalid for this run: "
                f"site_oov_rate={site_oov_rate:.4f} exceeds "
                f"stage2_site_max_oov_rate={args.stage2_site_max_oov_rate:.4f}. "
                "This means the whole-site vocabulary is collapsing most latent sites "
                "to fallback token 0, so stage2/gt_recons are not trustworthy. "
                "Increase --stage2_site_vocab_size substantially or use a different "
                "site-level tokenization scheme."
            )
        return site_tokens_flat, None, site_H, site_W, site_D, site_tokenizer, cache

    def _run_stage2(
        tokens_flat: torch.Tensor,
        coeffs_flat: Optional[torch.Tensor],
        quantize_sparse_coeffs: bool,
        H: int,
        W: int,
        D: int,
        site_tokenizer: Optional[SparseSiteTokenizer],
        sample_latent_shape: Optional[Tuple[int, int]],
        sliding_window_stride_latent: Optional[int],
        ae_model: LASER,
        fid_real_images: Optional[torch.Tensor],
        coeff_mean: float = 0.0,
        coeff_std: float = 1.0,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("Stage-2 Lightning multi-GPU training requires CUDA.")
        if torch.cuda.device_count() < args.stage2_devices:
            raise RuntimeError(
                f"Requested {args.stage2_devices} GPUs, but only {torch.cuda.device_count()} detected."
            )
        if not quantize_sparse_coeffs and coeffs_flat is None and site_tokenizer is None:
            raise ValueError("coeffs_flat is required when quantize_sparse_coeffs=False")

        dm = TokenDataModule(
            tokens_flat=tokens_flat,
            batch_size=args.stage2_batch_size,
            coeffs_flat=None if coeffs_flat is None else coeffs_flat,
            sliding_window_shape=((int(H), int(W)) if sample_latent_shape is not None else None),
            full_latent_shape=sample_latent_shape,
            latent_depth=(int(D) if sample_latent_shape is not None else None),
            sliding_window_stride_latent=(
                int(sliding_window_stride_latent)
                if (sample_latent_shape is not None and sliding_window_stride_latent is not None)
                else None
            ),
            num_workers=2,
        )

        if args.stage2_arch == "spatial_depth":
            from laser_transformer import SpatialDepthPrior, SpatialDepthPriorConfig
            sd_cfg = SpatialDepthPriorConfig(
                vocab_size=ae_model.bottleneck.num_embeddings,
                H=H,
                W=W,
                D=D,
                d_model=args.tf_d_model,
                n_heads=args.tf_heads,
                n_spatial_layers=args.tf_layers,
                n_depth_layers=args.tf_depth_layers,
                d_ff=args.tf_ff,
                dropout=args.tf_dropout,
                coeff_max=args.stage2_coeff_norm_clip,
            )
            transformer = SpatialDepthPrior(sd_cfg)
        elif args.stage2_arch == "diffusion":
            from laser_diffusion_prior import DiffusionPrior, DiffusionPriorConfig
            diff_cfg = DiffusionPriorConfig(
                vocab_size=ae_model.bottleneck.vocab_size,
                H=H,
                W=W,
                D=D,
                d_model=args.tf_d_model,
                n_heads=args.tf_heads,
                n_layers=args.tf_layers,
                d_ff=args.tf_ff,
                dropout=args.tf_dropout,
                coeff_max=args.stage2_coeff_norm_clip,
            )
            transformer = DiffusionPrior(diff_cfg)
        else:
            if args.stage2_arch == "site_flat":
                if site_tokenizer is None:
                    raise ValueError("site_tokenizer is required for stage2_arch=site_flat")
                vocab_size = site_tokenizer.vocab_size
                predict_coefficients = False
                bos_token_id = site_tokenizer.bos_token_id
                pad_token_id = site_tokenizer.pad_token_id
            else:
                vocab_size = ae_model.bottleneck.vocab_size
                predict_coefficients = not quantize_sparse_coeffs
                bos_token_id = ae_model.bottleneck.bos_token_id
                pad_token_id = ae_model.bottleneck.pad_token_id
            cfg = PriorConfig(
                vocab_size=vocab_size,
                H=H,
                W=W,
                D=D,
                predict_coefficients=predict_coefficients,
                coeff_loss_weight=args.stage2_coeff_loss_weight,
                coeff_max=args.stage2_coeff_norm_clip,
                d_model=args.tf_d_model,
                n_heads=args.tf_heads,
                n_layers=args.tf_layers,
                d_ff=args.tf_ff,
                dropout=args.tf_dropout,
            )
            transformer = Prior(
                cfg,
                bos_token_id=bos_token_id,
                pad_token_id=pad_token_id,
            )
        _load_transformer_weights(transformer, arch=args.stage2_arch)
        module = Stage2Module(
            transformer=transformer,
            lr=args.stage2_lr,
            pad_token_id=(site_tokenizer.pad_token_id if site_tokenizer is not None else ae_model.bottleneck.pad_token_id),
            out_dir=stage2_dir,
            laser=ae_model,
            H=H,
            W=W,
            D=D,
            image_size=args.image_size,
            sample_every_steps=args.stage2_sample_every_steps,
            sample_batch_size=args.stage2_sample_batch_size,
            sample_latent_shape=sample_latent_shape,
            sample_window_stride_latent=(
                int(sliding_window_stride_latent)
                if (sample_latent_shape is not None and sliding_window_stride_latent is not None)
                else 1
            ),
            fid_real_images=fid_real_images,
            fid_num_samples=args.stage2_fid_num_samples,
            fid_feature=args.stage2_fid_feature,
            fid_every_n_epochs=args.stage2_fid_every_n_epochs,
            lr_schedule=args.stage2_lr_schedule,
            warmup_epochs=args.stage2_warmup_epochs,
            min_lr_ratio=args.stage2_min_lr_ratio,
            coeff_loss_weight=args.stage2_coeff_loss_weight,
            coeff_mean=coeff_mean,
            coeff_std=coeff_std,
            coeff_norm_clip=args.stage2_coeff_norm_clip,
            latent_loss_weight=args.stage2_latent_loss_weight,
            coeff_energy_loss_weight=args.stage2_coeff_energy_loss_weight,
            projection_consistency_weight=args.stage2_projection_consistency_weight,
            projection_consistency_sites=args.stage2_projection_consistency_sites,
            site_tokenizer=site_tokenizer,
            arch_tag=args.stage2_arch,
            dump_sparse_debug=args.stage2_dump_sparse_debug,
            sparse_debug_topk=args.stage2_sparse_debug_topk,
        )

        if args.stage2_devices > 1:
            from lightning.pytorch.strategies import DDPStrategy
            needs_find_unused = args.stage2_arch == "diffusion"
            effective_strategy = DDPStrategy(broadcast_buffers=False, find_unused_parameters=needs_find_unused)
        else:
            effective_strategy = "auto"

        trainer = pl.Trainer(
            accelerator="gpu",
            devices=args.stage2_devices,
            strategy=effective_strategy,
            max_epochs=args.stage2_epochs,
            logger=_build_wandb_logger("stage2"),
            plugins=trainer_plugins,
            enable_checkpointing=False,
            gradient_clip_val=1.0,
            precision=args.stage2_precision,
            log_every_n_steps=10,
            deterministic=False,
        )
        trainer.fit(module, datamodule=dm)

    # During stage-2 DDP script re-entry, skip stage-1 and tokenization work.
    if os.environ.get("LASER_DDP_PHASE") == "stage2":
        if not os.path.exists(token_cache_path):
            raise FileNotFoundError(f"Missing token cache: {token_cache_path}")
        try:
            cache = torch.load(token_cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            cache = torch.load(token_cache_path, map_location="cpu")
        tokens_flat = cache["tokens_flat"]
        coeffs_flat = cache.get("coeffs_flat")
        quantize_sparse_coeffs = cache.get("quantize_sparse_coeffs", True)
        H, W, D = cache["shape"]
        sample_latent_shape = cache.get("sample_latent_shape")
        sliding_window_stride_latent = cache.get(
            "sliding_window_stride_latent",
            cache.get("sliding_window_stride_tokens"),
        )
        fid_real_images = cache.get("fid_real_images")
        ae = _build_ae(quantize_sparse_coeffs=quantize_sparse_coeffs)
        _load_laser_weights(ae, required=True, prefer_best=True)
        tokens_flat, coeffs_flat, H, W, D, site_tokenizer, cache = _prepare_stage2_token_data(cache, ae)
        _coeff_mean, _coeff_std = 0.0, 1.0
        if coeffs_flat is not None:
            _coeff_mean = float(coeffs_flat.mean().item())
            _coeff_std = float(coeffs_flat.std().item())
        _run_stage2(
            tokens_flat=tokens_flat,
            coeffs_flat=coeffs_flat,
            quantize_sparse_coeffs=quantize_sparse_coeffs,
            H=H,
            W=W,
            D=D,
            site_tokenizer=site_tokenizer,
            sample_latent_shape=sample_latent_shape,
            sliding_window_stride_latent=sliding_window_stride_latent,
            ae_model=ae,
            fid_real_images=fid_real_images,
            coeff_mean=_coeff_mean,
            coeff_std=_coeff_std,
        )
        return

    # Normalize to [-1, 1], with a single square resize.
    train_tfm = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    eval_tfm = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    val_vis_tfm = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    stage2_source_tfm = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    if args.dataset == "cifar10":
        train_set = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=train_tfm)
        val_set = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=eval_tfm)
        val_vis_set = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=val_vis_tfm)
        if args.stage2_sliding_window:
            stage2_source_set = datasets.CIFAR10(
                root=args.data_dir,
                train=True,
                download=True,
                transform=stage2_source_tfm,
            )
        else:
            stage2_source_set = train_set
    elif args.dataset == "celeba":
        print(f"[Data] building FlatImageDataset from: {args.data_dir}")
        train_full = FlatImageDataset(root=args.data_dir, transform=train_tfm)
        val_full = FlatImageDataset(root=args.data_dir, transform=eval_tfm)
        val_vis_full = FlatImageDataset(root=args.data_dir, transform=val_vis_tfm)
        stage2_source_full = FlatImageDataset(root=args.data_dir, transform=stage2_source_tfm)
        if len(train_full) < 2:
            raise RuntimeError("CelebA dataset needs at least 2 images for train/val split.")
        val_size = max(1, int(0.05 * len(train_full)))
        train_size = len(train_full) - val_size
        all_indices = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(args.seed))
        train_indices = all_indices[:train_size].tolist()
        val_indices = all_indices[train_size:].tolist()
        train_set = Subset(train_full, train_indices)
        val_set = Subset(val_full, val_indices)
        val_vis_set = Subset(val_vis_full, val_indices)
        if args.stage2_sliding_window:
            stage2_source_set = Subset(stage2_source_full, train_indices)
        else:
            stage2_source_set = train_set
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    print(f"[Data] dataset objects ready: train={len(train_set)} val={len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    stage1_val_vis_images = None
    if args.stage1_val_vis_batch_size > 0 and len(val_vis_set) > 0:
        vis_bs = min(int(args.stage1_val_vis_batch_size), len(val_vis_set))
        vis_loader = DataLoader(val_vis_set, batch_size=vis_bs, shuffle=False, num_workers=0, pin_memory=False)
        vis_images, _ = next(iter(vis_loader))
        stage1_val_vis_images = vis_images
        print(
            f"[Stage1] full-res validation visuals: "
            f"batch={stage1_val_vis_images.size(0)} size={stage1_val_vis_images.size(-2)}x{stage1_val_vis_images.size(-1)}"
        )

    ae = _build_ae()
    if args.stage1_epochs > 0:
        # Resume stage-1 from the most recent saved checkpoint when available.
        _load_laser_weights(ae, required=False, prefer_best=False)
    if args.stage1_epochs > 0:
        if not torch.cuda.is_available():
            raise RuntimeError("Stage-1 Lightning multi-GPU training requires CUDA.")
        if torch.cuda.device_count() < args.stage1_devices:
            raise RuntimeError(
                f"Requested {args.stage1_devices} GPUs for stage-1, but only {torch.cuda.device_count()} detected."
            )
        s1_module = Stage1Module(
            ae=ae,
            lr=args.stage1_lr,
            bottleneck_weight=args.bottleneck_weight,
            out_dir=stage1_dir,
            val_vis_images=stage1_val_vis_images,
            fid_num_samples=args.fid_num_samples,
            fid_feature=args.fid_feature,
            fid_compute_batch_size=args.fid_compute_batch_size,
            lr_schedule=args.stage1_lr_schedule,
            warmup_epochs=args.stage1_warmup_epochs,
            min_lr_ratio=args.stage1_min_lr_ratio,
        )
        s1_trainer = pl.Trainer(
            accelerator="gpu",
            devices=args.stage1_devices,
            strategy=(args.stage1_strategy if args.stage1_devices > 1 else "auto"),
            max_epochs=args.stage1_epochs,
            logger=_build_wandb_logger("stage1"),
            plugins=trainer_plugins,
            enable_checkpointing=False,
            gradient_clip_val=args.grad_clip,
            precision=args.stage1_precision,
            log_every_n_steps=10,
            deterministic=False,
        )
        s1_trainer.fit(s1_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank not in (None, "0"):
            # In stage-1 DDP non-zero ranks, stop here so only rank 0
            # proceeds to tokenization and stage-2 launch.
            return

    # For stage-2 tokenization, use the best available LASER checkpoint if present.
    # If stage-1 is skipped, require an existing checkpoint.
    if args.stage1_epochs <= 0:
        _load_laser_weights(ae, required=True, prefer_best=True)
    else:
        _load_laser_weights(ae, required=False, prefer_best=True)
    # ------------------------------------------------------------------
    # Token cache: load if available, otherwise recompute and save.
    # Delete the cache file manually (or pass --force_retokenize) to
    # force re-encoding after retraining stage 1.
    # ------------------------------------------------------------------
    _have_token_cache = (
        not args.force_retokenize
        and args.stage1_epochs <= 0
        and os.path.exists(token_cache_path)
    )

    if _have_token_cache:
        print(f"[Stage2] loading token cache: {token_cache_path}")
        try:
            cache = torch.load(token_cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            cache = torch.load(token_cache_path, map_location="cpu")
        cache_quantized = bool(cache.get("quantize_sparse_coeffs", True))
        if cache_quantized != bool(args.quantize_sparse_coeffs):
            raise ValueError(
                "Token cache quantization mode does not match current run "
                f"(cache quantize_sparse_coeffs={cache_quantized}, "
                f"args quantize_sparse_coeffs={args.quantize_sparse_coeffs}). "
                "Re-run with --force_retokenize."
            )
        tokens_flat = cache["tokens_flat"]
        coeffs_flat = cache.get("coeffs_flat")
        fid_real_images = cache.get("fid_real_images")

        cached_H, cached_W, D = cache["shape"]
        full_latent = cache.get("full_latent_shape") or cache.get("sample_latent_shape")

        if args.stage2_sliding_window:
            H = int(args.stage2_window_latent_h)
            W = int(args.stage2_window_latent_w)
            if full_latent is not None:
                full_H, full_W = int(full_latent[0]), int(full_latent[1])
            else:
                full_H, full_W = cached_H, cached_W
            if H > full_H or W > full_W:
                raise ValueError(
                    f"Sliding window ({H}, {W}) exceeds full latent ({full_H}, {full_W})."
                )
            sample_latent_shape = (full_H, full_W)
            sliding_window_stride_latent = int(args.stage2_window_stride_latent)
        else:
            if full_latent is not None:
                H, W = int(full_latent[0]), int(full_latent[1])
            else:
                H, W = cached_H, cached_W
            sample_latent_shape = None
            sliding_window_stride_latent = None
        full_latent_shape = full_latent if full_latent is not None else (cached_H, cached_W)

        print(f"[Stage2] token dataset: {tokens_flat.shape}   (H={H}, W={W}, D={D})")
    else:
        encode_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ae = ae.to(encode_device)
        token_loader_batch_size = int(args.batch_size)
        if args.stage2_sliding_window:
            token_loader_batch_size = max(1, min(token_loader_batch_size, 16))
        token_loader = DataLoader(
            stage2_source_set,
            batch_size=token_loader_batch_size,
            shuffle=False,
            num_workers=args.token_num_workers,
            pin_memory=True,
            persistent_workers=(args.token_num_workers > 0),
        )
        _token_max_items = None if args.token_subset <= 0 else min(args.token_subset, len(stage2_source_set))
        sample_latent_shape = None
        sliding_window_stride_latent = None
        if args.stage2_sliding_window:
            tokens_flat, coeffs_flat, full_H, full_W, D = precompute_tokens(
                ae,
                token_loader,
                encode_device,
                max_items=_token_max_items,
            )
            H = int(args.stage2_window_latent_h)
            W = int(args.stage2_window_latent_w)
            if H > int(full_H) or W > int(full_W):
                raise ValueError(
                    f"Sliding window latent size ({H}, {W}) exceeds full latent shape ({full_H}, {full_W})."
                )
            sample_latent_shape = (int(full_H), int(full_W))
            sliding_window_stride_latent = int(args.stage2_window_stride_latent)
        else:
            tokens_flat, coeffs_flat, H, W, D = precompute_tokens(
                ae,
                token_loader,
                encode_device,
                max_items=_token_max_items,
            )
        fid_real_loader = DataLoader(
            stage2_source_set if args.stage2_sliding_window else train_set,
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
        if sample_latent_shape is not None:
            full_h, full_w = int(sample_latent_shape[0]), int(sample_latent_shape[1])
            stride_lat = int(sliding_window_stride_latent or H)
            grid_h = ((full_h - int(H)) // stride_lat) + 1
            grid_w = ((full_w - int(W)) // stride_lat) + 1
            windows_per_image = int(grid_h * grid_w)
            print(
                f"[Stage2] sliding-window mode: "
                f"window_latent={H}x{W}, full_latent={sample_latent_shape[0]}x{sample_latent_shape[1]}, "
                f"stride_latent={stride_lat}, windows_per_image={windows_per_image}"
            )

        full_latent_shape = sample_latent_shape or (H, W)
        torch.save(
            {
                "tokens_flat": tokens_flat,
                "coeffs_flat": coeffs_flat,
                "quantize_sparse_coeffs": args.quantize_sparse_coeffs,
                "shape": (H, W, D),
                "full_latent_shape": full_latent_shape,
                "sample_latent_shape": sample_latent_shape,
                "sliding_window_stride_latent": sliding_window_stride_latent,
                "fid_real_images": fid_real_images,
            },
            token_cache_path,
        )
        print(f"[Stage2] token cache saved: {token_cache_path}")
    cache_for_stage2 = {
        "tokens_flat": tokens_flat,
        "coeffs_flat": coeffs_flat,
        "quantize_sparse_coeffs": args.quantize_sparse_coeffs,
        "shape": (H, W, D),
        "full_latent_shape": full_latent_shape,
        "sample_latent_shape": sample_latent_shape,
        "sliding_window_stride_latent": sliding_window_stride_latent,
        "fid_real_images": fid_real_images,
    }
    tokens_flat, coeffs_flat, H, W, D, site_tokenizer, cache_for_stage2 = _prepare_stage2_token_data(
        cache_for_stage2,
        ae,
    )
    if args.stage2_arch == "site_flat":
        torch.save(cache_for_stage2, token_cache_path)
        print(f"[Stage2] updated token cache with site tokens: {token_cache_path}")
    coeff_mean, coeff_std = 0.0, 1.0
    if coeffs_flat is not None:
        coeff_mean = float(coeffs_flat.mean().item())
        coeff_std = float(coeffs_flat.std().item())
        print(f"[Stage2] coefficient stats: mean={coeff_mean:.4f}, std={coeff_std:.4f}")

    if args.stage1_epochs <= 0:
        # When stage-1 is skipped, stage-2 DDP workers may re-exec this script.
        # Mark the phase now so child ranks load the saved token cache instead of
        # re-running dataset build + token precompute on every rank.
        os.environ["LASER_DDP_PHASE"] = "stage2"
        _run_stage2(
            tokens_flat=tokens_flat,
            coeffs_flat=coeffs_flat,
            quantize_sparse_coeffs=args.quantize_sparse_coeffs,
            H=H,
            W=W,
            D=D,
            site_tokenizer=site_tokenizer,
            sample_latent_shape=sample_latent_shape,
            sliding_window_stride_latent=sliding_window_stride_latent,
            ae_model=ae,
            fid_real_images=fid_real_images,
            coeff_mean=coeff_mean,
            coeff_std=coeff_std,
        )
        return
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
    cmd = [sys.executable, __file__, *sys.argv[1:]]
    env = os.environ.copy()
    ret = subprocess.call(cmd, env=env, close_fds=True)
    if ret != 0:
        raise RuntimeError(f"Stage-2 restart process failed with exit code {ret}.")
    return

    print("Done.")
    print(f"Outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
