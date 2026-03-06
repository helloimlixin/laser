
"""
laser.py

Minimal LASER training script.

It keeps the core two-stage workflow:
  - stage 1: train the LASER autoencoder
  - stage 2: train a transformer prior on flattened sparse tokens

The default dataset is CelebA under ../../data/celeba relative to this file.
For multi-GPU runs, launch with torchrun.
"""
import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.linalg import sqrtm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.models import Inception_V3_Weights, inception_v3
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
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CELEBA_DIR = (SCRIPT_DIR / "../../data/celeba").resolve()


def _default_image_size(dataset: str) -> int:
    return 128 if dataset == "celeba" else 32


def _default_data_dir(dataset: str) -> Path:
    if dataset == "celeba":
        return DEFAULT_CELEBA_DIR
    return (SCRIPT_DIR / "data").resolve()


def _default_out_dir(dataset: str, image_size: int) -> Path:
    return (SCRIPT_DIR / "runs" / f"laser_{dataset}_{image_size}").resolve()


def _init_distributed() -> Tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0, 1
    if not torch.cuda.is_available():
        raise RuntimeError("Multi-GPU training requires CUDA.")

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return True, rank, local_rank, world_size


def _cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _barrier():
    if _is_distributed():
        if torch.cuda.is_available():
            # NCCL warns if barrier cannot infer the rank-to-device mapping.
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def _distributed_mean(value: torch.Tensor) -> torch.Tensor:
    if not _is_distributed():
        return value.detach()
    reduced = value.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= dist.get_world_size()
    return reduced


def _unwrap_module(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, DDP) else module


_RFID_MODEL = None
_RFID_MODEL_DEVICE = None
_RFID_METRIC = None
_RFID_METRIC_DEVICE = None


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

class DictionaryLearningTokenized(nn.Module):
    """
    Dictionary-learning bottleneck with batched Orthogonal Matching Pursuit (OMP) sparse coding.
    Tokenization modes:
    - Quantized-mode: token = atom_id * n_bins + coefficient_bin (backward compatible).
    - Regressor-mode: token = atom_id only, coefficients are modeled with a separate head.

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
        n_bins: int = 16,
        coef_max: float = 3.0,
        quantize_sparse_coeffs: bool = True,
        coef_quantization: str = "mu_law",
        coef_mu: float = 50.0,
        commitment_cost: float = 0.25,
        epsilon: float = 1e-10,
    ):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.sparsity_level = int(sparsity_level)
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
        self.commitment_cost = float(commitment_cost)
        self.epsilon = float(epsilon)

        # Dictionary shape [C, K] (matches LASER)
        self.dictionary = nn.Parameter(torch.randn(self.embedding_dim, self.num_embeddings) * 0.02)

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

    def _normalize_dict(self) -> torch.Tensor:
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

    def _encode_sparse_codes(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run OMP and return support atom ids and continuous coefficients."""
        B, C, H, W = z_e.shape
        n_signals = B * H * W
        signals = z_e.permute(0, 2, 3, 1).contiguous().view(-1, C).t()
        dictionary = self._normalize_dict()
        with torch.no_grad():
            support, coeffs = self.batch_omp_with_support(signals, dictionary)
        if support.ndim != 2 or coeffs.ndim != 2:
            raise RuntimeError(
                f"OMP returned invalid rank: support={tuple(support.shape)} coeffs={tuple(coeffs.shape)}"
            )
        if support.size(0) != n_signals or coeffs.size(0) != n_signals:
            raise RuntimeError(
                f"OMP returned invalid batch size: expected {n_signals}, "
                f"got support={support.size(0)} coeffs={coeffs.size(0)}"
            )
        # Defensive shape guard: keep a fixed D stack even if OMP exits short due to numerical edge cases.
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
            support.view(B, H, W, self.sparsity_level),
            coeffs.view(B, H, W, self.sparsity_level),
        )

    def _reconstruct_sparse(
        self, support: torch.Tensor, coeffs: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct latent map from atom ids + coefficients."""
        if support.shape != coeffs.shape:
            raise ValueError(
                f"support and coeffs shape mismatch: {support.shape} vs {coeffs.shape}"
            )

        B, H, W, D = support.shape
        if D != self.sparsity_level:
            raise ValueError(f"Expected D={self.sparsity_level}, got {D}")

        dictionary = self._normalize_dict().t()  # [num_embeddings, C]
        support = support.to(torch.long)
        coeffs = coeffs.to(dictionary.dtype)
        support_flat = support.view(-1, D)
        coeffs_flat = coeffs.view(-1, D)
        atoms = dictionary[support_flat]  # [B*H*W, D, C]
        recon_flat = (atoms * coeffs_flat.unsqueeze(-1)).sum(dim=1)  # [N, C]
        return recon_flat.view(B, H, W, self.embedding_dim).permute(0, 3, 1, 2).contiguous()

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
        if self.sparsity_level > int(D.size(1)):
            raise ValueError(
                f"sparsity_level ({self.sparsity_level}) must be <= num_atoms ({int(D.size(1))})"
            )
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
        # Always run exactly K selections; relying on residual-based stopping can return short supports.
        while k < self.sparsity_level:
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

        support, coeffs = self._encode_sparse_codes(z_e)
        support_flat = support.view(-1, self.sparsity_level)
        coeffs_flat = coeffs.view(-1, self.sparsity_level)

        if self.quantize_sparse_coeffs:
            # Quantize coefficients (Option A)
            bin_idx, coeff_q = self._quantize_coeff(coeffs_flat)  # both [Nsig, D]
            # Tokens: [Nsig, D] -> [B,H,W,D]
            tokens = (support_flat * self.n_bins + bin_idx).view(B, H, W, self.sparsity_level)
            coeffs_for_recon = coeff_q
        else:
            tokens = support.view(B, H, W, self.sparsity_level).long()
            coeffs_for_recon = coeffs_flat

        coeffs_for_recon = coeffs_for_recon.reshape(B, H, W, self.sparsity_level)
        z_q = self._reconstruct_sparse(support, coeffs_for_recon)

        # Bottleneck loss (LASER-style)
        dl_latent_loss = F.mse_loss(z_q, z_e.detach())
        e_latent_loss = F.mse_loss(z_q.detach(), z_e)
        loss = dl_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator to encoder
        z_q_ste = z_e + (z_q - z_e).detach()
        return z_q_ste, loss, tokens

    @torch.no_grad()
    def tokens_to_latent(self, tokens: torch.Tensor, coeffs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode tokens back to a latent map.
        Args:
            tokens: [B, H, W, D]
            coeffs: [B, H, W, D] (only used in non-quantized mode)
        Returns:
            z_q: [B, C, H, W]
        """
        if tokens.dim() != 4:
            raise ValueError(f"Expected [B,H,W,D], got {tuple(tokens.shape)}")
        B, H, W, D = tokens.shape
        if D != self.sparsity_level:
            raise ValueError(f"Expected D={self.sparsity_level}, got {D}")

        if self.quantize_sparse_coeffs:
            dictionary = self._normalize_dict()
            tok = tokens.to(torch.long)
            special = tok >= self.pad_token_id  # pad or bos
            tok_clamped = tok.clamp_max(self.pad_token_id - 1)

            atom = tok_clamped // self.n_bins
            bin_idx = tok_clamped % self.n_bins

            coeff = self._dequantize_coeff(bin_idx).to(dictionary.dtype)
            coeff = coeff * (~special).float()
            atom = atom * (~special).long()
            return self._reconstruct_sparse(atom, coeff)

        if coeffs is None:
            raise ValueError("coeffs must be provided when quantize_sparse_coeffs=False")

        return self._reconstruct_sparse(tokens.to(torch.long), coeffs.to(self._normalize_dict().dtype))


SparseBottleneck = DictionaryLearningTokenized


# -----------------------------
# Stage-1 model: Encoder + Dictionary bottleneck + Decoder
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
        n_bins: int = 16,
        coef_max: float = 3.0,
        coef_quantization: str = "mu_law",
        coef_mu: float = 50.0,
        out_tanh: bool = True,
        quantize_sparse_coeffs: bool = True,
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
        self.pre = nn.Conv2d(num_hiddens, embedding_dim, kernel_size=1)
        self.bottleneck = DictionaryLearningTokenized(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            sparsity_level=sparsity_level,
            n_bins=n_bins,
            coef_max=coef_max,
            quantize_sparse_coeffs=quantize_sparse_coeffs,
            coef_quantization=coef_quantization,
            coef_mu=coef_mu,
            commitment_cost=commitment_cost,
        )
        self.post = nn.Conv2d(embedding_dim, num_hiddens, kernel_size=3, padding=1)
        self.decoder = Decoder(
            num_hiddens,
            num_hiddens,
            num_residual_layers,
            num_residual_hiddens,
            out_channels=in_channels,
            num_upsamples=num_downsamples,
        )

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
    def encode_to_atoms_and_coeffs(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        z = self.encoder(x)
        z = self.pre(z)
        atoms, coeffs = self.bottleneck._encode_sparse_codes(z)
        return atoms, coeffs, atoms.shape[1], atoms.shape[2]

    @torch.no_grad()
    def decode_from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        z_q = self.bottleneck.tokens_to_latent(tokens)
        z_q = self.post(z_q)
        recon = self.decoder(z_q)
        if self.out_tanh:
            recon = torch.tanh(recon)
        return recon

    @torch.no_grad()
    def decode_from_atoms_and_coeffs(self, atoms: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        z_q = self.bottleneck._reconstruct_sparse(atoms, coeffs)
        z_q = self.post(z_q)
        recon = self.decoder(z_q)
        if self.out_tanh:
            recon = torch.tanh(recon)
        return recon


# Backward-compatible alias for older scratch experiments.
SparseDictAE = LASER


# -----------------------------
# Stage-2: RQTransformer prior (GPT-style causal transformer over stacks)
# -----------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, max_seq_len: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = float(dropout)
        self.max_seq_len = int(max_seq_len)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=(self.dropout_p if self.training else 0.0),
            is_causal=True,
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
    Autoregressive prior over a flattened H x W x D token grid.
    """
    def __init__(self, cfg: RQTransformerConfig, bos_token_id: int, pad_token_id: int):
        super().__init__()
        self.cfg = cfg
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
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Position ids for [BOS] + flattened token sequence.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L] tokens, L <= max_len
        Returns:
            logits: [B, L, vocab]
        """
        _, L = x.shape
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
        return self.head(h)

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
    ) -> torch.Tensor:
        """Sample a batch of flattened token sequences."""
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
                k = min(int(top_k), int(logits.size(-1)))
                v, ix = torch.topk(logits, k, dim=-1)
                mask = torch.full_like(logits, float("-inf"))
                mask.scatter_(1, ix, v)
                logits = mask
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            seq = torch.cat([seq, nxt], dim=1)
        return seq[:, 1:]


PriorConfig = RQTransformerConfig
Prior = RQTransformerPrior


# -----------------------------
# Training helpers
# -----------------------------

def _make_image_grid(x: torch.Tensor, nrow: int = 8) -> torch.Tensor:
    """Build image grid tensor from a batch in [-1, 1]."""
    x = x.detach().cpu().clamp(-1, 1)
    x = (x + 1.0) / 2.0
    return utils.make_grid(x, nrow=nrow)


def save_image_grid(x: torch.Tensor, path: str, nrow: int = 8):
    """Save a batch in [-1, 1] as a single image grid."""
    grid = _make_image_grid(x, nrow=nrow)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    utils.save_image(grid, path)


def _to_unit_range(x: torch.Tensor) -> torch.Tensor:
    """Map images from [-1, 1] to [0, 1] for reconstruction metrics."""
    return x.detach().clamp(-1, 1).add(1.0).mul(0.5)


def _batch_psnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Average PSNR over a batch of images in [0, 1]."""
    mse = F.mse_loss(x, y, reduction="none").mean(dim=(1, 2, 3))
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10)).mean()


def _batch_ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Average SSIM over a batch of images in [0, 1] using a Gaussian window."""
    _, channels, height, width = x.shape
    window_size = min(11, height, width)
    if window_size % 2 == 0:
        window_size -= 1
    window_size = max(1, window_size)
    radius = window_size // 2

    coords = torch.arange(window_size, device=x.device, dtype=x.dtype) - radius
    kernel_1d = torch.exp(-(coords ** 2) / (2 * (1.5 ** 2)))
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-12)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()

    mu_x = F.conv2d(x, kernel, padding=radius, groups=channels)
    mu_y = F.conv2d(y, kernel, padding=radius, groups=channels)
    mu_x_sq = mu_x.square()
    mu_y_sq = mu_y.square()
    mu_xy = mu_x * mu_y

    sigma_x = F.conv2d(x * x, kernel, padding=radius, groups=channels) - mu_x_sq
    sigma_y = F.conv2d(y * y, kernel, padding=radius, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(x * y, kernel, padding=radius, groups=channels) - mu_xy

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x_sq + mu_y_sq + c1) * (sigma_x + sigma_y + c2)
    ssim_map = num / den.clamp_min(1e-10)
    return ssim_map.mean(dim=(1, 2, 3)).mean()


def _get_rfid_model(device: torch.device) -> nn.Module:
    """Build and cache the fallback Inception-V3 feature extractor used for reconstruction FID."""
    global _RFID_MODEL, _RFID_MODEL_DEVICE
    device_key = str(device)
    if _RFID_MODEL is None or _RFID_MODEL_DEVICE != device_key:
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        model.fc = nn.Identity()
        model.eval().to(device)
        _RFID_MODEL = model
        _RFID_MODEL_DEVICE = device_key
    return _RFID_MODEL


def _get_rfid_metric(device: torch.device, feature: int = 64):
    """Build and cache the canonical torchmetrics FID metric when available."""
    global _RFID_METRIC, _RFID_METRIC_DEVICE
    if FrechetInceptionDistance is None:
        return None
    device_key = f"{device}:{int(feature)}"
    if _RFID_METRIC is None or _RFID_METRIC_DEVICE != device_key:
        _RFID_METRIC = FrechetInceptionDistance(
            feature=int(feature),
            sync_on_compute=False,
            normalize=False,
        ).to(device)
        _RFID_METRIC_DEVICE = device_key
    _RFID_METRIC.reset()
    return _RFID_METRIC


@torch.no_grad()
def _extract_rfid_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Extract pooled Inception-V3 features from images in [-1, 1]."""
    x = _to_unit_range(x)
    if x.size(-2) != 299 or x.size(-1) != 299:
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    mean = x.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = x.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    x = (x - mean) / std
    feats = model(x)
    if isinstance(feats, tuple):
        feats = feats[0]
    return feats.float()


def _to_uint8_images(x: torch.Tensor) -> torch.Tensor:
    """Convert images from [-1, 1] to uint8 RGB tensors for FID."""
    return ((x.detach().cpu().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)


def _frechet_distance_from_features(real_feats: torch.Tensor, fake_feats: torch.Tensor) -> float:
    """Compute Fréchet distance between two feature clouds."""
    real_np = real_feats.detach().cpu().numpy().astype(np.float64, copy=False)
    fake_np = fake_feats.detach().cpu().numpy().astype(np.float64, copy=False)
    mu_real = np.mean(real_np, axis=0)
    mu_fake = np.mean(fake_np, axis=0)
    sigma_real = np.cov(real_np, rowvar=False)
    sigma_fake = np.cov(fake_np, rowvar=False)

    cov_prod = sigma_real @ sigma_fake
    cov_mean = sqrtm(cov_prod)
    if not np.isfinite(cov_mean).all():
        eps = 1e-6
        eye = np.eye(sigma_real.shape[0], dtype=np.float64)
        cov_mean = sqrtm((sigma_real + eps * eye) @ (sigma_fake + eps * eye))
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real

    diff = mu_real - mu_fake
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2.0 * cov_mean)
    return float(max(fid, 0.0))


@torch.no_grad()
def _compute_reconstruction_fid(
    ae: LASER,
    loader: Optional[DataLoader],
    device: torch.device,
    max_items: int,
) -> Optional[float]:
    """Compute reconstruction FID between validation images and their reconstructions."""
    if loader is None or max_items <= 1:
        return None

    ae.eval()
    fid_metric = _get_rfid_metric(device, feature=64)
    if fid_metric is not None:
        seen = 0
        for x, _ in tqdm(loader, desc="[Stage1] compute rFID", leave=False, dynamic_ncols=True):
            x = x.to(device)
            recon, _, _ = ae(x)
            keep = min(x.size(0), max_items - seen)
            if keep <= 0:
                break
            fid_metric.update(_to_uint8_images(x[:keep]).to(device), real=True)
            fid_metric.update(_to_uint8_images(recon[:keep]).to(device), real=False)
            seen += keep
            if seen >= max_items:
                break
        if seen <= 1:
            return None
        return float(fid_metric.compute().detach().cpu().item())

    model = _get_rfid_model(device)
    real_feats = []
    fake_feats = []
    seen = 0

    for x, _ in tqdm(loader, desc="[Stage1] compute rFID", leave=False, dynamic_ncols=True):
        x = x.to(device)
        recon, _, _ = ae(x)
        keep = min(x.size(0), max_items - seen)
        if keep <= 0:
            break
        real_feats.append(_extract_rfid_features(model, x[:keep]).cpu())
        fake_feats.append(_extract_rfid_features(model, recon[:keep]).cpu())
        seen += keep
        if seen >= max_items:
            break

    if seen <= 1:
        return None
    return _frechet_distance_from_features(torch.cat(real_feats, dim=0), torch.cat(fake_feats, dim=0))


def _init_wandb(args) -> Optional[object]:
    if not getattr(args, "wandb", True):
        return None
    if wandb is None:
        print("[W&B] wandb is not installed; continuing without logging.")
        return None
    try:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            dir=args.wandb_dir,
            mode=args.wandb_mode,
            config=dict(vars(args)),
        )
        run.define_metric("stage1/step")
        run.define_metric("stage1/*", step_metric="stage1/step")
        run.define_metric("stage2/step")
        run.define_metric("stage2/*", step_metric="stage2/step")
        return run
    except Exception as exc:
        print(f"[W&B] init failed ({exc}); continuing without logging.")
        return None


def _log_wandb(
    run: Optional[object],
    data: dict,
    step_metric: Optional[str] = None,
    step_value: Optional[int] = None,
):
    if run is None:
        return
    payload = dict(data)
    if step_metric is not None and step_value is not None:
        payload[step_metric] = int(step_value)
    run.log(payload)


def _log_wandb_image(
    run: Optional[object],
    key: str,
    x: torch.Tensor,
    step_metric: Optional[str] = None,
    step_value: Optional[int] = None,
    caption: Optional[str] = None,
):
    if run is None or wandb is None:
        return
    grid = _make_image_grid(x)
    image = grid.permute(1, 2, 0).mul(255).clamp(0, 255).byte().numpy()
    payload = {key: wandb.Image(image, caption=caption)}
    if step_metric is not None and step_value is not None:
        payload[step_metric] = int(step_value)
    run.log(payload)


def train_stage1_ae(
    ae: LASER,
    train_loader: DataLoader,
    val_loader: DataLoader,
    rfid_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int,
    lr: float,
    bottleneck_weight: float,
    grad_clip: float,
    out_dir: str,
    rfid_num_samples: int = 0,
    train_sampler: Optional[DistributedSampler] = None,
    is_main_process: bool = True,
    wandb_run: Optional[object] = None,
):
    """Train stage 1 with optional DDP and rank-0-only artifacts."""
    ae_module = _unwrap_module(ae)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    best_val = float("inf")
    global_step = 0
    rfid_warned = False

    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        ae.train()
        pbar = tqdm(train_loader, desc=f"[Stage1] epoch {epoch}/{epochs}", disable=(not is_main_process))
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

            # Keep dictionary atoms normalized after each optimizer step.
            with torch.no_grad():
                ae_module.bottleneck.dictionary.copy_(
                    F.normalize(
                        ae_module.bottleneck.dictionary,
                        p=2,
                        dim=0,
                        eps=ae_module.bottleneck.epsilon,
                    )
                )

            loss_log = _distributed_mean(loss)
            recon_log = _distributed_mean(recon_loss)
            b_log = _distributed_mean(b_loss)
            running += float(loss_log.item())
            global_step += 1
            if is_main_process:
                pbar.set_postfix(
                    loss=float(loss_log.item()),
                    recon=float(recon_log.item()),
                    b=float(b_log.item()),
                )
                _log_wandb(
                    wandb_run,
                    {
                        "stage1/train_loss": float(loss_log.item()),
                        "stage1/recon_loss": float(recon_log.item()),
                        "stage1/bottleneck_loss": float(b_log.item()),
                        "stage1/epoch": epoch,
                    },
                    step_metric="stage1/step",
                    step_value=global_step,
                )

        # Validation
        ae.eval()
        val_loss_sum = torch.zeros(1, device=device)
        val_psnr_sum = torch.zeros(1, device=device)
        val_ssim_sum = torch.zeros(1, device=device)
        val_count = torch.zeros(1, device=device)
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                recon, b_loss, _ = ae(x)
                recon_loss = F.mse_loss(recon, x)
                loss = recon_loss + bottleneck_weight * b_loss
                x_unit = _to_unit_range(x)
                recon_unit = _to_unit_range(recon)
                psnr = _batch_psnr(recon_unit, x_unit)
                ssim = _batch_ssim(recon_unit, x_unit)
                val_loss_sum += loss.detach() * x.size(0)
                val_psnr_sum += psnr.detach() * x.size(0)
                val_ssim_sum += ssim.detach() * x.size(0)
                val_count += x.size(0)
        if _is_distributed():
            dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_psnr_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_ssim_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_count, op=dist.ReduceOp.SUM)
        val_loss = float((val_loss_sum / val_count.clamp_min(1)).item())
        val_psnr = float((val_psnr_sum / val_count.clamp_min(1)).item())
        val_ssim = float((val_ssim_sum / val_count.clamp_min(1)).item())

        if is_main_process:
            print(
                f"[Stage1] epoch {epoch} val_loss={val_loss:.6f} "
                f"psnr={val_psnr:.3f} ssim={val_ssim:.4f}"
            )
            _log_wandb(
                wandb_run,
                {
                    "stage1/val_loss": float(val_loss),
                    "stage1/val_psnr": float(val_psnr),
                    "stage1/val_ssim": float(val_ssim),
                    "stage1/epoch": epoch,
                },
                step_metric="stage1/step",
                step_value=global_step,
            )

        _barrier()
        if is_main_process:
            val_rfid = None
            if rfid_num_samples > 0:
                try:
                    val_rfid = _compute_reconstruction_fid(
                        ae_module,
                        rfid_loader,
                        device,
                        max_items=rfid_num_samples,
                    )
                except Exception as exc:
                    if not rfid_warned:
                        print(f"[Stage1] rFID unavailable: {exc}")
                        rfid_warned = True
                    val_rfid = None
                if val_rfid is not None:
                    print(f"[Stage1] epoch {epoch} rfid={val_rfid:.4f}")
                    _log_wandb(
                        wandb_run,
                        {
                            "stage1/rfid": float(val_rfid),
                            "stage1/epoch": epoch,
                        },
                        step_metric="stage1/step",
                        step_value=global_step,
                    )

            x_vis, _ = next(iter(val_loader))
            x_vis = x_vis.to(device)[:64]
            with torch.no_grad():
                recon_vis, _, _ = ae_module(x_vis)
            save_image_grid(x_vis, os.path.join(out_dir, f"stage1_epoch{epoch:03d}_real.png"))
            save_image_grid(recon_vis, os.path.join(out_dir, f"stage1_epoch{epoch:03d}_recon.png"))
            _log_wandb_image(
                wandb_run,
                "stage1/real",
                x_vis,
                step_metric="stage1/step",
                step_value=global_step,
                caption=f"epoch={epoch} real",
            )
            _log_wandb_image(
                wandb_run,
                "stage1/recon",
                recon_vis,
                step_metric="stage1/step",
                step_value=global_step,
                caption=f"epoch={epoch} recon",
            )

            os.makedirs(out_dir, exist_ok=True)
            ckpt_path = os.path.join(out_dir, "ae_last.pt")
            torch.save(ae_module.state_dict(), ckpt_path)
            if val_loss < best_val:
                best_val = val_loss
                torch.save(ae_module.state_dict(), os.path.join(out_dir, "ae_best.pt"))
        _barrier()


@torch.no_grad()
def precompute_tokens(
    ae: LASER,
    loader: DataLoader,
    device: torch.device,
    max_items: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, int, int]:
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
            tokens, h, w = ae.encode_to_tokens(x)
            coeffs = None
        else:
            tokens, coeffs, h, w = ae.encode_to_atoms_and_coeffs(x)
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


def train_stage2_transformer(
    transformer: RQTransformerPrior,
    token_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    pad_token_id: int,
    out_dir: str,
    ae_for_decode: LASER,
    H: int,
    W: int,
    D: int,
    sample_every_steps: int = 200,
    sample_batch_size: int = 8,
    sample_temperature: float = 1.0,
    sample_top_k: Optional[int] = 256,
    sample_image_size: Optional[int] = None,
    token_sampler: Optional[DistributedSampler] = None,
    is_main_process: bool = True,
    wandb_run: Optional[object] = None,
):
    """Train stage 2 with optional DDP and synchronized rank-0 sampling."""
    transformer_module = _unwrap_module(transformer)
    ae_decode = _unwrap_module(ae_for_decode)
    opt = torch.optim.Adam(transformer.parameters(), lr=lr)
    vocab = transformer_module.cfg.vocab_size
    bos = transformer_module.bos_token_id
    global_step = 0
    sample_top_k = None if sample_top_k is None or int(sample_top_k) <= 0 else int(sample_top_k)

    for epoch in range(1, epochs + 1):
        if token_sampler is not None:
            token_sampler.set_epoch(epoch)
        transformer.train()
        pbar = tqdm(token_loader, desc=f"[Stage2] epoch {epoch}/{epochs}", disable=(not is_main_process))
        running = 0.0
        steps = 0

        for batch in pbar:
            tok_flat = batch[0] if isinstance(batch, (tuple, list)) else batch
            tok_flat = tok_flat.to(device).long()
            B = tok_flat.size(0)

            seq = torch.cat([torch.full((B, 1), bos, device=device, dtype=torch.long), tok_flat], dim=1)
            x_in = seq[:, :-1]
            y = seq[:, 1:]

            opt.zero_grad(set_to_none=True)
            logits = transformer(x_in)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab),
                y.reshape(-1),
                ignore_index=pad_token_id,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
            opt.step()
            global_step += 1

            loss_log = _distributed_mean(loss)
            running += float(loss_log.item())
            steps += 1
            if is_main_process:
                pbar.set_postfix(loss=float(loss_log.item()))
                _log_wandb(
                    wandb_run,
                    {
                        "stage2/train_loss": float(loss_log.item()),
                        "stage2/epoch": epoch,
                    },
                    step_metric="stage2/step",
                    step_value=global_step,
                )

            if sample_every_steps > 0 and (global_step % sample_every_steps == 0):
                _barrier()
                if is_main_process:
                    transformer.eval()
                    ae_decode.eval()
                    print(f"[Stage2] sampling at step {global_step} (batch_size={sample_batch_size})...")
                    with torch.no_grad():
                        flat_gen = transformer_module.generate(
                            batch_size=sample_batch_size,
                            temperature=sample_temperature,
                            top_k=sample_top_k,
                            show_progress=True,
                            progress_desc=f"[Stage2] sample step {global_step}",
                        )
                        tokens_gen = flat_gen.view(-1, H, W, D)
                        imgs = ae_decode.decode_from_tokens(tokens_gen.to(device))
                        if sample_image_size is not None and int(sample_image_size) > 0:
                            if imgs.size(-2) != int(sample_image_size) or imgs.size(-1) != int(sample_image_size):
                                imgs = F.interpolate(
                                    imgs,
                                    size=(int(sample_image_size), int(sample_image_size)),
                                    mode="bilinear",
                                    align_corners=False,
                                )
                    save_image_grid(imgs, os.path.join(out_dir, f"stage2_step{global_step:06d}_samples.png"))
                    _log_wandb_image(
                        wandb_run,
                        "stage2/samples",
                        imgs,
                        step_metric="stage2/step",
                        step_value=global_step,
                        caption=f"step={global_step}",
                    )
                    print(f"[Stage2] sampling done at step {global_step}")
                _barrier()
                transformer.train()

        epoch_loss = running / max(1, steps)
        if is_main_process:
            print(f"[Stage2] epoch {epoch} train_loss={epoch_loss:.6f}")
            _log_wandb(
                wandb_run,
                {
                    "stage2/epoch_loss": float(epoch_loss),
                    "stage2/epoch": epoch,
                },
                step_metric="stage2/step",
                step_value=global_step,
            )

        _barrier()
        if is_main_process:
            os.makedirs(out_dir, exist_ok=True)
            torch.save(transformer_module.state_dict(), os.path.join(out_dir, "transformer_last.pt"))
        _barrier()



# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Train the core LASER pipeline.")
    parser.add_argument("--dataset", type=str, default="celeba", choices=["cifar10", "celeba"])
    parser.add_argument("--data_dir", type=str, default=None, help="Root directory for dataset files.")
    parser.add_argument(
        "--image_size",
        type=int,
        default=128,
        help="Resize every image to this square size.",
    )
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4, help="Workers for image dataloaders.")
    parser.add_argument("--token_num_workers", type=int, default=0, help="Workers for token precompute.")
    parser.add_argument("--wandb", dest="wandb", action="store_true", default=True, help="Enable Weights & Biases logging.")
    parser.add_argument("--no_wandb", dest="wandb", action="store_false", help="Disable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="laser")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default="laser_celeba128")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_dir", type=str, default="./wandb")

    parser.add_argument("--stage1_epochs", type=int, default=100)
    parser.add_argument("--stage1_lr", type=float, default=2e-4)
    parser.add_argument("--stage2_epochs", type=int, default=100)
    parser.add_argument("--stage2_lr", type=float, default=4e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--stage2_batch_size", type=int, default=8)
    parser.add_argument("--bottleneck_weight", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--num_hiddens", type=int, default=128)
    parser.add_argument("--ae_num_downsamples", type=int, default=2)
    parser.add_argument("--num_res_layers", type=int, default=2)
    parser.add_argument("--num_res_hiddens", type=int, default=32)
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--num_atoms", type=int, default=128)
    parser.add_argument("--sparsity_level", type=int, default=3)
    parser.add_argument("--n_bins", type=int, default=32)
    parser.add_argument("--coef_max", type=float, default=30.0)
    parser.add_argument("--coef_quantization", type=str, default="uniform", choices=["uniform", "mu_law"])
    parser.add_argument("--coef_mu", type=float, default=0.0)
    parser.add_argument("--commitment_cost", type=float, default=0.25)

    parser.add_argument("--tf_d_model", type=int, default=256)
    parser.add_argument("--tf_heads", type=int, default=8)
    parser.add_argument("--tf_layers", type=int, default=6)
    parser.add_argument("--tf_ff", type=int, default=1024)
    parser.add_argument("--tf_dropout", type=float, default=0.1)
    parser.add_argument(
        "--token_subset",
        type=int,
        default=0,
        help="Number of stage-1 token grids to encode for stage-2 training (<= 0 uses the full set).",
    )
    parser.add_argument(
        "--rfid_num_samples",
        type=int,
        default=256,
        help="Number of validation images used for stage-1 reconstruction FID (0 disables it).",
    )
    parser.add_argument("--stage2_sample_every_steps", type=int, default=2000)
    parser.add_argument("--stage2_sample_batch_size", type=int, default=8)
    parser.add_argument("--stage2_sample_temperature", type=float, default=0.6)
    parser.add_argument("--stage2_sample_top_k", type=int, default=0)
    parser.add_argument("--stage2_sample_image_size", type=int, default=128)

    args = parser.parse_args()
    wandb_run = None
    distributed = False

    if args.ae_num_downsamples <= 0:
        raise ValueError(f"ae_num_downsamples must be positive, got {args.ae_num_downsamples}")
    if args.stage2_sample_temperature <= 0.0:
        raise ValueError("stage2_sample_temperature must be > 0.")
    if args.token_subset < 0:
        args.token_subset = 0
    if args.image_size is None:
        args.image_size = _default_image_size(args.dataset)
    args.image_size = int(args.image_size)
    if args.data_dir is None:
        args.data_dir = str(_default_data_dir(args.dataset))
    if args.out_dir is None:
        args.out_dir = str(_default_out_dir(args.dataset, args.image_size))

    distributed, rank, local_rank, world_size = _init_distributed()
    is_main_process = (rank == 0)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")

    if distributed:
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    if is_main_process:
        os.makedirs(args.out_dir, exist_ok=True)
    stage1_dir = os.path.join(args.out_dir, "stage1")
    stage2_dir = os.path.join(args.out_dir, "stage2")
    if is_main_process:
        os.makedirs(stage1_dir, exist_ok=True)
        os.makedirs(stage2_dir, exist_ok=True)
    _barrier()

    if is_main_process:
        print(
            f"[Setup] device={device} world_size={world_size} dataset={args.dataset} "
            f"data_dir={args.data_dir} image_size={args.image_size}"
        )
        wandb_run = _init_wandb(args)
    if wandb_run is not None:
        _log_wandb(
            wandb_run,
            {
                "setup/device": str(device),
                "setup/dataset": args.dataset,
            },
        )

    def _build_laser() -> LASER:
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
            out_tanh=True,
        )

    def _load_best_laser_weights(laser_model: LASER):
        best_path = os.path.join(stage1_dir, "ae_best.pt")
        if not os.path.exists(best_path):
            raise FileNotFoundError(f"Stage-1 checkpoint not found at {best_path}")
        try:
            state_dict = torch.load(best_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(best_path, map_location="cpu")
        laser_model.load_state_dict(state_dict)

    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    train_tfm = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normalize,
    ])
    eval_tfm = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    if args.dataset == "cifar10":
        train_set = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=train_tfm)
        val_set = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=eval_tfm)
        stage2_source_set = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=eval_tfm)
    elif args.dataset == "celeba":
        train_full = FlatImageDataset(root=args.data_dir, transform=train_tfm)
        val_full = FlatImageDataset(root=args.data_dir, transform=eval_tfm)
        token_full = FlatImageDataset(root=args.data_dir, transform=eval_tfm)
        if len(train_full) < 2:
            raise RuntimeError("CelebA dataset needs at least 2 images for train/val split.")
        val_size = max(1, int(0.05 * len(train_full)))
        train_size = len(train_full) - val_size
        indices = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(args.seed)).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        train_set = Subset(train_full, train_indices)
        val_set = Subset(val_full, val_indices)
        stage2_source_set = Subset(token_full, train_indices)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    train_sampler = DistributedSampler(train_set, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if distributed else None
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=min(64, args.batch_size),
        shuffle=False,
        sampler=val_sampler,
        num_workers=max(0, args.num_workers // 2),
        pin_memory=pin_memory,
    )
    rfid_loader = None
    if is_main_process and args.rfid_num_samples > 0:
        rfid_loader = DataLoader(
            val_set,
            batch_size=min(32, min(64, args.batch_size)),
            shuffle=False,
            num_workers=max(0, args.num_workers // 2),
            pin_memory=pin_memory,
        )

    laser = _build_laser().to(device)
    laser_stage1 = DDP(laser, device_ids=[local_rank], output_device=local_rank) if distributed else laser
    if args.stage1_epochs > 0:
        train_stage1_ae(
            ae=laser_stage1,
            train_loader=train_loader,
            val_loader=val_loader,
            rfid_loader=rfid_loader,
            device=device,
            epochs=args.stage1_epochs,
            lr=args.stage1_lr,
            bottleneck_weight=args.bottleneck_weight,
            grad_clip=args.grad_clip,
            out_dir=stage1_dir,
            rfid_num_samples=args.rfid_num_samples,
            train_sampler=train_sampler,
            is_main_process=is_main_process,
            wandb_run=wandb_run,
        )
    _barrier()

    _load_best_laser_weights(laser)
    laser = laser.to(device)

    token_cache_path = os.path.join(stage2_dir, "tokens_cache.pt")
    if is_main_process:
        token_subset = None if args.token_subset <= 0 else min(args.token_subset, len(stage2_source_set))
        token_source_loader = DataLoader(
            stage2_source_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.token_num_workers,
            pin_memory=pin_memory,
            persistent_workers=(args.token_num_workers > 0),
        )
        tokens_flat, coeffs_flat, H, W, D = precompute_tokens(
            laser,
            token_source_loader,
            device,
            max_items=token_subset,
        )
        if coeffs_flat is not None:
            raise RuntimeError("The simplified script only trains the quantized-token stage-2 prior.")
        torch.save({"tokens_flat": tokens_flat, "shape": (H, W, D)}, token_cache_path)
        print(f"[Stage2] token dataset: {tokens_flat.shape} (H={H}, W={W}, D={D})")
    _barrier()
    try:
        token_cache = torch.load(token_cache_path, map_location="cpu", weights_only=True)
    except TypeError:
        token_cache = torch.load(token_cache_path, map_location="cpu")
    tokens_flat = token_cache["tokens_flat"]
    H, W, D = token_cache["shape"]

    if args.stage2_epochs <= 0:
        _barrier()
        if is_main_process:
            if wandb_run is not None:
                wandb_run.finish()
            print(f"Outputs saved to: {args.out_dir}")
        _cleanup_distributed()
        return

    token_sampler = DistributedSampler(tokens_flat, shuffle=True) if distributed else None
    token_loader = DataLoader(
        tokens_flat,
        batch_size=args.stage2_batch_size,
        shuffle=(token_sampler is None),
        sampler=token_sampler,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=(len(tokens_flat) >= args.stage2_batch_size),
    )
    transformer = RQTransformerPrior(
        RQTransformerConfig(
            vocab_size=laser.bottleneck.vocab_size,
            H=H,
            W=W,
            D=D,
            d_model=args.tf_d_model,
            n_heads=args.tf_heads,
            n_layers=args.tf_layers,
            d_ff=args.tf_ff,
            dropout=args.tf_dropout,
        ),
        bos_token_id=laser.bottleneck.bos_token_id,
        pad_token_id=laser.bottleneck.pad_token_id,
    ).to(device)
    transformer_stage2 = DDP(transformer, device_ids=[local_rank], output_device=local_rank) if distributed else transformer

    train_stage2_transformer(
        transformer=transformer_stage2,
        token_loader=token_loader,
        device=device,
        epochs=args.stage2_epochs,
        lr=args.stage2_lr,
        pad_token_id=laser.bottleneck.pad_token_id,
        out_dir=stage2_dir,
        ae_for_decode=laser,
        H=H,
        W=W,
        D=D,
        sample_every_steps=args.stage2_sample_every_steps,
        sample_batch_size=args.stage2_sample_batch_size,
        sample_temperature=args.stage2_sample_temperature,
        sample_top_k=(None if args.stage2_sample_top_k <= 0 else args.stage2_sample_top_k),
        sample_image_size=args.stage2_sample_image_size,
        token_sampler=token_sampler,
        is_main_process=is_main_process,
        wandb_run=wandb_run,
    )

    if is_main_process:
        if wandb_run is not None:
            wandb_run.finish()
        print(f"Outputs saved to: {args.out_dir}")
    _barrier()
    _cleanup_distributed()


if __name__ == "__main__":
    main()
