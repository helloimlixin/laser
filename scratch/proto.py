
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
    from torchmetrics.image.fid import FrechetInceptionDistance
except Exception:
    FrechetInceptionDistance = None
try:
    import wandb
except Exception:
    wandb = None


def _disable_lightning_cuda_matmul_capability_probe():
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


_disable_lightning_cuda_matmul_capability_probe()


def _pick_free_tcp_port(excluded_ports: Optional[set[int]] = None) -> int:
    excluded = excluded_ports or set()
    for _ in range(32):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("0.0.0.0", 0))
            port = int(sock.getsockname()[1])
            if port in excluded:
                continue
            return port
        except OSError:
            continue
        finally:
            sock.close()
    raise RuntimeError("Unable to allocate a free TCP port for DDP rendezvous.")


def _ensure_free_master_port_for_ddp(stage_tag: str) -> None:
    # Only the launcher process should pick the rendezvous port.
    # Worker processes must inherit a stable value.
    local_rank_raw = os.environ.get("LOCAL_RANK")
    local_rank = None
    if local_rank_raw is not None:
        try:
            local_rank = int(local_rank_raw)
        except ValueError:
            # Treat malformed LOCAL_RANK as launcher context and proceed.
            local_rank = None
    if local_rank is not None and local_rank != 0:
        return

    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_ADDR", master_addr)

    raw_port = os.environ.get("MASTER_PORT")
    excluded_ports: set[int] = set()
    if raw_port is not None:
        try:
            excluded_ports.add(int(raw_port))
        except ValueError:
            pass

    # Always choose a fresh rendezvous port for this launch to avoid stale/fixed-port collisions.
    selected_port = _pick_free_tcp_port(excluded_ports=excluded_ports)
    os.environ["MASTER_PORT"] = str(selected_port)
    if raw_port is None:
        print(f"[{stage_tag}] MASTER_PORT not set; selected free port {selected_port}.")
    else:
        print(f"[{stage_tag}] overriding MASTER_PORT={raw_port!r}; selected free port {selected_port}.")


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
        n_bins: int = 129,
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
        support_flat = support.reshape(-1, D)
        coeffs_flat = coeffs.reshape(-1, D)
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
        coeffs_ordered = x[batch_idx, I]  # [B, K]

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


# -----------------------------
# Stage-1 model: Encoder + Dictionary bottleneck + Decoder
# -----------------------------

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
        n_bins: int = 129,
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


# -----------------------------
# Stage-2: RQTransformer prior (GPT-style causal transformer over stacks)
# -----------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
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
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out, new_kv


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
        self,
        x: torch.Tensor,
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
    num_patch_positions: int = 0
    context_tokens: int = 0
    predict_coefficients: bool = False
    coeff_loss_weight: float = 1.0
    coeff_max: float = 3.0
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
        self.predict_coefficients = bool(cfg.predict_coefficients)
        self.bos_token_id = int(bos_token_id)
        self.pad_token_id = int(pad_token_id)

        self.tokens_per_patch = cfg.H * cfg.W * cfg.D
        self.max_len = 1 + int(cfg.context_tokens) + self.tokens_per_patch

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.spatial_emb = nn.Embedding(cfg.H * cfg.W, cfg.d_model)
        self.depth_emb = nn.Embedding(cfg.D, cfg.d_model)
        self.type_emb = nn.Embedding(2, cfg.d_model)
        self.patch_pos_emb = (
            nn.Embedding(cfg.num_patch_positions, cfg.d_model)
            if cfg.num_patch_positions > 0
            else None
        )

        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.atom_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if self.predict_coefficients:
            self.coeff_head = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model // 2),
                nn.GELU(),
                nn.Linear(cfg.d_model // 2, cfg.vocab_size),
            )

        # Precompute position ids for [BOS] + optional context + current patch tokens.
        # For context tokens, repeat the local (spatial, depth) pattern per patch.
        spatial_ids = torch.zeros(self.max_len, dtype=torch.long)
        depth_ids = torch.zeros(self.max_len, dtype=torch.long)
        type_ids = torch.zeros(self.max_len, dtype=torch.long)  # 0 for BOS, 1 for normal
        if self.max_len > 1:
            idx = torch.arange(self.max_len - 1)
            local_idx = idx % self.tokens_per_patch
            spatial_ids[1:] = local_idx // cfg.D
            depth_ids[1:] = local_idx % cfg.D
            type_ids[1:] = 1
        self.register_buffer("_spatial_ids", spatial_ids)
        self.register_buffer("_depth_ids", depth_ids)
        self.register_buffer("_type_ids", type_ids)

    def forward(
        self,
        x: torch.Tensor,
        patch_pos_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[list] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], list]:
        """
        Args:
            x: [B, L] tokens
            patch_pos_ids: patch position ids, [B] or [B, L]
            kv_cache: list of per-layer (K, V) caches for incremental decoding
            start_pos: position offset into the precomputed spatial/depth/type ids
        Returns:
            atom_logits: [B, L, vocab]
            coeff: [B, L] or None
            new_kv_cache: list of per-layer (K, V) caches
        """
        B, L = x.shape
        pos_end = start_pos + L

        tok = self.token_emb(x)
        sp = self.spatial_emb(self._spatial_ids[start_pos:pos_end])
        dp = self.depth_emb(self._depth_ids[start_pos:pos_end])
        tp = self.type_emb(self._type_ids[start_pos:pos_end])

        h = tok + sp.unsqueeze(0) + dp.unsqueeze(0) + tp.unsqueeze(0)
        if self.patch_pos_emb is not None:
            if patch_pos_ids is None:
                raise ValueError("patch_pos_ids must be provided when num_patch_positions > 0")
            patch_pos_ids = patch_pos_ids.to(device=x.device, dtype=torch.long)
            if patch_pos_ids.dim() == 1:
                if patch_pos_ids.shape[0] != B:
                    raise ValueError(
                        f"patch_pos_ids must have shape [B], got {tuple(patch_pos_ids.shape)} for B={B}"
                    )
                h = h + self.patch_pos_emb(patch_pos_ids).unsqueeze(1)
            elif patch_pos_ids.dim() == 2:
                if patch_pos_ids.shape != (B, L):
                    raise ValueError(
                        f"patch_pos_ids must have shape [B, L]={B, L}, got {tuple(patch_pos_ids.shape)}"
                    )
                h = h + self.patch_pos_emb(patch_pos_ids)
            else:
                raise ValueError(
                    f"patch_pos_ids must have rank 1 or 2, got rank {patch_pos_ids.dim()}"
                )
        h = self.drop(h)

        new_kv_cache = []
        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            h, new_kv = block(h, kv_cache=layer_cache)
            new_kv_cache.append(new_kv)

        h = self.ln_f(h)
        atom_logits = self.atom_head(h)
        coeff = self.coeff_head(h) if self.predict_coefficients else None
        return atom_logits, coeff, new_kv_cache

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
        patch_pos_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with KV-cache.
        Returns:
            If predict_coefficients=False:
                flat_tokens: [B, H*W*D] (without BOS)
            If predict_coefficients=True:
                flat_tokens, flat_coeffs
        """
        device = next(self.parameters()).device
        T = self.cfg.H * self.cfg.W * self.cfg.D

        seq = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=device)
        if self.patch_pos_emb is not None:
            if patch_pos_ids is None:
                raise ValueError("patch_pos_ids must be provided when num_patch_positions > 0")
            patch_pos_ids = patch_pos_ids.to(device=device, dtype=torch.long)
            if patch_pos_ids.dim() != 1 or patch_pos_ids.shape[0] != batch_size:
                raise ValueError(
                    f"patch_pos_ids must have shape [batch_size], got {tuple(patch_pos_ids.shape)}"
                )
        special_ids = torch.tensor(
            [self.bos_token_id, self.pad_token_id], device=device,
        )
        coeffs: list[torch.Tensor] = []
        kv_cache = None
        steps = tqdm(
            range(T),
            desc=(progress_desc or "[Stage2] sampling tokens"),
            leave=False,
            dynamic_ncols=True,
            disable=(not show_progress),
        )
        for _ in steps:
            if kv_cache is None:
                inp = seq
                start_pos = 0
            else:
                inp = seq[:, -1:]
                start_pos = seq.size(1) - 1

            atom_logits, coeff_pred, kv_cache = self(
                inp, patch_pos_ids=patch_pos_ids,
                kv_cache=kv_cache, start_pos=start_pos,
            )

            logits = atom_logits[:, -1, :]
            logits[:, special_ids] = float("-inf")
            logits = logits / max(temperature, 1e-8)
            if top_k is not None and top_k > 0:
                v, ix = torch.topk(logits, top_k, dim=-1)
                mask = torch.full_like(logits, float("-inf"))
                mask.scatter_(1, ix, v)
                logits = mask
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            seq = torch.cat([seq, nxt], dim=1)

            if self.predict_coefficients:
                c = coeff_pred[:, -1, :].gather(-1, nxt).squeeze(-1)
                coeffs.append(c)

        if self.predict_coefficients:
            coeff_flat = torch.stack(coeffs, dim=1)
            coeff_flat = coeff_flat.clamp(-self.cfg.coeff_max, self.cfg.coeff_max)
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


def unfold_image_patches(
    x: torch.Tensor,
    patch_size: int,
    stride: int,
) -> Tuple[torch.Tensor, int, int]:
    """
    Split images into patches with unfold.
    Returns:
      patches: [B * (grid_h*grid_w), C, patch_size, patch_size]
      grid_h, grid_w
    """
    if x.dim() != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
    B, C, H, W = x.shape
    p = int(patch_size)
    s = int(stride)
    if p <= 0 or s <= 0:
        raise ValueError(f"patch_size and stride must be positive, got patch_size={p}, stride={s}")
    if H < p or W < p:
        raise ValueError(f"Image size {(H, W)} is smaller than patch_size={p}")
    if ((H - p) % s) != 0 or ((W - p) % s) != 0:
        raise ValueError(
            f"Invalid patch grid for HxW={H}x{W}, patch_size={p}, stride={s}."
        )
    grid_h = ((H - p) // s) + 1
    grid_w = ((W - p) // s) + 1
    cols = F.unfold(x, kernel_size=p, stride=s)  # [B, C*p*p, grid_h*grid_w]
    patches = cols.transpose(1, 2).contiguous().view(B * grid_h * grid_w, C, p, p)
    return patches, grid_h, grid_w


def fold_image_patches(
    patches: torch.Tensor,
    grid_h: int,
    grid_w: int,
    patch_size: int,
    stride: int,
) -> torch.Tensor:
    """
    Reconstruct images from patches with fold.
    Expects patches in [B, grid_h*grid_w, C, patch_size, patch_size].
    """
    if patches.dim() != 5:
        raise ValueError(f"Expected [B,P,C,H,W], got {tuple(patches.shape)}")
    B, P, C, PH, PW = patches.shape
    p = int(patch_size)
    s = int(stride)
    if PH != p or PW != p:
        raise ValueError(f"Patch tensor size ({PH}, {PW}) does not match patch_size={p}")
    expected_patches = int(grid_h) * int(grid_w)
    if P != expected_patches:
        raise ValueError(f"Expected P={expected_patches}, got P={P}")

    out_h = (int(grid_h) - 1) * s + p
    out_w = (int(grid_w) - 1) * s + p
    cols = patches.contiguous().view(B, P, C * p * p).transpose(1, 2).contiguous()
    imgs = F.fold(cols, output_size=(out_h, out_w), kernel_size=p, stride=s)

    if s < p:
        weight_cols = torch.ones((B, p * p, P), device=patches.device, dtype=patches.dtype)
        weights = F.fold(weight_cols, output_size=(out_h, out_w), kernel_size=p, stride=s)
        imgs = imgs / weights.clamp_min(1e-6)

    return imgs


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
    def __init__(
        self,
        tokens_flat: torch.Tensor,
        batch_size: int,
        num_workers: int = 2,
        coeffs_flat: Optional[torch.Tensor] = None,
        patch_pos_flat: Optional[torch.Tensor] = None,
        context_patches: int = 0,
        pad_token_id: Optional[int] = None,
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
        self.patch_pos_flat = patch_pos_flat
        self.context_patches = max(0, int(context_patches))
        self.pad_token_id = (None if pad_token_id is None else int(pad_token_id))
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
            tok_ds = SlidingWindowTokenDataset(
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
        elif (
            self.patch_pos_flat is not None
            and self.context_patches > 0
            and self.pad_token_id is not None
            and self.coeffs_flat is None
        ):
            tok_ds = PatchContextTokenDataset(
                tokens_flat=self.tokens_flat,
                patch_pos_flat=self.patch_pos_flat,
                context_patches=self.context_patches,
                pad_token_id=self.pad_token_id,
            )
        else:
            tensors = [self.tokens_flat]
            if self.coeffs_flat is not None:
                tensors.append(self.coeffs_flat)
            if self.patch_pos_flat is not None:
                tensors.append(self.patch_pos_flat)
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


class PatchContextTokenDataset(Dataset):
    """
    Training dataset for patch-grid stage-2 with local raster context.
    Each sample yields:
      tok_flat: [T]
      patch_pos_id: scalar
      ctx_tok_flat: [context_patches*T] (left-padded with pad_token_id)
      ctx_patch_pos_flat: [context_patches*T]
    """

    def __init__(
        self,
        tokens_flat: torch.Tensor,
        patch_pos_flat: torch.Tensor,
        context_patches: int,
        pad_token_id: int,
    ):
        if tokens_flat.dim() != 2:
            raise ValueError(f"tokens_flat must be [N, T], got {tuple(tokens_flat.shape)}")
        if patch_pos_flat.dim() != 1 or patch_pos_flat.shape[0] != tokens_flat.shape[0]:
            raise ValueError(
                f"patch_pos_flat must be [N] with N={tokens_flat.shape[0]}, got {tuple(patch_pos_flat.shape)}"
            )
        self.tokens_flat = tokens_flat
        self.patch_pos_flat = patch_pos_flat.to(torch.long)
        self.context_patches = max(0, int(context_patches))
        self.pad_token_id = int(pad_token_id)
        self.tokens_per_patch = int(tokens_flat.shape[1])

    def __len__(self) -> int:
        return int(self.tokens_flat.shape[0])

    def __getitem__(self, idx: int):
        tok = self.tokens_flat[idx]
        patch_pos = self.patch_pos_flat[idx]

        if self.context_patches <= 0:
            return tok, patch_pos

        ctx_len = self.context_patches * self.tokens_per_patch
        ctx_tok = torch.full((ctx_len,), self.pad_token_id, dtype=self.tokens_flat.dtype)
        ctx_patch_pos = torch.zeros((ctx_len,), dtype=torch.long)
        cur_pos = int(patch_pos.item())

        for k in range(self.context_patches):
            prev_pos = cur_pos - self.context_patches + k
            if prev_pos < 0:
                continue
            delta = cur_pos - prev_pos
            src_idx = idx - delta
            if src_idx < 0:
                continue
            if int(self.patch_pos_flat[src_idx].item()) != prev_pos:
                # Different image boundary.
                continue
            start = k * self.tokens_per_patch
            end = start + self.tokens_per_patch
            ctx_tok[start:end] = self.tokens_flat[src_idx]
            ctx_patch_pos[start:end] = prev_pos

        return tok, patch_pos, ctx_tok, ctx_patch_pos


class SlidingWindowTokenDataset(Dataset):
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


class Stage1LightningModule(pl.LightningModule):
    def __init__(
        self,
        ae: SparseDictAE,
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
        if self.trainer is not None:
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
        # Inputs/recon are normalized to [-1, 1], so peak value for PSNR is 2.0.
        psnr = 10.0 * torch.log10(4.0 / torch.clamp(recon_loss.detach(), min=1e-8))
        self.log("stage1/val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=x.size(0))
        self.log("stage1/val_psnr", psnr, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=x.size(0))

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
                    recon_vis, _, _ = self.ae(x_vis)
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
        sample_micro_batch_size: int = 8,
        sample_temperature: float = 0.8,
        sample_top_k: int = 64,
        sample_coef_extreme_penalty: float = 0.0,
        sample_image_size: Optional[int] = None,
        sample_latent_shape: Optional[Tuple[int, int]] = None,
        sample_window_stride_latent: int = 1,
        patch_grid_shape: Optional[Tuple[int, int]] = None,
        patch_size: int = 32,
        patch_stride: int = 32,
        patch_context_patches: int = 0,
        fid_real_images: Optional[torch.Tensor] = None,
        fid_num_samples: int = 64,
        fid_feature: int = 2048,
        fid_every_n_epochs: int = 1,
        coeff_loss_weight: float = 1.0,
        lr_schedule: str = "cosine",
        warmup_epochs: int = 1,
        min_lr_ratio: float = 0.1,
    ):
        super().__init__()
        self.transformer = transformer
        self.lr = float(lr)
        self.pad_token_id = int(pad_token_id)
        self.predict_coefficients = bool(self.transformer.cfg.predict_coefficients)
        self.coeff_loss_weight = float(coeff_loss_weight)
        self.out_dir = out_dir
        self.ae_for_decode = ae_for_decode
        self.H, self.W, self.D = int(H), int(W), int(D)
        self.sample_every_steps = int(sample_every_steps)
        self.sample_batch_size = int(sample_batch_size)
        self.sample_micro_batch_size = max(1, int(sample_micro_batch_size))
        self.sample_temperature = max(float(sample_temperature), 1e-8)
        self.sample_top_k = None if int(sample_top_k) <= 0 else int(sample_top_k)
        self.sample_coef_extreme_penalty = max(0.0, float(sample_coef_extreme_penalty))
        self.sample_image_size = (
            None
            if sample_image_size is None or int(sample_image_size) <= 0
            else int(sample_image_size)
        )
        if sample_latent_shape is None:
            self.sample_latent_h, self.sample_latent_w = self.H, self.W
        else:
            self.sample_latent_h = int(sample_latent_shape[0])
            self.sample_latent_w = int(sample_latent_shape[1])
        self.sample_window_stride_latent = max(1, int(sample_window_stride_latent))
        self.use_sliding_window_sampling = bool(sample_latent_shape is not None)
        if patch_grid_shape is None:
            self.patch_grid_h, self.patch_grid_w = 1, 1
        else:
            self.patch_grid_h, self.patch_grid_w = int(patch_grid_shape[0]), int(patch_grid_shape[1])
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)
        self.patch_context_patches = max(0, int(patch_context_patches))
        self.use_patch_positions = bool(getattr(self.transformer.cfg, "num_patch_positions", 0) > 0)
        self.context_tokens = int(getattr(self.transformer.cfg, "context_tokens", 0))
        self._sample_quantized_vocab_tokens = 0
        self._sample_n_bins = 0
        bottleneck = getattr(self.ae_for_decode, "bottleneck", None)
        if bottleneck is not None and bool(getattr(bottleneck, "quantize_sparse_coeffs", False)):
            num_atoms = int(getattr(bottleneck, "num_embeddings", 0))
            n_bins = int(getattr(bottleneck, "n_bins", 0))
            if num_atoms > 0 and n_bins > 1:
                self._sample_quantized_vocab_tokens = int(num_atoms * n_bins)
                self._sample_n_bins = int(n_bins)
        if self.sample_coef_extreme_penalty > 0.0 and self._sample_quantized_vocab_tokens > 0:
            center = 0.5 * float(self._sample_n_bins - 1)
            denom = max(center, 1.0)
            token_ids = torch.arange(self._sample_quantized_vocab_tokens, dtype=torch.long)
            bin_ids = torch.remainder(token_ids, self._sample_n_bins).float()
            dist = (bin_ids - center).abs() / denom
            sample_coef_penalty = self.sample_coef_extreme_penalty * (dist ** 2)
        else:
            sample_coef_penalty = torch.empty(0, dtype=torch.float32)
        self.register_buffer("_sample_coef_penalty", sample_coef_penalty, persistent=False)
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
        if self.trainer is not None:
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
        patch_pos_ids = None
        ctx_tokens = None
        ctx_patch_pos = None
        if self.predict_coefficients:
            if self.use_patch_positions:
                if len(batch) == 3:
                    tok_flat, coeff_flat, patch_pos_ids = batch
                elif len(batch) == 5:
                    tok_flat, coeff_flat, patch_pos_ids, ctx_tokens, ctx_patch_pos = batch
                else:
                    raise ValueError(f"Unexpected stage2 coefficient batch format with {len(batch)} entries.")
            else:
                tok_flat, coeff_flat = batch
            coeff_flat = coeff_flat.to(torch.float32)
        else:
            if self.use_patch_positions:
                if len(batch) == 2:
                    tok_flat, patch_pos_ids = batch
                elif len(batch) == 4:
                    tok_flat, patch_pos_ids, ctx_tokens, ctx_patch_pos = batch
                else:
                    raise ValueError(f"Unexpected stage2 patch batch format with {len(batch)} entries.")
            else:
                (tok_flat,) = batch
            coeff_flat = None
        tok_flat = tok_flat.long()
        if patch_pos_ids is not None:
            patch_pos_ids = patch_pos_ids.long()
        if ctx_tokens is not None:
            ctx_tokens = ctx_tokens.long()
        if ctx_patch_pos is not None:
            ctx_patch_pos = ctx_patch_pos.long()
        B = tok_flat.size(0)
        bos = self.transformer.bos_token_id

        if self.predict_coefficients and ctx_tokens is not None:
            raise NotImplementedError("Patch context with coefficient regression is not supported yet.")

        if ctx_tokens is not None:
            seq = torch.cat(
                [torch.full((B, 1), bos, device=tok_flat.device, dtype=torch.long), ctx_tokens, tok_flat],
                dim=1,
            )
            context_len = int(ctx_tokens.size(1))
            y = seq[:, 1:]
            y[:, :context_len] = self.pad_token_id
            if patch_pos_ids is None:
                raise ValueError("patch_pos_ids required when using patch context.")
            bos_patch = patch_pos_ids.view(B, 1)
            tok_patch = patch_pos_ids.view(B, 1).expand(B, tok_flat.size(1))
            seq_patch = torch.cat([bos_patch, ctx_patch_pos, tok_patch], dim=1)
            patch_pos_for_x = seq_patch[:, :-1]
        else:
            seq = torch.cat([torch.full((B, 1), bos, device=tok_flat.device, dtype=torch.long), tok_flat], dim=1)
            y = seq[:, 1:]
            patch_pos_for_x = patch_pos_ids

        x_in = seq[:, :-1]

        logits, coeff_pred, _ = self.transformer(x_in, patch_pos_ids=patch_pos_for_x)
        if self.predict_coefficients:
            atom_loss = F.cross_entropy(
                logits.reshape(-1, self.transformer.cfg.vocab_size),
                y.reshape(-1),
                ignore_index=self.pad_token_id,
            )
            coeff_for_target = coeff_pred.gather(-1, y.unsqueeze(-1)).squeeze(-1)
            coeff_loss = F.smooth_l1_loss(coeff_for_target.reshape(-1), coeff_flat.to(logits.device).reshape(-1))
            loss = atom_loss + self.coeff_loss_weight * coeff_loss
            self.log(
                "train/coeff_loss",
                coeff_loss,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=B,
            )
        else:
            atom_loss = F.cross_entropy(
                logits.reshape(-1, self.transformer.cfg.vocab_size),
                y.reshape(-1),
                ignore_index=self.pad_token_id,
            )
            coeff_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            loss = atom_loss
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
        self.ae_for_decode.to(self.device)
        self.ae_for_decode.eval()

    def _dist_initialized(self) -> bool:
        return bool(torch.distributed.is_available() and torch.distributed.is_initialized())

    def _dist_barrier(self) -> None:
        if self._dist_initialized():
            torch.distributed.barrier()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.sample_every_steps <= 0:
            return
        if self.global_step <= 0 or (self.global_step % self.sample_every_steps) != 0:
            return

        if self._dist_initialized():
            # Keep ranks synchronized while rank0 generates/saves.
            self._dist_barrier()
            sample_error_flag = torch.zeros(1, dtype=torch.int32, device=self.device)
            if self.trainer.is_global_zero:
                try:
                    self._sample_and_save(step=self.global_step)
                except Exception as exc:
                    sample_error_flag.fill_(1)
                    print(f"[Stage2] sampling failed at step {self.global_step}: {exc}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            print(f"[rank {self.global_rank}] entering post-sample sync (step {self.global_step})", flush=True)
            torch.distributed.all_reduce(sample_error_flag, op=torch.distributed.ReduceOp.MAX)
            self._dist_barrier()
            print(f"[rank {self.global_rank}] post-sample sync done (step {self.global_step})", flush=True)
            if int(sample_error_flag.item()) != 0:
                raise RuntimeError(f"Stage-2 sampling failed at step {self.global_step} on at least one rank.")
            return
        if self.trainer.is_global_zero:
            self._sample_and_save(step=self.global_step)

    def on_train_epoch_end(self):
        self._dist_barrier()

        if self.trainer.is_global_zero:
            os.makedirs(self.out_dir, exist_ok=True)
            torch.save(self.transformer.state_dict(), os.path.join(self.out_dir, "transformer_last.pt"))

        self._dist_barrier()

    def _maybe_resize_samples(self, imgs: torch.Tensor) -> torch.Tensor:
        if self.sample_image_size is None:
            return imgs
        if imgs.size(-2) == self.sample_image_size and imgs.size(-1) == self.sample_image_size:
            return imgs
        return F.interpolate(
            imgs,
            size=(self.sample_image_size, self.sample_image_size),
            mode="bilinear",
            align_corners=False,
        )

    def _apply_sampling_constraints(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply token constraints before multinomial sampling:
        - ban BOS/PAD content sampling
        - optionally downweight extreme coefficient bins in quantized-token mode
        """
        if 0 <= int(self.transformer.bos_token_id) < int(logits.size(-1)):
            logits[:, int(self.transformer.bos_token_id)] = float("-inf")
        if 0 <= int(self.pad_token_id) < int(logits.size(-1)):
            logits[:, int(self.pad_token_id)] = float("-inf")

        if self._sample_coef_penalty.numel() > 0:
            cutoff = min(int(self._sample_coef_penalty.numel()), int(logits.size(-1)))
            if cutoff > 0:
                logits[:, :cutoff] = logits[:, :cutoff] - self._sample_coef_penalty[:cutoff].to(logits.dtype).unsqueeze(0)
        return logits

    def _decode_generated_batch(self, gen) -> torch.Tensor:
        if self.predict_coefficients:
            flat_gen, coeff_gen = gen
            atoms_gen = flat_gen.view(-1, self.H, self.W, self.D).to(self.device)
            coeff_gen = coeff_gen.view(-1, self.H, self.W, self.D).to(self.device)
            return self.ae_for_decode.decode_from_atoms_and_coeffs(atoms_gen, coeff_gen)
        tokens_gen = gen.view(-1, self.H, self.W, self.D)
        return self.ae_for_decode.decode_from_tokens(tokens_gen.to(self.device))

    @torch.no_grad()
    def _sample_sliding_window_batch(self, step: int, batch_size: int) -> torch.Tensor:
        B = int(batch_size)
        bos_id = int(self.transformer.bos_token_id)
        win_tokens = int(self.H * self.W * self.D)
        full_h = int(self.sample_latent_h)
        full_w = int(self.sample_latent_w)
        if self.H > full_h or self.W > full_w:
            raise ValueError(
                f"Sliding window {(self.H, self.W)} must fit within full latent {(full_h, full_w)}"
            )
        # Fast path: when window == full latent field, use the transformer's
        # cached autoregressive sampler directly (much faster than per-step
        # re-encoding full context in the generic sliding-window loop).
        if self.H == full_h and self.W == full_w:
            gen = self.transformer.generate(
                batch_size=B,
                temperature=self.sample_temperature,
                top_k=self.sample_top_k,
                show_progress=False,
                progress_desc=f"[Stage2] sample step {step}",
                patch_pos_ids=None,
            )
            return self._decode_generated_batch(gen)
        full_tokens = int(self.sample_latent_h * self.sample_latent_w * self.D)
        generated = torch.zeros((B, full_h, full_w, self.D), dtype=torch.long, device=self.device)
        generated_coeffs = (
            torch.zeros((B, full_h, full_w, self.D), dtype=torch.float32, device=self.device)
            if self.predict_coefficients else None
        )
        bos = torch.full((B, 1), bos_id, dtype=torch.long, device=self.device)

        tqdm_disable = str(os.environ.get("TQDM_DISABLE", "")).strip().lower() in {"1", "true", "yes", "on"}
        steps = tqdm(
            range(full_tokens),
            desc=f"[Stage2] sliding-window sample step {step}",
            leave=False,
            dynamic_ncols=True,
            disable=tqdm_disable,
        )

        def _earliest_overlapping_start(coord: int, window: int, full: int, stride: int) -> int:
            max_start = int(full - window)
            low = max(0, int(coord - window + 1))
            high = min(int(coord), max_start)
            start = ((low + stride - 1) // stride) * stride
            if start > high:
                start = min((int(coord) // stride) * stride, max_start)
            return int(max(0, min(start, max_start)))

        for t in steps:
            spatial_idx = t // self.D
            depth_idx = t % self.D
            y = spatial_idx // full_w
            x = spatial_idx % full_w

            stride_lat = int(self.sample_window_stride_latent)
            h0 = _earliest_overlapping_start(y, self.H, full_h, stride_lat)
            w0 = _earliest_overlapping_start(x, self.W, full_w, stride_lat)
            local_y = y - h0
            local_x = x - w0
            local_pos = (local_y * self.W) + local_x
            local_t = (local_pos * self.D) + depth_idx

            window_flat = generated[:, h0 : h0 + self.H, w0 : w0 + self.W, :].contiguous().view(B, win_tokens)
            ctx = window_flat[:, :local_t]
            seq = torch.cat([bos, ctx], dim=1)
            logits, coeff_pred, _ = self.transformer(seq, patch_pos_ids=None)
            logits = logits[:, -1, :] / self.sample_temperature
            logits = self._apply_sampling_constraints(logits)
            if self.sample_top_k is not None:
                k = min(self.sample_top_k, int(logits.size(-1)))
                v, ix = torch.topk(logits, k, dim=-1)
                mask = torch.full_like(logits, float("-inf"))
                mask.scatter_(1, ix, v)
                logits = mask
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1).squeeze(1)
            generated[:, y, x, depth_idx] = nxt
            if self.predict_coefficients and coeff_pred is not None:
                c = coeff_pred[:, -1, :].gather(-1, nxt.unsqueeze(-1)).squeeze(-1)
                generated_coeffs[:, y, x, depth_idx] = c.clamp(
                    -self.transformer.cfg.coeff_max, self.transformer.cfg.coeff_max
                )

        if self.predict_coefficients:
            return self.ae_for_decode.decode_from_atoms_and_coeffs(
                generated.to(self.device), generated_coeffs.to(self.device)
            )
        return self.ae_for_decode.decode_from_tokens(generated.to(self.device))

    @torch.no_grad()
    def _generate_patch_tokens_with_context(
        self,
        patch_idx: int,
        prev_patch_tokens: list[torch.Tensor],
        batch_size: int,
        temperature: float = 1.0,
        top_k: Optional[int] = 256,
    ) -> torch.Tensor:
        if self.predict_coefficients:
            raise NotImplementedError("Patch-context sampling is only supported in quantized-token mode.")
        T = self.H * self.W * self.D
        B = int(batch_size)
        bos = torch.full((B, 1), self.transformer.bos_token_id, dtype=torch.long, device=self.device)

        ctx_needed = self.patch_context_patches
        if ctx_needed <= 0:
            seq = bos
            if self.use_patch_positions:
                patch_pos_seq = torch.full((B, 1), patch_idx, dtype=torch.long, device=self.device)
            else:
                patch_pos_seq = None
        else:
            ctx_tokens = []
            ctx_patch_pos = []
            for pidx in range(patch_idx - ctx_needed, patch_idx):
                if pidx < 0:
                    ctx_tokens.append(torch.full((B, T), self.pad_token_id, dtype=torch.long, device=self.device))
                else:
                    ctx_tokens.append(prev_patch_tokens[pidx])
                ctx_patch_pos.append(torch.full((B, T), max(pidx, 0), dtype=torch.long, device=self.device))
            ctx_tok = torch.cat(ctx_tokens, dim=1) if ctx_tokens else torch.empty((B, 0), dtype=torch.long, device=self.device)
            seq = torch.cat([bos, ctx_tok], dim=1)
            if self.use_patch_positions:
                bos_patch = torch.full((B, 1), patch_idx, dtype=torch.long, device=self.device)
                ctx_patch = torch.cat(ctx_patch_pos, dim=1) if ctx_patch_pos else torch.empty((B, 0), dtype=torch.long, device=self.device)
                patch_pos_seq = torch.cat([bos_patch, ctx_patch], dim=1)
            else:
                patch_pos_seq = None

        kv_cache = None
        nxt_patch_pos = (
            torch.full((B, 1), patch_idx, dtype=torch.long, device=self.device)
            if self.use_patch_positions else None
        )
        for _ in range(T):
            if kv_cache is None:
                inp = seq
                start_pos = 0
                pp = patch_pos_seq
            else:
                inp = seq[:, -1:]
                start_pos = seq.size(1) - 1
                pp = nxt_patch_pos

            atom_logits, _, kv_cache = self.transformer(
                inp, patch_pos_ids=pp, kv_cache=kv_cache, start_pos=start_pos,
            )
            logits = atom_logits[:, -1, :] / max(temperature, 1e-8)
            logits = self._apply_sampling_constraints(logits)
            if top_k is not None and top_k > 0:
                k = min(int(top_k), int(logits.size(-1)))
                v, ix = torch.topk(logits, k, dim=-1)
                mask = torch.full_like(logits, float("-inf"))
                mask.scatter_(1, ix, v)
                logits = mask
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            seq = torch.cat([seq, nxt], dim=1)

        return seq[:, -T:]

    @torch.no_grad()
    def _sample_patch_grid_batch(self, step: int, batch_size: int) -> torch.Tensor:
        total_patches = self.patch_grid_h * self.patch_grid_w
        generated_patch_tokens: list[torch.Tensor] = []
        patch_batches = []
        for patch_idx in range(total_patches):
            if self.patch_context_patches > 0:
                patch_tokens = self._generate_patch_tokens_with_context(
                    patch_idx=patch_idx,
                    prev_patch_tokens=generated_patch_tokens,
                    batch_size=batch_size,
                    temperature=self.sample_temperature,
                    top_k=self.sample_top_k,
                )
                generated_patch_tokens.append(patch_tokens)
                patch_img = self.ae_for_decode.decode_from_tokens(
                    patch_tokens.view(-1, self.H, self.W, self.D).to(self.device)
                )
            else:
                patch_pos_ids = None
                if self.use_patch_positions:
                    patch_pos_ids = torch.full(
                        (int(batch_size),),
                        patch_idx,
                        dtype=torch.long,
                        device=self.device,
                    )
                gen_patch = self.transformer.generate(
                    batch_size=int(batch_size),
                    temperature=self.sample_temperature,
                    top_k=self.sample_top_k,
                    show_progress=False,
                    progress_desc=f"[Stage2] patch {patch_idx + 1}/{total_patches} step {step}",
                    patch_pos_ids=patch_pos_ids,
                )
                patch_img = self._decode_generated_batch(gen_patch)
            if patch_img.size(-2) != self.patch_size or patch_img.size(-1) != self.patch_size:
                patch_img = F.interpolate(
                    patch_img,
                    size=(self.patch_size, self.patch_size),
                    mode="bilinear",
                    align_corners=False,
                )
            patch_batches.append(patch_img)

        patches_bpc = torch.stack(patch_batches, dim=1)  # [B, P, C, patch, patch]
        return fold_image_patches(
            patches_bpc,
            grid_h=self.patch_grid_h,
            grid_w=self.patch_grid_w,
            patch_size=self.patch_size,
            stride=self.patch_stride,
        )

    @torch.no_grad()
    def _sample_raw_images(self, step: int, batch_size: int) -> torch.Tensor:
        if self.use_sliding_window_sampling:
            return self._sample_sliding_window_batch(step=step, batch_size=batch_size)
        if self.patch_grid_h > 1 or self.patch_grid_w > 1:
            return self._sample_patch_grid_batch(step=step, batch_size=batch_size)
        gen = self.transformer.generate(
            batch_size=int(batch_size),
            temperature=self.sample_temperature,
            top_k=self.sample_top_k,
            show_progress=True,
            progress_desc=f"[Stage2] sample step {step}",
            patch_pos_ids=None,
        )
        return self._decode_generated_batch(gen)

    @torch.no_grad()
    def _sample_and_save(self, step: int):
        self.transformer.eval()
        self.ae_for_decode.eval()
        sample_size_str = (
            f"{self.sample_image_size}x{self.sample_image_size}"
            if self.sample_image_size is not None
            else "native"
        )
        print(
            f"[Stage2] sampling at step {step} "
            f"(batch_size={self.sample_batch_size}, output_size={sample_size_str}, "
            f"temp={self.sample_temperature:.3f}, "
            f"top_k={self.sample_top_k if self.sample_top_k is not None else 'none'}, "
            f"coef_penalty={self.sample_coef_extreme_penalty:.3f}, "
            f"micro_batch={self.sample_micro_batch_size})..."
        )
        if self.use_sliding_window_sampling:
            print(
                f"[Stage2] latent sliding-window sampling enabled "
                f"(window={self.H}x{self.W}, full_latent={self.sample_latent_h}x{self.sample_latent_w})"
            )
        elif self.patch_grid_h > 1 or self.patch_grid_w > 1:
            print(
                f"[Stage2] patch-grid sampling enabled "
                f"(grid={self.patch_grid_h}x{self.patch_grid_w}, patch={self.patch_size}, "
                f"stride={self.patch_stride}, context_patches={self.patch_context_patches})"
            )

        total_batch = int(self.sample_batch_size)
        micro_batch = max(1, min(int(self.sample_micro_batch_size), total_batch))
        raw_parts = []
        for start in range(0, total_batch, micro_batch):
            cur_bs = min(micro_batch, total_batch - start)
            if total_batch > micro_batch:
                chunk_idx = (start // micro_batch) + 1
                chunk_total = int(math.ceil(total_batch / float(micro_batch)))
                print(f"[Stage2] sampling chunk {chunk_idx}/{chunk_total} (batch={cur_bs})")
            raw_chunk = self._sample_raw_images(step=step, batch_size=cur_bs)
            raw_parts.append(raw_chunk)
        raw_imgs = raw_parts[0] if len(raw_parts) == 1 else torch.cat(raw_parts, dim=0)
        sample_imgs = self._maybe_resize_samples(raw_imgs)
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
        del raw_imgs, sample_imgs, raw_parts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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


@torch.no_grad()
def precompute_patch_grid_tokens(
    ae: SparseDictAE,
    loader: DataLoader,
    device: torch.device,
    patch_size: int,
    patch_stride: int,
    patch_encode_batch_size: int = 512,
    max_items: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, int, int, int, int, int]:
    """
    Encode full-resolution images as patch tokens for stage-2 patch-grid generation.
    Returns:
      tokens_flat: [N_patches, H*W*D] int32
      coeffs_flat: [N_patches, H*W*D] float32 (None if quantized)
      patch_pos_flat: [N_patches] int64 patch index in raster order
      H, W, D: latent patch shape
      grid_h, grid_w: patch grid shape over the full image
    """
    ae.eval()
    all_tokens = []
    all_coeffs = []
    all_patch_pos = []
    seen_images = 0
    H = W = D = None
    grid_h = grid_w = None
    chunk_bs = max(1, int(patch_encode_batch_size))

    for x, _ in tqdm(loader, desc="[Stage2] precompute patch-grid tokens"):
        if max_items is not None:
            remaining = int(max_items) - seen_images
            if remaining <= 0:
                break
            if x.size(0) > remaining:
                x = x[:remaining]
        if x.numel() == 0:
            continue

        patches, gh, gw = unfold_image_patches(x, patch_size=int(patch_size), stride=int(patch_stride))
        if grid_h is None:
            grid_h, grid_w = gh, gw
        elif gh != grid_h or gw != grid_w:
            raise ValueError(
                f"Inconsistent patch grid: expected {(grid_h, grid_w)}, got {(gh, gw)}"
            )

        patch_pos = torch.arange(gh * gw, dtype=torch.long).unsqueeze(0).expand(x.size(0), -1).reshape(-1)

        for start in range(0, patches.size(0), chunk_bs):
            end = min(start + chunk_bs, patches.size(0))
            patch_batch = patches[start:end].to(device)
            if ae.bottleneck.quantize_sparse_coeffs:
                tokens, h, w = ae.encode_to_tokens(patch_batch)
                coeffs = None
            else:
                tokens, coeffs, h, w = ae.encode_to_atoms_and_coeffs(patch_batch)

            if H is None:
                H, W = h, w
                D = tokens.shape[-1]

            all_tokens.append(tokens.view(tokens.size(0), -1).to(torch.int32).cpu())
            all_patch_pos.append(patch_pos[start:end].to(torch.long).cpu())
            if coeffs is not None:
                all_coeffs.append(coeffs.view(coeffs.size(0), -1).to(torch.float32).cpu())

        seen_images += x.size(0)
        if max_items is not None and seen_images >= max_items:
            break

    if not all_tokens:
        raise RuntimeError("No patch tokens were generated.")

    tokens_flat = torch.cat(all_tokens, dim=0)
    patch_pos_flat = torch.cat(all_patch_pos, dim=0)
    coeffs_flat = torch.cat(all_coeffs, dim=0) if len(all_coeffs) > 0 else None
    return tokens_flat, coeffs_flat, patch_pos_flat, H, W, D, grid_h, grid_w


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
    parser.add_argument("--image_size", type=int, default=None, help="Legacy alias for --crop_size.")
    parser.add_argument(
        "--resize_size",
        type=int,
        default=256,
        help="Resize shortest side to this size (keep aspect ratio) before cropping.",
    )
    parser.add_argument("--crop_size", type=int, default=32, help="Random crop size used for training/tokenization.")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)

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
    parser.add_argument(
        "--stage1_init_ckpt",
        type=str,
        default=None,
        help="Optional path to stage-1 AE state_dict (.pt) used to warm-start stage-1 training.",
    )
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
    parser.add_argument("--n_bins", type=int, default=129, help="Coefficient quantization bins (higher = lower quantization error, larger vocab).")
    parser.add_argument("--coef_max", type=float, default=3.0, help="Coefficient clipping range for quantization in [-coef_max, coef_max].")
    parser.add_argument("--coef_quantization", type=str, default="mu_law", choices=["uniform", "mu_law"])
    parser.add_argument("--coef_mu", type=float, default=50.0, help="Mu for mu-law quantization (only used when coef_quantization=mu_law).")
    parser.add_argument("--commitment_cost", type=float, default=0.25)
    parser.add_argument(
        "--quantize_sparse_coeffs",
        action="store_true",
        default=True,
        help="Quantize sparse coefficients into token IDs (legacy).",
    )
    parser.add_argument(
        "--no_quantize_sparse_coeffs",
        action="store_false",
        dest="quantize_sparse_coeffs",
        help="Disable quantized coefficients and use a coefficient regressor head.",
    )

    # Stage-2 (Transformer)
    parser.add_argument("--stage2_epochs", type=int, default=10)
    parser.add_argument("--stage2_lr", type=float, default=3e-4)
    parser.add_argument("--stage2_lr_schedule", type=str, default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--stage2_warmup_epochs", type=int, default=1)
    parser.add_argument("--stage2_min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--stage2_batch_size", type=int, default=16)
    parser.add_argument("--stage2_grad_accum", type=int, default=4, help="Gradient accumulation steps for stage-2.")
    parser.add_argument("--stage2_sample_every_steps", type=int, default=200)
    parser.add_argument("--stage2_sample_batch_size", type=int, default=32)
    parser.add_argument(
        "--stage2_sample_micro_batch_size",
        type=int,
        default=8,
        help="Generate stage-2 samples in micro-batches to avoid long stalls/OOM while keeping total sample count.",
    )
    parser.add_argument(
        "--stage2_sample_temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for stage-2 previews (lower usually improves coherence).",
    )
    parser.add_argument(
        "--stage2_sample_top_k",
        type=int,
        default=64,
        help="Top-k for stage-2 preview sampling (<=0 disables top-k filtering).",
    )
    parser.add_argument(
        "--stage2_sample_coef_extreme_penalty",
        type=float,
        default=0.0,
        help="Logit penalty for extreme coefficient bins during stage-2 sampling (0 disables).",
    )
    parser.add_argument(
        "--stage2_resume_from_last",
        action="store_true",
        default=True,
        help="If stage2/transformer_last.pt exists in out_dir, load it before stage-2 training.",
    )
    parser.add_argument(
        "--no_stage2_resume_from_last",
        action="store_false",
        dest="stage2_resume_from_last",
        help="Disable automatic stage-2 resume from stage2/transformer_last.pt.",
    )
    parser.add_argument(
        "--stage2_patchify",
        action="store_true",
        default=False,
        help="Train stage-2 on full images split into patch tokens via unfold/fold.",
    )
    parser.add_argument(
        "--stage2_sliding_window",
        action="store_true",
        default=False,
        help="Train stage-2 on full-image latent sequences with a fixed sliding attention window.",
    )
    parser.add_argument("--stage2_patch_size", type=int, default=32, help="Patch size for stage-2 patchify mode.")
    parser.add_argument("--stage2_patch_stride", type=int, default=32, help="Patch stride for stage-2 patchify mode.")
    parser.add_argument(
        "--stage2_patch_context_patches",
        type=int,
        default=1,
        help="Number of previous raster patches used as context during stage-2 patch sampling/training.",
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
        "--stage2_patch_encode_batch_size",
        type=int,
        default=512,
        help="Patch mini-batch size during stage-2 token precompute.",
    )
    parser.add_argument(
        "--stage2_sample_image_size",
        type=int,
        default=256,
        help="Output size for stage-2 sample grids (upsampled if needed).",
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
    parser.add_argument("--tf_d_model", type=int, default=384)
    parser.add_argument("--tf_heads", type=int, default=8)
    parser.add_argument("--tf_layers", type=int, default=8)
    parser.add_argument("--tf_ff", type=int, default=1536)
    parser.add_argument("--tf_dropout", type=float, default=0.1)
    parser.add_argument(
        "--stage2_coeff_loss_weight",
        type=float,
        default=2.0,
        help="Coefficient regression loss weight when using continuous coefficients.",
    )
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
    if args.stage2_sample_batch_size <= 0:
        raise ValueError("stage2_sample_batch_size must be > 0.")
    if args.stage2_sample_micro_batch_size <= 0:
        raise ValueError("stage2_sample_micro_batch_size must be > 0.")
    if args.stage2_sample_temperature <= 0.0:
        raise ValueError("stage2_sample_temperature must be > 0.")
    if args.stage2_sample_coef_extreme_penalty < 0.0:
        raise ValueError("stage2_sample_coef_extreme_penalty must be >= 0.")
    if args.ae_num_downsamples <= 0:
        raise ValueError(f"ae_num_downsamples must be positive, got {args.ae_num_downsamples}")
    if args.image_size is not None:
        args.crop_size = int(args.image_size)
    args.image_size = int(args.crop_size)
    if args.resize_size < args.image_size:
        raise ValueError(
            f"resize_size ({args.resize_size}) must be >= crop_size ({args.image_size})."
        )
    if args.resize_size >= (2 * args.image_size):
        print(
            f"[Data] WARNING: resize_size={args.resize_size} is much larger than crop_size={args.image_size}. "
            "This can make stage-1 reconstruction harder and depress PSNR."
        )
    if args.stage2_patchify and args.stage2_sliding_window:
        raise ValueError("stage2_patchify and stage2_sliding_window are mutually exclusive.")
    if args.stage2_patchify:
        if args.stage2_patch_size <= 0 or args.stage2_patch_stride <= 0:
            raise ValueError(
                "stage2_patch_size and stage2_patch_stride must be positive."
            )
        if args.stage2_patch_context_patches < 0:
            raise ValueError("stage2_patch_context_patches must be >= 0.")
        if args.resize_size < args.stage2_patch_size:
            raise ValueError(
                f"resize_size ({args.resize_size}) must be >= stage2_patch_size ({args.stage2_patch_size})."
            )
        if ((args.resize_size - args.stage2_patch_size) % args.stage2_patch_stride) != 0:
            raise ValueError(
                "resize_size, stage2_patch_size, and stage2_patch_stride produce a non-integer patch grid."
            )
    if args.stage2_sliding_window:
        if args.stage2_window_latent_h <= 0 or args.stage2_window_latent_w <= 0:
            raise ValueError("stage2_window_latent_h and stage2_window_latent_w must be positive.")
        if args.stage2_window_stride_latent <= 0:
            raise ValueError("stage2_window_stride_latent must be positive.")
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
        f"resize_size={args.resize_size} crop_size={args.image_size} "
        f"stage2_patchify={args.stage2_patchify} "
        f"stage2_sliding_window={args.stage2_sliding_window} "
        f"context_patches={args.stage2_patch_context_patches if args.stage2_patchify else 0}"
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

    def _build_ae(quantize_sparse_coeffs: bool = args.quantize_sparse_coeffs) -> SparseDictAE:
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
            n_bins=args.n_bins,
            coef_max=args.coef_max,
            coef_quantization=args.coef_quantization,
            coef_mu=args.coef_mu,
            quantize_sparse_coeffs=quantize_sparse_coeffs,
            out_tanh=True,
        )

    def _load_ae_weights(ae_model: SparseDictAE, ckpt_path: str, tag: str = "Stage1"):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"{tag} checkpoint not found at {ckpt_path}")
        try:
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(ckpt_path, map_location="cpu")
        ae_model.load_state_dict(state_dict)
        print(f"[{tag}] loaded AE weights from {ckpt_path}")

    def _load_best_ae_weights(ae_model: SparseDictAE):
        best_path = os.path.join(stage1_dir, "ae_best.pt")
        _load_ae_weights(ae_model, best_path, tag="Stage1")

    def _run_stage2_lightning(
        tokens_flat: torch.Tensor,
        coeffs_flat: Optional[torch.Tensor],
        patch_pos_flat: Optional[torch.Tensor],
        quantize_sparse_coeffs: bool,
        H: int,
        W: int,
        D: int,
        patch_grid_shape: Optional[Tuple[int, int]],
        patch_size: int,
        patch_stride: int,
        sample_latent_shape: Optional[Tuple[int, int]],
        sliding_window_stride_latent: Optional[int],
        ae_model: SparseDictAE,
        fid_real_images: Optional[torch.Tensor],
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("Stage-2 Lightning multi-GPU training requires CUDA.")
        if torch.cuda.device_count() < args.stage2_devices:
            raise RuntimeError(
                f"Requested {args.stage2_devices} GPUs, but only {torch.cuda.device_count()} detected."
            )
        if not quantize_sparse_coeffs and coeffs_flat is None:
            raise ValueError("coeffs_flat is required when quantize_sparse_coeffs=False")
        if patch_grid_shape is not None and args.stage2_patch_context_patches > 0 and patch_pos_flat is None:
            raise ValueError("patch_pos_flat is required for patch context training.")
        if sample_latent_shape is not None and patch_grid_shape is not None:
            raise ValueError("sample_latent_shape and patch_grid_shape cannot both be set.")

        stage2_dm = Stage2TokenDataModule(
            tokens_flat=tokens_flat,
            batch_size=args.stage2_batch_size,
            coeffs_flat=None if coeffs_flat is None else coeffs_flat,
            patch_pos_flat=patch_pos_flat,
            context_patches=(args.stage2_patch_context_patches if patch_grid_shape is not None else 0),
            pad_token_id=ae_model.bottleneck.pad_token_id,
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

        patch_positions = int(patch_grid_shape[0] * patch_grid_shape[1]) if patch_grid_shape is not None else 0
        context_tokens = 0
        if patch_grid_shape is not None and args.stage2_patch_context_patches > 0:
            context_tokens = int(args.stage2_patch_context_patches) * int(H * W * D)
        cfg = RQTransformerConfig(
            vocab_size=ae_model.bottleneck.vocab_size,
            H=H,
            W=W,
            D=D,
            num_patch_positions=patch_positions,
            context_tokens=context_tokens,
            predict_coefficients=not quantize_sparse_coeffs,
            coeff_loss_weight=args.stage2_coeff_loss_weight,
            coeff_max=args.coef_max,
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
        if args.stage2_resume_from_last:
            stage2_last_path = os.path.join(stage2_dir, "transformer_last.pt")
            if os.path.exists(stage2_last_path):
                try:
                    stage2_state = torch.load(stage2_last_path, map_location="cpu", weights_only=True)
                except TypeError:
                    stage2_state = torch.load(stage2_last_path, map_location="cpu")
                model_state = transformer.state_dict()
                filtered = {
                    k: v for k, v in stage2_state.items()
                    if k in model_state and model_state[k].shape == v.shape
                }
                skipped = set(stage2_state.keys()) - set(filtered.keys())
                transformer.load_state_dict(filtered, strict=False)
                if skipped:
                    print(f"[Stage2] skipped {len(skipped)} incompatible keys: {sorted(skipped)}")
                print(f"[Stage2] resumed transformer weights from {stage2_last_path}")
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
            sample_micro_batch_size=args.stage2_sample_micro_batch_size,
            sample_temperature=args.stage2_sample_temperature,
            sample_top_k=args.stage2_sample_top_k,
            sample_coef_extreme_penalty=args.stage2_sample_coef_extreme_penalty,
            sample_image_size=args.stage2_sample_image_size,
            sample_latent_shape=sample_latent_shape,
            sample_window_stride_latent=(
                int(sliding_window_stride_latent)
                if (sample_latent_shape is not None and sliding_window_stride_latent is not None)
                else 1
            ),
            patch_grid_shape=patch_grid_shape,
            patch_size=patch_size,
            patch_stride=patch_stride,
            patch_context_patches=(args.stage2_patch_context_patches if patch_grid_shape is not None else 0),
            fid_real_images=fid_real_images,
            fid_num_samples=args.stage2_fid_num_samples,
            fid_feature=args.stage2_fid_feature,
            fid_every_n_epochs=args.stage2_fid_every_n_epochs,
            lr_schedule=args.stage2_lr_schedule,
            warmup_epochs=args.stage2_warmup_epochs,
            min_lr_ratio=args.stage2_min_lr_ratio,
            coeff_loss_weight=args.stage2_coeff_loss_weight,
        )

        effective_strategy = (args.stage2_strategy if args.stage2_devices > 1 else "auto")
        if effective_strategy == "ddp_fork" and torch.cuda.is_initialized():
            print("[Stage2] CUDA already initialized; falling back from ddp_fork to ddp.")
            effective_strategy = "ddp"
        if args.stage2_devices > 1:
            os.environ["LASER_DDP_PHASE"] = "stage2"
        if args.stage2_devices > 1:
            _ensure_free_master_port_for_ddp("Stage2")
            print(
                f"[Stage2] rendezvous endpoint: "
                f"{os.environ.get('MASTER_ADDR', '127.0.0.1')}:{os.environ.get('MASTER_PORT')}"
            )

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
            accumulate_grad_batches=args.stage2_grad_accum,
        )
        trainer.fit(stage2_module, datamodule=stage2_dm)

    # During stage-2 DDP script re-entry (re-exec after stage-1, OR worker
    # spawned by the stage-2 trainer), skip stage-1 and tokenization.
    if os.environ.get("LASER_DDP_PHASE") == "stage2":
        if not os.path.exists(token_cache_path):
            raise FileNotFoundError(f"Missing token cache: {token_cache_path}")
        try:
            cache = torch.load(token_cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            cache = torch.load(token_cache_path, map_location="cpu")
        tokens_flat = cache["tokens_flat"]
        coeffs_flat = cache.get("coeffs_flat")
        patch_pos_flat = cache.get("patch_pos_flat")
        quantize_sparse_coeffs = cache.get("quantize_sparse_coeffs", True)
        H, W, D = cache["shape"]
        patch_grid_shape = cache.get("patch_grid_shape")
        patch_size = int(cache.get("patch_size", args.stage2_patch_size))
        patch_stride = int(cache.get("patch_stride", args.stage2_patch_stride))
        sample_latent_shape = cache.get("sample_latent_shape")
        sliding_window_stride_latent = cache.get(
            "sliding_window_stride_latent",
            cache.get("sliding_window_stride_tokens"),
        )
        fid_real_images = cache.get("fid_real_images")
        ae = _build_ae(quantize_sparse_coeffs=quantize_sparse_coeffs)
        _load_best_ae_weights(ae)
        _run_stage2_lightning(
            tokens_flat=tokens_flat,
            coeffs_flat=coeffs_flat,
            patch_pos_flat=patch_pos_flat,
            quantize_sparse_coeffs=quantize_sparse_coeffs,
            H=H,
            W=W,
            D=D,
            patch_grid_shape=patch_grid_shape,
            patch_size=patch_size,
            patch_stride=patch_stride,
            sample_latent_shape=sample_latent_shape,
            sliding_window_stride_latent=sliding_window_stride_latent,
            ae_model=ae,
            fid_real_images=fid_real_images,
        )
        return

    # Normalize to [-1, 1].
    # Keep aspect ratio on resize, then center-crop to a square canvas so
    # stage-1 random crops come from a more stable distribution.
    train_tfm = transforms.Compose([
        transforms.Resize(args.resize_size),
        transforms.CenterCrop((args.resize_size, args.resize_size)),
        transforms.RandomCrop((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    eval_tfm = transforms.Compose([
        transforms.Resize(args.resize_size),
        transforms.CenterCrop((args.resize_size, args.resize_size)),
        transforms.CenterCrop((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    val_vis_tfm = transforms.Compose([
        transforms.Resize(args.resize_size),
        transforms.CenterCrop((args.resize_size, args.resize_size)),
        transforms.CenterCrop((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    stage2_source_tfm = transforms.Compose([
        transforms.Resize(args.resize_size),
        transforms.CenterCrop((args.resize_size, args.resize_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    if args.dataset == "cifar10":
        train_set = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=train_tfm)
        val_set = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=eval_tfm)
        val_vis_set = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=val_vis_tfm)
        if args.stage2_patchify or args.stage2_sliding_window:
            stage2_source_set = datasets.CIFAR10(
                root=args.data_dir,
                train=True,
                download=True,
                transform=stage2_source_tfm,
            )
        else:
            stage2_source_set = train_set
    elif args.dataset == "celeba":
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
        if args.stage2_patchify or args.stage2_sliding_window:
            stage2_source_set = Subset(stage2_source_full, train_indices)
        else:
            stage2_source_set = train_set
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

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
    if args.stage1_init_ckpt:
        _load_ae_weights(ae, args.stage1_init_ckpt, tag="Stage1 init")
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
            val_vis_images=stage1_val_vis_images,
            fid_num_samples=args.fid_num_samples,
            fid_feature=args.fid_feature,
            fid_compute_batch_size=args.fid_compute_batch_size,
            lr_schedule=args.stage1_lr_schedule,
            warmup_epochs=args.stage1_warmup_epochs,
            min_lr_ratio=args.stage1_min_lr_ratio,
        )
        if args.stage1_devices > 1:
            os.environ["LASER_DDP_PHASE"] = "stage1"
            _ensure_free_master_port_for_ddp("Stage1")
            print(
                f"[Stage1] rendezvous endpoint: "
                f"{os.environ.get('MASTER_ADDR', '127.0.0.1')}:{os.environ.get('MASTER_PORT')}"
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

    # Reuse cached tokens when stage-1 was not (re-)trained this run.
    _use_token_cache = (
        args.stage1_epochs <= 0
        and os.path.exists(token_cache_path)
    )
    if _use_token_cache:
        print(f"[Stage2] loading cached tokens from {token_cache_path}")
        try:
            cache = torch.load(token_cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            cache = torch.load(token_cache_path, map_location="cpu")
        tokens_flat = cache["tokens_flat"]
        coeffs_flat = cache.get("coeffs_flat")
        patch_pos_flat = cache.get("patch_pos_flat")
        H, W, D = cache["shape"]
        patch_grid_shape = cache.get("patch_grid_shape")
        patch_size = int(cache.get("patch_size", args.stage2_patch_size))
        patch_stride = int(cache.get("patch_stride", args.stage2_patch_stride))
        sample_latent_shape = cache.get("sample_latent_shape")
        sliding_window_stride_latent = cache.get(
            "sliding_window_stride_latent",
            cache.get("sliding_window_stride_tokens"),
        )
        fid_real_images = cache.get("fid_real_images")
    else:
        encode_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ae = ae.to(encode_device)
        token_loader_batch_size = int(args.batch_size)
        if args.stage2_patchify or args.stage2_sliding_window:
            token_loader_batch_size = max(1, min(token_loader_batch_size, 16))
        token_loader = DataLoader(
            stage2_source_set,
            batch_size=token_loader_batch_size,
            shuffle=False,
            num_workers=args.token_num_workers,
            pin_memory=True,
            persistent_workers=(args.token_num_workers > 0),
        )
        patch_pos_flat = None
        patch_grid_shape = None
        patch_size = int(args.stage2_patch_size)
        patch_stride = int(args.stage2_patch_stride)
        sample_latent_shape = None
        sliding_window_stride_latent = None
        if args.stage2_patchify:
            (
                tokens_flat,
                coeffs_flat,
                patch_pos_flat,
                H,
                W,
                D,
                patch_grid_h,
                patch_grid_w,
            ) = precompute_patch_grid_tokens(
                ae,
                token_loader,
                encode_device,
                patch_size=patch_size,
                patch_stride=patch_stride,
                patch_encode_batch_size=args.stage2_patch_encode_batch_size,
                max_items=min(args.token_subset, len(stage2_source_set)),
            )
            patch_grid_shape = (patch_grid_h, patch_grid_w)
        elif args.stage2_sliding_window:
            tokens_flat, coeffs_flat, full_H, full_W, D = precompute_tokens(
                ae,
                token_loader,
                encode_device,
                max_items=min(args.token_subset, len(stage2_source_set)),
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
                max_items=min(args.token_subset, len(stage2_source_set)),
            )
    if not _use_token_cache:
        fid_real_loader = DataLoader(
            stage2_source_set if (args.stage2_patchify or args.stage2_sliding_window) else train_set,
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
        torch.save(
            {
                "tokens_flat": tokens_flat,
                "coeffs_flat": coeffs_flat,
                "patch_pos_flat": patch_pos_flat,
                "quantize_sparse_coeffs": args.quantize_sparse_coeffs,
                "shape": (H, W, D),
                "patch_grid_shape": patch_grid_shape,
                "patch_size": patch_size,
                "patch_stride": patch_stride,
                "sample_latent_shape": sample_latent_shape,
                "sliding_window_stride_latent": sliding_window_stride_latent,
                "fid_real_images": fid_real_images,
            },
            token_cache_path,
        )

    print(f"[Stage2] token dataset: {tokens_flat.shape}   (H={H}, W={W}, D={D})")
    if patch_grid_shape is not None:
        print(
            f"[Stage2] patch grid: {patch_grid_shape[0]}x{patch_grid_shape[1]} "
            f"(patch_size={patch_size}, stride={patch_stride})"
        )
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
    if args.stage1_epochs <= 0:
        # No prior DDP group to clean up; call directly. Stage-2 DDP workers
        # will re-enter via LASER_DDP_PHASE=stage2 and load from cache.
        _run_stage2_lightning(
            tokens_flat=tokens_flat,
            coeffs_flat=coeffs_flat,
            patch_pos_flat=patch_pos_flat,
            quantize_sparse_coeffs=args.quantize_sparse_coeffs,
            H=H,
            W=W,
            D=D,
            patch_grid_shape=patch_grid_shape,
            patch_size=patch_size,
            patch_stride=patch_stride,
            sample_latent_shape=sample_latent_shape,
            sliding_window_stride_latent=sliding_window_stride_latent,
            ae_model=ae,
            fid_real_images=fid_real_images,
        )
        return
    # Re-exec into a clean process before launching stage-2 DDP.
    # After stage-1 DDP, CUDA and process groups are already initialised;
    # a fresh process avoids hangs from nested DDP contexts.
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
