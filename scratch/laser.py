
"""
laser.py

Minimal LASER training script.

It keeps the core two-stage workflow:
  - stage 1: train the LASER autoencoder
  - stage 2: train a transformer prior on flattened sparse tokens

The default dataset is CelebA under ../../data/celeba relative to this file.
For multi-GPU runs, launch with torchrun.

@Copyright 2026 Xin Li (helloimlixin@gmail.com)
"""
import argparse
from math import e
import os
import socket
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.linalg import sqrtm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision import datasets, transforms, utils
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
import wandb


# -----------------------------
# VQ-VAE style building blocks
# -----------------------------

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"} # only support jpg, jpeg, and png


def soft_clamp(x: torch.Tensor, max_val: float) -> torch.Tensor:
    """Smoothly bound coefficients to a symmetric range during sampling/regression."""
    return max_val * torch.tanh(x / max(max_val, 1e-8))


def safe_atanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Numerically safe inverse tanh for latents expected to live in [-1, 1]."""
    return torch.atanh(x.clamp(min=-(1.0 - eps), max=(1.0 - eps)))


@dataclass
class DistributedContext:
    """
    Keeps all torch.distributed state in one place.

    Terminology:
    - world_size: total number of worker processes launched by `torchrun`
    - rank: global process id in `[0, world_size)`
    - local_rank: GPU index used by this process on the current machine

    In single-process runs, `enabled=False` and the helper methods become no-ops.
    That lets the rest of the training code call the same API in both single-GPU
    and multi-GPU modes.
    """
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: Optional[str] = None

    @classmethod
    def initialize(cls) -> "DistributedContext":
        # `torchrun` communicates distributed topology through environment variables.
        # If WORLD_SIZE is 1, we keep the same code path but treat distribution as disabled.
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size <= 1:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return cls(
                enabled=False,
                rank=0,
                local_rank=0,
                world_size=1,
                device=device,
                backend=None,
            )

        if not torch.cuda.is_available():
            raise RuntimeError("Multi-GPU training requires CUDA.")

        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        # Each process must bind itself to exactly one GPU before initializing NCCL.
        # With `torchrun --nproc_per_node=N`, local_rank is the GPU id for this node.
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            # NCCL = NVIDIA Collective Communications Library.
            # It is PyTorch's standard high-performance backend for multi-GPU CUDA training.
            # DDP uses it for collectives such as:
            # - gradient synchronization after backward()
            # - all_reduce for cross-rank metrics
            # - barrier for "everyone wait here" synchronization points
            # In short: each GPU process trains on its own shard of data, and NCCL is
            # the communication layer that keeps those processes in sync.
            dist.init_process_group(backend="nccl")

        return cls(
            enabled=True,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=torch.device("cuda", local_rank),
            backend=(dist.get_backend() if dist.is_initialized() else "nccl"),
        )

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @property
    def pin_memory(self) -> bool:
        return self.device.type == "cuda"

    @property
    def prefix(self) -> str:
        return f"[Rank {self.rank}]"

    def debug_summary(self) -> str:
        parts = [
            f"enabled={self.enabled}",
            f"rank={self.rank}/{self.world_size}",
            f"local_rank={self.local_rank}",
            f"device={self.device}",
            f"pid={os.getpid()}",
            f"host={socket.gethostname()}",
        ]
        if self.backend is not None:
            parts.append(f"backend={self.backend}")
        if self.device.type == "cuda":
            current_device = torch.cuda.current_device()
            parts.append(f"cuda_device={current_device}")
            parts.append(f"cuda_name={torch.cuda.get_device_name(current_device)}")
        return " ".join(parts)

    def debug(self, message: str, *, enabled: bool, all_ranks: bool = False):
        if enabled and (all_ranks or self.is_main_process):
            print(f"{self.prefix} {message}", flush=True)

    def set_epoch(self, sampler: Optional[DistributedSampler], epoch: int):
        if sampler is not None:
            sampler.set_epoch(epoch)

    def barrier(self):
        if not self.enabled:
            return
        if self.device.type == "cuda":
            # A barrier is a rendezvous point: every rank must reach it before any
            # rank is allowed to continue. We use it before rank-0-only file writes
            # and sampling so the workers stay in lock-step.
            # NCCL warns if barrier cannot infer the rank-to-device mapping.
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()

    def mean(self, value: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return value.detach()
        # Every rank computes the loss on a different mini-batch. For logging, we
        # usually want the cross-rank average instead of rank 0's local value.
        reduced = value.detach().clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced /= self.world_size
        return reduced

    def all_reduce_sum_(self, value: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            # SUM is useful when each rank has accumulated a partial numerator or
            # counter. After the reduction, every rank holds the global total.
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return value

    def reduce_sums_(self, *values: torch.Tensor):
        for value in values:
            self.all_reduce_sum_(value)

    def make_sampler(self, dataset, *, shuffle: bool) -> Optional[DistributedSampler]:
        if not self.enabled:
            return None
        # DistributedSampler shards the dataset so ranks do not all train on the
        # same examples each step.
        return DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle,
        )

    def wrap_model(self, module: nn.Module) -> nn.Module:
        if not self.enabled:
            return module
        # DDP = DistributedDataParallel.
        # It runs one training process per GPU, with one model replica per process.
        # Each rank sees a different shard of the data, computes its own forward and
        # backward pass, then DDP synchronizes gradients across ranks so every replica
        # applies the same optimizer step and stays identical.
        # Compared with the older DataParallel approach, DDP is the standard and
        # scalable way to do multi-GPU training in PyTorch.
        # DDP keeps one model replica per process/GPU and synchronizes gradients
        # during backward() so optimization stays equivalent to data parallel training.
        return DDP(module, device_ids=[self.local_rank], output_device=self.local_rank)

    def run_main(self, fn: Callable[[], object], *, sync: bool = False) -> Optional[object]:
        # Most shared side effects (saving files, generating samples, writing caches)
        # should happen only on rank 0. `sync=True` wraps that work in barriers so the
        # other ranks wait instead of racing ahead.
        if sync:
            self.barrier()
        result = fn() if self.is_main_process else None
        if sync:
            self.barrier()
        return result

    def cleanup(self):
        if dist.is_available() and dist.is_initialized():
            # Explicit cleanup avoids leaving the process group alive after early exits.
            dist.destroy_process_group()


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


class IndexedDataset(Dataset):
    """Wraps a dataset so precompute can keep a stable sample order across ranks."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        if isinstance(sample, (tuple, list)):
            image = sample[0]
        else:
            image = sample
        return {"image": image, "index": idx}

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
# Dictionary learning bottleneck (batch OMP)
# -----------------------------

class DictionaryLearning(nn.Module):
    """
    Dictionary-learning bottleneck with batched Orthogonal Matching Pursuit (OMP) sparse coding.
    Tokenization modes:
    - Quantized-mode: alternating token pairs [atom_id, coeff_bin + num_atoms].
    - Regressor-mode: token = atom_id only, coefficients are modeled with a separate head.

    Uses a reshape fast path for non-overlapping patches and a strided-view/scatter path for overlap.

    Outputs, per latent patch, a token stack of length:
    - 2 * sparsity_level in quantized mode
    - sparsity_level in regressor mode

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
        quantize_sparse_coeffs: bool = False,
        patch_size: int = 8,
        patch_stride: int = 4,
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
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)
        self.commitment_cost = float(commitment_cost)
        self.epsilon = float(epsilon)
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {self.patch_size}")
        if self.patch_stride <= 0:
            raise ValueError(f"patch_stride must be positive, got {self.patch_stride}")
        if self.patch_stride > self.patch_size:
            raise ValueError(
                f"patch_stride ({self.patch_stride}) must be <= patch_size ({self.patch_size})"
            )
        self.patch_dim = self.embedding_dim * self.patch_size * self.patch_size
        self._patch_reconstruct_cache = {}

        # Dictionary shape [C * patch_size^2, K].
        self.dictionary = nn.Parameter(torch.randn(self.patch_dim, self.num_embeddings) * 0.02)

        # Coefficient bin centers for uniform coefficient quantization.
        centers = torch.linspace(-self.coef_max, self.coef_max, steps=self.n_bins)
        self.register_buffer("coef_bin_centers", centers)
        coef_bin_step = 0.0 if self.n_bins <= 1 else (2.0 * self.coef_max) / float(self.n_bins - 1)
        coef_bin_scale = 0.0 if self.coef_max <= 0.0 else float(self.n_bins - 1) / (2.0 * self.coef_max)
        self.register_buffer("coef_bin_step", torch.tensor(coef_bin_step))
        self.register_buffer("coef_bin_scale", torch.tensor(coef_bin_scale))

        # Special tokens (for the transformer)
        coeff_vocab_size = self.n_bins if self.quantize_sparse_coeffs else 0
        self.coeff_token_offset = self.num_embeddings
        self.token_depth = self.sparsity_level * (2 if self.quantize_sparse_coeffs else 1)
        self.content_vocab_size = self.num_embeddings + coeff_vocab_size
        self.pad_token_id = self.content_vocab_size
        self.bos_token_id = self.pad_token_id + 1
        self.vocab_size = self.content_vocab_size + 2

    def _normalize_dict(self) -> torch.Tensor:
        return F.normalize(self.dictionary, p=2, dim=0, eps=self.epsilon)

    def _patch_grid_shape(self, height: int, width: int) -> Tuple[int, int]:
        if height < self.patch_size or width < self.patch_size:
            raise ValueError(
                f"Latent map {(height, width)} is smaller than patch_size {self.patch_size}"
            )
        if self.patch_size == 1:
            return height, width
        if (height - self.patch_size) % self.patch_stride != 0:
            raise ValueError(
                f"Latent height {height} is incompatible with patch_size={self.patch_size} "
                f"and patch_stride={self.patch_stride}"
            )
        if (width - self.patch_size) % self.patch_stride != 0:
            raise ValueError(
                f"Latent width {width} is incompatible with patch_size={self.patch_size} "
                f"and patch_stride={self.patch_stride}"
            )
        patch_h = ((height - self.patch_size) // self.patch_stride) + 1
        patch_w = ((width - self.patch_size) // self.patch_stride) + 1
        return patch_h, patch_w

    def _latent_shape_from_patch_grid(self, patch_h: int, patch_w: int) -> Tuple[int, int]:
        if self.patch_size == 1:
            return patch_h, patch_w
        if self.patch_stride == self.patch_size:
            return patch_h * self.patch_size, patch_w * self.patch_size
        height = self.patch_size + (patch_h - 1) * self.patch_stride
        width = self.patch_size + (patch_w - 1) * self.patch_stride
        return height, width

    def _extract_patches(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        B, C, H, W = z_e.shape
        patch_h, patch_w = self._patch_grid_shape(H, W)
        if self.patch_size == 1:
            patches = z_e.permute(0, 2, 3, 1).contiguous()
            return patches, patch_h, patch_w
        if self.patch_stride == self.patch_size:
            patches = (
                z_e.reshape(B, C, patch_h, self.patch_size, patch_w, self.patch_size)
                .permute(0, 2, 4, 1, 3, 5)
                .contiguous()
                .view(B, patch_h, patch_w, self.patch_dim)
            )
            return patches, patch_h, patch_w
        if not z_e.is_contiguous():
            z_e = z_e.contiguous()
        stride_b, stride_c, stride_h, stride_w = z_e.stride()
        patches = z_e.as_strided(
            size=(B, C, patch_h, patch_w, self.patch_size, self.patch_size),
            stride=(
                stride_b,
                stride_c,
                stride_h * self.patch_stride,
                stride_w * self.patch_stride,
                stride_h,
                stride_w,
            ),
        )
        patches = (
            patches.permute(0, 2, 3, 1, 4, 5)
            .contiguous()
            .view(B, patch_h, patch_w, self.patch_dim)
        )
        return patches, patch_h, patch_w

    def _overlap_patch_reconstruct_data(
        self,
        patch_h: int,
        patch_w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_key = (patch_h, patch_w, device.type, device.index, str(dtype))
        cached = self._patch_reconstruct_cache.get(cache_key)
        if cached is not None:
            return cached

        latent_h, latent_w = self._latent_shape_from_patch_grid(patch_h, patch_w)
        patch_rows = (
            torch.arange(patch_h, device=device)[:, None, None, None] * self.patch_stride
            + torch.arange(self.patch_size, device=device)[None, None, :, None]
        )
        patch_cols = (
            torch.arange(patch_w, device=device)[None, :, None, None] * self.patch_stride
            + torch.arange(self.patch_size, device=device)[None, None, None, :]
        )
        flat_idx = (patch_rows * latent_w + patch_cols).reshape(-1)

        norm = torch.zeros(1, 1, latent_h * latent_w, device=device, dtype=dtype)
        norm.scatter_add_(
            2,
            flat_idx.view(1, 1, -1),
            torch.ones(1, 1, flat_idx.numel(), device=device, dtype=dtype),
        )
        cached = (flat_idx, norm.clamp_min(self.epsilon))
        self._patch_reconstruct_cache[cache_key] = cached
        return cached

    def _fold_patches(self, patch_values: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
        B = patch_values.size(0)
        if self.patch_size == 1:
            return patch_values.view(B, patch_h, patch_w, self.embedding_dim).permute(0, 3, 1, 2).contiguous()

        if self.patch_stride == self.patch_size:
            return (
                patch_values.view(
                    B,
                    patch_h,
                    patch_w,
                    self.embedding_dim,
                    self.patch_size,
                    self.patch_size,
                )
                .permute(0, 3, 1, 4, 2, 5)
                .contiguous()
                .view(
                    B,
                    self.embedding_dim,
                    patch_h * self.patch_size,
                    patch_w * self.patch_size,
                )
            )

        latent_h, latent_w = self._latent_shape_from_patch_grid(patch_h, patch_w)
        flat_idx, norm = self._overlap_patch_reconstruct_data(
            patch_h,
            patch_w,
            patch_values.device,
            patch_values.dtype,
        )
        patches = (
            patch_values.view(
                B,
                patch_h,
                patch_w,
                self.embedding_dim,
                self.patch_size,
                self.patch_size,
            )
            .permute(0, 3, 1, 2, 4, 5)
            .contiguous()
            .view(B, self.embedding_dim, -1)
        )
        recon = torch.zeros(
            B,
            self.embedding_dim,
            latent_h * latent_w,
            device=patch_values.device,
            dtype=patch_values.dtype,
        )
        recon.scatter_add_(2, flat_idx.view(1, 1, -1).expand(B, self.embedding_dim, -1), patches)
        recon = recon / norm
        return recon.view(B, self.embedding_dim, latent_h, latent_w)

    def _quantize_coeff(self, coeff: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize coefficients to bins; return (bin_idx, bin_center_value)."""
        c = coeff.clamp(-self.coef_max, self.coef_max)
        scaled = (c + self.coef_max) * self.coef_bin_scale
        bin_idx = torch.round(scaled).to(torch.long).clamp(0, self.n_bins - 1)
        coeff_q = self._dequantize_coeff(bin_idx)
        return bin_idx, coeff_q

    def _dequantize_coeff(self, bin_idx: torch.Tensor) -> torch.Tensor:
        """Decode bin indices back to quantized coefficients."""
        return bin_idx.to(self.coef_bin_step.dtype) * self.coef_bin_step - self.coef_max

    def _pack_quantized_tokens(self, support: torch.Tensor, bin_idx: torch.Tensor) -> torch.Tensor:
        """Interleave atom tokens and coefficient-bin tokens along the token depth axis."""
        if support.shape != bin_idx.shape:
            raise ValueError(f"support and bin_idx shape mismatch: {support.shape} vs {bin_idx.shape}")
        if support.size(-1) != self.sparsity_level:
            raise ValueError(f"Expected sparse depth {self.sparsity_level}, got {support.size(-1)}")

        tokens = torch.empty(
            *support.shape[:-1],
            self.token_depth,
            device=support.device,
            dtype=torch.long,
        )
        tokens[..., 0::2] = support.to(torch.long)
        tokens[..., 1::2] = bin_idx.to(torch.long) + self.coeff_token_offset
        return tokens

    def _unpack_quantized_tokens(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode alternating [atom, coeff_bin] tokens back to atom ids and coefficients."""
        if tokens.size(-1) != self.token_depth:
            raise ValueError(f"Expected token depth {self.token_depth}, got {tokens.size(-1)}")

        atom_tokens = tokens[..., 0::2].to(torch.long)
        coeff_tokens = tokens[..., 1::2].to(torch.long)

        atom_invalid = (atom_tokens < 0) | (atom_tokens >= self.num_embeddings)
        coeff_bin = coeff_tokens - self.coeff_token_offset
        coeff_invalid = (coeff_bin < 0) | (coeff_bin >= self.n_bins)
        invalid = atom_invalid | coeff_invalid

        atom_ids = atom_tokens.clamp(0, self.num_embeddings - 1)
        coeff_bin = coeff_bin.clamp(0, self.n_bins - 1)
        coeffs = self._dequantize_coeff(coeff_bin)

        atom_ids = atom_ids.masked_fill(invalid, 0)
        coeffs = coeffs.masked_fill(invalid, 0.0)
        return atom_ids, coeffs

    def _bound_coeffs_by_signal_norm(self, coeffs: torch.Tensor, signals: torch.Tensor) -> torch.Tensor:
        """Clip each signal's sparse coefficients to that signal's L2 norm."""
        if coeffs.ndim != 2:
            raise ValueError(f"Expected coeffs with shape [B, K], got {tuple(coeffs.shape)}")
        if signals.ndim != 2:
            raise ValueError(f"Expected signals with shape [M, B], got {tuple(signals.shape)}")
        if coeffs.size(0) != signals.size(1):
            raise ValueError(
                f"coeff batch ({coeffs.size(0)}) does not match signal batch ({signals.size(1)})"
            )
        signal_norm = torch.linalg.vector_norm(signals, dim=0).to(coeffs.dtype)
        signal_bound = signal_norm.unsqueeze(1)
        return torch.maximum(torch.minimum(coeffs, signal_bound), -signal_bound)

    def _encode_sparse_codes(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run OMP and return support atom ids and continuous coefficients."""
        B, _, H, W = z_e.shape
        patches, patch_h, patch_w = self._extract_patches(z_e)
        n_signals = B * patch_h * patch_w
        signals = patches.view(-1, self.patch_dim).t()
        dictionary = self._normalize_dict()
        with torch.no_grad():
            support, coeffs = self.batch_omp(signals, dictionary)
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
        coeffs = self._bound_coeffs_by_signal_norm(coeffs, signals)
        return (
            support.view(B, patch_h, patch_w, self.sparsity_level),
            coeffs.view(B, patch_h, patch_w, self.sparsity_level),
        )

    def _reconstruct_sparse(
        self, support: torch.Tensor, coeffs: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct latent map from atom ids + coefficients."""
        if support.shape != coeffs.shape:
            raise ValueError(
                f"support and coeffs shape mismatch: {support.shape} vs {coeffs.shape}"
            )

        B, patch_h, patch_w, D = support.shape
        if D != self.sparsity_level:
            raise ValueError(f"Expected D={self.sparsity_level}, got {D}")

        dictionary = self._normalize_dict().t()  # [num_embeddings, patch_dim]
        support = support.to(torch.long)
        coeffs = coeffs.to(dictionary.dtype)
        support_flat = support.reshape(-1, D)
        coeffs_flat = coeffs.reshape(-1, D)
        atoms = dictionary[support_flat]  # [B*patch_h*patch_w, D, patch_dim]
        recon_patches = (atoms * coeffs_flat.unsqueeze(-1)).sum(dim=1)  # [N, patch_dim]
        recon_patches = recon_patches.view(B, patch_h, patch_w, self.patch_dim)
        return self._fold_patches(recon_patches, patch_h, patch_w)

    def batch_omp(self, X: torch.Tensor, D: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched OMP.
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
            tokens: [B, H, W, token_depth]
        """
        if z_e.dim() != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(z_e.shape)}")
        B, C, H, W = z_e.shape
        if C != self.embedding_dim:
            raise ValueError(f"Expected channel dim {self.embedding_dim}, got {C}")

        support, coeffs = self._encode_sparse_codes(z_e)

        if self.quantize_sparse_coeffs:
            # Quantize coefficients in-place over the native [B, H, W, D] layout.
            bin_idx, coeffs_for_recon = self._quantize_coeff(coeffs)
            tokens = self._pack_quantized_tokens(support, bin_idx)
        else:
            tokens = support.long()
            coeffs_for_recon = coeffs

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
            tokens: [B, H, W, token_depth]
            coeffs: [B, H, W, D] (only used in non-quantized mode)
        Returns:
            z_q: [B, C, H, W]
        """
        if tokens.dim() != 4:
            raise ValueError(f"Expected [B,H,W,D], got {tuple(tokens.shape)}")
        B, H, W, D = tokens.shape
        if D != self.token_depth:
            raise ValueError(f"Expected token depth {self.token_depth}, got {D}")

        if self.quantize_sparse_coeffs:
            atom_ids, coeff_q = self._unpack_quantized_tokens(tokens)
            return self._reconstruct_sparse(atom_ids, coeff_q)

        if coeffs is None:
            raise ValueError("coeffs must be provided when quantize_sparse_coeffs=False")

        return self._reconstruct_sparse(tokens.to(torch.long), coeffs.to(self._normalize_dict().dtype))

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
        embedding_dim: int = 16,
        num_embeddings: int = 256,
        sparsity_level: int = 4,
        commitment_cost: float = 0.25,
        n_bins: int = 16,
        coef_max: float = 3.0,
        latent_patch_size: int = 8,
        latent_patch_stride: int = 4,
        quantize_sparse_coeffs: bool = False
    ):
        super().__init__()

        self.encoder = Encoder(
            in_channels,
            num_hiddens,
            num_residual_layers,
            num_residual_hiddens,
            num_downsamples=num_downsamples,
        )
        self.pre_bottleneck = nn.Conv2d(num_hiddens, embedding_dim, kernel_size=1)

        if quantize_sparse_coeffs:
            print("[DEBUG] Using quantized sparse coefficients.")
        else:
            print("[DEBUG] Not using quantized sparse coefficients.")

        self.bottleneck = DictionaryLearning(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            sparsity_level=sparsity_level,
            n_bins=n_bins,
            coef_max=coef_max,
            quantize_sparse_coeffs=quantize_sparse_coeffs,
            patch_size=latent_patch_size,
            patch_stride=latent_patch_stride,
            commitment_cost=commitment_cost,
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        z = torch.tanh(self.pre_bottleneck(z))
        z_q, b_loss, tokens = self.bottleneck(z)
        z_q = safe_atanh(z_q)
        z_q = self.post_bottleneck(z_q)
        recon = self.decoder(z_q)
        return recon, b_loss, tokens

    @torch.no_grad()
    def encode(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, int]:
        z = self.encoder(x)
        z = torch.tanh(self.pre_bottleneck(z))
        atoms, coeffs = self.bottleneck._encode_sparse_codes(z)
        if self.bottleneck.quantize_sparse_coeffs:
            coeff_bin, _ = self.bottleneck._quantize_coeff(coeffs)
            codes = self.bottleneck._pack_quantized_tokens(atoms, coeff_bin)
            coeffs = None
        else:
            codes = atoms
        return codes, coeffs, codes.shape[1], codes.shape[2]

    @torch.no_grad()
    def decode(self, codes: torch.Tensor, coeffs: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.bottleneck.quantize_sparse_coeffs:
            z_q = self.bottleneck.tokens_to_latent(codes)
        else:
            if coeffs is None:
                raise ValueError("coeffs must be provided when quantize_sparse_coeffs=False")
            z_q = self.bottleneck._reconstruct_sparse(codes, coeffs)
        z_q = safe_atanh(z_q)
        z_q = self.post_bottleneck(z_q)
        return self.decoder(z_q)


# -----------------------------
# Stage-2: Transformer prior (GPT-style causal transformer over stacks)
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
class TransformerConfig:
    vocab_size: int
    H: int
    W: int
    D: int
    atom_vocab_size: Optional[int] = None
    coeff_vocab_size: Optional[int] = None
    predict_coefficients: bool = False
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    coeff_max: float = 3.0


class TransformerPrior(nn.Module):
    """
    Autoregressive prior over a flattened H x W x D token grid.
    In quantized LASER mode, D is the token depth after atom/coeff interleaving.
    """
    def __init__(self, cfg: TransformerConfig, bos_token_id: int, pad_token_id: int):
        super().__init__()
        self.cfg = cfg
        self.bos_token_id = int(bos_token_id)
        self.pad_token_id = int(pad_token_id)
        self.predict_coefficients = bool(cfg.predict_coefficients)
        self.coeff_max = float(cfg.coeff_max)
        self.atom_vocab_size = None if cfg.atom_vocab_size is None else int(cfg.atom_vocab_size)
        self.coeff_vocab_size = None if cfg.coeff_vocab_size is None else int(cfg.coeff_vocab_size)
        if (self.atom_vocab_size is None) != (self.coeff_vocab_size is None):
            raise ValueError("atom_vocab_size and coeff_vocab_size must both be set or both be None")
        if self.atom_vocab_size is not None and self.coeff_vocab_size is not None:
            if self.atom_vocab_size <= 0 or self.coeff_vocab_size <= 0:
                raise ValueError("atom_vocab_size and coeff_vocab_size must be positive")
            self.content_vocab_size = self.atom_vocab_size + self.coeff_vocab_size
            if self.content_vocab_size > self.pad_token_id:
                raise ValueError(
                    f"content vocab ({self.content_vocab_size}) exceeds pad token id ({self.pad_token_id})"
                )
        else:
            self.content_vocab_size = None

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
        self.coeff_head = None
        if self.predict_coefficients:
            self.coeff_head = nn.Sequential(
                nn.Linear(2 * cfg.d_model, cfg.d_model),
                nn.GELU(),
                nn.Linear(cfg.d_model, 1),
            )

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

    def _forward_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Return contextual states for the autoregressive prefix `x`."""
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

        return self.ln_f(h)

    def _predict_coefficients(self, hidden: torch.Tensor, coeff_tokens: torch.Tensor) -> torch.Tensor:
        """Predict coefficients conditioned on both the hidden state and aligned atom tokens."""
        if not self.predict_coefficients or self.coeff_head is None:
            raise RuntimeError("Coefficient prediction is disabled for this prior.")
        if hidden.shape[:2] != coeff_tokens.shape:
            raise ValueError(
                f"coeff_tokens shape {tuple(coeff_tokens.shape)} does not match hidden prefix "
                f"shape {tuple(hidden.shape[:2])}"
            )
        # Keep the coefficient head from backpropagating into the token embedding table
        # through the conditioning path; token embeddings are already trained by the
        # autoregressive token loss.
        tok = self.token_emb(coeff_tokens.to(torch.long)).detach()
        coeff_in = torch.cat([hidden, tok], dim=-1)
        return soft_clamp(self.coeff_head(coeff_in).squeeze(-1), self.coeff_max)

    def forward(self, x: torch.Tensor, coeff_tokens: Optional[torch.Tensor] = None):
        """
        Args:
            x: [B, L] tokens, L <= max_len
            coeff_tokens: [B, L] aligned target tokens whose coefficients should be predicted.
        Returns:
            logits: [B, L, vocab]
            coeff_pred: [B, L] when predict_coefficients=True
        """
        h = self._forward_hidden(x)
        logits = self.head(h)
        if not self.predict_coefficients:
            return logits
        if coeff_tokens is None:
            raise ValueError("coeff_tokens must be provided when predict_coefficients=True")
        coeff_pred = self._predict_coefficients(h, coeff_tokens)
        return logits, coeff_pred

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
    ):
        """Sample a batch of flattened token sequences."""
        device = next(self.parameters()).device
        T = self.cfg.H * self.cfg.W * self.cfg.D

        seq = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=device)
        coeff_steps = []
        steps = tqdm(
            range(T),
            desc=(progress_desc or "[Stage2] sampling tokens"),
            leave=False,
            dynamic_ncols=True,
            disable=(not show_progress),
        )
        for _ in steps:
            h = self._forward_hidden(seq)
            logits = self.head(h)[:, -1, :] / max(temperature, 1e-8)
            logits[:, self.pad_token_id] = float("-inf")
            logits[:, self.bos_token_id] = float("-inf")
            if self.content_vocab_size is not None:
                logits[:, self.content_vocab_size:] = float("-inf")
                if (_ % 2) == 0:
                    logits[:, self.atom_vocab_size:self.content_vocab_size] = float("-inf")
                else:
                    logits[:, :self.atom_vocab_size] = float("-inf")
            if top_k is not None and top_k > 0:
                k = min(int(top_k), int(logits.size(-1)))
                v, ix = torch.topk(logits, k, dim=-1)
                mask = torch.full_like(logits, float("-inf"))
                mask.scatter_(1, ix, v)
                logits = mask
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            if self.predict_coefficients:
                coeff_step = self._predict_coefficients(h[:, -1:, :], nxt).squeeze(1)
                coeff_steps.append(coeff_step)
            seq = torch.cat([seq, nxt], dim=1)
        if not self.predict_coefficients:
            return seq[:, 1:]
        coeffs = torch.stack(coeff_steps, dim=1)
        return seq[:, 1:], coeffs


PriorConfig = TransformerConfig
Prior = TransformerPrior


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
    dist_ctx: DistributedContext,
    epochs: int,
    lr: float,
    bottleneck_weight: float,
    out_dir: str,
    rfid_num_samples: int = 0,
    train_sampler: Optional[DistributedSampler] = None,
    wandb_run: Optional[object] = None,
):
    """Train stage 1 with optional DDP and rank-0-only artifacts."""
    ae_module = _unwrap_module(ae)
    device = dist_ctx.device
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    best_val = float("inf")
    global_step = 0
    rfid_warned = False

    for epoch in range(1, epochs + 1):
        # Without set_epoch(), every rank would shuffle in the same order every epoch.
        dist_ctx.set_epoch(train_sampler, epoch)
        ae.train()
        pbar = tqdm(train_loader, desc=f"[Stage1] epoch {epoch}/{epochs}", disable=(not dist_ctx.is_main_process))
        running = 0.0
        for x, _ in pbar:
            x = x.to(device)
            recon, b_loss, _ = ae(x)
            recon_loss = F.mse_loss(recon, x)
            loss = recon_loss + bottleneck_weight * b_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
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

            # These are reduced only for reporting. The optimization step above still
            # uses each rank's local loss, which is what DDP expects.
            loss_log = dist_ctx.mean(loss)
            recon_log = dist_ctx.mean(recon_loss)
            b_log = dist_ctx.mean(b_loss)
            running += float(loss_log.item())
            global_step += 1
            if dist_ctx.is_main_process:
                pbar.set_postfix(
                    total_loss=float(loss_log.item()),
                    reconstruction_loss=float(recon_log.item()),
                    bottleneck_loss=float(b_log.item()),
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
        # Validation is also sharded across ranks. We reduce the summed numerators
        # and counts, then divide once to get the true global averages.
        dist_ctx.reduce_sums_(val_loss_sum, val_psnr_sum, val_ssim_sum, val_count)
        val_loss = float((val_loss_sum / val_count.clamp_min(1)).item())
        val_psnr = float((val_psnr_sum / val_count.clamp_min(1)).item())
        val_ssim = float((val_ssim_sum / val_count.clamp_min(1)).item())

        if dist_ctx.is_main_process:
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

        def _write_stage1_artifacts():
            nonlocal best_val, rfid_warned
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

            # This preview comes from rank 0's validation shard. That is fine for
            # visualization because it is only used for quick qualitative inspection.
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

        # Only rank 0 writes images/checkpoints, but all ranks stay synchronized
        # around that work to keep the next epoch aligned.
        dist_ctx.run_main(_write_stage1_artifacts, sync=True)


@torch.inference_mode()
def precompute_tokens(
    ae: LASER,
    loader: DataLoader,
    device: torch.device,
    max_items: Optional[int] = None,
    show_progress: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], int, int, int, float, float]:
    """
    Encode dataset to tokens for stage-2 training.
    Returns:
      tokens_flat: [N, H*W*token_depth] int32
      coeffs_flat: [N, H*W*sparsity_level] float32 (None if quantized)
      indices_flat: [N] int64 original sample indices (None if unavailable)
      H, W, token_depth
      raw_coeff_min: global minimum raw sparse coefficient encountered
      raw_coeff_max: global maximum raw sparse coefficient encountered
    """
    ae.eval()
    all_tokens = []
    all_coeffs = []
    all_indices = []
    seen = 0
    H = W = D = None
    raw_coeff_min = None
    raw_coeff_max = None

    for batch in tqdm(loader, desc="[Stage2] precompute tokens", disable=(not show_progress)):
        batch_indices = None
        if isinstance(batch, dict):
            x = batch["image"]
            batch_indices = batch["index"]
        elif isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch

        x = x.to(device, non_blocking=True)
        z = ae.encoder(x)
        z = torch.tanh(ae.pre_bottleneck(z))
        atoms, raw_coeffs = ae.bottleneck._encode_sparse_codes(z)
        batch_min = float(raw_coeffs.min().item())
        batch_max = float(raw_coeffs.max().item())
        raw_coeff_min = batch_min if raw_coeff_min is None else min(raw_coeff_min, batch_min)
        raw_coeff_max = batch_max if raw_coeff_max is None else max(raw_coeff_max, batch_max)

        if ae.bottleneck.quantize_sparse_coeffs:
            coeff_bin, _ = ae.bottleneck._quantize_coeff(raw_coeffs)
            tokens = ae.bottleneck._pack_quantized_tokens(atoms, coeff_bin)
            coeffs = None
        else:
            tokens = atoms
            coeffs = raw_coeffs

        h, w = tokens.shape[1], tokens.shape[2]
        if H is None:
            H, W = h, w
            D = tokens.shape[-1]
        flat = tokens.view(tokens.size(0), -1).to(torch.int32).cpu()
        all_tokens.append(flat)
        if coeffs is not None:
            all_coeffs.append(coeffs.view(coeffs.size(0), -1).to(torch.float32).cpu())
        if batch_indices is not None:
            all_indices.append(batch_indices.to(torch.int64).cpu())
        seen += flat.size(0)
        if max_items is not None and seen >= max_items:
            break

    tokens_flat = torch.cat(all_tokens, dim=0)
    if len(all_coeffs) > 0:
        coeffs_flat = torch.cat(all_coeffs, dim=0)
    else:
        coeffs_flat = None
    if len(all_indices) > 0:
        indices_flat = torch.cat(all_indices, dim=0)
    else:
        indices_flat = None
    if max_items is not None:
        tokens_flat = tokens_flat[:max_items]
        if coeffs_flat is not None:
            coeffs_flat = coeffs_flat[:max_items]
        if indices_flat is not None:
            indices_flat = indices_flat[:max_items]
    if raw_coeff_min is None or raw_coeff_max is None:
        raise RuntimeError("Token precompute did not observe any sparse coefficients.")
    return tokens_flat, coeffs_flat, indices_flat, H, W, D, raw_coeff_min, raw_coeff_max


def train_stage2_transformer(
    transformer: TransformerPrior,
    token_loader: DataLoader,
    dist_ctx: DistributedContext,
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
    coeff_loss_weight: float = 1.0,
    wandb_run: Optional[object] = None,
):
    """Train stage 2 with optional DDP and synchronized rank-0 sampling."""
    transformer_module = _unwrap_module(transformer)
    ae_decode = _unwrap_module(ae_for_decode)
    device = dist_ctx.device
    opt = torch.optim.Adam(transformer.parameters(), lr=lr)
    vocab = transformer_module.cfg.vocab_size
    bos = transformer_module.bos_token_id
    global_step = 0
    sample_top_k = None if sample_top_k is None or int(sample_top_k) <= 0 else int(sample_top_k)

    for epoch in range(1, epochs + 1):
        # Same idea as stage 1: reshuffle shard assignments each epoch.
        dist_ctx.set_epoch(token_sampler, epoch)
        transformer.train()
        pbar = tqdm(token_loader, desc=f"[Stage2] epoch {epoch}/{epochs}", disable=(not dist_ctx.is_main_process))
        running = 0.0
        steps = 0

        for batch in pbar:
            coeff_flat = None
            if isinstance(batch, (tuple, list)):
                tok_flat = batch[0]
                if len(batch) > 1:
                    coeff_flat = batch[1]
            else:
                tok_flat = batch
            tok_flat = tok_flat.to(device).long()
            if coeff_flat is not None:
                coeff_flat = coeff_flat.to(device).float()
            B = tok_flat.size(0)

            # The prior is trained autoregressively, so we prepend a BOS token and
            # predict the next token at each position.
            seq = torch.cat([torch.full((B, 1), bos, device=device, dtype=torch.long), tok_flat], dim=1)
            x_in = seq[:, :-1]
            y = seq[:, 1:]

            opt.zero_grad(set_to_none=True)
            if coeff_flat is not None:
                forward_out = transformer(x_in, coeff_tokens=y)
            else:
                forward_out = transformer(x_in)
            if isinstance(forward_out, tuple):
                logits, coeff_pred = forward_out
            else:
                logits = forward_out
                coeff_pred = None
            token_loss = F.cross_entropy(
                logits.reshape(-1, vocab),
                y.reshape(-1),
                ignore_index=pad_token_id,
            )
            loss = token_loss
            coeff_loss = None
            if coeff_flat is not None:
                if coeff_pred is None:
                    raise RuntimeError("Stage-2 token loader provided coefficients, but the prior has no regression head.")
                coeff_loss = F.mse_loss(coeff_pred, coeff_flat)
                loss = loss + coeff_loss_weight * coeff_loss
            elif coeff_pred is not None:
                raise RuntimeError("Stage-2 prior predicts coefficients, but the token loader did not provide coefficient targets.")
            loss.backward()
            opt.step()
            global_step += 1

            loss_log = dist_ctx.mean(loss)
            token_loss_log = dist_ctx.mean(token_loss)
            coeff_loss_log = dist_ctx.mean(coeff_loss) if coeff_loss is not None else None
            running += float(loss_log.item())
            steps += 1
            if dist_ctx.is_main_process:
                metrics = {
                    "stage2/train_loss": float(loss_log.item()),
                    "stage2/token_loss": float(token_loss_log.item()),
                    "stage2/epoch": epoch,
                }
                postfix = {
                    "train_loss": float(loss_log.item()),
                    "token_loss": float(token_loss_log.item()),
                }
                if coeff_loss is not None:
                    metrics["stage2/coeff_loss"] = float(coeff_loss_log.item())
                    postfix["coeff_loss"] = float(coeff_loss_log.item())
                pbar.set_postfix(**postfix)
                _log_wandb(
                    wandb_run,
                    metrics,
                    step_metric="stage2/step",
                    step_value=global_step,
                )

            if sample_every_steps > 0 and (global_step % sample_every_steps == 0):
                def _sample_stage2():
                    transformer.eval()
                    ae_decode.eval()
                    print(f"[Stage2] sampling at step {global_step} (batch_size={sample_batch_size})...")
                    with torch.no_grad():
                        gen_out = transformer_module.generate(
                            batch_size=sample_batch_size,
                            temperature=sample_temperature,
                            top_k=sample_top_k,
                            show_progress=True,
                            progress_desc=f"[Stage2] sample step {global_step}",
                        )
                        if transformer_module.predict_coefficients:
                            tokens_gen_flat, coeffs_gen_flat = gen_out
                            tokens_gen = tokens_gen_flat.view(-1, H, W, D)
                            coeffs_gen = coeffs_gen_flat.view(-1, H, W, D)
                            imgs = ae_decode.decode(tokens_gen.to(device), coeffs_gen.to(device))
                        else:
                            tokens_gen = gen_out.view(-1, H, W, D)
                            imgs = ae_decode.decode(tokens_gen.to(device))
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

                # Sampling is intentionally rank-0-only to avoid generating the same
                # samples on every GPU and writing duplicate files.
                dist_ctx.run_main(_sample_stage2, sync=True)
                transformer.train()

        epoch_loss = running / max(1, steps)
        if dist_ctx.is_main_process:
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

        def _save_stage2_checkpoint():
            os.makedirs(out_dir, exist_ok=True)
            torch.save(transformer_module.state_dict(), os.path.join(out_dir, "transformer_last.pt"))

        # Checkpoint saving is also rank-0-only, so we synchronize around it.
        dist_ctx.run_main(_save_stage2_checkpoint, sync=True)



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
    parser.add_argument(
        "--debug_distributed",
        action="store_true",
        help="Print rank-aware distributed setup details for multi-GPU debugging.",
    )

    parser.add_argument("--stage1_epochs", type=int, default=0)
    parser.add_argument("--stage1_lr", type=float, default=2e-3)
    parser.add_argument("--stage2_epochs", type=int, default=5)
    parser.add_argument("--stage2_lr", type=float, default=2e-3)
    parser.add_argument("--stage1_batch_size", type=int, default=128)
    parser.add_argument("--stage2_batch_size", type=int, default=32)
    parser.add_argument("--bottleneck_weight", type=float, default=1.0)

    parser.add_argument("--num_hiddens", type=int, default=128)
    parser.add_argument("--ae_num_downsamples", type=int, default=2)
    parser.add_argument("--num_res_layers", type=int, default=2)
    parser.add_argument("--num_res_hiddens", type=int, default=32)
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--num_atoms", type=int, default=2048)
    parser.add_argument("--sparsity_level", type=int, default=8)
    parser.add_argument("--latent_patch_size", type=int, default=8)
    parser.add_argument("--latent_patch_stride", type=int, default=4)
    parser.add_argument("--n_bins", type=int, default=256)
    parser.add_argument("--coef_max", type=float, default=1.0)
    parser.add_argument(
        "--quantize_sparse_coeffs",
        dest="quantize_sparse_coeffs",
        action="store_true",
        default=True,
        help="Quantize sparse coefficients into discrete tokens for the stage-2 prior.",
    )
    parser.add_argument(
        "--no_quantize_sparse_coeffs",
        dest="quantize_sparse_coeffs",
        action="store_false",
        help="Keep coefficients continuous and train a regression head in stage 2.",
    )
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
    parser.add_argument("--stage2_sample_every_steps", type=int, default=200)
    parser.add_argument("--stage2_sample_batch_size", type=int, default=64)
    parser.add_argument("--stage2_sample_temperature", type=float, default=1.0)
    parser.add_argument("--stage2_sample_top_k", type=int, default=0)
    parser.add_argument("--stage2_sample_image_size", type=int, default=128)
    parser.add_argument("--stage2_coeff_loss_weight", type=float, default=1.0)

    args = parser.parse_args()
    wandb_run = None

    if args.ae_num_downsamples <= 0:
        raise ValueError(f"ae_num_downsamples must be positive, got {args.ae_num_downsamples}")
    if args.stage1_batch_size <= 0:
        raise ValueError(f"stage1_batch_size must be positive, got {args.stage1_batch_size}")
    if args.stage2_batch_size <= 0:
        raise ValueError(f"stage2_batch_size must be positive, got {args.stage2_batch_size}")
    if args.latent_patch_size <= 0:
        raise ValueError(f"latent_patch_size must be positive, got {args.latent_patch_size}")
    if args.latent_patch_stride <= 0:
        raise ValueError(f"latent_patch_stride must be positive, got {args.latent_patch_stride}")
    if args.latent_patch_stride > args.latent_patch_size:
        raise ValueError(
            f"latent_patch_stride ({args.latent_patch_stride}) must be <= "
            f"latent_patch_size ({args.latent_patch_size})"
        )
    if args.stage2_sample_temperature <= 0.0:
        raise ValueError("stage2_sample_temperature must be > 0.")
    if args.stage2_coeff_loss_weight < 0.0:
        raise ValueError("stage2_coeff_loss_weight must be >= 0.")
    if args.token_subset < 0:
        args.token_subset = 0
    if args.image_size is None:
        args.image_size = 128
    args.image_size = int(args.image_size)
    if args.data_dir is None:
        args.data_dir = '../../data/celeba'
    if args.out_dir is None:
        args.out_dir = './runs/laser_celeba128_quantized' if args.quantize_sparse_coeffs else './runs/laser_celeba128_no_quantized'

    dist_ctx = DistributedContext.initialize()

    try:
        # Seed all ranks the same way for reproducibility. The distributed sampler
        # still gives each rank different data shards, so identical seeds are okay.
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("medium")

        device = dist_ctx.device
        pin_memory = dist_ctx.pin_memory

        def _prepare_output_dirs():
            os.makedirs(args.out_dir, exist_ok=True)
            os.makedirs(stage1_dir, exist_ok=True)
            os.makedirs(stage2_dir, exist_ok=True)

        stage1_dir = os.path.join(args.out_dir, "stage1")
        stage2_dir = os.path.join(args.out_dir, "stage2")
        # Directories are shared state. Let rank 0 create them first, then release
        # the other ranks once the filesystem is ready.
        dist_ctx.run_main(_prepare_output_dirs, sync=True)
        dist_ctx.debug(
            f"initialized distributed context: {dist_ctx.debug_summary()}",
            enabled=args.debug_distributed,
            all_ranks=True,
        )

        def _init_main_process():
            print(
                f"[Setup] device={device} world_size={dist_ctx.world_size} dataset={args.dataset} "
                f"data_dir={args.data_dir} image_size={args.image_size}"
            )
            return _init_wandb(args)

        wandb_run = dist_ctx.run_main(_init_main_process)
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
                latent_patch_size=args.latent_patch_size,
                latent_patch_stride=args.latent_patch_stride,
                commitment_cost=args.commitment_cost,
                n_bins=args.n_bins,
                coef_max=args.coef_max,
                quantize_sparse_coeffs=args.quantize_sparse_coeffs,
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

        # In multi-GPU mode, each sampler hands a unique shard of the dataset to
        # the current rank. In single-process mode these are just `None`.
        train_sampler = dist_ctx.make_sampler(train_set, shuffle=True)
        val_sampler = dist_ctx.make_sampler(val_set, shuffle=False)
        train_loader = DataLoader(
            train_set,
            batch_size=args.stage1_batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=min(64, args.stage1_batch_size),
            shuffle=False,
            sampler=val_sampler,
            num_workers=max(0, args.num_workers // 2),
            pin_memory=pin_memory,
        )
        rfid_loader = None
        if dist_ctx.is_main_process and args.rfid_num_samples > 0:
            rfid_loader = DataLoader(
                val_set,
                batch_size=min(32, min(64, args.stage1_batch_size)),
                shuffle=False,
                num_workers=max(0, args.num_workers // 2),
                pin_memory=pin_memory,
            )

        dist_ctx.debug(
            f"loaders ready: train_batches={len(train_loader)} val_batches={len(val_loader)} "
            f"train_sampler={'distributed' if train_sampler is not None else 'none'} "
            f"val_sampler={'distributed' if val_sampler is not None else 'none'}",
            enabled=args.debug_distributed,
            all_ranks=True,
        )

        laser = _build_laser().to(device)
        # `laser` stays as the underlying module. `laser_stage1` is the DDP wrapper
        # used for training. Keeping both makes checkpointing and later reuse simpler.
        laser_stage1 = dist_ctx.wrap_model(laser)
        if args.stage1_epochs > 0:
            train_stage1_ae(
                ae=laser_stage1,
                train_loader=train_loader,
                val_loader=val_loader,
                rfid_loader=rfid_loader,
                dist_ctx=dist_ctx,
                epochs=args.stage1_epochs,
                lr=args.stage1_lr,
                bottleneck_weight=args.bottleneck_weight,
                out_dir=stage1_dir,
                rfid_num_samples=args.rfid_num_samples,
                train_sampler=train_sampler,
                wandb_run=wandb_run,
            )
        dist_ctx.barrier()

        _load_best_laser_weights(laser)
        laser = laser.to(device)

        token_cache_path = os.path.join(stage2_dir, "tokens_cache.pt")
        token_subset = None if args.token_subset <= 0 else min(args.token_subset, len(stage2_source_set))
        token_source_set = (
            stage2_source_set
            if token_subset is None
            else Subset(stage2_source_set, range(token_subset))
        )
        indexed_token_source_set = IndexedDataset(token_source_set)
        token_precompute_sampler = dist_ctx.make_sampler(indexed_token_source_set, shuffle=False)
        token_source_loader = DataLoader(
            indexed_token_source_set,
            batch_size=args.stage1_batch_size,
            shuffle=False,
            sampler=token_precompute_sampler,
            num_workers=args.token_num_workers,
            pin_memory=pin_memory,
            persistent_workers=(args.token_num_workers > 0),
        )
        dist_ctx.debug(
            f"token precompute loader ready: batches={len(token_source_loader)} "
            f"sampler={'distributed' if token_precompute_sampler is not None else 'none'}",
            enabled=args.debug_distributed,
            all_ranks=True,
        )
        tokens_flat, coeffs_flat, token_indices, H, W, D, raw_coeff_min, raw_coeff_max = precompute_tokens(
            laser,
            token_source_loader,
            device,
            show_progress=dist_ctx.is_main_process,
        )

        if dist_ctx.enabled:
            token_shard_path = os.path.join(stage2_dir, f"tokens_cache.rank{dist_ctx.rank:04d}.pt")
            shard_payload = {
                "tokens_flat": tokens_flat,
                "indices": token_indices,
                "shape": (H, W, D),
                "raw_coeff_min": raw_coeff_min,
                "raw_coeff_max": raw_coeff_max,
            }
            if coeffs_flat is not None:
                shard_payload["coeffs_flat"] = coeffs_flat
            torch.save(shard_payload, token_shard_path)

            def _merge_token_cache_shards():
                shard_tokens = []
                shard_coeffs = []
                shard_indices = []
                shard_shape = None
                merged_raw_coeff_min = None
                merged_raw_coeff_max = None
                shard_paths = [
                    os.path.join(stage2_dir, f"tokens_cache.rank{rank:04d}.pt")
                    for rank in range(dist_ctx.world_size)
                ]
                for shard_path in shard_paths:
                    try:
                        shard_payload = torch.load(shard_path, map_location="cpu", weights_only=True)
                    except TypeError:
                        shard_payload = torch.load(shard_path, map_location="cpu")
                    shard_tokens.append(shard_payload["tokens_flat"])
                    if shard_payload.get("coeffs_flat") is not None:
                        shard_coeffs.append(shard_payload["coeffs_flat"])
                    shard_indices.append(shard_payload["indices"])
                    if shard_shape is None:
                        shard_shape = shard_payload["shape"]
                    shard_min = shard_payload.get("raw_coeff_min")
                    shard_max = shard_payload.get("raw_coeff_max")
                    if shard_min is not None:
                        merged_raw_coeff_min = shard_min if merged_raw_coeff_min is None else min(merged_raw_coeff_min, float(shard_min))
                    if shard_max is not None:
                        merged_raw_coeff_max = shard_max if merged_raw_coeff_max is None else max(merged_raw_coeff_max, float(shard_max))

                merged_tokens = torch.cat(shard_tokens, dim=0)
                merged_coeffs = torch.cat(shard_coeffs, dim=0) if len(shard_coeffs) > 0 else None
                merged_indices = torch.cat(shard_indices, dim=0)
                order = torch.argsort(merged_indices)
                merged_tokens = merged_tokens[order]
                if merged_coeffs is not None:
                    merged_coeffs = merged_coeffs[order]
                merged_indices = merged_indices[order]

                # DistributedSampler may pad the last few samples so every rank gets
                # the same number of steps. Sorting by original index lets us drop
                # those duplicates while keeping a stable cache order.
                if merged_indices.numel() > 1:
                    keep = torch.ones(merged_indices.size(0), dtype=torch.bool)
                    keep[1:] = merged_indices[1:] != merged_indices[:-1]
                    merged_tokens = merged_tokens[keep]
                    if merged_coeffs is not None:
                        merged_coeffs = merged_coeffs[keep]
                    merged_indices = merged_indices[keep]

                cache_payload = {"tokens_flat": merged_tokens, "shape": shard_shape}
                if merged_coeffs is not None:
                    cache_payload["coeffs_flat"] = merged_coeffs
                if merged_raw_coeff_min is not None and merged_raw_coeff_max is not None:
                    cache_payload["raw_coeff_min"] = float(merged_raw_coeff_min)
                    cache_payload["raw_coeff_max"] = float(merged_raw_coeff_max)
                torch.save(cache_payload, token_cache_path)
                for shard_path in shard_paths:
                    if os.path.exists(shard_path):
                        os.remove(shard_path)
                print(
                    f"[Stage2] token dataset: {merged_tokens.shape} "
                    f"(H={shard_shape[0]}, W={shard_shape[1]}, D={shard_shape[2]}, "
                    f"coeffs={'yes' if merged_coeffs is not None else 'no'}, "
                    f"raw_coeff_min={float(merged_raw_coeff_min):.6f}, "
                    f"raw_coeff_max={float(merged_raw_coeff_max):.6f})"
                )

            # Every rank precomputes its own shard, then rank 0 merges the shards
            # into the final cache that all ranks will read.
            dist_ctx.run_main(_merge_token_cache_shards, sync=True)
        else:
            cache_payload = {
                "tokens_flat": tokens_flat,
                "shape": (H, W, D),
                "raw_coeff_min": float(raw_coeff_min),
                "raw_coeff_max": float(raw_coeff_max),
            }
            if coeffs_flat is not None:
                cache_payload["coeffs_flat"] = coeffs_flat
            torch.save(cache_payload, token_cache_path)
            print(
                f"[Stage2] token dataset: {tokens_flat.shape} "
                f"(H={H}, W={W}, D={D}, coeffs={'yes' if coeffs_flat is not None else 'no'}, "
                f"raw_coeff_min={float(raw_coeff_min):.6f}, raw_coeff_max={float(raw_coeff_max):.6f})"
            )

        try:
            token_cache = torch.load(token_cache_path, map_location="cpu", weights_only=True)
        except TypeError:
            token_cache = torch.load(token_cache_path, map_location="cpu")
        tokens_flat = token_cache["tokens_flat"]
        coeffs_flat = token_cache.get("coeffs_flat")
        H, W, D = token_cache["shape"]

        if args.stage2_epochs <= 0:
            dist_ctx.barrier()
            if dist_ctx.is_main_process:
                print(f"Outputs saved to: {args.out_dir}")
            return

        # Stage 2 also uses a distributed sampler so each rank sees different token sequences.
        token_dataset = (
            TensorDataset(tokens_flat, coeffs_flat)
            if coeffs_flat is not None
            else tokens_flat
        )
        token_sampler = dist_ctx.make_sampler(token_dataset, shuffle=True)
        token_loader = DataLoader(
            token_dataset,
            batch_size=args.stage2_batch_size,
            shuffle=(token_sampler is None),
            sampler=token_sampler,
            num_workers=0,
            pin_memory=pin_memory,
            drop_last=(len(token_dataset) >= args.stage2_batch_size),
        )
        dist_ctx.debug(
            f"token loader ready: token_batches={len(token_loader)} "
            f"token_sampler={'distributed' if token_sampler is not None else 'none'}",
            enabled=args.debug_distributed,
            all_ranks=True,
        )
        transformer = TransformerPrior(
            TransformerConfig(
                vocab_size=laser.bottleneck.vocab_size,
                H=H,
                W=W,
                D=D,
                atom_vocab_size=(laser.bottleneck.num_embeddings if laser.bottleneck.quantize_sparse_coeffs else None),
                coeff_vocab_size=(laser.bottleneck.n_bins if laser.bottleneck.quantize_sparse_coeffs else None),
                predict_coefficients=(coeffs_flat is not None),
                d_model=args.tf_d_model,
                n_heads=args.tf_heads,
                n_layers=args.tf_layers,
                d_ff=args.tf_ff,
                dropout=args.tf_dropout,
                coeff_max=args.coef_max,
            ),
            bos_token_id=laser.bottleneck.bos_token_id,
            pad_token_id=laser.bottleneck.pad_token_id,
        ).to(device)
        transformer_stage2 = dist_ctx.wrap_model(transformer)

        train_stage2_transformer(
            transformer=transformer_stage2,
            token_loader=token_loader,
            dist_ctx=dist_ctx,
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
            coeff_loss_weight=args.stage2_coeff_loss_weight,
            wandb_run=wandb_run,
        )

        if dist_ctx.is_main_process:
            print(f"Outputs saved to: {args.out_dir}")
        dist_ctx.barrier()
    except Exception as exc:
        if dist_ctx.enabled or args.debug_distributed:
            print(f"{dist_ctx.prefix} fatal error: {type(exc).__name__}: {exc}", flush=True)
            if args.debug_distributed:
                traceback.print_exc()
        raise
    finally:
        if dist_ctx.is_main_process and wandb_run is not None:
            wandb_run.finish()
        # Safe to call in both single-process and distributed modes.
        dist_ctx.cleanup()


if __name__ == "__main__":
    main()
