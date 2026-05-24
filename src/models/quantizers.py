"""Vector-quantization bottlenecks (codebook and EMA variants)."""

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """VQ baseline aligned with lucidrains vector-quantize-pytorch (no EMA)."""

    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(
            -1 / num_embeddings, 1 / num_embeddings
        )

    def forward(self, z_e: torch.Tensor):
        """
        Args:
            z_e: input tensor of shape [B, C, H, W] from the encoder

        Returns:
            Tuple of (quantized tensor, loss, perplexity, one-hot encodings)
        """
        if z_e.dim() != 4:
            raise ValueError(
                f"Expected input [B, C, H, W], got {tuple(z_e.shape)}"
            )
        if z_e.shape[1] != self.embedding_dim:
            raise ValueError(
                (
                    "Expected channel dim "
                    f"{self.embedding_dim} but received {z_e.shape[1]}"
                )
            )

        # [B, C, H, W] -> [B, H, W, C] -> [N, C]
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        z_flat = z_e.view(-1, self.embedding_dim)

        # Squared L2 distance to each codebook vector.
        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2 * z_flat @ self.embedding.weight.t()
        )
        indices = torch.argmin(distances, dim=1)

        encodings = torch.zeros(
            indices.shape[0],
            self.num_embeddings,
            device=z_flat.device,
            dtype=z_flat.dtype,
        )
        encodings.scatter_(1, indices.unsqueeze(1), 1)

        quantized = self.embedding(indices).view_as(z_e)

        # Commitment loss only (lucidrains-style); codebook learns via gradients.
        loss = self.commitment_cost * F.mse_loss(quantized, z_e)

        # Straight-through estimator: forward uses quantized, backward sees
        # identity.
        quantized = z_e + (quantized - z_e).detach()

        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        )

        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return quantized, loss, perplexity, encodings


class VectorQuantizerEMA(nn.Module):
    """VQ baseline aligned with lucidrains vector-quantize-pytorch (EMA)."""

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        commitment_cost=0.25,
        ema_decay=0.99,
        epsilon=1e-10,
        codebook_init=False,
        dead_code_threshold=0.0,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.epsilon = epsilon
        self.codebook_init = bool(codebook_init)
        self.dead_code_threshold = float(dead_code_threshold)

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.normal_()
        self.embedding.weight.requires_grad = False

        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer(
            "_ema_w", torch.randn(num_embeddings, embedding_dim)
        )
        self.register_buffer("_codebook_initialized", torch.tensor(False))

    def _sample_batch_vectors(self, z_flat: torch.Tensor, count: int) -> torch.Tensor:
        count = int(count)
        if z_flat.ndim != 2 or int(z_flat.size(0)) <= 0:
            raise ValueError("Cannot sample codebook vectors from an empty latent batch")
        if int(z_flat.size(0)) >= count:
            indices = torch.randperm(int(z_flat.size(0)), device=z_flat.device)[:count]
            return z_flat[indices]

        repeats = int(math.ceil(count / int(z_flat.size(0))))
        tiled = z_flat.repeat(repeats, 1)
        indices = torch.randperm(int(tiled.size(0)), device=z_flat.device)[:count]
        return tiled[indices]

    @torch.no_grad()
    def _initialize_codebook_from_batch(self, z_flat: torch.Tensor):
        samples = self._sample_batch_vectors(z_flat.detach(), self.num_embeddings)
        samples = samples.to(device=self.embedding.weight.device, dtype=self.embedding.weight.dtype)
        self.embedding.weight.copy_(samples)
        self._ema_w.copy_(samples)
        self.ema_cluster_size.fill_(1.0)
        self._codebook_initialized.fill_(True)

    @torch.no_grad()
    def _expire_dead_codes(self, z_flat: torch.Tensor):
        if self.dead_code_threshold <= 0.0:
            return
        dead_codes = self.ema_cluster_size < self.dead_code_threshold
        num_dead = int(dead_codes.sum().item())
        if num_dead <= 0:
            return
        samples = self._sample_batch_vectors(z_flat.detach(), num_dead)
        samples = samples.to(device=self._ema_w.device, dtype=self._ema_w.dtype)
        replacement_count = max(float(self.dead_code_threshold), 1.0)
        self.ema_cluster_size[dead_codes] = replacement_count
        self._ema_w[dead_codes] = samples * replacement_count

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        expected = {
            prefix + "ema_cluster_size": self.ema_cluster_size.detach().clone(),
            prefix + "_ema_w": self._ema_w.detach().clone(),
        }
        for key, value in expected.items():
            loaded = state_dict.get(key)
            if loaded is None or tuple(loaded.shape) != tuple(value.shape):
                state_dict[key] = value
        init_key = prefix + "_codebook_initialized"
        loaded_init = state_dict.get(init_key)
        if loaded_init is None or tuple(loaded_init.shape) != tuple(self._codebook_initialized.shape):
            state_dict[init_key] = torch.ones_like(self._codebook_initialized)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, z_e: torch.Tensor):
        """
        Args:
            z_e: input tensor of shape [B, C, H, W] from the encoder

        Returns:
            Tuple of (quantized tensor, loss, perplexity, one-hot encodings)
        """
        if z_e.dim() != 4:
            raise ValueError(
                f"Expected input [B, C, H, W], got {tuple(z_e.shape)}"
            )
        if z_e.shape[1] != self.embedding_dim:
            raise ValueError(
                (
                    "Expected channel dim "
                    f"{self.embedding_dim} but received {z_e.shape[1]}"
                )
            )

        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        z_flat = z_e.view(-1, self.embedding_dim)
        if self.training and self.codebook_init and not bool(self._codebook_initialized.item()):
            self._initialize_codebook_from_batch(z_flat)

        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2 * z_flat @ self.embedding.weight.t()
        )
        indices = torch.argmin(distances, dim=1)

        encodings = torch.zeros(
            indices.shape[0],
            self.num_embeddings,
            device=z_flat.device,
            dtype=z_flat.dtype,
        )
        encodings.scatter_(1, indices.unsqueeze(1), 1)

        quantized = self.embedding(indices).view_as(z_e)

        if self.training:
            with torch.no_grad():
                counts = encodings.sum(dim=0).to(
                    self.ema_cluster_size.dtype
                )
                self.ema_cluster_size.mul_(self.ema_decay).add_(
                    counts, alpha=1 - self.ema_decay
                )
                dw = (encodings.t() @ z_flat).to(self._ema_w.dtype)
                self._ema_w.mul_(self.ema_decay).add_(
                    dw, alpha=1 - self.ema_decay
                )
                self._expire_dead_codes(z_flat)

                n = self.ema_cluster_size.sum()
                cluster_size = (
                    (self.ema_cluster_size + self.epsilon)
                    / (n + self.num_embeddings * self.epsilon) * n
                )
                self.embedding.weight.copy_(
                    self._ema_w / cluster_size.unsqueeze(1)
                )

        loss = self.commitment_cost * F.mse_loss(quantized.detach(), z_e)

        quantized = z_e + (quantized - z_e).detach()

        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        )

        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return quantized, loss, perplexity, encodings
