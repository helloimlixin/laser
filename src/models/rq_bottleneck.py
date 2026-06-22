"""Residual Quantization bottleneck.

Ported from https://github.com/kakaobrain/rq-vae-transformer/blob/main/rqvae/models/rqvae/quantizations.py
to fit the LASER `(z_dl, bottleneck_loss, sparse_codes)` bottleneck interface.

Differences vs upstream:
- Wraps codes in :class:`SparseCodes` so downstream cache/stage-2 paths keep working.
- The bottleneck loss is ``commitment_cost * commitment_loss`` (matches the LASER
  ``commitment_cost`` semantics).
"""

import math
from typing import Iterable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .bottleneck_utils import SparseCodes


class VQEmbedding(nn.Embedding):
    """VQ embedding with EMA codebook updates and dead-code restart."""

    def __init__(self, n_embed, embed_dim, ema=True, decay=0.99, restart_unused_codes=True, eps=1e-5):
        super().__init__(n_embed + 1, embed_dim, padding_idx=n_embed)

        self.ema = ema
        self.decay = decay
        self.eps = eps
        self.restart_unused_codes = restart_unused_codes
        self.n_embed = n_embed

        if self.ema:
            for p in self.parameters():
                p.requires_grad_(False)
            self.register_buffer("cluster_size_ema", torch.zeros(n_embed))
            self.register_buffer("embed_ema", self.weight[:-1, :].detach().clone())

    @torch.no_grad()
    def compute_distances(self, inputs):
        codebook_t = self.weight[:-1, :].t()
        embed_dim, _ = codebook_t.shape
        inputs_shape = inputs.shape
        assert inputs_shape[-1] == embed_dim
        inputs_flat = inputs.reshape(-1, embed_dim)
        inputs_norm_sq = inputs_flat.pow(2.0).sum(dim=1, keepdim=True)
        codebook_t_norm_sq = codebook_t.pow(2.0).sum(dim=0, keepdim=True)
        distances = torch.addmm(
            inputs_norm_sq + codebook_t_norm_sq,
            inputs_flat,
            codebook_t,
            alpha=-2.0,
        )
        distances = distances.reshape(*inputs_shape[:-1], -1)
        return distances

    @torch.no_grad()
    def find_nearest_embedding(self, inputs):
        distances = self.compute_distances(inputs)
        return distances.argmin(dim=-1)

    @torch.no_grad()
    def _tile_with_noise(self, x, target_n):
        B, embed_dim = x.shape
        n_repeats = (target_n + B - 1) // B
        std = x.new_ones(embed_dim) * 0.01 / math.sqrt(embed_dim)
        x = x.repeat(n_repeats, 1)
        x = x + torch.rand_like(x) * std
        return x

    @torch.no_grad()
    def _update_buffers(self, vectors, idxs):
        n_embed = self.weight.shape[0] - 1
        embed_dim = self.weight.shape[-1]
        vectors = vectors.reshape(-1, embed_dim)
        idxs = idxs.reshape(-1)
        n_vectors = vectors.shape[0]
        n_total_embed = n_embed
        one_hot_idxs = vectors.new_zeros(n_total_embed, n_vectors)
        one_hot_idxs.scatter_(
            dim=0, index=idxs.unsqueeze(0), src=vectors.new_ones(1, n_vectors)
        )
        cluster_size = one_hot_idxs.sum(dim=1)
        vectors_sum_per_cluster = one_hot_idxs @ vectors
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(vectors_sum_per_cluster, op=dist.ReduceOp.SUM)
            dist.all_reduce(cluster_size, op=dist.ReduceOp.SUM)
        self.cluster_size_ema.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.embed_ema.mul_(self.decay).add_(vectors_sum_per_cluster, alpha=1 - self.decay)
        if self.restart_unused_codes:
            if n_vectors < n_embed:
                vectors = self._tile_with_noise(vectors, n_embed)
            n_vectors = vectors.shape[0]
            _vectors_random = vectors[torch.randperm(n_vectors, device=vectors.device)][:n_embed]
            if dist.is_available() and dist.is_initialized():
                dist.broadcast(_vectors_random, 0)
            usage = (self.cluster_size_ema.view(-1, 1) >= 1).float()
            self.embed_ema.mul_(usage).add_(_vectors_random * (1 - usage))
            self.cluster_size_ema.mul_(usage.view(-1))
            self.cluster_size_ema.add_(torch.ones_like(self.cluster_size_ema) * (1 - usage).view(-1))

    @torch.no_grad()
    def _update_embedding(self):
        n_embed = self.weight.shape[0] - 1
        n = self.cluster_size_ema.sum()
        normalized_cluster_size = (
            n * (self.cluster_size_ema + self.eps) / (n + n_embed * self.eps)
        )
        self.weight[:-1, :] = self.embed_ema / normalized_cluster_size.reshape(-1, 1)

    def forward(self, inputs):
        embed_idxs = self.find_nearest_embedding(inputs)
        if self.training and self.ema:
            self._update_buffers(inputs, embed_idxs)
        embeds = self.embed(embed_idxs)
        if self.ema and self.training:
            self._update_embedding()
        return embeds, embed_idxs

    def embed(self, idxs):
        return super().forward(idxs)


class RQBottleneck(nn.Module):
    """Residual Quantization with depth ``code_depth``.

    Acts as a drop-in for ``DictionaryLearning`` in LASER: takes ``z_e`` of shape
    ``[B, C, H, W]`` and returns ``(z_dl, commitment_cost * commitment_loss, sparse_codes)``.
    ``sparse_codes.support`` has shape ``[B, H, W, code_depth]`` (atom index per depth);
    ``sparse_codes.values`` is all-ones (RQ has no continuous coefficients).
    """

    def __init__(
        self,
        num_embeddings=2048,
        embedding_dim=256,
        code_depth=4,
        shared_codebook=True,
        decay=0.99,
        restart_unused_codes=True,
        commitment_cost=0.25,
        ema_eps=1e-5,
    ):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.code_depth = int(code_depth)
        self.shared_codebook = bool(shared_codebook)
        self.commitment_cost = float(commitment_cost)

        n_embed_list = [self.num_embeddings] * self.code_depth
        decay_list = [float(decay)] * self.code_depth

        if self.shared_codebook:
            codebook0 = VQEmbedding(
                n_embed_list[0],
                self.embedding_dim,
                decay=decay_list[0],
                restart_unused_codes=restart_unused_codes,
                eps=ema_eps,
            )
            self.codebooks = nn.ModuleList(
                [codebook0 for _ in range(self.code_depth)]
            )
        else:
            self.codebooks = nn.ModuleList(
                [
                    VQEmbedding(
                        n_embed_list[i],
                        self.embedding_dim,
                        decay=decay_list[i],
                        restart_unused_codes=restart_unused_codes,
                        eps=ema_eps,
                    )
                    for i in range(self.code_depth)
                ]
            )

        self._last_latent_loss = None
        self._last_dl_latent_loss = None
        self._last_e_latent_loss = None
        self._last_bottleneck_loss = torch.zeros(())

    def quantize(self, x):
        """Sequential residual quantization. x: [B, H, W, D]."""
        residual = x.detach().clone()
        quant_list = []
        code_list = []
        aggregated_quants = torch.zeros_like(x)
        for i in range(self.code_depth):
            quant, code = self.codebooks[i](residual)
            residual = residual - quant
            aggregated_quants = aggregated_quants + quant
            quant_list.append(aggregated_quants.clone())
            code_list.append(code.unsqueeze(-1))
        codes = torch.cat(code_list, dim=-1)
        return quant_list, codes

    def compute_commitment_loss(self, x, quant_list):
        loss_list = []
        for quant in quant_list:
            loss_list.append((x - quant.detach()).pow(2.0).mean())
        return torch.mean(torch.stack(loss_list))

    def forward(self, z_e):
        if z_e.dim() != 4:
            raise ValueError(f"Expected input [B, C, H, W], got {tuple(z_e.shape)}")
        B, C, H, W = z_e.shape
        if C != self.embedding_dim:
            raise ValueError(
                f"Expected channel dim {self.embedding_dim} but received {C}"
            )

        x = z_e.permute(0, 2, 3, 1).contiguous()
        quant_list, codes = self.quantize(x)
        commitment_loss = self.compute_commitment_loss(x, quant_list)
        z_dl_quant = quant_list[-1]
        z_dl = x + (z_dl_quant - x).detach()
        z_dl = z_dl.permute(0, 3, 1, 2).contiguous()

        bottleneck_loss = self.commitment_cost * commitment_loss

        self._last_latent_loss = bottleneck_loss.detach()
        self._last_dl_latent_loss = commitment_loss.detach()
        self._last_e_latent_loss = commitment_loss.detach()
        self._last_bottleneck_loss = bottleneck_loss.detach()

        sparse_codes = SparseCodes(
            support=codes,
            values=torch.ones_like(codes, dtype=z_e.dtype),
            num_embeddings=self.num_embeddings,
        )
        return z_dl, bottleneck_loss, sparse_codes

    # --- LASER compat shims (DictionaryLearning interface) -------------------

    @property
    def dictionary(self) -> torch.Tensor:
        """[embedding_dim, num_embeddings] tensor view of codebook 0 (sans padding row)."""
        return self.codebooks[0].weight[:-1, :].t()

    @property
    def dictionary_dtype(self) -> torch.dtype:
        return self.codebooks[0].weight.dtype

    @property
    def sparsity_level(self) -> int:
        return self.code_depth

    def is_dictionary_parameter(self, name: str) -> bool:
        return name.startswith("codebooks.")

    def normalize_dictionary_(self):
        return

    def project_dictionary_gradient_(self):
        return

    def dictionary_for_visualization(self, max_vectors: int) -> torch.Tensor:
        atoms = self.codebooks[0].weight[:-1, :].detach().cpu()
        max_vectors = max(1, int(max_vectors))
        if int(atoms.size(0)) <= max_vectors:
            return atoms
        indices = (
            torch.linspace(0, int(atoms.size(0)) - 1, steps=max_vectors)
            .round()
            .to(torch.long)
        )
        return atoms.index_select(0, indices)
