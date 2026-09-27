"""Lightning adapter around the original KakaoBrain residual quantizer."""

from __future__ import annotations

import torch
from torch import nn

from .bottleneck_utils import SparseCodes
from .rqvae.quantizations import RQBottleneck as UpstreamRQBottleneck


class RQBottleneck(nn.Module):
    """Expose upstream residual quantization through LASER's bottleneck contract."""

    def __init__(
        self,
        num_embeddings=2048,
        embedding_dim=256,
        code_depth=4,
        shared_codebook=True,
        decay=0.99,
        restart_unused_codes=True,
        commitment_cost=0.25,
        latent_shape=None,
        code_shape=None,
        **_ignored,
    ):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.code_depth = int(code_depth)
        self.shared_codebook = bool(shared_codebook)
        self.commitment_cost = float(commitment_cost)
        if latent_shape is None:
            raise ValueError("latent_shape=(height, width, channels) is required for RQ")
        latent_shape = tuple(int(v) for v in latent_shape)
        if code_shape is None:
            code_shape = (*latent_shape[:2], self.code_depth)
        self.quantizer = UpstreamRQBottleneck(
            latent_shape=latent_shape,
            code_shape=tuple(int(v) for v in code_shape),
            n_embed=self.num_embeddings,
            decay=float(decay),
            shared_codebook=self.shared_codebook,
            restart_unused_codes=bool(restart_unused_codes),
        )
        self._last_latent_loss = None
        self._last_dl_latent_loss = None
        self._last_e_latent_loss = None
        self._last_bottleneck_loss = torch.zeros(())

    @property
    def codebooks(self):
        return self.quantizer.codebooks

    def forward(self, z_e):
        if z_e.ndim != 4 or z_e.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected [B, {self.embedding_dim}, H, W], got {tuple(z_e.shape)}"
            )
        x = z_e.permute(0, 2, 3, 1).contiguous()
        z_q, commitment_loss, codes = self.quantizer(x)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        bottleneck_loss = self.commitment_cost * commitment_loss
        self._last_latent_loss = bottleneck_loss.detach()
        self._last_dl_latent_loss = commitment_loss.detach()
        self._last_e_latent_loss = commitment_loss.detach()
        self._last_bottleneck_loss = bottleneck_loss.detach()
        return z_q, bottleneck_loss, SparseCodes(
            support=codes,
            values=torch.ones_like(codes, dtype=z_e.dtype),
            num_embeddings=self.num_embeddings,
            code_format="rq",
        )

    @property
    def dictionary(self):
        return self.codebooks[0].weight[:-1].t()

    @property
    def dictionary_dtype(self):
        return self.codebooks[0].weight.dtype

    @property
    def sparsity_level(self):
        return self.code_depth

    def is_dictionary_parameter(self, name):
        return name.startswith(("quantizer.codebooks.", "codebooks."))

    def normalize_dictionary_(self):
        return None

    def project_dictionary_gradient_(self):
        return None

    def dictionary_for_visualization(self, max_vectors):
        atoms = self.codebooks[0].weight[:-1].detach().cpu()
        if atoms.shape[0] <= int(max_vectors):
            return atoms
        indices = torch.linspace(0, atoms.shape[0] - 1, int(max_vectors)).round().long()
        return atoms.index_select(0, indices)


__all__ = ["RQBottleneck"]
