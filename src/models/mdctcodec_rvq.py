"""Adapter for the authors' MDCTCodec quantizer, copied in mdctcodec_quantize.py.

The quantizer source is unchanged from PB20000090/MDCTCodec commit
1aff8b6287f4e2e66511bd4bc46f3a6bcf96edaf. Codebook and commitment losses
remain separate so the shared audio trainer can apply the upstream weights.
"""
import torch
from torch import nn

from .bottleneck_utils import SparseCodes
from .mdctcodec_quantize import ResidualVectorQuantize


class MDCTCodecRQBottleneck(nn.Module):
    def __init__(self, num_embeddings=1024, embedding_dim=32, code_depth=4, **_):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.code_depth = self.sparsity_level = int(code_depth)
        self.quantizer = ResidualVectorQuantize(
            input_dim=self.embedding_dim, codebook_dim=self.embedding_dim,
            n_codebooks=self.code_depth, codebook_size=self.num_embeddings,
            quantizer_dropout=0.0,
        )

    def forward(self, z):
        if z.ndim != 4 or z.shape[1] != self.embedding_dim:
            raise ValueError('Expected B,D,H,W latents')
        b, d, h, w = z.shape
        q, ids, _, commitment, codebook = self.quantizer(z.reshape(b, d, h*w))
        self._last_commitment_loss = commitment.detach()
        self._last_dictionary_loss = codebook.detach()
        self._last_dictionary_loss_for_backward = codebook
        ids = ids.transpose(1, 2).reshape(b, h, w, self.code_depth)
        return q.reshape_as(z), commitment, SparseCodes(
            support=ids, values=torch.ones_like(ids, dtype=z.dtype),
            num_embeddings=self.num_embeddings, code_format='rq',
        )

    @property
    def dictionary(self):
        return self.quantizer.quantizers[0].codebook.weight.t()

    @property
    def dictionary_dtype(self):
        return self.dictionary.dtype

    def normalize_dictionary_(self):
        pass  # Upstream performs normalized lookup, not parameter projection.

    def project_dictionary_gradient_(self):
        pass

    def dictionary_for_visualization(self, max_vectors):
        return self.dictionary.t().detach().cpu()[:max_vectors]
