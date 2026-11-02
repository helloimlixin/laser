"""Finite joint integer codes and a decoder which consumes only those codes.

Depth is implicit in the fixed 8x8x4 layout, so depth-specific lookup tables
do not transmit a second index. Both the plain and neural decoders use the
same integer stream. No continuous coefficient or image feature bypasses it.
"""
import math

import torch
from torch import nn


class JointVectorBook(nn.Module):
    def __init__(self, vectors):
        super().__init__()
        vectors = torch.as_tensor(vectors).float()
        if (vectors.ndim != 3 or vectors.shape[0] < 1 or vectors.shape[1] < 2 or
                vectors.shape[2] < 1 or not torch.isfinite(vectors).all()):
            raise ValueError('Expected finite [depth, vocabulary, channels] vectors')
        if vectors[:, 0].count_nonzero():
            raise ValueError('ID zero must decode to zero')
        self.register_buffer('vectors', vectors.clone())
        self.depth, self.vocab_size, self.channels = vectors.shape

    def embed(self, codes):
        if codes.dtype != torch.long or codes.shape[-1] != self.depth:
            raise ValueError('Expected integer codes ending in the depth axis')
        return torch.stack([self.vectors[d, codes[..., d]] for d in range(self.depth)], -2)

    def get_code_emb_with_depth(self, codes):
        """The same frozen-vector input interface as the native RQ backbone."""
        return self.embed(codes), None

    def embed_depth(self, codes, depth):
        return self.vectors[depth, codes]

    def expanded_codebook(self, depth):
        return self.vectors[depth]

    def scores(self, residual, depth):
        table = self.vectors[depth]
        return 2 * residual @ table.T - table.square().sum(-1)

    @torch.no_grad()
    def get_soft_codes(self, x, temp, stochastic=True, chunk_size=128):
        """Native RQ targets, with each residual conditioned on its chosen prefix."""
        try:
            temperatures = [float(temp)] * self.depth
        except (TypeError, ValueError):
            temperatures = [float(t) for t in temp]
        if (len(temperatures) != self.depth or
                any(not math.isfinite(t) or t <= 0 for t in temperatures) or chunk_size < 1):
            raise ValueError('Positive finite temperature and chunk size required')
        residual = x.reshape(-1, self.channels).float().clone()
        targets = residual.new_empty(len(residual), self.depth, self.vocab_size)
        codes = torch.empty(len(residual), self.depth, device=x.device, dtype=torch.long)
        for d in range(self.depth):
            for first in range(0, len(residual), chunk_size):
                r = residual[first:first + chunk_size]
                probability = (self.scores(r, d) / temperatures[d]).softmax(-1)
                chosen = (torch.multinomial(probability, 1).squeeze(-1)
                          if stochastic else probability.argmax(-1))
                targets[first:first + len(r), d] = probability
                codes[first:first + len(r), d] = chosen
                r.sub_(self.embed_depth(chosen, d))
        return (targets.reshape(*x.shape[:-1], self.depth, self.vocab_size),
                codes.reshape(*x.shape[:-1], self.depth))

    @torch.no_grad()
    def quantize(self, x, *, chunk_size=1024):
        residual = x.reshape(-1, self.channels).float().clone()
        codes = torch.empty(len(residual), self.depth, dtype=torch.long, device=x.device)
        for d in range(self.depth):
            table = self.vectors[d]
            norms = table.square().sum(-1)
            for first in range(0, len(residual), chunk_size):
                r = residual[first:first + chunk_size]
                chosen = (2 * r @ table.T - norms).argmax(-1)
                codes[first:first + len(r), d] = chosen
                r.sub_(table[chosen])
        return dict(codes=codes.reshape(*x.shape[:-1], self.depth),
                    quantized=x.float() - residual.reshape_as(x))

    @torch.no_grad()
    def refit(self, x, codes, *, prior_weight=8.):
        """Coordinate least squares on final reconstruction, with stable rare IDs."""
        x = x.reshape(-1, self.channels).float()
        codes = codes.reshape(-1, self.depth)
        error = x - self.embed(codes).sum(-2)
        before = float(error.square().mean())
        for d in range(self.depth):
            index = codes[:, d]
            old = self.vectors[d].clone()
            count = torch.bincount(index, minlength=self.vocab_size).float()
            sums = torch.zeros_like(old).index_add_(0, index, error + old[index])
            proposal = (sums + prior_weight * old) / (count[:, None] + prior_weight)
            proposal[0].zero_()
            self.vectors[d].copy_(proposal)
            error.sub_(proposal[index] - old[index])
        after = float(error.square().mean())
        if after > before + 1.e-6:
            raise RuntimeError('Fixed-code least squares increased distortion')
        return dict(fixed_code_mse_before=before, fixed_code_mse_after=after)


class NeuralJointDecoder(nn.Module):
    """Restore latent vectors from the four discrete lookups and their neighbors."""
    def __init__(self, depth=4, channels=256, width=256):
        super().__init__()
        self.depth, self.channels, self.width = depth, channels, width
        self.network = nn.Sequential(
            nn.Conv2d(depth * channels, width, 1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, channels, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, vectors):
        if vectors.ndim != 5 or vectors.shape[-2:] != (self.depth, self.channels):
            raise ValueError('Expected [batch, height, width, depth, channels] lookups')
        features = vectors.flatten(-2).permute(0, 3, 1, 2).contiguous()
        correction = self.network(features).permute(0, 2, 3, 1)
        return vectors.sum(-2) + correction


def rate_statistics(counts, *, pixels=256 * 256, tokens_per_image=256):
    """Marginal entropies are coding estimates, not autoregressive cross entropy."""
    counts = torch.as_tensor(counts).double().cpu()
    if counts.ndim != 2 or (counts < 0).any() or (counts.sum(-1) <= 0).any():
        raise ValueError('Expected nonnegative counts [depth, vocabulary], with observations')
    probability = counts / counts.sum(-1, keepdim=True)
    entropy = -(probability * probability.clamp_min(1.e-300).log2()).sum(-1)
    marginal = counts.sum(0) / counts.sum()
    marginal_entropy = float(-(marginal * marginal.clamp_min(1.e-300).log2()).sum())
    weights = counts.sum(-1) / counts.sum()
    conditional = float((entropy * weights).sum())
    vocabulary = counts.shape[-1]
    return dict(vocabulary=vocabulary, tokens_per_image=tokens_per_image,
        alphabet_bits_per_token=math.log2(vocabulary),
        fixed_width_bits_per_token=math.ceil(math.log2(vocabulary)),
        fixed_width_bpp=math.ceil(math.log2(vocabulary)) * tokens_per_image / pixels,
        marginal_entropy_bits_per_token=marginal_entropy,
        depth_conditional_entropy_bits_per_token=conditional,
        marginal_entropy_bpp=conditional * tokens_per_image / pixels,
        entropy_bits_by_depth=entropy.tolist(),
        used_codes_by_depth=(counts > 0).sum(-1).tolist(),
        entropy_is_autoregressive_rate=False)
