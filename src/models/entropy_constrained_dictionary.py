"""Entropy-constrained residual dictionaries with a fixed integer token budget.

Assignment minimizes final latent MSE per channel plus beta times code length
in bits. Depth is known from the stream layout. Probability tables are fitted
on training codes, frozen for evaluation, and never sent per image.
"""
import hashlib
import math

import torch
from torch import nn

from src.models.joint_rate_distortion import JointVectorBook, rate_statistics


def _counts(value):
    value = torch.as_tensor(value).detach().double()
    if (value.ndim != 2 or value.shape[1] < 2 or
            not torch.isfinite(value).all() or (value < 0).any() or
            (value.sum(-1) <= 0).any()):
        raise ValueError('Expected finite nonnegative [depth, vocabulary] counts with observations')
    return value


def fit_probabilities(counts, pseudocount=.5):
    """A full-support, depth-conditioned model fitted only on training counts."""
    counts = _counts(counts)
    if not math.isfinite(pseudocount) or pseudocount <= 0:
        raise ValueError('Positive finite pseudocount required')
    return (counts + pseudocount) / (counts.sum(-1, keepdim=True) + counts.shape[-1] * pseudocount)


def coding_statistics(training_counts, evaluation_counts, *, pixels=65536,
                      tokens_per_image=256, pseudocount=.5):
    """Separate empirical entropy from held-out model cross entropy.

    Neither quantity is an actual bitstream size or a trained AR rate. The
    model cost excludes the one-time shared codebook and probability tables.
    """
    train, evaluation = _counts(training_counts).cpu(), _counts(evaluation_counts).cpu()
    if train.shape != evaluation.shape:
        raise ValueError('Training and evaluation alphabets must match')
    if (not math.isfinite(pixels) or pixels <= 0 or
            not math.isfinite(tokens_per_image) or tokens_per_image <= 0):
        raise ValueError('Positive finite pixel and token counts required')
    if not torch.equal(evaluation.sum(-1), evaluation.sum(-1)[:1].expand(evaluation.shape[0])):
        raise ValueError('Fixed-depth streams need the same number of observations at each depth')
    probability = fit_probabilities(train, pseudocount)
    lengths = -probability.log2()
    per_depth = (evaluation * lengths).sum(-1) / evaluation.sum(-1)
    per_token = float((evaluation * lengths).sum() / evaluation.sum())
    base = rate_statistics(evaluation, pixels=pixels, tokens_per_image=tokens_per_image)
    depth_entropy = base['depth_conditional_entropy_bits_per_token']
    marginal_entropy = base['marginal_entropy_bits_per_token']
    return dict(base,
        marginal_entropy_bpp=marginal_entropy * tokens_per_image / pixels,
        depth_conditional_entropy_bpp=depth_entropy * tokens_per_image / pixels,
        effective_vocabulary=2. ** marginal_entropy,
        effective_vocabulary_by_depth=[2. ** h for h in base['entropy_bits_by_depth']],
        heldout_cross_entropy_bits_per_token=per_token,
        heldout_cross_entropy_bits_by_depth=per_depth.tolist(),
        heldout_model_bits_per_image=per_token * tokens_per_image,
        heldout_model_bpp=per_token * tokens_per_image / pixels,
        heldout_model_redundancy_bits_per_token=per_token - depth_entropy,
        training_observations_by_depth=train.sum(-1).tolist(),
        evaluation_observations_by_depth=evaluation.sum(-1).tolist(),
        probability_pseudocount=pseudocount,
        entropy_model_fitting_split='train', evaluation_used_for_entropy_model=False,
        actual_bitstream_measured=False, shared_model_overhead_included=False)


class EntropyConstrainedBook(JointVectorBook):
    def __init__(self, vectors, probabilities=None):
        super().__init__(vectors)
        if probabilities is None:
            probabilities = torch.ones(self.depth, self.vocab_size)
        probabilities = torch.as_tensor(probabilities, device=self.vectors.device).double()
        if (probabilities.shape != (self.depth, self.vocab_size) or
                not torch.isfinite(probabilities).all() or (probabilities <= 0).any()):
            raise ValueError('Expected positive finite probabilities [depth, vocabulary]')
        probabilities = probabilities / probabilities.sum(-1, keepdim=True)
        self.register_buffer('probabilities', probabilities)

    @torch.no_grad()
    def code_counts(self, codes):
        if codes.dtype != torch.long or codes.shape[-1] != self.depth:
            raise ValueError('Expected integer codes ending in the depth axis')
        return torch.stack([torch.bincount(codes[..., d].flatten(), minlength=self.vocab_size)
                            for d in range(self.depth)])

    @torch.no_grad()
    def update_probabilities(self, codes, *, pseudocount=.5):
        self.probabilities.copy_(fit_probabilities(self.code_counts(codes), pseudocount))

    @torch.no_grad()
    def quantize(self, x, *, chunk_size=1024, rate_weight=0., sweeps=0, initial_codes=None):
        if (not math.isfinite(rate_weight) or rate_weight < 0 or
                not isinstance(sweeps, int) or sweeps < 0 or chunk_size < 1):
            raise ValueError('Nonnegative finite rate weight, integer sweeps and positive chunk size required')
        if x.ndim < 2 or x.shape[-1] != self.channels or x.numel() == 0:
            raise ValueError('Expected nonempty signals ending in the channel axis')
        shape = (*x.shape[:-1], self.depth)
        residual = x.reshape(-1, self.channels).float().clone()
        lengths = -self.probabilities.log2().float()
        norms = self.vectors.square().sum(-1)
        if initial_codes is None:
            codes = torch.empty(len(residual), self.depth, dtype=torch.long, device=x.device)
        else:
            if (initial_codes.shape != shape or initial_codes.dtype != torch.long or
                    initial_codes.device != x.device or (initial_codes < 0).any() or
                    (initial_codes >= self.vocab_size).any()):
                raise ValueError('Initial codes must match the signal shape, device and alphabet')
            codes = initial_codes.reshape(-1, self.depth).clone()
            for d in range(self.depth):
                residual.sub_(self.vectors[d, codes[:, d]])

        def assign(d, replace):
            table = self.vectors[d]
            for first in range(0, len(residual), chunk_size):
                r = residual[first:first + chunk_size]
                old = codes[first:first + len(r), d].clone() if replace else None
                if replace:
                    r.add_(table[old])
                # Squared distance / channels + beta * (-log2 p).
                score = (2 * r @ table.T - norms[d]) / self.channels
                score.sub_(rate_weight * lengths[d])
                chosen = score.argmax(-1)
                if replace:
                    rows = torch.arange(len(r), device=r.device)
                    # Keep existing assignments on ties to avoid cycling.
                    chosen = torch.where(score[rows, chosen] > score[rows, old], chosen, old)
                codes[first:first + len(r), d] = chosen
                r.sub_(table[chosen])

        if initial_codes is None:
            for d in range(self.depth):
                assign(d, False)
        for _ in range(sweeps):
            for d in range(self.depth):
                assign(d, True)
        # The reconstruction is produced from transmitted IDs alone.
        quantized = torch.zeros_like(residual)
        for d in range(self.depth):
            quantized.add_(self.vectors[d, codes[:, d]])
        return dict(codes=codes.reshape(shape), quantized=quantized.reshape_as(x))

    @torch.no_grad()
    def objective(self, x, codes, *, rate_weight=0.):
        if not math.isfinite(rate_weight) or rate_weight < 0:
            raise ValueError('Expected a nonnegative finite rate weight')
        reconstruction = self.embed(codes).sum(-2)
        mse = (x.float() - reconstruction).double().square().mean()
        counts = self.code_counts(codes).double()
        bits_per_site = (counts * -self.probabilities.log2()).sum() / (codes.numel() / self.depth)
        return float(mse + rate_weight * bits_per_site)


class DictionaryPriorProbe(nn.Module):
    """Small causal diagnostic with identical body initialization across alphabets."""
    def __init__(self, vocabulary, *, width=256, layers=4, heads=8, tokens=256, dropout=.1):
        super().__init__()
        from src.models.transformer_core import TransformerBlock
        self.embedding = nn.Embedding(vocabulary, width)
        self.classes = nn.Embedding(1000, width)
        self.positions = nn.Embedding(tokens, width)
        self.blocks = nn.ModuleList([TransformerBlock(width, heads, 4*width, dropout,
            attention_dropout=0., ffn_intermediate_dropout=0.) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                seed = 261026 + int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'little')
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    nn.init.normal_(module.weight, std=.02)
                    if getattr(module, 'bias', None) is not None:nn.init.zeros_(module.bias)

    def forward(self, codes, labels):
        h = torch.cat((self.classes(labels)[:, None], self.embedding(codes[:, :-1])), 1)
        h = h + self.positions(torch.arange(codes.shape[1], device=codes.device))
        for block in self.blocks:h = block(h)[0]
        return torch.nn.functional.linear(self.norm(h), self.embedding.weight)
