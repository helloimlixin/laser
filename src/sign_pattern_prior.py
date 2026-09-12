"""Oracle sparse-sign priors with identical causal inputs for both factorizations.

The current site's support and magnitudes are observed. Signed coefficients
are visible only at strictly earlier raster sites. This is a conditional
diagnostic, not an unconditional image generator.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def sign_patterns(depth, *, device=None):
    if not 1 <= depth <= 8:
        raise ValueError("sign-pattern depth must be between 1 and 8")
    ids = torch.arange(2**depth, device=device)
    return ((ids[:, None] >> torch.arange(depth, device=device)) & 1).bool()


def pack_signs(positive):
    weights = 2 ** torch.arange(positive.shape[-1], device=positive.device)
    return (positive.long() * weights).sum(-1)


def pattern_log_probabilities(logits, mode, depth):
    """Represent either head as a normalized distribution over all sign tuples."""
    logits = logits.float()
    if mode == "joint":
        return logits.log_softmax(-1)
    if mode != "independent":
        raise ValueError(mode)
    patterns = sign_patterns(depth, device=logits.device)
    return torch.where(
        patterns, F.logsigmoid(logits).unsqueeze(-2),
        F.logsigmoid(-logits).unsqueeze(-2),
    ).sum(-1)


class CausalBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(),
            nn.Linear(4 * width, width), nn.Dropout(dropout),
        )

    def forward(self, x):
        batch, sites, width = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(
            batch, sites, 3, self.heads, width // self.heads
        ).permute(2, 0, 3, 1, 4)
        attention = F.scaled_dot_product_attention(
            *qkv.unbind(0), is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        ).transpose(1, 2).reshape(batch, sites, width)
        x = x + F.dropout(self.out(attention), self.dropout, self.training)
        return x + self.mlp(self.norm2(x))


class SignPatternPrior(nn.Module):
    def __init__(self, dictionary, coefficient_rms, *, depth=4, sites=64,
                 width=256, layers=4, heads=8, dropout=0.1, mode="joint"):
        super().__init__()
        sign_patterns(depth)
        if mode not in {"joint", "independent"}:
            raise ValueError(mode)
        if width % heads:
            raise ValueError("width must be divisible by heads")
        if dictionary.ndim != 2 or coefficient_rms.shape != (depth,):
            raise ValueError("expected dictionary [channels, atoms] and RMS [depth]")
        if (not torch.isfinite(dictionary).all() or not torch.isfinite(coefficient_rms).all()
                or not (coefficient_rms > 0).all()):
            raise ValueError("dictionary and scales must be finite and scales positive")
        self.mode, self.depth, self.sites = mode, depth, sites
        self.register_buffer("dictionary", dictionary.float().detach().clone())
        self.register_buffer("coefficient_rms", coefficient_rms.float().clone())
        self.atom_embedding = nn.Embedding(dictionary.shape[1], 32)
        feature_width = depth * (2 * dictionary.shape[0] + 33)
        self.unsigned_projection = nn.Linear(feature_width, width)
        self.history_projection = nn.Linear(feature_width, width)
        self.bos = nn.Parameter(torch.zeros(1, 1, width))
        self.position = nn.Parameter(torch.randn(1, sites, width) * 0.02)
        self.blocks = nn.Sequential(*[
            CausalBlock(width, heads, dropout) for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 2**depth if mode == "joint" else depth)

    def features(self, atoms, coefficients):
        normalized = coefficients / self.coefficient_rms
        directions = self.dictionary.t()[atoms.long()] * math.sqrt(self.dictionary.shape[0])
        return torch.cat((
            directions,
            directions * normalized.unsqueeze(-1),
            self.atom_embedding(atoms.long()),
            normalized.unsqueeze(-1),
        ), -1).flatten(-2)

    def forward(self, atoms, magnitudes, signed_history):
        """Inputs [batch, sites, depth]; history is shifted internally by one site."""
        if atoms.shape != magnitudes.shape or atoms.shape != signed_history.shape:
            raise ValueError("support, magnitude and history shapes must match")
        if atoms.ndim != 3 or atoms.shape[-1] != self.depth or atoms.shape[1] > self.sites:
            raise ValueError("expected [batch, sites, configured depth]")
        unsigned = self.unsigned_projection(self.features(atoms, magnitudes.abs()))
        signed = self.history_projection(self.features(atoms, signed_history))
        previous = torch.cat((self.bos.expand(atoms.shape[0], -1, -1), signed[:, :-1]), 1)
        return self.head(self.norm(self.blocks(
            unsigned + previous + self.position[:, :atoms.shape[1]]
        )))

    @torch.no_grad()
    def rollout(self, atoms, magnitudes):
        """Oracle supports/magnitudes, but generated signs in previous sites."""
        history = torch.zeros_like(magnitudes)
        for site in range(atoms.shape[1]):
            logits = self(atoms[:, :site + 1], magnitudes[:, :site + 1], history[:, :site + 1])[:, -1]
            log_probs = pattern_log_probabilities(logits, self.mode, self.depth)
            patterns = sign_patterns(self.depth, device=atoms.device)
            signs = patterns[log_probs.argmax(-1)].float() * 2 - 1
            history[:, site] = magnitudes[:, site] * signs
        return history


def sign_metrics(logits, coefficients, atoms, dictionary, mode):
    """Per-image sufficient statistics, suitable for exact population averaging."""
    depth = coefficients.shape[-1]
    patterns = sign_patterns(depth, device=coefficients.device)
    log_probs = pattern_log_probabilities(logits, mode, depth)
    target = coefficients >= 0
    nll = -log_probs.gather(-1, pack_signs(target).unsqueeze(-1)).squeeze(-1)
    # Elementwise reduction keeps metrics in FP32 even inside BF16 autocast.
    probability_positive = (log_probs.exp().unsqueeze(-1) * patterns.float()).sum(-2)
    map_signs = patterns[log_probs.argmax(-1)]
    marginal_signs = probability_positive >= 0.5
    predicted = coefficients.abs() * (map_signs.float() * 2 - 1)
    error = predicted - coefficients
    directions = dictionary.t()[atoms.long()]
    latent_error = (directions * error.unsqueeze(-1)).sum(-2)
    latent_target = (directions * coefficients.unsqueeze(-1)).sum(-2)
    return {
        "pattern_nll": nll.mean(1),
        "pattern_accuracy": (map_signs == target).all(-1).float().mean(1),
        "sign_accuracy": (map_signs == target).float().mean((1, 2)),
        "marginal_sign_accuracy": (marginal_signs == target).float().mean((1, 2)),
        "sign_brier": (probability_positive - target.float()).square().mean((1, 2)),
        "physical_coefficient_mse": error.square().mean((1, 2)),
        "latent_mse": latent_error.square().mean((1, 2)),
        "latent_energy": latent_target.square().mean((1, 2)),
        **{f"sign_accuracy_depth{d}": (map_signs[..., d] == target[..., d]).float().mean(1)
           for d in range(depth)},
    }
