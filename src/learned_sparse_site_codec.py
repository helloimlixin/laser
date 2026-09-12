"""Image-loss-trained, single-ID complete sparse codewords.

The encoder can use spatial context. Decoding an ID is pointwise: its learned
codeword is projected to four dictionary atoms and signed coefficient bins.
Hard projections are used in every forward; gradients use an explicit surrogate.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SparseSiteProjector(nn.Module):
    def __init__(self, dictionary_rows, bins, scales, ridge=1e-6):
        super().__init__()
        dictionary_rows, bins, scales = dictionary_rows.float(), bins.float(), scales.float()
        if dictionary_rows.ndim != 2 or bins.ndim != 1 or scales.ndim != 1:
            raise ValueError('Expected dictionary [atoms, channels], bins [bins], scales [depth]')
        if not all(torch.isfinite(x).all() for x in (dictionary_rows, bins, scales)):
            raise ValueError('Projector values must be finite')
        if not (scales > 0).all() or not (bins[1:] > bins[:-1]).all():
            raise ValueError('Expected positive scales and strictly increasing bins')
        if not 0 < len(scales) <= len(dictionary_rows) or ridge <= 0:
            raise ValueError('Invalid sparsity or solve regularization')
        self.ridge = float(ridge)
        self.register_buffer('dictionary', dictionary_rows.detach().clone(), persistent=False)
        self.register_buffer('bins', bins.detach().clone(), persistent=False)
        self.register_buffer('scales', scales.detach().clone(), persistent=False)
        self.register_buffer('gram', dictionary_rows @ dictionary_rows.t(), persistent=False)

    @torch.no_grad()
    def forward(self, vectors):
        shape = vectors.shape[:-1]
        with torch.autocast(vectors.device.type, enabled=False):
            signals = vectors.reshape(-1, vectors.shape[-1]).float()
            original = signals @ self.dictionary.t()
            correlations = original.clone()
            support = torch.empty(len(signals), 0, dtype=torch.long, device=signals.device)
            for depth in range(1, len(self.scales) + 1):
                scores = correlations.abs()
                if depth > 1:
                    scores.scatter_(1, support, -1.)
                selected = scores.argmax(-1)
                support = torch.cat((support, selected[:, None]), -1)
                matrices = self.gram[support[:, :, None], support[:, None, :]]
                matrices = matrices + self.ridge * torch.eye(depth, device=signals.device)
                coefficients = torch.linalg.solve(matrices, original.gather(1, support).unsqueeze(-1)).squeeze(-1)
                if depth < len(self.scales):
                    correlations = original - torch.bmm(coefficients[:, None, :], self.gram[support]).squeeze(1)
            normalized = coefficients / self.scales
            boundaries = (self.bins[1:] + self.bins[:-1]) / 2
            ids = torch.bucketize(normalized.contiguous(), boundaries)
            physical = self.bins[ids] * self.scales
            hard = (self.dictionary[support] * physical[..., None]).sum(-2)
            clipped = ((normalized < self.bins[0]) | (normalized > self.bins[-1])).float().mean()
        return {'latents': hard.reshape(*shape, signals.shape[-1]),
                'atoms': support.reshape(*shape, len(self.scales)),
                'coefficient_ids': ids.reshape(*shape, len(self.scales)),
                'clipped_fraction': clipped}


class CodecResidual(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.GroupNorm(8, width), nn.SiLU(), nn.Conv2d(width, width, 3, padding=1),
                                    nn.GroupNorm(8, width), nn.SiLU(), nn.Conv2d(width, width, 3, padding=1))

    def forward(self, x):
        return x + self.layers(x)


class LearnedSparseSiteCodec(nn.Module):
    def __init__(self, initial_codewords, width=256, layers=3):
        super().__init__()
        if initial_codewords.ndim != 2 or not torch.isfinite(initial_codewords).all():
            raise ValueError('Expected finite initial complete-codeword latents')
        if width % 8 or width < 8 or layers < 1:
            raise ValueError('Width must be a positive multiple of eight and layers positive')
        size, channels = initial_codewords.shape
        self.config = {'vocabulary': size, 'channels': channels, 'width': width, 'layers': layers}
        self.codewords = nn.Parameter(initial_codewords.float().clone())
        self.register_buffer('mean', torch.zeros(1, channels, 1, 1))
        self.register_buffer('scale', torch.ones(1, channels, 1, 1))
        self.encoder = nn.Sequential(nn.Conv2d(channels, width, 1),
            *[CodecResidual(width) for _ in range(layers)], nn.GroupNorm(8, width), nn.SiLU(), nn.Conv2d(width, channels, 1))
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)

    @torch.no_grad()
    def set_normalization(self, values):
        self.mean.copy_(values.mean((0, 2, 3), keepdim=True))
        self.scale.copy_(values.std((0, 2, 3), keepdim=True).clamp_min(.05))

    def encode(self, z):
        normalized = (z - self.mean) / self.scale
        encoded = z + self.scale * self.encoder(normalized)
        flat = encoded.permute(0, 2, 3, 1).reshape(-1, z.shape[1]).float()
        # Quantizer assignments are discrete and do not construct a huge backward graph.
        with torch.no_grad(), torch.autocast(z.device.type, enabled=False):
            table = self.codewords.float()
            distance = flat.square().sum(-1, keepdim=True) + table.square().sum(-1)[None]
            distance.addmm_(flat, table.t(), beta=1., alpha=-2.)
            ids = distance.argmin(-1).reshape(z.shape[0], z.shape[2], z.shape[3])
        return ids, encoded

    def decode_ids(self, ids, projector):
        if ids.is_floating_point() or (ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= len(self.codewords))):
            raise ValueError('Expected integer site IDs within the vocabulary')
        unique, inverse = torch.unique(ids.long(), return_inverse=True)
        projected = projector(self.codewords[unique])
        return {k: (v[inverse] if k != 'clipped_fraction' else v) for k, v in projected.items()}

    def forward(self, z, projector):
        ids, encoded = self.encode(z)
        selected = self.codewords[ids].permute(0, 3, 1, 2)
        projected = self.decode_ids(ids, projector)
        hard = projected['latents'].permute(0, 3, 1, 2)
        # Forward exactly equals ID-only decoding. The surrogate sends the image
        # gradient to the selected codewords AND the encoder, through hard OMP.
        carrier = selected + (encoded - encoded.detach())
        reconstructed = hard + (carrier - carrier.detach())
        losses = {
            'commitment': F.mse_loss(encoded / self.scale, selected.detach() / self.scale),
            'alignment': F.mse_loss(selected / self.scale, encoded.detach() / self.scale),
            'projection': F.mse_loss(selected / self.scale, hard.detach() / self.scale),
        }
        return reconstructed, ids, losses, projected

    @torch.no_grad()
    def export_codebook(self, projector, batch_size=512):
        rows = [projector(self.codewords[start:start + batch_size]) for start in range(0, len(self.codewords), batch_size)]
        return {name: torch.cat([r[name] for r in rows]).cpu() for name in ('atoms', 'coefficient_ids', 'latents')}
