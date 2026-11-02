"""Depth-specific LASER dictionaries with one integer per atom/bin pair.

Grouped OMP is only the deterministic frozen image tokenizer. The resulting
integer has a fixed physical vector, independent of the other generated IDs.
"""
import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from src.training.laser_hard_ar import nearest_coefficient_ids


class DepthIntegerCodebooks(nn.Module):
    def __init__(self, dictionaries, scales, *, coefficient_bins=512,
                 coefficient_max=3., ridge=1.e-6):
        super().__init__()
        books = torch.as_tensor(dictionaries).detach().float().clone()
        scales = torch.as_tensor(scales).detach().float().clone()
        if (books.ndim != 3 or min(books.shape) < 1 or
                not torch.isfinite(books).all() or
                not torch.allclose(books.norm(dim=1), torch.ones_like(books[:, 0]), atol=2.e-5)):
            raise ValueError('Expected finite unit atoms in [depth, channels, atoms]')
        if (scales.shape != (books.shape[0],) or not torch.isfinite(scales).all()
                or (scales <= 0).any() or coefficient_bins < 2
                or not 0 < coefficient_max < float('inf') or not 0 < ridge < 1.):
            raise ValueError('Invalid coefficient coordinates or LS regularization')
        self.register_buffer('dictionaries', books)
        self.register_buffer('scales', scales)
        self.register_buffer('bins', torch.linspace(-coefficient_max, coefficient_max, coefficient_bins))
        self.depth, self.channels, self.atoms_per_book = books.shape
        self.coefficient_bins = int(coefficient_bins)
        self.vocab_size = self.atoms_per_book * self.coefficient_bins
        self.ridge = float(ridge)

    def pack(self, atoms, bins):
        if (atoms.dtype != torch.long or bins.dtype != torch.long or atoms.shape != bins.shape
                or (atoms < 0).any() or (atoms >= self.atoms_per_book).any()
                or (bins < 0).any() or (bins >= self.coefficient_bins).any()):
            raise ValueError('Invalid local atom/bin IDs')
        return atoms * self.coefficient_bins + bins

    def unpack(self, codes):
        if (codes.dtype != torch.long or codes.ndim < 1 or
                not 0 < codes.shape[-1] <= self.depth or
                (codes < 0).any() or (codes >= self.vocab_size).any()):
            raise ValueError('Expected valid integer IDs with a depth prefix')
        return codes.div(self.coefficient_bins, rounding_mode='floor'), codes.remainder(self.coefficient_bins)

    def embed_depth(self, codes, depth):
        if not 0 <= depth < self.depth:
            raise ValueError('Book depth outside tokenizer')
        if codes.dtype != torch.long or (codes < 0).any() or (codes >= self.vocab_size).any():
            raise ValueError('Pair ID outside local vocabulary')
        atoms = codes.div(self.coefficient_bins, rounding_mode='floor')
        values = self.bins[codes.remainder(self.coefficient_bins)] * self.scales[depth]
        return self.dictionaries[depth].T[atoms] * values[..., None]

    def get_code_emb_with_depth(self, codes):
        self.unpack(codes)
        return torch.stack([self.embed_depth(codes[..., d], d)
                            for d in range(codes.shape[-1])], -2), None

    def reconstruct(self, atoms, coefficients):
        if atoms.shape != coefficients.shape or atoms.shape[-1] != self.depth:
            raise ValueError('Expected one atom and physical coefficient per book')
        return sum(self.dictionaries[d].T[atoms[..., d]] * coefficients[..., d, None]
                   for d in range(self.depth))

    @torch.no_grad()
    def encode_components(self, signals, *, chunk_size=2048):
        """Select one atom per book and jointly refit the selected coefficients."""
        if (signals.dtype != torch.float32 or signals.shape[-1] != self.channels
                or signals.device != self.dictionaries.device or chunk_size < 1):
            raise ValueError('Matching FP32 latent vectors and positive chunk size required')
        x = signals.reshape(-1, self.channels)
        atoms = torch.empty(len(x), self.depth, device=x.device, dtype=torch.long)
        coefficients = x.new_empty(len(x), self.depth)
        for first in range(0, len(x), chunk_size):
            target = x[first:first+chunk_size]
            residual = target
            selected = []
            for d in range(self.depth):
                atom = (residual @ self.dictionaries[d]).abs().argmax(-1)
                atoms[first:first+len(target), d] = atom
                selected.append(self.dictionaries[d].T[atom])
                vectors = torch.stack(selected, 1)
                gram = vectors @ vectors.transpose(1, 2)
                gram.diagonal(dim1=-2, dim2=-1).add_(self.ridge)
                rhs = (vectors * target[:, None]).sum(-1, keepdim=True)
                values = torch.linalg.solve(gram, rhs).squeeze(-1)
                residual = target - (vectors * values[..., None]).sum(1)
            coefficients[first:first+len(target)] = values
        if not torch.isfinite(coefficients).all():
            raise ValueError('Nonfinite grouped OMP coefficients')
        shape = (*signals.shape[:-1], self.depth)
        return atoms.reshape(shape), coefficients.reshape(shape)

    @torch.no_grad()
    def quantize(self, signals, *, chunk_size=2048):
        atoms, coefficients = self.encode_components(signals, chunk_size=chunk_size)
        bins = nearest_coefficient_ids(coefficients / self.scales, self.bins)
        codes = self.pack(atoms, bins)
        return dict(codes=codes, atoms=atoms, coefficients=coefficients,
                    quantized=self.get_code_emb_with_depth(codes)[0].sum(-2))


class FrozenDepthIntegerLaserTokenizer(nn.Module):
    def __init__(self, checkpoint, codebook):
        super().__init__()
        from src.models.laser_tokenizer_adapter import FrozenLaserTokenizer
        spec = torch.load(codebook, map_location='cpu', weights_only=True)
        if (spec.get('kind') != 'laser_depth_integer_codebooks' or spec.get('format_version') != 1
                or spec.get('encoder_decoder_frozen') is not True):
            raise ValueError('Expected a frozen-CNN depth integer codebook')
        with Path(checkpoint).open('rb') as handle:
            if hashlib.file_digest(handle, 'sha256').hexdigest() != spec['source_stage1_sha256']:
                raise ValueError('Stage-1 checkpoint SHA256 mismatch')
        self.quantizer = DepthIntegerCodebooks(spec['dictionaries'], spec['scales'],
            coefficient_bins=spec['coefficient_bins'], coefficient_max=spec['coefficient_max'],
            ridge=spec['ridge'])
        self.backbone = FrozenLaserTokenizer(checkpoint, num_atoms=spec['source_atoms'],
            coeff_vocab_size=2, coeff_max=1., coeff_scales=[1.] * self.quantizer.depth,
            sparsity_level=self.quantizer.depth)
        self.code_shape = torch.Size([8, 8, self.quantizer.depth])
        self.vocab_size = self.quantizer.vocab_size
        self.requires_grad_(False).eval()

    @torch.no_grad()
    def encode(self, images):
        return self.backbone.quant_conv(self.backbone.encoder(images.float())).permute(0, 2, 3, 1).contiguous()

    @torch.no_grad()
    def decode(self, latent):
        return self.backbone.decoder(self.backbone.post_quant_conv(latent.permute(0, 3, 1, 2).contiguous()))

    def get_code_emb_with_depth(self, codes):
        return self.quantizer.get_code_emb_with_depth(codes)

    @torch.no_grad()
    def decode_code(self, codes):
        if codes.shape[-1] != self.quantizer.depth:
            raise ValueError('Decoding requires all depth books')
        return self.decode(self.get_code_emb_with_depth(codes)[0].sum(-2))

    @torch.no_grad()
    def forward(self, images):
        latent = self.encode(images)
        result = self.quantizer.quantize(latent)
        return self.decode(result['quantized']), latent.new_zeros(()), result['codes']
