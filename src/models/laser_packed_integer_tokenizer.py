"""One ordinary integer for every LASER atom/coefficient-bin pair.

The mapping is the full Cartesian product, not an observed-pair shortlist.
Embedding and decoding retain the frozen source dictionary and its physical
coefficient coordinates. No learned pair adapter or coefficient decoder is
part of this tokenizer.
"""
import torch

from src.models.laser_tokenizer_adapter import FrozenLaserTokenizer
from src.training.laser_hard_ar import nearest_coefficient_ids, sparse_encoding


def pack_pair_ids(atoms, bins, num_atoms, coefficient_bins):
    if atoms.dtype != torch.long or bins.dtype != torch.long or atoms.shape != bins.shape:
        raise ValueError('Atom and coefficient IDs must be equal-shaped int64 tensors')
    if (num_atoms < 1 or coefficient_bins < 2 or (atoms < 0).any()
            or (atoms >= num_atoms).any() or (bins < 0).any() or (bins >= coefficient_bins).any()):
        raise ValueError('Pair ID outside the full atom/coefficient vocabulary')
    return atoms * coefficient_bins + bins


def unpack_pair_ids(codes, num_atoms, coefficient_bins):
    if codes.dtype != torch.long or num_atoms < 1 or coefficient_bins < 2:
        raise ValueError('Expected int64 pair IDs and a valid Cartesian vocabulary')
    if (codes < 0).any() or (codes >= num_atoms * coefficient_bins).any():
        raise ValueError('Packed ID outside the full atom/coefficient vocabulary')
    return codes.div(coefficient_bins, rounding_mode='floor'), codes.remainder(coefficient_bins)


class FrozenPackedIntegerLaserTokenizer(FrozenLaserTokenizer):
    def __init__(self, checkpoint, **options):
        super().__init__(checkpoint, **options)
        self.vocab_size = self.num_atoms * self.coeff_vocab_size
        self.code_shape = torch.Size([8, 8, self.sparsity_level])

    @torch.no_grad()
    def encode_components(self, images, encoder_batch_size=8):
        """One deterministic source sparse encoding, with no trajectory targets."""
        components = []
        with torch.autocast(device_type=images.device.type, enabled=False):
            if not hasattr(self, '_packed_gram'):
                self._packed_gram = (self.dictionary.T @ self.dictionary).contiguous()
            for chunk in images.split(encoder_batch_size):
                latent = self.quant_conv(self.encoder(chunk.float())).permute(0, 2, 3, 1).contiguous()
                atoms, values = sparse_encoding(latent.float(), self.dictionary,
                    self.sparsity_level, gram=self._packed_gram)
                components.append((atoms, values / self.coeff_scales))
        return tuple(torch.cat([row[i] for row in components]) for i in (0, 1))

    @torch.no_grad()
    def get_code_emb_with_depth(self, codes):
        atoms, bins = unpack_pair_ids(codes, self.num_atoms, self.coeff_vocab_size)
        depth = codes.shape[-1]
        if not 0 < depth <= self.sparsity_level:
            raise ValueError('Expected a complete or partial LASER depth prefix')
        values = self.coeff_bins[bins] * self.coeff_scales[:depth]
        return self.dictionary.T[atoms] * values[..., None], None

    @torch.no_grad()
    def decode_code(self, codes):
        if codes.shape[-1] != self.sparsity_level:
            raise ValueError('Decoding requires a complete depth stack')
        latent = self.get_code_emb_with_depth(codes)[0].sum(-2)
        return self.decoder(self.post_quant_conv(latent.permute(0, 3, 1, 2).contiguous())).clamp(-1., 1.)

    @torch.no_grad()
    def forward(self, images):
        atoms, values = self.encode_components(images)
        bins = nearest_coefficient_ids(values, self.coeff_bins)
        codes = pack_pair_ids(atoms, bins, self.num_atoms, self.coeff_vocab_size)
        return self.decode_code(codes), None, codes
