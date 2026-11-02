"""A compact, globally consistent integer catalog on the source 512-bin grid.

Each ID means exactly one (source atom, source coefficient-bin) pair at every
depth. Offline coefficient calibration restricts the available pairs; neither
the source dictionary nor the deterministic OMP support selection is changed.
"""
import hashlib
from pathlib import Path

import torch

from src.models.laser_packed_integer_tokenizer import FrozenPackedIntegerLaserTokenizer


def validate_catalog(bin_ids, num_atoms, coefficient_bins):
    if (bin_ids.dtype != torch.long or bin_ids.ndim != 2 or bin_ids.shape[0] != num_atoms
            or bin_ids.shape[1] < 2 or (bin_ids < 0).any() or (bin_ids >= coefficient_bins).any()
            or (bin_ids.diff(dim=-1) <= 0).any()):
        raise ValueError('Expected strictly increasing source coefficient-bin IDs per atom')


class FrozenCatalogIntegerLaserTokenizer(FrozenPackedIntegerLaserTokenizer):
    def __init__(self, checkpoint, catalog):
        spec = torch.load(catalog, map_location='cpu', weights_only=True)
        if spec.get('kind') != 'laser_catalog_pair_integer' or spec.get('format_version') != 1:
            raise ValueError('Expected a calibrated integer pair catalog')
        with Path(checkpoint).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != spec['source_stage1_sha256']:
                raise ValueError('Integer catalog source stage-1 SHA256 mismatch')
        super().__init__(checkpoint, num_atoms=int(spec['source_atoms']),
            coeff_vocab_size=int(spec['coefficient_bins']), coeff_max=float(spec['coefficient_max']),
            coeff_scales=spec['coefficient_scales'], sparsity_level=int(spec['depth']))
        validate_catalog(spec['bin_ids'], self.num_atoms, self.coeff_vocab_size)
        self.register_buffer('catalog_bin_ids', spec['bin_ids'].long().clone())
        self.levels_per_atom = self.catalog_bin_ids.shape[1]
        self.vocab_size = self.num_atoms * self.levels_per_atom

    def coefficient_values(self, atoms):
        return self.coeff_bins[self.catalog_bin_ids[atoms]]

    def pack_nearest(self, atoms, normalized_coefficients):
        slots = (self.coefficient_values(atoms)-normalized_coefficients[..., None]).abs().argmin(-1)
        return atoms*self.levels_per_atom+slots

    def unpack_catalog(self, codes):
        if (codes.dtype != torch.long or codes.ndim < 1 or not 0 < codes.shape[-1] <= self.sparsity_level
                or (codes < 0).any() or (codes >= self.vocab_size).any()):
            raise ValueError('Expected integer catalog IDs and a valid depth prefix')
        atoms = codes.div(self.levels_per_atom, rounding_mode='floor')
        slots = codes.remainder(self.levels_per_atom)
        return atoms, self.catalog_bin_ids[atoms, slots]

    @torch.no_grad()
    def get_code_emb_with_depth(self, codes):
        atoms, bins = self.unpack_catalog(codes)
        coefficients = self.coeff_bins[bins]*self.coeff_scales[:codes.shape[-1]]
        return self.dictionary.T[atoms]*coefficients[..., None], None

    @torch.no_grad()
    def forward(self, images):
        atoms, values = self.encode_components(images)
        codes = self.pack_nearest(atoms, values)
        return self.decode_code(codes), None, codes
