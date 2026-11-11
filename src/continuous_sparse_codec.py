"""Discrete supports paired with physical FP32 OMP values; no scalar bins."""
from dataclasses import dataclass

import torch

from src.training.laser_hard_ar import sparse_encoding


@dataclass(frozen=True)
class ContinuousSparseCodes:
    atoms: torch.Tensor
    coefficients: torch.Tensor

    @property
    def shape(self):
        return self.atoms.shape

    def split(self, size):
        return [ContinuousSparseCodes(a, c) for a, c in zip(
            self.atoms.split(size), self.coefficients.split(size))]

    def validate(self, num_atoms, depth):
        a, c = self.atoms, self.coefficients
        if (a.dtype != torch.long or c.dtype != torch.float32 or a.shape != c.shape
                or a.ndim != 4 or a.shape[-1] != depth or a.device != c.device):
            raise ValueError('Expected equal [B,H,W,K] atom IDs and physical FP32 coefficients')
        if ((a < 0) | (a >= num_atoms)).any() or not c.isfinite().all():
            raise ValueError('Invalid sparse support or nonfinite coefficient')
        if (a.sort(-1).values.diff(dim=-1) == 0).any():
            raise ValueError('OMP supports must contain distinct atoms')


@torch.no_grad()
def continuous_image_codes(aux, images, *, encoder_batch_size=8):
    """Use exactly the deterministic hard-compound support and final OMP refit."""
    parts = []
    with torch.autocast(device_type=images.device.type, enabled=False):
        if not hasattr(aux, '_hard_ar_gram'):
            aux._hard_ar_gram = (aux.dictionary.float().T @ aux.dictionary.float()).contiguous()
        for chunk in images.split(encoder_batch_size):
            signals = aux.quant_conv(aux.encoder(chunk.float())).permute(0, 2, 3, 1).float()
            parts.append(sparse_encoding(signals, aux.dictionary.float(), aux.sparsity_level,
                                         gram=aux._hard_ar_gram))
    return ContinuousSparseCodes(*(torch.cat([p[i] for p in parts]) for i in (0, 1)))


def continuous_latents(codes, dictionary):
    with torch.autocast(device_type=dictionary.device.type, enabled=False):
        return (dictionary.float().T[codes.atoms] * codes.coefficients[..., None]).sum(-2)
