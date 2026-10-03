"""Fresh, reproducible ImageNet views and memory-bounded sparse encoding."""
from __future__ import annotations

import torch
from torchvision.datasets import ImageFolder
from src.orthogonal_sparse_codec import dictionary_to_orthogonal_coefficients


class EpochImageFolder(ImageFolder):
    """Draw a fresh crop/flip per epoch, independent of worker prefetch/resumes."""

    def __init__(self, *args, augmentation_seed=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentation_seed = int(augmentation_seed)
        self._augmentation_epoch = torch.zeros((), dtype=torch.int64).share_memory_()

    def set_epoch(self, epoch):
        self._augmentation_epoch.fill_(int(epoch))

    def __getitem__(self, index):
        epoch = int(self._augmentation_epoch)
        # Stable per-image stream; no dependency on rank, batch size or workers.
        seed = (self.augmentation_seed + 1000003 * epoch + 1000000007 * int(index)) % (2**63 - 1)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            return super().__getitem__(index)


@torch.no_grad()
def encode_dictionary_images(aux, images, *, chunk_size=32, return_prefix_coeffs=False):
    """Keep the dictionary cache's FP32 encoder/OMP precision on fresh views.

    Reuse the dictionary Gram matrix across chunks, bounding encoder memory
    without changing the dictionary coefficients or their per-depth scales.
    """
    if chunk_size <= 0:
        raise ValueError('encoder chunk size must be positive')
    outputs = None
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.autocast(device_type=images.device.type, enabled=False):
            dictionary = aux.dictionary.float()
            gram = dictionary.t() @ dictionary
            for chunk in images.split(chunk_size):
                encoded = aux.encode_sparse_components(
                    chunk.float(), dictionary_gram=gram,
                    return_prefix_coeffs=return_prefix_coeffs,
                )
                if outputs is None:
                    outputs = [[] for _ in encoded]
                for batches, value in zip(outputs, encoded):
                    batches.append(value)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return tuple(torch.cat(batches) for batches in outputs)


@torch.no_grad()
def encode_orthogonal_images(aux, images, *, chunk_size=32):
    """Encode fresh images in the same coordinates as the orthogonal cache.

    The encoder and OMP solve use FP32 (TF32 convolutions are allowed by the
    caller), matching the source cache. Chunking bounds frozen encoder memory.
    Dictionary coefficients must be converted before orthogonal decoding.
    """
    if chunk_size <= 0:
        raise ValueError('encoder chunk size must be positive')
    if aux.clamp_coeffs:
        raise ValueError('orthogonal online encoding requires unclipped dictionary coefficients')
    atoms_out, coeffs_out = [], []
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.autocast(device_type=images.device.type, enabled=False):
            dictionary = aux.dictionary.float()
            gram = dictionary.t() @ dictionary
            for chunk in images.split(chunk_size):
                atoms, dictionary_coeffs = aux.encode_sparse_components(chunk.float(), dictionary_gram=gram)
                scales = aux.coeff_scales.view(*([1] * (dictionary_coeffs.ndim - 1)), -1)
                physical = dictionary_coeffs.float() * scales
                support = aux.dictionary.t()[atoms.long()]
                gamma, _ = dictionary_to_orthogonal_coefficients(support, physical)
                atoms_out.append(atoms)
                coeffs_out.append(gamma / scales)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return torch.cat(atoms_out), torch.cat(coeffs_out)
