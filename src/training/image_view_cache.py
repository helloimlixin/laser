"""Select coherent image views from paired sparse-token caches.

Views originate from pixel-space augmentation before encoding. One view is
selected for the entire image, never independently across sites or depths.
Draws use the training device's checkpointed RNG, outside DataLoader workers.
"""
import torch


IMAGE_VIEW_FORMAT = "laser_compound_image_views_v1"
ORTHOGONAL_IMAGE_VIEW_FORMAT = "laser_orthogonal_image_views_v1"
IMAGE_VIEW_FORMATS = {IMAGE_VIEW_FORMAT, ORTHOGONAL_IMAGE_VIEW_FORMAT}


def validate_image_view_cache(atoms, coefficients, labels, metadata):
    if atoms.shape != coefficients.shape or atoms.ndim != 5:
        raise ValueError("image views require equal [N,V,H,W,K] paired tensors")
    if atoms.shape[1] < 2 or atoms.shape[1] != metadata.get("views_per_image"):
        raise ValueError("image-view count differs from metadata")
    if list(atoms.shape[2:]) != metadata.get("shape"):
        raise ValueError("image-view token shape differs from metadata")
    if coefficients.dtype != torch.float32:
        raise ValueError("image-view coefficients must retain FP32 precision")
    if atoms.dtype not in (torch.int16, torch.int32, torch.int64):
        raise ValueError("image-view atoms must be integer IDs")
    if labels.ndim != 1 or len(labels) != len(atoms):
        raise ValueError("image-view labels must contain one entry per image")
    if not metadata.get("image_view_cache_identity"):
        raise ValueError("image-view cache needs a provenance identity")


def sample_image_views(atoms, coefficients, *, generator=None, choices=None):
    if atoms.shape != coefficients.shape or atoms.ndim != 5:
        raise ValueError("image views require equal [B,V,H,W,K] paired tensors")
    if atoms.device != coefficients.device or atoms.shape[1] < 2:
        raise ValueError("paired image views must share a device and contain multiple views")
    if choices is None:
        choices = torch.randint(atoms.shape[1], (len(atoms),),
                                device=atoms.device, generator=generator)
    if choices.shape != (len(atoms),) or choices.device != atoms.device:
        raise ValueError("choose exactly one view per image on the input device")
    rows = torch.arange(len(atoms), device=atoms.device)
    return atoms[rows, choices], coefficients[rows, choices], choices
