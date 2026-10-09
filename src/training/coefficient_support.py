"""Restrict coefficient sampling before selected tokens enter the AR history."""
from contextlib import contextmanager
import math

import torch


@contextmanager
def sampling_coefficient_support(model, bins, limits):
    """Mask each depth's logits, leaving training and the uniform bins intact."""
    if limits is None:
        yield
        return
    limits = tuple(float(value) for value in limits)
    if len(limits) != model.block_size[-1] or any(
            not math.isfinite(value) or value <= 0 for value in limits):
        raise ValueError('Require one finite positive coefficient limit per depth')
    if not isinstance(model.coeff_classifier, torch.nn.ModuleList):
        raise ValueError('Coefficient support requires depth-specific classifiers')
    masks = [bins.abs() > value for value in limits]
    if any(bool(mask.all()) for mask in masks):
        raise ValueError('Coefficient limit excludes every bin')
    handles = []
    try:
        for classifier, mask in zip(model.coeff_classifier, masks):
            def restrict(_module, _inputs, logits, mask=mask):
                return logits.masked_fill(mask.to(logits.device), -float('inf'))
            handles.append(classifier.register_forward_hook(restrict))
        yield
    finally:
        for handle in handles:
            handle.remove()
