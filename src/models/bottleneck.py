"""Bottleneck implementations.

The concrete classes live in focused modules; this module re-exports them so
existing ``from src.models.bottleneck import ...`` imports keep working:

- :mod:`src.models.quantizers` — :class:`VectorQuantizer`, :class:`VectorQuantizerEMA`
- :mod:`src.models.dictionary_learning` — :class:`DictionaryLearning`, :class:`SparseCodes`
"""

from .dictionary_learning import (
    DictionaryLearning,
    SparseCodes,
    _dictionary_abs_offdiag_cosines,
    _gaussian_kl_to_fixed_mean,
    _normalize_dictionary,
)
from .quantizers import VectorQuantizer, VectorQuantizerEMA

__all__ = [
    "DictionaryLearning",
    "SparseCodes",
    "VectorQuantizer",
    "VectorQuantizerEMA",
]
