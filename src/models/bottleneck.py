"""Bottleneck implementations.

The concrete classes live in focused modules; this module re-exports them so
existing ``from src.models.bottleneck import ...`` imports keep working:

- :mod:`src.models.quantizers` — :class:`VectorQuantizer`, :class:`VectorQuantizerEMA`
- :mod:`src.models.dictionary_learner` — :class:`DictionaryLearning`
- :mod:`src.models.bottleneck_utils` — :class:`SparseCodes` and math helpers
"""

from .bottleneck_utils import SparseCodes, _normalize_dictionary
from .dictionary_learner import DictionaryLearning
from .quantizers import VectorQuantizer, VectorQuantizerEMA

__all__ = [
    "DictionaryLearning",
    "SparseCodes",
    "VectorQuantizer",
    "VectorQuantizerEMA",
]
