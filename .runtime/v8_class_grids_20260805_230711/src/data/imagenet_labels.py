"""ImageNet-1k label helpers used for cache metadata and previews."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Mapping, Sequence


_IMAGENET_SYNSET_RE = re.compile(r"^n\d{8}$")


def is_imagenet_synset(value: object) -> bool:
    return bool(_IMAGENET_SYNSET_RE.match(str(value).strip()))


def looks_like_imagenet_synsets(values: Sequence[object]) -> bool:
    names = [str(item).strip() for item in values]
    if not names:
        return False
    probe = names[: min(32, len(names))]
    return all(is_imagenet_synset(item) for item in probe)


@lru_cache(maxsize=1)
def imagenet1k_categories() -> tuple[str, ...]:
    """Return ImageNet-1k class names in canonical class-index order.

    Torchvision is already part of the training environment. This helper keeps
    the dependency optional so metadata reads still work in lightweight shells.
    """

    categories = None
    try:
        from torchvision.models import ResNet50_Weights

        categories = ResNet50_Weights.IMAGENET1K_V1.meta.get("categories")
    except Exception:
        try:
            from torchvision.models._meta import _IMAGENET_CATEGORIES

            categories = _IMAGENET_CATEGORIES
        except Exception:
            categories = None

    if not isinstance(categories, (list, tuple)) or len(categories) < 1000:
        return ()
    return tuple(str(item) for item in categories[:1000])


def ordered_class_names(raw_names) -> list[str]:
    if isinstance(raw_names, Mapping):
        ordered: list[tuple[int, str]] = []
        for name, idx in raw_names.items():
            try:
                ordered.append((int(idx), str(name)))
            except (TypeError, ValueError):
                return []
        ordered.sort(key=lambda item: item[0])
        return [name for _, name in ordered]
    if isinstance(raw_names, (str, bytes)) or raw_names is None:
        return []
    if isinstance(raw_names, Sequence):
        return [str(item) for item in raw_names]
    try:
        return [str(item) for item in list(raw_names)]
    except TypeError:
        return []


def class_names_for_dataset(dataset: str, raw_names=None) -> list[str]:
    names = ordered_class_names(raw_names)
    if str(dataset).strip().lower() != "imagenet":
        return names

    categories = list(imagenet1k_categories())
    if not categories:
        return names
    if not names:
        return categories
    if looks_like_imagenet_synsets(names):
        return categories[: len(names)]
    return names


def imagenet_synsets_from_names(dataset: str, raw_names=None) -> list[str]:
    names = ordered_class_names(raw_names)
    if str(dataset).strip().lower() != "imagenet":
        return []
    if looks_like_imagenet_synsets(names):
        return names
    return []
