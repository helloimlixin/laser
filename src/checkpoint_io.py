"""Safe checkpoint helpers for LASER/VQ-VAE utilities.

These helpers avoid Lightning's default checkpoint loader path, which still
goes through ``torch.load(..., weights_only=False)`` in some releases and emits
the future-compatibility warning about unsafe pickle loads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import torch

T = TypeVar("T", bound=torch.nn.Module)


def load_torch_payload(path, *, map_location="cpu") -> Any:
    """Load a checkpoint payload with ``weights_only=True`` when available."""
    resolved = Path(path).expanduser().resolve()
    try:
        return torch.load(resolved, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(resolved, map_location=map_location)


def extract_state_dict(payload: Any) -> Any:
    if isinstance(payload, dict):
        if isinstance(payload.get("state_dict"), dict):
            return payload["state_dict"]
        module_blob = payload.get("module")
        if isinstance(module_blob, dict):
            return module_blob.get("state_dict", module_blob)
        for key in ("model", "ema", "model_state_dict", "net", "generator"):
            blob = payload.get(key)
            if isinstance(blob, dict):
                return blob
    return payload


def extract_hparams(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("hyper_parameters"), dict):
        return dict(payload["hyper_parameters"])
    return {}


def build_lightning_module(
    module_cls: type[T],
    payload: Any,
    *,
    strict: bool = False,
    **overrides,
) -> T:
    """Instantiate a LightningModule from a safe-loaded checkpoint payload."""
    state_dict = extract_state_dict(payload)
    if not isinstance(state_dict, dict):
        raise RuntimeError("Checkpoint payload does not contain a valid state_dict")

    init_kwargs = extract_hparams(payload)
    init_kwargs.update(overrides)
    model = module_cls(**init_kwargs)
    model.load_state_dict(state_dict, strict=bool(strict))
    return model


def load_lightning_module(
    module_cls: type[T],
    checkpoint_path,
    *,
    map_location="cpu",
    strict: bool = False,
    **overrides,
) -> T:
    payload = load_torch_payload(checkpoint_path, map_location=map_location)
    return build_lightning_module(
        module_cls,
        payload,
        strict=strict,
        **overrides,
    )
