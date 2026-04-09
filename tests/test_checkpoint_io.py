from __future__ import annotations

from pathlib import Path

import torch

from src.checkpoint_io import (
    build_lightning_module,
    extract_hparams,
    extract_state_dict,
    load_lightning_module,
    load_torch_payload,
)


class _ToyModule(torch.nn.Module):
    def __init__(self, width: int, scale: float = 1.0, enabled: bool = True):
        super().__init__()
        self.width = int(width)
        self.scale = float(scale)
        self.enabled = bool(enabled)
        self.linear = torch.nn.Linear(self.width, self.width, bias=False)


def _toy_payload(*, width: int = 3, scale: float = 2.0, enabled: bool = True) -> tuple[dict, _ToyModule]:
    model = _ToyModule(width=width, scale=scale, enabled=enabled)
    with torch.no_grad():
        model.linear.weight.copy_(torch.arange(width * width, dtype=torch.float32).view(width, width))
    payload = {
        "state_dict": model.state_dict(),
        "hyper_parameters": {
            "width": width,
            "scale": scale,
            "enabled": enabled,
        },
    }
    return payload, model


def test_extract_state_dict_prefers_lightning_key():
    payload, model = _toy_payload()
    state_dict = extract_state_dict(payload)
    assert state_dict.keys() == model.state_dict().keys()
    for key, value in state_dict.items():
        assert torch.equal(value, model.state_dict()[key])


def test_extract_hparams_reads_checkpoint_dict():
    payload, _ = _toy_payload(width=5, scale=3.5, enabled=False)
    assert extract_hparams(payload) == {"width": 5, "scale": 3.5, "enabled": False}


def test_build_lightning_module_restores_state_and_overrides():
    payload, ref = _toy_payload(width=4, scale=1.5, enabled=True)
    restored = build_lightning_module(_ToyModule, payload, strict=True, enabled=False)
    assert isinstance(restored, _ToyModule)
    assert restored.width == 4
    assert restored.scale == 1.5
    assert restored.enabled is False
    assert torch.equal(restored.linear.weight, ref.linear.weight)


def test_load_lightning_module_restores_safe_payload(tmp_path: Path):
    payload, ref = _toy_payload(width=2, scale=0.75, enabled=True)
    ckpt = tmp_path / "toy.ckpt"
    torch.save(payload, ckpt)

    raw = load_torch_payload(ckpt, map_location="cpu")
    assert extract_hparams(raw)["width"] == 2

    restored = load_lightning_module(_ToyModule, ckpt, map_location="cpu", strict=True)
    assert restored.width == 2
    assert restored.scale == 0.75
    assert restored.enabled is True
    assert torch.equal(restored.linear.weight, ref.linear.weight)
