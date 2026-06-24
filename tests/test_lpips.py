import torch
import torch.nn as nn

from src.models import lpips as lpips_module


def _fake_vgg_features():
    return nn.Sequential(*(nn.Conv2d(3, 3, 1) for _ in range(30)))


def test_vgg16_uses_local_weights_without_torchvision_download(monkeypatch, tmp_path):
    calls = []

    class FakeVGG:
        def __init__(self):
            self.features = _fake_vgg_features()

    def fake_vgg16(*, weights):
        calls.append(weights)
        return FakeVGG()

    reference_features = _fake_vgg_features()
    checkpoint = {
        f"features.{key}": torch.full_like(value, 0.25)
        for key, value in reference_features.state_dict().items()
    }
    checkpoint_path = tmp_path / "vgg16_features.pth"
    torch.save(checkpoint, checkpoint_path)

    monkeypatch.setattr(lpips_module, "vgg16", fake_vgg16)

    vgg = lpips_module.VGG16(weights_path=checkpoint_path)

    assert calls == [None]
    assert vgg.weights_path == str(checkpoint_path.resolve())
    first_weight = vgg.slice1[0].weight
    assert torch.allclose(first_weight, torch.full_like(first_weight, 0.25))
