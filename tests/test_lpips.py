import torch
import torch.nn as nn

from src.models import lpips as lpips_module


def _fake_vgg_features():
    return nn.Sequential(*(nn.Conv2d(3, 3, 1) for _ in range(30)))


def test_lpips_prefers_rqvae_cache_path(monkeypatch, tmp_path):
    rqvae_root = tmp_path / "rqvae" / "losses" / "vqgan" / ".caches"
    fallback_root = tmp_path / "vgg_lpips"
    rqvae_root.mkdir(parents=True)
    fallback_root.mkdir()
    rqvae_ckpt = rqvae_root / "vgg.pth"
    fallback_ckpt = fallback_root / "vgg.pth"
    rqvae_ckpt.write_bytes(b"rqvae")
    fallback_ckpt.write_bytes(b"fallback")

    monkeypatch.setattr(lpips_module, "RQVAE_CKPT_ROOT", rqvae_root)
    monkeypatch.setattr(lpips_module, "BUNDLED_CKPT_ROOT", fallback_root)

    assert lpips_module.get_ckpt_path("vgg_lpips") == str(rqvae_ckpt)


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


def test_lpips_stays_in_eval_when_parent_enters_train_mode():
    # Avoid loading VGG weights: this regression concerns nn.Module mode
    # propagation and can be tested on a minimally initialized LPIPS instance.
    lpips = lpips_module.LPIPS.__new__(lpips_module.LPIPS)
    nn.Module.__init__(lpips)
    lpips.dropout_probe = nn.Dropout()

    parent = nn.Sequential(lpips)
    parent.train()

    assert not lpips.training
    assert not lpips.dropout_probe.training


def test_lpips_v00_bypasses_only_internal_scaling():
    lpips = lpips_module.LPIPS.__new__(lpips_module.LPIPS)
    nn.Module.__init__(lpips)
    lpips.version = "0.0"
    lpips.scaling_layer = lpips_module.ScalingLayer()
    x = torch.randn(2, 3, 4, 4)

    assert lpips._prepare_input(x) is x

    lpips.version = "0.1"
    assert not torch.equal(lpips._prepare_input(x), x)
