import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import vqvae as vqvae_module
from src.models.bottleneck import VectorQuantizerEMA


def _build_model(**overrides):
    params = {
        "in_channels": 3,
        "num_hiddens": 16,
        "num_embeddings": 8,
        "embedding_dim": 4,
        "num_residual_blocks": 1,
        "num_residual_hiddens": 8,
        "commitment_cost": 0.25,
        "decay": 0.0,
        "perceptual_weight": 0.0,
        "learning_rate": 1e-3,
        "beta": 0.9,
        "compute_fid": False,
    }
    params.update(overrides)
    model = vqvae_module.VQVAE(**params)
    model.log = lambda *args, **kwargs: None
    return model


def test_train_psnr_is_logged_to_progress_bar():
    model = _build_model()
    batch = torch.randn(4, 3, 16, 16)
    log_calls = []

    def _record_log(name, value, **kwargs):
        log_calls.append((name, kwargs))

    model.log = _record_log

    metrics = model.compute_metrics(batch, prefix="train")

    assert torch.isfinite(metrics["loss"])
    assert any(
        name == "train/psnr" and kwargs.get("prog_bar") is True
        for name, kwargs in log_calls
    )


def test_vqvae_uses_ema_quantizer():
    model = _build_model(decay=0.99)

    assert isinstance(model.vector_quantizer, VectorQuantizerEMA)
    assert model.vector_quantizer.embedding.weight.requires_grad is False
    assert model.decay == 0.99


def test_val_psnr_and_ssim_are_logged_to_progress_bar():
    model = _build_model()
    batch = torch.randn(4, 3, 16, 16)
    log_calls = []

    def _record_log(name, value, **kwargs):
        log_calls.append((name, kwargs))

    model.log = _record_log

    metrics = model.compute_metrics(batch, prefix="val")

    assert torch.isfinite(metrics["loss"])
    assert metrics["ssim"] is not None
    assert any(
        name == "val/psnr" and kwargs.get("prog_bar") is True
        for name, kwargs in log_calls
    )
    assert any(
        name == "val/ssim" and kwargs.get("prog_bar") is True
        for name, kwargs in log_calls
    )
