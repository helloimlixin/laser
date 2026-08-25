from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ROOT / "third_party" / "rq-vae-transformer"

import sys

if str(THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY))

from rqvae.models.rqvae.rqvae import RQVAE
from rqvae.optimizer.optimizer import create_optimizer
from rqvae.trainers.trainer import TrainerTemplate, resolve_training_precision
from src.models.dictionary_learner import DictionaryLearning


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "float32"),
        ("float32", "float32"),
        ("fp32", "float32"),
        ("bfloat16", "bfloat16"),
        ("bf16", "bfloat16"),
    ],
)
def test_explicit_precision_resolution(raw, expected):
    experiment = AttrDict(amp=False)
    if raw is not None:
        experiment.precision = raw
    assert resolve_training_precision(experiment) == expected


def test_legacy_amp_true_is_rejected_instead_of_silently_running_fp32():
    with pytest.raises(ValueError, match="precision=bfloat16 explicitly"):
        resolve_training_precision(AttrDict(amp=True))


def test_bfloat16_context_is_explicit_and_fp32_context_is_disabled():
    trainer = object.__new__(TrainerTemplate)
    trainer.device = torch.device("cpu")
    trainer.config = AttrDict(
        experiment=AttrDict(amp=False, precision="bfloat16")
    )
    with trainer.autocast_context():
        assert torch.is_autocast_enabled("cpu")
        assert torch.get_autocast_dtype("cpu") == torch.bfloat16

    trainer.config.experiment.precision = "float32"
    with trainer.autocast_context():
        assert not torch.is_autocast_enabled("cpu")


class RecordingQuantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dtype = None
        self.autocast_enabled = None
        self._last_bottleneck_objective_for_backward = None

    def forward(self, z_e):
        self.input_dtype = z_e.dtype
        self.autocast_enabled = torch.is_autocast_enabled(z_e.device.type)
        objective = z_e.square().mean()
        self._last_bottleneck_objective_for_backward = objective
        support = torch.zeros(
            z_e.size(0), z_e.size(2), z_e.size(3), 1, dtype=torch.long
        )
        return z_e, objective, SimpleNamespace(support=support)


def test_laser_quantizer_exits_bfloat16_autocast_for_fp32_omp():
    model = RQVAE.__new__(RQVAE)
    nn.Module.__init__(model)
    model.encoder = nn.Conv2d(2, 2, kernel_size=1)
    model.quant_conv = nn.Conv2d(2, 2, kernel_size=1)
    model.quantizer = RecordingQuantizer()
    model.post_quant_conv = nn.Conv2d(2, 2, kernel_size=1)
    model.decoder = nn.Identity()
    model.bottleneck_type = "laser"

    inputs = torch.randn(2, 2, 4, 4)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        reconstructed, objective, _ = model(inputs)

    assert model.quantizer.input_dtype == torch.float32
    assert model.quantizer.autocast_enabled is False
    assert objective.dtype == torch.float32
    assert reconstructed.dtype == torch.bfloat16


def test_bypass_decode_is_an_opt_in_diagnostic():
    model = RQVAE.__new__(RQVAE)
    nn.Module.__init__(model)
    model.encoder = nn.Conv2d(2, 2, kernel_size=1)
    model.quant_conv = nn.Conv2d(2, 2, kernel_size=1)
    model.quantizer = RecordingQuantizer()
    model.post_quant_conv = nn.Conv2d(2, 2, kernel_size=1)
    model.decoder = nn.Identity()
    model.bottleneck_type = "laser"

    inputs = torch.randn(2, 2, 4, 4)
    default_outputs = model(inputs)
    diagnostic_outputs = model(inputs, return_bypass=True)

    assert len(default_outputs) == 3
    assert len(diagnostic_outputs) == 4
    expected_bypass = model.decode(model.encode(inputs))
    assert torch.allclose(diagnostic_outputs[3], expected_bypass)


def test_bypass_training_path_still_observes_quantizer_but_decodes_encoder_latent():
    class DistortingQuantizer(RecordingQuantizer):
        def forward(self, z_e):
            _, objective, code = super().forward(z_e)
            return 2.0 * z_e, objective, code

    model = RQVAE.__new__(RQVAE)
    nn.Module.__init__(model)
    model.encoder = nn.Conv2d(2, 2, kernel_size=1)
    model.quant_conv = nn.Conv2d(2, 2, kernel_size=1)
    model.quantizer = DistortingQuantizer()
    model.post_quant_conv = nn.Conv2d(2, 2, kernel_size=1, bias=False)
    model.decoder = nn.Identity()
    model.bottleneck_type = "laser"

    inputs = torch.randn(2, 2, 4, 4)
    full = model(inputs)[0]
    bypass = model(inputs, use_bypass=True)[0]
    expected_bypass = model.decode(model.encode(inputs))

    assert model.quantizer.input_dtype == torch.float32
    assert torch.allclose(bypass, expected_bypass)
    assert not torch.allclose(full, bypass)
    with pytest.raises(ValueError, match="mutually exclusive"):
        model(inputs, return_bypass=True, use_bypass=True)


def test_alternating_dictionary_is_excluded_from_adam():
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(4, 4)
            self.quantizer = DictionaryLearning(
                num_embeddings=8,
                embedding_dim=4,
                sparsity_level=2,
                dictionary_update_mode="alternating_residual",
            )

    model = TinyModel()
    config = AttrDict(
        arch=AttrDict(
            type="rq-vae",
            hparams=AttrDict(
                bottleneck_type="laser",
                dictionary_update_mode="alternating_residual",
                dict_learning_rate=None,
            ),
        ),
        optimizer=AttrDict(
            type="adam",
            init_lr=4e-5,
            weight_decay=0.0,
            betas=(0.5, 0.9),
        ),
    )

    optimizer = create_optimizer(model, config)
    optimized_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    assert id(model.backbone.weight) in optimized_ids
    assert id(model.quantizer.dictionary) not in optimized_ids
    assert not model.quantizer.dictionary.requires_grad
