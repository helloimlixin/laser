import random
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ROOT / "third_party" / "rq-vae-transformer"

import sys

if str(THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY))

from rqvae.trainers.trainer_rqvae import Trainer
from rqvae.utils.checkpoint import rank_resume_state, validate_resume_checkpoint


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class RandomizedToyDataset(Dataset):
    """Worker-side randomness stands in for ImageNet random crops/flips."""

    def __init__(self, length=10):
        self.values = torch.arange(length * 4, dtype=torch.float32).reshape(length, 4) / 40

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        jitter = (
            torch.rand(()) * 0.01
            + float(np.random.rand()) * 0.01
            + random.random() * 0.01
        )
        return self.values[index] + jitter, 0


class ToyQuantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.dictionary = nn.Parameter(torch.tensor([0.8, 1.0, 1.2, 1.4]))
        self.register_buffer("_revival_step", torch.zeros((), dtype=torch.long))
        self.register_buffer("_revival_noise", torch.zeros(4))

    def project_dictionary_gradient_(self):
        return None

    def normalize_dictionary_(self):
        return None

    def revive_dead_atoms_after_step_(self, optimizer):
        self._revival_step.add_(1)
        if int(self._revival_step) % 2 == 0:
            self._revival_noise.copy_(torch.rand_like(self._revival_noise))


class ToyRQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.quantizer = ToyQuantizer()
        self.decoder = nn.Linear(4, 4)

    def forward(self, xs):
        # Main-process RNG exercises the rank-local torch RNG checkpoint.
        hidden = torch.tanh(self.encoder(xs + torch.rand_like(xs) * 0.01))
        quantized = hidden * self.quantizer.dictionary
        reconstructed = self.decoder(quantized)
        quant_loss = (hidden - quantized).square().mean()
        codes = quantized.detach().argmax(dim=1).reshape(-1, 1, 1, 1)
        return reconstructed, quant_loss, codes

    def compute_loss(self, reconstructed, quant_loss, codes, *, xs, valid=False):
        loss_recon = (reconstructed - xs).square().mean()
        zero = loss_recon.detach().new_zeros(())
        return {
            "loss_total": loss_recon + 0.25 * quant_loss,
            "loss_recon": loss_recon,
            "loss_latent": quant_loss,
            "loss_dictionary": quant_loss,
            "loss_commitment": quant_loss.detach(),
            "latent_input_energy": xs.detach().square().mean(),
            "bottleneck_explained_variance": 1.0 - quant_loss.detach(),
            "atom_window_active_fraction": loss_recon.detach().new_ones(()),
            "revived_atom_count": zero,
            "codes": [codes],
        }


class Wrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class PerceptualLoss(nn.Module):
    def forward(self, xs, reconstructed):
        return (xs - reconstructed).abs().mean()


def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _toy_gan_loss(self, inputs, recons, mode="idle"):
    if mode == "gen":
        logits_fake = self.discriminator(recons)
        return -logits_fake.mean(), recons.new_zeros(()), {}
    if mode == "disc":
        logits_real = self.discriminator(inputs.detach())
        logits_fake = self.discriminator(recons.detach())
        loss = (1.0 - logits_real).square().mean() + logits_fake.square().mean()
        return recons.new_zeros(()), loss, {
            "logits_real": logits_real.detach().mean(),
            "logits_fake": logits_fake.detach().mean(),
        }
    return recons.new_zeros(()), recons.new_zeros(()), {}


def _make_training_state(result_path):
    model = Wrapper(ToyRQVAE())
    discriminator = Wrapper(nn.Linear(4, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1.0e-3)
    disc_scheduler = torch.optim.lr_scheduler.StepLR(
        disc_optimizer, step_size=2, gamma=0.9
    )

    trainer = object.__new__(Trainer)
    trainer.model = model
    trainer.model_ema = None
    trainer.discriminator = discriminator
    trainer.disc_optimizer = disc_optimizer
    trainer.disc_scheduler = disc_scheduler
    trainer.loader_trn = DataLoader(
        RandomizedToyDataset(),
        batch_size=2,
        shuffle=False,
        num_workers=2,
        prefetch_factor=2,
    )
    trainer.device = torch.device("cpu")
    trainer.distenv = SimpleNamespace(master=True, world_size=1, world_rank=0)
    trainer.writer = None
    trainer.n_codebook = 1
    trainer.gan_start_epoch = 0
    trainer.perceptual_loss = PerceptualLoss()
    trainer.perceptual_weight = 0.2
    trainer.disc_weight = 0.3
    trainer.get_last_layer = lambda: trainer.model.module.decoder.weight
    trainer.gan_loss = MethodType(_toy_gan_loss, trainer)
    trainer.resume_signature = {"version": 1, "sha256": "trajectory-test"}
    trainer.lineage_exact = True
    trainer.lineage_origin = "test"
    trainer._last_checkpoint_id = None
    trainer.config = AttrDict(
        result_path=str(result_path),
        experiment=AttrDict(
            recovery_ckpt_freq_steps=3,
            rfid_backend="original-rqvae",
        ),
        arch=AttrDict(
            code_hier=1,
            hparams=AttrDict(n_embed=8, use_padding_idx=False),
        ),
    )
    return trainer, optimizer, scheduler


def _clone_state_dict(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _assert_nested_equal(expected, actual):
    if torch.is_tensor(expected):
        assert torch.equal(expected, actual)
    elif isinstance(expected, np.ndarray):
        assert np.array_equal(expected, actual)
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            _assert_nested_equal(expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual):
            _assert_nested_equal(left, right)
    else:
        assert expected == actual


def test_uninterrupted_and_mid_epoch_resume_have_identical_trajectory(tmp_path):
    continuous_path = tmp_path / "continuous"
    continuous_path.mkdir()
    _seed_all(431)
    continuous, optimizer, scheduler = _make_training_state(continuous_path)
    continuous_summary = continuous.train(optimizer, scheduler, epoch=0)

    checkpoint = torch.load(
        continuous_path / "last_model.pt", map_location="cpu", weights_only=False
    )
    metadata = validate_resume_checkpoint(
        checkpoint,
        steps_per_epoch=len(continuous.loader_trn),
        world_size=1,
        expected_resume_signature=continuous.resume_signature,
    )
    assert metadata["batch_idx"] == 3
    assert checkpoint["train_accumulator_state_by_rank"][0]["counter"] == 3

    expected_model = _clone_state_dict(continuous.model.module)
    expected_discriminator = _clone_state_dict(continuous.discriminator.module)
    expected_optimizer = optimizer.state_dict()
    expected_scheduler = scheduler.state_dict()
    expected_disc_optimizer = continuous.disc_optimizer.state_dict()
    expected_disc_scheduler = continuous.disc_scheduler.state_dict()
    expected_rng = continuous._capture_rng_state()

    resumed_path = tmp_path / "resumed"
    resumed_path.mkdir()
    _seed_all(9999)
    resumed, resumed_optimizer, resumed_scheduler = _make_training_state(resumed_path)
    resumed.model.module.load_state_dict(checkpoint["state_dict"])
    resumed.discriminator.module.load_state_dict(checkpoint["discriminator"])
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    resumed_scheduler.load_state_dict(checkpoint["scheduler"])
    resumed.disc_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])
    resumed.disc_scheduler.load_state_dict(checkpoint["discriminator_scheduler"])

    resumed_summary = resumed.train(
        resumed_optimizer,
        resumed_scheduler,
        epoch=metadata["epoch"],
        start_batch_idx=metadata["batch_idx"],
        resume_rng_state=rank_resume_state(checkpoint, "rng_state_by_rank", 0),
        resume_epoch_start_rng_state=rank_resume_state(
            checkpoint, "epoch_start_rng_state_by_rank", 0
        ),
        resume_accumulator_state=rank_resume_state(
            checkpoint, "train_accumulator_state_by_rank", 0
        ),
    )

    _assert_nested_equal(expected_model, resumed.model.module.state_dict())
    _assert_nested_equal(expected_discriminator, resumed.discriminator.module.state_dict())
    _assert_nested_equal(expected_optimizer, resumed_optimizer.state_dict())
    _assert_nested_equal(expected_scheduler, resumed_scheduler.state_dict())
    _assert_nested_equal(expected_disc_optimizer, resumed.disc_optimizer.state_dict())
    _assert_nested_equal(expected_disc_scheduler, resumed.disc_scheduler.state_dict())
    _assert_nested_equal(expected_rng, resumed._capture_rng_state())
    _assert_nested_equal(continuous_summary.metrics, resumed_summary.metrics)
    _assert_nested_equal(
        continuous_summary.ent_codes_wo_pad, resumed_summary.ent_codes_wo_pad
    )
