"""Exercise the actual LASER manual optimizer loop through Lightning."""
import lightning as pl
from lightning.pytorch.plugins.precision import MixedPrecision
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.laser import LASER


class LinearLossLASER(LASER):
    """Replace expensive networks/losses; retain optimization and AMP hooks."""
    def __init__(self, accumulation=2, clip=0., discriminator=False):
        pl.LightningModule.__init__(self)
        self.weight = nn.Parameter(torch.tensor(10.))
        self.discriminator = nn.Linear(1, 1, bias=False)
        nn.init.constant_(self.discriminator.weight, 10.)
        self.automatic_optimization = False
        self._adversarial_enabled = True
        self.audio_discriminator_first = False
        self.manual_accumulate_grad_batches = accumulation
        self.manual_gradient_clip_val = clip
        self.bypass_bottleneck = True
        self.register_buffer('_manual_train_step', torch.zeros((), dtype=torch.long))
        self._lr_base_lrs = self._disc_lr_base_lrs = ()
        self.disc_factor = float(discriminator)
        self.disc_start_step = 0
        self.adversarial_weight = 1.
        self.ae_step_gradients = []
        self.disc_step_gradients = []

    def on_fit_start(self): pass
    def on_train_start(self): pass
    def on_fit_end(self): pass
    def _apply_scheduled_lrs(self, *args, **kwargs): pass
    def _should_log_images(self, *args, **kwargs): return False
    def _should_compute_train_audio_spectral_metrics(self, *args): return False
    def _adversarial_schedule_factor_at_step(self, *args): return 1.
    def _disc_update_selected(self, step): return True

    def compute_metrics(self, batch, **kwargs):
        x = batch[0]
        return self.weight * x.mean(), x, x, x, x

    def _discriminator_loss(self, real, fake):
        return self.discriminator.weight.sum() * real.mean(), real.new_zeros(()), real.new_zeros(())

    def configure_optimizers(self):
        ae = torch.optim.SGD([self.weight], lr=1.)
        disc = torch.optim.SGD(self.discriminator.parameters(), lr=1.)
        ae.register_step_pre_hook(lambda *_: self.ae_step_gradients.append(float(self.weight.grad)))
        disc.register_step_pre_hook(lambda *_: self.disc_step_gradients.append(float(self.discriminator.weight.grad)))
        return [ae, disc]


def fit(model, data, *, amp=False):
    plugins = [MixedPrecision('16-mixed', device='cpu',
               scaler=torch.amp.GradScaler('cpu', init_scale=128., growth_interval=1))] if amp else None
    trainer = pl.Trainer(max_epochs=1, accelerator='cpu', devices=1,
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, num_sanity_val_steps=0, limit_val_batches=0,
        plugins=plugins)
    trainer.fit(model, train_dataloaders=DataLoader(TensorDataset(data[:, None]), batch_size=2))
    return trainer


@pytest.mark.parametrize('count,expected', [(3, [2.]), (5, [2.5, 5.])])
def test_manual_accumulation_weights_examples_and_flushes_partial_windows(count, expected):
    model = LinearLossLASER()
    fit(model, torch.arange(1, count + 1, dtype=torch.float32))
    assert model.ae_step_gradients == pytest.approx(expected)
    assert float(model.weight.detach()) == pytest.approx(10. - sum(expected))
    assert int(model._manual_train_step) == len(expected)


def test_manual_amp_clips_unscaled_gradients_for_both_optimizers():
    model = LinearLossLASER(accumulation=1, clip=.5, discriminator=True)
    fit(model, torch.tensor([2., 2.]), amp=True)
    assert model.ae_step_gradients == pytest.approx([.5], abs=1e-6)
    assert model.disc_step_gradients == pytest.approx([.5], abs=1e-6)
    assert float(model.weight.detach()) == pytest.approx(9.5, abs=1e-6)
    assert float(model.discriminator.weight.detach()) == pytest.approx(9.5, abs=1e-6)


def test_manual_amp_skip_does_not_advance_generator_step_or_corrupt_critic_scale():
    model = LinearLossLASER(accumulation=1, discriminator=True)
    model.weight.register_hook(lambda gradient: gradient * float('inf'))
    fit(model, torch.tensor([2., 2.]), amp=True)
    assert model.ae_step_gradients == []
    assert int(model._manual_train_step) == 0
    assert float(model.weight.detach()) == 10.
    assert model.disc_step_gradients == pytest.approx([2.])
    assert float(model.discriminator.weight.detach()) == pytest.approx(8.)
