import pytest
import torch

from src.models.discriminator import (
    NLayerDiscriminator,
    adopt_weight,
    hinge_d_loss,
    hinge_g_loss,
    vanilla_d_loss,
)


@pytest.mark.parametrize("norm", ["batch", "group", "none"])
def test_discriminator_outputs_patch_logits(norm):
    torch.manual_seed(0)
    disc = NLayerDiscriminator(in_channels=3, num_filters=32, num_layers=3, norm=norm)
    x = torch.randn(2, 3, 64, 64)
    logits = disc(x)
    # PatchGAN returns a single-channel spatial logit map, not a scalar.
    assert logits.dim() == 4
    assert logits.shape[0] == 2 and logits.shape[1] == 1
    assert logits.shape[-1] > 1 and logits.shape[-2] > 1


def test_discriminator_spectral_norm_smoke():
    torch.manual_seed(0)
    disc = NLayerDiscriminator(num_filters=32, num_layers=3, norm="none", spectral=True)
    logits = disc(torch.randn(2, 3, 64, 64))
    logits.mean().backward()
    assert any(p.grad is not None for p in disc.parameters())


def test_hinge_losses_reward_correct_separation():
    # Confident-correct critic (real >> 1, fake << -1) should incur ~zero hinge D-loss.
    real = torch.full((4, 1, 8, 8), 3.0)
    fake = torch.full((4, 1, 8, 8), -3.0)
    assert hinge_d_loss(real, fake).item() == pytest.approx(0.0, abs=1e-6)
    # Swapped (critic fooled) should be strictly worse.
    assert hinge_d_loss(fake, real).item() > 1.0
    # Generator wants high fake logits => lower generator loss.
    assert hinge_g_loss(torch.full((4, 1, 8, 8), 2.0)).item() < hinge_g_loss(
        torch.full((4, 1, 8, 8), -2.0)
    ).item()


def test_vanilla_d_loss_finite_and_ordered():
    real = torch.randn(2, 1, 8, 8)
    fake = torch.randn(2, 1, 8, 8)
    loss = vanilla_d_loss(real, fake)
    assert torch.isfinite(loss)
    good = vanilla_d_loss(torch.full((2, 1, 4, 4), 5.0), torch.full((2, 1, 4, 4), -5.0))
    bad = vanilla_d_loss(torch.full((2, 1, 4, 4), -5.0), torch.full((2, 1, 4, 4), 5.0))
    assert good.item() < bad.item()


def test_adopt_weight_warmup_gate():
    assert adopt_weight(0.8, global_step=10, threshold=100) == 0.0
    assert adopt_weight(0.8, global_step=100, threshold=100) == 0.8
    assert adopt_weight(0.8, global_step=250, threshold=100) == 0.8


def test_discriminator_gradients_flow_to_generator_input():
    # The adversarial path must backprop into the (decoder) input image.
    torch.manual_seed(0)
    disc = NLayerDiscriminator(num_filters=32, num_layers=3, norm="group")
    fake = torch.randn(2, 3, 64, 64, requires_grad=True)
    g_loss = hinge_g_loss(disc(fake))
    g_loss.backward()
    assert fake.grad is not None and torch.isfinite(fake.grad).all()
