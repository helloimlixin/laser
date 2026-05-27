"""PatchGAN discriminator and adversarial losses for LASER stage-1 training.

The reconstruction-only objective (MSE + L1 + edge + LPIPS) used by the LASER
autoencoder produces over-smoothed images: it has no term that rewards
high-frequency texture, so the decoder regresses toward the conditional mean.
A PatchGAN discriminator with a hinge adversarial loss is the standard remedy
(VQGAN / taming-transformers); it pushes the decoder to synthesize plausible
fine detail without changing the latent/token budget at all.

This module is intentionally self-contained and has no LASER dependency, so it
can be unit-tested in isolation and stays a no-op until ``adversarial_weight``
is set in the model config.
"""

from __future__ import annotations

import functools

import torch
import torch.nn as nn


def weights_init(module: nn.Module) -> None:
    """taming-transformers weight init: N(0, 0.02) convs, BN gamma ~ N(1, 0.02)."""
    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator (Pix2Pix / VQGAN style).

    Outputs a map of real/fake logits, one per overlapping receptive-field
    patch, rather than a single image-level score. This keeps the adversarial
    signal local, which is what encourages crisp per-region texture.

    Args:
        in_channels: input image channels (3 for RGB).
        num_filters: base channel width (``ndf``); doubles each layer up to 8x.
        num_layers: number of stride-2 downsampling conv blocks.
        norm: ``"batch"`` (taming default), ``"group"`` (DDP/small-batch safe),
            or ``"none"``. ``norm="none"`` pairs well with ``spectral=True``.
        spectral: wrap convs in spectral normalization for a more stable critic.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_filters: int = 64,
        num_layers: int = 3,
        norm: str = "batch",
        spectral: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_layers = int(num_layers)

        norm = (norm or "none").lower()
        if norm not in {"batch", "group", "none"}:
            raise ValueError(f"norm must be one of batch|group|none, got {norm!r}")

        def make_norm(num_features: int) -> nn.Module:
            if norm == "batch":
                return nn.BatchNorm2d(num_features)
            if norm == "group":
                groups = min(32, num_features)
                while num_features % groups != 0:
                    groups -= 1
                return nn.GroupNorm(groups, num_features)
            return nn.Identity()

        # Affine bias is redundant in front of an affine norm layer.
        use_bias = norm in {"none", "group"}

        def conv(in_ch: int, out_ch: int, stride: int) -> nn.Module:
            layer = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=use_bias)
            return nn.utils.spectral_norm(layer) if spectral else layer

        layers: list[nn.Module] = [
            conv(self.in_channels, num_filters, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        ch_mult = 1
        prev_mult = 1
        for n in range(1, self.num_layers):
            prev_mult = ch_mult
            ch_mult = min(2 ** n, 8)
            layers += [
                conv(num_filters * prev_mult, num_filters * ch_mult, stride=2),
                make_norm(num_filters * ch_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        # Final stride-1 block widens the receptive field without downsampling.
        prev_mult = ch_mult
        ch_mult = min(2 ** self.num_layers, 8)
        layers += [
            conv(num_filters * prev_mult, num_filters * ch_mult, stride=1),
            make_norm(num_filters * ch_mult),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Project to a single-channel patch logit map.
        layers += [conv(num_filters * ch_mult, 1, stride=1)]

        self.main = nn.Sequential(*layers)
        if not spectral:
            self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Hinge discriminator loss: push real logits > 1, fake logits < -1."""
    loss_real = torch.mean(torch.relu(1.0 - logits_real))
    loss_fake = torch.mean(torch.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)


def vanilla_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Non-saturating BCE discriminator loss (softplus form)."""
    loss_real = torch.mean(nn.functional.softplus(-logits_real))
    loss_fake = torch.mean(nn.functional.softplus(logits_fake))
    return 0.5 * (loss_real + loss_fake)


def hinge_g_loss(logits_fake: torch.Tensor) -> torch.Tensor:
    """Generator adversarial loss (both hinge and vanilla use -E[logits_fake])."""
    return -torch.mean(logits_fake)


def adopt_weight(weight: float, global_step: int, threshold: int = 0, value: float = 0.0) -> float:
    """Gate ``weight`` to ``value`` until ``global_step`` reaches ``threshold``.

    Used to warm up the autoencoder before the discriminator starts, so the
    critic does not destabilize an untrained decoder.
    """
    if global_step < threshold:
        return value
    return weight


__all__ = [
    "NLayerDiscriminator",
    "weights_init",
    "hinge_d_loss",
    "vanilla_d_loss",
    "hinge_g_loss",
    "adopt_weight",
]
