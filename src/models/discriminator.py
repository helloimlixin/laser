"""PatchGAN discriminator and adversarial losses for LASER stage-1 training.

The reconstruction-only objective (MSE + L1 + edge + LPIPS) used by the LASER
autoencoder produces over-smoothed images: it has no term that rewards
high-frequency texture, so the decoder regresses toward the conditional mean.
A PatchGAN discriminator with a hinge adversarial loss is the standard remedy
(taming-transformers style); it pushes the decoder to synthesize plausible
fine detail without changing the latent/token budget at all.

This module is intentionally self-contained and has no LASER dependency, so it
can be unit-tested in isolation and stays a no-op until ``adversarial_weight``
is set in the model config.
"""

from __future__ import annotations

import functools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


def weights_init(module: nn.Module) -> None:
    """taming-transformers weight init: N(0, 0.02) convs, BN gamma ~ N(1, 0.02)."""
    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator (Pix2Pix / taming-transformers style).

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


def _conv1d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    *,
    stride: int = 1,
    padding: int = 0,
    groups: int = 1,
    spectral: bool = False,
) -> nn.Module:
    layer = nn.Conv1d(
        int(in_channels),
        int(out_channels),
        kernel_size=int(kernel_size),
        stride=int(stride),
        padding=int(padding),
        groups=int(groups),
    )
    return nn.utils.spectral_norm(layer) if spectral else layer


def _conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size,
    *,
    stride=(1, 1),
    padding=(0, 0),
    spectral: bool = False,
) -> nn.Module:
    layer = nn.Conv2d(
        int(in_channels),
        int(out_channels),
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )
    return nn.utils.spectral_norm(layer) if spectral else layer


class AudioScaleDiscriminator(nn.Module):
    """1D multi-scale waveform discriminator in the MelGAN/HiFi-GAN family."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        num_filters: int = 32,
        max_filters: int = 512,
        num_layers: int = 4,
        spectral: bool = False,
    ) -> None:
        super().__init__()
        num_layers = max(1, int(num_layers))
        num_filters = max(1, int(num_filters))
        max_filters = max(num_filters, int(max_filters))

        layers: list[nn.Module] = [
            _conv1d(in_channels, num_filters, 15, padding=7, spectral=spectral),
        ]
        channels = num_filters
        for idx in range(num_layers):
            next_channels = min(max_filters, channels * 2)
            groups = min(4, channels)
            while channels % groups != 0 or next_channels % groups != 0:
                groups -= 1
            layers.append(
                _conv1d(
                    channels,
                    next_channels,
                    41,
                    stride=4,
                    padding=20,
                    groups=max(groups, 1),
                    spectral=spectral,
                )
            )
            channels = next_channels
        layers += [
            _conv1d(channels, channels, 5, padding=2, spectral=spectral),
            _conv1d(channels, 1, 3, padding=1, spectral=spectral),
        ]
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feats: list[torch.Tensor] = []
        for layer in self.layers[:-1]:
            x = F.leaky_relu(layer(x), 0.2)
            if return_features:
                feats.append(x)
        logit = self.layers[-1](x)
        if return_features:
            return logit, feats
        return logit


class AudioPeriodDiscriminator(nn.Module):
    """Period-folded waveform discriminator from the HiFi-GAN discriminator stack."""

    def __init__(
        self,
        *,
        period: int,
        in_channels: int = 1,
        num_filters: int = 32,
        max_filters: int = 512,
        num_layers: int = 4,
        spectral: bool = False,
    ) -> None:
        super().__init__()
        self.period = max(1, int(period))
        num_layers = max(1, int(num_layers))
        num_filters = max(1, int(num_filters))
        max_filters = max(num_filters, int(max_filters))

        layers: list[nn.Module] = []
        channels = int(in_channels)
        out_channels = num_filters
        for _ in range(num_layers):
            layers.append(
                _conv2d(
                    channels,
                    out_channels,
                    kernel_size=(5, 1),
                    stride=(3, 1),
                    padding=(2, 0),
                    spectral=spectral,
                )
            )
            channels = out_channels
            out_channels = min(max_filters, out_channels * 2)
        layers += [
            _conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0), spectral=spectral),
            _conv2d(channels, 1, kernel_size=(3, 1), padding=(1, 0), spectral=spectral),
        ]
        self.layers = nn.ModuleList(layers)

    def _fold_period(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected waveform [B, C, T], got {tuple(x.shape)}")
        time = int(x.size(-1))
        pad = (self.period - (time % self.period)) % self.period
        if pad:
            x = F.pad(x, (0, pad), mode="reflect")
        return x.view(x.size(0), x.size(1), x.size(-1) // self.period, self.period)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x = self._fold_period(x)
        feats: list[torch.Tensor] = []
        for layer in self.layers[:-1]:
            x = F.leaky_relu(layer(x), 0.2)
            if return_features:
                feats.append(x)
        logit = self.layers[-1](x)
        if return_features:
            return logit, feats
        return logit


class AudioSTFTDiscriminator(nn.Module):
    """Complex-STFT discriminator (Encodec/DAC family) for one FFT resolution.

    Operates on the stacked real/imaginary STFT of the waveform with a small 2D
    conv stack. Complements the time-domain MPD/MSD critics with a frequency-domain
    view that is well suited to neural-codec reconstruction.
    """

    def __init__(
        self,
        *,
        n_fft: int,
        hop_length: int | None = None,
        in_channels: int = 1,
        num_filters: int = 32,
        max_filters: int = 512,
        num_layers: int = 4,
        spectral: bool = False,
    ) -> None:
        super().__init__()
        if int(in_channels) != 1:
            raise ValueError("AudioSTFTDiscriminator expects mono waveform input")
        self.n_fft = max(8, int(n_fft))
        self.hop_length = int(hop_length or max(1, self.n_fft // 4))
        self.win_length = int(self.n_fft)
        num_layers = max(1, int(num_layers))
        num_filters = max(1, int(num_filters))
        max_filters = max(num_filters, int(max_filters))
        # Two input channels: real and imaginary parts of the complex STFT.
        layers: list[nn.Module] = [
            _conv2d(2, num_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), spectral=spectral),
        ]
        channels = num_filters
        for _ in range(num_layers):
            next_channels = min(max_filters, channels * 2)
            layers.append(
                _conv2d(channels, next_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), spectral=spectral)
            )
            channels = next_channels
        layers += [
            _conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1), spectral=spectral),
            _conv2d(channels, 1, kernel_size=(3, 3), padding=(1, 1), spectral=spectral),
        ]
        self.layers = nn.ModuleList(layers)

    def _complex_stft(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            if int(x.size(1)) != 1:
                raise ValueError(f"expected mono waveform [B, 1, T], got {tuple(x.shape)}")
            x = x[:, 0, :]
        elif x.ndim != 2:
            raise ValueError(f"expected waveform [B, T] or [B, 1, T], got {tuple(x.shape)}")
        # STFT is computed in fp32 (no half/bf16 kernel) and is differentiable;
        # constant padding keeps the 1D backward deterministic (matches the loss).
        x32 = x.float()
        window = torch.hann_window(self.win_length, periodic=True, dtype=x32.dtype, device=x32.device)
        spec = torch.stft(
            x32,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        return torch.stack([spec.real, spec.imag], dim=1)  # [B, 2, F, T]

    def forward(self, x: torch.Tensor, return_features: bool = False):
        # fp32 STFT input; under bf16-mixed autocast the convs downcast as needed.
        h = self._complex_stft(x)
        feats: list[torch.Tensor] = []
        for layer in self.layers[:-1]:
            h = F.leaky_relu(layer(h), 0.2)
            if return_features:
                feats.append(h)
        logit = self.layers[-1](h)
        if return_features:
            return logit, feats
        return logit


class AudioMultiScalePeriodDiscriminator(nn.Module):
    """Compact HiFi-GAN-style discriminator for raw waveform reconstruction."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        num_filters: int = 32,
        max_filters: int = 512,
        num_layers: int = 4,
        periods=(2, 3, 5, 7, 11),
        num_scales: int = 3,
        stft_fft_sizes=(),
        spectral: bool = False,
    ) -> None:
        super().__init__()
        self.period_discriminators = nn.ModuleList(
            [
                AudioPeriodDiscriminator(
                    period=period,
                    in_channels=in_channels,
                    num_filters=num_filters,
                    max_filters=max_filters,
                    num_layers=num_layers,
                    spectral=spectral,
                )
                for period in periods
            ]
        )
        self.scale_discriminators = nn.ModuleList(
            [
                AudioScaleDiscriminator(
                    in_channels=in_channels,
                    num_filters=num_filters,
                    max_filters=max_filters,
                    num_layers=num_layers,
                    spectral=spectral,
                )
                for _ in range(max(0, int(num_scales)))
            ]
        )
        # Optional DAC/Encodec-style complex-STFT critics (frequency-domain view).
        self.stft_discriminators = nn.ModuleList(
            [
                AudioSTFTDiscriminator(
                    n_fft=int(n_fft),
                    in_channels=in_channels,
                    num_filters=num_filters,
                    max_filters=max_filters,
                    num_layers=num_layers,
                    spectral=spectral,
                )
                for n_fft in stft_fft_sizes
                if int(n_fft) > 0
            ]
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        logits: list[torch.Tensor] = []
        feats: list[list[torch.Tensor]] = []

        def _run(disc, inp):
            if return_features:
                logit, fmap = disc(inp, return_features=True)
                logits.append(logit)
                feats.append(fmap)
            else:
                logits.append(disc(inp))

        for disc in self.period_discriminators:
            _run(disc, x)
        scaled = x
        for idx, disc in enumerate(self.scale_discriminators):
            if idx > 0:
                scaled = F.avg_pool1d(scaled, kernel_size=4, stride=2, padding=1)
            _run(disc, scaled)
        for disc in self.stft_discriminators:
            _run(disc, x)

        if return_features:
            return logits, feats
        return logits


class AudioEncodecMultiScaleSTFTDiscriminator(nn.Module):
    """Adapter around Meta EnCodec's published MS-STFT discriminator.

    The upstream module always returns logits and feature maps together.
    LASER makes feature collection optional, so this adapter preserves the
    exact upstream STFT/convolution stack behind the local critic interface.
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        num_filters: int = 32,
        max_filters: int = 1024,
        fft_sizes=(2048, 1024, 512, 256, 128),
    ) -> None:
        super().__init__()
        if int(in_channels) != 1:
            raise ValueError("Meta EnCodec's MS-STFT discriminator expects mono audio")
        try:
            from encodec.msstftd import MultiScaleSTFTDiscriminator
        except ImportError as exc:  # pragma: no cover - project pins encodec
            raise RuntimeError(
                "audio_adversarial_type='encodec_msstft' requires encodec==0.1.1"
            ) from exc

        n_ffts = tuple(int(value) for value in fft_sizes if int(value) > 0)
        if not n_ffts:
            raise ValueError("Meta EnCodec's MS-STFT discriminator needs at least one FFT scale")
        self.model = MultiScaleSTFTDiscriminator(
            filters=int(num_filters),
            in_channels=int(in_channels),
            out_channels=1,
            n_ffts=list(n_ffts),
            hop_lengths=[max(1, value // 4) for value in n_ffts],
            win_lengths=list(n_ffts),
            max_filters=int(max_filters),
            normalized=True,
            norm="weight_norm",
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        # The upstream Spectrogram transform requires fp32 under mixed precision.
        logits, features = self.model(x.float())
        if return_features:
            return logits, features
        return logits


def _mdct_analysis_kernel(num_coefficients: int) -> torch.Tensor:
    """Sine-windowed MDCT analysis kernel used by the MDCT critic.

    Keeping the transform inside the discriminator makes every sub-critic see
    the signed, critically sampled representation used by MDCTCodec rather
    than a redundant complex STFT magnitude.  The kernel matches the fixed
    MDCT front end in :mod:`src.models.audio_codec`.
    """
    num_coefficients = int(num_coefficients)
    if num_coefficients <= 0:
        raise ValueError("num_coefficients must be positive")
    block_size = 2 * num_coefficients
    sample = torch.arange(block_size, dtype=torch.float64).unsqueeze(1)
    frequency = torch.arange(num_coefficients, dtype=torch.float64).unsqueeze(0)
    basis = torch.cos(
        math.pi
        / num_coefficients
        * (sample + 0.5 + num_coefficients / 2.0)
        * (frequency + 0.5)
    )
    window = torch.sin(
        math.pi
        / block_size
        * (torch.arange(block_size, dtype=torch.float64) + 0.5)
    ).unsqueeze(1)
    return (basis * window).t().unsqueeze(1).to(torch.float32).contiguous()


class AudioMDCTSubDiscriminator(nn.Module):
    """One signed-MDCT branch of the multi-resolution audio critic.

    The convolution kernel schedule follows the MR-MDCTD description in the
    MDCTCodec paper. Strided 2-D convolutions reduce both the frame and
    frequency axes, keeping the shortest-window branch inexpensive for the
    one-second VCTK crops used during training.
    """

    def __init__(self, *, num_coefficients: int, channels: int = 64) -> None:
        super().__init__()
        self.num_coefficients = int(num_coefficients)
        self.hop_length = self.num_coefficients
        self.register_buffer(
            "analysis_kernel",
            _mdct_analysis_kernel(self.num_coefficients),
            persistent=False,
        )
        channels = max(1, int(channels))
        kernel_sizes = ((7, 5), (5, 3), (5, 3), (3, 3), (3, 3))
        layers: list[nn.Module] = []
        in_channels = 1
        for index, kernel_size in enumerate(kernel_sizes):
            # Preserve more resolution in the first layer and then reduce both
            # axes. Same-style padding avoids favouring particular MDCT bins.
            stride = (1, 1) if index == 0 else (2, 2)
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)
            layers.append(
                nn.Conv2d(
                    in_channels,
                    channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            in_channels = channels
        self.layers = nn.ModuleList(layers)
        self.output = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.apply(weights_init)

    def _analysis(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3 or int(waveform.size(1)) != 1:
            raise ValueError(
                "MDCT discriminator expects mono waveform [B, 1, T], got "
                f"{tuple(waveform.shape)}"
            )
        # Boundary padding matches the codec MDCT and permits arbitrary crop
        # lengths; the final partial hop is intentionally omitted.
        padded = F.pad(waveform.float(), (self.hop_length, self.hop_length))
        coefficients = F.conv1d(
            padded,
            self.analysis_kernel,
            stride=self.hop_length,
        )
        # Conv2d layout: [batch, channel, time frames, signed MDCT bins].
        return coefficients.transpose(1, 2).unsqueeze(1)

    def forward(self, waveform: torch.Tensor, return_features: bool = False):
        hidden = self._analysis(waveform)
        features: list[torch.Tensor] = []
        for layer in self.layers:
            hidden = F.leaky_relu(layer(hidden), negative_slope=0.2)
            if return_features:
                features.append(hidden)
        logits = self.output(hidden)
        if return_features:
            return logits, features
        return logits


class AudioMultiResolutionMDCTDiscriminator(nn.Module):
    """MDCTCodec-style critic over multiple signed-MDCT resolutions."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        num_filters: int = 64,
        mdct_num_coefficients=(100, 25, 10),
    ) -> None:
        super().__init__()
        if int(in_channels) != 1:
            raise ValueError("multi-resolution MDCT discriminator expects mono audio")
        resolutions = tuple(
            int(value) for value in mdct_num_coefficients if int(value) > 0
        )
        if not resolutions:
            raise ValueError("MDCT discriminator needs at least one resolution")
        self.discriminators = nn.ModuleList(
            [
                AudioMDCTSubDiscriminator(
                    num_coefficients=value,
                    channels=int(num_filters),
                )
                for value in resolutions
            ]
        )

    def forward(self, waveform: torch.Tensor, return_features: bool = False):
        logits: list[torch.Tensor] = []
        features: list[list[torch.Tensor]] = []
        for discriminator in self.discriminators:
            if return_features:
                branch_logits, branch_features = discriminator(
                    waveform,
                    return_features=True,
                )
                logits.append(branch_logits)
                features.append(branch_features)
            else:
                logits.append(discriminator(waveform))
        if return_features:
            return logits, features
        return logits


class MDCTCodecOfficialSubDiscriminator(nn.Module):
    """One branch of the released MR-MDCTD critic.

    This is intentionally separate from the earlier LASER approximation so
    existing checkpoints retain their topology.  Transform normalization,
    convolution strides, weight normalization, activation slope, and feature
    map collection all match the authors' implementation.
    """

    def __init__(self, *, num_coefficients: int, channels: int = 64) -> None:
        super().__init__()
        self.num_coefficients = int(num_coefficients)
        self.hop_length = self.num_coefficients
        analysis = (
            math.sqrt(2.0 / float(self.num_coefficients))
            * _mdct_analysis_kernel(self.num_coefficients)
        )
        self.register_buffer("analysis_kernel", analysis, persistent=False)
        channels = int(channels)
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv2d(
                        1, channels, kernel_size=(7, 5), stride=(2, 2), padding=(3, 2)
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        channels, channels, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        channels, channels, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1)
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        channels, channels, kernel_size=(3, 3), stride=(2, 1), padding=1
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        channels, channels, kernel_size=(3, 3), stride=(2, 2), padding=1
                    )
                ),
            ]
        )
        self.conv_post = weight_norm(
            nn.Conv2d(channels, 1, kernel_size=(3, 3), padding=(1, 1))
        )

    def _analysis(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3 or int(waveform.size(1)) != 1:
            raise ValueError(
                "MR-MDCTD expects mono waveform [B, 1, T], got "
                f"{tuple(waveform.shape)}"
            )
        padded = F.pad(waveform.float(), (self.hop_length, self.hop_length))
        coefficients = F.conv1d(
            padded,
            self.analysis_kernel,
            stride=self.hop_length,
        )
        # Official Conv2d layout is [batch, channel, MDCT bin, frame].
        return coefficients.unsqueeze(1)

    def forward(self, waveform: torch.Tensor, return_features: bool = False):
        hidden = self._analysis(waveform)
        features: list[torch.Tensor] = []
        for layer in self.convs:
            hidden = F.leaky_relu(layer(hidden), negative_slope=0.1)
            if return_features:
                features.append(hidden)
        logits = self.conv_post(hidden)
        if return_features:
            # MDCTCodec includes the final logit map in feature matching.
            features.append(logits)
            return logits.flatten(1), features
        return logits.flatten(1)


class MDCTCodecOfficialDiscriminator(nn.Module):
    """The three-branch MDCTCodec discriminator with an exact public shell."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        num_filters: int = 64,
        mdct_num_coefficients=(50, 200, 20),
    ) -> None:
        super().__init__()
        if int(in_channels) != 1:
            raise ValueError("MDCTCodec discriminator expects mono audio")
        resolutions = tuple(int(value) for value in mdct_num_coefficients)
        if not resolutions or any(value <= 0 for value in resolutions):
            raise ValueError("MDCTCodec discriminator resolutions must be positive")
        self.discriminators = nn.ModuleList(
            [
                MDCTCodecOfficialSubDiscriminator(
                    num_coefficients=value,
                    channels=int(num_filters),
                )
                for value in resolutions
            ]
        )

    def forward(self, waveform: torch.Tensor, return_features: bool = False):
        logits: list[torch.Tensor] = []
        features: list[list[torch.Tensor]] = []
        for discriminator in self.discriminators:
            if return_features:
                branch_logits, branch_features = discriminator(
                    waveform, return_features=True
                )
                logits.append(branch_logits)
                features.append(branch_features)
            else:
                logits.append(discriminator(waveform))
        if return_features:
            return logits, features
        return logits


def _as_logit_list(logits) -> list[torch.Tensor]:
    if isinstance(logits, torch.Tensor):
        return [logits]
    items = list(logits)
    if not items:
        raise ValueError("expected at least one discriminator logit tensor")
    return items


def multi_hinge_d_loss(logits_real, logits_fake) -> torch.Tensor:
    real_list = _as_logit_list(logits_real)
    fake_list = _as_logit_list(logits_fake)
    if len(real_list) != len(fake_list):
        raise ValueError(
            "real/fake discriminator output count mismatch: "
            f"{len(real_list)} vs {len(fake_list)}"
        )
    losses = [hinge_d_loss(real, fake) for real, fake in zip(real_list, fake_list)]
    return torch.stack(losses).mean()


def multi_vanilla_d_loss(logits_real, logits_fake) -> torch.Tensor:
    real_list = _as_logit_list(logits_real)
    fake_list = _as_logit_list(logits_fake)
    if len(real_list) != len(fake_list):
        raise ValueError(
            "real/fake discriminator output count mismatch: "
            f"{len(real_list)} vs {len(fake_list)}"
        )
    losses = [vanilla_d_loss(real, fake) for real, fake in zip(real_list, fake_list)]
    return torch.stack(losses).mean()


def multi_hinge_g_loss(logits_fake) -> torch.Tensor:
    losses = [hinge_g_loss(fake) for fake in _as_logit_list(logits_fake)]
    return torch.stack(losses).mean()


def multi_encodec_hinge_d_loss(logits_real, logits_fake) -> torch.Tensor:
    """Meta EnCodec discriminator hinge loss, averaged across STFT scales."""
    real_list = _as_logit_list(logits_real)
    fake_list = _as_logit_list(logits_fake)
    if len(real_list) != len(fake_list):
        raise ValueError(
            "real/fake discriminator output count mismatch: "
            f"{len(real_list)} vs {len(fake_list)}"
        )
    losses = [
        torch.relu(1.0 - real).mean() + torch.relu(1.0 + fake).mean()
        for real, fake in zip(real_list, fake_list)
    ]
    return torch.stack(losses).mean()


def multi_encodec_hinge_g_loss(logits_fake) -> torch.Tensor:
    """Published EnCodec clipped generator objective ``relu(1 - D(fake))``."""
    losses = [torch.relu(1.0 - fake).mean() for fake in _as_logit_list(logits_fake)]
    return torch.stack(losses).mean()


def multi_mdctcodec_hinge_d_loss(logits_real, logits_fake) -> torch.Tensor:
    """Published MR-MDCTD hinge objective, summed across three resolutions."""
    real_list = _as_logit_list(logits_real)
    fake_list = _as_logit_list(logits_fake)
    if len(real_list) != len(fake_list):
        raise ValueError(
            "real/fake discriminator output count mismatch: "
            f"{len(real_list)} vs {len(fake_list)}"
        )
    losses = [
        torch.relu(1.0 - real).mean() + torch.relu(1.0 + fake).mean()
        for real, fake in zip(real_list, fake_list)
    ]
    return torch.stack(losses).sum()


def multi_mdctcodec_hinge_g_loss(logits_fake) -> torch.Tensor:
    """Published clipped generator hinge objective, summed over MR-MDCTD."""
    losses = [torch.relu(1.0 - fake).mean() for fake in _as_logit_list(logits_fake)]
    return torch.stack(losses).sum()


def lsgan_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Least-squares (LSGAN) discriminator loss — the canonical HiFi-GAN critic loss."""
    return 0.5 * (torch.mean((logits_real - 1.0) ** 2) + torch.mean(logits_fake ** 2))


def lsgan_g_loss(logits_fake: torch.Tensor) -> torch.Tensor:
    """Least-squares (LSGAN) generator loss — pushes fake logits toward 1."""
    return torch.mean((logits_fake - 1.0) ** 2)


def multi_lsgan_d_loss(logits_real, logits_fake) -> torch.Tensor:
    real_list = _as_logit_list(logits_real)
    fake_list = _as_logit_list(logits_fake)
    if len(real_list) != len(fake_list):
        raise ValueError(
            "real/fake discriminator output count mismatch: "
            f"{len(real_list)} vs {len(fake_list)}"
        )
    losses = [lsgan_d_loss(real, fake) for real, fake in zip(real_list, fake_list)]
    return torch.stack(losses).mean()


def multi_lsgan_g_loss(logits_fake) -> torch.Tensor:
    losses = [lsgan_g_loss(fake) for fake in _as_logit_list(logits_fake)]
    return torch.stack(losses).mean()


def feature_matching_loss(feats_real, feats_fake) -> torch.Tensor:
    """HiFi-GAN feature-matching loss: mean L1 between the discriminator's
    intermediate feature maps for real vs generated audio.

    ``feats_real``/``feats_fake`` are lists (one per sub-discriminator) of lists of
    feature tensors. Real features are detached so the loss only updates the
    generator. Returns a zero scalar if no features are provided.
    """
    terms: list[torch.Tensor] = []
    for real_maps, fake_maps in zip(feats_real, feats_fake):
        for real_feat, fake_feat in zip(real_maps, fake_maps):
            terms.append(F.l1_loss(fake_feat, real_feat.detach()))
    if not terms:
        # Caller guarantees at least one feature in practice; guard anyway.
        return torch.zeros((), device=feats_fake[0][0].device) if feats_fake and feats_fake[0] else torch.zeros(())
    return torch.stack(terms).mean()


def mdctcodec_feature_matching_loss(feats_real, feats_fake) -> torch.Tensor:
    """Reference spectral-codec feature matching: sum mean-L1 over all maps."""
    terms: list[torch.Tensor] = []
    for real_maps, fake_maps in zip(feats_real, feats_fake):
        for real_feat, fake_feat in zip(real_maps, fake_maps):
            terms.append(F.l1_loss(fake_feat, real_feat.detach()))
    if terms:
        return torch.stack(terms).sum()
    return (
        torch.zeros((), device=feats_fake[0][0].device)
        if feats_fake and feats_fake[0]
        else torch.zeros(())
    )


def relative_feature_matching_loss(feats_real, feats_fake) -> torch.Tensor:
    """Meta EnCodec feature matching normalized by real-feature magnitude."""
    terms: list[torch.Tensor] = []
    for real_maps, fake_maps in zip(feats_real, feats_fake):
        for real_feat, fake_feat in zip(real_maps, fake_maps):
            real_target = real_feat.detach()
            numerator = (fake_feat - real_target).abs().mean()
            denominator = real_target.abs().mean().clamp_min(1.0e-6)
            terms.append(numerator / denominator)
    if not terms:
        return (
            torch.zeros((), device=feats_fake[0][0].device)
            if feats_fake and feats_fake[0]
            else torch.zeros(())
        )
    return torch.stack(terms).mean()


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
    "AudioMultiScalePeriodDiscriminator",
    "AudioEncodecMultiScaleSTFTDiscriminator",
    "AudioMDCTSubDiscriminator",
    "AudioMultiResolutionMDCTDiscriminator",
    "MDCTCodecOfficialSubDiscriminator",
    "MDCTCodecOfficialDiscriminator",
    "weights_init",
    "hinge_d_loss",
    "vanilla_d_loss",
    "hinge_g_loss",
    "multi_hinge_d_loss",
    "multi_vanilla_d_loss",
    "multi_hinge_g_loss",
    "multi_encodec_hinge_d_loss",
    "multi_encodec_hinge_g_loss",
    "multi_mdctcodec_hinge_d_loss",
    "multi_mdctcodec_hinge_g_loss",
    "feature_matching_loss",
    "mdctcodec_feature_matching_loss",
    "relative_feature_matching_loss",
    "adopt_weight",
]
