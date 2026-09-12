"""Waveform encoder/decoder blocks for audio autoencoders.

The local ``AudioEncoder``/``AudioDecoder`` pair is intentionally lightweight.
``build_meta_encodec_24khz_backbone`` exposes Meta's official pretrained
EnCodec analysis/synthesis network so callers can replace EnCodec's RVQ with a
different bottleneck while retaining the released SEANet weights.
"""

from __future__ import annotations

import ast
import math
import weakref
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


META_ENCODEC_SAMPLE_RATE = 24_000
META_ENCODEC_LATENT_DIM = 128
MDCT_PITCH_FEATURE_DIM = 4


def _mdct_kernels(num_coefficients: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sine-windowed MDCT and IMDCT convolution kernels.

    The transform uses a ``2N`` sample window and ``N`` sample hop. With one
    zero block of boundary padding on each side, transposed-convolution
    overlap-add followed by removing that padding is perfect reconstruction.
    """
    n_coefficients = int(num_coefficients)
    if n_coefficients <= 0:
        raise ValueError("num_coefficients must be positive")
    block_size = 2 * n_coefficients
    n = torch.arange(block_size, dtype=torch.float64).unsqueeze(1)
    k = torch.arange(n_coefficients, dtype=torch.float64).unsqueeze(0)
    basis = torch.cos(
        math.pi
        / n_coefficients
        * (n + 0.5 + n_coefficients / 2.0)
        * (k + 0.5)
    )
    window = torch.sin(
        math.pi
        / block_size
        * (torch.arange(block_size, dtype=torch.float64) + 0.5)
    ).unsqueeze(1)
    analysis = (basis * window).t().unsqueeze(1).to(torch.float32)
    synthesis = (2.0 / n_coefficients * basis * window).t().unsqueeze(1).to(torch.float32)
    return analysis.contiguous(), synthesis.contiguous()


def _mdct_erb_filterbank(
    num_coefficients: int,
    *,
    sample_rate: int,
    num_bands: int,
    min_frequency: float = 30.0,
) -> torch.Tensor:
    """Return unit-area ERB bands evaluated at the MDCT bin centres."""
    num_coefficients = int(num_coefficients)
    sample_rate = int(sample_rate)
    num_bands = int(num_bands)
    if num_bands <= 0:
        return torch.empty(0, num_coefficients, dtype=torch.float32)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    def hz_to_erb(frequency: torch.Tensor) -> torch.Tensor:
        return 21.4 * torch.log10(1.0 + 4.37e-3 * frequency)

    def erb_to_hz(rate: torch.Tensor) -> torch.Tensor:
        return (torch.pow(10.0, rate / 21.4) - 1.0) / 4.37e-3

    nyquist = 0.5 * float(sample_rate)
    low = torch.tensor(max(0.0, float(min_frequency)), dtype=torch.float64)
    high = torch.tensor(nyquist, dtype=torch.float64)
    edges = erb_to_hz(
        torch.linspace(hz_to_erb(low), hz_to_erb(high), num_bands + 2)
    )
    # A type-IV DCT coefficient k is centred at (k + 1/2) Fs / (2N).
    frequencies = (
        (torch.arange(num_coefficients, dtype=torch.float64) + 0.5)
        * float(sample_rate)
        / (2.0 * num_coefficients)
    )
    left = edges[:-2, None]
    centre = edges[1:-1, None]
    right = edges[2:, None]
    rising = (frequencies[None, :] - left) / (centre - left).clamp_min(1.0e-12)
    falling = (right - frequencies[None, :]) / (right - centre).clamp_min(1.0e-12)
    filterbank = torch.minimum(rising, falling).clamp(0.0, 1.0)
    filterbank = filterbank / filterbank.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    return filterbank.to(torch.float32).contiguous()


class MDCTEncoder(nn.Module):
    """MDCT analysis optionally augmented with perceptual side information.

    The additional log-band-energy and pitch channels are aligned exactly to
    the MDCT frames and participate in sparse support selection. They are side
    information for the LASER dictionary, not extra IMDCT coefficients.
    """

    def __init__(
        self,
        num_coefficients: int = 320,
        *,
        learnable_gain: bool = True,
        sample_rate: int = META_ENCODEC_SAMPLE_RATE,
        num_log_bands: int = 0,
        log_band_scale: float = 0.25,
        include_pitch: bool = False,
        pitch_min_hz: float = 60.0,
        pitch_max_hz: float = 500.0,
        pitch_scale: float = 1.0,
    ):
        super().__init__()
        self.num_coefficients = int(num_coefficients)
        self.hop_length = self.num_coefficients
        self.sample_rate = int(sample_rate)
        self.num_log_bands = int(num_log_bands)
        self.log_band_scale = float(log_band_scale)
        self.include_pitch = bool(include_pitch)
        self.pitch_min_hz = float(pitch_min_hz)
        self.pitch_max_hz = float(pitch_max_hz)
        self.pitch_scale = float(pitch_scale)
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.num_log_bands < 0:
            raise ValueError("num_log_bands must be non-negative")
        if not math.isfinite(self.log_band_scale) or self.log_band_scale < 0.0:
            raise ValueError("log_band_scale must be finite and non-negative")
        if not math.isfinite(self.pitch_scale) or self.pitch_scale < 0.0:
            raise ValueError("pitch_scale must be finite and non-negative")
        if not 0.0 < self.pitch_min_hz < self.pitch_max_hz < 0.5 * self.sample_rate:
            raise ValueError(
                "pitch range must satisfy 0 < min < max < Nyquist, got "
                f"{self.pitch_min_hz}, {self.pitch_max_hz} at {self.sample_rate} Hz"
            )
        analysis, _ = _mdct_kernels(self.num_coefficients)
        self.register_buffer("analysis_kernel", analysis, persistent=False)
        self.register_buffer(
            "log_band_filterbank",
            _mdct_erb_filterbank(
                self.num_coefficients,
                sample_rate=self.sample_rate,
                num_bands=self.num_log_bands,
            ),
            persistent=False,
        )
        self.log_gain = nn.Parameter(
            torch.zeros(self.num_coefficients),
            requires_grad=bool(learnable_gain),
        )

    @property
    def dimension(self) -> int:
        pitch_dim = MDCT_PITCH_FEATURE_DIM if self.include_pitch else 0
        return self.num_coefficients + self.num_log_bands + pitch_dim

    def gain(self) -> torch.Tensor:
        # Geometric-mean normalization removes an unidentifiable global scale.
        centered = self.log_gain - self.log_gain.mean()
        return centered.exp()

    def _log_band_features(self, coefficients: torch.Tensor) -> torch.Tensor:
        if self.num_log_bands <= 0:
            return coefficients.new_empty(
                coefficients.size(0), 0, coefficients.size(-1)
            )
        coefficients_float = coefficients.float()
        filterbank = self.log_band_filterbank.to(
            device=coefficients.device,
            dtype=coefficients_float.dtype,
        )
        energy = torch.einsum(
            "kn,bnt->bkt", filterbank, coefficients_float.square()
        )
        # Additive epsilon only defines log(0); there is no value clipping.
        log_energy = torch.log(energy + 1.0e-8) * self.log_band_scale
        return log_energy.to(dtype=coefficients.dtype)

    def _pitch_features(self, padded_waveform: torch.Tensor) -> torch.Tensor:
        """Extract YIN-style pitch/voicing descriptors on the MDCT grid."""
        frames = padded_waveform.unfold(
            -1, 2 * self.num_coefficients, self.hop_length
        ).squeeze(1)
        work = frames.float()
        work = work - work.mean(dim=-1, keepdim=True)
        frame_length = int(work.size(-1))
        min_lag = max(1, int(math.floor(self.sample_rate / self.pitch_max_hz)))
        max_lag = min(
            frame_length - 2,
            int(math.ceil(self.sample_rate / self.pitch_min_hz)),
        )
        if max_lag <= min_lag:
            raise ValueError(
                "MDCT window is too short for the requested pitch range: "
                f"window={frame_length}, lag range=({min_lag}, {max_lag})"
            )

        fft_size = 1 << (2 * frame_length - 1).bit_length()
        spectrum = torch.fft.rfft(work, n=fft_size, dim=-1)
        autocorrelation = torch.fft.irfft(
            spectrum.abs().square(), n=fft_size, dim=-1
        )[..., : max_lag + 1]
        squared_prefix = F.pad(work.square().cumsum(dim=-1), (1, 0))
        all_lags = torch.arange(1, max_lag + 1, device=work.device)
        left_energy = squared_prefix[..., frame_length - all_lags]
        right_energy = squared_prefix[..., frame_length].unsqueeze(-1) - squared_prefix[..., all_lags]
        difference = (
            left_energy
            + right_energy
            - 2.0 * autocorrelation[..., 1 : max_lag + 1]
        )
        difference = difference.relu()
        cumulative_mean_normalized = (
            difference
            * all_lags.to(dtype=difference.dtype)
            / (difference.cumsum(dim=-1) + 1.0e-8)
        )
        search = cumulative_mean_normalized[..., min_lag - 1 : max_lag]

        # YIN selects the first plausible period, avoiding octave errors from a
        # later harmonic minimum. Fall back to the global trough when unvoiced.
        below_threshold = search < 0.15
        first_below = below_threshold.to(torch.int64).argmax(dim=-1)
        fallback = search.argmin(dim=-1)
        selected_index = torch.where(below_threshold.any(dim=-1), first_below, fallback)
        selected_lag = selected_index + min_lag
        selected_cmnd = search.gather(-1, selected_index.unsqueeze(-1)).squeeze(-1)
        periodicity = 1.0 - selected_cmnd
        voicing = torch.sigmoid((periodicity - 0.50) / 0.10)
        f0_hz = float(self.sample_rate) / selected_lag.to(dtype=work.dtype)
        log_min = math.log2(self.pitch_min_hz)
        log_span = math.log2(self.pitch_max_hz) - log_min
        normalized_log_f0 = (torch.log2(f0_hz) - log_min) / log_span
        phase = 2.0 * math.pi * torch.log2(f0_hz)
        pitch = torch.stack(
            (
                voicing,
                voicing * normalized_log_f0,
                voicing * torch.sin(phase),
                voicing * torch.cos(phase),
            ),
            dim=1,
        )
        return (pitch * self.pitch_scale).to(dtype=padded_waveform.dtype)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3 or int(waveform.size(1)) != 1:
            raise ValueError(
                f"MDCT expects mono waveform [B, 1, T], got {tuple(waveform.shape)}"
            )
        if int(waveform.size(-1)) % self.hop_length != 0:
            raise ValueError(
                "MDCT waveform length must be divisible by its hop length "
                f"({self.hop_length}), got {int(waveform.size(-1))}"
            )
        padded = F.pad(waveform, (self.hop_length, self.hop_length))
        coefficients = F.conv1d(
            padded,
            self.analysis_kernel.to(dtype=waveform.dtype),
            stride=self.hop_length,
        )
        coefficients = coefficients * self.gain().to(
            dtype=coefficients.dtype
        ).view(1, -1, 1)
        features = [coefficients]
        if self.num_log_bands > 0:
            features.append(self._log_band_features(coefficients))
        if self.include_pitch:
            features.append(self._pitch_features(padded))
        return torch.cat(features, dim=1)


class MDCTDecoder(nn.Module):
    """Matched IMDCT overlap-add synthesis transform."""

    def __init__(self, encoder: MDCTEncoder):
        super().__init__()
        _, synthesis = _mdct_kernels(encoder.num_coefficients)
        self.num_coefficients = int(encoder.num_coefficients)
        self.hop_length = self.num_coefficients
        self.register_buffer("synthesis_kernel", synthesis, persistent=False)
        # Do not register the encoder (or its shared gain) a second time. That
        # would duplicate the same parameter across optimizer groups/state_dict.
        object.__setattr__(self, "_encoder_ref", weakref.ref(encoder))

    def forward(self, coefficients: torch.Tensor) -> torch.Tensor:
        encoder = self._encoder_ref()
        if encoder is None:
            raise RuntimeError("MDCT analysis module is no longer available")
        if coefficients.ndim != 3 or int(coefficients.size(1)) != encoder.dimension:
            raise ValueError(
                "IMDCT expects the combined MDCT feature tensor [B, C, frames] "
                f"with C={encoder.dimension}, got {tuple(coefficients.shape)}"
            )
        # Perceptual side channels influence sparse support selection; only the
        # reconstructed transform coefficients are consumed by synthesis.
        coefficients = coefficients[:, : self.num_coefficients]
        gain = encoder.gain().to(dtype=coefficients.dtype).view(1, -1, 1)
        normalized = coefficients / gain
        padded = F.conv_transpose1d(
            normalized,
            self.synthesis_kernel.to(dtype=coefficients.dtype),
            stride=self.hop_length,
        )
        return padded[..., self.hop_length : -self.hop_length]


def build_mdct_backbone(
    *,
    num_coefficients: int = 320,
    learnable_gain: bool = True,
    sample_rate: int = META_ENCODEC_SAMPLE_RATE,
    num_log_bands: int = 0,
    log_band_scale: float = 0.25,
    include_pitch: bool = False,
    pitch_min_hz: float = 60.0,
    pitch_max_hz: float = 500.0,
    pitch_scale: float = 1.0,
) -> tuple[MDCTEncoder, MDCTDecoder]:
    """Build a matched MDCT/IMDCT analysis-synthesis pair."""
    encoder = MDCTEncoder(
        num_coefficients=num_coefficients,
        learnable_gain=learnable_gain,
        sample_rate=sample_rate,
        num_log_bands=num_log_bands,
        log_band_scale=log_band_scale,
        include_pitch=include_pitch,
        pitch_min_hz=pitch_min_hz,
        pitch_max_hz=pitch_max_hz,
        pitch_scale=pitch_scale,
    )
    return encoder, MDCTDecoder(encoder)


def build_meta_encodec_24khz_model(*, pretrained: bool = True):
    """Return Meta's complete released 24 kHz EnCodec model.

    Most LASER callers only retain the analysis/synthesis backbone through
    :func:`build_meta_encodec_24khz_backbone`.  The complete model is also
    needed when the released RVQ is used as a frozen decoder-input teacher or
    as a principled dictionary initializer.
    """

    try:
        from encodec import EncodecModel
    except ImportError as exc:  # pragma: no cover - exercised in minimal envs
        raise RuntimeError(
            "Meta EnCodec integration requires the `encodec` package; "
            "install the project requirements (encodec==0.1.1)."
        ) from exc

    model = EncodecModel.encodec_model_24khz(pretrained=bool(pretrained))
    if int(model.sample_rate) != META_ENCODEC_SAMPLE_RATE:
        raise RuntimeError(
            "Unexpected Meta EnCodec sample rate: "
            f"{model.sample_rate} (expected {META_ENCODEC_SAMPLE_RATE})"
        )
    if int(model.encoder.dimension) != META_ENCODEC_LATENT_DIM:
        raise RuntimeError(
            "Unexpected Meta EnCodec latent dimension: "
            f"{model.encoder.dimension} (expected {META_ENCODEC_LATENT_DIM})"
        )
    return model


def build_meta_encodec_24khz_backbone(*, pretrained: bool = True):
    """Return Meta EnCodec's mono 24 kHz encoder and decoder.

    Only the analysis/synthesis modules are retained.  The upstream residual
    vector quantizer is deliberately discarded so the caller can install the
    LASER sparse dictionary bottleneck in its place.
    """

    model = build_meta_encodec_24khz_model(pretrained=pretrained)
    return model.encoder, model.decoder


def canonical_int_tuple(values, *, default: Iterable[int]) -> Tuple[int, ...]:
    if values is None:
        values = tuple(default)
    if isinstance(values, str):
        raw = values.strip()
        if not raw:
            values = tuple(default)
        elif raw[0] in "[(":
            values = ast.literal_eval(raw)
        else:
            values = [part for part in raw.split(",") if part.strip()]
    out = tuple(int(value) for value in values)
    if not out:
        raise ValueError("Expected a non-empty integer tuple")
    if any(value <= 0 for value in out):
        raise ValueError(f"All tuple values must be positive, got {out}")
    return out


class AudioResidualUnit(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, *, dilation: int, kernel_size: int = 7):
        super().__init__()
        channels = int(channels)
        hidden_channels = int(hidden_channels)
        dilation = int(dilation)
        kernel_size = int(kernel_size)
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(
                channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class AudioResidualStack(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        *,
        num_layers: int,
        dilation_cycle=(1, 3, 9),
    ):
        super().__init__()
        dilations = canonical_int_tuple(dilation_cycle, default=(1, 3, 9))
        self.layers = nn.ModuleList(
            [
                AudioResidualUnit(
                    channels,
                    hidden_channels,
                    dilation=dilations[idx % len(dilations)],
                )
                for idx in range(int(num_layers))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class AudioGlobalResponseNorm(nn.Module):
    """ConvNeXt-v2 global response normalization for ``[B, T, C]`` audio."""

    def __init__(self, channels: int):
        super().__init__()
        channels = int(channels)
        self.gamma = nn.Parameter(torch.zeros(1, 1, channels))
        self.beta = nn.Parameter(torch.zeros(1, 1, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        response = torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)
        normalized = response / response.mean(dim=-1, keepdim=True).clamp_min(1.0e-6)
        return x + self.gamma * (x * normalized) + self.beta


class AudioConvNeXtV2Block(nn.Module):
    """Modified ConvNeXt-v2 block used by the published MDCTCodec backbone."""

    def __init__(
        self,
        channels: int,
        intermediate_channels: int,
        *,
        kernel_size: int = 7,
    ):
        super().__init__()
        channels = int(channels)
        intermediate_channels = int(intermediate_channels)
        kernel_size = int(kernel_size)
        if channels <= 0 or intermediate_channels <= 0:
            raise ValueError("ConvNeXt channel dimensions must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("ConvNeXt kernel_size must be a positive odd integer")
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.norm = nn.LayerNorm(channels, eps=1.0e-6)
        self.expand = nn.Linear(channels, intermediate_channels)
        self.activation = nn.GELU()
        self.grn = AudioGlobalResponseNorm(intermediate_channels)
        self.project = nn.Linear(intermediate_channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x).transpose(1, 2)
        x = self.norm(x)
        x = self.expand(x)
        x = self.activation(x)
        x = self.grn(x)
        x = self.project(x).transpose(1, 2)
        return residual + x


class AudioConvNeXtV2Stack(nn.Module):
    def __init__(
        self,
        channels: int,
        intermediate_channels: int,
        *,
        num_layers: int,
        kernel_size: int = 7,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                AudioConvNeXtV2Block(
                    channels,
                    intermediate_channels,
                    kernel_size=kernel_size,
                )
                for _ in range(int(num_layers))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def _initialize_mdctcodec_module_(module: nn.Module) -> None:
    """Initialization used by the ConvNeXt spectral codec family."""
    if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _initialize_identity_projection_(projection: nn.Conv1d) -> None:
    """Initialize a 1x1 convolution as a rectangular identity map."""
    if projection.kernel_size != (1,):
        raise ValueError("identity projection initialization requires kernel_size=1")
    with torch.no_grad():
        projection.weight.zero_()
        diagonal = min(int(projection.in_channels), int(projection.out_channels))
        indices = torch.arange(diagonal)
        projection.weight[indices, indices, 0] = 1.0
        if projection.bias is not None:
            projection.bias.zero_()


def _initialize_residual_stack_as_identity_(stack: AudioResidualStack) -> None:
    """Zero residual branch outputs while retaining trainable internal weights."""
    with torch.no_grad():
        for layer in stack.layers:
            output = layer.net[-1]
            output.weight.zero_()
            if output.bias is not None:
                output.bias.zero_()


class MDCTVAEEncoder(nn.Module):
    """Learned temporal encoder operating on fixed MDCT frames.

    The outer MDCT is deterministic and critically sampled. The neural network
    learns the compact latent coordinate system in which LASER performs K-sparse
    dictionary coding, rather than forcing K atoms to approximate raw transform
    bins directly.
    """

    def __init__(
        self,
        analysis: MDCTEncoder,
        *,
        latent_dim: int,
        hidden_channels: int = 256,
        residual_hidden_channels: int = 128,
        num_residual_layers: int = 6,
        dilation_cycle=(1, 3, 9),
        temporal_downsample_factor: int = 1,
        use_convnext_v2: bool = False,
        convnext_intermediate_channels: int = 512,
    ):
        super().__init__()
        if analysis.dimension != analysis.num_coefficients:
            raise ValueError("MDCT-VAE analysis cannot include auxiliary feature channels")
        self.analysis = analysis
        self.num_coefficients = int(analysis.num_coefficients)
        self.latent_dim = int(latent_dim)
        self.hidden_channels = int(hidden_channels)
        self.temporal_downsample_factor = int(temporal_downsample_factor)
        self.use_convnext_v2 = bool(use_convnext_v2)
        if self.latent_dim <= 0 or self.hidden_channels <= 0:
            raise ValueError("MDCT-VAE latent and hidden dimensions must be positive")
        if self.temporal_downsample_factor <= 0:
            raise ValueError("temporal_downsample_factor must be positive")
        # MDCTCodec Figure 2: Conv1D -> LayerNorm -> 8 ConvNeXt-v2 blocks ->
        # LayerNorm -> Linear -> stride-D Conv1D -> latent Conv1D.  Every
        # Conv1D has kernel seven; the upsampling DeConv1D is the sole
        # exception and mirrors APCodec's exact factor-eight geometry.
        projection_kernel = 7 if self.use_convnext_v2 else 1
        projection_padding = 3 if self.use_convnext_v2 else 0
        self.input_projection = nn.Conv1d(
            self.num_coefficients,
            self.hidden_channels,
            kernel_size=projection_kernel,
            padding=projection_padding,
        )
        self.input_norm = (
            nn.LayerNorm(self.hidden_channels, eps=1.0e-6)
            if self.use_convnext_v2
            else nn.Identity()
        )
        if self.use_convnext_v2:
            self.temporal = AudioConvNeXtV2Stack(
                self.hidden_channels,
                int(convnext_intermediate_channels),
                num_layers=int(num_residual_layers),
                kernel_size=7,
            )
        else:
            self.temporal = AudioResidualStack(
                self.hidden_channels,
                int(residual_hidden_channels),
                num_layers=int(num_residual_layers),
                dilation_cycle=dilation_cycle,
            )
        self.output_norm = (
            nn.LayerNorm(self.hidden_channels, eps=1.0e-6)
            if self.use_convnext_v2
            else nn.Identity()
        )
        self.output_linear = (
            nn.Linear(self.hidden_channels, self.hidden_channels)
            if self.use_convnext_v2
            else nn.Identity()
        )
        if self.temporal_downsample_factor > 1:
            factor = self.temporal_downsample_factor
            downsample_kernel = 7 if self.use_convnext_v2 else 2 * factor
            downsample_padding = 3 if self.use_convnext_v2 else factor // 2
            self.downsample = nn.Conv1d(
                self.hidden_channels,
                self.hidden_channels,
                kernel_size=downsample_kernel,
                stride=factor,
                padding=downsample_padding,
            )
        else:
            self.downsample = nn.Identity()
        self.latent_projection = nn.Conv1d(
            self.hidden_channels,
            self.latent_dim,
            kernel_size=projection_kernel,
            padding=projection_padding,
        )
        if self.use_convnext_v2:
            self.apply(_initialize_mdctcodec_module_)
        else:
            # Retain the identity-oriented initialization for the older
            # residual MDCT-VAE family only.  It is not the MDCTCodec recipe.
            _initialize_identity_projection_(self.input_projection)
            _initialize_residual_stack_as_identity_(self.temporal)
            _initialize_identity_projection_(self.latent_projection)

    @property
    def dimension(self) -> int:
        return self.latent_dim

    @property
    def hop_length(self) -> int:
        return int(self.analysis.hop_length) * self.temporal_downsample_factor

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        coefficients = self.analysis(waveform)
        factor = self.temporal_downsample_factor
        if factor > 1 and int(coefficients.size(-1)) % factor != 0:
            raise ValueError(
                "MDCTCodec temporal downsampling expects waveform length divisible by "
                "a frame-aligned crop (the padded MDCT frame count must be divisible "
                f"by {factor}); got "
                f"{int(waveform.size(-1))} samples and {int(coefficients.size(-1))} frames"
            )
        hidden = self.input_projection(coefficients)
        hidden = self.input_norm(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = self.temporal(hidden)
        hidden = self.output_norm(hidden.transpose(1, 2))
        hidden = self.output_linear(hidden).transpose(1, 2)
        hidden = self.downsample(hidden)
        return self.latent_projection(hidden)


class MDCTVAEDecoder(nn.Module):
    """Learned temporal decoder followed by matched IMDCT synthesis."""

    def __init__(
        self,
        synthesis: MDCTDecoder,
        *,
        latent_dim: int,
        num_coefficients: int,
        hidden_channels: int = 256,
        residual_hidden_channels: int = 128,
        num_residual_layers: int = 6,
        dilation_cycle=(1, 3, 9),
        temporal_upsample_factor: int = 1,
        use_convnext_v2: bool = False,
        convnext_intermediate_channels: int = 512,
    ):
        super().__init__()
        self.synthesis = synthesis
        self.latent_dim = int(latent_dim)
        self.num_coefficients = int(num_coefficients)
        self.hidden_channels = int(hidden_channels)
        self.temporal_upsample_factor = int(temporal_upsample_factor)
        self.use_convnext_v2 = bool(use_convnext_v2)
        if self.temporal_upsample_factor <= 0:
            raise ValueError("temporal_upsample_factor must be positive")
        projection_kernel = 7 if self.use_convnext_v2 else 1
        projection_padding = 3 if self.use_convnext_v2 else 0
        self.latent_projection = nn.Conv1d(
            self.latent_dim,
            self.hidden_channels,
            kernel_size=projection_kernel,
            padding=projection_padding,
        )
        if self.temporal_upsample_factor > 1:
            factor = self.temporal_upsample_factor
            upsample_kwargs = dict(
                kernel_size=2 * factor,
                stride=factor,
                padding=factor // 2,
            )
            if not self.use_convnext_v2:
                upsample_kwargs["output_padding"] = 1
            self.upsample = nn.ConvTranspose1d(
                self.hidden_channels,
                self.hidden_channels,
                **upsample_kwargs,
            )
        else:
            self.upsample = nn.Identity()
        if self.use_convnext_v2:
            self.temporal = AudioConvNeXtV2Stack(
                self.hidden_channels,
                int(convnext_intermediate_channels),
                num_layers=int(num_residual_layers),
                kernel_size=7,
            )
        else:
            self.temporal = AudioResidualStack(
                self.hidden_channels,
                int(residual_hidden_channels),
                num_layers=int(num_residual_layers),
                dilation_cycle=dilation_cycle,
            )
        self.input_linear = (
            nn.Linear(self.hidden_channels, self.hidden_channels)
            if self.use_convnext_v2
            else nn.Identity()
        )
        self.input_norm = (
            nn.LayerNorm(self.hidden_channels, eps=1.0e-6)
            if self.use_convnext_v2
            else nn.Identity()
        )
        self.output_norm = (
            nn.LayerNorm(self.hidden_channels, eps=1.0e-6)
            if self.use_convnext_v2
            else nn.Identity()
        )
        self.output_projection = nn.Conv1d(
            self.hidden_channels,
            self.num_coefficients,
            kernel_size=projection_kernel,
            padding=projection_padding,
        )
        if self.use_convnext_v2:
            self.apply(_initialize_mdctcodec_module_)
        else:
            _initialize_identity_projection_(self.latent_projection)
            _initialize_residual_stack_as_identity_(self.temporal)
            _initialize_identity_projection_(self.output_projection)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 3 or int(latent.size(1)) != self.latent_dim:
            raise ValueError(
                f"MDCT-VAE decoder expects [B, {self.latent_dim}, frames], "
                f"got {tuple(latent.shape)}"
            )
        hidden = self.latent_projection(latent)
        hidden = self.upsample(hidden)
        hidden = self.input_linear(hidden.transpose(1, 2))
        hidden = self.input_norm(hidden).transpose(1, 2)
        hidden = self.temporal(hidden)
        hidden = self.output_norm(hidden.transpose(1, 2)).transpose(1, 2)
        coefficients = self.output_projection(hidden)
        return self.synthesis(coefficients)


def _mdctcodec_orthonormal_kernels(
    num_coefficients: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MDCTCodec's orthonormal sine-windowed analysis/synthesis kernels.

    The legacy LASER MDCT classes intentionally retain their original scaling
    for checkpoint compatibility.  MDCTCodec instead multiplies both the MDCT
    and IMDCT bases by ``sqrt(2 / N)``.  Keeping this in a separate builder
    makes the paper's ``250 * MSE(MDCT)`` coefficient numerically identical to
    the released implementation without invalidating older runs.
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
    kernel = (
        math.sqrt(2.0 / float(num_coefficients))
        * basis
        * window
    ).t().unsqueeze(1).to(torch.float32).contiguous()
    return kernel, kernel.clone()


class MDCTCodecOrthonormalAnalysis(nn.Module):
    """Exact fixed MDCT front end used by the released MDCTCodec."""

    def __init__(self, num_coefficients: int = 40):
        super().__init__()
        self.num_coefficients = int(num_coefficients)
        self.hop_length = self.num_coefficients
        analysis, _ = _mdctcodec_orthonormal_kernels(self.num_coefficients)
        self.register_buffer("analysis_kernel", analysis, persistent=False)

    @property
    def dimension(self) -> int:
        return self.num_coefficients

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3 or int(waveform.size(1)) != 1:
            raise ValueError(
                "MDCTCodec analysis expects mono waveform [B, 1, T], got "
                f"{tuple(waveform.shape)}"
            )
        padded = F.pad(waveform.float(), (self.hop_length, self.hop_length))
        return F.conv1d(
            padded,
            self.analysis_kernel,
            stride=self.hop_length,
        )


class MDCTCodecOrthonormalSynthesis(nn.Module):
    """Exact overlap-add IMDCT paired with :class:`MDCTCodecOrthonormalAnalysis`."""

    def __init__(self, num_coefficients: int = 40):
        super().__init__()
        self.num_coefficients = int(num_coefficients)
        self.hop_length = self.num_coefficients
        _, synthesis = _mdctcodec_orthonormal_kernels(self.num_coefficients)
        self.register_buffer("synthesis_kernel", synthesis, persistent=False)

    def forward(self, coefficients: torch.Tensor) -> torch.Tensor:
        if coefficients.ndim != 3 or int(coefficients.size(1)) != self.num_coefficients:
            raise ValueError(
                "MDCTCodec synthesis expects [B, coefficients, frames], got "
                f"{tuple(coefficients.shape)}"
            )
        padded = F.conv_transpose1d(
            coefficients.float(),
            self.synthesis_kernel,
            stride=self.hop_length,
        )
        return padded[..., self.hop_length : -self.hop_length]


class MDCTCodecGRN(nn.Module):
    """ConvNeXt-v2 GRN with parameter names matching the released checkpoint."""

    def __init__(self, channels: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, int(channels)))
        self.beta = nn.Parameter(torch.zeros(1, 1, int(channels)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        response = torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)
        normalized = response / (response.mean(dim=-1, keepdim=True) + 1.0e-6)
        return self.gamma * (x * normalized) + self.beta + x


class MDCTCodecConvNeXtBlock(nn.Module):
    """The modified ConvNeXt-v2 block in the official MDCTCodec source."""

    def __init__(self, channels: int, intermediate_channels: int):
        super().__init__()
        channels = int(channels)
        intermediate_channels = int(intermediate_channels)
        self.dwconv = nn.Conv1d(
            channels, channels, kernel_size=7, padding=3, groups=channels
        )
        self.norm = nn.LayerNorm(channels, eps=1.0e-6)
        self.pwconv1 = nn.Linear(channels, intermediate_channels)
        self.act = nn.GELU()
        self.grn = MDCTCodecGRN(intermediate_channels)
        self.pwconv2 = nn.Linear(intermediate_channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x).transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x).transpose(1, 2)
        return residual + x


def _mdctcodec_trunc_normal_init_(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv1d, nn.Linear)):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _mdctcodec_conv_init_(module: nn.Module) -> None:
    if "Conv" in module.__class__.__name__:
        nn.init.normal_(module.weight, mean=0.0, std=0.01)


class MDCTCodecOfficialEncoder(nn.Module):
    """Published MDCTCodec encoder, with RVQ deliberately omitted.

    Attribute names mirror the authors' released checkpoint so its non-RVQ
    weights can be loaded as a structural regression test.  LASER is attached
    only after ``latent_output_conv`` by the parent model.
    """

    def __init__(
        self,
        analysis: MDCTCodecOrthonormalAnalysis,
        *,
        latent_dim: int = 32,
        hidden_channels: int = 256,
        intermediate_channels: int = 512,
        num_layers: int = 8,
        ratio: int = 8,
    ):
        super().__init__()
        self.analysis = analysis
        self.input_channels = int(analysis.num_coefficients)
        self.dim = int(hidden_channels)
        self.num_layers = int(num_layers)
        self.ratio = int(ratio)
        self.latent_dim = int(latent_dim)
        if self.ratio <= 0:
            raise ValueError("MDCTCodec ratio must be positive")

        self.embed_logamp = nn.Conv1d(
            self.input_channels, self.dim, kernel_size=7, padding=3
        )
        self.norm_logamp = nn.LayerNorm(self.dim, eps=1.0e-6)
        self.convnext_logamp = nn.ModuleList(
            [
                MDCTCodecConvNeXtBlock(self.dim, int(intermediate_channels))
                for _ in range(self.num_layers)
            ]
        )
        self.final_layer_norm_logamp = nn.LayerNorm(self.dim, eps=1.0e-6)
        self.apply(_mdctcodec_trunc_normal_init_)

        # The official source defines this layer after its trunc-normal pass,
        # so PyTorch's Linear initialization is part of the published recipe.
        self.out_logamp = nn.Linear(self.dim, self.dim)
        self.AMP_Encoder_downsample_output_conv = weight_norm(
            nn.Conv1d(self.dim, self.dim, kernel_size=7, stride=self.ratio, padding=3)
        )
        self.latent_output_conv = weight_norm(
            nn.Conv1d(self.dim, self.latent_dim, kernel_size=7, padding=3)
        )
        self.AMP_Encoder_downsample_output_conv.apply(_mdctcodec_conv_init_)
        self.latent_output_conv.apply(_mdctcodec_conv_init_)

    @property
    def dimension(self) -> int:
        return self.latent_dim

    @property
    def hop_length(self) -> int:
        return int(self.analysis.hop_length) * self.ratio

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        coefficients = self.analysis(waveform)
        if int(coefficients.size(-1)) % self.ratio != 0:
            raise ValueError(
                "MDCTCodec requires its padded MDCT frame count to be divisible "
                f"by ratio={self.ratio}; got {int(coefficients.size(-1))} frames"
            )
        hidden = self.embed_logamp(coefficients)
        hidden = self.norm_logamp(hidden.transpose(1, 2)).transpose(1, 2)
        for block in self.convnext_logamp:
            hidden = block(hidden)
        hidden = self.final_layer_norm_logamp(hidden.transpose(1, 2))
        hidden = self.out_logamp(hidden).transpose(1, 2)
        hidden = self.AMP_Encoder_downsample_output_conv(hidden)
        return self.latent_output_conv(hidden)


class MDCTCodecOfficialDecoder(nn.Module):
    """Released MDCTCodec decoder topology followed by its fixed IMDCT.

    The repository's current source accidentally omits ``embed_logamp`` from
    both construction and forward, while the released 200k-step checkpoint
    contains that trained layer.  The symmetric topology below is therefore
    reconstructed from the paper diagram and released state dictionary.
    """

    def __init__(
        self,
        synthesis: MDCTCodecOrthonormalSynthesis,
        *,
        latent_dim: int = 32,
        hidden_channels: int = 256,
        intermediate_channels: int = 512,
        num_layers: int = 8,
        ratio: int = 8,
    ):
        super().__init__()
        self.synthesis = synthesis
        self.latent_dim = int(latent_dim)
        self.dim = int(hidden_channels)
        self.num_layers = int(num_layers)
        self.ratio = int(ratio)
        num_coefficients = int(synthesis.num_coefficients)

        self.latent_input_conv = weight_norm(
            nn.Conv1d(self.latent_dim, self.dim, kernel_size=7, padding=3)
        )
        self.AMP_Decoder_upsample_input_conv = weight_norm(
            nn.ConvTranspose1d(
                self.dim,
                self.dim,
                kernel_size=2 * self.ratio,
                stride=self.ratio,
                padding=self.ratio // 2,
            )
        )
        self.embed_logamp = nn.Conv1d(self.dim, self.dim, kernel_size=7, padding=3)
        self.norm_logamp = nn.LayerNorm(self.dim, eps=1.0e-6)
        self.convnext_logamp = nn.ModuleList(
            [
                MDCTCodecConvNeXtBlock(self.dim, int(intermediate_channels))
                for _ in range(self.num_layers)
            ]
        )
        self.final_layer_norm_logamp = nn.LayerNorm(self.dim, eps=1.0e-6)
        self.apply(_mdctcodec_trunc_normal_init_)

        self.out_logamp = nn.Linear(self.dim, self.dim)
        self.PHA_Decoder_output_R_conv = weight_norm(
            nn.Conv1d(self.dim, num_coefficients, kernel_size=7, padding=3)
        )
        # Present in the official source/checkpoint but intentionally unused by
        # its real-valued MDCT forward path. Retaining it makes the shell exact.
        self.PHA_Decoder_output_I_conv = weight_norm(
            nn.Conv1d(self.dim, num_coefficients, kernel_size=7, padding=3)
        )
        self.PHA_Decoder_output_R_conv.apply(_mdctcodec_conv_init_)
        self.PHA_Decoder_output_I_conv.apply(_mdctcodec_conv_init_)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 3 or int(latent.size(1)) != self.latent_dim:
            raise ValueError(
                f"MDCTCodec decoder expects [B, {self.latent_dim}, frames], got "
                f"{tuple(latent.shape)}"
            )
        hidden = self.latent_input_conv(latent)
        hidden = self.AMP_Decoder_upsample_input_conv(hidden)
        hidden = self.embed_logamp(hidden)
        hidden = self.norm_logamp(hidden.transpose(1, 2)).transpose(1, 2)
        for block in self.convnext_logamp:
            hidden = block(hidden)
        hidden = self.final_layer_norm_logamp(hidden.transpose(1, 2))
        hidden = self.out_logamp(hidden).transpose(1, 2)
        coefficients = self.PHA_Decoder_output_R_conv(hidden)
        return self.synthesis(coefficients)


def build_mdctcodec_official_backbone(
    *,
    num_coefficients: int = 40,
    latent_dim: int = 32,
    hidden_channels: int = 256,
    intermediate_channels: int = 512,
    num_layers: int = 8,
    temporal_ratio: int = 8,
) -> tuple[MDCTCodecOfficialEncoder, MDCTCodecOfficialDecoder]:
    """Build MDCTCodec's encoder/decoder with only its RVQ slot empty."""
    analysis = MDCTCodecOrthonormalAnalysis(num_coefficients)
    synthesis = MDCTCodecOrthonormalSynthesis(num_coefficients)
    encoder = MDCTCodecOfficialEncoder(
        analysis,
        latent_dim=latent_dim,
        hidden_channels=hidden_channels,
        intermediate_channels=intermediate_channels,
        num_layers=num_layers,
        ratio=temporal_ratio,
    )
    decoder = MDCTCodecOfficialDecoder(
        synthesis,
        latent_dim=latent_dim,
        hidden_channels=hidden_channels,
        intermediate_channels=intermediate_channels,
        num_layers=num_layers,
        ratio=temporal_ratio,
    )
    return encoder, decoder


def build_mdct_vae_backbone(
    *,
    num_coefficients: int = 320,
    latent_dim: int = 128,
    hidden_channels: int = 256,
    residual_hidden_channels: int = 128,
    num_residual_layers: int = 6,
    dilation_cycle=(1, 3, 9),
    temporal_downsample_factor: int = 1,
    use_convnext_v2: bool = False,
    convnext_intermediate_channels: int = 512,
    sample_rate: int = META_ENCODEC_SAMPLE_RATE,
) -> tuple[MDCTVAEEncoder, MDCTVAEDecoder]:
    """Build a fixed-MDCT learned analysis/synthesis autoencoder."""
    analysis, synthesis = build_mdct_backbone(
        num_coefficients=num_coefficients,
        learnable_gain=False,
        sample_rate=sample_rate,
        num_log_bands=0,
        include_pitch=False,
    )
    encoder = MDCTVAEEncoder(
        analysis,
        latent_dim=latent_dim,
        hidden_channels=hidden_channels,
        residual_hidden_channels=residual_hidden_channels,
        num_residual_layers=num_residual_layers,
        dilation_cycle=dilation_cycle,
        temporal_downsample_factor=temporal_downsample_factor,
        use_convnext_v2=use_convnext_v2,
        convnext_intermediate_channels=convnext_intermediate_channels,
    )
    decoder = MDCTVAEDecoder(
        synthesis,
        latent_dim=latent_dim,
        num_coefficients=num_coefficients,
        hidden_channels=hidden_channels,
        residual_hidden_channels=residual_hidden_channels,
        num_residual_layers=num_residual_layers,
        dilation_cycle=dilation_cycle,
        temporal_upsample_factor=temporal_downsample_factor,
        use_convnext_v2=use_convnext_v2,
        convnext_intermediate_channels=convnext_intermediate_channels,
    )
    return encoder, decoder


class AudioEncoder(nn.Module):
    """SEANet/SoundStream-inspired 1D residual strided convolution encoder."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
        downsample_rates=(4, 4, 4),
        dilation_cycle=(1, 3, 9),
    ):
        super().__init__()
        rates = canonical_int_tuple(downsample_rates, default=(4, 4, 4))
        num_hiddens = int(num_hiddens)
        self.downsample_rates = rates
        self.conv_in = nn.Conv1d(int(in_channels), num_hiddens, kernel_size=7, padding=3)
        self.blocks = nn.ModuleList()
        for rate in rates:
            if rate % 2 != 0:
                raise ValueError(f"Audio downsample rates must be even for exact shape recovery, got {rates}")
            self.blocks.append(
                nn.Sequential(
                    AudioResidualStack(
                        num_hiddens,
                        int(num_residual_hiddens),
                        num_layers=int(num_residual_layers),
                        dilation_cycle=dilation_cycle,
                    ),
                    nn.SiLU(),
                    nn.Conv1d(
                        num_hiddens,
                        num_hiddens,
                        kernel_size=2 * int(rate),
                        stride=int(rate),
                        padding=int(rate) // 2,
                    ),
                )
            )
        self.conv_out = nn.Conv1d(num_hiddens, num_hiddens, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for block in self.blocks:
            x = block(x)
        return self.conv_out(F.silu(x))


class AudioDecoder(nn.Module):
    """1D residual decoder matching :class:`AudioEncoder` downsampling rates."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
        out_channels: int,
        upsample_rates=(4, 4, 4),
        dilation_cycle=(1, 3, 9),
    ):
        super().__init__()
        rates = canonical_int_tuple(upsample_rates, default=(4, 4, 4))
        num_hiddens = int(num_hiddens)
        self.upsample_rates = rates
        self.conv_in = nn.Conv1d(int(in_channels), num_hiddens, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList()
        for rate in reversed(rates):
            if rate % 2 != 0:
                raise ValueError(f"Audio upsample rates must be even for exact shape recovery, got {rates}")
            self.blocks.append(
                nn.Sequential(
                    AudioResidualStack(
                        num_hiddens,
                        int(num_residual_hiddens),
                        num_layers=int(num_residual_layers),
                        dilation_cycle=dilation_cycle,
                    ),
                    nn.SiLU(),
                    nn.ConvTranspose1d(
                        num_hiddens,
                        num_hiddens,
                        kernel_size=2 * int(rate),
                        stride=int(rate),
                        padding=int(rate) // 2,
                    ),
                )
            )
        self.res_out = AudioResidualStack(
            num_hiddens,
            int(num_residual_hiddens),
            num_layers=max(1, int(num_residual_layers)),
            dilation_cycle=dilation_cycle,
        )
        self.conv_out = nn.Conv1d(num_hiddens, int(out_channels), kernel_size=7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for block in self.blocks:
            x = block(x)
        x = self.res_out(x)
        return self.conv_out(F.silu(x))
