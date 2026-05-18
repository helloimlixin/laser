from __future__ import annotations

import csv
from functools import lru_cache
import importlib.util
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Optional, Sequence
import uuid
import wave

import numpy as np
import torch
import torch.nn.functional as F
from src.data.vctk import (
    _apply_edge_fade,
    _normalize_pcm,
    _preprocess_waveform,
    _read_audio_file,
    _resample_if_needed,
    _rms_normalize_waveform,
)


AUDIO_META_KEYS = (
    "path",
    "crop_mode",
    "crop_offset",
    "source_num_samples",
    "spec_min",
    "spec_max",
    "spec_shape",
)

# Datasets for which the audio logging / metrics path is wired and validated.
# Historically this was a single "vctk" check baked into every function below,
# which silently dropped maestro audio. Add new audio datasets here once their
# DataModule produces the AUDIO_META_KEYS schema and the loss-side normalization
# is known to be in [-1, 1].
_AUDIO_DATASETS_WITH_LOGGING = frozenset({"vctk", "maestro"})


def _dataset_supports_audio_logging(config: Mapping[str, Any]) -> bool:
    name = str(config.get("dataset", "") or "").strip().lower()
    return name in _AUDIO_DATASETS_WITH_LOGGING

_VISQOL_AUDIO_MODEL = "libsvm_nu_svr_model.txt"
_VISQOL_SPEECH_MODEL = (
    "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
)


def _get_value(source: Any, key: str, default=None):
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def audio_config_from_source(source: Any) -> dict:
    win_length = _get_value(source, "stft_win_length", None)
    n_fft = int(_get_value(source, "stft_n_fft", 1024))
    return {
        "dataset": str(_get_value(source, "dataset", "") or ""),
        "mean": tuple(float(v) for v in (_get_value(source, "mean", (0.5,)) or (0.5,))),
        "std": tuple(float(v) for v in (_get_value(source, "std", (0.5,)) or (0.5,))),
        "sample_rate": int(_get_value(source, "sample_rate", 16000)),
        "audio_num_samples": int(_get_value(source, "audio_num_samples", 32768)),
        "audio_representation": str(_get_value(source, "audio_representation", "spectrogram") or "spectrogram"),
        "stft_n_fft": n_fft,
        "stft_hop_length": int(_get_value(source, "stft_hop_length", 256)),
        "stft_win_length": int(n_fft if win_length in (None, 0) else win_length),
        "stft_power": float(_get_value(source, "stft_power", 2.0)),
        "stft_log_offset": float(_get_value(source, "stft_log_offset", 1.0e-5)),
        "griffin_lim_iters": int(_get_value(source, "griffin_lim_iters", 16)),
        "mel_bins": int(_get_value(source, "mel_bins", 80)),
        "audio_dc_remove": bool(_get_value(source, "audio_dc_remove", False)),
        "audio_peak_normalize": bool(_get_value(source, "audio_peak_normalize", False)),
        "audio_target_peak": float(_get_value(source, "audio_target_peak", 0.95)),
        "audio_rms_normalize": bool(_get_value(source, "audio_rms_normalize", False)),
        "audio_target_rms": float(_get_value(source, "audio_target_rms", 0.12)),
        "audio_max_gain": float(_get_value(source, "audio_max_gain", 8.0)),
        "audio_min_crop_rms": float(_get_value(source, "audio_min_crop_rms", 0.0)),
        "audio_crop_attempts": int(_get_value(source, "audio_crop_attempts", 1)),
        "audio_fade_samples": int(_get_value(source, "audio_fade_samples", 0)),
    }


def extract_audio_metadata_from_batch(batch) -> Optional[dict]:
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        return None
    candidate = batch[-1]
    if not isinstance(candidate, Mapping):
        return None
    if not all(key in candidate for key in AUDIO_META_KEYS):
        return None
    return dict(candidate)


def has_audio_metadata(meta: Optional[Mapping[str, Any]]) -> bool:
    return isinstance(meta, Mapping) and all(key in meta for key in AUDIO_META_KEYS)


def _audio_format(meta: Optional[Mapping[str, Any]], inputs: Optional[torch.Tensor] = None) -> str:
    raw = None if not isinstance(meta, Mapping) else meta.get("audio_format")
    if isinstance(raw, (list, tuple)) and raw:
        raw = raw[0]
    if torch.is_tensor(raw):
        raw = None
    text = str(raw or "").strip().lower()
    if text:
        return text
    if torch.is_tensor(inputs) and inputs.ndim == 3:
        return "waveform"
    return "spectrogram"


def _is_waveform_batch(inputs: torch.Tensor, meta: Optional[Mapping[str, Any]] = None) -> bool:
    return torch.is_tensor(inputs) and inputs.ndim == 3 and _audio_format(meta, inputs) == "waveform"


def _slice_audio_meta(meta: Mapping[str, Any], limit: int) -> dict:
    out = {}
    for key in AUDIO_META_KEYS:
        value = meta[key]
        if torch.is_tensor(value):
            out[key] = value[:limit].detach().cpu()
        elif isinstance(value, np.ndarray):
            out[key] = value[:limit]
        elif isinstance(value, (list, tuple)):
            out[key] = list(value[:limit])
        else:
            out[key] = value
    return out


def _meta_item(meta: Mapping[str, Any], index: int) -> dict:
    item = {}
    for key in AUDIO_META_KEYS:
        value = meta[key]
        if torch.is_tensor(value):
            value = value[index]
            item[key] = value.detach().cpu()
        elif isinstance(value, np.ndarray):
            item[key] = value[index]
        elif isinstance(value, (list, tuple)):
            item[key] = value[index]
        else:
            item[key] = value
    return item


def _scalar_int(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.item())
    if isinstance(value, np.ndarray):
        return int(np.asarray(value).item())
    return int(value)


def _scalar_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.item())
    if isinstance(value, np.ndarray):
        return float(np.asarray(value).item())
    return float(value)


def _tensor_1d(value: Any, *, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().cpu().to(dtype=dtype).reshape(-1)
    return torch.as_tensor(value, dtype=dtype).reshape(-1)


def _tensor_2d(value: Any, *, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().cpu().to(dtype=dtype).reshape(-1, 2)
    return torch.as_tensor(value, dtype=dtype).reshape(-1, 2)


def _load_cropped_waveform(meta_item: Mapping[str, Any], config: Mapping[str, Any]) -> torch.Tensor:
    path = Path(str(meta_item["path"])).expanduser()
    sample_rate, samples = _read_audio_file(path)
    waveform = _normalize_pcm(samples)
    waveform = _resample_if_needed(waveform, int(sample_rate), int(config["sample_rate"]))
    waveform = _preprocess_waveform(
        waveform,
        dc_remove=bool(config.get("audio_dc_remove", False)),
        peak_normalize=bool(config.get("audio_peak_normalize", False)),
        target_peak=float(config.get("audio_target_peak", 0.95)),
    )
    waveform = torch.from_numpy(waveform.astype(np.float32, copy=False))

    target = int(config["audio_num_samples"])
    crop_mode = _scalar_int(meta_item["crop_mode"])
    crop_offset = max(0, _scalar_int(meta_item["crop_offset"]))
    source_num_samples = max(0, _scalar_int(meta_item["source_num_samples"]))
    if source_num_samples > 0 and waveform.numel() > source_num_samples:
        waveform = waveform[:source_num_samples]

    if crop_mode == 0:
        start = min(crop_offset, max(0, int(waveform.numel()) - target))
        clipped = waveform[start:start + target]
        if clipped.numel() < target:
            clipped = F.pad(clipped, (0, target - int(clipped.numel())))
        return _rms_normalize_waveform(
            _apply_edge_fade(clipped.to(torch.float32), int(config.get("audio_fade_samples", 0))),
            enabled=bool(config.get("audio_rms_normalize", False)),
            target_rms=float(config.get("audio_target_rms", 0.12)),
            max_gain=float(config.get("audio_max_gain", 8.0)),
            peak_limit=float(config.get("audio_target_peak", 0.95)),
        )

    padded = torch.zeros(target, dtype=torch.float32)
    usable = waveform[: min(int(waveform.numel()), target)].to(torch.float32)
    start = min(crop_offset, max(0, target - int(usable.numel())))
    padded[start:start + int(usable.numel())] = usable
    return _rms_normalize_waveform(
        _apply_edge_fade(padded, int(config.get("audio_fade_samples", 0))),
        enabled=bool(config.get("audio_rms_normalize", False)),
        target_rms=float(config.get("audio_target_rms", 0.12)),
        max_gain=float(config.get("audio_max_gain", 8.0)),
        peak_limit=float(config.get("audio_target_peak", 0.95)),
    )


def _normalized_to_unit(spec: torch.Tensor, config: Mapping[str, Any]) -> torch.Tensor:
    mean = torch.tensor(config["mean"], dtype=spec.dtype, device=spec.device).view(-1, 1, 1)
    std = torch.tensor(config["std"], dtype=spec.dtype, device=spec.device).view(-1, 1, 1)
    return (spec * std + mean).clamp(0.0, 1.0)


def _logmag_and_magnitude(
    spec: torch.Tensor,
    meta_item: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    unit = _normalized_to_unit(spec, config)
    if int(unit.size(0)) != 1:
        raise ValueError(f"Audio logging expects a single-channel spectrogram, got {tuple(unit.shape)}")

    spec_min = _scalar_float(meta_item["spec_min"])
    spec_max = _scalar_float(meta_item["spec_max"])
    logmag = unit[0] * (spec_max - spec_min) + spec_min

    spec_shape = _tensor_1d(meta_item["spec_shape"], dtype=torch.int64)
    original_hw = (int(spec_shape[0].item()), int(spec_shape[1].item()))
    if tuple(logmag.shape) != original_hw:
        logmag = F.interpolate(
            logmag.view(1, 1, *logmag.shape),
            size=original_hw,
            mode="bilinear",
            align_corners=False,
        ).view(*original_hw)

    magnitude = torch.exp(logmag).sub(float(config["stft_log_offset"])).clamp_min(0.0)
    power = float(config["stft_power"])
    if abs(power - 1.0) > 1e-6:
        magnitude = magnitude.pow(1.0 / power)
    return logmag.to(torch.float32), magnitude.to(torch.float32)


def _griffin_lim(
    magnitude: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
    length: int,
    num_iters: int,
) -> torch.Tensor:
    magnitude = magnitude.to(torch.float32)
    window = torch.hann_window(win_length, periodic=True, dtype=magnitude.dtype, device=magnitude.device)
    angles = torch.exp(2j * torch.pi * torch.rand_like(magnitude))
    complex_spec = magnitude.to(torch.complex64) * angles.to(torch.complex64)
    waveform = torch.istft(
        complex_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        length=length,
    )

    for _ in range(max(1, int(num_iters))):
        rebuilt = torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            return_complex=True,
        )
        phase = rebuilt / rebuilt.abs().clamp_min(1.0e-8)
        complex_spec = magnitude.to(torch.complex64) * phase.to(torch.complex64)
        waveform = torch.istft(
            complex_spec,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            length=length,
        )
    return waveform.clamp(-1.0, 1.0).to(torch.float32)


def _hz_to_mel(freq_hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + freq_hz / 700.0)


def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)


def _mel_filterbank(*, sample_rate: int, n_fft: int, n_mels: int) -> torch.Tensor:
    n_freqs = n_fft // 2 + 1
    hz = np.linspace(0.0, sample_rate / 2.0, n_freqs, dtype=np.float32)
    mel_edges = np.linspace(_hz_to_mel(np.array([0.0], dtype=np.float32))[0], _hz_to_mel(np.array([sample_rate / 2.0], dtype=np.float32))[0], n_mels + 2)
    hz_edges = _mel_to_hz(mel_edges)
    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for mel_idx in range(n_mels):
        left, center, right = hz_edges[mel_idx : mel_idx + 3]
        if center <= left or right <= center:
            continue
        rising = (hz - left) / max(center - left, 1.0e-8)
        falling = (right - hz) / max(right - center, 1.0e-8)
        fb[mel_idx] = np.maximum(0.0, np.minimum(rising, falling))
    return torch.from_numpy(fb)


def _mel_db(waveform: torch.Tensor, config: Mapping[str, Any]) -> np.ndarray:
    n_fft = int(config["stft_n_fft"])
    hop = int(config["stft_hop_length"])
    win = int(config["stft_win_length"])
    window = torch.hann_window(win, periodic=True, dtype=waveform.dtype)
    spec = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        window=window,
        center=True,
        return_complex=True,
    ).abs()
    mel_fb = _mel_filterbank(
        sample_rate=int(config["sample_rate"]),
        n_fft=n_fft,
        n_mels=int(config["mel_bins"]),
    ).to(spec.dtype)
    mel = mel_fb @ spec
    mel = torch.log10(mel.clamp_min(1.0e-5))
    mel = mel - mel.min()
    mel = mel / mel.max().clamp_min(1.0e-6)
    return mel.detach().cpu().numpy()


@lru_cache(maxsize=1)
def _resolve_visqol_binary() -> Optional[str]:
    for env_key in ("VISQOL_BINARY", "VISQOL_BIN"):
        raw = os.environ.get(env_key)
        if not raw:
            continue
        candidate = Path(raw).expanduser()
        if candidate.exists():
            return str(candidate)
    candidate = shutil.which("visqol")
    return candidate if candidate else None


@lru_cache(maxsize=1)
def _has_visqol_python_module() -> bool:
    return importlib.util.find_spec("visqol") is not None


def _visqol_mode(sample_rate: int) -> str:
    return "speech" if int(sample_rate) <= 16000 else "audio"


def _measure_visqol_python(
    reference_waveform: torch.Tensor,
    degraded_waveform: torch.Tensor,
    *,
    sample_rate: int,
) -> Optional[float]:
    from visqol import visqol_lib_py
    from visqol.pb2 import visqol_config_pb2

    config = visqol_config_pb2.VisqolConfig()
    config.audio.sample_rate = int(sample_rate)
    mode = _visqol_mode(sample_rate)
    if mode == "speech":
        config.options.use_speech_scoring = True
        model_name = _VISQOL_SPEECH_MODEL
    else:
        config.options.use_speech_scoring = False
        model_name = _VISQOL_AUDIO_MODEL
    config.options.svr_model_path = os.path.join(
        os.path.dirname(visqol_lib_py.__file__),
        "model",
        model_name,
    )

    api = visqol_lib_py.VisqolApi()
    api.Create(config)
    result = api.Measure(
        np.asarray(reference_waveform.detach().cpu().numpy(), dtype=np.float64),
        np.asarray(degraded_waveform.detach().cpu().numpy(), dtype=np.float64),
    )
    score = float(result.moslqo)
    if not np.isfinite(score):
        return None
    return score


def _measure_visqol_cli(
    reference_waveform: torch.Tensor,
    degraded_waveform: torch.Tensor,
    *,
    sample_rate: int,
    binary: str,
) -> Optional[float]:
    with tempfile.TemporaryDirectory(prefix="visqol_") as tmpdir:
        root = Path(tmpdir)
        ref_path = _write_wav_audio_file(root, "reference", reference_waveform, sample_rate=sample_rate)
        deg_path = _write_wav_audio_file(root, "degraded", degraded_waveform, sample_rate=sample_rate)
        input_csv = root / "pairs.csv"
        output_csv = root / "results.csv"
        with input_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("reference", "degraded"))
            writer.writerow((str(ref_path), str(deg_path)))

        cmd = [
            str(binary),
            "--batch_input_csv",
            str(input_csv),
            "--results_csv",
            str(output_csv),
        ]
        if _visqol_mode(sample_rate) == "speech":
            cmd.append("--use_speech_mode")
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=120,
        )
        if result.returncode != 0 or not output_csv.exists():
            return None
        with output_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            row = next(reader, None)
        if not row:
            return None
        try:
            score = float(row["moslqo"])
        except (KeyError, TypeError, ValueError):
            return None
        return score if np.isfinite(score) else None


def _measure_visqol(
    reference_waveform: torch.Tensor,
    degraded_waveform: torch.Tensor,
    *,
    sample_rate: int,
) -> Optional[float]:
    if _has_visqol_python_module():
        try:
            return _measure_visqol_python(
                reference_waveform,
                degraded_waveform,
                sample_rate=sample_rate,
            )
        except Exception:
            pass
    binary = _resolve_visqol_binary()
    if not binary:
        return None
    try:
        return _measure_visqol_cli(
            reference_waveform,
            degraded_waveform,
            sample_rate=sample_rate,
            binary=binary,
        )
    except Exception:
        return None


def _canonical_stft_sizes(values: Optional[Sequence[int]], default: Sequence[int]) -> tuple[int, ...]:
    if values is None:
        values = default
    if isinstance(values, str):
        raw = values.strip()
        if raw.startswith("[") or raw.startswith("("):
            import ast

            values = ast.literal_eval(raw)
        elif raw:
            values = [part for part in raw.split(",") if part.strip()]
        else:
            values = default
    out = tuple(int(value) for value in values)
    if not out or any(value <= 0 for value in out):
        raise ValueError(f"Expected positive STFT sizes, got {out}")
    return out


def _stft_magnitude_batch(
    waveform: torch.Tensor,
    *,
    n_fft: int,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
) -> torch.Tensor:
    if waveform.ndim == 3:
        if int(waveform.size(1)) != 1:
            raise ValueError(f"Expected mono waveform [B, 1, T], got {tuple(waveform.shape)}")
        waveform = waveform[:, 0, :]
    elif waveform.ndim != 2:
        raise ValueError(f"Expected waveform [B, T] or [B, 1, T], got {tuple(waveform.shape)}")
    hop_length = int(hop_length or max(1, n_fft // 4))
    win_length = int(win_length or n_fft)
    window = torch.hann_window(win_length, periodic=True, dtype=waveform.dtype, device=waveform.device)
    return torch.stft(
        waveform,
        n_fft=int(n_fft),
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        # Reflect padding has no deterministic CUDA backward for 1D tensors.
        # Constant padding is stable for both metrics and differentiable losses.
        pad_mode="constant",
        return_complex=True,
    ).abs()


def _waveform_spectral_convergence(
    target_mag: torch.Tensor,
    recon_mag: torch.Tensor,
    *,
    eps: float = 1.0e-7,
    magnitude_floor: float = 1.0e-3,
    max_value: float = 10.0,
) -> torch.Tensor:
    target_flat = target_mag.reshape(int(target_mag.size(0)), -1).float()
    diff_flat = (recon_mag - target_mag).reshape(int(target_mag.size(0)), -1).float()
    diff_norm = torch.linalg.vector_norm(diff_flat, dim=1)
    target_norm = torch.linalg.vector_norm(target_flat, dim=1)
    floor = math.sqrt(max(1, int(target_flat.size(1)))) * float(magnitude_floor)
    denom = target_norm.clamp_min(max(float(eps), floor))
    return (diff_norm / denom).clamp_max(float(max_value)).mean()


def compute_waveform_multires_stft_loss(
    inputs: torch.Tensor,
    reconstructions: torch.Tensor,
    *,
    fft_sizes: Optional[Sequence[int]] = None,
    hop_lengths: Optional[Sequence[int]] = None,
    win_lengths: Optional[Sequence[int]] = None,
) -> dict[str, torch.Tensor]:
    """Differentiable multi-resolution STFT reconstruction loss for waveform batches."""
    if not torch.is_tensor(inputs) or not torch.is_tensor(reconstructions):
        return {}
    if inputs.ndim != 3 or reconstructions.ndim != 3 or int(inputs.size(1)) != 1 or int(reconstructions.size(1)) != 1:
        return {}

    fft_sizes = _canonical_stft_sizes(fft_sizes, (512, 1024, 2048))
    if hop_lengths is None:
        hop_lengths = tuple(max(1, size // 4) for size in fft_sizes)
    else:
        hop_lengths = _canonical_stft_sizes(hop_lengths, tuple(max(1, size // 4) for size in fft_sizes))
    if win_lengths is None:
        win_lengths = fft_sizes
    else:
        win_lengths = _canonical_stft_sizes(win_lengths, fft_sizes)
    if not (len(fft_sizes) == len(hop_lengths) == len(win_lengths)):
        raise ValueError(
            "fft_sizes, hop_lengths, and win_lengths must have the same length "
            f"(got {len(fft_sizes)}, {len(hop_lengths)}, {len(win_lengths)})"
        )

    eps = 1.0e-7
    spectral_terms = []
    logmag_terms = []
    linmag_terms = []
    result_dtype = inputs.dtype if torch.is_floating_point(inputs) else torch.float32
    recon = reconstructions.to(dtype=torch.float32)
    target = inputs.to(dtype=torch.float32)
    for n_fft, hop, win in zip(fft_sizes, hop_lengths, win_lengths):
        target_mag = _stft_magnitude_batch(target, n_fft=n_fft, hop_length=hop, win_length=win)
        recon_mag = _stft_magnitude_batch(recon, n_fft=n_fft, hop_length=hop, win_length=win)
        spectral_terms.append(
            _waveform_spectral_convergence(
                target_mag,
                recon_mag,
                eps=eps,
            )
        )
        logmag_terms.append(F.l1_loss(torch.log(recon_mag.clamp_min(eps)), torch.log(target_mag.clamp_min(eps))))
        linmag_terms.append(F.l1_loss(recon_mag, target_mag))

    spectral_loss = torch.stack(spectral_terms).mean().to(dtype=result_dtype)
    logmag_loss = torch.stack(logmag_terms).mean().to(dtype=result_dtype)
    linmag_loss = torch.stack(linmag_terms).mean().to(dtype=result_dtype)
    total = spectral_loss + logmag_loss + 0.1 * linmag_loss
    return {
        "audio_multires_stft_loss": total,
        "audio_multires_stft_spectral_convergence": spectral_loss,
        "audio_multires_stft_logmag_l1": logmag_loss,
        "audio_multires_stft_mag_l1": linmag_loss,
    }


def _compute_waveform_reconstruction_metrics(
    inputs: torch.Tensor,
    reconstructions: torch.Tensor,
    *,
    config: Mapping[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    target = inputs.detach().to(torch.float32)
    recon = reconstructions.detach().to(torch.float32)
    if target.ndim != 3 or recon.ndim != 3 or int(target.size(1)) != 1 or int(recon.size(1)) != 1:
        return {}
    limit = min(int(target.size(0)), int(recon.size(0)))
    if limit <= 0:
        return {}
    target = target[:limit].cpu()
    recon = recon[:limit].cpu()
    diff = recon - target
    eps = 1.0e-8
    waveform_mse = diff.pow(2).mean()
    waveform_l1 = diff.abs().mean()
    signal = target.pow(2).mean(dim=(1, 2))
    noise = diff.pow(2).mean(dim=(1, 2)).clamp_min(eps)
    snr = (10.0 * torch.log10(signal.clamp_min(eps) / noise)).mean()
    base_fft = int(config["stft_n_fft"])
    stft_metrics = compute_waveform_multires_stft_loss(
        target,
        recon,
        fft_sizes=(max(16, base_fft // 2), base_fft, base_fft * 2),
    )
    logmag_l1 = stft_metrics.get("audio_multires_stft_logmag_l1", torch.tensor(0.0))
    spectral_convergence = stft_metrics.get(
        "audio_multires_stft_spectral_convergence",
        torch.tensor(0.0),
    )
    lsd = logmag_l1.to(torch.float32)
    return {
        "audio_lsd": lsd.to(device=device, dtype=dtype),
        "audio_log_spectral_distance": lsd.to(device=device, dtype=dtype),
        "audio_waveform_mse": waveform_mse.to(device=device, dtype=dtype),
        "audio_waveform_l1": waveform_l1.to(device=device, dtype=dtype),
        "audio_snr_db": snr.to(device=device, dtype=dtype),
        "audio_logmag_l1": logmag_l1.to(device=device, dtype=dtype),
        "audio_spectral_convergence": spectral_convergence.to(device=device, dtype=dtype),
    }


def compute_audio_reconstruction_metrics(
    inputs: torch.Tensor,
    reconstructions: torch.Tensor,
    *,
    audio_meta: Mapping[str, Any],
    audio_source: Any,
    compute_visqol: bool = False,
) -> dict:
    """Compute audio-domain reconstruction metrics for normalized log-spectrogram batches."""
    if not has_audio_metadata(audio_meta):
        return {}

    config = audio_config_from_source(audio_source)
    if not _dataset_supports_audio_logging(config):
        return {}

    if _is_waveform_batch(inputs, audio_meta):
        dtype = inputs.dtype if torch.is_floating_point(inputs) else torch.float32
        return _compute_waveform_reconstruction_metrics(
            inputs,
            reconstructions,
            config=config,
            device=inputs.device,
            dtype=dtype,
        )

    limit = min(int(inputs.size(0)), int(reconstructions.size(0)), len(audio_meta["path"]))
    if limit <= 0:
        return {}

    device = inputs.device
    dtype = inputs.dtype if torch.is_floating_point(inputs) else torch.float32
    inputs_cpu = inputs[:limit].detach().cpu().to(torch.float32)
    recon_cpu = reconstructions[:limit].detach().cpu().to(torch.float32)
    meta = _slice_audio_meta(audio_meta, limit)
    mel_fb = _mel_filterbank(
        sample_rate=int(config["sample_rate"]),
        n_fft=int(config["stft_n_fft"]),
        n_mels=int(config["mel_bins"]),
    ).to(torch.float32)
    log_offset = float(config["stft_log_offset"])
    should_compute_visqol = bool(compute_visqol) and (
        _has_visqol_python_module() or _resolve_visqol_binary() is not None
    )

    logmag_mses = []
    log_spectral_distances = []
    logmag_l1s = []
    spectral_convergences = []
    logmel_l1s = []
    visqol_scores = []
    with torch.no_grad():
        for idx in range(limit):
            item = _meta_item(meta, idx)
            try:
                original_logmag, original_magnitude = _logmag_and_magnitude(inputs_cpu[idx], item, config)
                recon_logmag, recon_magnitude = _logmag_and_magnitude(recon_cpu[idx], item, config)
            except Exception:
                continue

            logmag_diff = recon_logmag - original_logmag
            logmag_mses.append(logmag_diff.pow(2).mean())
            log_spectral_distances.append(logmag_diff.pow(2).mean().sqrt())
            logmag_l1s.append(logmag_diff.abs().mean())
            spectral_convergences.append(
                torch.linalg.vector_norm(recon_magnitude - original_magnitude)
                / torch.linalg.vector_norm(original_magnitude).clamp_min(1.0e-8)
            )
            original_logmel = torch.log((mel_fb @ original_magnitude).clamp_min(log_offset))
            recon_logmel = torch.log((mel_fb @ recon_magnitude).clamp_min(log_offset))
            logmel_l1s.append((recon_logmel - original_logmel).abs().mean())
            if should_compute_visqol:
                try:
                    original_waveform = _load_cropped_waveform(item, config)
                    recon_waveform = _griffin_lim(
                        recon_magnitude,
                        n_fft=int(config["stft_n_fft"]),
                        hop_length=int(config["stft_hop_length"]),
                        win_length=int(config["stft_win_length"]),
                        length=int(original_waveform.numel()),
                        num_iters=int(config["griffin_lim_iters"]),
                    )
                    visqol_score = _measure_visqol(
                        original_waveform,
                        recon_waveform,
                        sample_rate=int(config["sample_rate"]),
                    )
                except Exception:
                    visqol_score = None
                if visqol_score is not None:
                    visqol_scores.append(torch.tensor(float(visqol_score), dtype=torch.float32))

    if not logmag_mses:
        return {}

    def _mean_tensor(values):
        return torch.stack(values).mean().to(device=device, dtype=dtype)

    mean_lsd = _mean_tensor(log_spectral_distances)
    metrics = {
        "audio_lsd": mean_lsd,
        "audio_log_spectral_distance": mean_lsd,
        "audio_logmag_mse": _mean_tensor(logmag_mses),
        "audio_logmag_l1": _mean_tensor(logmag_l1s),
        "audio_spectral_convergence": _mean_tensor(spectral_convergences),
        "audio_logmel_l1": _mean_tensor(logmel_l1s),
    }
    if visqol_scores:
        metrics["audio_visqol"] = _mean_tensor(visqol_scores)
    return metrics


def _audio_logmel_feature(waveform: torch.Tensor, config: Mapping[str, Any]) -> torch.Tensor:
    mel = torch.as_tensor(_mel_db(waveform.to(torch.float32), config), dtype=torch.float32)
    if mel.ndim != 2 or mel.numel() == 0:
        raise ValueError("Expected a non-empty [mel, time] feature map")
    means = mel.mean(dim=1)
    stds = mel.std(dim=1, unbiased=False)
    if mel.size(1) > 1:
        dynamics = (mel[:, 1:] - mel[:, :-1]).abs().mean(dim=1)
    else:
        dynamics = torch.zeros_like(means)
    return torch.cat([means, stds, dynamics], dim=0)


def _diag_frechet_distance(real_features: torch.Tensor, generated_features: torch.Tensor) -> torch.Tensor:
    real_mean = real_features.mean(dim=0)
    generated_mean = generated_features.mean(dim=0)
    if real_features.size(0) > 1:
        real_var = real_features.var(dim=0, unbiased=False)
    else:
        real_var = torch.zeros_like(real_mean)
    if generated_features.size(0) > 1:
        generated_var = generated_features.var(dim=0, unbiased=False)
    else:
        generated_var = torch.zeros_like(generated_mean)
    mean_term = (real_mean - generated_mean).pow(2).sum()
    std_term = (real_var.clamp_min(0.0).sqrt() - generated_var.clamp_min(0.0).sqrt()).pow(2).sum()
    return mean_term + std_term


def compute_audio_generation_metrics(
    generated: torch.Tensor,
    *,
    audio_source: Any,
    audio_meta: Optional[Mapping[str, Any]] = None,
    max_items: int = 16,
) -> dict:
    """Compute a lightweight FAD-style log-mel perceptual proxy for generated VCTK audio."""
    config = audio_config_from_source(audio_source)
    if not _dataset_supports_audio_logging(config):
        return {}
    if not has_audio_metadata(audio_meta):
        return {}
    is_waveform = torch.is_tensor(generated) and generated.ndim == 3 and int(generated.size(1)) == 1
    is_spectrogram = torch.is_tensor(generated) and generated.ndim == 4 and int(generated.size(1)) == 1
    if not (is_waveform or is_spectrogram):
        return {}

    limit = min(int(max_items), int(generated.size(0)), len(audio_meta["path"]))
    if limit <= 0:
        return {}

    device = generated.device
    dtype = generated.dtype if torch.is_floating_point(generated) else torch.float32
    generated_cpu = generated[:limit].detach().cpu().to(torch.float32)
    meta = _slice_audio_meta(audio_meta, limit)
    generated_spec_item = _representative_audio_spec_item(meta, config)
    generated_features = []
    real_features = []

    with torch.no_grad():
        for idx in range(limit):
            real_item = _meta_item(meta, idx)
            try:
                if is_waveform:
                    generated_waveform = generated_cpu[idx, 0].clamp(-1.0, 1.0)
                else:
                    _, generated_magnitude = _logmag_and_magnitude(generated_cpu[idx], generated_spec_item, config)
                    generated_waveform = _griffin_lim(
                        generated_magnitude,
                        n_fft=int(config["stft_n_fft"]),
                        hop_length=int(config["stft_hop_length"]),
                        win_length=int(config["stft_win_length"]),
                        length=int(config["audio_num_samples"]),
                        num_iters=int(config["griffin_lim_iters"]),
                    )
                real_waveform = _load_cropped_waveform(real_item, config)
                generated_features.append(_audio_logmel_feature(generated_waveform, config))
                real_features.append(_audio_logmel_feature(real_waveform, config))
            except Exception:
                continue

    if not generated_features or not real_features:
        return {}

    generated_stack = torch.stack(generated_features, dim=0)
    real_stack = torch.stack(real_features, dim=0)
    frechet = _diag_frechet_distance(real_stack, generated_stack).to(device=device, dtype=dtype)
    mean_l1 = (real_stack.mean(dim=0) - generated_stack.mean(dim=0)).abs().mean().to(device=device, dtype=dtype)
    return {
        "audio_generation_logmel_frechet": frechet,
        "audio_generation_logmel_mean_l1": mean_l1,
    }


def compute_audio_energy_matching_loss(
    inputs: torch.Tensor,
    reconstructions: torch.Tensor,
    *,
    audio_meta: Mapping[str, Any],
    audio_source: Any,
    frame_weight: float = 1.0,
    global_weight: float = 0.5,
) -> dict:
    """Compute a differentiable energy-matching loss in magnitude-spectrogram space.

    This penalizes systematic loudness collapse without requiring waveform phase
    reconstruction in the training graph.
    """
    if not has_audio_metadata(audio_meta):
        return {}

    config = audio_config_from_source(audio_source)
    if not _dataset_supports_audio_logging(config):
        return {}
    if _is_waveform_batch(inputs, audio_meta):
        return {}

    limit = min(int(inputs.size(0)), int(reconstructions.size(0)), len(audio_meta["path"]))
    if limit <= 0:
        return {}

    eps = 1.0e-8
    device = reconstructions.device
    dtype = reconstructions.dtype if torch.is_floating_point(reconstructions) else torch.float32
    frame_losses = []
    global_losses = []
    rms_ratios = []

    for idx in range(limit):
        item = _meta_item(audio_meta, idx)
        try:
            _, original_magnitude = _logmag_and_magnitude(inputs[idx], item, config)
            _, recon_magnitude = _logmag_and_magnitude(reconstructions[idx], item, config)
        except Exception:
            continue

        original_power = original_magnitude.pow(2).clamp_min(0.0)
        recon_power = recon_magnitude.pow(2).clamp_min(0.0)

        original_frame_log_energy = torch.log(original_power.mean(dim=0).clamp_min(eps))
        recon_frame_log_energy = torch.log(recon_power.mean(dim=0).clamp_min(eps))
        frame_losses.append(F.l1_loss(recon_frame_log_energy, original_frame_log_energy))

        original_global_log_energy = torch.log(original_power.mean().clamp_min(eps))
        recon_global_log_energy = torch.log(recon_power.mean().clamp_min(eps))
        global_losses.append((recon_global_log_energy - original_global_log_energy).abs())

        original_rms = original_power.mean().clamp_min(eps).sqrt()
        recon_rms = recon_power.mean().clamp_min(eps).sqrt()
        rms_ratios.append(recon_rms / original_rms.clamp_min(eps))

    if not frame_losses:
        return {}

    def _mean_tensor(values: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack([value.to(device=device, dtype=dtype) for value in values]).mean()

    frame_loss = _mean_tensor(frame_losses)
    global_loss = _mean_tensor(global_losses)
    total_loss = float(frame_weight) * frame_loss + float(global_weight) * global_loss
    return {
        "audio_energy_loss": total_loss,
        "audio_frame_log_energy_l1": frame_loss,
        "audio_global_log_energy_l1": global_loss,
        "audio_rms_ratio": _mean_tensor(rms_ratios),
    }


def _audio_artifact_root(artifact_dir: Any, split: str) -> Path:
    base = Path("." if artifact_dir in (None, "") else artifact_dir).expanduser().resolve()
    root = base / "audio_media" / str(split)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_wav_audio_file(root: Path, stem: str, waveform: torch.Tensor, *, sample_rate: int) -> Path:
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(stem)).strip("._") or "audio"
    suffix = uuid.uuid4().hex[:8]
    path = root / f"{safe_stem}_{suffix}.wav"
    pcm = torch.clamp(waveform.detach().cpu().to(torch.float32), -1.0, 1.0).mul(32767.0).round().to(torch.int16).numpy()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())
    return path


def _representative_audio_spec_item(audio_meta: Optional[Mapping[str, Any]], config: Mapping[str, Any]) -> dict:
    if isinstance(audio_meta, Mapping) and "spec_min" in audio_meta and "spec_max" in audio_meta:
        spec_min_values = _tensor_1d(audio_meta["spec_min"], dtype=torch.float32)
        spec_max_values = _tensor_1d(audio_meta["spec_max"], dtype=torch.float32)
        if spec_min_values.numel() > 0 and spec_max_values.numel() > 0:
            spec_min = float(torch.quantile(spec_min_values, 0.5).item())
            spec_max = float(torch.quantile(spec_max_values, 0.5).item())
        else:
            spec_min, spec_max = -12.0, 2.0
        if "spec_shape" in audio_meta:
            shapes = _tensor_2d(audio_meta["spec_shape"], dtype=torch.int64)
            if shapes.numel() > 0:
                spec_shape = shapes[0]
            else:
                spec_shape = torch.tensor(
                    [int(config["stft_n_fft"]) // 2 + 1, int(config["audio_num_samples"]) // int(config["stft_hop_length"]) + 1],
                    dtype=torch.int64,
                )
        else:
            spec_shape = torch.tensor(
                [int(config["stft_n_fft"]) // 2 + 1, int(config["audio_num_samples"]) // int(config["stft_hop_length"]) + 1],
                dtype=torch.int64,
            )
    else:
        spec_min, spec_max = -12.0, 2.0
        spec_shape = torch.tensor(
            [int(config["stft_n_fft"]) // 2 + 1, int(config["audio_num_samples"]) // int(config["stft_hop_length"]) + 1],
            dtype=torch.int64,
        )

    if spec_max <= spec_min:
        spec_max = spec_min + 1.0
    return {
        "path": "generated",
        "crop_mode": torch.tensor(0, dtype=torch.int64),
        "crop_offset": torch.tensor(0, dtype=torch.int64),
        "source_num_samples": torch.tensor(int(config["audio_num_samples"]), dtype=torch.int64),
        "spec_min": torch.tensor(float(spec_min), dtype=torch.float32),
        "spec_max": torch.tensor(float(spec_max), dtype=torch.float32),
        "spec_shape": spec_shape.to(torch.int64),
    }


def _wandb_audio_array(waveform: torch.Tensor) -> np.ndarray:
    return torch.clamp(waveform.detach().cpu().to(torch.float32), -1.0, 1.0).numpy().copy()


def _image_media_payload(items: list[np.ndarray], captions: Optional[list[Optional[str]]] = None) -> dict:
    payload = {
        "kind": "image",
        "items": list(items),
    }
    if captions is not None:
        payload["caption"] = list(captions)
    return payload


def _audio_media_payload(
    items: list[np.ndarray],
    *,
    sample_rates: list[int],
    captions: Optional[list[Optional[str]]] = None,
) -> dict:
    payload = {
        "kind": "audio",
        "items": list(items),
        "sample_rate": [int(rate) for rate in sample_rates],
    }
    if captions is not None:
        payload["caption"] = list(captions)
    return payload


def build_generated_audio_log_payload(
    generated: torch.Tensor,
    *,
    audio_source: Any,
    audio_meta: Optional[Mapping[str, Any]] = None,
    split: str = "generation",
    max_items: int = 4,
    artifact_dir: Any = None,
) -> dict:
    config = audio_config_from_source(audio_source)
    if not _dataset_supports_audio_logging(config):
        return {}
    is_waveform = torch.is_tensor(generated) and generated.ndim == 3 and int(generated.size(1)) == 1
    is_spectrogram = torch.is_tensor(generated) and generated.ndim == 4 and int(generated.size(1)) == 1
    if not (is_waveform or is_spectrogram):
        return {}

    limit = min(int(max_items), int(generated.size(0)))
    if limit <= 0:
        return {}

    generated = generated[:limit].detach().cpu().to(torch.float32)
    meta_item = _representative_audio_spec_item(audio_meta, config)
    audio_items = []
    audio_captions = []
    audio_sample_rates = []

    for idx in range(limit):
        try:
            if is_waveform:
                waveform = generated[idx, 0].clamp(-1.0, 1.0)
            else:
                _, magnitude = _logmag_and_magnitude(generated[idx], meta_item, config)
                waveform = _griffin_lim(
                    magnitude,
                    n_fft=int(config["stft_n_fft"]),
                    hop_length=int(config["stft_hop_length"]),
                    win_length=int(config["stft_win_length"]),
                    length=int(config["audio_num_samples"]),
                    num_iters=int(config["griffin_lim_iters"]),
                )
        except Exception:
            continue
        audio_items.append(_wandb_audio_array(waveform))
        audio_captions.append(f"generated audio {idx}")
        audio_sample_rates.append(int(config["sample_rate"]))

    if not audio_items:
        return {}
    primary_key = f"{str(split).strip() or 'generation'}/audio"
    media_payload = _audio_media_payload(
        audio_items,
        sample_rates=audio_sample_rates,
        captions=audio_captions,
    )
    return {
        primary_key: media_payload,
        "s2/audio": media_payload,
    }


def _figure_to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    if hasattr(fig.canvas, "buffer_rgba"):
        buffer = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image = buffer.reshape(height, width, 4)[..., :3]
        return image.copy()
    if hasattr(fig.canvas, "tostring_rgb"):
        buffer = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = buffer.reshape(height, width, 3)
        return image.copy()
    if hasattr(fig.canvas, "print_to_buffer"):
        buffer, _ = fig.canvas.print_to_buffer()
        image = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 4)[..., :3]
        return image.copy()
    raise AttributeError("Figure canvas does not expose an RGB buffer API.")


def _waveform_plot(original: torch.Tensor, reconstructed: torch.Tensor, *, sample_rate: int, title: str) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    time = np.arange(original.numel(), dtype=np.float32) / float(sample_rate)
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(time, original.detach().cpu().numpy(), label="original", linewidth=0.8)
    ax.plot(time, reconstructed.detach().cpu().numpy(), label="reconstructed", linewidth=0.8, alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel("seconds")
    ax.set_ylabel("amplitude")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    image = _figure_to_rgb(fig)
    plt.close(fig)
    return image


def _triple_panel_plot(
    left: np.ndarray,
    middle: np.ndarray,
    right: np.ndarray,
    *,
    title: str,
    left_title: str,
    middle_title: str,
    right_title: str,
) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, image, panel_title in zip(axes, (left, middle, right), (left_title, middle_title, right_title)):
        ax.imshow(image, origin="lower", aspect="auto", cmap="magma")
        ax.set_title(panel_title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    image = _figure_to_rgb(fig)
    plt.close(fig)
    return image


def build_audio_log_payload(
    inputs: torch.Tensor,
    reconstructions: torch.Tensor,
    *,
    audio_meta: Mapping[str, Any],
    audio_source: Any,
    split: str,
    max_items: int = 4,
    artifact_dir: Any = None,
) -> dict:
    if not has_audio_metadata(audio_meta):
        return {}

    config = audio_config_from_source(audio_source)
    if not _dataset_supports_audio_logging(config):
        return {}

    limit = min(int(max_items), int(inputs.size(0)), len(audio_meta["path"]))
    if limit <= 0:
        return {}

    inputs = inputs[:limit].detach().cpu().to(torch.float32)
    reconstructions = reconstructions[:limit].detach().cpu().to(torch.float32)
    meta = _slice_audio_meta(audio_meta, limit)

    if _is_waveform_batch(inputs, meta):
        audio_items = []
        audio_captions = []
        audio_sample_rates = []
        mel_items = []
        mel_captions = []
        waveform_items = []
        waveform_captions = []
        waveform_l1s = []
        for idx in range(limit):
            item = _meta_item(meta, idx)
            basename = Path(str(item["path"])).stem
            original_waveform = inputs[idx, 0].clamp(-1.0, 1.0)
            recon_waveform = reconstructions[idx, 0].clamp(-1.0, 1.0)
            waveform_l1s.append(float((recon_waveform - original_waveform).abs().mean().item()))
            audio_items.extend(
                [
                    _wandb_audio_array(original_waveform),
                    _wandb_audio_array(recon_waveform),
                ]
            )
            audio_captions.extend(
                [
                    f"{basename} original",
                    f"{basename} reconstructed",
                ]
            )
            audio_sample_rates.extend([int(config["sample_rate"]), int(config["sample_rate"])])
            try:
                original_mel = _mel_db(original_waveform, config)
                recon_mel = _mel_db(recon_waveform, config)
                mel_diff = np.abs(recon_mel - original_mel)
                mel_diff = mel_diff / max(float(mel_diff.max()), 1.0e-6)
                mel_items.append(
                    _triple_panel_plot(
                        original_mel,
                        recon_mel,
                        mel_diff,
                        title=f"{basename} mel comparison",
                        left_title="original mel",
                        middle_title="reconstructed mel",
                        right_title="abs diff",
                    )
                )
                mel_captions.append(f"{basename} mel comparison")
            except Exception:
                pass
            waveform_items.append(
                _waveform_plot(
                    original_waveform,
                    recon_waveform,
                    sample_rate=int(config["sample_rate"]),
                    title=f"{basename} waveform comparison",
                )
            )
            waveform_captions.append(f"{basename} waveform comparison")

        payload = {}
        if audio_items:
            payload[f"{split}/audio_clips"] = _audio_media_payload(
                audio_items,
                sample_rates=audio_sample_rates,
                captions=audio_captions,
            )
        if mel_items:
            payload[f"{split}/audio_mels"] = _image_media_payload(mel_items, mel_captions)
        if waveform_items:
            payload[f"{split}/audio_waveforms"] = _image_media_payload(waveform_items, waveform_captions)
        if waveform_l1s:
            payload[f"{split}/audio_waveform_l1"] = float(np.mean(waveform_l1s))
        return payload

    audio_items = []
    audio_captions = []
    audio_sample_rates = []
    mel_items = []
    mel_captions = []
    waveform_items = []
    waveform_captions = []
    logmag_items = []
    logmag_captions = []
    waveform_l1s = []
    spectral_mses = []

    for idx in range(limit):
        item = _meta_item(meta, idx)
        basename = Path(str(item["path"])).stem
        try:
            original_logmag, original_magnitude = _logmag_and_magnitude(inputs[idx], item, config)
            recon_logmag, recon_magnitude = _logmag_and_magnitude(reconstructions[idx], item, config)
            original_waveform = _griffin_lim(
                original_magnitude,
                n_fft=int(config["stft_n_fft"]),
                hop_length=int(config["stft_hop_length"]),
                win_length=int(config["stft_win_length"]),
                length=int(config["audio_num_samples"]),
                num_iters=int(config["griffin_lim_iters"]),
            )
            recon_waveform = _griffin_lim(
                recon_magnitude,
                n_fft=int(config["stft_n_fft"]),
                hop_length=int(config["stft_hop_length"]),
                win_length=int(config["stft_win_length"]),
                length=int(config["audio_num_samples"]),
                num_iters=int(config["griffin_lim_iters"]),
            )
        except Exception as exc:
            logmag_items.append(np.zeros((48, 256, 3), dtype=np.uint8))
            logmag_captions.append(f"{basename}: audio logging failed ({exc})")
            continue

        waveform_l1s.append(float((recon_waveform - original_waveform).abs().mean().item()))
        spectral_mses.append(float((recon_logmag - original_logmag).pow(2).mean().item()))

        audio_items.extend(
            [
                _wandb_audio_array(original_waveform),
                _wandb_audio_array(recon_waveform),
            ]
        )
        audio_captions.extend(
            [
                f"{basename} original",
                f"{basename} reconstructed",
            ]
        )
        audio_sample_rates.extend(
            [
                int(config["sample_rate"]),
                int(config["sample_rate"]),
            ]
        )

        original_mel = _mel_db(original_waveform, config)
        recon_mel = _mel_db(recon_waveform, config)
        mel_diff = np.abs(recon_mel - original_mel)
        mel_diff = mel_diff / max(float(mel_diff.max()), 1.0e-6)
        mel_items.append(
            _triple_panel_plot(
                original_mel,
                recon_mel,
                mel_diff,
                title=f"{basename} mel comparison",
                left_title="original mel",
                middle_title="reconstructed mel",
                right_title="abs diff",
            )
        )
        mel_captions.append(f"{basename} mel comparison")

        waveform_items.append(
            _waveform_plot(
                original_waveform,
                recon_waveform,
                sample_rate=int(config["sample_rate"]),
                title=f"{basename} waveform comparison",
            )
        )
        waveform_captions.append(f"{basename} waveform comparison")

        original_logmag_np = original_logmag.numpy()
        recon_logmag_np = recon_logmag.numpy()
        diff_np = np.abs(recon_logmag_np - original_logmag_np)
        diff_np = diff_np / max(float(diff_np.max()), 1.0e-6)
        logmag_items.append(
            _triple_panel_plot(
                original_logmag_np,
                recon_logmag_np,
                diff_np,
                title=f"{basename} log-magnitude comparison",
                left_title="original log-mag",
                middle_title="reconstructed log-mag",
                right_title="abs diff",
            )
        )
        logmag_captions.append(f"{basename} log-magnitude comparison")

    payload = {}
    if audio_items:
        payload[f"{split}/audio_clips"] = _audio_media_payload(
            audio_items,
            sample_rates=audio_sample_rates,
            captions=audio_captions,
        )
    if mel_items:
        payload[f"{split}/audio_mels"] = _image_media_payload(mel_items, mel_captions)
    if waveform_items:
        payload[f"{split}/audio_waveforms"] = _image_media_payload(waveform_items, waveform_captions)
    if logmag_items:
        payload[f"{split}/audio_logmag"] = _image_media_payload(logmag_items, logmag_captions)
    if waveform_l1s:
        payload[f"{split}/audio_waveform_l1"] = float(np.mean(waveform_l1s))
    if spectral_mses:
        payload[f"{split}/audio_logmag_mse"] = float(np.mean(spectral_mses))
    return payload
