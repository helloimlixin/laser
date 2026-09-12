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
    audio_max_gain = float(_get_value(source, "audio_max_gain", 8.0))
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
        "audio_max_gain": audio_max_gain,
        "generated_audio_rms_normalize": bool(
            _get_value(source, "generated_audio_rms_normalize", _get_value(source, "audio_rms_normalize", False))
        ),
        "generated_audio_target_rms": float(
            _get_value(source, "generated_audio_target_rms", _get_value(source, "audio_target_rms", 0.12))
        ),
        "generated_audio_target_peak": float(
            _get_value(source, "generated_audio_target_peak", _get_value(source, "audio_target_peak", 0.95))
        ),
        "generated_audio_max_gain": float(
            _get_value(source, "generated_audio_max_gain", max(audio_max_gain, 64.0))
        ),
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
    )
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


def is_visqol_available() -> bool:
    """Return whether an official ViSQOL Python binding or CLI is available."""
    return _has_visqol_python_module() or _resolve_visqol_binary() is not None


def _has_pesq() -> bool:
    return importlib.util.find_spec("pesq") is not None


def _has_stoi() -> bool:
    return importlib.util.find_spec("pystoi") is not None


def _visqol_mode(sample_rate: int, *, dataset: Optional[str] = None) -> str:
    dataset_key = str(dataset or "").strip().lower()
    if dataset_key == "vctk":
        return "speech"
    if dataset_key == "maestro":
        return "audio"
    return "speech" if int(sample_rate) <= 16000 else "audio"


def _prepare_visqol_waveforms(
    reference_waveform: torch.Tensor,
    degraded_waveform: torch.Tensor,
    *,
    sample_rate: int,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Resample inputs to the canonical rate required by official ViSQOL."""
    canonical_rate = 16_000 if str(mode) == "speech" else 48_000

    def _prepare(waveform: torch.Tensor) -> torch.Tensor:
        values = np.asarray(
            waveform.detach().cpu().to(torch.float32).reshape(-1).numpy(),
            dtype=np.float32,
        )
        values = _resample_if_needed(values, int(sample_rate), canonical_rate)
        return torch.from_numpy(np.ascontiguousarray(values))

    return _prepare(reference_waveform), _prepare(degraded_waveform), canonical_rate


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
        mode = _visqol_mode(sample_rate)
        model_name = _VISQOL_SPEECH_MODEL if mode == "speech" else _VISQOL_AUDIO_MODEL
        model_candidates = []
        explicit_model = os.environ.get(
            "VISQOL_SPEECH_MODEL" if mode == "speech" else "VISQOL_AUDIO_MODEL"
        )
        if explicit_model:
            model_candidates.append(Path(explicit_model).expanduser())
        binary_path = Path(binary).expanduser().resolve()
        model_candidates.extend(
            (
                Path(f"{binary_path}.runfiles") / "__main__" / "model" / model_name,
                binary_path.parent / "model" / model_name,
                Path.cwd() / "model" / model_name,
            )
        )
        model_path = next((path for path in model_candidates if path.is_file()), None)
        if model_path is not None:
            cmd.extend(("--similarity_to_quality_model", str(model_path)))
        if mode == "speech":
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
    mode: Optional[str] = None,
) -> Optional[float]:
    mode = str(mode or _visqol_mode(sample_rate)).strip().lower()
    if mode not in {"speech", "audio"}:
        raise ValueError(f"ViSQOL mode must be 'speech' or 'audio', got {mode!r}")
    reference_waveform, degraded_waveform, sample_rate = _prepare_visqol_waveforms(
        reference_waveform,
        degraded_waveform,
        sample_rate=sample_rate,
        mode=mode,
    )
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
    return _stft_complex_batch(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    ).abs()


def _stft_complex_batch(
    waveform: torch.Tensor,
    *,
    n_fft: int,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
) -> torch.Tensor:
    """Return a differentiable complex STFT for a mono waveform batch."""
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
    )


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
    phase_weight: float = 0.0,
    phase_activity_floor_db: float = -40.0,
    sample_rate: int = 24_000,
    tonal_grid_hz: float = 0.0,
    tonal_margin_db: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Differentiable multi-resolution magnitude and phase reconstruction loss.

    Magnitude-only objectives can reward spectrally plausible but source-
    incoherent high-frequency energy.  When ``phase_weight`` is positive, this
    also measures anti-wrapped instantaneous phase, group-delay, and temporal
    phase-increment errors on reference-active bins.  This is the waveform-
    decoder analogue of the explicit phase supervision used by APCodec; the
    activity mask avoids undefined phase in silent bins.
    """
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
    complex_spectral_terms = []
    phase_ip_terms = []
    phase_gd_terms = []
    phase_iaf_terms = []
    tonal_excess_terms = []
    result_dtype = inputs.dtype if torch.is_floating_point(inputs) else torch.float32
    recon = reconstructions.to(dtype=torch.float32)
    target = inputs.to(dtype=torch.float32)
    phase_weight = max(0.0, float(phase_weight))
    phase_activity_ratio = 10.0 ** (float(phase_activity_floor_db) / 20.0)
    sample_rate = max(1, int(sample_rate))
    tonal_grid_hz = max(0.0, float(tonal_grid_hz))
    tonal_margin_nepers = max(0.0, float(tonal_margin_db)) * math.log(10.0) / 10.0
    largest_fft = max(fft_sizes)

    def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(dtype=value.dtype)
        return (value * mask).sum() / mask.sum().clamp_min(1.0)

    for n_fft, hop, win in zip(fft_sizes, hop_lengths, win_lengths):
        target_spec = _stft_complex_batch(
            target, n_fft=n_fft, hop_length=hop, win_length=win
        )
        recon_spec = _stft_complex_batch(
            recon, n_fft=n_fft, hop_length=hop, win_length=win
        )
        target_mag = target_spec.abs()
        recon_mag = recon_spec.abs()
        spectral_terms.append(
            _waveform_spectral_convergence(
                target_mag,
                recon_mag,
                eps=eps,
            )
        )
        logmag_terms.append(F.l1_loss(torch.log(recon_mag.clamp_min(eps)), torch.log(target_mag.clamp_min(eps))))
        linmag_terms.append(F.l1_loss(recon_mag, target_mag))
        if phase_weight > 0.0:
            complex_diff = (recon_spec - target_spec).reshape(
                int(target_spec.size(0)), -1
            )
            complex_target = target_spec.reshape(int(target_spec.size(0)), -1)
            complex_spectral_terms.append(
                (
                    torch.linalg.vector_norm(complex_diff, dim=1)
                    / torch.linalg.vector_norm(complex_target, dim=1).clamp_min(eps)
                ).mean()
            )

            # The angle of recon * conj(target) is already wrapped to [-pi, pi].
            # Its finite difference along frequency/time gives anti-wrapped
            # group-delay and instantaneous-frequency errors without unwrapping
            # either phase surface independently.
            phase_error = torch.angle(recon_spec * target_spec.conj())
            local_peak = target_mag.amax(dim=1, keepdim=True)
            active = (local_peak > eps) & (
                target_mag >= local_peak * phase_activity_ratio
            )
            phase_ip_terms.append(_masked_mean(phase_error.abs(), active))
            if int(phase_error.size(1)) > 1:
                group_delay_delta = (
                    phase_error[:, 1:, :] - phase_error[:, :-1, :]
                )
                group_delay_error = torch.atan2(
                    torch.sin(group_delay_delta),
                    torch.cos(group_delay_delta),
                )
                phase_gd_terms.append(
                    _masked_mean(
                        group_delay_error.abs(),
                        active[:, 1:, :] & active[:, :-1, :],
                    )
                )
            if int(phase_error.size(2)) > 1:
                temporal_phase_delta = (
                    phase_error[:, :, 1:] - phase_error[:, :, :-1]
                )
                temporal_phase_error = torch.atan2(
                    torch.sin(temporal_phase_delta),
                    torch.cos(temporal_phase_delta),
                )
                phase_iaf_terms.append(
                    _masked_mean(
                        temporal_phase_error.abs(),
                        active[:, :, 1:] & active[:, :, :-1],
                    )
                )

        if tonal_grid_hz > 0.0 and int(n_fft) == int(largest_fft):
            # Transposed-convolution images appear as narrow, stationary lines
            # at multiples of an intermediate sample rate. Compare each line's
            # prominence to the source's local spectral neighborhood so real
            # speech harmonics are retained and only reconstruction excess is
            # penalized.
            frequencies = torch.linspace(
                0.0,
                0.5 * float(sample_rate),
                steps=int(target_mag.size(1)),
                device=target_mag.device,
                dtype=target_mag.dtype,
            )
            nyquist = 0.5 * float(sample_rate)
            first_harmonic = max(1, int(math.ceil(3_000.0 / tonal_grid_hz)))
            last_harmonic = int(math.floor((nyquist - 180.0) / tonal_grid_hz))
            if last_harmonic >= first_harmonic:
                harmonic_centers = (
                    torch.arange(
                        first_harmonic,
                        last_harmonic + 1,
                        device=target_mag.device,
                        dtype=target_mag.dtype,
                    )
                    * tonal_grid_hz
                )
                distance = (
                    frequencies.unsqueeze(0) - harmonic_centers.unsqueeze(1)
                ).abs()
                center_weights = (distance <= 18.0).to(target_mag.dtype)
                shoulder_weights = (
                    (distance >= 60.0) & (distance <= 180.0)
                ).to(target_mag.dtype)
                center_weights = center_weights / center_weights.sum(
                    dim=1, keepdim=True
                ).clamp_min(1.0)
                shoulder_weights = shoulder_weights / shoulder_weights.sum(
                    dim=1, keepdim=True
                ).clamp_min(1.0)
                target_power = target_mag.square()
                recon_power = recon_mag.square()
                target_center = torch.einsum(
                    "hf,bft->bht", center_weights, target_power
                )
                recon_center = torch.einsum(
                    "hf,bft->bht", center_weights, recon_power
                )
                target_shoulder = torch.einsum(
                    "hf,bft->bht", shoulder_weights, target_power
                )
                recon_shoulder = torch.einsum(
                    "hf,bft->bht", shoulder_weights, recon_power
                )
                target_tonality = torch.log(
                    (target_center + eps) / (target_shoulder + eps)
                )
                recon_tonality = torch.log(
                    (recon_center + eps) / (recon_shoulder + eps)
                )
                tonal_excess_terms.append(
                    (recon_tonality - target_tonality - tonal_margin_nepers)
                    .clamp_min(0.0)
                    .mean()
                )

    spectral_loss = torch.stack(spectral_terms).mean().to(dtype=result_dtype)
    logmag_loss = torch.stack(logmag_terms).mean().to(dtype=result_dtype)
    linmag_loss = torch.stack(linmag_terms).mean().to(dtype=result_dtype)
    total = spectral_loss + logmag_loss + 0.1 * linmag_loss
    metrics = {
        "audio_multires_stft_loss": total,
        "audio_multires_stft_spectral_convergence": spectral_loss,
        "audio_multires_stft_logmag_l1": logmag_loss,
        "audio_multires_stft_mag_l1": linmag_loss,
    }
    if phase_weight > 0.0:
        complex_spectral_loss = torch.stack(complex_spectral_terms).mean()
        phase_ip_loss = torch.stack(phase_ip_terms).mean()
        phase_gd_loss = (
            torch.stack(phase_gd_terms).mean()
            if phase_gd_terms
            else phase_ip_loss.new_zeros(())
        )
        phase_iaf_loss = (
            torch.stack(phase_iaf_terms).mean()
            if phase_iaf_terms
            else phase_ip_loss.new_zeros(())
        )
        # Optimize only the scale-invariant wrapped phase terms here. Complex
        # convergence is retained as a diagnostic, but including it in the
        # phase objective lets an under-capacity decoder lower loss by turning
        # down hard-to-predict high-frequency energy. Magnitude, mel, ERB, and
        # time-domain objectives already supervise amplitude explicitly.
        phase_loss = (
            phase_ip_loss + phase_gd_loss + phase_iaf_loss
        ) / 3.0
        total = total + phase_weight * phase_loss.to(dtype=total.dtype)
        metrics.update(
            {
                "audio_multires_stft_loss": total,
                "audio_multires_stft_phase_loss": phase_loss.to(dtype=result_dtype),
                "audio_multires_stft_complex_convergence": complex_spectral_loss.to(
                    dtype=result_dtype
                ),
                "audio_multires_stft_phase_ip": phase_ip_loss.to(dtype=result_dtype),
                "audio_multires_stft_phase_gd": phase_gd_loss.to(dtype=result_dtype),
                "audio_multires_stft_phase_iaf": phase_iaf_loss.to(dtype=result_dtype),
            }
        )
    if tonal_excess_terms:
        tonal_loss = torch.stack(tonal_excess_terms).mean().to(dtype=result_dtype)
        metrics.update(
            {
                "audio_upsampling_tone_loss": tonal_loss,
                "audio_upsampling_tone_excess_db": (
                    tonal_loss * (10.0 / math.log(10.0))
                ),
            }
        )
    return metrics


def compute_waveform_preemphasis_loss(
    inputs: torch.Tensor,
    reconstructions: torch.Tensor,
    *,
    coefficient: float = 0.97,
) -> dict[str, torch.Tensor]:
    """Source-aligned high-frequency waveform loss.

    Matching only the RMS of a first difference can restore brightness with
    unrelated noise.  Pre-emphasis instead compares the filtered waveforms
    sample-for-sample, directly penalizing the audible high-band residual.
    """
    if not torch.is_tensor(inputs) or not torch.is_tensor(reconstructions):
        return {}
    if (
        inputs.ndim != 3
        or reconstructions.ndim != 3
        or int(inputs.size(1)) != 1
        or int(reconstructions.size(1)) != 1
        or int(inputs.size(-1)) < 2
        or int(reconstructions.size(-1)) < 2
    ):
        return {}
    coefficient = float(coefficient)
    if not math.isfinite(coefficient) or not 0.0 <= coefficient <= 1.0:
        raise ValueError("coefficient must be finite and in [0, 1]")
    target = inputs.to(dtype=torch.float32)
    recon = reconstructions.to(dtype=torch.float32)
    target_pre = target[..., 1:] - coefficient * target[..., :-1]
    recon_pre = recon[..., 1:] - coefficient * recon[..., :-1]
    difference = recon_pre - target_pre
    l1_loss = difference.abs().mean()
    convergence = (
        torch.linalg.vector_norm(difference.flatten(1), dim=1)
        / torch.linalg.vector_norm(target_pre.flatten(1), dim=1).clamp_min(1.0e-8)
    ).mean()
    result_dtype = inputs.dtype if torch.is_floating_point(inputs) else torch.float32
    return {
        "audio_preemphasis_l1_loss": l1_loss.to(dtype=result_dtype),
        "audio_preemphasis_convergence": convergence.to(dtype=result_dtype),
    }


_MEL_FB_CACHE: dict[tuple[int, int, int], torch.Tensor] = {}
_SLANEY_MEL_FB_CACHE: dict[tuple[int, int, int], torch.Tensor] = {}


def _cached_mel_filterbank(*, sample_rate: int, n_fft: int, n_mels: int, device, dtype) -> torch.Tensor:
    key = (int(sample_rate), int(n_fft), int(n_mels))
    fb = _MEL_FB_CACHE.get(key)
    if fb is None:
        fb = _mel_filterbank(sample_rate=int(sample_rate), n_fft=int(n_fft), n_mels=int(n_mels))
        _MEL_FB_CACHE[key] = fb
    return fb.to(device=device, dtype=dtype)


def _slaney_mel_filterbank(*, sample_rate: int, n_fft: int, n_mels: int) -> torch.Tensor:
    """Librosa-compatible Slaney mel bank used by the authors' codec code."""
    key = (int(sample_rate), int(n_fft), int(n_mels))
    cached = _SLANEY_MEL_FB_CACHE.get(key)
    if cached is not None:
        return cached

    def hz_to_mel(frequencies: np.ndarray) -> np.ndarray:
        frequencies = np.asarray(frequencies, dtype=np.float64)
        f_sp = 200.0 / 3.0
        mels = frequencies / f_sp
        log_region = frequencies >= 1000.0
        mels[log_region] = 15.0 + np.log(frequencies[log_region] / 1000.0) / (
            np.log(6.4) / 27.0
        )
        return mels

    def mel_to_hz(mels: np.ndarray) -> np.ndarray:
        mels = np.asarray(mels, dtype=np.float64)
        f_sp = 200.0 / 3.0
        frequencies = f_sp * mels
        log_region = mels >= 15.0
        frequencies[log_region] = 1000.0 * np.exp(
            (np.log(6.4) / 27.0) * (mels[log_region] - 15.0)
        )
        return frequencies

    mel_edges = np.linspace(
        hz_to_mel(np.array([0.0]))[0],
        hz_to_mel(np.array([0.5 * float(sample_rate)]))[0],
        int(n_mels) + 2,
    )
    hz_edges = mel_to_hz(mel_edges)
    fft_frequencies = np.linspace(
        0.0, 0.5 * float(sample_rate), 1 + int(n_fft) // 2
    )
    ramps = hz_edges[:, None] - fft_frequencies[None, :]
    weights = np.zeros((int(n_mels), 1 + int(n_fft) // 2), dtype=np.float64)
    for index in range(int(n_mels)):
        lower = -ramps[index] / max(hz_edges[index + 1] - hz_edges[index], 1.0e-12)
        upper = ramps[index + 2] / max(hz_edges[index + 2] - hz_edges[index + 1], 1.0e-12)
        weights[index] = np.maximum(0.0, np.minimum(lower, upper))
    weights *= (2.0 / np.maximum(hz_edges[2:] - hz_edges[:-2], 1.0e-12))[:, None]
    cached = torch.from_numpy(weights.astype(np.float32, copy=False)).contiguous()
    _SLANEY_MEL_FB_CACHE[key] = cached
    return cached


def compute_waveform_mdctcodec_mel_loss(
    inputs: torch.Tensor,
    reconstructions: torch.Tensor,
    *,
    sample_rate: int = 48_000,
    n_fft: int = 1024,
    hop_length: int = 40,
    win_length: int = 320,
    n_mels: int = 80,
) -> dict[str, torch.Tensor]:
    """MDCTCodec/APCodec log-mel MAE+MSE objective.

    This deliberately differs from the repository's generic multi-resolution
    mel objective: the published spectral codec uses one 1024-point STFT, a
    320-sample Hann window, 40-sample hop, 80 Slaney mel bands, log compression,
    and the sum of elementwise MAE and MSE.
    """
    if not torch.is_tensor(inputs) or not torch.is_tensor(reconstructions):
        return {}
    if (
        inputs.ndim != 3
        or reconstructions.ndim != 3
        or int(inputs.size(1)) != 1
        or int(reconstructions.size(1)) != 1
    ):
        return {}

    target = inputs[:, 0].to(torch.float32)
    recon = reconstructions[:, 0].to(torch.float32)
    window = torch.hann_window(
        int(win_length), periodic=True, device=target.device, dtype=target.dtype
    )

    def log_mel(waveform: torch.Tensor) -> torch.Tensor:
        spectrum = torch.stft(
            waveform,
            n_fft=int(n_fft),
            hop_length=int(hop_length),
            win_length=int(win_length),
            window=window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        magnitude = torch.sqrt(spectrum.real.square() + spectrum.imag.square() + 1.0e-9)
        filterbank = _slaney_mel_filterbank(
            sample_rate=int(sample_rate),
            n_fft=int(n_fft),
            n_mels=int(n_mels),
        ).to(device=magnitude.device, dtype=magnitude.dtype)
        mel = torch.einsum("mf,bft->bmt", filterbank, magnitude)
        return torch.log(mel.clamp_min(1.0e-5))

    target_mel = log_mel(target).detach()
    recon_mel = log_mel(recon)
    difference = recon_mel - target_mel
    mae = difference.abs().mean()
    mse = difference.square().mean()
    result_dtype = inputs.dtype if torch.is_floating_point(inputs) else torch.float32
    return {
        "audio_mel_loss": (mae + mse).to(dtype=result_dtype),
        "audio_mel_mae_loss": mae.to(dtype=result_dtype),
        "audio_mel_mse_loss": mse.to(dtype=result_dtype),
    }


def compute_waveform_multiscale_mel_loss(
    inputs: torch.Tensor,
    reconstructions: torch.Tensor,
    *,
    sample_rate: int = 16000,
    fft_sizes: Optional[Sequence[int]] = None,
    n_mels: Sequence[int] | int = 80,
) -> dict[str, torch.Tensor]:
    """Differentiable multi-scale log-mel L1 loss (DAC/HiFi-GAN style).

    For each STFT resolution the magnitude spectrogram is projected onto a mel
    filterbank, log-compressed, and compared with L1. This is the perceptual
    reconstruction objective that pairs with the adversarial + feature-matching
    losses. Returns ``{}`` for non-waveform inputs.
    """
    if not torch.is_tensor(inputs) or not torch.is_tensor(reconstructions):
        return {}
    if inputs.ndim != 3 or reconstructions.ndim != 3 or int(inputs.size(1)) != 1 or int(reconstructions.size(1)) != 1:
        return {}

    fft_sizes = _canonical_stft_sizes(fft_sizes, (512, 1024, 2048))
    if isinstance(n_mels, int):
        n_mels_list = [int(n_mels)] * len(fft_sizes)
    else:
        n_mels_list = [int(m) for m in n_mels]
        if len(n_mels_list) != len(fft_sizes):
            raise ValueError(
                f"n_mels must be an int or match fft_sizes length (got {len(n_mels_list)} vs {len(fft_sizes)})"
            )

    eps = 1.0e-5
    result_dtype = inputs.dtype if torch.is_floating_point(inputs) else torch.float32
    recon = reconstructions.to(dtype=torch.float32)
    target = inputs.to(dtype=torch.float32)
    terms = []
    for n_fft, mels in zip(fft_sizes, n_mels_list):
        hop = max(1, n_fft // 4)
        target_mag = _stft_magnitude_batch(target, n_fft=n_fft, hop_length=hop, win_length=n_fft)
        recon_mag = _stft_magnitude_batch(recon, n_fft=n_fft, hop_length=hop, win_length=n_fft)
        fb = _cached_mel_filterbank(
            sample_rate=sample_rate, n_fft=n_fft, n_mels=mels, device=target_mag.device, dtype=target_mag.dtype
        )
        # [n_mels, F] @ [B, F, T] -> [B, n_mels, T]
        target_mel = torch.log(torch.einsum("mf,bft->bmt", fb, target_mag).clamp_min(eps))
        recon_mel = torch.log(torch.einsum("mf,bft->bmt", fb, recon_mag).clamp_min(eps))
        terms.append(F.l1_loss(recon_mel, target_mel))
    mel_loss = torch.stack(terms).mean().to(dtype=result_dtype)
    return {"audio_mel_loss": mel_loss}


_CRITICAL_BAND_FB_CACHE: dict[tuple[int, int, int, float, float], torch.Tensor] = {}


def _cached_erb_filterbank(
    *,
    sample_rate: int,
    n_fft: int,
    num_bands: int,
    min_frequency: float,
    max_frequency: float,
    device,
    dtype,
) -> torch.Tensor:
    """Return unit-area triangular bands uniformly spaced on the ERB-rate scale."""
    nyquist = 0.5 * float(sample_rate)
    min_frequency = max(0.0, float(min_frequency))
    max_frequency = min(max(float(max_frequency), min_frequency + 1.0), nyquist)
    key = (
        int(sample_rate),
        int(n_fft),
        int(num_bands),
        round(min_frequency, 4),
        round(max_frequency, 4),
    )
    filterbank = _CRITICAL_BAND_FB_CACHE.get(key)
    if filterbank is None:
        # Glasberg-Moore ERB-rate mapping. Uniform spacing here approximates
        # auditory critical-band integration much more closely than linear FFT
        # bins, especially below 2 kHz.
        def hz_to_erb(frequency: torch.Tensor) -> torch.Tensor:
            return 21.4 * torch.log10(1.0 + 4.37e-3 * frequency)

        def erb_to_hz(rate: torch.Tensor) -> torch.Tensor:
            return (torch.pow(10.0, rate / 21.4) - 1.0) / 4.37e-3

        low_rate = hz_to_erb(torch.tensor(min_frequency, dtype=torch.float32))
        high_rate = hz_to_erb(torch.tensor(max_frequency, dtype=torch.float32))
        edges = erb_to_hz(
            torch.linspace(low_rate, high_rate, steps=int(num_bands) + 2)
        )
        frequencies = torch.linspace(0.0, nyquist, steps=int(n_fft) // 2 + 1)
        left = edges[:-2, None]
        center = edges[1:-1, None]
        right = edges[2:, None]
        rising = (frequencies[None, :] - left) / (center - left).clamp_min(1.0e-8)
        falling = (right - frequencies[None, :]) / (right - center).clamp_min(1.0e-8)
        filterbank = torch.minimum(rising, falling).clamp(0.0, 1.0)
        filterbank = filterbank / filterbank.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        _CRITICAL_BAND_FB_CACHE[key] = filterbank
    return filterbank.to(device=device, dtype=dtype)


def compute_waveform_critical_band_energy_loss(
    inputs: torch.Tensor,
    reconstructions: torch.Tensor,
    *,
    sample_rate: int = 24_000,
    fft_sizes: Optional[Sequence[int]] = (512, 2048),
    num_bands: int = 32,
    min_frequency: float = 30.0,
    max_frequency: Optional[float] = None,
    deficit_weight: float = 0.5,
    activity_floor_db: float = -50.0,
) -> dict[str, torch.Tensor]:
    """Match short-time energy in perceptual critical bands.

    The symmetric log-energy error preserves the spectral envelope, while an
    additional one-sided term penalizes missing audible energy more strongly
    than excess energy. The latter directly targets dull consonants and weak
    attacks without applying a broadband gain. Bands below the reference's
    local activity floor are excluded from the asymmetric term.
    """
    if not torch.is_tensor(inputs) or not torch.is_tensor(reconstructions):
        return {}
    if (
        inputs.ndim != 3
        or reconstructions.ndim != 3
        or int(inputs.size(1)) != 1
        or int(reconstructions.size(1)) != 1
    ):
        return {}
    sample_rate = max(1, int(sample_rate))
    num_bands = max(1, int(num_bands))
    fft_sizes = _canonical_stft_sizes(fft_sizes, (512, 2048))
    max_frequency = (
        0.5 * float(sample_rate)
        if max_frequency is None
        else min(float(max_frequency), 0.5 * float(sample_rate))
    )
    eps = 1.0e-8
    target = inputs.to(dtype=torch.float32)
    recon = reconstructions.to(dtype=torch.float32)
    result_dtype = inputs.dtype if torch.is_floating_point(inputs) else torch.float32
    symmetric_terms = []
    deficit_terms = []
    log_ratio_terms = []
    activity_power_ratio = 10.0 ** (float(activity_floor_db) / 10.0)

    for n_fft in fft_sizes:
        hop = max(1, int(n_fft) // 4)
        target_magnitude = _stft_magnitude_batch(
            target, n_fft=n_fft, hop_length=hop, win_length=n_fft
        )
        recon_magnitude = _stft_magnitude_batch(
            recon, n_fft=n_fft, hop_length=hop, win_length=n_fft
        )
        filterbank = _cached_erb_filterbank(
            sample_rate=sample_rate,
            n_fft=n_fft,
            num_bands=num_bands,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            device=target_magnitude.device,
            dtype=target_magnitude.dtype,
        )
        target_energy = torch.einsum(
            "kf,bft->bkt", filterbank, target_magnitude.square()
        ).clamp_min(eps)
        recon_energy = torch.einsum(
            "kf,bft->bkt", filterbank, recon_magnitude.square()
        ).clamp_min(eps)
        target_log = torch.log(target_energy)
        recon_log = torch.log(recon_energy)
        log_error = recon_log - target_log
        symmetric_terms.append(log_error.abs().mean())

        local_peak = target_energy.amax(dim=1, keepdim=True)
        active = (target_energy >= local_peak * activity_power_ratio).to(target_energy.dtype)
        active_count = active.sum().clamp_min(1.0)
        deficit_terms.append(((-log_error).clamp_min(0.0) * active).sum() / active_count)
        log_ratio_terms.append((log_error * active).sum() / active_count)

    symmetric_loss = torch.stack(symmetric_terms).mean()
    deficit_loss = torch.stack(deficit_terms).mean()
    total = symmetric_loss + float(deficit_weight) * deficit_loss
    energy_ratio = torch.exp(torch.stack(log_ratio_terms).mean())
    return {
        "audio_critical_band_loss": total.to(dtype=result_dtype),
        "audio_critical_band_log_energy_l1": symmetric_loss.to(dtype=result_dtype),
        "audio_critical_band_deficit": deficit_loss.to(dtype=result_dtype),
        "audio_critical_band_energy_ratio": energy_ratio.to(dtype=result_dtype),
    }


def compute_waveform_energy_matching_loss(
    inputs: torch.Tensor,
    reconstructions: torch.Tensor,
    *,
    sample_rate: int = 24_000,
    frame_durations_ms: Sequence[float] = (10.0, 40.0, 160.0),
    frame_weight: float = 0.5,
    global_weight: float = 0.25,
    transient_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Match waveform loudness envelopes and high-frequency/transient energy.

    Sample and spectral reconstruction losses can have nearly optimal values
    while a trainable decoder remains perceptually flat: the global RMS is
    preserved by low-frequency/voiced energy, but consonants and attacks are
    attenuated.  This objective therefore combines three scale-independent
    log-RMS terms:

    * global waveform energy;
    * active-speech frame energy at multiple time scales; and
    * first-difference energy, a cheap phase-insensitive proxy for high-frequency
      and transient content.

    Silence is excluded from the frame term using a per-example threshold 35 dB
    below the reference RMS. The loss is differentiable with respect to
    ``reconstructions`` and returns direct energy ratios for monitoring.
    """
    if not torch.is_tensor(inputs) or not torch.is_tensor(reconstructions):
        return {}
    if (
        inputs.ndim != 3
        or reconstructions.ndim != 3
        or int(inputs.size(1)) != 1
        or int(reconstructions.size(1)) != 1
    ):
        return {}

    sample_rate = max(1, int(sample_rate))
    eps = 1.0e-8
    result_dtype = inputs.dtype if torch.is_floating_point(inputs) else torch.float32
    target = inputs.to(dtype=torch.float32)
    recon = reconstructions.to(dtype=torch.float32)

    target_rms = target.square().mean(dim=-1).clamp_min(eps).sqrt()
    recon_rms = recon.square().mean(dim=-1).clamp_min(eps).sqrt()
    global_log_ratio = torch.log(recon_rms / target_rms.clamp_min(eps))
    global_loss = global_log_ratio.abs().mean()
    global_ratio = (recon_rms / target_rms.clamp_min(eps)).mean()

    frame_losses: list[torch.Tensor] = []
    frame_log_ratios: list[torch.Tensor] = []
    for duration_ms in frame_durations_ms:
        frame_size = max(1, int(round(float(duration_ms) * sample_rate / 1_000.0)))
        frame_size = min(frame_size, int(target.size(-1)))
        hop_size = max(1, frame_size // 2)
        target_frame_rms = F.avg_pool1d(
            target.square(), frame_size, stride=hop_size
        ).clamp_min(eps).sqrt()
        recon_frame_rms = F.avg_pool1d(
            recon.square(), frame_size, stride=hop_size
        ).clamp_min(eps).sqrt()
        # Keep real speech/background structure in scope but prevent padded or
        # truly silent frames from dominating a logarithmic objective.
        activity_floor = target_rms.unsqueeze(-1) * (10.0 ** (-35.0 / 20.0))
        active = (target_frame_rms >= activity_floor).to(target_frame_rms.dtype)
        active_count = active.sum().clamp_min(1.0)
        log_ratio = torch.log(recon_frame_rms / target_frame_rms.clamp_min(eps))
        frame_losses.append((log_ratio.abs() * active).sum() / active_count)
        frame_log_ratios.append((log_ratio * active).sum() / active_count)

    if frame_losses:
        frame_loss = torch.stack(frame_losses).mean()
        active_frame_ratio = torch.exp(torch.stack(frame_log_ratios).mean())
    else:
        frame_loss = global_loss.new_zeros(())
        active_frame_ratio = global_ratio

    if int(target.size(-1)) > 1:
        target_delta = target[..., 1:] - target[..., :-1]
        recon_delta = recon[..., 1:] - recon[..., :-1]
        target_delta_rms = target_delta.square().mean(dim=-1).clamp_min(eps).sqrt()
        recon_delta_rms = recon_delta.square().mean(dim=-1).clamp_min(eps).sqrt()
        transient_log_ratio = torch.log(
            recon_delta_rms / target_delta_rms.clamp_min(eps)
        )
        transient_loss = transient_log_ratio.abs().mean()
        transient_ratio = (
            recon_delta_rms / target_delta_rms.clamp_min(eps)
        ).mean()
    else:
        transient_loss = global_loss.new_zeros(())
        transient_ratio = global_ratio

    total = (
        float(frame_weight) * frame_loss
        + float(global_weight) * global_loss
        + float(transient_weight) * transient_loss
    ).to(dtype=result_dtype)
    return {
        "audio_energy_loss": total,
        "audio_frame_log_rms_l1": frame_loss.to(dtype=result_dtype),
        "audio_global_log_rms_l1": global_loss.to(dtype=result_dtype),
        "audio_transient_log_rms_l1": transient_loss.to(dtype=result_dtype),
        "audio_rms_ratio": global_ratio.to(dtype=result_dtype),
        "audio_active_frame_rms_ratio": active_frame_ratio.to(dtype=result_dtype),
        "audio_transient_rms_ratio": transient_ratio.to(dtype=result_dtype),
    }


def _compute_waveform_reconstruction_metrics(
    inputs: torch.Tensor,
    reconstructions: torch.Tensor,
    *,
    config: Mapping[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    compute_visqol: bool = False,
    compute_spectral_metrics: bool = True,
    compute_paper_visqol: bool = False,
) -> dict:
    target = inputs.detach().to(torch.float32)
    recon = reconstructions.detach().to(torch.float32)
    if target.ndim != 3 or recon.ndim != 3 or int(target.size(1)) != 1 or int(recon.size(1)) != 1:
        return {}
    limit = min(int(target.size(0)), int(recon.size(0)))
    if limit <= 0:
        return {}
    # Core train/validation metrics are tensor operations and should stay on the
    # accelerator. Moving every 48 kHz clip to CPU here made an 8-rank job run
    # hundreds of threaded CPU STFTs per step, dominating the actual codec
    # update. Reference tool metrics below copy only their small evaluation
    # inputs when those optional packages require NumPy/CPU arrays.
    target = target[:limit].to(device=device)
    recon = recon[:limit].to(device=device)
    diff = recon - target
    eps = 1.0e-8
    waveform_mse = diff.pow(2).mean()
    waveform_l1 = diff.abs().mean()
    signal = target.pow(2).mean(dim=(1, 2))
    noise = diff.pow(2).mean(dim=(1, 2)).clamp_min(eps)
    snr = (10.0 * torch.log10(signal.clamp_min(eps) / noise)).mean()
    target_centered = target - target.mean(dim=-1, keepdim=True)
    recon_centered = recon - recon.mean(dim=-1, keepdim=True)
    projection_scale = (
        (recon_centered * target_centered).sum(dim=-1, keepdim=True)
        / target_centered.square().sum(dim=-1, keepdim=True).clamp_min(eps)
    )
    projected = projection_scale * target_centered
    residual = recon_centered - projected
    si_sdr = (
        10.0
        * torch.log10(
            projected.square().sum(dim=-1).clamp_min(eps)
            / residual.square().sum(dim=-1).clamp_min(eps)
        )
    ).mean()
    metrics = {
        "audio_waveform_mse": waveform_mse.to(device=device, dtype=dtype),
        "audio_waveform_l1": waveform_l1.to(device=device, dtype=dtype),
        "audio_snr_db": snr.to(device=device, dtype=dtype),
        "audio_si_sdr_db": si_sdr.to(device=device, dtype=dtype),
    }
    # Training already evaluates multi-resolution STFT and seven-scale mel
    # objectives. Repeating another three STFT pairs merely for telemetry on
    # every batch is expensive, particularly because Lightning only publishes
    # step metrics every ``log_every_n_steps``. Callers can therefore sample
    # these diagnostics at the logger cadence while validation keeps computing
    # the complete metric set.
    if compute_spectral_metrics:
        base_fft = int(config["stft_n_fft"])
        stft_metrics = compute_waveform_multires_stft_loss(
            target,
            recon,
            fft_sizes=(max(16, base_fft // 2), base_fft, base_fft * 2),
        )
        logmag_l1 = stft_metrics.get(
            "audio_multires_stft_logmag_l1",
            torch.zeros((), device=device, dtype=torch.float32),
        )
        spectral_convergence = stft_metrics.get(
            "audio_multires_stft_spectral_convergence",
            torch.zeros((), device=device, dtype=torch.float32),
        )
        lsd = logmag_l1.to(torch.float32)
        metrics.update(
            {
                "audio_lsd": lsd.to(device=device, dtype=dtype),
                "audio_log_spectral_distance": lsd.to(device=device, dtype=dtype),
                "audio_logmag_l1": logmag_l1.to(device=device, dtype=dtype),
                "audio_spectral_convergence": spectral_convergence.to(
                    device=device,
                    dtype=dtype,
                ),
            }
        )
    should_compute_visqol = bool(compute_visqol) and is_visqol_available()
    if should_compute_visqol:
        # VCTK crops are two seconds, whereas ViSQOL recommends approximately
        # 8--10 seconds per comparison. Concatenating the equal-length batch
        # yields one stable 8-second paired sample and avoids four subprocesses.
        try:
            score = _measure_visqol(
                target[:limit, 0].clamp(-1.0, 1.0).reshape(-1),
                recon[:limit, 0].clamp(-1.0, 1.0).reshape(-1),
                sample_rate=int(config["sample_rate"]),
                mode=_visqol_mode(
                    int(config["sample_rate"]),
                    dataset=str(config.get("dataset", "") or ""),
                ),
            )
        except Exception:
            score = None
        if score is not None:
            metrics["audio_visqol"] = torch.tensor(
                float(score), dtype=torch.float32, device=device
            ).to(dtype=dtype)
        # Most 24 kHz codec papers report ViSQOL's general-audio model, whose
        # canonical input rate is 48 kHz. VCTK checkpoint selection deliberately
        # remains on speech mode; this separately named value exists solely as
        # a paper-facing comparison column and never changes historical ranking.
        if bool(compute_paper_visqol):
            try:
                paper_score = _measure_visqol(
                    target[:limit, 0].clamp(-1.0, 1.0).reshape(-1),
                    recon[:limit, 0].clamp(-1.0, 1.0).reshape(-1),
                    sample_rate=int(config["sample_rate"]),
                    mode="audio",
                )
            except Exception:
                paper_score = None
            if paper_score is not None:
                metrics["audio_visqol_audio48k"] = torch.tensor(
                    float(paper_score), dtype=torch.float32, device=device
                ).to(dtype=dtype)

    # PESQ + STOI: pip-installable reference-based perceptual metrics, used as the
    # practical alternative to ViSQOL (which has no wheel and needs a bazel C++
    # build). Gated by the same val/test `compute_visqol` flag and computed only
    # when the packages are importable, so train-time and image runs are unaffected.
    sr = int(config["sample_rate"])
    if bool(compute_visqol) and _has_pesq() and sr in (8000, 16000):
        try:
            from pesq import pesq as _pesq_fn

            mode = "wb" if sr == 16000 else "nb"
            pesq_scores = []
            for idx in range(limit):
                try:
                    ref = target[idx, 0].clamp(-1.0, 1.0).cpu().numpy()
                    deg = recon[idx, 0].clamp(-1.0, 1.0).cpu().numpy()
                    pesq_scores.append(float(_pesq_fn(sr, ref, deg, mode)))
                except Exception:
                    continue
            if pesq_scores:
                metrics["audio_pesq"] = torch.tensor(
                    sum(pesq_scores) / len(pesq_scores), dtype=torch.float32
                ).to(device=device, dtype=dtype)
        except Exception:
            pass
    if bool(compute_visqol) and _has_stoi():
        try:
            from pystoi import stoi as _stoi_fn

            stoi_scores = []
            for idx in range(limit):
                try:
                    ref = target[idx, 0].clamp(-1.0, 1.0).cpu().numpy()
                    deg = recon[idx, 0].clamp(-1.0, 1.0).cpu().numpy()
                    stoi_scores.append(float(_stoi_fn(ref, deg, sr, extended=False)))
                except Exception:
                    continue
            if stoi_scores:
                metrics["audio_stoi"] = torch.tensor(
                    sum(stoi_scores) / len(stoi_scores), dtype=torch.float32
                ).to(device=device, dtype=dtype)
        except Exception:
            pass
    return metrics


def compute_audio_reconstruction_metrics(
    inputs: torch.Tensor,
    reconstructions: torch.Tensor,
    *,
    audio_meta: Mapping[str, Any],
    audio_source: Any,
    compute_visqol: bool = False,
    compute_spectral_metrics: bool = True,
    compute_paper_visqol: bool = False,
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
            compute_visqol=compute_visqol,
            compute_spectral_metrics=compute_spectral_metrics,
            compute_paper_visqol=compute_paper_visqol,
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
    should_compute_visqol = bool(compute_visqol) and is_visqol_available()

    logmag_mses = []
    log_spectral_distances = []
    logmag_l1s = []
    spectral_convergences = []
    logmel_l1s = []
    visqol_scores = []
    visqol_references = []
    visqol_degraded = []
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
                    visqol_references.append(original_waveform.reshape(-1))
                    visqol_degraded.append(recon_waveform.reshape(-1))
                except Exception:
                    pass

    if visqol_references and len(visqol_references) == len(visqol_degraded):
        try:
            visqol_score = _measure_visqol(
                torch.cat(visqol_references),
                torch.cat(visqol_degraded),
                sample_rate=int(config["sample_rate"]),
                mode=_visqol_mode(
                    int(config["sample_rate"]),
                    dataset=str(config.get("dataset", "") or ""),
                ),
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
    """Compute lightweight distribution and health metrics for generated audio."""
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
    generated_rms = []
    real_rms = []
    generated_peak = []
    generated_clip_fraction = []
    generated_silence = []
    generated_zcr = []
    real_zcr = []

    def _waveform_stats(waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        wave = waveform.to(torch.float32).reshape(-1)
        if wave.numel() == 0:
            raise ValueError("empty waveform")
        rms = wave.pow(2).mean().sqrt()
        peak = wave.abs().max()
        clip_fraction = (wave.abs() >= 0.999).to(torch.float32).mean()
        silence = (rms < 1.0e-4).to(torch.float32)
        if wave.numel() > 1:
            zcr = (wave[1:].signbit() != wave[:-1].signbit()).to(torch.float32).mean()
        else:
            zcr = wave.new_zeros(())
        return {
            "rms": rms,
            "peak": peak,
            "clip_fraction": clip_fraction,
            "silence": silence,
            "zcr": zcr,
        }

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
                gen_stats = _waveform_stats(generated_waveform)
                real_stats = _waveform_stats(real_waveform)
                generated_rms.append(gen_stats["rms"])
                real_rms.append(real_stats["rms"])
                generated_peak.append(gen_stats["peak"])
                generated_clip_fraction.append(gen_stats["clip_fraction"])
                generated_silence.append(gen_stats["silence"])
                generated_zcr.append(gen_stats["zcr"])
                real_zcr.append(real_stats["zcr"])
            except Exception:
                continue

    if not generated_features or not real_features:
        return {}

    generated_stack = torch.stack(generated_features, dim=0)
    real_stack = torch.stack(real_features, dim=0)
    frechet = _diag_frechet_distance(real_stack, generated_stack).to(device=device, dtype=dtype)
    mean_l1 = (real_stack.mean(dim=0) - generated_stack.mean(dim=0)).abs().mean().to(device=device, dtype=dtype)
    gen_rms = torch.stack(generated_rms).to(device=device, dtype=dtype)
    ref_rms = torch.stack(real_rms).to(device=device, dtype=dtype)
    gen_zcr = torch.stack(generated_zcr).to(device=device, dtype=dtype)
    ref_zcr = torch.stack(real_zcr).to(device=device, dtype=dtype)
    return {
        "audio_generation_logmel_frechet": frechet,
        "audio_generation_logmel_mean_l1": mean_l1,
        "audio_generation_rms_mean": gen_rms.mean(),
        "audio_generation_rms_std": gen_rms.std(unbiased=False),
        "audio_generation_rms_mean_l1": (gen_rms.mean() - ref_rms.mean()).abs(),
        "audio_generation_peak_mean": torch.stack(generated_peak).to(device=device, dtype=dtype).mean(),
        "audio_generation_clip_fraction": torch.stack(generated_clip_fraction).to(device=device, dtype=dtype).mean(),
        "audio_generation_silence_fraction": torch.stack(generated_silence).to(device=device, dtype=dtype).mean(),
        "audio_generation_zcr_mean": gen_zcr.mean(),
        "audio_generation_zcr_mean_l1": (gen_zcr.mean() - ref_zcr.mean()).abs(),
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


def _write_audio_preview_pair(
    artifact_dir: Any,
    split: str,
    basename: str,
    original_waveform: torch.Tensor,
    reconstructed_waveform: torch.Tensor,
    *,
    sample_rate: int,
) -> None:
    if artifact_dir in (None, ""):
        return
    try:
        root = _audio_artifact_root(artifact_dir, split)
        _write_wav_audio_file(root, f"{basename}_original", original_waveform, sample_rate=sample_rate)
        _write_wav_audio_file(root, f"{basename}_reconstructed", reconstructed_waveform, sample_rate=sample_rate)
    except Exception:
        # W&B media still carries the payload; local WAV export is best-effort.
        return


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


def normalize_generated_waveform_for_preview(waveform: torch.Tensor, config: Mapping[str, Any]) -> torch.Tensor:
    """Apply listening-only RMS normalization to generated audio previews."""
    return _rms_normalize_waveform(
        waveform.detach().to(torch.float32).reshape(-1).clamp(-1.0, 1.0),
        enabled=bool(config.get("generated_audio_rms_normalize", config.get("audio_rms_normalize", False))),
        target_rms=float(config.get("generated_audio_target_rms", config.get("audio_target_rms", 0.12))),
        max_gain=float(config.get("generated_audio_max_gain", max(float(config.get("audio_max_gain", 8.0)), 64.0))),
        peak_limit=float(config.get("generated_audio_target_peak", config.get("audio_target_peak", 0.95))),
    )


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
    captions: Optional[list[str]] = None,
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
            waveform = normalize_generated_waveform_for_preview(waveform, config)
        except Exception:
            continue
        audio_items.append(_wandb_audio_array(waveform))
        if captions is not None and idx < len(captions) and str(captions[idx]).strip():
            audio_captions.append(str(captions[idx]))
        else:
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
            _write_audio_preview_pair(
                artifact_dir,
                split,
                basename,
                original_waveform,
                recon_waveform,
                sample_rate=int(config["sample_rate"]),
            )
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
        _write_audio_preview_pair(
            artifact_dir,
            split,
            basename,
            original_waveform,
            recon_waveform,
            sample_rate=int(config["sample_rate"]),
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
