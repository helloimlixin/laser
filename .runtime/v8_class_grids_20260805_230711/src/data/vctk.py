import math
import os
import wave
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import lightning as pl
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import wavfile
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, Dataset

from src.data.config import DataConfig


WAV_EXTENSIONS = {".wav"}
FLAC_EXTENSIONS = {".flac"}
CROP_MODE_SLICE = 0
CROP_MODE_PAD = 1


def _vctk_utterance_stem(path: Union[str, Path]) -> str:
    stem = Path(path).stem
    pieces = stem.split("_")
    if len(pieces) >= 2 and pieces[0].startswith("p"):
        return f"{pieces[0]}_{pieces[1]}"
    return stem


def _find_vctk_transcript_path(path: Union[str, Path]) -> Path | None:
    audio_path = Path(path)
    speaker = audio_path.parent.name
    utterance = _vctk_utterance_stem(audio_path)
    candidates = []
    for root in (audio_path.parent, *audio_path.parents):
        candidates.append(root / "txt" / speaker / f"{utterance}.txt")
        candidates.append(root / "VCTK-Corpus" / "txt" / speaker / f"{utterance}.txt")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _read_vctk_transcript(path: Union[str, Path]) -> str:
    transcript = _find_vctk_transcript_path(path)
    if transcript is None:
        return ""
    try:
        return transcript.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return ""


def _target_hw(image_size: Union[int, Sequence[int]]) -> Tuple[int, int]:
    if isinstance(image_size, int):
        side = int(image_size)
        return side, side
    if len(image_size) != 2:
        raise ValueError(f"image_size must be an int or length-2 sequence, got {image_size!r}")
    return int(image_size[0]), int(image_size[1])


def _normalize_pcm(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    elif samples.ndim != 1:
        raise ValueError(f"Unsupported audio shape {samples.shape}; expected mono or multi-channel waveform")

    if np.issubdtype(samples.dtype, np.floating):
        audio = samples.astype(np.float32, copy=False)
    elif np.issubdtype(samples.dtype, np.signedinteger):
        scale = float(max(np.iinfo(samples.dtype).max, 1))
        audio = samples.astype(np.float32) / scale
    elif np.issubdtype(samples.dtype, np.unsignedinteger):
        info = np.iinfo(samples.dtype)
        midpoint = (info.max + 1) / 2.0
        audio = (samples.astype(np.float32) - midpoint) / midpoint
    else:
        raise TypeError(f"Unsupported PCM dtype {samples.dtype}")

    return np.clip(audio, -1.0, 1.0)


def _resample_if_needed(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if int(src_sr) == int(dst_sr):
        return audio.astype(np.float32, copy=False)
    if int(src_sr) <= 0 or int(dst_sr) <= 0:
        raise ValueError(f"Sample rates must be positive, got src={src_sr} dst={dst_sr}")
    divisor = math.gcd(int(src_sr), int(dst_sr))
    up = int(dst_sr) // divisor
    down = int(src_sr) // divisor
    return resample_poly(audio, up, down).astype(np.float32, copy=False)


def _preprocess_waveform(
    audio: np.ndarray,
    *,
    dc_remove: bool,
    peak_normalize: bool,
    target_peak: float,
) -> np.ndarray:
    audio = audio.astype(np.float32, copy=False)
    if bool(dc_remove) and audio.size:
        audio = audio - float(np.mean(audio, dtype=np.float64))
    if bool(peak_normalize) and audio.size:
        peak = float(np.max(np.abs(audio)))
        if peak > 1.0e-8:
            audio = audio * (float(target_peak) / peak)
    return np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)


def _apply_edge_fade(waveform: torch.Tensor, fade_samples: int) -> torch.Tensor:
    fade = min(int(fade_samples), int(waveform.numel()) // 2)
    if fade <= 0:
        return waveform
    out = waveform.clone()
    ramp = torch.linspace(0.0, 1.0, steps=fade, dtype=out.dtype, device=out.device)
    out[:fade] = out[:fade] * ramp
    out[-fade:] = out[-fade:] * ramp.flip(0)
    return out


def _rms_normalize_waveform(
    waveform: torch.Tensor,
    *,
    enabled: bool,
    target_rms: float,
    max_gain: float,
    peak_limit: float = 1.0,
) -> torch.Tensor:
    if not bool(enabled):
        return waveform
    out = waveform.to(torch.float32)
    target = float(target_rms)
    if target <= 0.0 or out.numel() <= 0:
        return out
    rms = out.pow(2).mean().sqrt()
    if float(rms.item()) <= 1.0e-8:
        return out
    gain = target / rms
    gain_cap = float(max_gain)
    if gain_cap > 0.0:
        gain = torch.minimum(gain, torch.tensor(gain_cap, dtype=out.dtype, device=out.device))
    out = out * gain
    peak = out.abs().max()
    limit = max(1.0e-8, min(float(peak_limit), 1.0))
    if float(peak.item()) > limit:
        out = out * (limit / peak)
    return out.clamp(-1.0, 1.0)


def _read_audio_file(path: Union[str, Path]) -> Tuple[int, np.ndarray]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".wav":
        sample_rate, samples = wavfile.read(path)
        return int(sample_rate), samples
    if suffix == ".flac":
        try:
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError(
                "Reading VCTK FLAC files requires soundfile. Install soundfile or convert the corpus to WAV."
            ) from exc
        samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
        return int(sample_rate), samples
    raise ValueError(f"Unsupported VCTK audio extension for {path}")


def _audio_duration_seconds(path: Union[str, Path]) -> float | None:
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        if suffix == ".wav":
            with wave.open(str(path), "rb") as handle:
                rate = int(handle.getframerate())
                frames = int(handle.getnframes())
            return None if rate <= 0 else frames / float(rate)
        if suffix == ".flac":
            try:
                import soundfile as sf
                info = sf.info(str(path))
                rate = int(info.samplerate)
                frames = int(info.frames)
                return None if rate <= 0 else frames / float(rate)
            except Exception:
                sample_rate, samples = _read_audio_file(path)
                return None if int(sample_rate) <= 0 else int(np.asarray(samples).shape[0]) / float(sample_rate)
    except Exception:
        return None
    return None


class VCTKSpectrogramDataset(Dataset):
    def __init__(
        self,
        paths: Sequence[Path],
        config: DataConfig,
        *,
        train: bool,
        speaker_to_index: dict[str, int] | None = None,
    ):
        if not paths:
            raise RuntimeError("No VCTK audio files provided for dataset split.")
        self.paths = [Path(p) for p in paths]
        self.train = bool(train)
        self.sample_rate = int(config.sample_rate)
        self.audio_num_samples = int(config.audio_num_samples)
        self.n_fft = int(config.stft_n_fft)
        self.hop_length = int(config.stft_hop_length)
        self.win_length = int(config.stft_win_length or config.stft_n_fft)
        self.power = float(config.stft_power)
        self.log_offset = float(config.stft_log_offset)
        self.target_hw = _target_hw(config.image_size)
        self.augment = bool(config.augment)
        self.mean = torch.tensor(tuple(float(x) for x in config.mean), dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(tuple(float(x) for x in config.std), dtype=torch.float32).view(-1, 1, 1)
        if tuple(self.mean.shape) != tuple(self.std.shape):
            raise ValueError(f"mean/std shape mismatch: {tuple(self.mean.shape)} vs {tuple(self.std.shape)}")
        if self.mean.shape[0] != 1:
            raise ValueError(
                f"VCTK spectrogram dataset expects single-channel mean/std, got {self.mean.shape[0]} channels"
            )
        if self.audio_num_samples <= 0:
            raise ValueError(f"audio_num_samples must be positive, got {self.audio_num_samples}")
        if self.n_fft <= 0 or self.hop_length <= 0 or self.win_length <= 0:
            raise ValueError(
                "stft_n_fft, stft_hop_length, and stft_win_length must all be positive "
                f"(got n_fft={self.n_fft}, hop={self.hop_length}, win={self.win_length})"
            )
        if self.win_length > self.n_fft:
            raise ValueError(f"stft_win_length ({self.win_length}) must be <= stft_n_fft ({self.n_fft})")
        self.window = torch.hann_window(self.win_length, periodic=True)
        self.texts = [_read_vctk_transcript(path) for path in self.paths]
        self.speaker_to_index = dict(speaker_to_index or {
            speaker: idx for idx, speaker in enumerate(sorted({path.parent.name for path in self.paths}))
        })

    def __len__(self) -> int:
        return len(self.paths)

    def _crop_or_pad(self, waveform: np.ndarray) -> Tuple[torch.Tensor, dict]:
        audio = torch.from_numpy(waveform.astype(np.float32, copy=False))
        length = int(audio.numel())
        target = self.audio_num_samples
        if length >= target:
            max_offset = length - target
            if self.train and self.augment and max_offset > 0:
                start = int(torch.randint(0, max_offset + 1, (1,)).item())
            else:
                start = max_offset // 2
            return audio[start:start + target], {
                "crop_mode": CROP_MODE_SLICE,
                "crop_offset": start,
                "source_num_samples": length,
            }

        padded = torch.zeros(target, dtype=torch.float32)
        pad_total = target - length
        if self.train and self.augment and pad_total > 0:
            start = int(torch.randint(0, pad_total + 1, (1,)).item())
        else:
            start = pad_total // 2
        padded[start:start + length] = audio
        return padded, {
            "crop_mode": CROP_MODE_PAD,
            "crop_offset": start,
            "source_num_samples": length,
        }

    def _waveform_to_spectrogram(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        ).abs()
        if self.power != 1.0:
            spec = spec.pow(self.power)
        spec = torch.log(spec + self.log_offset)
        original_shape = (int(spec.size(0)), int(spec.size(1)))
        spec = spec.unsqueeze(0).unsqueeze(0)
        spec = F.interpolate(spec, size=self.target_hw, mode="bilinear", align_corners=False).squeeze(0)
        spec_min = spec.amin(dim=(-2, -1), keepdim=True)
        spec_max = spec.amax(dim=(-2, -1), keepdim=True)
        spec = (spec - spec_min) / (spec_max - spec_min + 1e-6)
        normalized = (spec - self.mean) / self.std
        return normalized, {
            "spec_min": float(spec_min.squeeze().item()),
            "spec_max": float(spec_max.squeeze().item()),
            "spec_shape": original_shape,
        }

    def __getitem__(self, index: int):
        path = self.paths[index]
        sample_rate, samples = _read_audio_file(path)
        waveform = _normalize_pcm(samples)
        waveform = _resample_if_needed(waveform, int(sample_rate), self.sample_rate)
        waveform, crop_meta = self._crop_or_pad(waveform)
        spectrogram, spec_meta = self._waveform_to_spectrogram(waveform)
        meta = {
            "path": str(path),
            "speaker_id": path.parent.name,
            "speaker_index": torch.tensor(
                int(self.speaker_to_index.get(path.parent.name, 0)),
                dtype=torch.int64,
            ),
            "text": self.texts[index],
            "crop_mode": torch.tensor(int(crop_meta["crop_mode"]), dtype=torch.int64),
            "crop_offset": torch.tensor(int(crop_meta["crop_offset"]), dtype=torch.int64),
            "source_num_samples": torch.tensor(int(crop_meta["source_num_samples"]), dtype=torch.int64),
            "spec_min": torch.tensor(float(spec_meta["spec_min"]), dtype=torch.float32),
            "spec_max": torch.tensor(float(spec_meta["spec_max"]), dtype=torch.float32),
            "spec_shape": torch.tensor(spec_meta["spec_shape"], dtype=torch.int64),
        }
        return spectrogram, 0, meta


class VCTKWaveformDataset(Dataset):
    def __init__(
        self,
        paths: Sequence[Path],
        config: DataConfig,
        *,
        train: bool,
        speaker_to_index: dict[str, int] | None = None,
    ):
        if not paths:
            raise RuntimeError("No VCTK audio files provided for dataset split.")
        self.paths = [Path(p) for p in paths]
        self.train = bool(train)
        self.sample_rate = int(config.sample_rate)
        self.audio_num_samples = int(config.audio_num_samples)
        self.augment = bool(config.augment)
        self.dc_remove = bool(getattr(config, "audio_dc_remove", False))
        self.peak_normalize = bool(getattr(config, "audio_peak_normalize", False))
        self.target_peak = float(getattr(config, "audio_target_peak", 0.95))
        self.rms_normalize = bool(getattr(config, "audio_rms_normalize", False))
        self.target_rms = float(getattr(config, "audio_target_rms", 0.12))
        self.max_gain = float(getattr(config, "audio_max_gain", 8.0))
        self.min_crop_rms = max(0.0, float(getattr(config, "audio_min_crop_rms", 0.0)))
        self.crop_attempts = max(1, int(getattr(config, "audio_crop_attempts", 1)))
        self.fade_samples = max(0, int(getattr(config, "audio_fade_samples", 0)))
        if self.audio_num_samples <= 0:
            raise ValueError(f"audio_num_samples must be positive, got {self.audio_num_samples}")
        self.texts = [_read_vctk_transcript(path) for path in self.paths]
        self.speaker_to_index = dict(speaker_to_index or {
            speaker: idx for idx, speaker in enumerate(sorted({path.parent.name for path in self.paths}))
        })

    def __len__(self) -> int:
        return len(self.paths)

    def _crop_rms(self, audio: torch.Tensor, start: int, target: int) -> float:
        clip = audio[int(start):int(start) + int(target)]
        if clip.numel() <= 0:
            return 0.0
        return float(clip.to(torch.float32).pow(2).mean().sqrt().item())

    def _select_crop_start(self, audio: torch.Tensor, *, max_offset: int, target: int) -> int:
        if self.min_crop_rms <= 0.0 or max_offset <= 0:
            if self.train and self.augment:
                return int(torch.randint(0, max_offset + 1, (1,)).item())
            return max_offset // 2

        attempts = min(max_offset + 1, self.crop_attempts)
        if self.train and self.augment:
            starts = torch.randint(0, max_offset + 1, (attempts,), dtype=torch.int64).tolist()
        elif attempts <= 1:
            starts = [max_offset // 2]
        else:
            starts = torch.linspace(0, max_offset, steps=attempts).round().to(torch.int64).tolist()

        best_start = int(starts[0])
        best_rms = -1.0
        for raw_start in starts:
            start = int(raw_start)
            rms = self._crop_rms(audio, start, target)
            if rms >= self.min_crop_rms:
                return start
            if rms > best_rms:
                best_start = start
                best_rms = rms
        return best_start

    def _crop_or_pad(self, waveform: np.ndarray) -> Tuple[torch.Tensor, dict]:
        audio = torch.from_numpy(waveform.astype(np.float32, copy=False))
        length = int(audio.numel())
        target = self.audio_num_samples
        if length >= target:
            max_offset = length - target
            start = self._select_crop_start(audio, max_offset=max_offset, target=target)
            return _apply_edge_fade(audio[start:start + target], self.fade_samples), {
                "crop_mode": CROP_MODE_SLICE,
                "crop_offset": start,
                "source_num_samples": length,
            }

        padded = torch.zeros(target, dtype=torch.float32)
        pad_total = target - length
        if self.train and self.augment and pad_total > 0:
            start = int(torch.randint(0, pad_total + 1, (1,)).item())
        else:
            start = pad_total // 2
        padded[start:start + length] = audio
        return _apply_edge_fade(padded, self.fade_samples), {
            "crop_mode": CROP_MODE_PAD,
            "crop_offset": start,
            "source_num_samples": length,
        }

    def __getitem__(self, index: int):
        path = self.paths[index]
        sample_rate, samples = _read_audio_file(path)
        waveform = _normalize_pcm(samples)
        waveform = _resample_if_needed(waveform, int(sample_rate), self.sample_rate)
        waveform = _preprocess_waveform(
            waveform,
            dc_remove=self.dc_remove,
            peak_normalize=self.peak_normalize,
            target_peak=self.target_peak,
        )
        waveform, crop_meta = self._crop_or_pad(waveform)
        waveform = _rms_normalize_waveform(
            waveform,
            enabled=self.rms_normalize,
            target_rms=self.target_rms,
            max_gain=self.max_gain,
            peak_limit=self.target_peak,
        )
        meta = {
            "path": str(path),
            "speaker_id": path.parent.name,
            "speaker_index": torch.tensor(
                int(self.speaker_to_index.get(path.parent.name, 0)),
                dtype=torch.int64,
            ),
            "text": self.texts[index],
            "crop_mode": torch.tensor(int(crop_meta["crop_mode"]), dtype=torch.int64),
            "crop_offset": torch.tensor(int(crop_meta["crop_offset"]), dtype=torch.int64),
            "source_num_samples": torch.tensor(int(crop_meta["source_num_samples"]), dtype=torch.int64),
            "spec_min": torch.tensor(0.0, dtype=torch.float32),
            "spec_max": torch.tensor(1.0, dtype=torch.float32),
            "spec_shape": torch.tensor([0, int(waveform.numel())], dtype=torch.int64),
            "audio_format": "waveform",
        }
        return waveform.unsqueeze(0), 0, meta


class VCTKDataModule(pl.LightningDataModule):
    def __init__(self, config: DataConfig):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.speaker_ids: list[str] = []
        self.speaker_to_index: dict[str, int] = {}
        self.num_speakers = 0

    def _loader_generator(self, offset: int = 0) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(int(self.config.seed) + int(offset))
        return generator

    def _list_audio_files(self, root: Union[str, Path]) -> List[Path]:
        base = Path(root)
        wav_paths = sorted(
            path for path in base.rglob("*")
            if path.is_file() and path.suffix.lower() in WAV_EXTENSIONS
        )
        flac_paths = sorted(
            path for path in base.rglob("*")
            if path.is_file() and path.suffix.lower() in FLAC_EXTENSIONS
        )
        audio_paths = wav_paths + flac_paths
        if audio_paths:
            return audio_paths
        raise RuntimeError(f"No WAV/FLAC audio files found under {base}")

    def _filter_audio_files(self, paths: Sequence[Path]) -> List[Path]:
        min_duration = max(0.0, float(getattr(self.config, "audio_min_duration_seconds", 0.0) or 0.0))
        max_duration = max(0.0, float(getattr(self.config, "audio_max_duration_seconds", 0.0) or 0.0))
        require_text = bool(getattr(self.config, "audio_require_text", False))
        if min_duration <= 0.0 and max_duration <= 0.0 and not require_text:
            return list(paths)

        kept: list[Path] = []
        dropped_no_text = 0
        dropped_duration = 0
        for path in paths:
            if require_text and not _read_vctk_transcript(path).strip():
                dropped_no_text += 1
                continue
            duration = _audio_duration_seconds(path)
            if duration is None:
                if min_duration > 0.0 or max_duration > 0.0:
                    dropped_duration += 1
                    continue
            else:
                if min_duration > 0.0 and duration < min_duration:
                    dropped_duration += 1
                    continue
                if max_duration > 0.0 and duration > max_duration:
                    dropped_duration += 1
                    continue
            kept.append(path)

        if not kept:
            raise RuntimeError(
                "VCTK duration/text filters removed every audio file "
                f"(min_duration={min_duration}, max_duration={max_duration}, require_text={require_text})."
            )
        if len(kept) != len(paths):
            print(
                "VCTK filter kept "
                f"{len(kept)}/{len(paths)} files "
                f"(dropped_no_text={dropped_no_text}, dropped_duration={dropped_duration}, "
                f"min_duration={min_duration}, max_duration={max_duration}, require_text={require_text})"
            )
        return kept

    def _resolve_data_dir(self) -> str:
        flac_error = None

        def is_vctk_dir(path_str: str) -> bool:
            nonlocal flac_error
            if not path_str:
                return False
            path = Path(path_str)
            if not path.exists() or not path.is_dir():
                return False
            try:
                self._list_audio_files(path)
                return True
            except RuntimeError as exc:
                if "FLAC files" in str(exc):
                    flac_error = exc
                return False

        configured_dir = str(self.config.data_dir) if getattr(self.config, "data_dir", "") else ""
        configured_path = Path(configured_dir).expanduser() if configured_dir else None
        env_path = Path(os.environ["VCTK_DIR"]).expanduser() if os.environ.get("VCTK_DIR") else None
        runtime_data = Path.cwd() / ".." / "data"
        project_data = Path(__file__).resolve().parents[3] / "data"
        candidates = [
            str(env_path) if env_path is not None else "",
            configured_dir,
            str(configured_path / "VCTK-Corpus") if configured_path is not None else "",
            str(configured_path / "VCTK-Corpus-0.92") if configured_path is not None else "",
            str(configured_path / "wav48") if configured_path is not None else "",
            str(configured_path / "wav48_silence_trimmed") if configured_path is not None else "",
            str(configured_path.parent / "VCTK-Corpus") if configured_path is not None else "",
            str(configured_path.parent / "VCTK-Corpus-0.92") if configured_path is not None else "",
            str(configured_path.parent / "vctk") if configured_path is not None else "",
            str(configured_path.parent / "VCTK") if configured_path is not None else "",
            str(configured_path.parent / "vctk-corpus") if configured_path is not None else "",
            str(configured_path.parent / "vctk-corpus-0.92") if configured_path is not None else "",
            str(runtime_data / "vctk"),
            str(runtime_data / "VCTK"),
            str(runtime_data / "VCTK-Corpus"),
            str(runtime_data / "VCTK-Corpus-0.92"),
            str(project_data / "vctk"),
            str(project_data / "VCTK"),
            str(project_data / "VCTK-Corpus"),
            str(project_data / "VCTK-Corpus-0.92"),
            "/home/xl598/Data/vctk",
            "/home/xl598/Data/VCTK",
            "/home/xl598/Data/VCTK-Corpus",
            "/home/xl598/Data/VCTK-Corpus-0.92",
        ]
        for candidate in candidates:
            if is_vctk_dir(candidate):
                return candidate
        if flac_error is not None:
            raise flac_error
        raise RuntimeError(
            "VCTK data not found. Set VCTK_DIR/data.data_dir to a directory containing VCTK WAV files "
            "or a standard VCTK root such as VCTK-Corpus, VCTK-Corpus-0.92, wav48, or wav48_silence_trimmed."
        )

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if self.train_dataset is not None:
            return
        data_dir = self._resolve_data_dir()
        audio_paths = self._list_audio_files(data_dir)
        audio_paths = self._filter_audio_files(audio_paths)
        num_items = len(audio_paths)
        if num_items < 3:
            raise RuntimeError("VCTK dataset must contain at least three WAV files for train/val/test splits.")
        self.speaker_ids = sorted({path.parent.name for path in audio_paths})
        self.speaker_to_index = {speaker: idx for idx, speaker in enumerate(self.speaker_ids)}
        self.num_speakers = len(self.speaker_ids)

        generator = torch.Generator().manual_seed(int(self.config.seed))
        indices = torch.randperm(num_items, generator=generator)
        num_val = max(1, int(round(0.05 * num_items)))
        num_test = max(1, int(round(0.05 * num_items)))
        num_val = min(num_val, num_items - 2)
        num_test = min(num_test, num_items - num_val - 1)
        num_train = num_items - num_val - num_test
        train_idx = indices[:num_train].tolist()
        val_idx = indices[num_train:num_train + num_val].tolist()
        test_idx = indices[num_train + num_val:num_train + num_val + num_test].tolist()

        def gather(idxs: Sequence[int]) -> List[Path]:
            return [audio_paths[int(i)] for i in idxs]

        representation = str(getattr(self.config, "audio_representation", "spectrogram")).strip().lower()
        if representation in {"waveform", "raw", "wav"}:
            dataset_cls = VCTKWaveformDataset
        elif representation in {"spectrogram", "stft", "logmag", ""}:
            dataset_cls = VCTKSpectrogramDataset
        else:
            raise ValueError(
            "Unsupported audio_representation "
            f"{representation!r}; expected 'spectrogram' or 'waveform'"
        )
        self.train_dataset = dataset_cls(
            gather(train_idx),
            self.config,
            train=True,
            speaker_to_index=self.speaker_to_index,
        )
        self.val_dataset = dataset_cls(
            gather(val_idx),
            self.config,
            train=False,
            speaker_to_index=self.speaker_to_index,
        )
        self.test_dataset = dataset_cls(
            gather(test_idx),
            self.config,
            train=False,
            speaker_to_index=self.speaker_to_index,
        )

    def train_dataloader(self):
        return self._build_loader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            seed_offset=0,
        )

    def val_dataloader(self):
        val_workers = min(2, self.config.num_workers) if self.config.num_workers > 0 else 0
        return self._build_loader(
            dataset=self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=val_workers,
            seed_offset=1,
        )

    def test_dataloader(self):
        test_workers = min(2, self.config.num_workers) if self.config.num_workers > 0 else 0
        return self._build_loader(
            dataset=self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=test_workers,
            seed_offset=2,
        )

    def _build_loader(self, dataset, batch_size, shuffle, num_workers, seed_offset):
        if dataset is None:
            return None
        kwargs = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=(num_workers > 0),
            generator=self._loader_generator(seed_offset),
        )
        if num_workers > 0:
            kwargs["timeout"] = 120
            kwargs["multiprocessing_context"] = "spawn"
        else:
            kwargs["timeout"] = 0
        return DataLoader(**kwargs)
