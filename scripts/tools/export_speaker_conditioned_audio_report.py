#!/usr/bin/env python3
"""Export labeled speaker-conditioned VCTK generation previews and mel plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import wave

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.audio_logging import (
    AUDIO_META_KEYS,
    _load_cropped_waveform,
    _mel_db,
    audio_config_from_source,
    normalize_generated_waveform_for_preview,
)
from src.data.token_cache import load_token_cache
from src.s2 import load_run, sample


def _safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._") or "item"


def _write_wav(path: Path, waveform: torch.Tensor, *, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (
        waveform.detach()
        .cpu()
        .to(torch.float32)
        .reshape(-1)
        .clamp(-1.0, 1.0)
        .mul(32767.0)
        .round()
        .to(torch.int16)
        .numpy()
    )
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())


def _read_wav(path: Path) -> tuple[int, torch.Tensor]:
    with wave.open(str(path), "rb") as handle:
        channels = int(handle.getnchannels())
        sample_width = int(handle.getsampwidth())
        sample_rate = int(handle.getframerate())
        frames = handle.readframes(int(handle.getnframes()))
    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM WAV is supported for generated-dir input, got {sample_width} bytes")
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32767.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return sample_rate, torch.from_numpy(audio.copy()).to(torch.float32)


def _save_mel(path: Path, waveform: torch.Tensor, config: dict, *, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    mel = _mel_db(waveform.detach().cpu().to(torch.float32).reshape(-1), config)
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    ax.imshow(mel, origin="lower", aspect="auto", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("frame")
    ax.set_ylabel("mel bin")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_mel_comparison(path: Path, panels: list[tuple[str, torch.Tensor]], config: dict, *, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    cols = max(1, len(panels))
    fig, axes = plt.subplots(1, cols, figsize=(4.2 * cols, 3.4), squeeze=False)
    for ax, (panel_title, waveform) in zip(axes[0], panels):
        mel = _mel_db(waveform.detach().cpu().to(torch.float32).reshape(-1), config)
        ax.imshow(mel, origin="lower", aspect="auto", cmap="magma")
        ax.set_title(panel_title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _class_names(cache: dict) -> list[str]:
    raw = (cache.get("meta", {}) or {}).get("class_names")
    if isinstance(raw, dict):
        ordered = sorted(((int(value), str(key)) for key, value in raw.items()), key=lambda item: item[0])
        return [name for _, name in ordered]
    if isinstance(raw, (list, tuple)):
        return [str(item) for item in raw]
    return []


def _parse_labels(raw: str | None, *, class_names: list[str], n: int) -> list[int]:
    if raw is None or str(raw).strip() == "":
        base = list(range(min(max(0, int(n)), len(class_names)))) if class_names else list(range(max(0, int(n))))
    else:
        name_to_id = {name: idx for idx, name in enumerate(class_names)}
        base = []
        for item in re.split(r"[,\s]+", str(raw).strip()):
            if not item:
                continue
            if item in name_to_id:
                base.append(int(name_to_id[item]))
            else:
                base.append(int(item))
    if not base:
        raise ValueError("No speaker/class labels were provided or inferred.")
    repeats = int(np.ceil(float(max(1, int(n))) / float(len(base))))
    return (base * repeats)[: int(n)]


def _speaker_name(class_names: list[str], label: int) -> str:
    if 0 <= int(label) < len(class_names):
        return str(class_names[int(label)])
    return f"class{int(label)}"


def _meta_item(audio_meta: dict, index: int) -> dict:
    item = {}
    for key in AUDIO_META_KEYS:
        value = audio_meta[key]
        if torch.is_tensor(value):
            item[key] = value[index].detach().cpu()
        elif isinstance(value, np.ndarray):
            item[key] = value[index]
        elif isinstance(value, (list, tuple)):
            item[key] = value[index]
        else:
            item[key] = value
    return item


def _matching_original_indices(cache: dict, label: int, limit: int) -> list[int]:
    labels = cache.get("class_labels")
    if labels is None:
        return []
    labels = torch.as_tensor(labels, dtype=torch.long).reshape(-1)
    hits = torch.nonzero(labels == int(label), as_tuple=False).reshape(-1)
    return [int(idx) for idx in hits[: max(0, int(limit))].tolist()]


def _wave_stats(waveform: torch.Tensor) -> dict[str, float]:
    wave = waveform.detach().cpu().to(torch.float32).reshape(-1)
    if wave.numel() == 0:
        return {"rms": 0.0, "peak": 0.0, "zcr": 0.0}
    rms = float(wave.pow(2).mean().sqrt().item())
    peak = float(wave.abs().max().item())
    zcr = 0.0
    if wave.numel() > 1:
        zcr = float((wave[1:].signbit() != wave[:-1].signbit()).to(torch.float32).mean().item())
    return {"rms": rms, "peak": peak, "zcr": zcr}


def export_report(args: argparse.Namespace) -> Path:
    torch.manual_seed(int(args.seed))
    generated_dir = Path(str(args.generated_dir or "")).expanduser()
    use_existing_generated = bool(str(args.generated_dir or "").strip())
    if use_existing_generated:
        cache = load_token_cache(args.token_cache)
    else:
        if not str(args.ckpt or "").strip():
            raise ValueError("--ckpt is required unless --generated-dir is provided.")
        run = load_run(
            ckpt=args.ckpt,
            cache_pt=args.token_cache,
            out_root=args.out_root,
            ar_dir=args.ar_dir,
            dev=args.device,
        )
        cache = run.cache
    class_names = _class_names(cache)
    labels = _parse_labels(args.class_labels, class_names=class_names, n=int(args.num_samples))

    config = audio_config_from_source(cache.get("meta", {}) or {})
    sample_rate = int(config["sample_rate"])
    if use_existing_generated:
        wav_paths = sorted(generated_dir.glob("*.wav"))
        if not wav_paths:
            raise FileNotFoundError(f"No generated WAV files found under {generated_dir}")
        wav_paths = wav_paths[: int(args.num_samples)]
        labels = labels[: len(wav_paths)]
        generated_items = []
        for wav_path in wav_paths:
            wav_sample_rate, waveform = _read_wav(wav_path)
            if int(wav_sample_rate) != sample_rate:
                raise ValueError(f"{wav_path} has sample rate {wav_sample_rate}, expected {sample_rate}")
            generated_items.append(waveform)
        generated = torch.stack(generated_items, dim=0)
    else:
        labels_tensor = torch.as_tensor(labels, device=run.dev, dtype=torch.long)
        batch = sample(
            run.net,
            run.s1,
            run.shape,
            n=int(args.num_samples),
            temp=float(args.temperature),
            top_k=int(args.top_k),
            ctemp=args.coeff_temperature,
            cmode=args.coeff_mode,
            class_labels=labels_tensor,
            dev=run.dev,
        )
        if not torch.is_tensor(batch.imgs) or batch.imgs.ndim != 3 or int(batch.imgs.size(1)) != 1:
            raise ValueError(f"Expected waveform samples [B, 1, T], got {tuple(batch.imgs.shape)}")
        generated = batch.imgs.detach().cpu().to(torch.float32)[:, 0]
    out_dir = Path(args.out_dir).expanduser().resolve()
    gen_dir = out_dir / "generated"
    ref_dir = out_dir / "originals"
    mel_dir = out_dir / "mels"
    compare_dir = out_dir / "mel_comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_meta = cache.get("audio_meta")
    if not isinstance(audio_meta, dict) or not all(key in audio_meta for key in AUDIO_META_KEYS):
        raise ValueError("Token cache does not contain complete audio_meta entries for original audio export.")

    entries = []
    for idx, (waveform, label) in enumerate(zip(generated, labels)):
        speaker = _speaker_name(class_names, int(label))
        safe_speaker = _safe_name(speaker)
        stem = f"generated_{idx:02d}_class{int(label):03d}_{safe_speaker}"
        gen_wave = waveform.clamp(-1.0, 1.0) if use_existing_generated else normalize_generated_waveform_for_preview(waveform, config)
        gen_wav = gen_dir / f"{stem}.wav"
        gen_mel = mel_dir / f"{stem}_mel.png"
        _write_wav(gen_wav, gen_wave, sample_rate=sample_rate)
        _save_mel(gen_mel, gen_wave, config, title=f"generated {idx:02d} speaker {speaker} class {int(label)}")

        original_entries = []
        panels = [(f"generated {idx:02d}", gen_wave)]
        for ref_idx, cache_idx in enumerate(_matching_original_indices(cache, int(label), int(args.orig_per_speaker))):
            item = _meta_item(audio_meta, cache_idx)
            ref_wave = _load_cropped_waveform(item, config).clamp(-1.0, 1.0)
            ref_stem = f"original_for_{idx:02d}_ref{ref_idx:02d}_class{int(label):03d}_{safe_speaker}_{_safe_name(Path(str(item['path'])).stem)}"
            ref_wav = ref_dir / f"{ref_stem}.wav"
            ref_mel = mel_dir / f"{ref_stem}_mel.png"
            _write_wav(ref_wav, ref_wave, sample_rate=sample_rate)
            _save_mel(ref_mel, ref_wave, config, title=f"original {ref_idx:02d} speaker {speaker} class {int(label)}")
            panels.append((f"original {ref_idx:02d}", ref_wave))
            original_entries.append(
                {
                    "cache_index": int(cache_idx),
                    "source_path": str(item["path"]),
                    "wav": str(ref_wav),
                    "mel": str(ref_mel),
                    **_wave_stats(ref_wave),
                }
            )

        compare_path = compare_dir / f"{stem}_vs_original_mels.png"
        _save_mel_comparison(
            compare_path,
            panels,
            config,
            title=f"speaker {speaker} class {int(label)} generated vs originals",
        )
        entries.append(
            {
                "index": int(idx),
                "class_id": int(label),
                "speaker_id": speaker,
                "generated_wav": str(gen_wav),
                "generated_mel": str(gen_mel),
                "comparison_mel": str(compare_path),
                "originals": original_entries,
                **{f"generated_{key}": value for key, value in _wave_stats(gen_wave).items()},
            }
        )

    manifest = {
        "checkpoint": str(Path(args.ckpt).expanduser().resolve()) if str(args.ckpt or "").strip() else "",
        "token_cache": str(Path(args.token_cache).expanduser().resolve()),
        "generated_dir": str(generated_dir.resolve()) if use_existing_generated else "",
        "sample_rate": sample_rate,
        "num_samples": int(args.num_samples),
        "orig_per_speaker": int(args.orig_per_speaker),
        "temperature": float(args.temperature),
        "top_k": int(args.top_k),
        "coeff_temperature": args.coeff_temperature,
        "coeff_mode": args.coeff_mode,
        "entries": entries,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "manifest.tsv").open("w", encoding="utf-8") as handle:
        handle.write("index\tclass_id\tspeaker_id\tgenerated_wav\tgenerated_mel\tcomparison_mel\toriginal_count\n")
        for entry in entries:
            handle.write(
                "\t".join(
                    [
                        str(entry["index"]),
                        str(entry["class_id"]),
                        str(entry["speaker_id"]),
                        str(entry["generated_wav"]),
                        str(entry["generated_mel"]),
                        str(entry["comparison_mel"]),
                        str(len(entry["originals"])),
                    ]
                )
                + "\n"
            )
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default="", help="Stage-2 checkpoint to sample from.")
    parser.add_argument("--token-cache", required=True, help="Speaker-conditioned token cache.")
    parser.add_argument("--generated-dir", default="", help="Existing generated WAV directory to label instead of sampling.")
    parser.add_argument("--out-dir", required=True, help="Directory for the exported audio report.")
    parser.add_argument("--out-root", default="outputs", help="Output root for resolving the stage-1 decoder.")
    parser.add_argument("--ar-dir", default="outputs/ar", help="Stage-2 output root used only for cache inference fallback.")
    parser.add_argument("--device", default="auto", help="Sampling device: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--orig-per-speaker", type=int, default=2)
    parser.add_argument("--class-labels", default="", help="Comma/space separated class IDs or speaker IDs such as p225,p226.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--coeff-temperature", type=float, default=None)
    parser.add_argument("--coeff-mode", default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    out_dir = export_report(parse_args())
    print(f"wrote speaker-conditioned audio report: {out_dir}")


if __name__ == "__main__":
    main()
