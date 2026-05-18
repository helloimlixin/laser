#!/usr/bin/env python3

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import shlex
import shutil
import subprocess
import sys

if sys.version_info < (3, 8):
    raise SystemExit("ERROR: scripts/submit_multimodal_sweep.py requires Python >= 3.8.")

import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


EXCLUDES = (
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tmp",
    ".tmp_*",
    "cluster_logs",
    "wandb",
    "runs",
    "source_snapshot_*",
    "pre_variation_snapshot_*",
    "*.out",
    "*.err",
    "*.pyc",
    "*.pyo",
    "*.swp",
)


@dataclass(frozen=True)
class SmokeCase:
    name: str
    data_config: str
    model_config: str
    data_dir: str
    image_size: int
    batch_size: int
    stage2_batch_size: int
    num_workers: int
    cache_max_items: int
    stage1_max_steps: int
    stage2_max_steps: int
    stage1_limit_train_batches: int
    stage2_limit_train_batches: int
    extra_cache_args: tuple[str, ...] = ()
    extra_stage1_overrides: tuple[str, ...] = ()
    extra_stage2_overrides: tuple[str, ...] = ()


def _user() -> str:
    return os.environ.get("USER", "unknown")


def _scratch_path(*parts: str) -> str:
    return str(Path("/scratch") / _user() / Path(*parts))


def _cache_path(*parts: str) -> str:
    return str(Path("/cache/home") / _user() / Path(*parts))


def _snapshot_ignore(repo: Path):
    repo = repo.resolve()

    def ignore(current_dir: str, names: Iterable[str]):
        rel_dir = Path(current_dir).resolve().relative_to(repo)
        ignored = set()
        for name in names:
            rel_path = (rel_dir / name) if rel_dir != Path(".") else Path(name)
            if len(rel_path.parts) >= 2 and rel_path.parts[0] == "configs" and rel_path.parts[1] == "wandb":
                continue
            if any(fnmatch.fnmatch(name, pattern) for pattern in EXCLUDES):
                ignored.add(name)
        return ignored

    return ignore


def _has_audio_files(root: Path) -> bool:
    for pattern in ("*.wav", "*.flac"):
        if any(path.is_file() for path in root.rglob(pattern)):
            return True
    return False


def ensure_vctk_smoke_data(root: Path, *, sample_rate: int = 16000, clip_samples: int = 32768, num_files: int = 24) -> tuple[Path, bool]:
    import numpy as np

    root = root.expanduser().resolve()
    marker = root / ".codex_synthetic_vctk_smoke"
    if marker.is_file():
        return root, True
    if _has_audio_files(root):
        synthetic_names = [path for path in root.rglob("p225_smoke_*.wav")]
        if synthetic_names:
            marker.write_text("synthetic_vctk_smoke\n", encoding="utf-8")
            return root, True
        return root, False

    wav_root = root / "wav48" / "p225"
    wav_root.mkdir(parents=True, exist_ok=True)

    total_seconds = clip_samples / float(sample_rate)
    t = np.linspace(0.0, total_seconds, clip_samples, endpoint=False, dtype=np.float32)
    for idx in range(num_files):
        base = 120.0 + 7.0 * idx
        chirp = np.sin(2.0 * np.pi * (base * t + 25.0 * t * t))
        harmonic = 0.5 * np.sin(2.0 * np.pi * (base * 2.03) * t + 0.17 * idx)
        slow = 0.3 * np.sin(2.0 * np.pi * (2.0 + 0.1 * idx) * t)
        envelope_base = np.maximum(np.sin(np.pi * np.linspace(0.0, 1.0, clip_samples, dtype=np.float32)), 0.0)
        envelope = np.clip(envelope_base**1.3, 0.0, 1.0)
        waveform = 0.42 * envelope * (chirp + harmonic + slow)
        waveform = np.clip(waveform, -0.95, 0.95)
        pcm = (waveform * 32767.0).astype(np.int16)
        with wave.open(str(wav_root / f"p225_smoke_{idx:03d}.wav"), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(pcm.tobytes())
    marker.write_text("synthetic_vctk_smoke\n", encoding="utf-8")
    return root, True


def resolve_full_training_vctk_dir(raw_path: str) -> Path:
    supplied = Path(raw_path).expanduser().resolve()
    candidates = [
        supplied,
        supplied.parent / "VCTK-Corpus-0.92",
        supplied.parent / "VCTK-Corpus",
        supplied.parent / "vctk",
        supplied.parent / "VCTK",
    ]
    seen = set()
    ordered = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(candidate)

    for candidate in ordered:
        if not candidate.exists():
            continue
        if (candidate / ".codex_synthetic_vctk_smoke").is_file():
            continue
        if _has_audio_files(candidate):
            return candidate

    raise SystemExit(
        "Full-training mode requires a real VCTK WAV/FLAC corpus. "
        f"Tried: {', '.join(str(path) for path in ordered)}"
    )


def resolve_full_training_coco_dir(raw_path: str) -> Path:
    supplied = Path(raw_path).expanduser().resolve()
    candidates = [
        supplied,
        Path(_scratch_path("data", "coco")),
        Path(_scratch_path("datasets", "coco")),
        Path(_scratch_path("datasets", "COCO")),
    ]
    seen = set()
    ordered = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(candidate)

    for candidate in ordered:
        if (candidate / "train2017").is_dir() and (candidate / "val2017").is_dir():
            return candidate

    raise SystemExit(
        "Full-training mode requires a COCO directory with train2017/ and val2017/. "
        f"Tried: {', '.join(str(path) for path in ordered)}"
    )


def snapshot_repo(repo: Path, snapshot_root: Path, *, stem: str) -> Path:
    snapshot_root.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_root / stem
    if snapshot_path.exists():
        shutil.rmtree(snapshot_path)
    shutil.copytree(repo, snapshot_path, ignore=_snapshot_ignore(repo))
    return snapshot_path


def _dataset_cases(
    vctk_dir: Path,
    *,
    coco_dir: Optional[Path] = None,
    ffhq_dir: Optional[Path] = None,
    maestro_dir: Optional[Path] = None,
    model_family: str = "vqvae",
) -> list[SmokeCase]:
    model_family = str(model_family).strip().lower()
    if model_family == "vqvae":
        image_model_config = "vqvae"
        audio_model_config = "vqvae_audio"
        audio_waveform_model_config = "vqvae_audio_waveform"
    elif model_family == "laser":
        image_model_config = "laser"
        audio_model_config = "laser_audio"
        audio_waveform_model_config = "laser_audio_waveform"
    else:
        raise ValueError(f"Unsupported model family: {model_family!r}")

    return [
        SmokeCase(
            name="celeba",
            data_config="celeba",
            model_config=image_model_config,
            data_dir=_scratch_path("datasets", "celeba_packed_128"),
            image_size=128,
            batch_size=1,
            stage2_batch_size=2,
            num_workers=0,
            cache_max_items=64,
            stage1_max_steps=8,
            stage2_max_steps=16,
            stage1_limit_train_batches=8,
            stage2_limit_train_batches=8,
            extra_stage1_overrides=(
                "model.compute_fid=false",
                "model.num_embeddings=256",
                "model.embedding_dim=32",
                "model.num_hiddens=96",
                "model.num_residual_blocks=1",
                "model.num_residual_hiddens=32",
            ),
            extra_stage2_overrides=("ar.d_model=192", "ar.n_heads=4", "ar.n_layers=4", "ar.d_ff=384"),
        ),
        SmokeCase(
            name="celebahq",
            data_config="celebahq",
            model_config=image_model_config,
            data_dir=_scratch_path("datasets", "celebahq_packed_256"),
            image_size=256,
            batch_size=2,
            stage2_batch_size=2,
            num_workers=0,
            cache_max_items=32,
            stage1_max_steps=6,
            stage2_max_steps=12,
            stage1_limit_train_batches=6,
            stage2_limit_train_batches=6,
            extra_stage1_overrides=(
                "model.compute_fid=false",
                "model.num_embeddings=256",
                "model.embedding_dim=32",
                "model.num_hiddens=96",
                "model.num_residual_blocks=1",
                "model.num_residual_hiddens=32",
            ),
            extra_stage2_overrides=("ar.d_model=192", "ar.n_heads=4", "ar.n_layers=4", "ar.d_ff=384"),
        ),
        SmokeCase(
            name="stl10",
            data_config="stl10",
            model_config=image_model_config,
            data_dir=_scratch_path("datasets", "stl10"),
            image_size=96,
            batch_size=4,
            stage2_batch_size=4,
            num_workers=4,
            cache_max_items=64,
            stage1_max_steps=8,
            stage2_max_steps=16,
            stage1_limit_train_batches=8,
            stage2_limit_train_batches=8,
            extra_stage1_overrides=(
                "model.compute_fid=false",
                "model.num_embeddings=256",
                "model.embedding_dim=32",
                "model.num_hiddens=96",
                "model.num_residual_blocks=1",
                "model.num_residual_hiddens=32",
            ),
            extra_stage2_overrides=("ar.d_model=192", "ar.n_heads=4", "ar.n_layers=4", "ar.d_ff=384"),
        ),
        SmokeCase(
            name="coco",
            data_config="coco",
            model_config=image_model_config,
            data_dir=str(coco_dir or Path(_scratch_path("data", "coco"))),
            image_size=512,
            batch_size=1,
            stage2_batch_size=1,
            num_workers=2,
            cache_max_items=12,
            stage1_max_steps=4,
            stage2_max_steps=8,
            stage1_limit_train_batches=4,
            stage2_limit_train_batches=4,
            extra_stage1_overrides=(
                "model.compute_fid=false",
                "model.num_embeddings=512",
                "model.embedding_dim=64",
                "model.num_hiddens=96",
                "model.num_residual_blocks=1",
                "model.num_residual_hiddens=32",
            ),
            extra_stage2_overrides=("ar.d_model=192", "ar.n_heads=4", "ar.n_layers=4", "ar.d_ff=384"),
        ),
        SmokeCase(
            name="ffhq",
            data_config="ffhq",
            model_config=image_model_config,
            data_dir=str(ffhq_dir or Path(_scratch_path("datasets", "ffhq"))),
            image_size=256,
            batch_size=2,
            stage2_batch_size=2,
            num_workers=4,
            cache_max_items=32,
            stage1_max_steps=6,
            stage2_max_steps=12,
            stage1_limit_train_batches=6,
            stage2_limit_train_batches=6,
            extra_stage1_overrides=(
                "model.compute_fid=false",
                "model.num_embeddings=256",
                "model.embedding_dim=32",
                "model.num_hiddens=96",
                "model.num_residual_blocks=1",
                "model.num_residual_hiddens=32",
            ),
            extra_stage2_overrides=("ar.d_model=192", "ar.n_heads=4", "ar.n_layers=4", "ar.d_ff=384"),
        ),
        SmokeCase(
            name="vctk",
            data_config="vctk",
            model_config=audio_model_config,
            data_dir=str(vctk_dir),
            image_size=128,
            batch_size=2,
            stage2_batch_size=2,
            num_workers=2,
            cache_max_items=24,
            stage1_max_steps=6,
            stage2_max_steps=12,
            stage1_limit_train_batches=6,
            stage2_limit_train_batches=6,
            extra_cache_args=(
                "--sample-rate", "16000",
                "--audio-num-samples", "32768",
                "--stft-n-fft", "1024",
                "--stft-hop-length", "256",
                "--stft-win-length", "1024",
                "--stft-power", "2.0",
                "--stft-log-offset", "1.0e-5",
            ),
            extra_stage1_overrides=(
                "model.compute_fid=false",
                "model.num_embeddings=256",
                "model.embedding_dim=32",
                "model.num_hiddens=96",
                "model.num_residual_blocks=1",
                "model.num_residual_hiddens=32",
            ),
            extra_stage2_overrides=("ar.d_model=160", "ar.n_heads=4", "ar.n_layers=4", "ar.d_ff=320"),
        ),
        SmokeCase(
            name="maestro",
            data_config="maestro_waveform",
            model_config=audio_waveform_model_config,
            data_dir=str(maestro_dir or Path(_scratch_path("datasets", "maestro", "maestro-v3.0.0"))),
            image_size=128,
            batch_size=2,
            stage2_batch_size=2,
            num_workers=2,
            cache_max_items=24,
            stage1_max_steps=6,
            stage2_max_steps=12,
            stage1_limit_train_batches=6,
            stage2_limit_train_batches=6,
            extra_cache_args=(
                "--sample-rate", "22050",
                "--audio-num-samples", "65536",
                "--audio-dc-remove",
                "--audio-peak-normalize",
                "--audio-target-peak", "0.95",
                "--audio-rms-normalize",
                "--audio-target-rms", "0.12",
                "--audio-max-gain", "8.0",
                "--audio-min-crop-rms", "0.03",
                "--audio-crop-attempts", "64",
                "--audio-fade-samples", "1024",
            ),
            extra_stage1_overrides=(
                "model.compute_fid=false",
                "model.num_embeddings=256",
                "model.embedding_dim=32",
                "model.num_hiddens=96",
                "model.num_residual_blocks=1",
                "model.num_residual_hiddens=32",
            ),
            extra_stage2_overrides=("ar.d_model=160", "ar.n_heads=4", "ar.n_layers=4", "ar.d_ff=320"),
        ),
    ]


def _full_stage1_batch_size(case: SmokeCase) -> int:
    return {
        "celebahq": 8,
        "coco": 2,
        "ffhq": 8,
        "maestro": 8,
        "vctk": 8,
    }.get(case.name, case.batch_size)


def _full_stage2_batch_size(case: SmokeCase) -> int:
    return {
        "celebahq": 8,
        "coco": 4,
        "ffhq": 8,
        "maestro": 8,
        "vctk": 8,
    }.get(case.name, case.stage2_batch_size)


def _bash_array(items: Iterable[str]) -> str:
    lines = []
    for item in items:
        if item.startswith("$"):
            lines.append(f'  "{item}"')
        else:
            lines.append(f"  {shlex.quote(item)}")
    return "\n".join(lines)


def _safe_label(raw: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_-]+", "-", str(raw).strip())
    label = label.strip("-_")
    if not label:
        raise ValueError(f"run label produced an empty token: {raw!r}")
    return label


def _per_process_batch_size(global_batch_size: int, num_devices: int) -> int:
    devices = max(1, int(num_devices))
    batch_size = max(1, int(global_batch_size))
    return max(1, (batch_size + devices - 1) // devices)


def write_job_files(
    *,
    snapshot_path: Path,
    run_root: Path,
    group_name: str,
    project: str,
    case: SmokeCase,
    synthetic_vctk: bool,
    full_training: bool,
    model_family: str,
    stage1_epochs: int,
    stage2_epochs: int,
    num_gpus: int,
    stage1_only: bool = False,
    extra_cache_args: tuple[str, ...] = (),
    extra_stage1_overrides: tuple[str, ...] = (),
    extra_stage2_overrides: tuple[str, ...] = (),
) -> tuple[Path, Path, Path]:
    run_dir = run_root / case.name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_base = run_dir / case.name
    stage1_dir = run_dir / "stage1"
    stage2_dir = run_dir / "stage2"
    token_cache = run_dir / "token_cache.pt"
    run_script = run_dir / "run.sh"
    sbatch_script = run_dir / "sbatch.sh"
    stage1_batch_size = _full_stage1_batch_size(case) if full_training else case.batch_size
    stage2_batch_size = _full_stage2_batch_size(case) if full_training else case.stage2_batch_size
    cache_max_items = 0 if full_training else case.cache_max_items
    stage1_max_steps = -1 if full_training else case.stage1_max_steps
    stage2_max_steps = -1 if full_training else case.stage2_max_steps
    stage1_limit_train = "1.0" if full_training else str(case.stage1_limit_train_batches)
    stage2_limit_train = "1.0" if full_training else str(case.stage2_limit_train_batches)
    limit_eval_batches = "1.0" if full_training else "2"
    log_every_n_steps = "50" if full_training else "1"
    run_tag = "train" if full_training else "smoke"
    is_audio = case.name in {"vctk", "maestro"}
    generation_metric_samples = 32 if is_audio else 64
    model_family = str(model_family).strip().lower()
    trainer_devices = max(1, int(num_gpus))
    trainer_strategy = "ddp" if trainer_devices > 1 else "auto"
    stage1_data_batch_size = _per_process_batch_size(stage1_batch_size, trainer_devices)
    stage2_data_batch_size = _per_process_batch_size(stage2_batch_size, trainer_devices)

    stage1_overrides = [
        f"output_dir={stage1_dir}",
        f"model={case.model_config}",
        f"data={case.data_config}",
        f"data.data_dir={case.data_dir}",
        f"data.image_size={case.image_size}",
        f"data.batch_size={stage1_data_batch_size}",
        f"data.num_workers={case.num_workers}",
        "seed=42",
        f"train.max_epochs={stage1_epochs}",
        f"train.max_steps={stage1_max_steps}",
        f"train.limit_train_batches={stage1_limit_train}",
        f"train.limit_val_batches={limit_eval_batches}",
        f"train.limit_test_batches={limit_eval_batches}",
        f"train.log_every_n_steps={log_every_n_steps}",
        # Keep sweep jobs moving into token extraction. Full reconstruction
        # evaluation is already covered by validation; the dedicated post-fit
        # test pass is too expensive for these chained SLURM jobs and can leave
        # the allocation idle before cache extraction starts.
        "train.run_test_after_fit=false",
        f"train.devices={trainer_devices}",
        f"train.strategy={trainer_strategy}",
        "train.precision=bf16-mixed",
        "train.accelerator=gpu",
        f"wandb.project={project}",
        f"wandb.group={group_name}",
        f"wandb.name={case.name}-stage1-autoencoder",
        f"wandb.tags=[{run_tag},{model_family},{case.name},stage1,autoencoder{',synthetic_vctk' if synthetic_vctk and case.name == 'vctk' else ''}]",
        "wandb.append_timestamp=false",
        f"wandb.save_dir={stage1_dir / 'wandb'}",
    ]
    if is_audio:
        stage1_overrides.extend(("model.compute_fid=false", "model.perceptual_weight=0.0"))
    elif full_training:
        stage1_overrides.append("model.compute_fid=true")
        if model_family == "laser":
            # Keep full image LASER submissions comparable to the VQ-VAE baseline:
            # same encoder/decoder family and capacity, different bottleneck.
            # Explicit user overrides are appended below and still win.
            stage1_overrides.extend(
                (
                    "model.backbone=simple",
                    "model.num_downsamples=2",
                    "model.num_hiddens=128",
                    "model.num_residual_blocks=2",
                    "model.num_residual_hiddens=64",
                    "model.embedding_dim=64",
                    "model.num_embeddings=512",
                    "model.sparsity_level=8",
                    "model.commitment_cost=0.25",
                    "model.bottleneck_loss_weight=1.0",
                    "model.dict_learning_rate=2.5e-4",
                    "model.recon_mse_weight=1.0",
                    "model.recon_l1_weight=0.0",
                    "model.recon_edge_weight=0.0",
                    "model.out_tanh=false",
                    "train.learning_rate=1.0e-3",
                )
            )
    if not full_training:
        stage1_overrides.extend(case.extra_stage1_overrides)
    stage1_overrides.extend(extra_stage1_overrides)

    stage2_overrides = [
        f"token_cache_path={token_cache}",
        f"output_dir={stage2_dir}",
        "seed=42",
        "ar.type=sparse_spatial_depth",
        f"ar.max_steps={stage2_max_steps}",
        f"train_ar.max_epochs={stage2_epochs}",
        f"train_ar.batch_size={stage2_data_batch_size}",
        f"train_ar.max_items={cache_max_items}",
        f"train_ar.limit_train_batches={stage2_limit_train}",
        f"train_ar.limit_val_batches={limit_eval_batches}",
        f"train_ar.limit_test_batches={limit_eval_batches}",
        f"train_ar.log_every_n_steps={log_every_n_steps}",
        "train_ar.sample_every_n_epochs=1",
        "train_ar.sample_log_to_wandb=true",
        f"train_ar.sample_num_images={generation_metric_samples}",
        f"train_ar.generation_metric_num_samples={generation_metric_samples}",
        f"train_ar.compute_generation_fid={'false' if is_audio else 'true'}",
        f"train_ar.compute_audio_generation_metrics={'true' if is_audio else 'false'}",
        f"train_ar.devices={trainer_devices}",
        f"train_ar.strategy={trainer_strategy}",
        "train_ar.precision=bf16-mixed",
        "train_ar.accelerator=gpu",
        f"data.dataset={case.name}",
        f"data.data_dir={case.data_dir}",
        f"data.image_size={case.image_size}",
        f"data.num_workers={case.num_workers}",
        f"wandb.project={project}",
        f"wandb.group={group_name}",
        f"wandb.name={case.name}-stage2-transformer",
        f"wandb.tags=[{run_tag},{model_family},{case.name},stage2,transformer,generation{',synthetic_vctk' if synthetic_vctk and case.name == 'vctk' else ''}]",
        "wandb.append_timestamp=false",
        f"wandb.save_dir={stage2_dir / 'wandb'}",
    ]
    if not full_training:
        stage2_overrides.extend(case.extra_stage2_overrides)
    stage2_overrides.extend(extra_stage2_overrides)

    cache_args = [
        "--stage1-checkpoint", "$CKPT",
        "--output-path", str(token_cache),
        "--dataset", case.name,
        "--data-dir", case.data_dir,
        "--image-size", str(case.image_size),
        "--batch-size", str(stage1_batch_size),
        "--num-workers", str(case.num_workers),
        "--seed", "42",
        "--max-items", str(cache_max_items),
        "--model-type", model_family,
    ]
    cache_args.extend(case.extra_cache_args)
    cache_args.extend(extra_cache_args)

    run_script.write_text(
        f"""#!/bin/bash
set -euo pipefail

export PYTHONUSERBASE="{_scratch_path('.pydeps', 'laser_src_py311')}"
export PATH="$PYTHONUSERBASE/bin:$PATH"
export PYTHONPATH="{snapshot_path}${{PYTHONPATH:+:$PYTHONPATH}}"
export WANDB_MODE="${{WANDB_MODE:-online}}"
export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
export HYDRA_FULL_ERROR=1
export TMPDIR="/tmp/laser_${{SLURM_JOB_ID:-$$}}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

mkdir -p "$TMPDIR" "{stage1_dir / 'wandb'}" "{stage2_dir / 'wandb'}"

PYTHON_BIN="${{PYTHON_BIN:-$(command -v python3 || command -v python || true)}}"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "python3/python not found" >&2
  exit 127
fi

python_version() {{
  "$1" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null || printf 'unknown\n'
}}

if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
  echo "ERROR: $PYTHON_BIN is Python $(python_version "$PYTHON_BIN"); LASER requires Python >= 3.10." >&2
  echo "Set PYTHON_BIN to a supported environment before running this job." >&2
  exit 2
fi

"$PYTHON_BIN" -m pip install --user --quiet \
  numpy scipy wandb lightning omegaconf hydra-core rich 'torchmetrics[image]' \
  torch-fidelity matplotlib lpips soundfile 2>/dev/null || true

if ! "$PYTHON_BIN" - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("visqol") is not None else 1)
PY
then
  "$PYTHON_BIN" -m pip install --user --quiet bazelisk pybind11 protobuf 2>/dev/null || true
  if ! command -v bazel >/dev/null 2>&1 && command -v bazelisk >/dev/null 2>&1; then
    ln -sf "$(command -v bazelisk)" "$PYTHONUSERBASE/bin/bazel" 2>/dev/null || true
  fi
  "$PYTHON_BIN" -m pip install --user --quiet "git+https://github.com/google/visqol.git" 2>/dev/null || true
fi

"$PYTHON_BIN" - <<'PY'
import importlib.util
print(f"ViSQOL available: {{importlib.util.find_spec('visqol') is not None}}")
PY

"$PYTHON_BIN" - <<'PY'
import torch

cuda_available = torch.cuda.is_available()
print(f"CUDA available: {{cuda_available}}")
if not cuda_available:
    raise SystemExit("CUDA is not available inside this job; failing before training.")
print(f"CUDA device: {{torch.cuda.get_device_name(0)}}")
PY

cd "{snapshot_path}"

STAGE1_ARGS=(
{_bash_array(stage1_overrides)}
)

echo "=== Stage 1: autoencoder training ({case.name}) ==="
"$PYTHON_BIN" train_stage1_autoencoder.py "${{STAGE1_ARGS[@]}}"

if [[ "{'1' if stage1_only else '0'}" == "1" ]]; then
  echo "=== Stage 1 only requested; skipping token cache and stage 2 ({case.name}) ==="
  exit 0
fi

CKPT="$(find "{stage1_dir}" -path '*/final.ckpt' -type f | sort | tail -1)"
if [[ -z "$CKPT" ]]; then
  CKPT="$(find "{stage1_dir}" -path '*/last.ckpt' -type f | sort | tail -1)"
fi
if [[ -z "$CKPT" ]]; then
  echo "No stage-1 checkpoint found under {stage1_dir}" >&2
  exit 1
fi

CACHE_ARGS=(
{_bash_array(cache_args)}
)

echo "=== Token cache extraction ({case.name}) ==="
"$PYTHON_BIN" cache.py "${{CACHE_ARGS[@]}}"

STAGE2_ARGS=(
{_bash_array(stage2_overrides)}
)

echo "=== Stage 2: transformer prior training + generation ({case.name}) ==="
"$PYTHON_BIN" train_stage2_prior.py "${{STAGE2_ARGS[@]}}"
""",
        encoding="utf-8",
    )
    os.chmod(run_script, 0o755)

    sbatch_script.write_text(
        f"""#!/bin/bash
set -euo pipefail

if ! command -v module >/dev/null 2>&1; then
  if [[ -f /usr/share/lmod/lmod/init/bash ]]; then
    set +u; source /usr/share/lmod/lmod/init/bash; set -u
  elif [[ -f /usr/share/Modules/init/bash ]]; then
    set +u; source /usr/share/Modules/init/bash; set -u
  fi
fi

CONTAINER_BIN=""
for candidate in singularity apptainer; do
  if command -v "$candidate" >/dev/null 2>&1; then
    CONTAINER_BIN="$candidate"
    break
  fi
done

if [[ -z "$CONTAINER_BIN" ]]; then
  module load singularity 2>/dev/null || true
  module load singularityce 2>/dev/null || true
  module load singularity-ce 2>/dev/null || true
  module load apptainer 2>/dev/null || true
  for candidate in singularity apptainer; do
    if command -v "$candidate" >/dev/null 2>&1; then
      CONTAINER_BIN="$candidate"
      break
    fi
  done
fi

IMAGE="${{IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}}"

if [[ -n "$CONTAINER_BIN" ]]; then
  "$CONTAINER_BIN" exec --nv \
    --bind "{snapshot_path}" \
    --bind "/scratch/{_user()}" \
    --bind "{run_dir}" \
    --bind /dev/shm \
    "$IMAGE" \
    bash "{run_script}"
else
  bash "{run_script}"
fi
""",
        encoding="utf-8",
    )
    os.chmod(sbatch_script, 0o755)
    return run_dir, run_script, sbatch_script


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Snapshot the maintained repo and submit a multimodal two-stage sweep.")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]), help="Live repo to snapshot.")
    parser.add_argument("--snapshot-root", default=_scratch_path("submission_snapshots"), help="Parent directory for frozen snapshots.")
    parser.add_argument("--run-root-base", default=None, help="Parent directory for job logs and outputs.")
    parser.add_argument("--partition", default="gpu-redhat")
    parser.add_argument("--time-limit", default=None)
    parser.add_argument("--cpus-per-task", type=int, default=8)
    parser.add_argument("--mem-mb", type=int, default=64000)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--project", default=None)
    parser.add_argument("--model-family", default="laser", choices=("vqvae", "laser"))
    parser.add_argument("--run-label", default="", help="Optional label inserted into snapshot, run root, and W&B group names.")
    parser.add_argument("--vctk-dir", default=_scratch_path("datasets", "VCTK-Corpus-smoke"))
    parser.add_argument("--coco-dir", default=_scratch_path("data", "coco"))
    parser.add_argument("--ffhq-dir", default=_scratch_path("datasets", "ffhq"))
    parser.add_argument("--maestro-dir", default=_scratch_path("datasets", "maestro", "maestro-v3.0.0"))
    parser.add_argument("--cases", default="", help="Comma-separated subset of cases to submit, e.g. celeba,celebahq,stl10,vctk.")
    parser.add_argument("--full-training", action="store_true", help="Use full configs/data limits instead of smoke settings.")
    parser.add_argument("--stage1-only", action="store_true", help="Train stage 1, then skip token cache extraction and stage 2.")
    parser.add_argument("--stage1-epochs", type=int, default=None, help="Stage-1 autoencoder epochs.")
    parser.add_argument("--stage2-epochs", type=int, default=None, help="Stage-2 transformer epochs.")
    parser.add_argument(
        "--stage1-override",
        action="append",
        default=[],
        help="Extra Hydra override appended to every stage-1 run. Repeat as needed.",
    )
    parser.add_argument(
        "--stage2-override",
        action="append",
        default=[],
        help="Extra Hydra override appended to every stage-2 run. Repeat as needed.",
    )
    parser.add_argument(
        "--cache-arg",
        action="append",
        default=[],
        help="Extra argument appended to cache.py. Repeat for flags and values, e.g. --cache-arg --coeff-bins --cache-arg 256.",
    )
    parser.add_argument("--exclude-nodes", default="", help="Optional SLURM node exclusion list passed to sbatch --exclude.")
    parser.add_argument("--dependency", default="", help="Optional SLURM dependency expression, e.g. afterok:12345.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    model_family = str(args.model_family).strip().lower()
    full_training = bool(args.full_training)
    stage1_only = bool(args.stage1_only)
    stage1_epochs = int(args.stage1_epochs if args.stage1_epochs is not None else (10 if full_training else 1))
    stage2_epochs = int(args.stage2_epochs if args.stage2_epochs is not None else (10 if full_training else 1))
    if stage1_epochs <= 0 or stage2_epochs <= 0:
        raise SystemExit("Stage epoch counts must be positive.")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_kind = "train" if full_training else "smoke"
    time_limit = str(args.time_limit or ("2-00:00:00" if full_training else "04:00:00"))
    label = _safe_label(args.run_label) if str(args.run_label or "").strip() else ""
    label_part = f"_{label}" if label else ""
    group_label_part = f"-{label}" if label else ""
    snapshot_name = f"laser_{model_family}_{run_kind}{label_part}_{stamp}"
    snapshot_path = snapshot_repo(repo, Path(args.snapshot_root).expanduser().resolve(), stem=snapshot_name)

    wanted = {name.strip().lower() for name in str(args.cases or "").split(",") if name.strip()}
    needs_vctk = (not wanted) or ("vctk" in wanted)
    needs_coco = (not wanted) or ("coco" in wanted)
    if full_training and needs_vctk:
        vctk_dir = resolve_full_training_vctk_dir(str(args.vctk_dir))
        synthetic_vctk = False
    elif full_training:
        vctk_dir = Path(args.vctk_dir).expanduser().resolve()
        synthetic_vctk = False
    else:
        if needs_vctk:
            vctk_dir, synthetic_vctk = ensure_vctk_smoke_data(Path(args.vctk_dir))
        else:
            vctk_dir = Path(args.vctk_dir).expanduser().resolve()
            synthetic_vctk = False
    if full_training and needs_coco:
        coco_dir = resolve_full_training_coco_dir(str(args.coco_dir))
    else:
        coco_dir = Path(args.coco_dir).expanduser().resolve()
    cases = _dataset_cases(
        vctk_dir,
        coco_dir=coco_dir,
        ffhq_dir=Path(args.ffhq_dir).expanduser().resolve(),
        maestro_dir=Path(args.maestro_dir).expanduser().resolve(),
        model_family=model_family,
    )
    if wanted:
        known = {case.name for case in cases}
        unknown = sorted(wanted - known)
        if unknown:
            raise SystemExit(f"Unknown case(s): {', '.join(unknown)}")
        cases = [case for case in cases if case.name in wanted]
        if not cases:
            raise SystemExit("No cases selected for submission.")

    group_name = f"{model_family}-{run_kind}{group_label_part}-{stamp}"
    run_root_base = args.run_root_base or _scratch_path("runs", f"{model_family}_{run_kind}")
    run_root = Path(run_root_base).expanduser().resolve() / group_name
    run_root.mkdir(parents=True, exist_ok=True)
    project = args.project or ("laser-multimodal-training" if full_training else "laser-multimodal-smoke")
    job_prefix = "ls" if model_family == "laser" else "vq"

    submissions = []
    for case in cases:
        run_dir, _, sbatch_script = write_job_files(
            snapshot_path=snapshot_path,
            run_root=run_root,
            group_name=group_name,
            project=project,
            case=case,
            synthetic_vctk=synthetic_vctk,
            full_training=full_training,
            model_family=model_family,
            stage1_epochs=stage1_epochs,
            stage2_epochs=stage2_epochs,
            num_gpus=int(args.gpus),
            stage1_only=stage1_only,
            extra_cache_args=tuple(str(v) for v in args.cache_arg),
            extra_stage1_overrides=tuple(str(v) for v in args.stage1_override),
            extra_stage2_overrides=tuple(str(v) for v in args.stage2_override),
        )
        log_base = run_dir / case.name
        sbatch_cmd = [
            "sbatch",
            f"--partition={args.partition}",
            f"--job-name={job_prefix}{run_kind[:3]}-{case.name}",
            "--nodes=1",
            "--ntasks=1",
            f"--cpus-per-task={int(args.cpus_per_task)}",
            f"--gres=gpu:{int(args.gpus)}",
            f"--mem={int(args.mem_mb)}",
            f"--time={time_limit}",
            f"--chdir={snapshot_path}",
            f"--output={log_base}_%j.out",
            f"--error={log_base}_%j.err",
        ]
        if str(args.exclude_nodes or "").strip():
            sbatch_cmd.append(f"--exclude={str(args.exclude_nodes).strip()}")
        if str(args.dependency or "").strip():
            sbatch_cmd.append(f"--dependency={str(args.dependency).strip()}")
        sbatch_cmd.append(str(sbatch_script))
        if args.dry_run:
            job_id = "dry-run"
            print(" ".join(sbatch_cmd))
        else:
            proc = subprocess.run(sbatch_cmd, check=True, text=True, capture_output=True)
            text = (proc.stdout or proc.stderr).strip()
            job_id = text.split()[-1]
        submissions.append(
            {
                "case": case.name,
                "job_id": job_id,
                "run_dir": str(run_dir),
                "stdout": f"{log_base}_{job_id}.out" if job_id != "dry-run" else f"{log_base}_<jobid>.out",
                "stderr": f"{log_base}_{job_id}.err" if job_id != "dry-run" else f"{log_base}_<jobid>.err",
            }
        )

    print(f"Snapshot: {snapshot_path}")
    print(f"Run root:  {run_root}")
    print(f"Model:     {model_family}")
    if label:
        print(f"Label:     {label}")
    print(f"Mode:      {('full training' if full_training else 'smoke')}{' stage1-only' if stage1_only else ''}")
    print(f"Epochs:    stage1={stage1_epochs}" + ("" if stage1_only else f" stage2={stage2_epochs}"))
    print(f"GPUs/job:  {int(args.gpus)}")
    if synthetic_vctk:
        print(f"VCTK path: {vctk_dir} (synthetic smoke corpus generated because no real WAV/FLAC corpus was present)")
    else:
        print(f"VCTK path: {vctk_dir}")
    for item in submissions:
        print(
            f"[{item['case']}] job={item['job_id']} stdout={item['stdout']} stderr={item['stderr']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
