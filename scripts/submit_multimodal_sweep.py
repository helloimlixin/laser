#!/usr/bin/env python3

from __future__ import annotations

import argparse
import fnmatch
import hashlib
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
    ".cache",
    ".pydeps",
    ".tmp",
    ".tmp_*",
    "cluster_logs",
    "wandb",
    "runs",
    "submission_snapshots",
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
    celeba_dir: Optional[Path] = None,
    celebahq_dir: Optional[Path] = None,
    coco_dir: Optional[Path] = None,
    ffhq_dir: Optional[Path] = None,
    imagenet_dir: Optional[Path] = None,
    imagenette2_dir: Optional[Path] = None,
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
            data_dir=str(celeba_dir or Path(_scratch_path("datasets", "celeba_packed_128"))),
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
            data_dir=str(celebahq_dir or Path(_scratch_path("datasets", "celebahq_packed_256"))),
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
            name="imagenette2",
            data_config="imagenette2",
            model_config=image_model_config,
            data_dir=str(imagenette2_dir or Path(_scratch_path("datasets", "imagenette2"))),
            image_size=224,
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
            name="imagenet",
            data_config="imagenet",
            model_config=image_model_config,
            data_dir=str(imagenet_dir or Path(_scratch_path("Projects", "data", "imagenet"))),
            image_size=256,
            batch_size=4,
            stage2_batch_size=4,
            num_workers=8,
            cache_max_items=128,
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
        "imagenet": 8,
        "maestro": 8,
        "vctk": 8,
    }.get(case.name, case.batch_size)


def _full_stage2_batch_size(case: SmokeCase) -> int:
    return {
        "celebahq": 8,
        "coco": 4,
        "ffhq": 8,
        "imagenet": 8,
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


def _hydra_override_key(item: str) -> Optional[str]:
    """Return the logical key for a Hydra override, if it sets one.

    Later overrides win in Hydra. Collapsing earlier duplicates keeps generated
    run scripts readable while preserving the resolved config.
    """
    raw = str(item).strip()
    if "=" not in raw:
        return None
    key = raw.split("=", 1)[0].strip()
    while key.startswith("+"):
        key = key[1:]
    if key.startswith("~"):
        key = key[1:]
    return key or None


def _dedupe_hydra_overrides(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped_reversed: list[str] = []
    for item in reversed([str(value) for value in items]):
        key = _hydra_override_key(item)
        if key is not None:
            if key in seen:
                continue
            seen.add(key)
        deduped_reversed.append(item)
    return list(reversed(deduped_reversed))


def _safe_label(raw: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_-]+", "-", str(raw).strip())
    label = label.strip("-_")
    if not label:
        raise ValueError(f"run label produced an empty token: {raw!r}")
    return label


def _bounded_label(raw: str, *, max_length: int = 120) -> str:
    label = _safe_label(raw)
    if len(label) <= max_length:
        return label
    digest = hashlib.sha1(label.encode("utf-8")).hexdigest()[:10]
    keep = max(1, max_length - len(digest) - 1)
    return f"{label[:keep].rstrip('-_')}-{digest}"


def _per_process_batch_size(global_batch_size: int, num_processes: int) -> int:
    devices = max(1, int(num_processes))
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
    stage1_adv_epochs: int,
    stage2_epochs: int,
    stage2_kind: str,
    num_gpus: int,
    stage1_checkpoint: Optional[Path] = None,
    stage1_only: bool = False,
    extra_cache_args: tuple[str, ...] = (),
    extra_stage1_overrides: tuple[str, ...] = (),
    extra_stage1_adv_overrides: tuple[str, ...] = (),
    extra_stage2_overrides: tuple[str, ...] = (),
    extra_diffusion_args: tuple[str, ...] = (),
    num_nodes: int = 1,
) -> tuple[Path, Path, Path]:
    run_dir = run_root / case.name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_base = run_dir / case.name
    stage1_dir = run_dir / "stage1"
    stage1_adv_dir = run_dir / "stage1_adv"
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
    trainer_nodes = max(1, int(num_nodes))
    trainer_world_size = trainer_devices * trainer_nodes
    trainer_strategy = "ddp" if trainer_world_size > 1 else "auto"
    stage1_data_batch_size = _per_process_batch_size(stage1_batch_size, trainer_world_size)
    stage2_data_batch_size = _per_process_batch_size(stage2_batch_size, trainer_world_size)

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
        f"train.num_nodes={trainer_nodes}",
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
    stage1_overrides = _dedupe_hydra_overrides(stage1_overrides)

    stage1_adv_overrides: list[str] = []
    if int(stage1_adv_epochs) > 0:
        stage1_adv_overrides = list(stage1_overrides)
        stage1_adv_overrides.extend(
            (
                f"output_dir={stage1_adv_dir}",
                f"train.max_epochs={int(stage1_adv_epochs)}",
                f"wandb.name={case.name}-stage1-adversarial",
                f"wandb.tags=[{run_tag},{model_family},{case.name},stage1,adversarial{',synthetic_vctk' if synthetic_vctk and case.name == 'vctk' else ''}]",
                f"wandb.save_dir={stage1_adv_dir / 'wandb'}",
            )
        )
        stage1_adv_overrides.extend(extra_stage1_adv_overrides)
        stage1_adv_overrides = _dedupe_hydra_overrides(stage1_adv_overrides)

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
        f"train_ar.num_nodes={trainer_nodes}",
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
    stage2_overrides = _dedupe_hydra_overrides(stage2_overrides)

    diffusion_args = [
        "--token-cache-path", str(token_cache),
        "--output-dir", str(stage2_dir),
        "--output-root", str(run_dir),
        "--seed", "42",
        "--batch-size", str(stage2_data_batch_size),
        "--num-workers", str(case.num_workers),
        "--max-epochs", str(stage2_epochs),
        "--max-steps", str(stage2_max_steps),
        "--accelerator", "gpu",
        "--devices", str(trainer_devices),
        "--precision", "bf16-mixed",
        "--gradient-clip-val", "1.0",
        "--log-every-n-steps", log_every_n_steps,
        "--val-check-interval", "1.0",
        "--limit-train-batches", stage2_limit_train,
        "--limit-val-batches", limit_eval_batches,
        "--max-items", str(cache_max_items),
        # W&B logging so the diffusion run's loss + post-fit FID land in the same
        # project/group as the AR-prior runs.
        "--wandb-project", project,
        "--wandb-group", group_name,
        "--wandb-name", f"{case.name}-stage2-diffusion",
        "--wandb-save-dir", str(stage2_dir / "wandb"),
    ]
    # Post-fit generation FID is image-only (src/stage2_metrics supports
    # celeba/celebahq; other datasets are skipped gracefully). 256 samples gives a
    # more stable estimate than the AR runs' in-loop count; a unified matched-N
    # post-hoc FID is the rigorous AR-vs-diffusion comparison.
    if not is_audio:
        diffusion_args += [
            "--compute-generation-fid",
            "--generation-metric-num-samples", "256",
            "--fid-dataset", str(case.name),
            "--fid-data-dir", str(case.data_dir),
            "--fid-image-size", str(case.image_size),
        ]
    diffusion_args.extend(extra_diffusion_args)

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

    supplied_ckpt = str(stage1_checkpoint.expanduser().resolve()) if stage1_checkpoint is not None else ""
    stage1_block = f"""CKPT={shlex.quote(supplied_ckpt)}
if [[ ! -f "$CKPT" ]]; then
  echo "Supplied stage-1 checkpoint not found: $CKPT" >&2
  exit 1
fi
echo "=== Using supplied stage-1 checkpoint ({case.name}): $CKPT ==="
""" if supplied_ckpt else f"""STAGE1_ARGS=(
{_bash_array(stage1_overrides)}
)

echo "=== Stage 1: autoencoder training ({case.name}) ==="
"$PYTHON_BIN" train.py stage1 "${{STAGE1_ARGS[@]}}"

CKPT="$(select_stage1_checkpoint "{stage1_dir}")"
if [[ -z "$CKPT" ]]; then
  echo "No stage-1 checkpoint found under {stage1_dir}" >&2
  exit 1
fi

if [[ "{'1' if int(stage1_adv_epochs) > 0 else '0'}" == "1" ]]; then
  RECON_CKPT="$CKPT"
  STAGE1_ADV_ARGS=(
{_bash_array(stage1_adv_overrides)}
    "init_ckpt_path=${{RECON_CKPT}}"
  )

  echo "=== Stage 1 adversarial continuation ({case.name}) ==="
  "$PYTHON_BIN" train.py stage1 "${{STAGE1_ADV_ARGS[@]}}"

  CKPT="$(select_stage1_checkpoint "{stage1_adv_dir}")"
  if [[ -z "$CKPT" ]]; then
    echo "No adversarial stage-1 checkpoint found under {stage1_adv_dir}" >&2
    exit 1
  fi
fi
"""

    run_script.write_text(
        f"""#!/bin/bash
set -euo pipefail

export PYTHONUSERBASE="${{PYTHONUSERBASE:-{_scratch_path('.pydeps', 'laser_src_py311')}}}"
export PATH="$PYTHONUSERBASE/bin:$PATH"
export PYTHONPATH="$PYTHONUSERBASE/lib/python3.11/site-packages:$PYTHONUSERBASE/lib/python3.12/site-packages:{snapshot_path}${{PYTHONPATH:+:$PYTHONPATH}}"
export WANDB_MODE="${{WANDB_MODE:-online}}"
export LASER_DISABLE_WANDB_MEDIA="${{LASER_DISABLE_WANDB_MEDIA:-0}}"
export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="${{XDG_CACHE_HOME:-/scratch/{_user()}/.cache}}"
export TORCH_HOME="${{TORCH_HOME:-$XDG_CACHE_HOME/torch}}"
export PIP_CACHE_DIR="${{PIP_CACHE_DIR:-$XDG_CACHE_HOME/pip}}"
export WANDB_CACHE_DIR="${{WANDB_CACHE_DIR:-$XDG_CACHE_HOME/wandb}}"
export WANDB_CONFIG_DIR="${{WANDB_CONFIG_DIR:-/scratch/{_user()}/.config/wandb}}"
export MPLCONFIGDIR="${{MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}}"
export HYDRA_FULL_ERROR=1
export TMPDIR="/tmp/laser_${{SLURM_JOB_ID:-$$}}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$MPLCONFIGDIR" "{stage1_dir / 'wandb'}" "{stage1_adv_dir / 'wandb'}" "{stage2_dir / 'wandb'}"

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

run_user_pip_install() {{
  mkdir -p "$PYTHONUSERBASE"
  if command -v flock >/dev/null 2>&1; then
    (
      flock 9
      "$PYTHON_BIN" -m pip install --user --quiet "$@" 2>/dev/null || true
    ) 9>"$PYTHONUSERBASE/.install.lock"
  else
    "$PYTHON_BIN" -m pip install --user --quiet "$@" 2>/dev/null || true
  fi
}}

run_user_pip_install \
  numpy scipy wandb lightning omegaconf hydra-core rich 'torchmetrics[image]' \
  torch-fidelity matplotlib lpips soundfile pystoi
# pesq has no cp311 wheel and the runtime container has no C compiler, so it is
# pre-built once and staged into the shared $PYTHONUSERBASE (see
# scripts/tools/build_pesq_into_pydeps.sh); _has_pesq() in src/audio_logging.py
# picks it up if present, otherwise PESQ is skipped gracefully (STOI still logs).

if ! "$PYTHON_BIN" - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("visqol") is not None else 1)
PY
then
  run_user_pip_install bazelisk pybind11 protobuf
  if ! command -v bazel >/dev/null 2>&1 && command -v bazelisk >/dev/null 2>&1; then
    ln -sf "$(command -v bazelisk)" "$PYTHONUSERBASE/bin/bazel" 2>/dev/null || true
  fi
  run_user_pip_install "git+https://github.com/google/visqol.git"
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

select_stage1_checkpoint() {{
  local root="$1"
  local ckpt=""
  ckpt="$(
    find "$root" -path '*/checkpoints/*' -type f -name '*.ckpt' \
      ! -name 'last.ckpt' ! -name 'final.ckpt' -printf '%T@ %p\\n' \
      | sort -n | tail -1 | cut -d ' ' -f2-
  )"
  if [[ -z "$ckpt" ]]; then
    ckpt="$(find "$root" -path '*/checkpoints/*' -type f -name 'last.ckpt' | sort | tail -1)"
  fi
  if [[ -z "$ckpt" ]]; then
    ckpt="$(find "$root" -path '*/checkpoints/*' -type f -name 'final.ckpt' | sort | tail -1)"
  fi
  printf '%s\\n' "$ckpt"
}}

cd "{snapshot_path}"

{stage1_block}

if [[ "{'1' if stage1_only else '0'}" == "1" ]]; then
  echo "=== Stage 1 only requested; skipping token cache and stage 2 ({case.name}) ==="
  exit 0
fi

CACHE_ARGS=(
{_bash_array(cache_args)}
)

NODE_RANK="${{SLURM_PROCID:-${{SLURM_NODEID:-0}}}}"
CACHE_DONE="{token_cache}.done"
CACHE_FAILED="{token_cache}.failed"

if [[ "$NODE_RANK" == "0" ]]; then
  rm -f "$CACHE_DONE" "$CACHE_FAILED"
  echo "=== Token cache extraction ({case.name}) ==="
  if "$PYTHON_BIN" cache.py "${{CACHE_ARGS[@]}}"; then
    touch "$CACHE_DONE"
  else
    rc=$?
    touch "$CACHE_FAILED"
    exit "$rc"
  fi
else
  echo "=== Waiting for token cache extraction ({case.name}) ==="
  while [[ ! -f "$CACHE_DONE" && ! -f "$CACHE_FAILED" ]]; do
    sleep 30
  done
  if [[ -f "$CACHE_FAILED" ]]; then
    echo "Token cache extraction failed on rank 0." >&2
    exit 1
  fi
fi

if [[ "{stage2_kind}" == "diffusion" ]]; then
  DIFFUSION_ARGS=(
{_bash_array(diffusion_args)}
  )

  echo "=== Stage 2: sparse coefficient diffusion prior ({case.name}) ==="
  "$PYTHON_BIN" train_stage2_diffusion_prior.py "${{DIFFUSION_ARGS[@]}}"
  exit 0
fi

STAGE2_ARGS=(
{_bash_array(stage2_overrides)}
)

echo "=== Stage 2: transformer prior training + generation ({case.name}) ==="
"$PYTHON_BIN" train.py stage2 "${{STAGE2_ARGS[@]}}"
""",
        encoding="utf-8",
    )
    os.chmod(run_script, 0o755)

    data_bind = f'    --bind "{case.data_dir}" \\\n'
    sbatch_script.write_text(
        f"""#!/bin/bash
set -euo pipefail

BOOTSTRAP_LOG="{run_dir}/bootstrap_${{SLURM_JOB_ID:-manual}}.log"
exec > "$BOOTSTRAP_LOG" 2>&1
set -x
echo "bootstrap-start host=$(hostname) date=$(date) job=${{SLURM_JOB_ID:-manual}}"
LASER_NODES={trainer_nodes}
LASER_GPUS_PER_NODE={trainer_devices}
LASER_WORLD_SIZE={trainer_world_size}
LAUNCH=()
if [[ "$LASER_NODES" -gt 1 ]]; then
  LAUNCH=(
    srun
    --nodes="$LASER_NODES"
    --ntasks="$LASER_WORLD_SIZE"
    --ntasks-per-node="$LASER_GPUS_PER_NODE"
  )
fi

if ! command -v module >/dev/null 2>&1; then
  if [[ -f /usr/share/lmod/lmod/init/bash ]]; then
    set +u; source /usr/share/lmod/lmod/init/bash; set -u
  elif [[ -f /usr/share/Modules/init/bash ]]; then
    set +u; source /usr/share/Modules/init/bash; set -u
  elif [[ -f /etc/profile.d/modules.sh ]]; then
    set +u; source /etc/profile.d/modules.sh; set -u
  fi
fi
echo "module=$(command -v module || true)"

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
echo "container_bin=$CONTAINER_BIN"

CONTAINER_CACHE_DIR="${{CONTAINER_CACHE_DIR:-/scratch/{_user()}/Projects/laser/.cache/laser_container_shared}}"
export APPTAINER_CACHEDIR="${{APPTAINER_CACHEDIR:-$CONTAINER_CACHE_DIR}}"
export SINGULARITY_CACHEDIR="${{SINGULARITY_CACHEDIR:-$CONTAINER_CACHE_DIR}}"
mkdir -p "$APPTAINER_CACHEDIR" "$SINGULARITY_CACHEDIR"

IMAGE="${{IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}}"
echo "image=$IMAGE"
nvidia-smi || true

if [[ -n "$CONTAINER_BIN" ]]; then
  "${{LAUNCH[@]}}" "$CONTAINER_BIN" exec --nv \
    --bind "{snapshot_path}" \
    --bind "/scratch/{_user()}" \
{data_bind}    --bind "/projects" \
    --bind "{run_dir}" \
    --bind /dev/shm \
    "$IMAGE" \
    bash "{run_script}"
else
  "${{LAUNCH[@]}}" bash "{run_script}"
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
    parser.add_argument("--gpus", type=int, default=1, help="GPUs per node.")
    parser.add_argument("--nodes", type=int, default=1, help="Number of SLURM nodes to request.")
    parser.add_argument("--project", default=None)
    parser.add_argument("--model-family", default="laser", choices=("vqvae", "laser"))
    parser.add_argument("--run-label", default="", help="Optional label inserted into snapshot, run root, and W&B group names.")
    parser.add_argument("--vctk-dir", default=_scratch_path("datasets", "VCTK-Corpus-smoke"))
    parser.add_argument("--celeba-dir", default=_scratch_path("datasets", "celeba_packed_128"))
    parser.add_argument("--celebahq-dir", default=_scratch_path("datasets", "celebahq_packed_256"))
    parser.add_argument("--coco-dir", default=_scratch_path("data", "coco"))
    parser.add_argument("--ffhq-dir", default=_scratch_path("datasets", "ffhq"))
    parser.add_argument("--imagenet-dir", default=_scratch_path("Projects", "data", "imagenet"))
    parser.add_argument("--imagenette2-dir", default=_scratch_path("datasets", "imagenette2"))
    parser.add_argument("--maestro-dir", default=_scratch_path("datasets", "maestro", "maestro-v3.0.0"))
    parser.add_argument("--cases", default="", help="Comma-separated subset of cases to submit, e.g. celeba,celebahq,imagenet,stl10,vctk.")
    parser.add_argument("--full-training", action="store_true", help="Use full configs/data limits instead of smoke settings.")
    parser.add_argument("--stage1-only", action="store_true", help="Train stage 1, then skip token cache extraction and stage 2.")
    parser.add_argument("--stage1-epochs", type=int, default=None, help="Stage-1 autoencoder epochs.")
    parser.add_argument(
        "--stage1-adv-epochs",
        type=int,
        default=0,
        help="Optional extra stage-1 epochs run after the first stage-1 phase, resuming from its checkpoint.",
    )
    parser.add_argument("--stage2-epochs", type=int, default=None, help="Stage-2 prior epochs.")
    parser.add_argument(
        "--stage1-checkpoint",
        type=Path,
        default=None,
        help="Existing stage-1 checkpoint to use for cache extraction; skips stage-1 training.",
    )
    parser.add_argument(
        "--stage2-kind",
        choices=("transformer", "diffusion"),
        default="transformer",
        help="Stage-2 prior type to train after token cache extraction.",
    )
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
        "--stage1-adv-override",
        action="append",
        default=[],
        help="Extra Hydra override appended only to the optional adversarial stage-1 continuation.",
    )
    parser.add_argument(
        "--cache-arg",
        action="append",
        default=[],
        help="Extra argument appended to cache.py. Repeat for flags and values, e.g. --cache-arg --coeff-bins --cache-arg 256.",
    )
    parser.add_argument(
        "--diffusion-arg",
        action="append",
        default=[],
        help="Extra argument appended to train_stage2_diffusion_prior.py when --stage2-kind=diffusion.",
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
    stage2_kind = str(args.stage2_kind or "transformer").strip().lower()
    num_nodes = max(1, int(args.nodes))
    if int(args.nodes) < 1:
        raise SystemExit("--nodes must be >= 1.")
    if int(args.gpus) < 1:
        raise SystemExit("--gpus must be >= 1.")
    stage1_checkpoint = Path(args.stage1_checkpoint).expanduser().resolve() if args.stage1_checkpoint else None
    if stage1_checkpoint is not None and not stage1_checkpoint.is_file():
        raise SystemExit(f"--stage1-checkpoint not found: {stage1_checkpoint}")
    if num_nodes > 1 and not stage1_only and stage2_kind != "transformer":
        raise SystemExit("--nodes > 1 full-pipeline orchestration is currently supported only for transformer stage 2.")
    stage1_epochs = int(args.stage1_epochs if args.stage1_epochs is not None else (10 if full_training else 1))
    stage1_adv_epochs = int(args.stage1_adv_epochs or 0)
    stage2_epochs = int(args.stage2_epochs if args.stage2_epochs is not None else (10 if full_training else 1))
    if stage1_epochs <= 0 or stage2_epochs <= 0:
        raise SystemExit("Stage epoch counts must be positive.")
    if stage1_adv_epochs < 0:
        raise SystemExit("Stage-1 adversarial epoch count must be non-negative.")
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
        celeba_dir=Path(args.celeba_dir).expanduser().resolve(),
        celebahq_dir=Path(args.celebahq_dir).expanduser().resolve(),
        coco_dir=coco_dir,
        ffhq_dir=Path(args.ffhq_dir).expanduser().resolve(),
        imagenet_dir=Path(args.imagenet_dir).expanduser().resolve(),
        imagenette2_dir=Path(args.imagenette2_dir).expanduser().resolve(),
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

    group_name = _bounded_label(f"{model_family}-{run_kind}{group_label_part}-{stamp}")
    run_root_base = args.run_root_base or _scratch_path("runs", f"{model_family}_{run_kind}")
    run_root = Path(run_root_base).expanduser().resolve() / group_name
    run_root.mkdir(parents=True, exist_ok=True)
    project = args.project or ("laser-multimodal-training" if full_training else "laser-multimodal-smoke")
    job_prefix = "ls" if model_family == "laser" else "vq"
    slurm_ntasks = num_nodes * int(args.gpus) if num_nodes > 1 else 1
    slurm_ntasks_per_node = int(args.gpus) if num_nodes > 1 else 1

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
            stage1_adv_epochs=stage1_adv_epochs,
            stage2_epochs=stage2_epochs,
            stage2_kind=stage2_kind,
            num_gpus=int(args.gpus),
            num_nodes=num_nodes,
            stage1_checkpoint=stage1_checkpoint,
            stage1_only=stage1_only,
            extra_cache_args=tuple(str(v) for v in args.cache_arg),
            extra_stage1_overrides=tuple(str(v) for v in args.stage1_override),
            extra_stage1_adv_overrides=tuple(str(v) for v in args.stage1_adv_override),
            extra_stage2_overrides=tuple(str(v) for v in args.stage2_override),
            extra_diffusion_args=tuple(str(v) for v in args.diffusion_arg),
        )
        log_base = run_dir / case.name
        sbatch_cmd = [
            "sbatch",
            f"--partition={args.partition}",
            f"--job-name={job_prefix}{run_kind[:3]}-{case.name}",
            f"--nodes={num_nodes}",
            f"--ntasks={slurm_ntasks}",
            f"--ntasks-per-node={slurm_ntasks_per_node}",
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
    if not stage1_only:
        print(f"Stage 2:   {stage2_kind}")
    adv_part = f" stage1_adv={stage1_adv_epochs}" if stage1_adv_epochs > 0 else ""
    print(f"Epochs:    stage1={stage1_epochs}{adv_part}" + ("" if stage1_only else f" stage2={stage2_epochs}"))
    print(f"Nodes/job: {num_nodes}")
    print(f"GPUs/node: {int(args.gpus)}")
    print(f"GPUs/job:  {num_nodes * int(args.gpus)}")
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
