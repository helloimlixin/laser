#!/usr/bin/env python3
"""Watch W&B for upstream-loss checkpoints and run full ImageNet rFID.

The July upstream-loss sweep wrote checkpoints to cluster-local scratch rather
than W&B.  This watcher waits for those files to be added to their source runs
(as run files or logged model artifacts), validates the dictionary size, and
launches one 50k-image evaluation per GPU.  State is durable so restarting the
watcher does not repeat completed evaluations.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import wandb


ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ROOT / "third_party" / "rq-vae-transformer"
for import_root in (THIRD_PARTY, ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))
DEFAULT_OUTPUT = ROOT / "outputs" / "x3h5cl0h_lw075_upstreamloss_rfid_sweep"
ENTITY = "helloimlixin-rutgers"
PROJECT = "laser"


@dataclass(frozen=True)
class Target:
    atoms: int
    run_id: str

    @property
    def key(self) -> str:
        return f"a{self.atoms}"

    @property
    def run_path(self) -> str:
        return f"{ENTITY}/{PROJECT}/{self.run_id}"


TARGETS = (
    Target(2_048, "x3h5cl0h-lw075-upstreamloss-4gpu-a2048-k2-20260724-225650"),
    Target(4_096, "x3h5cl0h-lw075-upstreamloss-4gpu-a4096-k2-20260724-225650"),
    Target(8_192, "x3h5cl0h-lw075-upstreamloss-4gpu-a8192-k2-20260724-225650"),
)

CHECKPOINT_BASENAME_PRIORITY = (
    "last_model.pt",
    "last.pt",
    "epoch10_model.pt",
    "best_rfid_slot1_model.pt",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def emit(message: str) -> None:
    print(f"{utc_now()} {message}", flush=True)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_state(path: Path) -> dict[str, Any]:
    if path.is_file():
        try:
            payload = json.loads(path.read_text())
            if isinstance(payload, dict):
                return payload
        except (OSError, ValueError, TypeError):
            pass
    return {"created_at": utc_now(), "targets": {}}


def checkpoint_rank(name: str) -> tuple[int, str]:
    basename = Path(name).name.lower()
    try:
        priority = CHECKPOINT_BASENAME_PRIORITY.index(basename)
    except ValueError:
        priority = len(CHECKPOINT_BASENAME_PRIORITY)
    return priority, name


def is_checkpoint_name(name: str) -> bool:
    basename = Path(name).name.lower()
    if not basename.endswith((".pt", ".pth", ".ckpt")):
        return False
    return (
        basename in CHECKPOINT_BASENAME_PRIORITY
        or "last" in basename
        or "epoch10" in basename
        or "checkpoint" in basename
        or "model" in basename
    )


def select_run_file(run) -> Any | None:
    candidates = [item for item in run.files() if is_checkpoint_name(item.name)]
    if not candidates:
        return None
    candidates.sort(key=lambda item: checkpoint_rank(item.name))
    return candidates[0]


def select_artifact(run) -> tuple[Any, Any] | None:
    candidates: list[tuple[Any, Any]] = []
    for artifact in run.logged_artifacts():
        if str(artifact.type).lower() not in {"model", "checkpoint"}:
            continue
        for item in artifact.files():
            if is_checkpoint_name(item.name):
                candidates.append((artifact, item))
    if not candidates:
        return None
    candidates.sort(key=lambda pair: checkpoint_rank(pair[1].name))
    return candidates[0]


def download_run_file(item, destination_root: Path) -> Path:
    destination_root.mkdir(parents=True, exist_ok=True)
    downloaded = item.download(root=str(destination_root), replace=True)
    path = Path(downloaded.name if hasattr(downloaded, "name") else downloaded)
    if not path.is_absolute():
        path = destination_root / path
    return path.resolve()


def download_artifact_file(artifact, item, destination_root: Path) -> Path:
    destination_root.mkdir(parents=True, exist_ok=True)
    # Artifact entries support selective downloads, avoiding unrelated large
    # files when a model artifact contains more than one retained checkpoint.
    try:
        downloaded = item.download(root=str(destination_root))
        path = Path(downloaded)
    except (AttributeError, TypeError):
        artifact_root = Path(artifact.download(root=str(destination_root)))
        path = artifact_root / item.name
    if not path.is_absolute():
        path = destination_root / path
    return path.resolve()


def checkpoint_dictionary_shape(path: Path) -> tuple[int, int]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError as exc:
        if "mmap can only be used" not in str(exc):
            raise
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a checkpoint dictionary")
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict):
        raise ValueError(f"{path} has no state_dict")
    dictionary = None
    for name in ("quantizer.dictionary", "module.quantizer.dictionary"):
        if name in state:
            dictionary = state[name]
            break
    if dictionary is None or dictionary.ndim != 2:
        raise ValueError(f"{path} has no two-dimensional LASER dictionary")
    shape = tuple(int(value) for value in dictionary.shape)
    del dictionary, state, payload
    gc.collect()
    return shape


def discover_and_download(api, target: Target, target_dir: Path) -> tuple[Path, str] | None:
    run = api.run(target.run_path)
    run_file = select_run_file(run)
    if run_file is not None:
        emit(f"{target.key}: found W&B run file {run_file.name}; downloading")
        return download_run_file(run_file, target_dir / "checkpoint"), f"run-file:{run_file.name}"
    artifact_match = select_artifact(run)
    if artifact_match is not None:
        artifact, item = artifact_match
        emit(
            f"{target.key}: found W&B artifact {artifact.name}/{item.name}; downloading"
        )
        path = download_artifact_file(artifact, item, target_dir / "checkpoint")
        return path, f"artifact:{artifact.name}/{item.name}"
    return None


def launch_evaluation(
    target: Target,
    checkpoint: Path,
    target_dir: Path,
    gpu: str,
    args: argparse.Namespace,
) -> tuple[subprocess.Popen, Any, Path, Path]:
    result_path = target_dir / "full_imagenet_val_rfid.json"
    log_path = target_dir / "rfid.log"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_upstream_laser_rfid.py"),
        "--checkpoint", str(checkpoint),
        "--data", str(args.data),
        "--output", str(result_path),
        "--dataset", "imagenet",
        "--num-images", str(args.num_images),
        "--num-atoms", str(target.atoms),
        "--sparsity-level", "2",
        "--batch-size", str(args.batch_size),
        "--backend", "torchmetrics",
        "--wandb-mode", "disabled",
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    environment.setdefault("WANDB_DIR", str(ROOT / "wandb"))
    environment.setdefault("WANDB_CACHE_DIR", str(ROOT / ".cache" / "wandb"))
    environment.setdefault("WANDB_DATA_DIR", str(ROOT / ".local" / "share" / "wandb"))
    environment.setdefault("OMP_NUM_THREADS", "8")
    target_dir.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", buffering=1)
    emit(f"{target.key}: launching rFID50k on physical GPU {gpu}")
    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        env=environment,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, log_handle, result_path, log_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/workspace/Projects/data/imagenet"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--retry-seconds", type=int, default=300)
    parser.add_argument("--num-images", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--once", action="store_true", help="Poll once and do not wait")
    parser.add_argument("--dry-run", action="store_true", help="Discover only; do not download or evaluate")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output = args.output.expanduser().resolve()
    args.data = args.data.expanduser().resolve()
    if not (args.data / "val").is_dir():
        raise SystemExit(f"ImageNet validation directory is missing: {args.data / 'val'}")
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise SystemExit("--gpus must contain at least one physical GPU index")
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "monitor_state.json"
    state = load_state(state_path)
    target_by_key = {target.key: target for target in TARGETS}
    running: dict[str, tuple[subprocess.Popen, Any, Path, Path, str]] = {}
    gpu_owner: dict[str, str] = {}
    emit(
        "watching W&B for " + ", ".join(target.run_path for target in TARGETS)
    )

    while True:
        now = time.time()
        # Reap completed evaluators before assigning newly uploaded checkpoints.
        for key, (process, log_handle, result_path, log_path, gpu) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            log_handle.close()
            gpu_owner.pop(gpu, None)
            target_state = state["targets"].setdefault(key, {})
            target_state["evaluation_finished_at"] = utc_now()
            target_state["evaluation_exit_code"] = int(return_code)
            target_state["log"] = str(log_path)
            if return_code == 0 and result_path.is_file():
                result = json.loads(result_path.read_text())
                try:
                    source_run = wandb.Api(timeout=90).run(target_by_key[key].run_path)
                    source_run.summary["diagnostics/continuous_reconstruction_rfid"] = float(
                        result["rfid"]
                    )
                    source_run.summary["diagnostics/reconstruction_rfid_num_images"] = int(
                        result["num_images"]
                    )
                    source_run.summary["diagnostics/reconstruction_rfid_backend"] = str(
                        result["fid_backend"]
                    )
                    source_run.summary["diagnostics/reconstruction_rfid_evaluated_at"] = utc_now()
                    source_run.summary.update()
                    target_state["wandb_summary_updated"] = True
                except Exception as exc:
                    target_state["wandb_summary_updated"] = False
                    target_state["wandb_summary_error"] = f"{type(exc).__name__}: {exc}"
                    emit(f"{key}: rFID computed but W&B summary update failed: {exc}")
                target_state.update(
                    status="completed",
                    result=str(result_path),
                    rfid=float(result["rfid"]),
                )
                emit(f"{key}: completed rFID50k={float(result['rfid']):.6f}")
            else:
                target_state.update(
                    status="retry",
                    last_error=f"evaluator exited {return_code}; see {log_path}",
                    retry_after=now + args.retry_seconds,
                )
                emit(f"{key}: evaluator failed with exit {return_code}; retry scheduled")
            del running[key]
            atomic_json(state_path, state)

        completed = {
            key for key, item in state.get("targets", {}).items()
            if item.get("status") == "completed"
        }
        if completed == {target.key for target in TARGETS}:
            state["completed_at"] = utc_now()
            atomic_json(state_path, state)
            emit("all requested upstream-loss rFID evaluations are complete")
            return 0

        free_gpus = [gpu for gpu in gpus if gpu not in gpu_owner]
        api = wandb.Api(timeout=90)
        for target in TARGETS:
            if target.key in completed or target.key in running:
                continue
            target_state = state["targets"].setdefault(
                target.key,
                {"status": "waiting", "run_path": target.run_path, "atoms": target.atoms},
            )
            try:
                target_dir = args.output / target.key
                local_candidates = sorted(
                    (
                        path for path in (target_dir / "checkpoint").rglob("*")
                        if path.is_file() and is_checkpoint_name(path.name)
                    ),
                    key=lambda path: checkpoint_rank(path.name),
                )
                checkpoint = local_candidates[0] if local_candidates else None
                source = target_state.get("source", "local-cache") if checkpoint else None
                run_file = None
                artifact_match = None
                if checkpoint is None:
                    if now < float(target_state.get("retry_after", 0)):
                        continue
                    run = api.run(target.run_path)
                    run_file = select_run_file(run)
                    artifact_match = select_artifact(run) if run_file is None else None
                    if run_file is None and artifact_match is None:
                        target_state.update(status="waiting", last_checked_at=utc_now())
                        atomic_json(state_path, state)
                        continue
                if args.dry_run:
                    if checkpoint is not None:
                        source = f"local-cache:{checkpoint}"
                    else:
                        source = (
                            f"run-file:{run_file.name}"
                            if run_file is not None
                            else f"artifact:{artifact_match[0].name}/{artifact_match[1].name}"
                        )
                    emit(f"{target.key}: discovered {source} (dry run)")
                    target_state.update(status="discovered", source=source, last_checked_at=utc_now())
                    atomic_json(state_path, state)
                    continue
                if not free_gpus:
                    target_state.update(status="ready", last_checked_at=utc_now())
                    continue
                if checkpoint is None:
                    target_state.update(status="downloading", last_checked_at=utc_now())
                    atomic_json(state_path, state)
                    if run_file is not None:
                        emit(f"{target.key}: found W&B run file {run_file.name}; downloading")
                        checkpoint = download_run_file(
                            run_file, args.output / target.key / "checkpoint"
                        )
                        source = f"run-file:{run_file.name}"
                    else:
                        artifact, item = artifact_match
                        emit(
                            f"{target.key}: found W&B artifact "
                            f"{artifact.name}/{item.name}; downloading"
                        )
                        checkpoint = download_artifact_file(
                            artifact, item, args.output / target.key / "checkpoint"
                        )
                        source = f"artifact:{artifact.name}/{item.name}"
                shape = checkpoint_dictionary_shape(checkpoint)
                if shape != (256, target.atoms):
                    raise ValueError(
                        f"dictionary shape {shape} does not match expected (256, {target.atoms})"
                    )
                gpu = free_gpus.pop(0)
                process, log_handle, result_path, log_path = launch_evaluation(
                    target, checkpoint, args.output / target.key, gpu, args
                )
                running[target.key] = (process, log_handle, result_path, log_path, gpu)
                gpu_owner[gpu] = target.key
                target_state.update(
                    status="running",
                    source=source,
                    checkpoint=str(checkpoint),
                    dictionary_shape=list(shape),
                    gpu=gpu,
                    evaluator_pid=process.pid,
                    evaluation_started_at=utc_now(),
                )
            except Exception as exc:
                target_state.update(
                    status="retry",
                    last_checked_at=utc_now(),
                    last_error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(),
                    retry_after=now + args.retry_seconds,
                )
                emit(f"{target.key}: {type(exc).__name__}: {exc}; retry scheduled")
            atomic_json(state_path, state)

        if args.once:
            if running:
                emit("--once requested; leaving launched evaluator processes detached")
            return 0
        time.sleep(max(10, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
