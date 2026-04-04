
"""
laser.py

Minimal LASER training script.

It keeps the core two-stage workflow:
  - stage 1: train the LASER autoencoder
  - stage 2: train a transformer prior on flattened sparse tokens

The default dataset is CelebA under ../../data/celeba relative to this file.
For multi-GPU runs, launch with torchrun.
"""
import argparse
import datetime
import hashlib
import importlib
import importlib.util
import math
import os
import shutil
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
try:
    from spatial_prior import SpatialDepthPrior, build_spatial_depth_prior_config
except ModuleNotFoundError:
    from laser_transformer import SpatialDepthPrior, build_spatial_depth_prior_config
from mingpt_prior import MinGPTQuantizedPrior, build_mingpt_quantized_prior_config
from PIL import Image
from scipy.linalg import sqrtm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision import datasets, transforms, utils
from tqdm import tqdm
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except Exception:
    FrechetInceptionDistance = None
def _import_real_wandb():
    """Import the real wandb package even if the workspace has a local wandb/ directory."""

    def _is_real_wandb(module) -> bool:
        return bool(module) and hasattr(module, "init") and hasattr(module, "Api")

    try:
        module = importlib.import_module("wandb")
        if _is_real_wandb(module):
            return module
    except Exception:
        module = None

    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    blocked_paths = {script_dir, cwd}
    original_sys_path = list(sys.path)

    def _iter_site_packages_paths():
        seen = set()
        pyuserbase = os.environ.get("PYTHONUSERBASE")
        if pyuserbase:
            for candidate in Path(pyuserbase).expanduser().glob("lib/python*/site-packages"):
                resolved = candidate.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    yield resolved
        for entry in original_sys_path:
            try:
                resolved = Path(entry).resolve()
            except Exception:
                continue
            if "site-packages" in resolved.parts and resolved not in seen:
                seen.add(resolved)
                yield resolved

    try:
        sys.modules.pop("wandb", None)
        filtered = []
        for entry in original_sys_path:
            try:
                resolved = cwd if entry in ("", ".") else Path(entry).resolve()
            except Exception:
                resolved = None
            if resolved in blocked_paths:
                continue
            filtered.append(entry)
        sys.path = filtered
        module = importlib.import_module("wandb")
        if _is_real_wandb(module):
            return module

        for site_packages in _iter_site_packages_paths():
            init_py = site_packages / "wandb" / "__init__.py"
            if not init_py.exists():
                continue
            spec = importlib.util.spec_from_file_location(
                "wandb",
                init_py,
                submodule_search_locations=[str(init_py.parent)],
            )
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules["wandb"] = module
            spec.loader.exec_module(module)
            if _is_real_wandb(module):
                return module
            sys.modules.pop("wandb", None)
    except Exception:
        return None
    finally:
        sys.path = original_sys_path
    return None


wandb = _import_real_wandb()


def _nan_to_num_tensor(
    x: torch.Tensor,
    *,
    nan: float = 0.0,
    posinf: float = 0.0,
    neginf: float = 0.0,
) -> torch.Tensor:
    if torch.isfinite(x).all():
        return x
    return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


def _gaussian_kl_to_fixed_mean(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: float,
) -> torch.Tensor:
    """KL(q || p) for diagonal Gaussians where p uses a fixed std and mean."""
    target_std = float(max(target_std, 1e-6))
    target_var = target_std * target_std
    var = logvar.exp().clamp_min(1e-8)
    sq_mean = (mu - target_mean).square()
    kl = 0.5 * ((var + sq_mean) / target_var - 1.0 + math.log(target_var) - logvar)
    return kl.mean()


# -----------------------------
# VQ-VAE style building blocks
# -----------------------------

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CELEBA_DIR = (SCRIPT_DIR / "../../data/celeba").resolve()
DEFAULT_STAGE1_SOURCE_RUN = "helloimlixin-rutgers/laser/4psikzda"
DEFAULT_STAGE1_SOURCE_CKPT_QUANTIZED = (SCRIPT_DIR / "checkpoints" / "quantized" / "stage1" / "ae_best.pt").resolve()
DEFAULT_STAGE1_SOURCE_CKPT_NOQUANTIZED = (SCRIPT_DIR / "checkpoints" / "noquantized" / "stage1" / "ae_best.pt").resolve()
LEGACY_DEFAULT_STAGE1_SOURCE_CKPT = (
    SCRIPT_DIR / "runs" / "laser_celeba128_quantized" / "20260311_000321" / "stage1" / "ae_best.pt"
).resolve()
DEFAULT_STAGE2_SOURCE_RUN = "helloimlixin-rutgers/laser/q5l0g3jn"
DEFAULT_STAGE2_SOURCE_TOKEN_CACHE = (
    SCRIPT_DIR / "runs" / "laser_celeba128_quantized" / "20260311_112058" / "stage2" / "tokens_cache.pt"
).resolve()
STAGE1_AUTORESUME_CHECKPOINT_NAMES = ("ae_resume_latest.pt", "ae_last.pt", "ae_best.pt")


def _parse_cli_bool(value) -> bool:
    """Parse a bool flag from common CLI spellings."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def _default_image_size(dataset: str) -> int:
    return 128 if dataset == "celeba" else 32


def _default_data_dir(dataset: str) -> Path:
    if dataset == "celeba":
        return DEFAULT_CELEBA_DIR
    return (SCRIPT_DIR / "data").resolve()


def _default_run_name(dataset: str, image_size: int, quantized: bool) -> str:
    suffix = "_quantized" if quantized else ""
    return f"laser_{dataset}{image_size}{suffix}"


def _default_out_dir(dataset: str, image_size: int, quantized: bool) -> Path:
    return (SCRIPT_DIR / "runs" / _default_run_name(dataset, image_size, quantized)).resolve()


def _default_stage1_source_ckpt(quantized: bool) -> Optional[Path]:
    primary = DEFAULT_STAGE1_SOURCE_CKPT_QUANTIZED if quantized else DEFAULT_STAGE1_SOURCE_CKPT_NOQUANTIZED
    if primary.exists():
        return primary
    if quantized and LEGACY_DEFAULT_STAGE1_SOURCE_CKPT.exists():
        return LEGACY_DEFAULT_STAGE1_SOURCE_CKPT
    return None


def _broadcast_optional_string(value: Optional[str], src: int = 0) -> str:
    """Broadcast a short string from src to all ranks."""
    if not _is_distributed():
        if value is None:
            raise ValueError("value must be provided when distributed training is disabled")
        return str(value)
    obj_list = [value if dist.get_rank() == src else None]
    dist.broadcast_object_list(obj_list, src=src)
    if obj_list[0] is None:
        raise RuntimeError("failed to broadcast launch timestamp")
    return str(obj_list[0])


def _launch_timestamp() -> str:
    """Return a single launch timestamp shared across all ranks."""
    value = None
    if not _is_distributed() or dist.get_rank() == 0:
        value = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return _broadcast_optional_string(value, src=0)


def _resolve_run_out_dir(base_dir: str, launch_timestamp: str) -> Path:
    """Create a per-launch run directory under the provided experiment root."""
    return Path(base_dir).expanduser().resolve() / launch_timestamp


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA256 hex digest for a file."""
    resolved = Path(path).expanduser().resolve()
    hasher = hashlib.sha256()
    with resolved.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _stage1_checkpoint_cache_key(path: Optional[Path]) -> Optional[str]:
    """Build a stable cache key for the stage-1 checkpoint used by stage 2."""
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Stage-1 checkpoint does not exist: {resolved}")
    return f"sha256:{_sha256_file(resolved)}"


def _find_latest_stage1_checkpoint(experiment_root: Path, current_run_dir: Path) -> Optional[Path]:
    """Find the newest prior stage-1 checkpoint under the experiment root."""
    candidates = []
    legacy_ckpt = experiment_root / "stage1" / "ae_best.pt"
    if legacy_ckpt.exists():
        candidates.append(legacy_ckpt)

    if experiment_root.exists():
        for child in experiment_root.iterdir():
            if not child.is_dir():
                continue
            if child.resolve() == current_run_dir.resolve():
                continue
            ckpt = child / "stage1" / "ae_best.pt"
            if ckpt.exists():
                candidates.append(ckpt)

    if not candidates:
        return None
    return max(candidates, key=lambda p: (p.stat().st_mtime, str(p)))


def _find_latest_stage1_autoresume_checkpoint(
    experiment_root: Path,
    current_run_dir: Path,
) -> Optional[Path]:
    """Find the newest stage-1 snapshot suitable for warm-starting a rerun."""
    run_dirs = []
    seen_dirs = set()

    def _maybe_add_run_dir(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen_dirs or not path.exists() or not path.is_dir():
            return
        seen_dirs.add(resolved)
        run_dirs.append(path)

    _maybe_add_run_dir(current_run_dir)
    _maybe_add_run_dir(experiment_root)
    if experiment_root.exists():
        for child in experiment_root.iterdir():
            if child.is_dir():
                _maybe_add_run_dir(child)

    candidates = []
    name_priority = {name: idx for idx, name in enumerate(STAGE1_AUTORESUME_CHECKPOINT_NAMES)}
    for run_dir in run_dirs:
        stage1_dir = run_dir / "stage1"
        if not stage1_dir.exists():
            continue
        for checkpoint_name in STAGE1_AUTORESUME_CHECKPOINT_NAMES:
            checkpoint_path = stage1_dir / checkpoint_name
            if checkpoint_path.exists():
                candidates.append(checkpoint_path)

    if not candidates:
        return None
    return max(
        candidates,
        key=lambda p: (
            p.stat().st_mtime,
            -name_priority.get(p.name, len(STAGE1_AUTORESUME_CHECKPOINT_NAMES)),
            str(p),
        ),
    )


def _resolve_stage1_checkpoint_from_wandb_run(run_path: str, cache_root: Path, quantized: bool) -> Path:
    """Resolve ae_best.pt from a W&B run via its recorded out_dir or uploaded files."""
    run_ref = str(run_path).strip()
    if not run_ref:
        raise ValueError("stage1_source_run must not be empty")
    if run_ref == DEFAULT_STAGE1_SOURCE_RUN:
        default_ckpt = _default_stage1_source_ckpt(quantized)
        if default_ckpt is not None:
            return default_ckpt
    if wandb is None:
        raise RuntimeError("wandb is not installed; cannot resolve a stage-1 checkpoint from a W&B run")

    api = wandb.Api()
    run = api.run(run_ref)

    out_dir = run.config.get("out_dir")
    out_root = run.config.get("out_root")
    launch_timestamp = run.config.get("launch_timestamp")
    if out_dir:
        out_dir_path = Path(str(out_dir)).expanduser().resolve()
        local_candidates = (
            out_dir_path / "stage1" / "ae_best.pt",
            out_dir_path / "ae_best.pt",
        )
        for candidate in local_candidates:
            if candidate.exists():
                return candidate
    if out_root and launch_timestamp:
        candidate = Path(str(out_root)).expanduser().resolve() / str(launch_timestamp) / "stage1" / "ae_best.pt"
        if candidate.exists():
            return candidate

    uploaded_file = None
    for file_obj in run.files():
        name = str(file_obj.name).replace("\\", "/")
        if name == "ae_best.pt" or name.endswith("/ae_best.pt"):
            uploaded_file = file_obj
            break

    if uploaded_file is not None:
        download_root = cache_root / "wandb_stage1_ckpts" / run.entity / run.project / run.id
        download_root.mkdir(parents=True, exist_ok=True)
        downloaded = uploaded_file.download(root=str(download_root), replace=True, exist_ok=True)
        downloaded_path = Path(downloaded.name).resolve()
        downloaded.close()
        return downloaded_path

    raise FileNotFoundError(
        f"Could not resolve stage-1 checkpoint for W&B run {run_ref}. "
        f"Checked run.config['out_dir']={out_dir!r}, out_root={out_root!r}, "
        f"launch_timestamp={launch_timestamp!r}, and uploaded run files for ae_best.pt."
    )


def _resolve_stage2_token_cache_from_wandb_run(run_path: str, cache_root: Path) -> Path:
    """Resolve tokens_cache.pt from a W&B run via its recorded out_dir or uploaded files."""
    run_ref = str(run_path).strip()
    if not run_ref:
        raise ValueError("stage2_source_run must not be empty")
    if run_ref == DEFAULT_STAGE2_SOURCE_RUN and DEFAULT_STAGE2_SOURCE_TOKEN_CACHE.exists():
        return DEFAULT_STAGE2_SOURCE_TOKEN_CACHE
    if wandb is None:
        raise RuntimeError("wandb is not installed; cannot resolve a stage-2 token cache from a W&B run")

    api = wandb.Api()
    run = api.run(run_ref)

    out_dir = run.config.get("out_dir")
    out_root = run.config.get("out_root")
    launch_timestamp = run.config.get("launch_timestamp")
    if out_dir:
        out_dir_path = Path(str(out_dir)).expanduser().resolve()
        local_candidates = (
            out_dir_path / "stage2" / "tokens_cache.pt",
            out_dir_path / "tokens_cache.pt",
        )
        for candidate in local_candidates:
            if candidate.exists():
                return candidate
    if out_root and launch_timestamp:
        candidate = Path(str(out_root)).expanduser().resolve() / str(launch_timestamp) / "stage2" / "tokens_cache.pt"
        if candidate.exists():
            return candidate

    uploaded_file = None
    for file_obj in run.files():
        name = str(file_obj.name).replace("\\", "/")
        if name == "tokens_cache.pt" or name.endswith("/tokens_cache.pt"):
            uploaded_file = file_obj
            break

    if uploaded_file is not None:
        download_root = cache_root / "wandb_stage2_token_caches" / run.entity / run.project / run.id
        download_root.mkdir(parents=True, exist_ok=True)
        downloaded = uploaded_file.download(root=str(download_root), replace=True, exist_ok=True)
        downloaded_path = Path(downloaded.name).resolve()
        downloaded.close()
        return downloaded_path

    raise FileNotFoundError(
        f"Could not resolve stage-2 token cache for W&B run {run_ref}. "
        f"Checked run.config['out_dir']={out_dir!r}, out_root={out_root!r}, "
        f"launch_timestamp={launch_timestamp!r}, and uploaded run files for tokens_cache.pt."
    )


def _find_latest_stage2_token_cache(experiment_root: Path, current_run_dir: Path) -> Optional[Path]:
    """Find the newest prior stage-2 token cache under the experiment root."""
    candidates = []
    legacy_cache = experiment_root / "stage2" / "tokens_cache.pt"
    if legacy_cache.exists():
        candidates.append(legacy_cache)

    if experiment_root.exists():
        for child in experiment_root.iterdir():
            if not child.is_dir():
                continue
            if child.resolve() == current_run_dir.resolve():
                continue
            cache_path = child / "stage2" / "tokens_cache.pt"
            if cache_path.exists():
                candidates.append(cache_path)

    if not candidates:
        return None
    return max(candidates, key=lambda p: (p.stat().st_mtime, str(p)))


def _init_distributed(timeout_minutes: int) -> Tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0, 1
    if not torch.cuda.is_available():
        raise RuntimeError("Multi-GPU training requires CUDA.")

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    visible_device_count = torch.cuda.device_count()
    if visible_device_count <= 0:
        raise RuntimeError("CUDA is available but no devices are visible to this process.")
    device_index = local_rank
    if device_index >= visible_device_count:
        if visible_device_count == 1:
            # Some SLURM launches expose one GPU per task but still assign LOCAL_RANK
            # in the node-local task range. In that case the only valid visible device is 0.
            device_index = 0
        else:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {visible_device_count} CUDA devices are visible"
            )
    torch.cuda.set_device(device_index)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(minutes=int(timeout_minutes)),
        )
    return True, rank, device_index, world_size


def _cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _barrier():
    if _is_distributed():
        if torch.cuda.is_available():
            # NCCL warns if barrier cannot infer the rank-to-device mapping.
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _write_atomic_text(path: Path, text: str) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp_path, path)


def _wait_for_file_signal(
    ready_path: Path,
    error_path: Optional[Path],
    timeout_seconds: float,
    description: str,
    poll_interval: float = 1.0,
) -> None:
    deadline = time.monotonic() + max(1.0, float(timeout_seconds))
    while True:
        if ready_path.exists():
            return
        if error_path is not None and error_path.exists():
            try:
                error_message = error_path.read_text(encoding="utf-8").strip()
            except OSError:
                error_message = ""
            if not error_message:
                error_message = f"{description} failed on rank 0"
            raise RuntimeError(error_message)
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for {description}. "
                f"Expected ready signal at {ready_path} within {int(timeout_seconds)} seconds."
            )
        time.sleep(max(0.1, float(poll_interval)))


def _load_module_checkpoint(module: nn.Module, checkpoint_path: Path) -> None:
    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    module.load_state_dict(state_dict)


def _save_module_checkpoint(module: nn.Module, checkpoint_path: Path) -> None:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(f"{checkpoint_path.suffix}.tmp")
    torch.save(module.state_dict(), tmp_path)
    os.replace(tmp_path, checkpoint_path)


def _artifact_safe_token(value) -> str:
    raw = "" if value is None else str(value).strip()
    if not raw:
        return ""
    cleaned = []
    last_dash = False
    for ch in raw:
        out = ch if (ch.isalnum() or ch in {"-", "_", "."}) else "-"
        if out == "-" and last_dash:
            continue
        cleaned.append(out)
        last_dash = (out == "-")
    return "".join(cleaned).strip("-_.")


def _resolve_run_id(explicit: Optional[str] = None, fallback_timestamp: Optional[str] = None) -> str:
    for candidate in (
        explicit,
        os.environ.get("PROTO_RUN_ID"),
        os.environ.get("WANDB_RUN_ID"),
    ):
        token = _artifact_safe_token(candidate)
        if token:
            return token
    fallback = _artifact_safe_token(fallback_timestamp)
    if fallback:
        return f"run-{fallback}"
    return ""


def _build_run_artifact_tag(args) -> str:
    patch_reconstruction = _canonical_patch_reconstruction(
        getattr(args, "patch_reconstruction", None),
        patch_size=getattr(args, "patch_size", None),
        patch_stride=getattr(args, "patch_stride", None),
    )
    payload_fields = (
        ("dataset", getattr(args, "dataset", None)),
        ("image_size", getattr(args, "image_size", None)),
        ("patch_based", getattr(args, "patch_based", None)),
        ("patch_size", getattr(args, "patch_size", None)),
        ("patch_stride", getattr(args, "patch_stride", None)),
        ("patch_reconstruction", patch_reconstruction),
        ("quantize_sparse_coeffs", getattr(args, "quantize_sparse_coeffs", None)),
        ("variational_coeffs", getattr(args, "variational_coeffs", None)),
        ("embedding_dim", getattr(args, "embedding_dim", None)),
        ("num_atoms", getattr(args, "num_atoms", None)),
        ("sparsity_level", getattr(args, "sparsity_level", None)),
        ("n_bins", getattr(args, "n_bins", None)),
        ("coef_max", getattr(args, "coef_max", None)),
        ("stage1_epochs", getattr(args, "stage1_epochs", None)),
        ("stage2_epochs", getattr(args, "stage2_epochs", None)),
        ("stage2_arch", getattr(args, "stage2_arch", None)),
        ("stage2_autoregressive_coeffs", getattr(args, "stage2_autoregressive_coeffs", None)),
    )
    payload = "|".join(f"{key}={value}" for key, value in payload_fields)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    run_id = _resolve_run_id(
        getattr(args, "run_id", None),
        getattr(args, "launch_timestamp", None) or os.environ.get("PROTO_LAUNCH_TIMESTAMP"),
    )
    slurm_job_id = _artifact_safe_token(
        os.environ.get("PROTO_RUN_SLURM_JOB_ID") or os.environ.get("SLURM_JOB_ID")
    )
    launch_timestamp = _artifact_safe_token(
        getattr(args, "launch_timestamp", None) or os.environ.get("PROTO_LAUNCH_TIMESTAMP")
    )
    parts = []
    if run_id:
        parts.append(f"r{run_id}")
    else:
        if slurm_job_id:
            parts.append(f"j{slurm_job_id}")
        if launch_timestamp:
            parts.append(f"t{launch_timestamp}")
    parts.append(f"h{digest}")
    return "__".join(parts)


def _canonical_patch_reconstruction(
    patch_reconstruction: Optional[str],
    *,
    patch_size: Optional[int],
    patch_stride: Optional[int],
) -> str:
    mode = "center_crop" if patch_reconstruction is None else str(patch_reconstruction).strip().lower()
    if mode not in ("center_crop", "hann", "tile"):
        raise ValueError("patch_reconstruction must be 'center_crop', 'hann', or 'tile'")
    if patch_size is not None and patch_stride is not None:
        try:
            patch_size_i = int(patch_size)
            patch_stride_i = int(patch_stride)
        except (TypeError, ValueError):
            patch_size_i = None
            patch_stride_i = None
        if patch_size_i is not None and patch_stride_i is not None and patch_stride_i == patch_size_i:
            # Non-overlapping patches can be stitched directly without overlap-aware blending.
            return "tile"
    return mode


def _tagged_artifact_path(path: Path, artifact_tag: Optional[str]) -> Path:
    path = Path(path).expanduser().resolve()
    if not artifact_tag:
        return path
    suffix = "".join(path.suffixes)
    stem = path.name[:-len(suffix)] if suffix else path.name
    return path.with_name(f"{stem}__{artifact_tag}{suffix}")


def _copy_artifact_to_tagged_path(path: Path, artifact_tag: Optional[str]) -> Path:
    path = Path(path).expanduser().resolve()
    tagged_path = _tagged_artifact_path(path, artifact_tag)
    if tagged_path != path:
        shutil.copy2(path, tagged_path)
    return tagged_path


def _save_module_checkpoint_with_tag(
    module: nn.Module,
    checkpoint_path: Path,
    artifact_tag: Optional[str],
) -> Path:
    _save_module_checkpoint(module, checkpoint_path)
    return _copy_artifact_to_tagged_path(checkpoint_path, artifact_tag)


def _distributed_mean(value: torch.Tensor) -> torch.Tensor:
    if not _is_distributed():
        return value.detach()
    reduced = value.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= dist.get_world_size()
    return reduced


def _unwrap_module(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, DDP) else module


def _project_real_valued_stage2_coeffs(ae: nn.Module, coeffs: torch.Tensor) -> torch.Tensor:
    """
    Project real-valued sparse coefficients into the bounded stage-2 model space.

    Scratch stage-1 runs may emit deterministic sparse coefficients with a much
    wider dynamic range than the stage-2 prior head and sampling path assume.
    Keep cached teacher-forcing inputs, regression targets, and preview decode
    coeffs on the same bounded manifold.
    """
    ae_module = _unwrap_module(ae)
    coeffs = _nan_to_num_tensor(ae_module.clamp_sparse_coeffs(coeffs))
    bottleneck = getattr(ae_module, "bottleneck", None)
    if bottleneck is None or bool(getattr(bottleneck, "quantize_sparse_coeffs", False)):
        return coeffs
    coef_max = getattr(bottleneck, "coef_max", None)
    if coef_max is None:
        return coeffs
    try:
        coef_max_value = float(coef_max)
    except (TypeError, ValueError):
        return coeffs
    if not math.isfinite(coef_max_value) or coef_max_value <= 0.0:
        return coeffs
    return coeffs.clamp(-coef_max_value, coef_max_value)


def _clamp_generated_sparse_coeffs_for_decode(ae: nn.Module, coeffs: torch.Tensor) -> torch.Tensor:
    return _project_real_valued_stage2_coeffs(ae, coeffs)


def build_stage2_model(
    bottleneck,
    *,
    stage2_arch: str,
    H: int,
    W: int,
    D: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    d_ff: int,
    dropout: float,
    n_global_spatial_tokens: int,
    real_valued_coeffs: bool,
    coeff_max_fallback: float,
    autoregressive_coeffs: bool,
) -> nn.Module:
    stage2_arch = str(stage2_arch).strip().lower()
    if (not real_valued_coeffs) and (not autoregressive_coeffs):
        raise ValueError(
            "Quantized stage-2 priors require stage2_autoregressive_coeffs=true so the "
            "shared atom/coeff token stream stays interleaved."
        )
    if stage2_arch == "spatial_depth":
        model = SpatialDepthPrior(
            build_spatial_depth_prior_config(
                bottleneck,
                H=H,
                W=W,
                D=D,
                d_model=d_model,
                n_heads=n_heads,
                n_spatial_layers=n_layers,
                n_depth_layers=max(1, n_layers // 2),
                d_ff=d_ff,
                dropout=dropout,
                n_global_spatial_tokens=n_global_spatial_tokens,
                real_valued_coeffs=real_valued_coeffs,
                coeff_max_fallback=coeff_max_fallback,
                autoregressive_coeffs=autoregressive_coeffs,
            )
        )
    elif stage2_arch == "mingpt":
        if real_valued_coeffs:
            raise ValueError("stage2_arch='mingpt' currently supports quantized sparse coefficients only")
        if not autoregressive_coeffs:
            raise ValueError("stage2_arch='mingpt' requires stage2_autoregressive_coeffs=true")
        if int(n_global_spatial_tokens) != 0:
            raise ValueError("stage2_arch='mingpt' does not support tf_global_tokens; use 0")
        model = MinGPTQuantizedPrior(
            build_mingpt_quantized_prior_config(
                bottleneck,
                H=H,
                W=W,
                D=D,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                d_ff=d_ff,
                dropout=dropout,
            )
        )
    else:
        raise ValueError(f"Unsupported stage2_arch: {stage2_arch!r}")
    setattr(model, "stage2_arch", stage2_arch)
    return model


def _describe_stage2_model(
    model: nn.Module,
    *,
    stage2_arch: str,
    tf_layers: int,
    tf_global_tokens: int,
) -> str:
    stage2_arch = str(stage2_arch).strip().lower()
    if stage2_arch == "spatial_depth":
        if not getattr(model, "real_valued_coeffs", False):
            coeff_mode = "quantized shared atom/coeff vocab"
            rollout_mode = "interleaved token rollout"
        else:
            coeff_mode = (
                "real-valued coeffs, Gaussian head"
                if getattr(model, "gaussian_coeffs", False)
                else "real-valued coeffs"
            )
            rollout_mode = (
                "autoregressive coeff rollout"
                if getattr(model, "autoregressive_coeffs", True)
                else "support-only coeff conditioning"
            )
        return (
            "[Stage2] using RQ spatial-depth prior "
            f"({coeff_mode}, {rollout_mode}, spatial_layers={tf_layers}, "
            f"depth_layers={max(1, tf_layers // 2)}, global_tokens={tf_global_tokens})"
        )
    return (
        "[Stage2] using quantized minGPT prior "
        f"(shared_vocab={getattr(model, 'content_vocab_size', 'unknown')}, "
        f"layers={tf_layers}, causal_token_stream=H*W*D)"
    )


def _dictionary_abs_offdiag_cosines(
    dictionary: torch.Tensor,
    eps: float = 1e-12,
    *,
    detach: bool = True,
) -> Tuple[torch.Tensor, int]:
    atoms = dictionary.detach() if detach else dictionary
    atoms = F.normalize(atoms, p=2, dim=0, eps=eps)
    gram = atoms.t() @ atoms
    if gram.size(0) <= 1:
        return torch.zeros_like(gram), 0
    gram = gram - torch.diag_embed(torch.diagonal(gram))
    return gram.abs(), int(gram.size(0) * (gram.size(0) - 1))


def _dictionary_coherence_stats(
    dictionary: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    abs_offdiag, pair_count = _dictionary_abs_offdiag_cosines(dictionary, eps=eps, detach=True)
    if pair_count <= 0:
        zero = torch.zeros((), device=dictionary.device, dtype=dictionary.dtype)
        return zero, zero, zero
    mean_abs = abs_offdiag.sum() / float(pair_count)
    rms_abs = torch.sqrt(abs_offdiag.square().sum() / float(pair_count))
    return abs_offdiag.max(), mean_abs, rms_abs


def _dictionary_coherence(dictionary: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return the maximum absolute off-diagonal atom cosine similarity."""
    coherence_max, _, _ = _dictionary_coherence_stats(dictionary, eps=eps)
    return coherence_max


def _dictionary_coherence_penalty(
    dictionary: torch.Tensor,
    margin: float = 0.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    abs_offdiag, pair_count = _dictionary_abs_offdiag_cosines(dictionary, eps=eps, detach=False)
    if pair_count <= 0:
        return torch.zeros((), device=dictionary.device, dtype=dictionary.dtype)
    excess = F.relu(abs_offdiag - float(max(0.0, margin)))
    return excess.square().sum() / float(pair_count)


def _normalize_dictionary_in_place(dictionary: torch.Tensor, eps: float = 1e-12) -> None:
    with torch.no_grad():
        safe_dictionary = _nan_to_num_tensor(dictionary.detach())
        dictionary.copy_(F.normalize(safe_dictionary, p=2, dim=0, eps=eps))


def _project_dictionary_gradient_in_place(dictionary: torch.Tensor, eps: float = 1e-12) -> None:
    """Project dictionary gradients onto the unit-sphere tangent space per atom."""
    if dictionary.grad is None:
        return
    with torch.no_grad():
        atoms = F.normalize(_nan_to_num_tensor(dictionary.detach()), p=2, dim=0, eps=eps)
        grad = dictionary.grad
        grad.copy_(_nan_to_num_tensor(grad))
        radial = (atoms * grad).sum(dim=0, keepdim=True)
        grad.sub_(atoms * radial)
        grad.copy_(_nan_to_num_tensor(grad))


def _batched_omp_with_support(
    X: torch.Tensor,
    D: torch.Tensor,
    sparsity_level: int,
    diag_eps: float = 1e-4,
    cholesky_eps: float = 1e-6,
    return_history: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """Numerically damped batched OMP that returns support indices and ordered coefficients."""
    if X.ndim != 2 or D.ndim != 2:
        raise ValueError(f"Expected 2D tensors, got X={tuple(X.shape)} D={tuple(D.shape)}")
    if sparsity_level > int(D.size(1)):
        raise ValueError(
            f"sparsity_level ({int(sparsity_level)}) must be <= num_atoms ({int(D.size(1))})"
        )

    X = _nan_to_num_tensor(X)
    D = _nan_to_num_tensor(D)

    _, batch_size = X.size()
    device = D.device
    dtype = D.dtype
    batch_idx = torch.arange(batch_size, device=device)

    Dt = D.t()
    G = Dt.mm(D)
    if diag_eps > 0.0:
        G = G + float(diag_eps) * torch.eye(G.size(0), device=device, dtype=dtype)
    h_bar = Dt.mm(X).t()
    h = h_bar.clone()
    x = torch.zeros_like(h_bar)
    L = torch.empty(batch_size, 0, 0, device=device, dtype=dtype)
    I = torch.empty(batch_size, 0, device=device, dtype=torch.long)
    I_logic = torch.zeros_like(h_bar, dtype=torch.bool)
    support_history = [] if return_history else None
    coeff_history = [] if return_history else None

    def _update_logical(logical: torch.Tensor, to_add: torch.Tensor) -> None:
        logical[batch_idx, to_add] = True

    while I.size(1) < int(sparsity_level):
        scores = h.abs().masked_fill(I_logic, -1.0)
        index = scores.argmax(dim=1)
        _update_logical(I_logic, index)

        selected = int(I.size(1))
        diag_g = G[index, index].view(batch_size, 1, 1)
        if selected == 0:
            L = torch.sqrt(torch.clamp(diag_g, min=cholesky_eps))
        else:
            expanded_batch_idx = batch_idx.unsqueeze(0).expand(selected, batch_size).t()
            G_stack = G[I[batch_idx, :], index[expanded_batch_idx]].view(batch_size, selected, 1)
            w = torch.linalg.solve_triangular(L, G_stack, upper=False)
            w_t = w.transpose(1, 2)
            w_corner = torch.sqrt(
                torch.clamp(diag_g - (w_t ** 2).sum(dim=2, keepdim=True), min=cholesky_eps)
            )
            k_zeros = torch.zeros(batch_size, selected, 1, device=device, dtype=dtype)
            L = torch.cat(
                (
                    torch.cat((L, k_zeros), dim=2),
                    torch.cat((w_t, w_corner), dim=2),
                ),
                dim=1,
            )

        I = torch.cat([I, index.unsqueeze(1)], dim=1)
        support_size = int(I.size(1))
        expanded_batch_idx = batch_idx.unsqueeze(0).expand(support_size, batch_size).t()
        h_stack = h_bar[expanded_batch_idx, I].view(batch_size, support_size, 1)
        try:
            x_stack = torch.cholesky_solve(h_stack, L)
        except RuntimeError:
            gram_support = torch.bmm(L, L.transpose(1, 2))
            reg_eye = torch.eye(support_size, device=device, dtype=dtype).expand(batch_size, -1, -1)
            x_stack = torch.linalg.solve(gram_support + cholesky_eps * reg_eye, h_stack)
        x_stack = torch.nan_to_num(x_stack, nan=0.0, posinf=0.0, neginf=0.0)
        x[batch_idx.unsqueeze(1), I] = x_stack.squeeze(-1)
        coeffs_ordered = x[batch_idx.unsqueeze(1), I]
        coeffs_ordered = torch.nan_to_num(coeffs_ordered, nan=0.0, posinf=0.0, neginf=0.0)
        if return_history:
            padded_support = torch.zeros(batch_size, int(sparsity_level), device=device, dtype=torch.long)
            padded_coeffs = torch.zeros(batch_size, int(sparsity_level), device=device, dtype=dtype)
            padded_support[:, :support_size] = I
            padded_coeffs[:, :support_size] = coeffs_ordered
            support_history.append(padded_support)
            coeff_history.append(padded_coeffs)

        beta = (
            coeffs_ordered
            .unsqueeze(1)
            .bmm(G[I[batch_idx], :])
            .squeeze(1)
        )
        h = torch.nan_to_num(h_bar - beta, nan=0.0, posinf=0.0, neginf=0.0)

    coeffs_ordered = x[batch_idx.unsqueeze(1), I]
    coeffs_ordered = torch.nan_to_num(coeffs_ordered, nan=0.0, posinf=0.0, neginf=0.0)
    if not return_history:
        return I, coeffs_ordered
    return (
        I,
        coeffs_ordered,
        torch.stack(support_history, dim=1),
        torch.stack(coeff_history, dim=1),
    )


_RFID_MODEL = None
_RFID_MODEL_DEVICE = None
_RFID_METRIC = None
_RFID_METRIC_DEVICE = None
_WANDB_LOG_STEP = 0
_WANDB_DISABLE_REASON = None


_FLAT_IMAGE_DATASET_INDEX_CACHE = {}


def _flat_image_index_cache_file(root: Path) -> Path:
    cache_base = Path(os.environ.get("LASER_DATASET_CACHE_DIR", f"/scratch/{os.environ.get('USER', 'unknown')}/.cache/laser_dataset_index")).expanduser()
    root_key = hashlib.sha1(str(root).encode("utf-8")).hexdigest()
    return cache_base / f"{root_key}.txt"


def _packed_celeba_file(root: Path, image_size: int) -> Path:
    return root / f"celeba_{int(image_size)}x{int(image_size)}_rgb_uint8.npy"


def _scan_image_paths(root: Path):
    image_paths = []
    for dirpath, _, filenames in os.walk(root):
        base = Path(dirpath)
        for name in filenames:
            path = base / name
            if path.suffix.lower() in IMG_EXTENSIONS:
                image_paths.append(path)
    image_paths.sort()
    return image_paths


class FlatImageDataset(Dataset):
    """
    Recursively loads images from a directory tree.
    Returns (image_tensor, dummy_label) to match torchvision dataset API.
    """

    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root}")

        cached = _FLAT_IMAGE_DATASET_INDEX_CACHE.get(self.root)
        if cached is None:
            cache_file = _flat_image_index_cache_file(self.root)
            image_paths = None
            if cache_file.exists():
                try:
                    image_paths = [Path(line.strip()) for line in cache_file.read_text(encoding="utf-8").splitlines() if line.strip()]
                except Exception:
                    image_paths = None
            if image_paths is None:
                image_paths = _scan_image_paths(self.root)
                try:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    cache_file.write_text("\n".join(str(p) for p in image_paths) + "\n", encoding="utf-8")
                except Exception:
                    pass
            _FLAT_IMAGE_DATASET_INDEX_CACHE[self.root] = image_paths
            cached = image_paths
        if not cached:
            raise RuntimeError(f"No images found under: {self.root}")
        self.image_paths = cached

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


class PackedRGBImageDataset(Dataset):
    """Read resized RGB uint8 images from a single NumPy memmap file."""

    def __init__(self, root: str, image_size: int, random_horizontal_flip: bool = False):
        self.root = Path(root)
        self.image_size = int(image_size)
        self.random_horizontal_flip = bool(random_horizontal_flip)
        self.packed_path = _packed_celeba_file(self.root, self.image_size)
        if not self.packed_path.exists():
            raise FileNotFoundError(f"Packed dataset not found: {self.packed_path}")
        self.images = np.load(self.packed_path, mmap_mode="r")
        expected_shape = (self.image_size, self.image_size, 3)
        if self.images.ndim != 4 or tuple(self.images.shape[1:]) != expected_shape:
            raise ValueError(
                f"Packed dataset at {self.packed_path} has shape {tuple(self.images.shape)}; "
                f"expected [N, {self.image_size}, {self.image_size}, 3]"
            )
        if self.images.dtype != np.uint8:
            raise ValueError(
                f"Packed dataset at {self.packed_path} has dtype {self.images.dtype}; expected uint8"
            )

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, idx: int):
        image = torch.from_numpy(np.array(self.images[idx], copy=True)).permute(2, 0, 1).float().div_(255.0)
        if self.random_horizontal_flip and torch.rand((), dtype=torch.float32).item() < 0.5:
            image = torch.flip(image, dims=[2])
        image = image.mul_(2.0).sub_(1.0)
        return image, 0


# -----------------------------
# RQ-VAE style building blocks (borrowed from https://github.com/kakaobrain/rq-vae-transformer)
# -----------------------------

def nonlinearity(x):
    return F.silu(x)


def Normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb=None):
        h = self.norm1(x)
        h = nonlinearity(h)
        h = self.conv1(h)
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        w_ = torch.bmm(q, k) * (int(c) ** -0.5)
        w_ = F.softmax(w_, dim=2)
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_).reshape(b, c, h, w)
        h_ = self.proj_out(h_)
        return x + h_


class Encoder(nn.Module):
    """RQ-VAE encoder with ResNet blocks, optional attention, and progressive downsampling."""
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=False, use_mid_attention=True, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.use_mid_attention = bool(use_mid_attention)

        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out,
                                         temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in) if self.use_mid_attention else nn.Identity()
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=dropout)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels if double_z else z_channels,
                                  kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        temb = None
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    """RQ-VAE decoder with ResNet blocks, optional attention, and progressive upsampling."""
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, use_mid_attention=True, extra_res_blocks=1, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.use_mid_attention = bool(use_mid_attention)
        self.extra_res_blocks = max(0, int(extra_res_blocks))

        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in) if self.use_mid_attention else nn.Identity()
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=dropout)
        self.blocks_per_level = max(1, self.num_res_blocks + self.extra_res_blocks)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.blocks_per_level):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out,
                                         temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        self.last_z_shape = z.shape
        temb = None
        h = self.conv_in(z)
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.blocks_per_level):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        if self.give_pre_end:
            return h
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


# -----------------------------
# Dictionary learning bottleneck (batch OMP) + Option-A tokenization
# -----------------------------

class DictionaryLearningTokenized(nn.Module):
    """
    Dictionary-learning bottleneck with batched Orthogonal Matching Pursuit (OMP) sparse coding.
    Tokenization modes:
    - Quantized-mode: alternating token pairs [atom_id, coeff_bin + num_atoms].
    - Regressor-mode: token = atom_id only, coefficients are modeled with a separate head.

    Outputs, per latent pixel, a token stack of length:
    - 2 * sparsity_level in quantized mode
    - sparsity_level in regressor mode

    Important simplifications (good for a quick test):
      - OMP runs under torch.no_grad() like in LASER: we do NOT backprop through sparse coding.
      - We reconstruct the latent using quantized coefficients, then apply STE so the encoder
        still receives gradients (VQ-VAE style).
    """
    def __init__(
        self,
        num_embeddings: int = 1024,
        embedding_dim: int = 16,
        sparsity_level: int = 8,
        n_bins: int = 16,
        coef_max: float = 3.0,
        quantize_sparse_coeffs: bool = False,
        coef_quantization: str = "uniform",
        coef_mu: float = 0.0,
        commitment_cost: float = 0.25,
        epsilon: float = 1e-10,
        canonicalize_sparse_slots: bool = True,
        variational_coeffs: bool = False,
        variational_coeff_kl_weight: float = 0.0,
        variational_coeff_prior_std: float = 0.25,
        variational_coeff_min_std: float = 0.01,
    ):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.sparsity_level = int(sparsity_level)
        self.n_bins = int(n_bins)
        self.coef_max = float(coef_max)
        self.quantize_sparse_coeffs = bool(quantize_sparse_coeffs)
        self.coef_quantization = str(coef_quantization)
        self.coef_mu = float(coef_mu)
        if self.coef_quantization not in ("uniform", "mu_law"):
            raise ValueError(
                "coef_quantization must be one of {'uniform', 'mu_law'}"
            )
        if self.coef_quantization == "mu_law" and self.coef_mu <= 0.0:
            raise ValueError(f"coef_mu must be > 0, got {self.coef_mu}")
        self.commitment_cost = float(commitment_cost)
        self.epsilon = float(epsilon)
        self.canonicalize_sparse_slots = bool(canonicalize_sparse_slots)
        self.variational_coeffs = bool(variational_coeffs)
        self.variational_coeff_kl_weight = float(variational_coeff_kl_weight)
        self.variational_coeff_prior_std = float(variational_coeff_prior_std)
        self.variational_coeff_min_std = float(variational_coeff_min_std)
        if self.variational_coeff_kl_weight < 0.0:
            raise ValueError(
                f"variational_coeff_kl_weight must be >= 0, got {self.variational_coeff_kl_weight}"
            )
        if self.variational_coeff_prior_std <= 0.0:
            raise ValueError(
                f"variational_coeff_prior_std must be > 0, got {self.variational_coeff_prior_std}"
            )
        if self.variational_coeff_min_std <= 0.0:
            raise ValueError(
                f"variational_coeff_min_std must be > 0, got {self.variational_coeff_min_std}"
            )
        if self.variational_coeff_min_std > self.variational_coeff_prior_std:
            raise ValueError(
                "variational_coeff_min_std cannot exceed variational_coeff_prior_std: "
                f"{self.variational_coeff_min_std} > {self.variational_coeff_prior_std}"
            )
        if self.variational_coeffs and self.quantize_sparse_coeffs:
            raise ValueError("variational_coeffs currently requires quantize_sparse_coeffs=False")

        # Dictionary shape [C, K] (matches LASER)
        self.dictionary = nn.Parameter(torch.randn(self.embedding_dim, self.num_embeddings) * 0.02)
        self._last_coeff_kl_loss = torch.zeros(())
        self._last_weighted_coeff_kl_loss = torch.zeros(())
        self._last_extra_bottleneck_loss = torch.zeros(())
        self._last_coeff_posterior_std = torch.zeros(())
        self._last_coeff_prior_std = torch.tensor(self.variational_coeff_prior_std)
        if self.variational_coeffs:
            hidden_dim = max(32, min(128, self.embedding_dim * 4))
            self.coeff_variational_atom_emb = nn.Embedding(self.num_embeddings, hidden_dim)
            self.coeff_variational_posterior = nn.Sequential(
                nn.Linear(hidden_dim + 1, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2),
            )
        else:
            self.coeff_variational_atom_emb = None
            self.coeff_variational_posterior = None

        # Coefficient bin centers (uniform)
        centers = torch.linspace(-self.coef_max, self.coef_max, steps=self.n_bins)
        self.register_buffer("coef_bin_centers", centers)
        mu_invlog1p = 1.0
        if self.coef_quantization == "mu_law":
            mu_invlog1p = 1.0 / math.log1p(self.coef_mu)
        self.register_buffer(
            "coef_mu_invlog1p",
            torch.tensor(mu_invlog1p),
        )

        # Special tokens (for the transformer)
        if self.quantize_sparse_coeffs:
            self.coeff_token_offset = self.num_embeddings
            self.token_depth = 2 * self.sparsity_level
            self.content_vocab_size = self.num_embeddings + self.n_bins
            self.pad_token_id = self.content_vocab_size
            self.bos_token_id = self.pad_token_id + 1
            self.vocab_size = self.content_vocab_size + 2
        else:
            self.coeff_token_offset = self.num_embeddings
            self.token_depth = self.sparsity_level
            self.content_vocab_size = self.num_embeddings
            self.pad_token_id = self.num_embeddings
            self.bos_token_id = self.num_embeddings + 1
            self.vocab_size = self.num_embeddings + 2

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        variational_prefixes = (
            prefix + "coeff_variational_atom_emb.",
            prefix + "coeff_variational_posterior.",
        )
        if self.variational_coeffs and self.coeff_variational_atom_emb is not None and self.coeff_variational_posterior is not None:
            expected_state = {
                prefix + "coeff_variational_atom_emb.weight": self.coeff_variational_atom_emb.weight.detach().clone(),
            }
            for subkey, value in self.coeff_variational_posterior.state_dict().items():
                expected_state[prefix + "coeff_variational_posterior." + subkey] = value.detach().clone()
            for key, expected in expected_state.items():
                loaded = state_dict.get(key, None)
                if loaded is None or tuple(loaded.shape) != tuple(expected.shape):
                    state_dict[key] = expected
        else:
            for key in list(state_dict.keys()):
                if key.startswith(variational_prefixes):
                    state_dict.pop(key, None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _normalize_dict(self) -> torch.Tensor:
        return F.normalize(self.dictionary, p=2, dim=0, eps=self.epsilon)

    def _quantize_coeff(self, coeff: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize coefficients to bins; return (bin_idx, bin_center_value)."""
        if self.coef_quantization == "uniform":
            c = coeff.clamp(-self.coef_max, self.coef_max)
            scaled = (c + self.coef_max) / (2 * self.coef_max)  # [0,1]
            bin_f = scaled * (self.n_bins - 1)
            bin_idx = torch.round(bin_f).to(torch.long).clamp(0, self.n_bins - 1)
            coeff_q = self.coef_bin_centers[bin_idx]
            return bin_idx, coeff_q

        # μ-law companding: finer resolution near zero for sparse code magnitudes.
        c = coeff.clamp(-self.coef_max, self.coef_max) / self.coef_max
        c_abs = c.abs()
        encoded = torch.sign(c) * torch.log1p(c_abs * self.coef_mu) * self.coef_mu_invlog1p
        scaled = (encoded + 1.0) * ((self.n_bins - 1) / 2.0)
        bin_idx = torch.round(scaled).to(torch.long).clamp(0, self.n_bins - 1)
        decoded = self._dequantize_coeff(bin_idx)
        return bin_idx, decoded

    def _dequantize_coeff(self, bin_idx: torch.Tensor) -> torch.Tensor:
        """Decode bin indices back to quantized coefficients."""
        if self.coef_quantization == "uniform":
            return self.coef_bin_centers[bin_idx]

        # Inverse μ-law companding.
        z = bin_idx.float() * (2.0 / (self.n_bins - 1)) - 1.0
        z_abs = z.abs()
        decoded_norm = torch.sign(z) * (torch.expm1(z_abs / self.coef_mu_invlog1p) / self.coef_mu)
        return decoded_norm * self.coef_max

    def _pack_quantized_tokens(self, support: torch.Tensor, bin_idx: torch.Tensor) -> torch.Tensor:
        """Interleave atom tokens and coefficient-bin tokens along the token depth axis."""
        if support.shape != bin_idx.shape:
            raise ValueError(f"support and bin_idx shape mismatch: {support.shape} vs {bin_idx.shape}")
        if support.size(-1) != self.sparsity_level:
            raise ValueError(f"Expected sparse depth {self.sparsity_level}, got {support.size(-1)}")

        tokens = torch.empty(
            *support.shape[:-1],
            self.token_depth,
            device=support.device,
            dtype=torch.long,
        )
        tokens[..., 0::2] = support.to(torch.long)
        tokens[..., 1::2] = bin_idx.to(torch.long) + self.coeff_token_offset
        return tokens

    def _unpack_quantized_tokens(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode alternating [atom, coeff_bin] tokens back to atom ids and coefficients."""
        if tokens.size(-1) != self.token_depth:
            raise ValueError(f"Expected token depth {self.token_depth}, got {tokens.size(-1)}")

        atom_tokens = tokens[..., 0::2].to(torch.long)
        coeff_tokens = tokens[..., 1::2].to(torch.long)

        atom_invalid = (atom_tokens < 0) | (atom_tokens >= self.num_embeddings)
        coeff_bin = coeff_tokens - self.coeff_token_offset
        coeff_invalid = (coeff_bin < 0) | (coeff_bin >= self.n_bins)
        invalid = atom_invalid | coeff_invalid

        atom_ids = atom_tokens.clamp(0, self.num_embeddings - 1)
        coeff_bin = coeff_bin.clamp(0, self.n_bins - 1)
        coeffs = self._dequantize_coeff(coeff_bin)

        atom_ids = atom_ids.masked_fill(invalid, 0)
        coeffs = coeffs.masked_fill(invalid, 0.0)
        return atom_ids, coeffs

    def _encode_sparse_codes(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run OMP and return support atom ids and continuous coefficients."""
        B, C, H, W = z_e.shape
        n_signals = B * H * W
        dictionary = self._normalize_dict()
        signals = z_e.permute(0, 2, 3, 1).contiguous().view(-1, C).t().to(dictionary.dtype)
        with torch.no_grad():
            support, coeffs = self.batch_omp_with_support(signals, dictionary)
        if support.ndim != 2 or coeffs.ndim != 2:
            raise RuntimeError(
                f"OMP returned invalid rank: support={tuple(support.shape)} coeffs={tuple(coeffs.shape)}"
            )
        if support.size(0) != n_signals or coeffs.size(0) != n_signals:
            raise RuntimeError(
                f"OMP returned invalid batch size: expected {n_signals}, "
                f"got support={support.size(0)} coeffs={coeffs.size(0)}"
            )
        # Defensive shape guard: keep a fixed D stack even if OMP exits short due to numerical edge cases.
        if support.size(1) != self.sparsity_level or coeffs.size(1) != self.sparsity_level:
            cur_d = min(support.size(1), coeffs.size(1))
            if cur_d > 0:
                support = support[:, :cur_d]
                coeffs = coeffs[:, :cur_d]
            else:
                support = torch.zeros((n_signals, 0), device=support.device, dtype=support.dtype)
                coeffs = torch.zeros((n_signals, 0), device=coeffs.device, dtype=coeffs.dtype)
            if cur_d < self.sparsity_level:
                pad = self.sparsity_level - cur_d
                support_pad = torch.zeros((n_signals, pad), device=support.device, dtype=support.dtype)
                coeffs_pad = torch.zeros((n_signals, pad), device=coeffs.device, dtype=coeffs.dtype)
                support = torch.cat([support, support_pad], dim=1)
                coeffs = torch.cat([coeffs, coeffs_pad], dim=1)
        if self.canonicalize_sparse_slots:
            # Canonicalize sparse slots so stage-2 does not need to model arbitrary OMP selection order.
            order = coeffs.abs().argsort(dim=1, descending=True)
            support = support.gather(1, order)
            coeffs = coeffs.gather(1, order)
        return (
            support.view(B, H, W, self.sparsity_level),
            coeffs.view(B, H, W, self.sparsity_level),
        )

    def clamp_sparse_coeffs(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Project coefficients onto the stage-1 decoder manifold."""
        coeffs = _nan_to_num_tensor(coeffs)
        if self.quantize_sparse_coeffs:
            return coeffs.clamp(-self.coef_max, self.coef_max)
        return coeffs

    def _coeff_posterior_stats(
        self,
        support: torch.Tensor,
        coeffs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.variational_coeffs:
            raise RuntimeError("_coeff_posterior_stats requires variational_coeffs=True")
        if self.coeff_variational_atom_emb is None or self.coeff_variational_posterior is None:
            raise RuntimeError("variational coefficient modules were not initialized")
        if support.shape != coeffs.shape:
            raise ValueError(f"support and coeffs shape mismatch: {support.shape} vs {coeffs.shape}")

        support_clamped = support.to(torch.long).clamp(0, self.num_embeddings - 1)
        coeffs_base = self.clamp_sparse_coeffs(coeffs.to(torch.float32))
        atom_emb = self.coeff_variational_atom_emb(support_clamped)
        posterior_in = torch.cat([atom_emb, coeffs_base.unsqueeze(-1)], dim=-1)
        posterior_raw = self.coeff_variational_posterior(posterior_in)

        mean_offset = self.variational_coeff_prior_std * torch.tanh(posterior_raw[..., 0])
        posterior_mu = self.clamp_sparse_coeffs(coeffs_base + mean_offset)

        std_range = max(self.variational_coeff_prior_std - self.variational_coeff_min_std, 0.0)
        posterior_std = self.variational_coeff_min_std + std_range * torch.sigmoid(posterior_raw[..., 1])
        posterior_logvar = 2.0 * torch.log(posterior_std.clamp_min(1e-6))
        return posterior_mu, posterior_logvar

    def project_sparse_coeffs(self, support: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        coeffs_clamped = self.clamp_sparse_coeffs(coeffs)
        if not self.variational_coeffs:
            return coeffs_clamped
        coeff_mu, _ = self._coeff_posterior_stats(support, coeffs_clamped)
        return coeff_mu

    def _reconstruct_sparse(
        self, support: torch.Tensor, coeffs: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct latent map from atom ids + coefficients."""
        if support.shape != coeffs.shape:
            raise ValueError(
                f"support and coeffs shape mismatch: {support.shape} vs {coeffs.shape}"
            )

        B, H, W, D = support.shape
        if D != self.sparsity_level:
            raise ValueError(f"Expected D={self.sparsity_level}, got {D}")

        dictionary = self._normalize_dict().t()  # [num_embeddings, C]
        support = support.to(torch.long).clamp(0, self.num_embeddings - 1)
        coeffs = self.clamp_sparse_coeffs(coeffs.to(dictionary.dtype))
        support_flat = support.reshape(-1, D)
        coeffs_flat = coeffs.reshape(-1, D)
        atoms = dictionary[support_flat]  # [B*H*W, D, C]
        recon_flat = (atoms * coeffs_flat.unsqueeze(-1)).sum(dim=1)  # [N, C]
        return recon_flat.view(B, H, W, self.embedding_dim).permute(0, 3, 1, 2).contiguous()

    def batch_omp_with_support(self, X: torch.Tensor, D: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched OMP adapted from LASER's DictionaryLearning.batch_omp.
        Runs exactly sparsity_level steps (no early-stop) so stack depth is fixed.

        Args:
            X: [M, B] signals
            D: [M, N] normalized dictionary
        Returns:
            support: [B, K] indices in selection order (K = sparsity_level)
            coeffs:  [B, K] coefficients aligned with support (same order)
        """
        return _batched_omp_with_support(
            X=X,
            D=D,
            sparsity_level=self.sparsity_level,
        )

    def batch_omp_with_trajectory(
        self,
        X: torch.Tensor,
        D: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batched OMP with exact per-iteration refit history.

        Returns:
            support: [B, K] final support indices in greedy selection order.
            coeffs: [B, K] final coefficients aligned with support.
            support_history: [B, K, K] padded support prefixes after each OMP step.
            coeff_history: [B, K, K] padded refit coefficients after each OMP step.
        """
        return _batched_omp_with_support(
            X=X,
            D=D,
            sparsity_level=self.sparsity_level,
            return_history=True,
        )

    def _encode_sparse_codes_with_trajectory(
        self,
        z_e: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run OMP and return the exact greedy refit trajectory.

        The returned history tensors are padded to the full sparsity level so the
        k-th OMP state can be decoded directly with `_reconstruct_sparse`.
        """
        if self.canonicalize_sparse_slots:
            raise RuntimeError(
                "Exact OMP trajectories require canonicalize_sparse_slots=False because canonicalization destroys "
                "the greedy step order."
            )
        B, C, H, W = z_e.shape
        n_signals = B * H * W
        signals = z_e.permute(0, 2, 3, 1).contiguous().view(-1, C).t()
        dictionary = self._normalize_dict()
        with torch.no_grad():
            support, coeffs, support_hist, coeff_hist = self.batch_omp_with_trajectory(signals, dictionary)
        if support.ndim != 2 or coeffs.ndim != 2 or support_hist.ndim != 3 or coeff_hist.ndim != 3:
            raise RuntimeError(
                "OMP trajectory returned invalid ranks: "
                f"support={tuple(support.shape)} coeffs={tuple(coeffs.shape)} "
                f"support_hist={tuple(support_hist.shape)} coeff_hist={tuple(coeff_hist.shape)}"
            )
        if support.size(0) != n_signals or coeffs.size(0) != n_signals:
            raise RuntimeError(
                f"OMP trajectory returned invalid batch size: expected {n_signals}, "
                f"got support={support.size(0)} coeffs={coeffs.size(0)}"
            )
        expected_shape = (n_signals, self.sparsity_level, self.sparsity_level)
        if tuple(support_hist.shape) != expected_shape or tuple(coeff_hist.shape) != expected_shape:
            raise RuntimeError(
                f"OMP trajectory returned invalid history shapes: expected {expected_shape}, "
                f"got support_hist={tuple(support_hist.shape)} coeff_hist={tuple(coeff_hist.shape)}"
            )
        return (
            support.view(B, H, W, self.sparsity_level),
            coeffs.view(B, H, W, self.sparsity_level),
            support_hist.view(B, H, W, self.sparsity_level, self.sparsity_level),
            coeff_hist.view(B, H, W, self.sparsity_level, self.sparsity_level),
        )

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_e: [B, C, H, W]
        Returns:
            z_q_ste: [B, C, H, W]
            loss: scalar bottleneck loss
            tokens: [B, H, W, token_depth]
        """
        if z_e.dim() != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(z_e.shape)}")
        B, C, H, W = z_e.shape
        if C != self.embedding_dim:
            raise ValueError(f"Expected channel dim {self.embedding_dim}, got {C}")

        z_e = _nan_to_num_tensor(z_e)
        support, coeffs = self._encode_sparse_codes(z_e)
        support_flat = support.view(-1, self.sparsity_level)
        coeffs_flat = self.clamp_sparse_coeffs(coeffs.view(-1, self.sparsity_level))

        if self.quantize_sparse_coeffs:
            # Quantize coefficients and interleave atom + coefficient-bin tokens.
            bin_idx, coeff_q = self._quantize_coeff(coeffs_flat)  # both [Nsig, D]
            tokens = self._pack_quantized_tokens(
                support_flat.view(B, H, W, self.sparsity_level),
                bin_idx.view(B, H, W, self.sparsity_level),
            )
            coeffs_for_recon = coeff_q
        else:
            tokens = support.view(B, H, W, self.sparsity_level).long()
            coeffs_for_recon = self.clamp_sparse_coeffs(coeffs_flat)

        coeff_kl_loss = z_e.new_zeros(())
        weighted_coeff_kl_loss = z_e.new_zeros(())
        if (not self.quantize_sparse_coeffs) and self.variational_coeffs:
            coeffs_base = coeffs_for_recon.reshape(B, H, W, self.sparsity_level)
            coeff_mu, coeff_logvar = self._coeff_posterior_stats(support, coeffs_base)
            if self.training:
                coeff_eps = torch.randn_like(coeff_mu)
                coeff_std = (0.5 * coeff_logvar).exp()
                coeffs_for_recon = self.clamp_sparse_coeffs(coeff_mu + coeff_std * coeff_eps)
            else:
                coeffs_for_recon = coeff_mu
            coeff_kl_loss = _gaussian_kl_to_fixed_mean(
                coeff_mu,
                coeff_logvar,
                coeffs_base,
                target_std=self.variational_coeff_prior_std,
            )
            weighted_coeff_kl_loss = float(self.variational_coeff_kl_weight) * coeff_kl_loss
            self._last_coeff_posterior_std = (0.5 * coeff_logvar).exp().mean().detach()
            self._last_coeff_prior_std = torch.as_tensor(
                self.variational_coeff_prior_std,
                device=z_e.device,
                dtype=z_e.dtype,
            )
        else:
            coeffs_for_recon = coeffs_for_recon.reshape(B, H, W, self.sparsity_level)
            self._last_coeff_posterior_std = z_e.new_zeros(())
            self._last_coeff_prior_std = torch.as_tensor(
                self.variational_coeff_prior_std,
                device=z_e.device,
                dtype=z_e.dtype,
            )

        z_q = _nan_to_num_tensor(self._reconstruct_sparse(support, coeffs_for_recon))

        # Bottleneck loss (LASER-style)
        dl_latent_loss = F.mse_loss(z_q, z_e.detach())
        e_latent_loss = F.mse_loss(z_q.detach(), z_e)
        loss = dl_latent_loss + self.commitment_cost * e_latent_loss + weighted_coeff_kl_loss
        self._last_dl_latent_loss = dl_latent_loss
        self._last_e_latent_loss = e_latent_loss
        self._last_coeff_kl_loss = coeff_kl_loss
        self._last_weighted_coeff_kl_loss = weighted_coeff_kl_loss
        self._last_extra_bottleneck_loss = weighted_coeff_kl_loss
        self._last_bottleneck_loss = loss

        # Straight-through estimator to encoder
        z_q_ste = z_e + (z_q - z_e).detach()
        return z_q_ste, loss, tokens

    @torch.no_grad()
    def tokens_to_latent(self, tokens: torch.Tensor, coeffs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode tokens back to a latent map.
        Args:
            tokens: [B, H, W, token_depth]
            coeffs: [B, H, W, D] (only used in non-quantized mode)
        Returns:
            z_q: [B, C, H, W]
        """
        if tokens.dim() != 4:
            raise ValueError(f"Expected [B,H,W,D], got {tuple(tokens.shape)}")
        B, H, W, D = tokens.shape
        if D != self.token_depth:
            raise ValueError(f"Expected token depth {self.token_depth}, got {D}")

        if self.quantize_sparse_coeffs:
            atom_ids, coeff_q = self._unpack_quantized_tokens(tokens)
            return self._reconstruct_sparse(atom_ids, coeff_q)

        if coeffs is None:
            raise ValueError("coeffs must be provided when quantize_sparse_coeffs=False")

        coeffs_clamped = self.clamp_sparse_coeffs(coeffs.to(self._normalize_dict().dtype))
        return self._reconstruct_sparse(tokens.to(torch.long), coeffs_clamped)


SparseBottleneck = DictionaryLearningTokenized


# -----------------------------
# Patch-based Dictionary Learning bottleneck
# -----------------------------

class PatchDictionaryLearningTokenized(nn.Module):
    """
    Patch-based dictionary learning bottleneck.

    Extracts overlapping patches from the latent feature map using F.unfold,
    runs batched OMP on each patch vector, then reconstructs the latent via
    one of three stitching strategies:

      "center_crop" (default) — take only the center patch_stride×patch_stride
          region of each reconstructed patch and tile non-overlappingly.
          No averaging; each output pixel comes from exactly one patch center.

      "hann" — weighted overlap-add using a 2D Hann window. The window
          up-weights the patch center and fades to zero at edges. With 50%%
          overlap (patch_stride = patch_size // 2) this satisfies the COLA
          condition so the weight map is constant and there is no blurring
          on the exact signal; for OMP-approximated patches it blends
          smoothly rather than averaging equally.

      "tile" — direct patch tiling with no crop or overlap weighting. This is
          the natural reconstruction for non-overlapping patches
          (patch_stride == patch_size), and overlapping requests are
          automatically normalized to this mode when patches do not overlap.

    All modes pad by (patch_size - patch_stride) // 2 before unfolding so
    that the output covers the full H × W spatial extent.

    The dictionary has shape [patch_dim, num_embeddings] where
        patch_dim = patch_size * patch_size * embedding_dim.

    Token output shape: [B, nph, npw, token_depth]  (nph = H // patch_stride).
    """

    def __init__(
        self,
        num_embeddings: int = 1024,
        embedding_dim: int = 16,
        patch_size: int = 8,
        patch_stride: int = 4,
        sparsity_level: int = 8,
        n_bins: int = 16,
        coef_max: float = 3.0,
        quantize_sparse_coeffs: bool = False,
        coef_quantization: str = "uniform",
        coef_mu: float = 0.0,
        commitment_cost: float = 0.25,
        epsilon: float = 1e-10,
        patch_reconstruction: str = "center_crop",
        canonicalize_sparse_slots: bool = True,
        variational_coeffs: bool = False,
        variational_coeff_kl_weight: float = 0.0,
        variational_coeff_prior_std: float = 0.25,
        variational_coeff_min_std: float = 0.01,
    ):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)
        self.patch_dim = self.patch_size * self.patch_size * self.embedding_dim
        self.sparsity_level = int(sparsity_level)
        self.n_bins = int(n_bins)
        self.coef_max = float(coef_max)
        self.quantize_sparse_coeffs = bool(quantize_sparse_coeffs)
        self.coef_quantization = str(coef_quantization)
        self.coef_mu = float(coef_mu)
        if self.coef_quantization not in ("uniform", "mu_law"):
            raise ValueError("coef_quantization must be one of {'uniform', 'mu_law'}")
        if self.coef_quantization == "mu_law" and self.coef_mu <= 0.0:
            raise ValueError(f"coef_mu must be > 0, got {self.coef_mu}")
        self.commitment_cost = float(commitment_cost)
        self.epsilon = float(epsilon)
        self.canonicalize_sparse_slots = bool(canonicalize_sparse_slots)
        self.variational_coeffs = bool(variational_coeffs)
        self.variational_coeff_kl_weight = float(variational_coeff_kl_weight)
        self.variational_coeff_prior_std = float(variational_coeff_prior_std)
        self.variational_coeff_min_std = float(variational_coeff_min_std)
        if self.variational_coeff_kl_weight < 0.0:
            raise ValueError(
                f"variational_coeff_kl_weight must be >= 0, got {self.variational_coeff_kl_weight}"
            )
        if self.variational_coeff_prior_std <= 0.0:
            raise ValueError(
                f"variational_coeff_prior_std must be > 0, got {self.variational_coeff_prior_std}"
            )
        if self.variational_coeff_min_std <= 0.0:
            raise ValueError(
                f"variational_coeff_min_std must be > 0, got {self.variational_coeff_min_std}"
            )
        if self.variational_coeff_min_std > self.variational_coeff_prior_std:
            raise ValueError(
                "variational_coeff_min_std cannot exceed variational_coeff_prior_std: "
                f"{self.variational_coeff_min_std} > {self.variational_coeff_prior_std}"
            )
        self.patch_reconstruction = _canonical_patch_reconstruction(
            patch_reconstruction,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
        )
        if self.variational_coeffs and self.quantize_sparse_coeffs:
            raise ValueError("variational_coeffs currently requires quantize_sparse_coeffs=False")

        # Dictionary shape: [patch_dim, num_embeddings]
        self.dictionary = nn.Parameter(
            torch.randn(self.patch_dim, self.num_embeddings) * 0.02
        )
        self._last_coeff_kl_loss = torch.zeros(())
        self._last_weighted_coeff_kl_loss = torch.zeros(())
        self._last_extra_bottleneck_loss = torch.zeros(())
        self._last_coeff_posterior_std = torch.zeros(())
        self._last_coeff_prior_std = torch.tensor(self.variational_coeff_prior_std)
        if self.variational_coeffs:
            hidden_dim = max(64, min(256, self.embedding_dim * self.patch_size))
            self.coeff_variational_atom_emb = nn.Embedding(self.num_embeddings, hidden_dim)
            self.coeff_variational_posterior = nn.Sequential(
                nn.Linear(hidden_dim + 1, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2),
            )
        else:
            self.coeff_variational_atom_emb = None
            self.coeff_variational_posterior = None

        centers = torch.linspace(-self.coef_max, self.coef_max, steps=self.n_bins)
        self.register_buffer("coef_bin_centers", centers)
        mu_invlog1p = 1.0
        if self.coef_quantization == "mu_law":
            mu_invlog1p = 1.0 / math.log1p(self.coef_mu)
        self.register_buffer("coef_mu_invlog1p", torch.tensor(mu_invlog1p))

        # Pre-compute the 2D Hann window (channel-tiled) as a buffer.
        # Shape: [patch_dim] = [C * patch_size * patch_size]
        hann_1d = torch.hann_window(self.patch_size, periodic=False)
        window_2d = hann_1d.unsqueeze(1) * hann_1d.unsqueeze(0)   # [ps, ps]
        window_flat = window_2d.flatten().unsqueeze(0).expand(
            self.embedding_dim, -1
        ).reshape(-1)                                               # [patch_dim]
        self.register_buffer("_hann_win", window_flat.clone())

        if self.quantize_sparse_coeffs:
            self.coeff_token_offset = self.num_embeddings
            self.token_depth = 2 * self.sparsity_level
            self.content_vocab_size = self.num_embeddings + self.n_bins
            self.pad_token_id = self.content_vocab_size
            self.bos_token_id = self.pad_token_id + 1
            self.vocab_size = self.content_vocab_size + 2
        else:
            self.coeff_token_offset = self.num_embeddings
            self.token_depth = self.sparsity_level
            self.content_vocab_size = self.num_embeddings
            self.pad_token_id = self.num_embeddings
            self.bos_token_id = self.num_embeddings + 1
            self.vocab_size = self.num_embeddings + 2

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        variational_prefixes = (
            prefix + "coeff_variational_atom_emb.",
            prefix + "coeff_variational_posterior.",
        )
        if self.variational_coeffs and self.coeff_variational_atom_emb is not None and self.coeff_variational_posterior is not None:
            expected_state = {
                prefix + "coeff_variational_atom_emb.weight": self.coeff_variational_atom_emb.weight.detach().clone(),
            }
            for subkey, value in self.coeff_variational_posterior.state_dict().items():
                expected_state[prefix + "coeff_variational_posterior." + subkey] = value.detach().clone()
            for key, expected in expected_state.items():
                loaded = state_dict.get(key, None)
                if loaded is None or tuple(loaded.shape) != tuple(expected.shape):
                    state_dict[key] = expected
        else:
            for key in list(state_dict.keys()):
                if key.startswith(variational_prefixes):
                    state_dict.pop(key, None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _normalize_dict(self) -> torch.Tensor:
        """Return column-normalised dictionary [patch_dim, num_embeddings]."""
        return F.normalize(self.dictionary, p=2, dim=0, eps=self.epsilon)

    # ---- coefficient quantisation (identical to DictionaryLearningTokenized) ----

    def _quantize_coeff(self, coeff: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.coef_quantization == "uniform":
            c = coeff.clamp(-self.coef_max, self.coef_max)
            scaled = (c + self.coef_max) / (2 * self.coef_max)
            bin_f = scaled * (self.n_bins - 1)
            bin_idx = torch.round(bin_f).to(torch.long).clamp(0, self.n_bins - 1)
            coeff_q = self.coef_bin_centers[bin_idx]
            return bin_idx, coeff_q
        c = coeff.clamp(-self.coef_max, self.coef_max) / self.coef_max
        c_abs = c.abs()
        encoded = torch.sign(c) * torch.log1p(c_abs * self.coef_mu) * self.coef_mu_invlog1p
        scaled = (encoded + 1.0) * ((self.n_bins - 1) / 2.0)
        bin_idx = torch.round(scaled).to(torch.long).clamp(0, self.n_bins - 1)
        decoded = self._dequantize_coeff(bin_idx)
        return bin_idx, decoded

    def _dequantize_coeff(self, bin_idx: torch.Tensor) -> torch.Tensor:
        if self.coef_quantization == "uniform":
            return self.coef_bin_centers[bin_idx]
        z = bin_idx.float() * (2.0 / (self.n_bins - 1)) - 1.0
        z_abs = z.abs()
        decoded_norm = torch.sign(z) * (torch.expm1(z_abs / self.coef_mu_invlog1p) / self.coef_mu)
        return decoded_norm * self.coef_max

    def _pack_quantized_tokens(self, support: torch.Tensor, bin_idx: torch.Tensor) -> torch.Tensor:
        tokens = torch.empty(
            *support.shape[:-1],
            self.token_depth,
            device=support.device,
            dtype=torch.long,
        )
        tokens[..., 0::2] = support.to(torch.long)
        tokens[..., 1::2] = bin_idx.to(torch.long) + self.coeff_token_offset
        return tokens

    def _unpack_quantized_tokens(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        atom_tokens = tokens[..., 0::2].to(torch.long)
        coeff_tokens = tokens[..., 1::2].to(torch.long)
        atom_invalid = (atom_tokens < 0) | (atom_tokens >= self.num_embeddings)
        coeff_bin = coeff_tokens - self.coeff_token_offset
        coeff_invalid = (coeff_bin < 0) | (coeff_bin >= self.n_bins)
        invalid = atom_invalid | coeff_invalid
        atom_ids = atom_tokens.clamp(0, self.num_embeddings - 1)
        coeff_bin = coeff_bin.clamp(0, self.n_bins - 1)
        coeffs = self._dequantize_coeff(coeff_bin)
        atom_ids = atom_ids.masked_fill(invalid, 0)
        coeffs = coeffs.masked_fill(invalid, 0.0)
        return atom_ids, coeffs

    # ---- patch extraction / reconstruction ----

    def _extract_patches(
        self, z_e: torch.Tensor
    ) -> Tuple[torch.Tensor, int, int, int, int]:
        """
        Pad then unfold z_e into overlapping patches.

        Two padding passes are applied:
          1. Symmetric padding of cs = (patch_size - patch_stride) // 2 so that
             the center of each reconstructed patch aligns with a non-overlapping
             patch_stride × patch_stride tile of the original latent.
          2. Asymmetric right/bottom padding to cover any remainder when H (or W)
             is not divisible by patch_stride — making this work for any stride.

        Returns:
            patches      : [B, patch_dim, L]  (L = nph * npw)
            nph, npw     : patch grid height / width
            H_orig, W_orig : original spatial dims before any padding
        """
        _, _, H, W = z_e.shape
        cs = (self.patch_size - self.patch_stride) // 2

        # Minimum patch count to cover the original extent after centering.
        nph = math.ceil(H / self.patch_stride)
        npw = math.ceil(W / self.patch_stride)

        # Total padded size required by unfold for nph / npw patches.
        H_pad_need = (nph - 1) * self.patch_stride + self.patch_size
        W_pad_need = (npw - 1) * self.patch_stride + self.patch_size

        pad_top = cs
        pad_left = cs
        pad_bottom = H_pad_need - H - cs   # may be > cs when H % stride != 0
        pad_right  = W_pad_need - W - cs

        z_e = F.pad(z_e, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")
        patches = F.unfold(z_e, kernel_size=self.patch_size, stride=self.patch_stride)
        return patches, nph, npw, H, W

    # ---- OMP (same algorithm as DictionaryLearningTokenized) ----

    def batch_omp_with_support(
        self, X: torch.Tensor, D: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched OMP.
        Args:
            X: [M, B] signals  (M = patch_dim)
            D: [M, N] normalised dictionary
        Returns:
            support: [B, K]
            coeffs : [B, K]
        """
        return _batched_omp_with_support(
            X=X,
            D=D,
            sparsity_level=self.sparsity_level,
        )

    def _encode_sparse_codes(
        self, z_e: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run OMP on every patch and return atom ids + coefficients.

        Returns:
            support : [B, nph, npw, K]
            coeffs  : [B, nph, npw, K]
        """
        patches, nph, npw, H, W = self._extract_patches(z_e)
        B = z_e.shape[0]
        L = patches.shape[2]  # nph * npw
        dictionary = self._normalize_dict()
        signals = patches.permute(0, 2, 1).contiguous().view(-1, self.patch_dim).t().to(dictionary.dtype)
        n_signals = B * L
        with torch.no_grad():
            support, coeffs = self.batch_omp_with_support(signals, dictionary)
        cur_d = min(support.size(1), coeffs.size(1))
        if cur_d < self.sparsity_level:
            pad = self.sparsity_level - cur_d
            support = torch.cat([support, torch.zeros(n_signals, pad, device=support.device, dtype=support.dtype)], dim=1)
            coeffs = torch.cat([coeffs, torch.zeros(n_signals, pad, device=coeffs.device, dtype=coeffs.dtype)], dim=1)
        else:
            support = support[:, :self.sparsity_level]
            coeffs = coeffs[:, :self.sparsity_level]
        if self.canonicalize_sparse_slots:
            # Canonicalize sparse slots so stage-2 sees a stable per-patch ordering.
            order = coeffs.abs().argsort(dim=1, descending=True)
            support = support.gather(1, order)
            coeffs = coeffs.gather(1, order)
        return (
            support.view(B, nph, npw, self.sparsity_level),
            coeffs.view(B, nph, npw, self.sparsity_level),
            H, W,
        )

    def clamp_sparse_coeffs(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Project coefficients onto the stage-1 decoder manifold."""
        coeffs = _nan_to_num_tensor(coeffs)
        if self.quantize_sparse_coeffs:
            return coeffs.clamp(-self.coef_max, self.coef_max)
        return coeffs

    def _coeff_posterior_stats(
        self,
        support: torch.Tensor,
        coeffs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.variational_coeffs:
            raise RuntimeError("_coeff_posterior_stats requires variational_coeffs=True")
        if self.coeff_variational_atom_emb is None or self.coeff_variational_posterior is None:
            raise RuntimeError("variational coefficient modules were not initialized")
        if support.shape != coeffs.shape:
            raise ValueError(f"support and coeffs shape mismatch: {support.shape} vs {coeffs.shape}")

        support_clamped = support.to(torch.long).clamp(0, self.num_embeddings - 1)
        coeffs_base = self.clamp_sparse_coeffs(coeffs.to(torch.float32))
        atom_emb = self.coeff_variational_atom_emb(support_clamped)
        posterior_in = torch.cat([atom_emb, coeffs_base.unsqueeze(-1)], dim=-1)
        posterior_raw = self.coeff_variational_posterior(posterior_in)

        mean_offset = self.variational_coeff_prior_std * torch.tanh(posterior_raw[..., 0])
        posterior_mu = self.clamp_sparse_coeffs(coeffs_base + mean_offset)

        std_range = max(self.variational_coeff_prior_std - self.variational_coeff_min_std, 0.0)
        posterior_std = self.variational_coeff_min_std + std_range * torch.sigmoid(posterior_raw[..., 1])
        posterior_logvar = 2.0 * torch.log(posterior_std.clamp_min(1e-6))
        return posterior_mu, posterior_logvar

    def project_sparse_coeffs(self, support: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        coeffs_clamped = self.clamp_sparse_coeffs(coeffs)
        if not self.variational_coeffs:
            return coeffs_clamped
        coeff_mu, _ = self._coeff_posterior_stats(support, coeffs_clamped)
        return coeff_mu

    def _reconstruct_sparse(
        self,
        support: torch.Tensor,
        coeffs: torch.Tensor,
        H: Optional[int] = None,
        W: Optional[int] = None,
    ) -> torch.Tensor:
        """Dispatch to the requested patch stitching strategy."""
        if self.patch_reconstruction == "tile":
            return self._reconstruct_tile(support, coeffs, H, W)
        if self.patch_reconstruction == "hann":
            return self._reconstruct_hann(support, coeffs, H, W)
        return self._reconstruct_center_crop(support, coeffs, H, W)

    def _reconstruct_tile(
        self,
        support: torch.Tensor,
        coeffs: torch.Tensor,
        H: Optional[int] = None,
        W: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Direct patch tiling for non-overlapping patches.

        H, W: original latent spatial dims. When provided, output is cropped
        to [B, C, H, W].
        """
        if self.patch_stride != self.patch_size:
            raise ValueError(
                "tile reconstruction requires non-overlapping patches: "
                f"patch_stride={self.patch_stride}, patch_size={self.patch_size}"
            )
        B, nph, npw, D = support.shape
        C = self.embedding_dim

        dictionary = self._normalize_dict().t()
        support_flat = support.to(torch.long).clamp(0, self.num_embeddings - 1).reshape(-1, D)
        coeffs_flat = self.clamp_sparse_coeffs(coeffs.to(dictionary.dtype)).reshape(-1, D)
        atoms = dictionary[support_flat]
        recon = (atoms * coeffs_flat.unsqueeze(-1)).sum(dim=1)

        recon = recon.view(B, nph, npw, C, self.patch_size, self.patch_size)
        recon = recon.permute(0, 3, 1, 4, 2, 5).contiguous()
        recon = recon.view(B, C, nph * self.patch_size, npw * self.patch_size)
        if H is not None and W is not None:
            recon = recon[:, :, :H, :W]
        return recon

    def _reconstruct_center_crop(
        self,
        support: torch.Tensor,
        coeffs: torch.Tensor,
        H: Optional[int] = None,
        W: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Center-crop stitching: each patch contributes only its center
        patch_stride × patch_stride region, forming a non-overlapping tiling.
        No averaging; every output pixel comes from exactly one patch.

        H, W: original latent spatial dims. When provided, output is cropped
        to [B, C, H, W], supporting any patch_stride regardless of divisibility.
        """
        B, nph, npw, D = support.shape
        s = self.patch_stride
        cs = (self.patch_size - self.patch_stride) // 2
        C = self.embedding_dim

        dictionary = self._normalize_dict().t()
        support_flat = support.to(torch.long).clamp(0, self.num_embeddings - 1).reshape(-1, D)
        coeffs_flat = self.clamp_sparse_coeffs(coeffs.to(dictionary.dtype)).reshape(-1, D)
        atoms = dictionary[support_flat]                          # [N, D, patch_dim]
        recon = (atoms * coeffs_flat.unsqueeze(-1)).sum(dim=1)   # [N, patch_dim]

        recon = recon.view(B * nph * npw, C, self.patch_size, self.patch_size)
        recon = recon[:, :, cs:cs + s, cs:cs + s]                # [N, C, s, s]

        recon = recon.view(B, nph, npw, C, s, s)
        recon = recon.permute(0, 3, 1, 4, 2, 5).contiguous()
        recon = recon.view(B, C, nph * s, npw * s)
        if H is not None and W is not None:
            recon = recon[:, :, :H, :W]
        return recon

    def _reconstruct_hann(
        self,
        support: torch.Tensor,
        coeffs: torch.Tensor,
        H: Optional[int] = None,
        W: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Weighted overlap-add with a 2D Hann window.

        H, W: original latent spatial dims. When provided, output is cropped
        to [B, C, H, W] after stripping the centering pad.
        """
        B, nph, npw, D = support.shape
        s = self.patch_stride
        cs = (self.patch_size - self.patch_stride) // 2
        C = self.embedding_dim
        # Padded fold dimensions matching what _extract_patches produced.
        H_pad = (nph - 1) * s + self.patch_size
        W_pad = (npw - 1) * s + self.patch_size

        dictionary = self._normalize_dict().t()
        support_flat = support.to(torch.long).clamp(0, self.num_embeddings - 1).reshape(-1, D)
        coeffs_flat = self.clamp_sparse_coeffs(coeffs.to(dictionary.dtype)).reshape(-1, D)
        atoms = dictionary[support_flat]                          # [N, D, patch_dim]
        recon = (atoms * coeffs_flat.unsqueeze(-1)).sum(dim=1)   # [N, patch_dim]

        win = self._hann_win.to(recon.dtype)
        recon = recon * win.unsqueeze(0)
        recon = recon.view(B, nph * npw, self.patch_dim).permute(0, 2, 1)

        weighted = F.fold(recon, output_size=(H_pad, W_pad),
                          kernel_size=self.patch_size, stride=s)
        win_map = F.fold(
            win.view(1, -1, 1).expand(B, -1, nph * npw),
            output_size=(H_pad, W_pad),
            kernel_size=self.patch_size, stride=s,
        )
        out = weighted / win_map.clamp_min(1e-8)

        # strip centering pad then crop to original H × W
        out = out[:, :, cs:cs + nph * s, cs:cs + npw * s]
        if H is not None and W is not None:
            out = out[:, :, :H, :W]
        return out

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_e : [B, C, H, W]
        Returns:
            z_q_ste : [B, C, H, W]
            loss    : scalar bottleneck loss
            tokens  : [B, nph, npw, token_depth]
        """
        if z_e.dim() != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(z_e.shape)}")
        B, C, _, _ = z_e.shape
        if C != self.embedding_dim:
            raise ValueError(f"Expected channel dim {self.embedding_dim}, got {C}")

        z_e = _nan_to_num_tensor(z_e)
        support, coeffs, H, W = self._encode_sparse_codes(z_e)
        _, nph, npw, _ = support.shape

        support_flat = support.view(-1, self.sparsity_level)
        coeffs_flat = self.clamp_sparse_coeffs(coeffs.view(-1, self.sparsity_level))

        if self.quantize_sparse_coeffs:
            bin_idx, coeff_q = self._quantize_coeff(coeffs_flat)
            tokens = self._pack_quantized_tokens(
                support_flat.view(B, nph, npw, self.sparsity_level),
                bin_idx.view(B, nph, npw, self.sparsity_level),
            )
            coeffs_for_recon = coeff_q.reshape(B, nph, npw, self.sparsity_level)
        else:
            tokens = support.view(B, nph, npw, self.sparsity_level).long()
            coeffs_for_recon = self.clamp_sparse_coeffs(coeffs_flat).reshape(B, nph, npw, self.sparsity_level)

        coeff_kl_loss = z_e.new_zeros(())
        weighted_coeff_kl_loss = z_e.new_zeros(())
        if (not self.quantize_sparse_coeffs) and self.variational_coeffs:
            coeffs_base = coeffs_for_recon
            coeff_mu, coeff_logvar = self._coeff_posterior_stats(support, coeffs_base)
            if self.training:
                coeff_eps = torch.randn_like(coeff_mu)
                coeff_std = (0.5 * coeff_logvar).exp()
                coeffs_for_recon = self.clamp_sparse_coeffs(coeff_mu + coeff_std * coeff_eps)
            else:
                coeffs_for_recon = coeff_mu
            coeff_kl_loss = _gaussian_kl_to_fixed_mean(
                coeff_mu,
                coeff_logvar,
                coeffs_base,
                target_std=self.variational_coeff_prior_std,
            )
            weighted_coeff_kl_loss = float(self.variational_coeff_kl_weight) * coeff_kl_loss
            self._last_coeff_posterior_std = (0.5 * coeff_logvar).exp().mean().detach()
            self._last_coeff_prior_std = torch.as_tensor(
                self.variational_coeff_prior_std,
                device=z_e.device,
                dtype=z_e.dtype,
            )
        else:
            self._last_coeff_posterior_std = z_e.new_zeros(())
            self._last_coeff_prior_std = torch.as_tensor(
                self.variational_coeff_prior_std,
                device=z_e.device,
                dtype=z_e.dtype,
            )

        z_q = _nan_to_num_tensor(self._reconstruct_sparse(support, coeffs_for_recon, H, W))

        dl_latent_loss = F.mse_loss(z_q, z_e.detach())
        e_latent_loss = F.mse_loss(z_q.detach(), z_e)
        loss = dl_latent_loss + self.commitment_cost * e_latent_loss + weighted_coeff_kl_loss
        self._last_dl_latent_loss = dl_latent_loss
        self._last_e_latent_loss = e_latent_loss
        self._last_coeff_kl_loss = coeff_kl_loss
        self._last_weighted_coeff_kl_loss = weighted_coeff_kl_loss
        self._last_extra_bottleneck_loss = weighted_coeff_kl_loss
        self._last_bottleneck_loss = loss

        z_q_ste = z_e + (z_q - z_e).detach()
        return z_q_ste, loss, tokens

    @torch.no_grad()
    def tokens_to_latent(
        self,
        tokens: torch.Tensor,
        coeffs: Optional[torch.Tensor] = None,
        latent_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Decode tokens back to a latent map.

        Args:
            tokens : [B, nph, npw, token_depth]
            coeffs : [B, nph, npw, K]  (only used in non-quantized mode)
            latent_hw : optional original latent spatial size (H, W)
        Returns:
            z_q : [B, C, H, W]
        """
        if tokens.dim() != 4:
            raise ValueError(f"Expected [B,nph,npw,D], got {tuple(tokens.shape)}")
        _, _, _, D = tokens.shape
        if D != self.token_depth:
            raise ValueError(f"Expected token depth {self.token_depth}, got {D}")

        if self.quantize_sparse_coeffs:
            atom_ids, coeff_q = self._unpack_quantized_tokens(tokens)
            if latent_hw is None:
                return self._reconstruct_sparse(atom_ids, coeff_q)
            return self._reconstruct_sparse(atom_ids, coeff_q, int(latent_hw[0]), int(latent_hw[1]))

        if coeffs is None:
            raise ValueError("coeffs must be provided when quantize_sparse_coeffs=False")
        coeffs_clamped = self.clamp_sparse_coeffs(coeffs.to(self._normalize_dict().dtype))
        if latent_hw is None:
            return self._reconstruct_sparse(
                tokens.to(torch.long),
                coeffs_clamped,
            )
        return self._reconstruct_sparse(
            tokens.to(torch.long),
            coeffs_clamped,
            int(latent_hw[0]),
            int(latent_hw[1]),
        )


# -----------------------------
# Stage-1 model: Encoder + Dictionary bottleneck + Decoder
# -----------------------------

class LASER(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_hiddens: int = 128,
        num_downsamples: int = 2,
        num_residual_layers: int = 2,
        resolution: int = 128,
        attn_resolutions: tuple = (),
        dropout: float = 0.0,
        max_ch_mult: int = 2,
        decoder_extra_residual_layers: int = 1,
        use_mid_attention: bool = True,
        embedding_dim: int = 16,
        num_embeddings: int = 1024,
        sparsity_level: int = 8,
        commitment_cost: float = 0.25,
        n_bins: int = 16,
        coef_max: float = 3.0,
        coef_quantization: str = "uniform",
        coef_mu: float = 50.0,
        out_tanh: bool = True,
        quantize_sparse_coeffs: bool = False,
        patch_based: bool = False,
        patch_size: int = 8,
        patch_stride: int = 4,
        patch_reconstruction: str = "center_crop",
        variational_coeffs: bool = False,
        variational_coeff_kl_weight: float = 0.0,
        variational_coeff_prior_std: float = 0.25,
        variational_coeff_min_std: float = 0.01,
    ):
        super().__init__()
        self.out_tanh = bool(out_tanh)
        self.max_ch_mult = int(max_ch_mult)
        self.decoder_extra_residual_layers = int(decoder_extra_residual_layers)
        self.use_mid_attention = bool(use_mid_attention)

        if self.max_ch_mult <= 0:
            raise ValueError(f"max_ch_mult must be positive, got {self.max_ch_mult}")
        if self.decoder_extra_residual_layers < 0:
            raise ValueError(
                f"decoder_extra_residual_layers must be non-negative, got {self.decoder_extra_residual_layers}"
            )

        # ch_mult controls the channel multiplier at each resolution level;
        # len(ch_mult) - 1 equals the number of spatial downsampling steps.
        # Cap multipliers to keep the max width bounded without changing the
        # number of encoder or decoder stages.
        ch_mult = tuple(min(2 ** i, self.max_ch_mult) for i in range(num_downsamples + 1))

        enc_dec_kwargs = dict(
            ch=num_hiddens,
            out_ch=in_channels,
            ch_mult=ch_mult,
            num_res_blocks=num_residual_layers,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            use_mid_attention=self.use_mid_attention,
            resamp_with_conv=True,
            in_channels=in_channels,
            resolution=resolution,
            z_channels=embedding_dim,
        )
        dec_kwargs = dict(enc_dec_kwargs, extra_res_blocks=self.decoder_extra_residual_layers)
        self.encoder = Encoder(**enc_dec_kwargs)
        bottleneck_kwargs = dict(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            sparsity_level=sparsity_level,
            n_bins=n_bins,
            coef_max=coef_max,
            quantize_sparse_coeffs=quantize_sparse_coeffs,
            coef_quantization=coef_quantization,
            coef_mu=coef_mu,
            commitment_cost=commitment_cost,
            variational_coeffs=variational_coeffs,
            variational_coeff_kl_weight=variational_coeff_kl_weight,
            variational_coeff_prior_std=variational_coeff_prior_std,
            variational_coeff_min_std=variational_coeff_min_std,
        )
        if patch_based:
            self.bottleneck = PatchDictionaryLearningTokenized(
                patch_size=patch_size,
                patch_stride=patch_stride,
                patch_reconstruction=patch_reconstruction,
                **bottleneck_kwargs,
            )
        else:
            self.bottleneck = DictionaryLearningTokenized(**bottleneck_kwargs)
        self.decoder = Decoder(**dec_kwargs)
        self._last_latent_hw: Optional[Tuple[int, int]] = None

    def _remember_latent_hw(self, z: torch.Tensor) -> None:
        self._last_latent_hw = (int(z.shape[-2]), int(z.shape[-1]))

    def _resolve_patch_latent_hw(self, latent_hw: Optional[Tuple[int, int]] = None) -> Optional[Tuple[int, int]]:
        if not isinstance(self.bottleneck, PatchDictionaryLearningTokenized):
            return None
        if latent_hw is not None:
            return (int(latent_hw[0]), int(latent_hw[1]))
        decoder_z_shape = getattr(self.decoder, "z_shape", None)
        if isinstance(decoder_z_shape, tuple) and len(decoder_z_shape) >= 4:
            return (int(decoder_z_shape[-2]), int(decoder_z_shape[-1]))
        if self._last_latent_hw is None:
            raise RuntimeError(
                "Patch-based decoding requires the original latent spatial size. "
                "Run an encode/forward pass first or pass latent_hw explicitly."
            )
        return self._last_latent_hw

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        self._remember_latent_hw(z)
        bottleneck_ctx = (
            torch.autocast(device_type=z.device.type, enabled=False)
            if z.is_cuda and torch.is_autocast_enabled()
            else nullcontext()
        )
        with bottleneck_ctx:
            z_q, b_loss, tokens = self.bottleneck(z.float())
        recon = self.decoder(z_q)
        if self.out_tanh:
            recon = torch.tanh(recon)
        return recon, b_loss, tokens

    @torch.no_grad()
    def encode_to_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        z = self.encoder(x)
        self._remember_latent_hw(z)
        _, _, tokens = self.bottleneck(z)
        return tokens, tokens.shape[1], tokens.shape[2]

    @torch.no_grad()
    def encode_to_atoms_and_coeffs(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        z = self.encoder(x)
        self._remember_latent_hw(z)
        encoded = self.bottleneck._encode_sparse_codes(z)
        if isinstance(self.bottleneck, PatchDictionaryLearningTokenized):
            atoms, coeffs, _, _ = encoded
        else:
            atoms, coeffs = encoded
        coeffs = self.bottleneck.project_sparse_coeffs(atoms, coeffs)
        return atoms, coeffs, atoms.shape[1], atoms.shape[2]

    def clamp_sparse_coeffs(self, coeffs: torch.Tensor) -> torch.Tensor:
        return self.bottleneck.clamp_sparse_coeffs(coeffs)

    @torch.no_grad()
    def decode_from_tokens(
        self,
        tokens: torch.Tensor,
        latent_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        patch_latent_hw = self._resolve_patch_latent_hw(latent_hw)
        if isinstance(self.bottleneck, PatchDictionaryLearningTokenized):
            z_q = self.bottleneck.tokens_to_latent(tokens, latent_hw=patch_latent_hw)
        else:
            z_q = self.bottleneck.tokens_to_latent(tokens)
        recon = self.decoder(z_q)
        if self.out_tanh:
            recon = torch.tanh(recon)
        return recon

    @torch.no_grad()
    def decode_from_atoms_and_coeffs(
        self,
        atoms: torch.Tensor,
        coeffs: torch.Tensor,
        latent_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        coeffs = self.clamp_sparse_coeffs(coeffs)
        patch_latent_hw = self._resolve_patch_latent_hw(latent_hw)
        if isinstance(self.bottleneck, PatchDictionaryLearningTokenized):
            z_q = self.bottleneck._reconstruct_sparse(
                atoms,
                coeffs,
                int(patch_latent_hw[0]),
                int(patch_latent_hw[1]),
            )
        else:
            z_q = self.bottleneck._reconstruct_sparse(atoms, coeffs)
        recon = self.decoder(z_q)
        if self.out_tanh:
            recon = torch.tanh(recon)
        return recon


# Backward-compatible alias for older scratch experiments.
SparseDictAE = LASER


# -----------------------------
# Patch spectrum analysis
# -----------------------------

@torch.no_grad()
def analyze_patch_spectrum(
    ae: LASER,
    loader: DataLoader,
    device: torch.device,
    n_patches: int = 50_000,
    var_targets: Tuple[float, ...] = (0.80, 0.90, 0.95, 0.99),
    bar_width: int = 60,
) -> dict:
    """
    Compute the PCA spectrum of latent patches to guide sparsity-level selection.

    Uses the patch covariance approach (O(patch_dim²) memory) rather than a
    full SVD of the data matrix, so it remains tractable even for large n_patches.

    Args:
        ae          : trained (or partially trained) LASER with a patch bottleneck
        loader      : DataLoader of (image, label) pairs
        device      : compute device
        n_patches   : how many patches to accumulate before computing the spectrum
        var_targets : cumulative-variance thresholds to report K* for
        bar_width   : width of the ASCII bar chart

    Returns dict with keys:
        eigenvalues   : [patch_dim] tensor (descending)
        cumvar        : [patch_dim] cumulative variance fraction
        k_for_target  : {target: K} minimum K to reach each variance target
        patch_dim     : int
        n_patches     : int  (actual count used)
    """
    ae_module = _unwrap_module(ae)
    if not isinstance(ae_module.bottleneck, PatchDictionaryLearningTokenized):
        raise ValueError(
            "analyze_patch_spectrum requires a patch-based bottleneck "
            "(LASER built with patch_based=True)."
        )
    bn = ae_module.bottleneck
    patch_dim = bn.patch_dim

    ae_module.eval()

    # Accumulate patch covariance online to avoid storing all patches.
    # cov = (1/N) * sum_i (p_i - mu)(p_i - mu)^T  via Welford-style update.
    running_sum = torch.zeros(patch_dim, device=device, dtype=torch.float64)
    running_cov = torch.zeros(patch_dim, patch_dim, device=device, dtype=torch.float64)
    seen = 0

    pbar = tqdm(loader, desc="[Spectrum] collecting patches", leave=False, dynamic_ncols=True)
    for x, _ in pbar:
        if seen >= n_patches:
            break
        x = x.to(device)
        z = ae_module.encoder(x)
        z = ae_module.pre(z)
        patches, _, _, _, _ = bn._extract_patches(z)
        # patches: [B, patch_dim, L]  →  flat: [N_local, patch_dim]
        P = patches.permute(0, 2, 1).reshape(-1, patch_dim).double()
        keep = min(P.shape[0], n_patches - seen)
        P = P[:keep]

        running_sum += P.sum(0)
        running_cov += P.T @ P
        seen += keep
        pbar.set_postfix(patches=seen)

    if seen == 0:
        raise RuntimeError("No patches collected — check the loader.")

    mu = running_sum / seen
    # cov = E[pp^T] - mu mu^T
    cov = running_cov / seen - mu.unsqueeze(1) * mu.unsqueeze(0)

    # Symmetric eigen-decomposition (more stable than SVD for covariance matrices).
    eigvals = torch.linalg.eigvalsh(cov)   # ascending
    eigvals = eigvals.flip(0).clamp_min(0) # descending, clamp floating-point negatives

    total_var = eigvals.sum()
    cumvar = eigvals.cumsum(0) / total_var.clamp_min(1e-12)

    k_for_target = {}
    for t in var_targets:
        k = int((cumvar < t).sum().item()) + 1
        k_for_target[float(t)] = min(k, patch_dim)

    # ---- pretty print ----
    print(f"\n{'─'*bar_width}")
    print(f"  Patch spectrum analysis")
    print(f"  patch_dim = {patch_dim}  |  patches used = {seen:,}")
    print(f"{'─'*bar_width}")

    # bar chart: top-64 eigenvalues (or patch_dim, whichever is smaller)
    n_show = min(64, patch_dim)
    ev_show = eigvals[:n_show].float()
    ev_max = ev_show[0].item()
    print(f"  Top-{n_show} eigenvalues (normalised):")
    for i, v in enumerate(ev_show):
        frac = v.item() / ev_max if ev_max > 0 else 0
        bar = "█" * int(frac * 20)
        pct = float(cumvar[i].item()) * 100
        print(f"  {i+1:3d}  {bar:<20s}  cumvar={pct:5.1f}%")

    print(f"{'─'*bar_width}")
    print(f"  Recommended minimum K (sparsity_level) per quality target:")
    for t, k in k_for_target.items():
        print(f"    {t*100:.0f}% variance  →  K ≥ {k}")
    print(f"{'─'*bar_width}\n")

    return {
        "eigenvalues": eigvals.float().cpu(),
        "cumvar": cumvar.float().cpu(),
        "k_for_target": k_for_target,
        "patch_dim": patch_dim,
        "n_patches": seen,
    }


# -----------------------------
# Training helpers
# -----------------------------

def _make_image_grid(x: torch.Tensor, nrow: int = 8) -> torch.Tensor:
    """Build image grid tensor from a batch in [-1, 1]."""
    x = x.detach().cpu().clamp(-1, 1)
    x = (x + 1.0) / 2.0
    return utils.make_grid(x, nrow=nrow)


def save_image_grid(x: torch.Tensor, path: str, nrow: int = 8):
    """Save a batch in [-1, 1] as a single image grid."""
    grid = _make_image_grid(x, nrow=nrow)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    utils.save_image(grid, path)


def _to_unit_range(x: torch.Tensor) -> torch.Tensor:
    """Map images from [-1, 1] to [0, 1] for reconstruction metrics."""
    return x.detach().clamp(-1, 1).add(1.0).mul(0.5)


def _sample_quality_features(x: torch.Tensor) -> torch.Tensor:
    """Compact image features for filtering off-manifold stage-2 samples."""
    x_unit = _to_unit_range(x)
    B, C, H, W = x_unit.shape
    rgb_mean = x_unit.mean(dim=(2, 3))
    rgb_std = x_unit.std(dim=(2, 3))
    saturation = (x_unit.amax(dim=1) - x_unit.amin(dim=1)).mean(dim=(1, 2)).unsqueeze(1)
    brightness = x_unit.mean(dim=(1, 2, 3)).unsqueeze(1)
    luma = (0.2126 * x_unit[:, 0] + 0.7152 * x_unit[:, 1] + 0.0722 * x_unit[:, 2]).reshape(B, -1)
    luma_p10 = torch.quantile(luma, 0.10, dim=1, keepdim=True)
    luma_p25 = torch.quantile(luma, 0.25, dim=1, keepdim=True)
    dark_frac20 = (luma < 0.20).float().mean(dim=1, keepdim=True)
    dark_frac30 = (luma < 0.30).float().mean(dim=1, keepdim=True)

    border_width = max(1, min(H, W) // 16)
    border_pixels = torch.cat([
        x_unit[:, :, :border_width, :].reshape(B, C, -1),
        x_unit[:, :, -border_width:, :].reshape(B, C, -1),
        x_unit[:, :, border_width:-border_width, :border_width].reshape(B, C, -1),
        x_unit[:, :, border_width:-border_width, -border_width:].reshape(B, C, -1),
    ], dim=2)
    if H > 2 * border_width and W > 2 * border_width:
        center_region = x_unit[:, :, border_width:-border_width, border_width:-border_width]
    else:
        center_region = x_unit
    center_pixels = center_region.reshape(B, C, -1)
    center_luma = (
        0.2126 * center_region[:, 0]
        + 0.7152 * center_region[:, 1]
        + 0.0722 * center_region[:, 2]
    ).reshape(B, -1)
    center_luma_p10 = torch.quantile(center_luma, 0.10, dim=1, keepdim=True)
    center_dark_frac20 = (center_luma < 0.20).float().mean(dim=1, keepdim=True)
    center_dark_frac30 = (center_luma < 0.30).float().mean(dim=1, keepdim=True)
    border_mean = border_pixels.mean(dim=2)
    border_std = border_pixels.std(dim=2)
    center_mean = center_pixels.mean(dim=2)
    border_center_gap = border_mean - center_mean

    return torch.cat([
        rgb_mean,
        rgb_std,
        saturation,
        brightness,
        border_mean,
        border_std,
        border_center_gap,
        luma_p10,
        luma_p25,
        dark_frac20,
        dark_frac30,
        center_luma_p10,
        center_dark_frac20,
        center_dark_frac30,
    ], dim=1)


def _sample_tensor_stats_dict(x: Optional[torch.Tensor], prefix: str) -> dict:
    """Compact scalar summary for a sampled tensor batch."""
    if x is None:
        return {}
    flat = x.detach().float().reshape(-1)
    if flat.numel() == 0:
        return {}
    return {
        f"{prefix}mean": float(flat.mean().item()),
        f"{prefix}std": float(flat.std(unbiased=False).item()),
        f"{prefix}min": float(flat.min().item()),
        f"{prefix}max": float(flat.max().item()),
        f"{prefix}abs_mean": float(flat.abs().mean().item()),
    }


def _sample_token_stats_dict(
    tokens: Optional[torch.Tensor],
    prefix: str,
    vocab_size: Optional[int] = None,
) -> dict:
    """Compact scalar summary for quantized sampled token grids."""
    if tokens is None:
        return {}
    flat = tokens.detach().reshape(-1).to(dtype=torch.long)
    if flat.numel() == 0:
        return {}
    unique = int(torch.unique(flat).numel())
    payload = {
        f"{prefix}mean": float(flat.float().mean().item()),
        f"{prefix}std": float(flat.float().std(unbiased=False).item()),
        f"{prefix}min": int(flat.min().item()),
        f"{prefix}max": int(flat.max().item()),
        f"{prefix}unique_count": unique,
    }
    if vocab_size is not None:
        payload[f"{prefix}unique_vocab_fraction"] = float(unique) / float(max(1, int(vocab_size)))
    return payload


def _sample_image_stats_dict(x: Optional[torch.Tensor], prefix: str) -> dict:
    """Compact scalar summary for decoded stage-2 preview images."""
    if x is None:
        return {}
    x_detached = x.detach().float()
    if x_detached.numel() == 0:
        return {}
    x_unit = _to_unit_range(x_detached)
    brightness = x_unit.mean(dim=(1, 2, 3))
    saturation = (x_unit.amax(dim=1) - x_unit.amin(dim=1)).mean(dim=(1, 2))
    contrast = x_unit.std(dim=(1, 2, 3), unbiased=False)
    return {
        f"{prefix}pixel_mean": float(x_detached.mean().item()),
        f"{prefix}pixel_std": float(x_detached.std(unbiased=False).item()),
        f"{prefix}pixel_min": float(x_detached.min().item()),
        f"{prefix}pixel_max": float(x_detached.max().item()),
        f"{prefix}brightness_mean": float(brightness.mean().item()),
        f"{prefix}brightness_std": float(brightness.std(unbiased=False).item()),
        f"{prefix}saturation_mean": float(saturation.mean().item()),
        f"{prefix}contrast_mean": float(contrast.mean().item()),
    }


_SAMPLE_FEAT_BRIGHTNESS_SLICE = slice(7, 8)
_SAMPLE_FEAT_BORDER_MEAN_SLICE = slice(8, 11)
_SAMPLE_FEAT_BORDER_CENTER_GAP_SLICE = slice(14, 17)
_SAMPLE_FEAT_LUMA_P10_SLICE = slice(17, 18)
_SAMPLE_FEAT_LUMA_P25_SLICE = slice(18, 19)
_SAMPLE_FEAT_DARK_FRAC20_SLICE = slice(19, 20)
_SAMPLE_FEAT_DARK_FRAC30_SLICE = slice(20, 21)
_SAMPLE_FEAT_CENTER_LUMA_P10_SLICE = slice(21, 22)
_SAMPLE_FEAT_CENTER_DARK_FRAC20_SLICE = slice(22, 23)
_SAMPLE_FEAT_CENTER_DARK_FRAC30_SLICE = slice(23, 24)


def _sample_feature_center_brightness(feats: torch.Tensor) -> torch.Tensor:
    border_mean = feats[:, _SAMPLE_FEAT_BORDER_MEAN_SLICE]
    border_center_gap = feats[:, _SAMPLE_FEAT_BORDER_CENTER_GAP_SLICE]
    center_mean = border_mean - border_center_gap
    return center_mean.mean(dim=1, keepdim=True)


def _sample_feature_low_brightness_penalty(
    feats: torch.Tensor,
    ref_mean: torch.Tensor,
    ref_std: torch.Tensor,
) -> torch.Tensor:
    brightness = feats[:, _SAMPLE_FEAT_BRIGHTNESS_SLICE]
    ref_brightness = ref_mean[:, _SAMPLE_FEAT_BRIGHTNESS_SLICE]
    ref_brightness_std = ref_std[:, _SAMPLE_FEAT_BRIGHTNESS_SLICE].clamp_min(1e-6)
    global_darkness = torch.relu((ref_brightness - brightness) / ref_brightness_std)

    center_brightness = _sample_feature_center_brightness(feats)
    ref_center_brightness = _sample_feature_center_brightness(ref_mean)
    ref_center_std = torch.sqrt(
        ref_std[:, _SAMPLE_FEAT_BORDER_MEAN_SLICE].pow(2)
        + ref_std[:, _SAMPLE_FEAT_BORDER_CENTER_GAP_SLICE].pow(2)
    ).mean(dim=1, keepdim=True).clamp_min(1e-6)
    center_darkness = torch.relu((ref_center_brightness - center_brightness) / ref_center_std)
    luma_p10 = feats[:, _SAMPLE_FEAT_LUMA_P10_SLICE]
    ref_luma_p10 = ref_mean[:, _SAMPLE_FEAT_LUMA_P10_SLICE]
    ref_luma_p10_std = ref_std[:, _SAMPLE_FEAT_LUMA_P10_SLICE].clamp_min(1e-6)
    luma_p10_darkness = torch.relu((ref_luma_p10 - luma_p10) / ref_luma_p10_std)

    luma_p25 = feats[:, _SAMPLE_FEAT_LUMA_P25_SLICE]
    ref_luma_p25 = ref_mean[:, _SAMPLE_FEAT_LUMA_P25_SLICE]
    ref_luma_p25_std = ref_std[:, _SAMPLE_FEAT_LUMA_P25_SLICE].clamp_min(1e-6)
    luma_p25_darkness = torch.relu((ref_luma_p25 - luma_p25) / ref_luma_p25_std)

    dark_frac20 = feats[:, _SAMPLE_FEAT_DARK_FRAC20_SLICE]
    ref_dark_frac20 = ref_mean[:, _SAMPLE_FEAT_DARK_FRAC20_SLICE]
    ref_dark_frac20_std = ref_std[:, _SAMPLE_FEAT_DARK_FRAC20_SLICE].clamp_min(1e-6)
    dark_frac20_excess = torch.relu((dark_frac20 - ref_dark_frac20) / ref_dark_frac20_std)

    dark_frac30 = feats[:, _SAMPLE_FEAT_DARK_FRAC30_SLICE]
    ref_dark_frac30 = ref_mean[:, _SAMPLE_FEAT_DARK_FRAC30_SLICE]
    ref_dark_frac30_std = ref_std[:, _SAMPLE_FEAT_DARK_FRAC30_SLICE].clamp_min(1e-6)
    dark_frac30_excess = torch.relu((dark_frac30 - ref_dark_frac30) / ref_dark_frac30_std)

    center_luma_p10 = feats[:, _SAMPLE_FEAT_CENTER_LUMA_P10_SLICE]
    ref_center_luma_p10 = ref_mean[:, _SAMPLE_FEAT_CENTER_LUMA_P10_SLICE]
    ref_center_luma_p10_std = ref_std[:, _SAMPLE_FEAT_CENTER_LUMA_P10_SLICE].clamp_min(1e-6)
    center_luma_p10_darkness = torch.relu((ref_center_luma_p10 - center_luma_p10) / ref_center_luma_p10_std)

    center_dark_frac20 = feats[:, _SAMPLE_FEAT_CENTER_DARK_FRAC20_SLICE]
    ref_center_dark_frac20 = ref_mean[:, _SAMPLE_FEAT_CENTER_DARK_FRAC20_SLICE]
    ref_center_dark_frac20_std = ref_std[:, _SAMPLE_FEAT_CENTER_DARK_FRAC20_SLICE].clamp_min(1e-6)
    center_dark_frac20_excess = torch.relu((center_dark_frac20 - ref_center_dark_frac20) / ref_center_dark_frac20_std)

    center_dark_frac30 = feats[:, _SAMPLE_FEAT_CENTER_DARK_FRAC30_SLICE]
    ref_center_dark_frac30 = ref_mean[:, _SAMPLE_FEAT_CENTER_DARK_FRAC30_SLICE]
    ref_center_dark_frac30_std = ref_std[:, _SAMPLE_FEAT_CENTER_DARK_FRAC30_SLICE].clamp_min(1e-6)
    center_dark_frac30_excess = torch.relu((center_dark_frac30 - ref_center_dark_frac30) / ref_center_dark_frac30_std)

    penalty_terms = torch.cat([
        global_darkness,
        center_darkness,
        luma_p10_darkness,
        luma_p25_darkness,
        dark_frac20_excess,
        dark_frac30_excess,
        center_luma_p10_darkness,
        center_dark_frac20_excess,
        center_dark_frac30_excess,
    ], dim=1)
    return penalty_terms.mean(dim=1)


def _sample_feature_high_brightness_penalty(
    feats: torch.Tensor,
    ref_mean: torch.Tensor,
    ref_std: torch.Tensor,
) -> torch.Tensor:
    brightness = feats[:, _SAMPLE_FEAT_BRIGHTNESS_SLICE]
    ref_brightness = ref_mean[:, _SAMPLE_FEAT_BRIGHTNESS_SLICE]
    ref_brightness_std = ref_std[:, _SAMPLE_FEAT_BRIGHTNESS_SLICE].clamp_min(1e-6)
    global_brightness = torch.relu((brightness - ref_brightness) / ref_brightness_std)

    center_brightness = _sample_feature_center_brightness(feats)
    ref_center_brightness = _sample_feature_center_brightness(ref_mean)
    ref_center_std = torch.sqrt(
        ref_std[:, _SAMPLE_FEAT_BORDER_MEAN_SLICE].pow(2)
        + ref_std[:, _SAMPLE_FEAT_BORDER_CENTER_GAP_SLICE].pow(2)
    ).mean(dim=1, keepdim=True).clamp_min(1e-6)
    center_brightness_excess = torch.relu((center_brightness - ref_center_brightness) / ref_center_std)

    luma_p10 = feats[:, _SAMPLE_FEAT_LUMA_P10_SLICE]
    ref_luma_p10 = ref_mean[:, _SAMPLE_FEAT_LUMA_P10_SLICE]
    ref_luma_p10_std = ref_std[:, _SAMPLE_FEAT_LUMA_P10_SLICE].clamp_min(1e-6)
    luma_p10_brightness = torch.relu((luma_p10 - ref_luma_p10) / ref_luma_p10_std)

    luma_p25 = feats[:, _SAMPLE_FEAT_LUMA_P25_SLICE]
    ref_luma_p25 = ref_mean[:, _SAMPLE_FEAT_LUMA_P25_SLICE]
    ref_luma_p25_std = ref_std[:, _SAMPLE_FEAT_LUMA_P25_SLICE].clamp_min(1e-6)
    luma_p25_brightness = torch.relu((luma_p25 - ref_luma_p25) / ref_luma_p25_std)

    center_luma_p10 = feats[:, _SAMPLE_FEAT_CENTER_LUMA_P10_SLICE]
    ref_center_luma_p10 = ref_mean[:, _SAMPLE_FEAT_CENTER_LUMA_P10_SLICE]
    ref_center_luma_p10_std = ref_std[:, _SAMPLE_FEAT_CENTER_LUMA_P10_SLICE].clamp_min(1e-6)
    center_luma_p10_brightness = torch.relu(
        (center_luma_p10 - ref_center_luma_p10) / ref_center_luma_p10_std
    )

    penalty_terms = torch.cat([
        global_brightness,
        center_brightness_excess,
        luma_p10_brightness,
        luma_p25_brightness,
        center_luma_p10_brightness,
    ], dim=1)
    return penalty_terms.mean(dim=1)


def _sample_feature_dark_rejection_mask(
    feats: torch.Tensor,
    ref_mean: torch.Tensor,
    ref_std: torch.Tensor,
    reject_dark_z: float,
) -> torch.Tensor:
    if float(reject_dark_z) <= 0.0:
        return torch.ones(feats.size(0), device=feats.device, dtype=torch.bool)

    z = float(reject_dark_z)
    brightness = feats[:, _SAMPLE_FEAT_BRIGHTNESS_SLICE]
    ref_brightness = ref_mean[:, _SAMPLE_FEAT_BRIGHTNESS_SLICE]
    ref_brightness_std = ref_std[:, _SAMPLE_FEAT_BRIGHTNESS_SLICE].clamp_min(1e-6)

    center_brightness = _sample_feature_center_brightness(feats)
    ref_center_brightness = _sample_feature_center_brightness(ref_mean)
    ref_center_std = torch.sqrt(
        ref_std[:, _SAMPLE_FEAT_BORDER_MEAN_SLICE].pow(2)
        + ref_std[:, _SAMPLE_FEAT_BORDER_CENTER_GAP_SLICE].pow(2)
    ).mean(dim=1, keepdim=True).clamp_min(1e-6)

    luma_p10 = feats[:, _SAMPLE_FEAT_LUMA_P10_SLICE]
    ref_luma_p10 = ref_mean[:, _SAMPLE_FEAT_LUMA_P10_SLICE]
    ref_luma_p10_std = ref_std[:, _SAMPLE_FEAT_LUMA_P10_SLICE].clamp_min(1e-6)

    center_luma_p10 = feats[:, _SAMPLE_FEAT_CENTER_LUMA_P10_SLICE]
    ref_center_luma_p10 = ref_mean[:, _SAMPLE_FEAT_CENTER_LUMA_P10_SLICE]
    ref_center_luma_p10_std = ref_std[:, _SAMPLE_FEAT_CENTER_LUMA_P10_SLICE].clamp_min(1e-6)

    dark_frac20 = feats[:, _SAMPLE_FEAT_DARK_FRAC20_SLICE]
    ref_dark_frac20 = ref_mean[:, _SAMPLE_FEAT_DARK_FRAC20_SLICE]
    ref_dark_frac20_std = ref_std[:, _SAMPLE_FEAT_DARK_FRAC20_SLICE].clamp_min(1e-6)

    dark_frac30 = feats[:, _SAMPLE_FEAT_DARK_FRAC30_SLICE]
    ref_dark_frac30 = ref_mean[:, _SAMPLE_FEAT_DARK_FRAC30_SLICE]
    ref_dark_frac30_std = ref_std[:, _SAMPLE_FEAT_DARK_FRAC30_SLICE].clamp_min(1e-6)

    center_dark_frac20 = feats[:, _SAMPLE_FEAT_CENTER_DARK_FRAC20_SLICE]
    ref_center_dark_frac20 = ref_mean[:, _SAMPLE_FEAT_CENTER_DARK_FRAC20_SLICE]
    ref_center_dark_frac20_std = ref_std[:, _SAMPLE_FEAT_CENTER_DARK_FRAC20_SLICE].clamp_min(1e-6)

    center_dark_frac30 = feats[:, _SAMPLE_FEAT_CENTER_DARK_FRAC30_SLICE]
    ref_center_dark_frac30 = ref_mean[:, _SAMPLE_FEAT_CENTER_DARK_FRAC30_SLICE]
    ref_center_dark_frac30_std = ref_std[:, _SAMPLE_FEAT_CENTER_DARK_FRAC30_SLICE].clamp_min(1e-6)

    checks = torch.cat([
        brightness >= (ref_brightness - z * ref_brightness_std),
        center_brightness >= (ref_center_brightness - z * ref_center_std),
        luma_p10 >= (ref_luma_p10 - z * ref_luma_p10_std),
        center_luma_p10 >= (ref_center_luma_p10 - z * ref_center_luma_p10_std),
        dark_frac20 <= (ref_dark_frac20 + z * ref_dark_frac20_std),
        dark_frac30 <= (ref_dark_frac30 + z * ref_dark_frac30_std),
        center_dark_frac20 <= (ref_center_dark_frac20 + z * ref_center_dark_frac20_std),
        center_dark_frac30 <= (ref_center_dark_frac30 + z * ref_center_dark_frac30_std),
    ], dim=1)
    return checks.all(dim=1)


def _sample_feature_bright_rejection_mask(
    feats: torch.Tensor,
    ref_mean: torch.Tensor,
    ref_std: torch.Tensor,
    reject_bright_z: float,
) -> torch.Tensor:
    if float(reject_bright_z) <= 0.0:
        return torch.ones(feats.size(0), device=feats.device, dtype=torch.bool)

    z = float(reject_bright_z)
    brightness = feats[:, _SAMPLE_FEAT_BRIGHTNESS_SLICE]
    ref_brightness = ref_mean[:, _SAMPLE_FEAT_BRIGHTNESS_SLICE]
    ref_brightness_std = ref_std[:, _SAMPLE_FEAT_BRIGHTNESS_SLICE].clamp_min(1e-6)

    center_brightness = _sample_feature_center_brightness(feats)
    ref_center_brightness = _sample_feature_center_brightness(ref_mean)
    ref_center_std = torch.sqrt(
        ref_std[:, _SAMPLE_FEAT_BORDER_MEAN_SLICE].pow(2)
        + ref_std[:, _SAMPLE_FEAT_BORDER_CENTER_GAP_SLICE].pow(2)
    ).mean(dim=1, keepdim=True).clamp_min(1e-6)

    luma_p10 = feats[:, _SAMPLE_FEAT_LUMA_P10_SLICE]
    ref_luma_p10 = ref_mean[:, _SAMPLE_FEAT_LUMA_P10_SLICE]
    ref_luma_p10_std = ref_std[:, _SAMPLE_FEAT_LUMA_P10_SLICE].clamp_min(1e-6)

    luma_p25 = feats[:, _SAMPLE_FEAT_LUMA_P25_SLICE]
    ref_luma_p25 = ref_mean[:, _SAMPLE_FEAT_LUMA_P25_SLICE]
    ref_luma_p25_std = ref_std[:, _SAMPLE_FEAT_LUMA_P25_SLICE].clamp_min(1e-6)

    center_luma_p10 = feats[:, _SAMPLE_FEAT_CENTER_LUMA_P10_SLICE]
    ref_center_luma_p10 = ref_mean[:, _SAMPLE_FEAT_CENTER_LUMA_P10_SLICE]
    ref_center_luma_p10_std = ref_std[:, _SAMPLE_FEAT_CENTER_LUMA_P10_SLICE].clamp_min(1e-6)

    checks = torch.cat([
        brightness <= (ref_brightness + z * ref_brightness_std),
        center_brightness <= (ref_center_brightness + z * ref_center_std),
        luma_p10 <= (ref_luma_p10 + z * ref_luma_p10_std),
        luma_p25 <= (ref_luma_p25 + z * ref_luma_p25_std),
        center_luma_p10 <= (ref_center_luma_p10 + z * ref_center_luma_p10_std),
    ], dim=1)
    return checks.all(dim=1)


@torch.no_grad()
def _compute_stage2_sample_reference_stats(
    ae: LASER,
    tokens_flat: torch.Tensor,
    coeffs_flat: Optional[torch.Tensor],
    H: int,
    W: int,
    D: int,
    device: torch.device,
    max_items: int = 256,
    batch_size: int = 32,
) -> Optional[dict]:
    """Compute reference image-statistics from stage-1 codes for sample filtering."""
    keep = min(int(tokens_flat.size(0)), max(1, int(max_items)))
    if keep <= 0:
        return None

    was_training = ae.training
    ae.eval()
    feats_all = []
    for start in range(0, keep, max(1, int(batch_size))):
        end = min(keep, start + max(1, int(batch_size)))
        tok = tokens_flat[start:end].view(-1, H, W, D).to(device=device, dtype=torch.long)
        if coeffs_flat is not None:
            coeff = coeffs_flat[start:end].view(-1, H, W, D).to(device=device, dtype=torch.float32)
            imgs = ae.decode_from_atoms_and_coeffs(tok, coeff)
        else:
            imgs = ae.decode_from_tokens(tok)
        feats_all.append(_sample_quality_features(imgs))
    if was_training:
        ae.train()

    feats = torch.cat(feats_all, dim=0)
    return {
        "mean": feats.mean(dim=0, keepdim=True).cpu(),
        "std": feats.std(dim=0, keepdim=True).clamp_min(1e-6).cpu(),
    }


@torch.no_grad()
def _select_best_stage2_sample_indices(
    imgs: torch.Tensor,
    keep: int,
    reference_stats: Optional[dict],
    quality_weight: float = 0.15,
    brightness_weight: float = 0.0,
    overbright_weight: float = 0.0,
    reject_dark_z: float = 0.0,
    reject_bright_z: float = 0.0,
    selection_mode: str = "diverse",
    sort_by_quality: bool = False,
) -> torch.Tensor:
    """Select indices for a reranked subset of decoded stage-2 candidates."""
    keep = min(int(keep), int(imgs.size(0)))
    if keep <= 0 or imgs.size(0) <= keep:
        return torch.arange(keep, device=imgs.device, dtype=torch.long)
    selection_mode = str(selection_mode).strip().lower()
    if selection_mode not in {"diverse", "quality_only"}:
        raise ValueError(f"Unsupported selection_mode={selection_mode!r}")

    feats = _sample_quality_features(imgs)
    candidate_mask = None
    if reference_stats is None:
        quality = torch.zeros(imgs.size(0), device=imgs.device, dtype=feats.dtype)
    else:
        ref_mean = reference_stats["mean"].to(device=imgs.device, dtype=feats.dtype)
        ref_std = reference_stats["std"].to(device=imgs.device, dtype=feats.dtype)
        quality = (((feats - ref_mean) / ref_std) ** 2).mean(dim=1)
        if float(brightness_weight) > 0.0:
            quality = quality + float(brightness_weight) * _sample_feature_low_brightness_penalty(
                feats,
                ref_mean=ref_mean,
                ref_std=ref_std,
            )
        if float(overbright_weight) > 0.0:
            quality = quality + float(overbright_weight) * _sample_feature_high_brightness_penalty(
                feats,
                ref_mean=ref_mean,
                ref_std=ref_std,
            )
        if float(reject_dark_z) > 0.0 or float(reject_bright_z) > 0.0:
            candidate_mask = torch.ones(imgs.size(0), device=imgs.device, dtype=torch.bool)
            if float(reject_dark_z) > 0.0:
                candidate_mask = candidate_mask & _sample_feature_dark_rejection_mask(
                    feats,
                    ref_mean=ref_mean,
                    ref_std=ref_std,
                    reject_dark_z=float(reject_dark_z),
                )
            if float(reject_bright_z) > 0.0:
                candidate_mask = candidate_mask & _sample_feature_bright_rejection_mask(
                    feats,
                    ref_mean=ref_mean,
                    ref_std=ref_std,
                    reject_bright_z=float(reject_bright_z),
                )
            if int(candidate_mask.sum().item()) < keep:
                candidate_mask = None

    if candidate_mask is not None:
        candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(1)
    else:
        candidate_indices = torch.arange(imgs.size(0), device=imgs.device)
    candidate_quality = quality.index_select(0, candidate_indices)
    candidate_feats = feats.index_select(0, candidate_indices)

    if selection_mode == "quality_only":
        order = torch.argsort(candidate_quality, dim=0)
        selected_idx = candidate_indices.index_select(0, order[:keep])
        if sort_by_quality:
            return selected_idx
        return selected_idx

    pool_size = min(int(candidate_indices.numel()), max(keep, 4 * keep))
    if pool_size < int(candidate_indices.numel()):
        rel_pool = torch.topk(-candidate_quality, k=pool_size).indices
        pool = candidate_indices.index_select(0, rel_pool)
        pool_feats = candidate_feats.index_select(0, rel_pool)
        pool_quality = candidate_quality.index_select(0, rel_pool)
    else:
        pool = candidate_indices
        pool_feats = candidate_feats
        pool_quality = candidate_quality

    if pool.numel() <= keep:
        selected_idx = pool[:keep]
        if sort_by_quality:
            selected_quality = quality.index_select(0, selected_idx)
            order = torch.argsort(selected_quality, dim=0)
            selected_idx = selected_idx.index_select(0, order)
        return selected_idx

    quality_norm = pool_quality - pool_quality.min()
    quality_norm = quality_norm / quality_norm.max().clamp_min(1e-6)

    selected = [int(torch.argmin(pool_quality).item())]
    selected_mask = torch.zeros(pool_feats.size(0), device=pool_feats.device, dtype=torch.bool)
    selected_mask[selected[0]] = True
    min_feat_dist = torch.full(
        (pool_feats.size(0),),
        float("inf"),
        device=pool_feats.device,
        dtype=pool_feats.dtype,
    )

    while len(selected) < keep:
        last = selected[-1]
        feat_dist = ((pool_feats - pool_feats[last:last + 1]) ** 2).mean(dim=1)
        min_feat_dist = torch.minimum(min_feat_dist, feat_dist)
        diversity_norm = min_feat_dist / min_feat_dist.max().clamp_min(1e-6)
        score = diversity_norm - float(quality_weight) * quality_norm
        score = score.masked_fill(selected_mask, float('-inf'))
        next_idx = int(torch.argmax(score).item())
        selected.append(next_idx)
        selected_mask[next_idx] = True

    selected_idx = pool.index_select(0, torch.tensor(selected, device=pool.device, dtype=torch.long))
    if sort_by_quality:
        selected_quality = quality.index_select(0, selected_idx)
        order = torch.argsort(selected_quality, dim=0)
        selected_idx = selected_idx.index_select(0, order)
    return selected_idx


@torch.no_grad()
def _select_best_stage2_samples(
    imgs: torch.Tensor,
    keep: int,
    reference_stats: Optional[dict],
    quality_weight: float = 0.15,
    brightness_weight: float = 0.0,
    overbright_weight: float = 0.0,
    reject_dark_z: float = 0.0,
    reject_bright_z: float = 0.0,
    selection_mode: str = "diverse",
    sort_by_quality: bool = False,
) -> torch.Tensor:
    """Select a reranked subset of decoded stage-2 candidates."""
    selected_idx = _select_best_stage2_sample_indices(
        imgs,
        keep=keep,
        reference_stats=reference_stats,
        quality_weight=quality_weight,
        brightness_weight=brightness_weight,
        overbright_weight=overbright_weight,
        reject_dark_z=reject_dark_z,
        reject_bright_z=reject_bright_z,
        selection_mode=selection_mode,
        sort_by_quality=sort_by_quality,
    )
    return imgs.index_select(0, selected_idx)


@torch.no_grad()
def _decode_stage2_candidates_in_chunks(
    ae: LASER,
    tokens: torch.Tensor,
    coeffs: Optional[torch.Tensor] = None,
    decode_batch_size: int = 16,
) -> torch.Tensor:
    """Decode stage-2 preview candidates in small batches to limit GPU peak memory."""
    total = int(tokens.size(0))
    if total <= 0:
        raise ValueError("Expected at least one stage-2 sample candidate to decode")

    device = next(ae.parameters()).device
    decode_batch_size = max(1, int(decode_batch_size))
    while True:
        imgs_all = []
        tok = None
        coeff = None
        imgs = None
        try:
            for start in range(0, total, decode_batch_size):
                end = min(total, start + decode_batch_size)
                tok = tokens[start:end].to(device=device, dtype=torch.long)
                if coeffs is not None:
                    coeff = coeffs[start:end].to(device=device, dtype=torch.float32)
                    imgs = ae.decode_from_atoms_and_coeffs(tok, coeff)
                else:
                    imgs = ae.decode_from_tokens(tok)
                imgs_all.append(imgs.cpu())
                del tok
                tok = None
                if coeff is not None:
                    del coeff
                    coeff = None
                del imgs
                imgs = None
            return torch.cat(imgs_all, dim=0)
        except torch.OutOfMemoryError:
            if imgs is not None:
                del imgs
                imgs = None
            if coeff is not None:
                del coeff
                coeff = None
            if tok is not None:
                del tok
                tok = None
            for cached_imgs in imgs_all:
                del cached_imgs
            imgs_all.clear()
            if device.type != "cuda" or decode_batch_size <= 1:
                raise
            next_decode_batch_size = max(1, decode_batch_size // 2)
            if next_decode_batch_size == decode_batch_size:
                raise
            torch.cuda.empty_cache()
            print(
                "[Stage2] CUDA OOM while decoding preview candidates "
                f"with decode_batch_size={decode_batch_size}; retrying with "
                f"decode_batch_size={next_decode_batch_size}"
            )
            decode_batch_size = next_decode_batch_size


def _reconstruct_stage2_sparse_latent(
    ae: LASER,
    atoms: torch.Tensor,
    coeffs: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct stage-2 sparse latents, including patch-based bottlenecks.

    This helper must stay differentiable for recon-based stage-2 losses, so the
    caller is responsible for wrapping target-only paths in ``torch.no_grad()``.
    """
    ae_module = _unwrap_module(ae)
    coeffs = ae_module.clamp_sparse_coeffs(coeffs)
    if isinstance(ae_module.bottleneck, PatchDictionaryLearningTokenized):
        latent_hw = ae_module._resolve_patch_latent_hw()
        return ae_module.bottleneck._reconstruct_sparse(
            atoms,
            coeffs,
            int(latent_hw[0]),
            int(latent_hw[1]),
        )
    return ae_module.bottleneck._reconstruct_sparse(atoms, coeffs)


def _batch_psnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Average PSNR over a batch of images in [0, 1]."""
    mse = F.mse_loss(x, y, reduction="none").mean(dim=(1, 2, 3))
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10)).mean()


def _batch_ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Average SSIM over a batch of images in [0, 1] using a Gaussian window."""
    # SSIM is a logging/eval metric; run it in fp32 so AMP bf16/fp16 activations
    # do not trip conv2d dtype mismatches against the Gaussian kernel.
    x = x.to(dtype=torch.float32)
    y = y.to(dtype=torch.float32)
    _, channels, height, width = x.shape
    window_size = min(11, height, width)
    if window_size % 2 == 0:
        window_size -= 1
    window_size = max(1, window_size)
    radius = window_size // 2

    coords = torch.arange(window_size, device=x.device, dtype=x.dtype) - radius
    kernel_1d = torch.exp(-(coords ** 2) / (2 * (1.5 ** 2)))
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-12)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()

    mu_x = F.conv2d(x, kernel, padding=radius, groups=channels)
    mu_y = F.conv2d(y, kernel, padding=radius, groups=channels)
    mu_x_sq = mu_x.square()
    mu_y_sq = mu_y.square()
    mu_xy = mu_x * mu_y

    sigma_x = F.conv2d(x * x, kernel, padding=radius, groups=channels) - mu_x_sq
    sigma_y = F.conv2d(y * y, kernel, padding=radius, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(x * y, kernel, padding=radius, groups=channels) - mu_xy

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x_sq + mu_y_sq + c1) * (sigma_x + sigma_y + c2)
    ssim_map = num / den.clamp_min(1e-10)
    return ssim_map.mean(dim=(1, 2, 3)).mean()


def _get_rfid_model(device: torch.device) -> nn.Module:
    """Build and cache the fallback Inception-V3 feature extractor used for reconstruction FID."""
    global _RFID_MODEL, _RFID_MODEL_DEVICE
    device_key = str(device)
    if _RFID_MODEL is None or _RFID_MODEL_DEVICE != device_key:
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        model.fc = nn.Identity()
        model.eval().to(device)
        _RFID_MODEL = model
        _RFID_MODEL_DEVICE = device_key
    return _RFID_MODEL


def _get_rfid_metric(device: torch.device, feature: int = 64):
    """Build and cache the canonical torchmetrics FID metric when available."""
    global _RFID_METRIC, _RFID_METRIC_DEVICE
    if FrechetInceptionDistance is None:
        return None
    device_key = f"{device}:{int(feature)}"
    if _RFID_METRIC is None or _RFID_METRIC_DEVICE != device_key:
        _RFID_METRIC = FrechetInceptionDistance(
            feature=int(feature),
            sync_on_compute=False,
            normalize=False,
        ).to(device)
        _RFID_METRIC_DEVICE = device_key
    _RFID_METRIC.reset()
    return _RFID_METRIC


@torch.no_grad()
def _extract_rfid_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Extract pooled Inception-V3 features from images in [-1, 1]."""
    x = _to_unit_range(x)
    if x.size(-2) != 299 or x.size(-1) != 299:
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    mean = x.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = x.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    x = (x - mean) / std
    feats = model(x)
    if isinstance(feats, tuple):
        feats = feats[0]
    return feats.float()


def _to_uint8_images(x: torch.Tensor) -> torch.Tensor:
    """Convert images from [-1, 1] to uint8 RGB tensors for FID."""
    return ((x.detach().cpu().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)


def _frechet_distance_from_features(real_feats: torch.Tensor, fake_feats: torch.Tensor) -> float:
    """Compute Fréchet distance between two feature clouds."""
    real_np = real_feats.detach().cpu().numpy().astype(np.float64, copy=False)
    fake_np = fake_feats.detach().cpu().numpy().astype(np.float64, copy=False)
    mu_real = np.mean(real_np, axis=0)
    mu_fake = np.mean(fake_np, axis=0)
    sigma_real = np.cov(real_np, rowvar=False)
    sigma_fake = np.cov(fake_np, rowvar=False)

    cov_prod = sigma_real @ sigma_fake
    cov_mean = sqrtm(cov_prod)
    if not np.isfinite(cov_mean).all():
        eps = 1e-6
        eye = np.eye(sigma_real.shape[0], dtype=np.float64)
        cov_mean = sqrtm((sigma_real + eps * eye) @ (sigma_fake + eps * eye))
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real

    diff = mu_real - mu_fake
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2.0 * cov_mean)
    return float(max(fid, 0.0))


@torch.no_grad()
def _compute_reconstruction_fid(
    ae: LASER,
    loader: Optional[DataLoader],
    device: torch.device,
    max_items: int,
) -> Optional[float]:
    """Compute reconstruction FID between validation images and their reconstructions."""
    if loader is None or max_items <= 1:
        return None

    ae.eval()
    fid_metric = _get_rfid_metric(device, feature=64)
    if fid_metric is not None:
        seen = 0
        for x, _ in tqdm(loader, desc="[Stage1] compute rFID", leave=False, dynamic_ncols=True):
            x = x.to(device)
            recon, _, _ = ae(x)
            keep = min(x.size(0), max_items - seen)
            if keep <= 0:
                break
            fid_metric.update(_to_uint8_images(x[:keep]).to(device), real=True)
            fid_metric.update(_to_uint8_images(recon[:keep]).to(device), real=False)
            seen += keep
            if seen >= max_items:
                break
        if seen <= 1:
            return None
        return float(fid_metric.compute().detach().cpu().item())

    model = _get_rfid_model(device)
    real_feats = []
    fake_feats = []
    seen = 0

    for x, _ in tqdm(loader, desc="[Stage1] compute rFID", leave=False, dynamic_ncols=True):
        x = x.to(device)
        recon, _, _ = ae(x)
        keep = min(x.size(0), max_items - seen)
        if keep <= 0:
            break
        real_feats.append(_extract_rfid_features(model, x[:keep]).cpu())
        fake_feats.append(_extract_rfid_features(model, recon[:keep]).cpu())
        seen += keep
        if seen >= max_items:
            break

    if seen <= 1:
        return None
    return _frechet_distance_from_features(torch.cat(real_feats, dim=0), torch.cat(fake_feats, dim=0))


@torch.no_grad()
def _compute_image_fid_from_batches(
    real_batches: Iterable[torch.Tensor],
    fake_batches: Iterable[torch.Tensor],
    device: torch.device,
    max_items: int,
) -> Optional[float]:
    """Compute FID between two image streams in [-1, 1]."""
    if max_items <= 1:
        return None

    fid_metric = _get_rfid_metric(device, feature=64)
    if fid_metric is not None:
        seen_real = 0
        for real_batch in real_batches:
            real_batch = real_batch.to(device)
            keep = min(int(real_batch.size(0)), max_items - seen_real)
            if keep <= 0:
                break
            fid_metric.update(_to_uint8_images(real_batch[:keep]).to(device), real=True)
            seen_real += keep
            if seen_real >= max_items:
                break

        seen_fake = 0
        for fake_batch in fake_batches:
            fake_batch = fake_batch.to(device)
            keep = min(int(fake_batch.size(0)), max_items - seen_fake)
            if keep <= 0:
                break
            fid_metric.update(_to_uint8_images(fake_batch[:keep]).to(device), real=False)
            seen_fake += keep
            if seen_fake >= max_items:
                break

        if min(seen_real, seen_fake) <= 1:
            return None
        return float(fid_metric.compute().detach().cpu().item())

    model = _get_rfid_model(device)
    real_feats = []
    fake_feats = []
    seen_real = 0
    seen_fake = 0

    for real_batch in real_batches:
        real_batch = real_batch.to(device)
        keep = min(int(real_batch.size(0)), max_items - seen_real)
        if keep <= 0:
            break
        real_feats.append(_extract_rfid_features(model, real_batch[:keep]).cpu())
        seen_real += keep
        if seen_real >= max_items:
            break

    for fake_batch in fake_batches:
        fake_batch = fake_batch.to(device)
        keep = min(int(fake_batch.size(0)), max_items - seen_fake)
        if keep <= 0:
            break
        fake_feats.append(_extract_rfid_features(model, fake_batch[:keep]).cpu())
        seen_fake += keep
        if seen_fake >= max_items:
            break

    if min(seen_real, seen_fake) <= 1:
        return None
    return _frechet_distance_from_features(torch.cat(real_feats, dim=0), torch.cat(fake_feats, dim=0))


@torch.no_grad()
def _compute_stage2_sample_fid(
    transformer: nn.Module,
    ae: LASER,
    loader: Optional[DataLoader],
    device: torch.device,
    max_items: int,
    H: int,
    W: int,
    D: int,
    sample_batch_size: int = 32,
    sample_candidate_factor: int = 4,
    sample_temperature: float = 1.0,
    sample_top_k: Optional[int] = None,
    sample_coeff_temperature: Optional[float] = None,
    sample_coeff_mode: str = "gaussian",
    sample_reference_stats: Optional[dict] = None,
    sample_selection_quality_weight: float = 1.0,
    sample_brightness_weight: float = 1.0,
    sample_overbright_weight: float = 1.0,
    sample_reject_dark_z: float = 1.5,
    sample_reject_bright_z: float = 1.5,
    sample_selection_mode: str = "quality_only",
    sample_sort_by_quality: bool = True,
    sample_image_size: Optional[int] = None,
) -> Optional[float]:
    """Compute FID between validation images and unconditional stage-2 samples."""
    if loader is None or max_items <= 1:
        return None

    real_valued = bool(transformer.real_valued_coeffs)
    sample_batch_size = max(1, int(sample_batch_size))
    sample_candidate_factor = max(1, int(sample_candidate_factor))
    base_candidate_batch_size = sample_batch_size
    if sample_reference_stats is not None:
        base_candidate_batch_size = max(sample_batch_size, sample_batch_size * sample_candidate_factor)
    decode_batch_size = max(1, min(8, sample_batch_size))

    transformer_was_training = transformer.training
    ae_was_training = ae.training
    ae_device = next(ae.parameters()).device
    moved_ae = ae_device != device

    transformer.eval()
    ae.eval()
    ae.requires_grad_(False)
    if moved_ae:
        ae.to(device)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def _iter_real_batches():
        seen = 0
        for x, _ in tqdm(loader, desc="[Stage2] compute FID real", leave=False, dynamic_ncols=True):
            keep = min(int(x.size(0)), max_items - seen)
            if keep <= 0:
                break
            yield x[:keep]
            seen += keep
            if seen >= max_items:
                break

    def _iter_fake_batches():
        produced = 0
        while produced < max_items:
            keep = min(sample_batch_size, max_items - produced)
            candidate_batch_size = max(keep, base_candidate_batch_size)
            if real_valued:
                atoms_gen, coeffs_gen = transformer.generate(
                    batch_size=candidate_batch_size,
                    temperature=sample_temperature,
                    top_k=sample_top_k,
                    coeff_temperature=sample_coeff_temperature,
                    coeff_sample_mode=sample_coeff_mode,
                    show_progress=False,
                )
                coeffs_gen = _clamp_generated_sparse_coeffs_for_decode(ae, coeffs_gen)
                atoms_gen = atoms_gen.view(-1, H, W, D)
                coeffs_gen = coeffs_gen.view(-1, H, W, D)
                imgs = _decode_stage2_candidates_in_chunks(
                    ae,
                    atoms_gen,
                    coeffs=coeffs_gen,
                    decode_batch_size=decode_batch_size,
                )
            else:
                tokens_gen = transformer.generate(
                    batch_size=candidate_batch_size,
                    temperature=sample_temperature,
                    top_k=sample_top_k,
                    show_progress=False,
                ).view(-1, H, W, D)
                imgs = _decode_stage2_candidates_in_chunks(
                    ae,
                    tokens_gen,
                    decode_batch_size=decode_batch_size,
                )
            imgs = _select_best_stage2_samples(
                imgs,
                keep=keep,
                reference_stats=sample_reference_stats,
                quality_weight=sample_selection_quality_weight,
                brightness_weight=sample_brightness_weight,
                overbright_weight=sample_overbright_weight,
                reject_dark_z=sample_reject_dark_z,
                reject_bright_z=sample_reject_bright_z,
                selection_mode=sample_selection_mode,
                sort_by_quality=sample_sort_by_quality,
            )
            if sample_image_size is not None and int(sample_image_size) > 0:
                if imgs.size(-2) != int(sample_image_size) or imgs.size(-1) != int(sample_image_size):
                    imgs = F.interpolate(
                        imgs,
                        size=(int(sample_image_size), int(sample_image_size)),
                        mode="bilinear",
                        align_corners=False,
                    )
            produced += int(imgs.size(0))
            yield imgs

    try:
        return _compute_image_fid_from_batches(
            real_batches=_iter_real_batches(),
            fake_batches=_iter_fake_batches(),
            device=device,
            max_items=max_items,
        )
    finally:
        if moved_ae:
            ae.to(ae_device)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if transformer_was_training:
            transformer.train()
        if ae_was_training:
            ae.train()


def _maybe_load_wandb_api_key() -> Optional[str]:
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        return api_key

    candidates = []
    key_file_env = os.environ.get("WANDB_API_KEY_FILE")
    if key_file_env:
        candidates.append(Path(key_file_env).expanduser())

    user_name = os.environ.get("USER") or os.environ.get("LOGNAME")
    if user_name:
        candidates.append(Path("/scratch") / user_name / ".secrets" / "wandb_api_key")
    try:
        home_user = Path.home().name
    except Exception:
        home_user = None
    if home_user and home_user != user_name:
        candidates.append(Path("/scratch") / home_user / ".secrets" / "wandb_api_key")

    for candidate in candidates:
        try:
            if not candidate.is_file():
                continue
            api_key = candidate.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if api_key:
            os.environ["WANDB_API_KEY"] = api_key
            return api_key
    return None


def _init_wandb(args) -> Optional[object]:
    global _WANDB_DISABLE_REASON, _WANDB_LOG_STEP
    if not getattr(args, "wandb", True):
        return None
    if wandb is None:
        print("[W&B] wandb is not installed; continuing without logging.")
        return None
    api_key = _maybe_load_wandb_api_key()
    if api_key:
        try:
            wandb.login(key=api_key, relogin=True, verify=True)
        except Exception as exc:
            print(f"[W&B] login failed ({type(exc).__name__}: {exc}); continuing to init with existing env.")
    try:
        run_id = _resolve_run_id(getattr(args, "run_id", None), getattr(args, "launch_timestamp", None))
        init_kwargs = dict(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            dir=args.wandb_dir,
            mode=args.wandb_mode,
            config=dict(vars(args)),
        )
        if run_id:
            init_kwargs["id"] = run_id
            init_kwargs["resume"] = "allow"
        run = wandb.init(
            **init_kwargs,
        )
        _WANDB_LOG_STEP = 0
        _WANDB_DISABLE_REASON = None
        run.define_metric("stage1/step")
        run.define_metric("stage1/*", step_metric="stage1/step")
        run.define_metric("stage2/step")
        run.define_metric("stage2/*", step_metric="stage2/step")
        return run
    except Exception as exc:
        print(f"[W&B] init failed ({exc}); continuing without logging.")
        return None


def _next_wandb_log_step() -> int:
    global _WANDB_LOG_STEP
    step = int(_WANDB_LOG_STEP)
    _WANDB_LOG_STEP += 1
    return step


def _disable_wandb_logging(run: Optional[object], context: str, exc: Exception) -> None:
    global _WANDB_DISABLE_REASON
    if _WANDB_DISABLE_REASON is not None:
        return
    _WANDB_DISABLE_REASON = f"{context}: {type(exc).__name__}: {exc}"
    print(f"[W&B] disabling logging after {context} failed ({type(exc).__name__}: {exc})")
    if run is None:
        return
    try:
        run.summary["wandb_disabled_reason"] = _WANDB_DISABLE_REASON
    except Exception:
        pass


def _finish_wandb(run: Optional[object]) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception as exc:
        print(f"[W&B] finish failed ({type(exc).__name__}: {exc})")


def _log_wandb(
    run: Optional[object],
    data: dict,
    step_metric: Optional[str] = None,
    step_value: Optional[int] = None,
):
    if run is None or _WANDB_DISABLE_REASON is not None:
        return
    payload = dict(data)
    if step_metric is not None and step_value is not None:
        payload[step_metric] = int(step_value)
    try:
        run.log(payload, step=_next_wandb_log_step())
    except Exception as exc:
        _disable_wandb_logging(run, "scalar log", exc)


def _log_wandb_image(
    run: Optional[object],
    key: str,
    x: torch.Tensor,
    step_metric: Optional[str] = None,
    step_value: Optional[int] = None,
    caption: Optional[str] = None,
):
    if run is None or wandb is None or _WANDB_DISABLE_REASON is not None:
        return
    grid = _make_image_grid(x)
    image = grid.permute(1, 2, 0).mul(255).clamp(0, 255).byte().numpy()
    payload = {key: wandb.Image(image, caption=caption)}
    if step_metric is not None and step_value is not None:
        payload[step_metric] = int(step_value)
    try:
        run.log(payload, step=_next_wandb_log_step())
    except Exception as exc:
        _disable_wandb_logging(run, f"image log {key}", exc)


def _stage1_lr_scale(
    epoch: float,
    max_epochs: int,
    schedule: str,
    warmup_epochs: int,
    min_lr_ratio: float,
) -> float:
    if str(schedule) != "cosine":
        return 1.0

    epoch = float(epoch)
    max_epochs = max(1, int(max_epochs))
    warmup_epochs = min(max(0, int(warmup_epochs)), max_epochs - 1)
    min_lr_ratio = float(max(0.0, min(float(min_lr_ratio), 1.0)))
    warmup_start_ratio = 0.1

    if warmup_epochs > 0 and epoch < warmup_epochs:
        progress = max(0.0, min(epoch / float(warmup_epochs), 1.0))
        return warmup_start_ratio + (1.0 - warmup_start_ratio) * progress

    if max_epochs <= warmup_epochs + 1:
        return 1.0

    decay_progress = float(epoch - warmup_epochs) / float(max_epochs - warmup_epochs)
    decay_progress = max(0.0, min(decay_progress, 1.0))
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def train_stage1_ae(
    ae: LASER,
    train_loader: DataLoader,
    val_loader: DataLoader,
    rfid_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int,
    lr: float,
    bottleneck_weight: float,
    grad_clip: float,
    stage1_amp: bool,
    stage1_amp_dtype: str,
    out_dir: str,
    rfid_num_samples: int = 0,
    lr_schedule: str = "cosine",
    warmup_epochs: int = 1,
    min_lr_ratio: float = 0.1,
    dict_optimizer: str = "shared_adam",
    dict_lr_multiplier: float = 1.0,
    dict_lr_schedule: str = "cosine",
    dict_warmup_epochs: int = 1,
    dict_min_lr_ratio: float = 0.05,
    dict_grad_clip: float = 0.1,
    dict_max_update_norm: float = 0.0,
    loss_spike_skip_ratio: float = 0.0,
    loss_ema_beta: float = 0.98,
    bottleneck_weight_start: float = 1.0,
    bottleneck_warmup_epochs: int = 0,
    dict_loss_weight: float = float("nan"),
    dict_loss_weight_start: float = float("nan"),
    dict_loss_warmup_epochs: int = 0,
    commitment_loss_weight: float = float("nan"),
    commitment_loss_weight_start: float = float("nan"),
    commitment_loss_warmup_epochs: int = 0,
    coherence_weight: float = 0.0,
    coherence_weight_start: float = float("nan"),
    coherence_warmup_epochs: int = 0,
    coherence_margin: float = 0.0,
    checkpoint_every_steps: int = 0,
    train_sampler: Optional[DistributedSampler] = None,
    is_main_process: bool = True,
    wandb_run: Optional[object] = None,
    artifact_tag: Optional[str] = None,
):
    """Train stage 1 with optional DDP and rank-0-only artifacts."""
    ae_module = _unwrap_module(ae)
    dict_param = ae_module.bottleneck.dictionary
    dict_eps = float(ae_module.bottleneck.epsilon)
    _normalize_dictionary_in_place(dict_param, eps=dict_eps)
    dict_optimizer = str(dict_optimizer)
    if dict_optimizer not in {"shared_adam", "separate_sgd"}:
        raise ValueError(
            f"Unsupported stage-1 dictionary optimizer mode: {dict_optimizer!r}"
        )
    non_dict_params = [p for p in ae.parameters() if p is not dict_param]
    use_separate_dict_opt = dict_optimizer == "separate_sgd"
    if use_separate_dict_opt:
        opt = torch.optim.Adam(non_dict_params, lr=lr)
        # Separate SGD remains available for experiments, but it is more sensitive because
        # OMP support selection changes discontinuously as the dictionary moves.
        dict_opt = torch.optim.SGD([dict_param], lr=lr * float(dict_lr_multiplier))
    else:
        # Keep the default Adam path, but give the dictionary its own param group so the
        # dictionary-specific LR schedule actually takes effect and can be damped separately.
        opt = torch.optim.Adam(
            [
                {"params": non_dict_params, "lr": lr},
                {"params": [dict_param], "lr": lr * float(dict_lr_multiplier)},
            ]
        )
        dict_opt = None
    best_val_recon = float("inf")
    global_step = 0
    rfid_warned = False
    loss_ema: Optional[float] = None
    commitment_beta = float(getattr(ae_module.bottleneck, "commitment_cost", 0.25))
    amp_dtype_name = str(stage1_amp_dtype).strip().lower()
    if amp_dtype_name not in {"auto", "float16", "bfloat16"}:
        raise ValueError(f"Unsupported stage1_amp_dtype: {stage1_amp_dtype!r}")
    amp_enabled = bool(stage1_amp) and device.type == "cuda"
    if amp_dtype_name == "auto":
        amp_dtype = torch.bfloat16 if (amp_enabled and torch.cuda.is_bf16_supported()) else torch.float16
    elif amp_dtype_name == "bfloat16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
    scaler_enabled = amp_enabled and amp_dtype == torch.float16
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(device="cuda", enabled=scaler_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    if amp_enabled and is_main_process:
        amp_label = "bfloat16" if amp_dtype == torch.bfloat16 else "float16"
        print(f"[Stage1] using AMP ({amp_label})")
    resume_checkpoint_path = Path(out_dir) / "ae_resume_latest.pt"

    def _stage1_loss_weight(
        epoch_progress: float,
        final_weight: float,
        start_weight: float,
        warmup_epochs: int,
        fallback_weight: float,
    ) -> float:
        if not math.isfinite(final_weight):
            return float(fallback_weight)
        resolved_start = float(final_weight) if not math.isfinite(start_weight) else float(start_weight)
        if warmup_epochs > 0:
            progress = max(0.0, min(epoch_progress / float(max(1, warmup_epochs)), 1.0))
            return resolved_start + (float(final_weight) - resolved_start) * progress
        return float(final_weight)

    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        ae.train()
        pbar = tqdm(train_loader, desc=f"[Stage1] epoch {epoch}/{epochs}", disable=(not is_main_process))
        running = 0.0
        num_train_steps = max(1, len(train_loader))
        current_lr = float(lr)
        for step_idx, (x, _) in enumerate(pbar):
            epoch_progress = float(epoch - 1) + float(step_idx + 1) / float(num_train_steps)
            lr_scale = _stage1_lr_scale(
                epoch=epoch_progress,
                max_epochs=epochs,
                schedule=lr_schedule,
                warmup_epochs=warmup_epochs,
                min_lr_ratio=min_lr_ratio,
            )
            current_lr = float(lr) * float(lr_scale)
            if bottleneck_warmup_epochs > 0:
                bw_progress = max(0.0, min(epoch_progress / float(max(1, bottleneck_warmup_epochs)), 1.0))
                current_bottleneck_weight = float(bottleneck_weight_start) + (float(bottleneck_weight) - float(bottleneck_weight_start)) * bw_progress
            else:
                current_bottleneck_weight = float(bottleneck_weight)
            current_dict_loss_weight = _stage1_loss_weight(
                epoch_progress=epoch_progress,
                final_weight=dict_loss_weight,
                start_weight=dict_loss_weight_start,
                warmup_epochs=dict_loss_warmup_epochs,
                fallback_weight=current_bottleneck_weight,
            )
            current_commitment_loss_weight = _stage1_loss_weight(
                epoch_progress=epoch_progress,
                final_weight=commitment_loss_weight,
                start_weight=commitment_loss_weight_start,
                warmup_epochs=commitment_loss_warmup_epochs,
                fallback_weight=current_bottleneck_weight * commitment_beta,
            )
            current_coherence_weight = _stage1_loss_weight(
                epoch_progress=epoch_progress,
                final_weight=coherence_weight,
                start_weight=coherence_weight_start,
                warmup_epochs=coherence_warmup_epochs,
                fallback_weight=0.0,
            )
            dict_lr_scale = _stage1_lr_scale(
                epoch=epoch_progress,
                max_epochs=epochs,
                schedule=dict_lr_schedule,
                warmup_epochs=dict_warmup_epochs,
                min_lr_ratio=dict_min_lr_ratio,
            )
            current_dict_lr = float(lr) * float(dict_lr_multiplier) * float(dict_lr_scale)
            if dict_opt is not None:
                for param_group in opt.param_groups:
                    param_group["lr"] = current_lr
                for param_group in dict_opt.param_groups:
                    param_group["lr"] = current_dict_lr
            else:
                opt.param_groups[0]["lr"] = current_lr
                opt.param_groups[1]["lr"] = current_dict_lr
            x = _nan_to_num_tensor(x.to(device, non_blocking=True))
            autocast_ctx = (
                torch.autocast(device_type=device.type, dtype=amp_dtype)
                if amp_enabled
                else nullcontext()
            )
            with autocast_ctx:
                recon, b_loss, _ = ae(x)
                recon = _nan_to_num_tensor(recon)
                b_loss = _nan_to_num_tensor(b_loss)
                dict_loss = getattr(ae_module.bottleneck, "_last_dl_latent_loss", None)
                commitment_loss = getattr(ae_module.bottleneck, "_last_e_latent_loss", None)
                extra_bottleneck_loss = getattr(ae_module.bottleneck, "_last_extra_bottleneck_loss", None)
                coeff_kl_loss = getattr(ae_module.bottleneck, "_last_coeff_kl_loss", None)
                coeff_posterior_std = getattr(ae_module.bottleneck, "_last_coeff_posterior_std", None)
                coeff_prior_std = getattr(ae_module.bottleneck, "_last_coeff_prior_std", None)
                if dict_loss is None or commitment_loss is None:
                    latent_mse = b_loss / max(1.0 + commitment_beta, 1e-8)
                    dict_loss = latent_mse
                    commitment_loss = latent_mse
                if extra_bottleneck_loss is None:
                    extra_bottleneck_loss = x.new_zeros(())
                if coeff_kl_loss is None:
                    coeff_kl_loss = x.new_zeros(())
                if coeff_posterior_std is None:
                    coeff_posterior_std = x.new_zeros(())
                if coeff_prior_std is None:
                    coeff_prior_std = x.new_zeros(())
                dict_loss = _nan_to_num_tensor(dict_loss)
                commitment_loss = _nan_to_num_tensor(commitment_loss)
                extra_bottleneck_loss = _nan_to_num_tensor(extra_bottleneck_loss)
                coeff_kl_loss = _nan_to_num_tensor(coeff_kl_loss)
                coeff_posterior_std = _nan_to_num_tensor(coeff_posterior_std)
                coeff_prior_std = _nan_to_num_tensor(coeff_prior_std)
                recon_loss = F.mse_loss(recon, x)
                coherence_loss = _dictionary_coherence_penalty(
                    dict_param,
                    margin=coherence_margin,
                    eps=dict_eps,
                )
                weighted_dict_loss = float(current_dict_loss_weight) * dict_loss
                weighted_commitment_loss = float(current_commitment_loss_weight) * commitment_loss
                weighted_coherence_loss = float(current_coherence_weight) * coherence_loss
                loss = (
                    recon_loss
                    + weighted_dict_loss
                    + weighted_commitment_loss
                    + weighted_coherence_loss
                    + extra_bottleneck_loss
                )

            loss_finite_local = bool(torch.isfinite(loss).item())
            if dist.is_available() and dist.is_initialized():
                finite_flag = torch.tensor(1 if loss_finite_local else 0, device=device, dtype=torch.int32)
                dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
                loss_is_finite = bool(finite_flag.item())
            else:
                loss_is_finite = loss_finite_local

            loss_for_ema = _distributed_mean(loss.detach()) if loss_is_finite else None
            loss_value = float(loss_for_ema.item()) if loss_for_ema is not None else float('nan')

            should_skip_for_loss = not loss_is_finite
            spike_threshold = None if loss_ema is None else loss_ema * float(loss_spike_skip_ratio)
            if (
                not should_skip_for_loss
                and loss_ema is not None
                and loss_spike_skip_ratio > 0.0
                and global_step >= 10
                and loss_value > float(spike_threshold)
            ):
                should_skip_for_loss = True

            if should_skip_for_loss:
                if is_main_process:
                    if not loss_is_finite:
                        print(
                            f"[Stage1] Warning: non-finite loss detected across ranks at step {global_step + 1}, skipping update"
                        )
                    else:
                        print(
                            f"[Stage1] Warning: loss spike ({loss_value:.4f} > {float(spike_threshold):.4f}) "
                            f"at step {global_step + 1}, skipping update"
                        )
                if loss_is_finite:
                    capped_loss = loss_value if spike_threshold is None else min(loss_value, float(spike_threshold))
                    if loss_ema is None:
                        loss_ema = capped_loss
                    else:
                        loss_ema = float(loss_ema_beta) * loss_ema + (1.0 - float(loss_ema_beta)) * capped_loss
                global_step += 1
                continue

            if loss_ema is None:
                loss_ema = loss_value

            opt.zero_grad(set_to_none=True)
            if dict_opt is not None:
                dict_opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                if dict_opt is not None:
                    scaler.unscale_(dict_opt)
            else:
                loss.backward()
            dict_before_step = F.normalize(dict_param.detach(), p=2, dim=0, eps=dict_eps)
            if dict_param.grad is None:
                dict_grad_norm_raw = torch.zeros((), device=device)
                dict_grad_norm_preclip = torch.zeros((), device=device)
                dict_grad_norm_postclip = torch.zeros((), device=device)
            else:
                dict_grad_norm_raw = torch.linalg.vector_norm(dict_param.grad.detach())
                _project_dictionary_gradient_in_place(dict_param, eps=dict_eps)
                dict_grad_norm_preclip = torch.linalg.vector_norm(dict_param.grad.detach())
                dict_grad_norm_postclip = dict_grad_norm_preclip
            found_nonfinite_grad = False
            for param in ae.parameters():
                if param.grad is None:
                    continue
                if not torch.isfinite(param.grad).all():
                    found_nonfinite_grad = True
                    param.grad.copy_(_nan_to_num_tensor(param.grad))
            if use_separate_dict_opt:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(non_dict_params, grad_clip)
                if dict_grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_([dict_param], dict_grad_clip)
                    if dict_param.grad is None:
                        dict_grad_norm_postclip = torch.zeros((), device=device)
                    else:
                        dict_grad_norm_postclip = torch.linalg.vector_norm(dict_param.grad.detach())
            else:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(non_dict_params, grad_clip)
                if dict_grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_([dict_param], dict_grad_clip)
                if dict_param.grad is None:
                    dict_grad_norm_postclip = torch.zeros((), device=device)
                else:
                    dict_grad_norm_postclip = torch.linalg.vector_norm(dict_param.grad.detach())
            if found_nonfinite_grad:
                if is_main_process:
                    print(f"[Stage1] Warning: non-finite gradients at step {global_step + 1}, sanitized before optimizer step")
                if dict_param.grad is not None:
                    dict_grad_norm_postclip = torch.linalg.vector_norm(dict_param.grad.detach())
            if scaler.is_enabled():
                scaler.step(opt)
                if dict_opt is not None:
                    scaler.step(dict_opt)
                scaler.update()
            else:
                opt.step()
                if dict_opt is not None:
                    dict_opt.step()

            # Keep dictionary atoms normalized after each optimizer step.
            _normalize_dictionary_in_place(ae_module.bottleneck.dictionary, eps=dict_eps)
            dict_update_norm = torch.linalg.vector_norm(
                F.normalize(dict_param.detach(), p=2, dim=0, eps=dict_eps) - dict_before_step
            )
            reject_dict_update = False
            rejected_update_norm = float(dict_update_norm.item())
            if dict_max_update_norm > 0.0:
                reject_dict_update = rejected_update_norm > float(dict_max_update_norm)
                if dist.is_available() and dist.is_initialized():
                    reject_flag = torch.tensor(1 if reject_dict_update else 0, device=device, dtype=torch.int32)
                    dist.all_reduce(reject_flag, op=dist.ReduceOp.MAX)
                    reject_dict_update = bool(reject_flag.item())
                    rejected_norm_tensor = torch.tensor(rejected_update_norm, device=device, dtype=torch.float32)
                    dist.all_reduce(rejected_norm_tensor, op=dist.ReduceOp.MAX)
                    rejected_update_norm = float(rejected_norm_tensor.item())
            if reject_dict_update:
                with torch.no_grad():
                    dict_param.copy_(dict_before_step)
                dict_update_norm = torch.zeros((), device=device)
                if is_main_process:
                    print(
                        f"[Stage1] Warning: dictionary update norm {rejected_update_norm:.4f} exceeded "
                        f"limit {float(dict_max_update_norm):.4f} at step {global_step + 1}, restored previous dictionary"
                    )
            dict_coherence, dict_coherence_mean_abs, dict_coherence_rms = _dictionary_coherence_stats(
                ae_module.bottleneck.dictionary,
                eps=dict_eps,
            )

            loss_log = _distributed_mean(loss)
            recon_log = _distributed_mean(recon_loss)
            b_log = _distributed_mean(b_loss)
            dict_loss_log = _distributed_mean(dict_loss)
            commitment_loss_log = _distributed_mean(commitment_loss)
            coherence_loss_log = _distributed_mean(coherence_loss)
            weighted_dict_log = _distributed_mean(weighted_dict_loss)
            weighted_commitment_log = _distributed_mean(weighted_commitment_loss)
            weighted_coherence_log = _distributed_mean(weighted_coherence_loss)
            extra_bottleneck_log = _distributed_mean(extra_bottleneck_loss)
            coeff_kl_log = _distributed_mean(coeff_kl_loss)
            coeff_posterior_std_log = _distributed_mean(coeff_posterior_std)
            coeff_prior_std_log = _distributed_mean(coeff_prior_std)
            latent_mse_log = _distributed_mean(0.5 * (dict_loss + commitment_loss))
            dict_grad_raw_log = _distributed_mean(dict_grad_norm_raw)
            dict_grad_preclip_log = _distributed_mean(dict_grad_norm_preclip)
            dict_grad_postclip_log = _distributed_mean(dict_grad_norm_postclip)
            dict_update_log = _distributed_mean(dict_update_norm)
            dict_coherence_log = _distributed_mean(dict_coherence)
            dict_coherence_mean_abs_log = _distributed_mean(dict_coherence_mean_abs)
            dict_coherence_rms_log = _distributed_mean(dict_coherence_rms)
            running += float(loss_log.item())
            if loss_ema is None:
                loss_ema = float(loss_log.item())
            else:
                loss_ema = float(loss_ema_beta) * loss_ema + (1.0 - float(loss_ema_beta)) * float(loss_log.item())
            global_step += 1
            if is_main_process:
                pbar.set_postfix(
                    loss=float(loss_log.item()),
                    recon=float(recon_log.item()),
                    b=float(b_log.item()),
                )
                _log_wandb(
                    wandb_run,
                    {
                        "stage1/train_loss": float(loss_log.item()),
                        "stage1/recon_loss": float(recon_log.item()),
                        "stage1/bottleneck_loss": float(b_log.item()),
                        "stage1/dict_loss": float(dict_loss_log.item()),
                        "stage1/commitment_loss": float(commitment_loss_log.item()),
                        "stage1/coherence_loss": float(coherence_loss_log.item()),
                        "stage1/latent_mse": float(latent_mse_log.item()),
                        "stage1/bottleneck_weight": float(current_dict_loss_weight + current_commitment_loss_weight),
                        "stage1/dict_loss_weight": float(current_dict_loss_weight),
                        "stage1/commitment_loss_weight": float(current_commitment_loss_weight),
                        "stage1/coherence_weight": float(current_coherence_weight),
                        "stage1/weighted_dict_loss": float(weighted_dict_log.item()),
                        "stage1/weighted_commitment_loss": float(weighted_commitment_log.item()),
                        "stage1/weighted_coherence_loss": float(weighted_coherence_log.item()),
                        "stage1/extra_bottleneck_loss": float(extra_bottleneck_log.item()),
                        "stage1/coeff_kl_loss": float(coeff_kl_log.item()),
                        "stage1/coeff_posterior_std": float(coeff_posterior_std_log.item()),
                        "stage1/coeff_prior_std": float(coeff_prior_std_log.item()),
                        "stage1/variational_coeff_kl_weight": float(
                            getattr(ae_module.bottleneck, "variational_coeff_kl_weight", 0.0)
                        ),
                        "stage1/effective_commitment": float(current_commitment_loss_weight),
                        "stage1/dict_lr": float(current_dict_lr),
                        "stage1/dict_grad_norm_raw": float(dict_grad_raw_log.item()),
                        "stage1/dict_grad_norm_preclip": float(dict_grad_preclip_log.item()),
                        "stage1/dict_grad_norm_postclip": float(dict_grad_postclip_log.item()),
                        "stage1/dict_update_norm": float(dict_update_log.item()),
                        "stage1/dict_coherence": float(dict_coherence_log.item()),
                        "stage1/dict_coherence_mean_abs": float(dict_coherence_mean_abs_log.item()),
                        "stage1/dict_coherence_rms": float(dict_coherence_rms_log.item()),
                        "stage1/loss_ema": float(loss_ema if loss_ema is not None else loss_log.item()),
                        "stage1/batch_in_epoch": int(step_idx + 1),
                        "stage1/epoch": epoch,
                    },
                    step_metric="stage1/step",
                    step_value=global_step,
                )
                if checkpoint_every_steps > 0 and (step_idx + 1) % checkpoint_every_steps == 0:
                    _save_module_checkpoint_with_tag(ae_module, resume_checkpoint_path, artifact_tag)

        # Validation
        ae.eval()
        val_loss_sum = torch.zeros(1, device=device)
        val_recon_sum = torch.zeros(1, device=device)
        val_b_sum = torch.zeros(1, device=device)
        val_psnr_sum = torch.zeros(1, device=device)
        val_ssim_sum = torch.zeros(1, device=device)
        val_count = torch.zeros(1, device=device)
        with torch.no_grad():
            for x, _ in val_loader:
                x = _nan_to_num_tensor(x.to(device, non_blocking=True))
                autocast_ctx = (
                    torch.autocast(device_type=device.type, dtype=amp_dtype)
                    if amp_enabled
                    else nullcontext()
                )
                with autocast_ctx:
                    recon, b_loss, _ = ae(x)
                    recon = _nan_to_num_tensor(recon)
                    b_loss = _nan_to_num_tensor(b_loss)
                    dict_loss = getattr(ae_module.bottleneck, "_last_dl_latent_loss", None)
                    commitment_loss = getattr(ae_module.bottleneck, "_last_e_latent_loss", None)
                    extra_bottleneck_loss = getattr(ae_module.bottleneck, "_last_extra_bottleneck_loss", None)
                    if dict_loss is None or commitment_loss is None:
                        latent_mse = b_loss / max(1.0 + commitment_beta, 1e-8)
                        dict_loss = latent_mse
                        commitment_loss = latent_mse
                    if extra_bottleneck_loss is None:
                        extra_bottleneck_loss = x.new_zeros(())
                    dict_loss = _nan_to_num_tensor(dict_loss)
                    commitment_loss = _nan_to_num_tensor(commitment_loss)
                    extra_bottleneck_loss = _nan_to_num_tensor(extra_bottleneck_loss)
                    recon_loss = F.mse_loss(recon, x)
                    loss = (
                        recon_loss
                        + float(current_dict_loss_weight) * dict_loss
                        + float(current_commitment_loss_weight) * commitment_loss
                        + extra_bottleneck_loss
                    )
                x_unit = _to_unit_range(x)
                recon_unit = _to_unit_range(recon)
                psnr = _batch_psnr(recon_unit, x_unit)
                ssim = _batch_ssim(recon_unit, x_unit)
                val_loss_sum += loss.detach() * x.size(0)
                val_recon_sum += recon_loss.detach() * x.size(0)
                val_b_sum += b_loss.detach() * x.size(0)
                val_psnr_sum += psnr.detach() * x.size(0)
                val_ssim_sum += ssim.detach() * x.size(0)
                val_count += x.size(0)
        if _is_distributed():
            dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_recon_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_b_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_psnr_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_ssim_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_count, op=dist.ReduceOp.SUM)
        val_loss = float((val_loss_sum / val_count.clamp_min(1)).item())
        val_recon_loss = float((val_recon_sum / val_count.clamp_min(1)).item())
        val_bottleneck_loss = float((val_b_sum / val_count.clamp_min(1)).item())
        val_psnr = float((val_psnr_sum / val_count.clamp_min(1)).item())
        val_ssim = float((val_ssim_sum / val_count.clamp_min(1)).item())

        if is_main_process:
            print(
                f"[Stage1] epoch {epoch} val_loss={val_loss:.6f} recon={val_recon_loss:.6f} "
                f"psnr={val_psnr:.3f} ssim={val_ssim:.4f}"
            )
            _log_wandb(
                wandb_run,
                {
                    "stage1/val_loss": float(val_loss),
                    "stage1/val_recon_loss": float(val_recon_loss),
                    "stage1/val_bottleneck_loss": float(val_bottleneck_loss),
                    "stage1/val_psnr": float(val_psnr),
                    "stage1/val_ssim": float(val_ssim),
                    "stage1/lr": float(current_lr),
                    "stage1/epoch": epoch,
                },
                step_metric="stage1/step",
                step_value=global_step,
            )

        _barrier()
        if is_main_process:
            val_rfid = None
            if rfid_num_samples > 0:
                try:
                    val_rfid = _compute_reconstruction_fid(
                        ae_module,
                        rfid_loader,
                        device,
                        max_items=rfid_num_samples,
                    )
                except Exception as exc:
                    if not rfid_warned:
                        print(f"[Stage1] rFID unavailable: {exc}")
                        rfid_warned = True
                    val_rfid = None
                if val_rfid is not None:
                    print(f"[Stage1] epoch {epoch} rfid={val_rfid:.4f}")
                    _log_wandb(
                        wandb_run,
                        {
                            "stage1/rfid": float(val_rfid),
                            "stage1/epoch": epoch,
                        },
                        step_metric="stage1/step",
                        step_value=global_step,
                    )

            x_vis, _ = next(iter(val_loader))
            x_vis = x_vis.to(device, non_blocking=True)[:64]
            with torch.no_grad():
                autocast_ctx = (
                    torch.autocast(device_type=device.type, dtype=amp_dtype)
                    if amp_enabled
                    else nullcontext()
                )
                with autocast_ctx:
                    recon_vis, _, _ = ae_module(x_vis)
            save_image_grid(x_vis, os.path.join(out_dir, f"stage1_epoch{epoch:03d}_real.png"))
            save_image_grid(recon_vis, os.path.join(out_dir, f"stage1_epoch{epoch:03d}_recon.png"))
            _log_wandb_image(
                wandb_run,
                "stage1/real",
                x_vis,
                step_metric="stage1/step",
                step_value=global_step,
                caption=f"epoch={epoch} real",
            )
            _log_wandb_image(
                wandb_run,
                "stage1/recon",
                recon_vis,
                step_metric="stage1/step",
                step_value=global_step,
                caption=f"epoch={epoch} recon",
            )

            os.makedirs(out_dir, exist_ok=True)
            ckpt_path = Path(out_dir) / "ae_last.pt"
            _save_module_checkpoint_with_tag(ae_module, ckpt_path, artifact_tag)
            _save_module_checkpoint_with_tag(ae_module, resume_checkpoint_path, artifact_tag)
            if val_recon_loss < best_val_recon:
                best_val_recon = val_recon_loss
                _save_module_checkpoint_with_tag(
                    ae_module,
                    Path(out_dir) / "ae_best.pt",
                    artifact_tag,
                )
        _barrier()


@torch.no_grad()
def precompute_tokens(
    ae: LASER,
    loader: DataLoader,
    device: torch.device,
    max_items: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, int, int]:
    """
    Encode dataset to tokens for stage-2 training.
    Returns:
      tokens_flat: [N, H*W*token_depth] int32
      coeffs_flat: [N, H*W*sparsity_level] float32 (None if quantized)
      H, W, token_depth
    """
    ae.eval()
    all_tokens = []
    all_coeffs = []
    seen = 0
    H = W = D = None

    for x, _ in tqdm(loader, desc="[Stage2] precompute tokens"):
        x = x.to(device, non_blocking=True)
        if ae.bottleneck.quantize_sparse_coeffs:
            tokens, h, w = ae.encode_to_tokens(x)
            coeffs = None
        else:
            tokens, coeffs, h, w = ae.encode_to_atoms_and_coeffs(x)
        if H is None:
            H = int(tokens.shape[1])
            W = int(tokens.shape[2])
            D = int(tokens.shape[-1])
        elif (H, W, D) != (int(tokens.shape[1]), int(tokens.shape[2]), int(tokens.shape[-1])):
            raise RuntimeError(
                "Stage-2 token grid changed across batches: "
                f"expected {(H, W, D)}, got {(int(tokens.shape[1]), int(tokens.shape[2]), int(tokens.shape[-1]))}"
            )
        flat = tokens.view(tokens.size(0), -1).to(torch.int32).cpu()
        if flat.size(1) != H * W * D:
            raise RuntimeError(
                "Flattened stage-2 token length does not match token-grid metadata: "
                f"flat={flat.size(1)} vs H*W*D={H * W * D}"
            )
        all_tokens.append(flat)
        if coeffs is not None:
            coeffs = _project_real_valued_stage2_coeffs(ae, coeffs)
            all_coeffs.append(coeffs.view(coeffs.size(0), -1).to(torch.float32).cpu())
        seen += flat.size(0)
        if max_items is not None and seen >= max_items:
            break

    tokens_flat = torch.cat(all_tokens, dim=0)
    if len(all_coeffs) > 0:
        coeffs_flat = torch.cat(all_coeffs, dim=0)
    else:
        coeffs_flat = None
    if max_items is not None:
        tokens_flat = tokens_flat[:max_items]
        if coeffs_flat is not None:
            coeffs_flat = coeffs_flat[:max_items]
    return tokens_flat, coeffs_flat, H, W, D


def _load_token_cache(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _compute_atom_conditioned_coeff_stats(
    tokens_flat: torch.Tensor,
    coeffs_flat: torch.Tensor,
    vocab_size: int,
    min_count: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute atom-conditioned coefficient mean/std with global fallbacks for rare atoms."""
    atom_ids = tokens_flat.to(torch.long).flatten()
    coeffs = coeffs_flat.to(torch.float32).flatten()
    count = torch.bincount(atom_ids, minlength=vocab_size).to(torch.float32)
    sumv = torch.bincount(atom_ids, weights=coeffs, minlength=vocab_size)
    sumsq = torch.bincount(atom_ids, weights=coeffs * coeffs, minlength=vocab_size)

    global_mean = coeffs.mean()
    global_std = coeffs.std().clamp_min(1e-6)
    mean = torch.full((vocab_size,), float(global_mean), dtype=torch.float32)
    std = torch.full((vocab_size,), float(global_std), dtype=torch.float32)

    enough = count >= float(max(1, int(min_count)))
    if enough.any():
        mean_enough = sumv[enough] / count[enough].clamp_min(1.0)
        var_enough = (sumsq[enough] / count[enough].clamp_min(1.0) - mean_enough.square()).clamp_min(0.0)
        mean[enough] = mean_enough
        std[enough] = var_enough.sqrt().clamp_min(1e-6)

    return mean, std


def _expected_token_cache_meta(
    args,
    stage2_source_set,
    token_subset: Optional[int],
    ae: LASER,
    stage1_checkpoint_path: Optional[Path],
) -> dict:
    effective_items = len(stage2_source_set) if token_subset is None else int(token_subset)
    return {
        "version": 3,
        "dataset": str(args.dataset),
        "image_size": int(args.image_size),
        "seed": int(args.seed),
        "source_items": int(len(stage2_source_set)),
        "effective_items": int(effective_items),
        "quantize_sparse_coeffs": bool(ae.bottleneck.quantize_sparse_coeffs),
        "ae_num_downsamples": int(args.ae_num_downsamples),
        "embedding_dim": int(args.embedding_dim),
        "num_atoms": int(args.num_atoms),
        "sparsity_level": int(args.sparsity_level),
        "stage1_checkpoint_sha256": _stage1_checkpoint_cache_key(stage1_checkpoint_path),
        "max_ch_mult": int(getattr(ae, "max_ch_mult", 2)),
        "decoder_extra_residual_layers": int(getattr(ae, "decoder_extra_residual_layers", 1)),
        "use_mid_attention": bool(getattr(ae, "use_mid_attention", True)),
        "patch_based": bool(args.patch_based),
        "patch_size": int(args.patch_size),
        "patch_stride": int(args.patch_stride),
        "patch_reconstruction": str(getattr(ae.bottleneck, "patch_reconstruction", args.patch_reconstruction)),
        "variational_coeffs": bool(args.variational_coeffs),
        "variational_coeff_kl_weight": float(args.variational_coeff_kl_weight),
        "variational_coeff_prior_std": float(args.variational_coeff_prior_std),
        "variational_coeff_min_std": float(args.variational_coeff_min_std),
    }


def _token_cache_is_compatible(cache, expected_meta: dict) -> Tuple[bool, str]:
    if not isinstance(cache, dict):
        return False, "cache payload is not a dict"

    tokens_flat = cache.get("tokens_flat")
    coeffs_flat = cache.get("coeffs_flat", None)
    shape = cache.get("shape")
    if not torch.is_tensor(tokens_flat) or tokens_flat.ndim != 2:
        return False, "tokens_flat must be a rank-2 tensor"
    if not isinstance(shape, (tuple, list)) or len(shape) != 3:
        return False, "shape must be a 3-tuple/list"

    H, W, D = (int(shape[0]), int(shape[1]), int(shape[2]))
    if tokens_flat.size(1) != H * W * D:
        return False, "tokens_flat width does not match cached shape metadata"

    expect_real_valued = not bool(expected_meta["quantize_sparse_coeffs"])
    if expect_real_valued != (coeffs_flat is not None):
        mode = "real-valued coefficients" if expect_real_valued else "quantized coefficients"
        return False, f"cache coefficient mode does not match current run ({mode})"

    if coeffs_flat is not None:
        if not torch.is_tensor(coeffs_flat) or coeffs_flat.ndim != 2:
            return False, "coeffs_flat must be a rank-2 tensor when present"
        if coeffs_flat.size(0) != tokens_flat.size(0):
            return False, "coeffs_flat row count does not match tokens_flat"
        if coeffs_flat.size(1) != H * W * D:
            return False, "coeffs_flat width does not match cached shape metadata"

    required_items = int(expected_meta["effective_items"])
    if tokens_flat.size(0) < required_items:
        return False, f"cache has {tokens_flat.size(0)} items but run needs {required_items}"

    cache_meta = cache.get("meta")
    if cache_meta is not None:
        for key, expected_value in expected_meta.items():
            cache_value = cache_meta.get(key)
            if key == "patch_reconstruction":
                expected_value = _canonical_patch_reconstruction(
                    expected_value,
                    patch_size=expected_meta.get("patch_size"),
                    patch_stride=expected_meta.get("patch_stride"),
                )
                cache_value = _canonical_patch_reconstruction(
                    cache_value,
                    patch_size=cache_meta.get("patch_size"),
                    patch_stride=cache_meta.get("patch_stride"),
                )
            if cache_value != expected_value:
                return False, f"meta mismatch for {key}: cache={cache_value!r}, expected={expected_value!r}"

    return True, "ok"


def _compute_quantized_rq_losses(
    per_token_ce: torch.Tensor,
    atom_loss_weight: float,
    coeff_loss_weight: float,
    coeff_depth_weighting: str = "none",
    coeff_focal_gamma: float = 0.0,
    coeff_logits: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
    """Split shared-vocab CE into atom/coeff views while keeping one total loss."""
    if per_token_ce.numel() == 0:
        raise ValueError("Expected non-empty per-token CE tensor")
    del coeff_logits

    def _depth_weights(depth_steps: int, mode: str, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if depth_steps <= 0:
            raise ValueError("depth_steps must be positive")
        mode = str(mode).strip().lower()
        if mode == "none":
            return torch.ones(depth_steps, device=device, dtype=dtype)
        if mode == "linear":
            weights = torch.arange(depth_steps, 0, -1, device=device, dtype=dtype)
        elif mode == "inverse_rank":
            weights = 1.0 / torch.arange(1, depth_steps + 1, device=device, dtype=dtype)
        else:
            raise ValueError(f"Unsupported coeff_depth_weighting: {mode!r}")
        return weights / weights.mean().clamp_min(1e-8)

    token_ce_loss = per_token_ce.mean()
    atom_terms = per_token_ce[..., 0::2]
    coeff_terms = per_token_ce[..., 1::2]

    atom_ce_loss = atom_terms.mean() if atom_terms.numel() > 0 else None

    coeff_ce_loss = None
    coeff_weighted_terms = None
    if coeff_terms.numel() > 0:
        coeff_weighted_terms = coeff_terms
        depth_weights = _depth_weights(
            coeff_terms.size(-1),
            coeff_depth_weighting,
            device=coeff_terms.device,
            dtype=coeff_terms.dtype,
        )
        coeff_weighted_terms = coeff_weighted_terms * depth_weights.view(
            *([1] * (coeff_terms.ndim - 1)),
            coeff_terms.size(-1),
        )
        coeff_focal_gamma = float(max(0.0, coeff_focal_gamma))
        if coeff_focal_gamma > 0.0:
            pt = torch.exp(-coeff_terms.clamp_min(0.0))
            coeff_weighted_terms = coeff_weighted_terms * (1.0 - pt).pow(coeff_focal_gamma)
        coeff_ce_loss = coeff_weighted_terms.mean()

    total_numerator = per_token_ce.new_tensor(0.0)
    total_denominator = per_token_ce.new_tensor(0.0)
    atom_loss_weight = float(atom_loss_weight)
    coeff_loss_weight = float(coeff_loss_weight)
    if atom_ce_loss is not None:
        total_numerator = total_numerator + atom_loss_weight * atom_terms.sum()
        total_denominator = total_denominator + atom_loss_weight * atom_terms.numel()
    if coeff_ce_loss is not None and coeff_weighted_terms is not None:
        total_numerator = total_numerator + coeff_loss_weight * coeff_weighted_terms.sum()
        total_denominator = total_denominator + coeff_loss_weight * coeff_weighted_terms.numel()
    if total_denominator.item() <= 0.0:
        raise ValueError("Expected positive quantized loss denominator")
    total_loss = total_numerator / total_denominator
    return token_ce_loss, atom_ce_loss, coeff_ce_loss, total_loss


def train_stage2_transformer(
    transformer: nn.Module,
    token_loader: DataLoader,
    stage2_fid_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int,
    lr: float,
    rq_atom_loss_weight: float,
    rq_coeff_loss_weight: float,
    coeff_loss_weight: float,
    coeff_loss_type: str,
    coeff_huber_delta: float,
    sched_sampling_final_prob: float,
    stage2_amp: bool,
    stage2_amp_dtype: str,
    pad_token_id: int,
    out_dir: str,
    ae_for_decode: LASER,
    H: int,
    W: int,
    D: int,
    sample_every_steps: int = 200,
    sample_start_step: int = 0,
    # ---- LR schedule ----
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.01,
    weight_decay: float = 0.0,
    # ---- coefficient depth/focal weighting ----
    coeff_depth_weighting: str = "none",
    coeff_focal_gamma: float = 0.0,
    # ---- ordinal coefficient regression ----
    ordinal_coeff_weight: float = 0.0,
    ordinal_coeff_huber_delta: float = 0.5,
    ordinal_magnitude_weighted: bool = False,
    ordinal_zero_drift_margin: float = 0.0,
    ordinal_zero_drift_threshold: float = 0.3,
    sample_batch_size: int = 8,
    sample_candidate_factor: int = 4,
    sample_temperature: float = 1.0,
    sample_top_k: Optional[int] = 256,
    sample_coeff_temperature: Optional[float] = None,
    sample_coeff_mode: str = "gaussian",
    sample_selection_quality_weight: float = 1.0,
    sample_brightness_weight: float = 1.0,
    sample_overbright_weight: float = 1.0,
    sample_reject_dark_z: float = 1.5,
    sample_reject_bright_z: float = 1.5,
    sample_selection_mode: str = "quality_only",
    sample_sort_by_quality: bool = True,
    sample_image_size: Optional[int] = None,
    sample_reference_stats: Optional[dict] = None,
    stage2_fid_num_samples: int = 0,
    stage2_fid_every_epochs: int = 0,
    token_sampler: Optional[DistributedSampler] = None,
    is_main_process: bool = True,
    wandb_run: Optional[object] = None,
    artifact_tag: Optional[str] = None,
):
    """Train stage 2 with optional DDP and synchronized rank-0 sampling."""
    transformer_module = _unwrap_module(transformer)
    if not hasattr(transformer_module, "cfg") or not hasattr(transformer_module, "generate"):
        raise TypeError(
            "train_stage2_transformer expects a stage-2 prior with cfg/generate support; "
            f"got {type(transformer_module)!r}"
        )
    ae_decode = _unwrap_module(ae_for_decode)
    ae_decode.eval()
    ae_decode.requires_grad_(False)
    if weight_decay > 0:
        opt = torch.optim.AdamW(transformer.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.Adam(transformer.parameters(), lr=lr)
    total_steps = max(1, epochs * max(1, len(token_loader)))
    warmup_steps = max(0, int(warmup_steps))
    min_lr_ratio = float(max(0.0, min(float(min_lr_ratio), 1.0)))
    sample_start_step = max(0, int(sample_start_step))
    vocab = transformer_module.cfg.vocab_size
    global_step = 0
    sample_top_k = None if sample_top_k is None or int(sample_top_k) <= 0 else int(sample_top_k)
    if sample_coeff_temperature is None:
        resolved_sample_coeff_temperature = None
    else:
        resolved_sample_coeff_temperature = float(sample_coeff_temperature)
        if not math.isfinite(resolved_sample_coeff_temperature):
            resolved_sample_coeff_temperature = None
        elif resolved_sample_coeff_temperature <= 0.0:
            raise ValueError("sample_coeff_temperature must be > 0 when set.")
    sample_coeff_mode = str(sample_coeff_mode).strip().lower()
    if sample_coeff_mode not in {"gaussian", "mean"}:
        raise ValueError(
            f"sample_coeff_mode must be 'gaussian' or 'mean', got {sample_coeff_mode!r}"
        )
    sample_selection_quality_weight = float(sample_selection_quality_weight)
    if sample_selection_quality_weight < 0.0:
        raise ValueError("sample_selection_quality_weight must be >= 0.")
    sample_brightness_weight = float(sample_brightness_weight)
    if sample_brightness_weight < 0.0:
        raise ValueError("sample_brightness_weight must be >= 0.")
    sample_overbright_weight = float(sample_overbright_weight)
    if sample_overbright_weight < 0.0:
        raise ValueError("sample_overbright_weight must be >= 0.")
    sample_reject_dark_z = float(sample_reject_dark_z)
    if sample_reject_dark_z < 0.0:
        raise ValueError("sample_reject_dark_z must be >= 0.")
    sample_reject_bright_z = float(sample_reject_bright_z)
    if sample_reject_bright_z < 0.0:
        raise ValueError("sample_reject_bright_z must be >= 0.")
    sample_selection_mode = str(sample_selection_mode).strip().lower()
    if sample_selection_mode not in {"quality_only", "diverse"}:
        raise ValueError(
            "sample_selection_mode must be 'quality_only' or 'diverse', "
            f"got {sample_selection_mode!r}."
        )
    sample_sort_by_quality = bool(sample_sort_by_quality)
    sample_candidate_factor = max(1, int(sample_candidate_factor))
    stage2_fid_num_samples = max(0, int(stage2_fid_num_samples))
    stage2_fid_every_epochs = max(0, int(stage2_fid_every_epochs))
    real_valued = transformer_module.real_valued_coeffs
    gaussian_coeffs = bool(getattr(transformer_module, "gaussian_coeffs", False))
    rq_atom_loss_weight = float(rq_atom_loss_weight)
    rq_coeff_loss_weight = float(rq_coeff_loss_weight)
    coeff_loss_weight = float(coeff_loss_weight)
    coeff_loss_type = str(coeff_loss_type).lower()
    if coeff_loss_type not in {"huber", "mse", "recon_mse", "gt_atom_recon_mse", "gaussian_nll"}:
        raise ValueError(f"Unsupported coeff_loss_type: {coeff_loss_type!r}")
    coeff_huber_delta = float(coeff_huber_delta)
    amp_dtype_name = str(stage2_amp_dtype).strip().lower()
    if amp_dtype_name not in {"auto", "float16", "bfloat16"}:
        raise ValueError(f"Unsupported stage2_amp_dtype: {stage2_amp_dtype!r}")
    amp_enabled = bool(stage2_amp) and device.type == "cuda"
    if amp_dtype_name == "auto":
        amp_dtype = torch.bfloat16 if (amp_enabled and torch.cuda.is_bf16_supported()) else torch.float16
    elif amp_dtype_name == "bfloat16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
    scaler_enabled = amp_enabled and amp_dtype == torch.float16
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(
            device="cuda",
            enabled=scaler_enabled,
        )
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    sched_sampling_final_prob = max(0.0, float(sched_sampling_final_prob))
    if sched_sampling_final_prob > 0.0 and is_main_process:
        print(
            "[Stage2] ignoring deprecated scheduled sampling setting "
            f"(requested final_prob={sched_sampling_final_prob:.3f}); using pure teacher forcing"
        )
    if not real_valued and is_main_process:
        print(
            "[Stage2] quantized stage-2 loss: shared-vocab CE on the cached interleaved token stream"
        )
    if amp_enabled and is_main_process:
        amp_label = "bfloat16" if amp_dtype == torch.bfloat16 else "float16"
        print(f"[Stage2] using AMP ({amp_label})")
    if real_valued and is_main_process:
        decode_coef_max = getattr(getattr(ae_decode, "bottleneck", None), "coef_max", None)
        try:
            decode_coef_max = float(decode_coef_max)
        except (TypeError, ValueError):
            decode_coef_max = None
        if decode_coef_max is not None and math.isfinite(decode_coef_max) and decode_coef_max > 0.0:
            print(
                f"[Stage2] clamping sampled real-valued coeffs to +/-{decode_coef_max:g} before stage-1 decode"
            )
    offload_decode_model = (
        (not real_valued)
        and device.type == "cuda"
        and next(ae_decode.parameters()).device.type == "cuda"
    )
    if offload_decode_model:
        ae_decode.to("cpu")
        torch.cuda.empty_cache()
        if is_main_process:
            print("[Stage2] parking stage-1 decoder on CPU between sampling steps")
    if sample_every_steps > 0 and sample_start_step > 0 and is_main_process:
        print(
            f"[Stage2] delaying preview sampling until step {sample_start_step} "
            f"(then every {sample_every_steps} steps)"
        )
    num_batches = max(1, len(token_loader))
    if coeff_loss_type == "gaussian_nll" and not real_valued:
        raise ValueError("stage2 coeff_loss_type='gaussian_nll' requires real-valued sparse coefficients")
    if coeff_loss_type == "gaussian_nll" and not gaussian_coeffs:
        raise ValueError("stage2 coeff_loss_type='gaussian_nll' requires variational_coeffs=True")
    stage2_fid_warned = False
    # ---- ordinal coefficient regression setup ----
    use_ordinal = False
    coeff_bin_values_for_ordinal = None
    if (not real_valued) and ordinal_coeff_weight > 0 and is_main_process:
        print("[Stage2] ignoring ordinal coefficient aux loss for quantized shared-vocab training")
    if use_ordinal:
        from contrastive_sparse import ordinal_coeff_loss
        coeff_bin_values_for_ordinal = ae_decode.bottleneck._dequantize_coeff(
            torch.arange(ae_decode.bottleneck.n_bins, dtype=torch.long)
        ).detach().to(device)
        if is_main_process:
            parts = [f"ordinal_coeff(w={ordinal_coeff_weight}, mag_w={ordinal_magnitude_weighted})"]
            if ordinal_zero_drift_margin > 0:
                parts.append(f"zero_drift(margin={ordinal_zero_drift_margin}, thresh={ordinal_zero_drift_threshold})")
            print(f"[Stage2] aux losses: {', '.join(parts)}")

    for epoch in range(1, epochs + 1):
        if token_sampler is not None:
            token_sampler.set_epoch(epoch)
        transformer.train()
        pbar = tqdm(token_loader, desc=f"[Stage2] epoch {epoch}/{epochs}", disable=(not is_main_process))
        running = 0.0
        steps = 0

        for batch_idx, batch in enumerate(pbar):
            if real_valued:
                tok_flat = batch[0].to(device=device, dtype=torch.long, non_blocking=True)
                coeff_flat = batch[1].to(device=device, dtype=torch.float32, non_blocking=True)
            else:
                tok_flat = batch[0] if isinstance(batch, (tuple, list)) else batch
                tok_flat = tok_flat.to(device=device, dtype=torch.long, non_blocking=True)
            B = tok_flat.size(0)

            opt.zero_grad(set_to_none=True)
            ce_loss = None
            atom_ce_loss = None
            coeff_ce_loss = None
            coeff_reg_loss = None
            ordinal_loss = None
            loss = None
            autocast_ctx = (
                torch.autocast(device_type=device.type, dtype=amp_dtype)
                if amp_enabled
                else nullcontext()
            )
            with autocast_ctx:
                if real_valued:
                    # Take private copies before stage-2 forward/recon paths so
                    # CE targets, masking tokens, and latent-reconstruction
                    # indices do not alias each other across the rollout paths.
                    tok_grid = tok_flat.view(B, H * W, D).clone()
                    coeff_grid = _project_real_valued_stage2_coeffs(
                        ae_decode,
                        coeff_flat.view(B, H * W, D).clone(),
                    )
                    ce_tok_grid = tok_grid.detach().clone()
                    forward_tok_grid = tok_grid.detach().clone()
                    forward_mask_tok_grid = ce_tok_grid.detach().clone()
                    if gaussian_coeffs:
                        atom_logits, coeff_pred, coeff_logvar_pred, depth_h = transformer(
                            forward_tok_grid,
                            coeff_grid,
                            mask_tokens=forward_mask_tok_grid,
                            return_features=True,
                        )
                    else:
                        atom_logits, coeff_pred, depth_h = transformer(
                            forward_tok_grid,
                            coeff_grid,
                            mask_tokens=forward_mask_tok_grid,
                            return_features=True,
                        )
                        coeff_logvar_pred = None
                    ce_loss = F.cross_entropy(
                        atom_logits.reshape(-1, vocab),
                        ce_tok_grid.reshape(-1),
                    )
                    pred_coeff = ae_decode.clamp_sparse_coeffs(coeff_pred)
                    target_coeff = ae_decode.clamp_sparse_coeffs(coeff_grid)
                    if coeff_loss_type == "mse":
                        coeff_reg_loss = F.mse_loss(pred_coeff, target_coeff)
                    elif coeff_loss_type == "huber":
                        coeff_reg_loss = F.huber_loss(
                            pred_coeff,
                            target_coeff,
                            delta=coeff_huber_delta,
                        )
                    elif coeff_loss_type == "gaussian_nll":
                        if coeff_logvar_pred is None:
                            raise RuntimeError("gaussian_nll requested but transformer did not return coeff_logvar")
                        pred_var = coeff_logvar_pred.exp().clamp_min(1e-6)
                        coeff_reg_loss = 0.5 * (
                            coeff_logvar_pred + (pred_coeff - target_coeff).square() / pred_var
                        )
                        coeff_reg_loss = coeff_reg_loss.mean()
                    elif coeff_loss_type == "recon_mse":
                        # Use a self-consistent support mask for the recon loss.
                        # The CE branch still scores ground-truth atoms under the
                        # ground-truth support constraints, but recon_mse is meant
                        # to approximate free-running support generation rather
                        # than argmax under target-only masking.
                        rollout_context_tok_grid = tok_grid.detach().clone()
                        rollout_context_coeff_grid = coeff_grid.detach().clone()
                        rollout_mask_tok_grid = rollout_context_tok_grid.detach().clone()
                        if gaussian_coeffs:
                            rollout_atom_logits, _, _, rollout_depth_h = transformer(
                                rollout_context_tok_grid,
                                rollout_context_coeff_grid,
                                mask_tokens=rollout_mask_tok_grid,
                                return_features=True,
                            )
                        else:
                            rollout_atom_logits, _, rollout_depth_h = transformer(
                                rollout_context_tok_grid,
                                rollout_context_coeff_grid,
                                mask_tokens=rollout_mask_tok_grid,
                                return_features=True,
                            )
                        pred_atoms = rollout_atom_logits.argmax(dim=-1)
                        pred_atoms_for_coeff = pred_atoms.detach().clone()
                        pred_atoms_for_recon = pred_atoms.detach().clone()
                        pred_coeff = ae_decode.clamp_sparse_coeffs(
                            transformer_module.predict_coeffs_for_atoms(rollout_depth_h, pred_atoms_for_coeff)
                        )
                        pred_latent = _reconstruct_stage2_sparse_latent(
                            ae_decode,
                            pred_atoms_for_recon.view(B, H, W, D),
                            pred_coeff.view(B, H, W, D),
                        )
                        with torch.no_grad():
                            target_tok_grid = ce_tok_grid.detach().clone()
                            target_latent = _reconstruct_stage2_sparse_latent(
                                ae_decode,
                                target_tok_grid.view(B, H, W, D),
                                target_coeff.view(B, H, W, D),
                            )
                        coeff_reg_loss = F.mse_loss(pred_latent, target_latent)
                    else:
                        pred_tok_grid = ce_tok_grid.detach().clone()
                        pred_latent = _reconstruct_stage2_sparse_latent(
                            ae_decode,
                            pred_tok_grid.view(B, H, W, D),
                            pred_coeff.view(B, H, W, D),
                        )
                        with torch.no_grad():
                            target_tok_grid = ce_tok_grid.detach().clone()
                            target_latent = _reconstruct_stage2_sparse_latent(
                                ae_decode,
                                target_tok_grid.view(B, H, W, D),
                                target_coeff.view(B, H, W, D),
                            )
                        coeff_reg_loss = F.mse_loss(pred_latent, target_latent)
                    loss = ce_loss + coeff_loss_weight * coeff_reg_loss
                else:
                    tok_grid = tok_flat.view(B, H * W, D).clone()
                    logits = transformer(tok_grid)
                    per_token_ce = F.cross_entropy(
                        logits.reshape(-1, vocab),
                        tok_grid.reshape(-1),
                        reduction="none",
                    ).view(B, H * W, D)
                    ce_loss, atom_ce_loss, coeff_ce_loss, loss = _compute_quantized_rq_losses(
                        per_token_ce,
                        atom_loss_weight=rq_atom_loss_weight,
                        coeff_loss_weight=rq_coeff_loss_weight,
                        coeff_depth_weighting=coeff_depth_weighting,
                        coeff_focal_gamma=coeff_focal_gamma,
                        coeff_logits=None,
                    )
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
                opt.step()
            global_step += 1

            # ---- LR schedule: linear warmup + cosine decay ----
            if warmup_steps > 0 or min_lr_ratio < 1.0:
                if global_step <= warmup_steps:
                    scale = max(0.01, global_step / max(1, warmup_steps))
                else:
                    progress = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
                    progress = min(progress, 1.0)
                    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                    scale = min_lr_ratio + (1.0 - min_lr_ratio) * cosine
                for pg in opt.param_groups:
                    pg["lr"] = lr * scale

            loss_log = _distributed_mean(loss)
            ce_log = _distributed_mean(ce_loss.detach())
            atom_ce_log = (
                _distributed_mean(atom_ce_loss.detach())
                if atom_ce_loss is not None else None
            )
            coeff_ce_log = (
                _distributed_mean(coeff_ce_loss.detach())
                if coeff_ce_loss is not None else None
            )
            coeff_reg_log = (
                _distributed_mean(coeff_reg_loss.detach())
                if coeff_reg_loss is not None else None
            )
            running += float(loss_log.item())
            steps += 1
            if is_main_process:
                postfix = {
                    "loss": float(loss_log.item()),
                    "ce": float(ce_log.item()),
                }
                if atom_ce_log is not None:
                    postfix["atom_ce"] = float(atom_ce_log.item())
                if coeff_ce_log is not None:
                    postfix["coeff_ce"] = float(coeff_ce_log.item())
                if coeff_reg_log is not None:
                    postfix[coeff_loss_type] = float(coeff_reg_log.item())
                pbar.set_postfix(**postfix)
                log_payload = {
                    "stage2/train_loss": float(loss_log.item()),
                    "stage2/ce_loss": float(ce_log.item()),
                    "stage2/epoch": epoch,
                    "stage2/lr": float(opt.param_groups[0]["lr"]),
                }
                if atom_ce_log is not None:
                    log_payload["stage2/atom_ce_loss"] = float(atom_ce_log.item())
                    log_payload["stage2/rq_atom_loss_weight"] = rq_atom_loss_weight
                    log_payload["stage2/weighted_atom_ce_loss"] = float(
                        rq_atom_loss_weight * atom_ce_log.item()
                    )
                if coeff_ce_log is not None:
                    log_payload["stage2/coeff_ce_loss"] = float(coeff_ce_log.item())
                    log_payload["stage2/rq_coeff_loss_weight"] = rq_coeff_loss_weight
                    log_payload["stage2/weighted_coeff_ce_loss"] = float(
                        rq_coeff_loss_weight * coeff_ce_log.item()
                    )
                if coeff_reg_log is not None:
                    log_payload["stage2/coeff_reg_loss"] = float(coeff_reg_log.item())
                    log_payload["stage2/coeff_loss_type"] = coeff_loss_type
                    log_payload["stage2/coeff_loss_weight"] = coeff_loss_weight
                    log_payload["stage2/weighted_coeff_loss"] = float(coeff_loss_weight * coeff_reg_log.item())
                    if coeff_loss_type == "mse":
                        log_payload["stage2/coeff_mse_loss"] = float(coeff_reg_log.item())
                    elif coeff_loss_type == "gaussian_nll":
                        log_payload["stage2/coeff_gaussian_nll"] = float(coeff_reg_log.item())
                    elif coeff_loss_type in {"recon_mse", "gt_atom_recon_mse"}:
                        log_payload["stage2/recon_mse_loss"] = float(coeff_reg_log.item())
                    else:
                        log_payload["stage2/coeff_huber_loss"] = float(coeff_reg_log.item())
                        log_payload["stage2/coeff_huber_delta"] = coeff_huber_delta
                if use_ordinal and ordinal_loss is not None:
                    log_payload["stage2/ordinal_coeff"] = float(ordinal_loss.item())
                    log_payload["stage2/ordinal_coeff_weight"] = ordinal_coeff_weight
                _log_wandb(
                    wandb_run,
                    log_payload,
                    step_metric="stage2/step",
                    step_value=global_step,
                )

            should_sample = False
            if sample_every_steps > 0 and global_step >= sample_start_step:
                should_sample = ((global_step - sample_start_step) % sample_every_steps == 0)

            if should_sample:
                opt.zero_grad(set_to_none=True)
                _barrier()
                if is_main_process:
                    transformer.eval()
                    if offload_decode_model:
                        ae_decode.to(device)
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                    ae_decode.eval()
                    candidate_batch_size = int(sample_batch_size)
                    if sample_reference_stats is not None:
                        candidate_batch_size = max(
                            candidate_batch_size,
                            int(sample_batch_size) * sample_candidate_factor,
                        )
                    decode_batch_size = max(1, min(8, int(sample_batch_size)))
                    sample_parts = [
                        f"keep={sample_batch_size}",
                        f"candidates={candidate_batch_size}",
                        f"atom_temp={sample_temperature}",
                        f"atom_top_k={sample_top_k}",
                    ]
                    if real_valued:
                        sample_parts.append(f"coeff_mode={sample_coeff_mode}")
                        sample_parts.append(
                            "coeff_temp="
                            f"{sample_temperature if resolved_sample_coeff_temperature is None else resolved_sample_coeff_temperature}"
                        )
                    sample_parts.extend([
                        f"select_q={sample_selection_quality_weight}",
                        f"brightness_q={sample_brightness_weight}",
                        f"overbright_q={sample_overbright_weight}",
                        f"reject_dark_z={sample_reject_dark_z}",
                        f"reject_bright_z={sample_reject_bright_z}",
                        f"selection_mode={sample_selection_mode}",
                        f"sort_by_quality={sample_sort_by_quality}",
                    ])
                    print(f"[Stage2] sampling at step {global_step} ({', '.join(sample_parts)})...")
                    try:
                        with torch.no_grad():
                            if device.type == "cuda":
                                torch.cuda.empty_cache()
                            sample_debug_payload = {}
                            if real_valued:
                                atoms_gen, coeffs_gen = transformer_module.generate(
                                    batch_size=candidate_batch_size,
                                    temperature=sample_temperature,
                                    top_k=sample_top_k,
                                    coeff_temperature=resolved_sample_coeff_temperature,
                                    coeff_sample_mode=sample_coeff_mode,
                                    show_progress=True,
                                    progress_desc=f"[Stage2] sample step {global_step}",
                                )
                                coeffs_gen = _clamp_generated_sparse_coeffs_for_decode(ae_decode, coeffs_gen)
                                atoms_gen = atoms_gen.view(-1, H, W, D)
                                coeffs_gen = coeffs_gen.view(-1, H, W, D)
                                if device.type == "cuda":
                                    torch.cuda.empty_cache()
                                imgs = _decode_stage2_candidates_in_chunks(
                                    ae_decode,
                                    atoms_gen,
                                    coeffs=coeffs_gen,
                                    decode_batch_size=decode_batch_size,
                                )
                                sample_debug_payload.update(
                                    _sample_tensor_stats_dict(atoms_gen, "stage2/sample_atoms_")
                                )
                                sample_debug_payload.update(
                                    _sample_tensor_stats_dict(coeffs_gen, "stage2/sample_coeffs_")
                                )
                                del atoms_gen
                                del coeffs_gen
                            else:
                                tokens_gen = transformer_module.generate(
                                    batch_size=candidate_batch_size,
                                    temperature=sample_temperature,
                                    top_k=sample_top_k,
                                    show_progress=True,
                                    progress_desc=f"[Stage2] sample step {global_step}",
                                ).view(-1, H, W, D)
                                if device.type == "cuda":
                                    torch.cuda.empty_cache()
                                imgs = _decode_stage2_candidates_in_chunks(
                                    ae_decode,
                                    tokens_gen,
                                    decode_batch_size=decode_batch_size,
                                )
                                sample_debug_payload.update(
                                    _sample_token_stats_dict(
                                        tokens_gen,
                                        "stage2/sample_tokens_",
                                        vocab_size=vocab,
                                    )
                                )
                                del tokens_gen
                            imgs_raw = imgs[:min(int(sample_batch_size), int(imgs.size(0)))].clone()
                            sample_debug_payload.update(
                                _sample_image_stats_dict(imgs_raw, "stage2/sample_raw_")
                            )
                            imgs = _select_best_stage2_samples(
                                imgs,
                                keep=sample_batch_size,
                                reference_stats=sample_reference_stats,
                                quality_weight=sample_selection_quality_weight,
                                brightness_weight=sample_brightness_weight,
                                overbright_weight=sample_overbright_weight,
                                reject_dark_z=sample_reject_dark_z,
                                reject_bright_z=sample_reject_bright_z,
                                selection_mode=sample_selection_mode,
                                sort_by_quality=sample_sort_by_quality,
                            )
                            if sample_image_size is not None and int(sample_image_size) > 0:
                                if imgs.size(-2) != int(sample_image_size) or imgs.size(-1) != int(sample_image_size):
                                    imgs = F.interpolate(
                                        imgs,
                                        size=(int(sample_image_size), int(sample_image_size)),
                                        mode="bilinear",
                                        align_corners=False,
                                    )
                            sample_debug_payload.update(
                                _sample_image_stats_dict(imgs, "stage2/sample_selected_")
                            )
                    finally:
                        if offload_decode_model:
                            ae_decode.to("cpu")
                            if device.type == "cuda":
                                torch.cuda.empty_cache()
                    save_image_grid(
                        imgs_raw,
                        os.path.join(out_dir, f"stage2_step{global_step:06d}_raw_samples.png"),
                    )
                    save_image_grid(imgs, os.path.join(out_dir, f"stage2_step{global_step:06d}_samples.png"))
                    _log_wandb_image(
                        wandb_run,
                        "stage2/raw_samples",
                        imgs_raw,
                        step_metric="stage2/step",
                        step_value=global_step,
                        caption=f"step={global_step} raw",
                    )
                    _log_wandb_image(
                        wandb_run,
                        "stage2/samples",
                        imgs,
                        step_metric="stage2/step",
                        step_value=global_step,
                        caption=f"step={global_step}",
                    )
                    if sample_debug_payload:
                        _log_wandb(
                            wandb_run,
                            sample_debug_payload,
                            step_metric="stage2/step",
                            step_value=global_step,
                        )
                        raw_brightness = float(sample_debug_payload.get("stage2/sample_raw_brightness_mean", float("nan")))
                        selected_brightness = float(sample_debug_payload.get("stage2/sample_selected_brightness_mean", float("nan")))
                        raw_std = float(sample_debug_payload.get("stage2/sample_raw_pixel_std", float("nan")))
                        selected_std = float(sample_debug_payload.get("stage2/sample_selected_pixel_std", float("nan")))
                        if real_valued:
                            coeff_abs_mean = float(sample_debug_payload.get("stage2/sample_coeffs_abs_mean", float("nan")))
                            print(
                                f"[Stage2] sample stats step {global_step}: "
                                f"raw_brightness={raw_brightness:.4f} raw_std={raw_std:.4f} "
                                f"selected_brightness={selected_brightness:.4f} selected_std={selected_std:.4f} "
                                f"coeff_abs_mean={coeff_abs_mean:.4f}"
                            )
                        else:
                            token_mean = float(sample_debug_payload.get("stage2/sample_tokens_mean", float("nan")))
                            token_std = float(sample_debug_payload.get("stage2/sample_tokens_std", float("nan")))
                            token_vocab_frac = float(sample_debug_payload.get("stage2/sample_tokens_unique_vocab_fraction", float("nan")))
                            print(
                                f"[Stage2] sample stats step {global_step}: "
                                f"token_mean={token_mean:.2f} token_std={token_std:.2f} "
                                f"token_vocab_frac={token_vocab_frac:.4f} "
                                f"raw_brightness={raw_brightness:.4f} raw_std={raw_std:.4f} "
                                f"selected_brightness={selected_brightness:.4f} selected_std={selected_std:.4f}"
                            )
                    print(f"[Stage2] sampling done at step {global_step}")
                _barrier()
                transformer.train()

        epoch_loss = running / max(1, steps)
        if is_main_process:
            print(f"[Stage2] epoch {epoch} train_loss={epoch_loss:.6f}")
            _log_wandb(
                wandb_run,
                {
                    "stage2/epoch_loss": float(epoch_loss),
                    "stage2/epoch": epoch,
                },
                step_metric="stage2/step",
                step_value=global_step,
            )
            stage2_fid = None
            if (
                stage2_fid_loader is not None
                and stage2_fid_num_samples > 0
                and stage2_fid_every_epochs > 0
                and (epoch % stage2_fid_every_epochs == 0)
            ):
                print(
                    f"[Stage2] computing sample FID at epoch {epoch} "
                    f"(num_samples={stage2_fid_num_samples})"
                )
                if offload_decode_model:
                    ae_decode.to(device)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                try:
                    stage2_fid = _compute_stage2_sample_fid(
                        transformer=transformer_module,
                        ae=ae_decode,
                        loader=stage2_fid_loader,
                        device=device,
                        max_items=stage2_fid_num_samples,
                        H=H,
                        W=W,
                        D=D,
                        sample_batch_size=sample_batch_size,
                        sample_candidate_factor=sample_candidate_factor,
                        sample_temperature=sample_temperature,
                        sample_top_k=sample_top_k,
                        sample_coeff_temperature=resolved_sample_coeff_temperature,
                        sample_coeff_mode=sample_coeff_mode,
                        sample_reference_stats=sample_reference_stats,
                        sample_selection_quality_weight=sample_selection_quality_weight,
                        sample_brightness_weight=sample_brightness_weight,
                        sample_overbright_weight=sample_overbright_weight,
                        sample_reject_dark_z=sample_reject_dark_z,
                        sample_reject_bright_z=sample_reject_bright_z,
                        sample_selection_mode=sample_selection_mode,
                        sample_sort_by_quality=sample_sort_by_quality,
                        sample_image_size=sample_image_size,
                    )
                except Exception as exc:
                    if not stage2_fid_warned:
                        print(f"[Stage2] sample FID disabled after failure: {exc}")
                        stage2_fid_warned = True
                    stage2_fid = None
                finally:
                    if offload_decode_model:
                        ae_decode.to("cpu")
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                if stage2_fid is not None:
                    print(f"[Stage2] epoch {epoch} fid={stage2_fid:.4f}")
                    _log_wandb(
                        wandb_run,
                        {
                            "stage2/fid": float(stage2_fid),
                            "stage2/fid_num_samples": stage2_fid_num_samples,
                            "stage2/fid_eval_epoch_interval": stage2_fid_every_epochs,
                            "stage2/epoch": epoch,
                        },
                        step_metric="stage2/step",
                        step_value=global_step,
                    )
        _barrier()
        if is_main_process:
            os.makedirs(out_dir, exist_ok=True)
            transformer_last_path = Path(out_dir) / "transformer_last.pt"
            _save_module_checkpoint(transformer_module, transformer_last_path)
            _copy_artifact_to_tagged_path(transformer_last_path, artifact_tag)
        _barrier()



# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Train the core LASER pipeline.")
    parser.add_argument("--dataset", type=str, default="celeba", choices=["cifar10", "celeba"])
    parser.add_argument("--data_dir", type=str, default=None, help="Root directory for dataset files.")
    parser.add_argument(
        "--image_size",
        type=int,
        default=128,
        help="Resize every image to this square size.",
    )
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stage1_source_run",
        type=str,
        default=None,
        help=(
            "W&B run path to source the stage-1 ae_best.pt checkpoint from, "
            "for example helloimlixin-rutgers/laser/4psikzda."
        ),
    )
    parser.add_argument(
        "--stage1_source_ckpt",
        type=str,
        default=None,
        help="Local path to a stage-1 ae_best.pt checkpoint to use before any W&B or prior-run fallback.",
    )
    parser.add_argument(
        "--stage1_auto_resume_from_latest",
        dest="stage1_auto_resume_from_latest",
        action="store_true",
        default=False,
        help=(
            "When stage 1 is training from scratch, warm-start from the newest stage-1 checkpoint "
            "already present under the same experiment root."
        ),
    )
    parser.add_argument(
        "--no_stage1_auto_resume_from_latest",
        dest="stage1_auto_resume_from_latest",
        action="store_false",
        help="Disable automatic stage-1 warm-start from prior checkpoints in the same experiment root.",
    )
    parser.add_argument(
        "--stage2_source_run",
        type=str,
        default=None,
        help=(
            "W&B run path to source the stage-2 tokens_cache.pt from, "
            "for example helloimlixin-rutgers/laser/q5l0g3jn."
        ),
    )
    parser.add_argument(
        "--stage2_source_token_cache",
        type=str,
        default=None,
        help="Local path to a stage-2 tokens_cache.pt to reuse before rebuilding.",
    )
    parser.add_argument(
        "--stage2_source_ckpt",
        type=str,
        default=None,
        help="Local path to a stage-2 transformer checkpoint to warm-start before training.",
    )
    parser.add_argument(
        "--dist_timeout_minutes",
        type=int,
        default=120,
        help="Distributed process-group timeout in minutes. Increase this when only rank 0 is busy building a token cache.",
    )
    parser.add_argument("--num_workers", type=int, default=4, help="Workers for image dataloaders.")
    parser.add_argument("--token_num_workers", type=int, default=0, help="Workers for token precompute.")
    parser.add_argument("--wandb", dest="wandb", action="store_true", default=True, help="Enable Weights & Biases logging.")
    parser.add_argument("--no_wandb", dest="wandb", action="store_false", help="Disable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="laser")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_dir", type=str, default="./wandb")

    parser.add_argument(
        "--analyze_spectrum",
        dest="analyze_spectrum",
        action="store_true",
        default=False,
        help="After loading/training stage-1, run patch spectrum analysis and exit.",
    )
    parser.add_argument(
        "--spectrum_n_patches",
        type=int,
        default=50_000,
        help="Number of latent patches to accumulate for spectrum analysis.",
    )

    parser.add_argument("--stage1_epochs", type=int, default=5)
    parser.add_argument(
        "--stage1_checkpoint_every_steps",
        type=int,
        default=0,
        help="Save ae_resume_latest.pt every N stage-1 optimizer steps. Set to 0 to disable intra-epoch snapshots.",
    )
    parser.add_argument("--stage1_lr", type=float, default=2e-4)
    parser.add_argument(
        "--stage1_dict_optimizer",
        type=str,
        default="shared_adam",
        choices=["shared_adam", "separate_sgd"],
        help="Stage-1 dictionary optimizer mode. 'shared_adam' is the stable default; 'separate_sgd' keeps the experimental split optimizer path.",
    )
    parser.add_argument(
        "--stage1_dict_lr_multiplier",
        type=float,
        default=1.0,
        help="Learning-rate multiplier used by the stage-1 dictionary SGD optimizer.",
    )
    parser.add_argument("--stage1_lr_schedule", type=str, default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--stage1_warmup_epochs", type=int, default=1)
    parser.add_argument("--stage1_min_lr_ratio", type=float, default=0.1)
    parser.add_argument(
        "--stage1_dict_lr_schedule",
        type=str,
        default="cosine",
        choices=["constant", "cosine"],
        help="Learning-rate schedule used by the stage-1 dictionary optimizer.",
    )
    parser.add_argument(
        "--stage1_dict_warmup_epochs",
        type=int,
        default=1,
        help="Warmup epochs for the stage-1 dictionary LR schedule.",
    )
    parser.add_argument(
        "--stage1_dict_min_lr_ratio",
        type=float,
        default=0.05,
        help="Minimum LR ratio for the stage-1 dictionary cosine schedule.",
    )
    parser.add_argument(
        "--stage1_dict_grad_clip",
        type=float,
        default=0.1,
        help="Clip norm applied only to the stage-1 dictionary gradient after tangent projection (<=0 disables).",
    )
    parser.add_argument(
        "--stage1_dict_max_update_norm",
        type=float,
        default=0.0,
        help="Maximum allowed normalized dictionary update per stage-1 step. >0 restores the previous dictionary on oversized updates.",
    )
    parser.add_argument(
        "--stage1_loss_spike_skip_ratio",
        type=float,
        default=0.0,
        help="Skip stage-1 updates when the loss exceeds this multiple of the running EMA. 0 disables the guard.",
    )
    parser.add_argument(
        "--stage1_loss_ema_beta",
        type=float,
        default=0.98,
        help="EMA coefficient used by the stage-1 loss-spike skip guard.",
    )
    parser.add_argument(
        "--stage1_bottleneck_weight_start",
        type=float,
        default=1.0,
        help="Initial stage-1 bottleneck weight at epoch 0 before ramping to --bottleneck_weight.",
    )
    parser.add_argument(
        "--stage1_bottleneck_warmup_epochs",
        type=int,
        default=0,
        help="Ramp stage-1 bottleneck weight from --stage1_bottleneck_weight_start to --bottleneck_weight over this many epochs.",
    )
    parser.add_argument(
        "--stage1_dict_loss_weight",
        type=float,
        default=float("nan"),
        help="Optional explicit stage-1 weight for the dictionary-fit term. NaN keeps the legacy coupled bottleneck objective.",
    )
    parser.add_argument(
        "--stage1_dict_loss_weight_start",
        type=float,
        default=float("nan"),
        help="Optional initial stage-1 dictionary-fit weight before ramping to --stage1_dict_loss_weight.",
    )
    parser.add_argument(
        "--stage1_dict_loss_warmup_epochs",
        type=int,
        default=0,
        help="Ramp stage-1 dictionary-fit weight from --stage1_dict_loss_weight_start to --stage1_dict_loss_weight over this many epochs.",
    )
    parser.add_argument(
        "--stage1_commitment_loss_weight",
        type=float,
        default=float("nan"),
        help="Optional explicit stage-1 weight for the encoder commitment term. NaN keeps the legacy coupled bottleneck objective.",
    )
    parser.add_argument(
        "--stage1_commitment_loss_weight_start",
        type=float,
        default=float("nan"),
        help="Optional initial stage-1 encoder commitment weight before ramping to --stage1_commitment_loss_weight.",
    )
    parser.add_argument(
        "--stage1_commitment_loss_warmup_epochs",
        type=int,
        default=0,
        help="Ramp stage-1 encoder commitment weight from --stage1_commitment_loss_weight_start to --stage1_commitment_loss_weight over this many epochs.",
    )
    parser.add_argument(
        "--stage1_amp",
        type=_parse_cli_bool,
        nargs="?",
        const=True,
        default=True,
        help="Enable CUDA AMP for stage-1 training. Ignored on CPU.",
    )
    parser.add_argument(
        "--stage1_amp_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16"],
        help="Autocast dtype for stage-1 AMP. 'auto' prefers bfloat16 when supported.",
    )
    parser.add_argument("--stage2_epochs", type=int, default=100)
    parser.add_argument("--stage2_lr", type=float, default=1e-3)
    parser.add_argument(
        "--stage2_rq_atom_loss_weight",
        type=float,
        default=1.0,
        help="Weight applied to atom-token CE in the quantized RQ stage-2 loss.",
    )
    parser.add_argument(
        "--stage2_rq_coeff_loss_weight",
        type=float,
        default=1.0,
        help="Weight applied to coefficient-bin CE in the quantized RQ stage-2 loss.",
    )
    parser.add_argument(
        "--stage2_coeff_loss_weight",
        type=float,
        default=0.1,
        help="Weight applied to the real-valued coefficient regression term during stage-2 training.",
    )
    parser.add_argument(
        "--stage2_coeff_loss_type",
        type=str,
        default=None,
        choices=["huber", "mse", "recon_mse", "gt_atom_recon_mse", "gaussian_nll"],
        help=(
            "Auxiliary loss used with real-valued sparse coefficients during stage-2 training: "
            "'huber'/'mse' regress normalized coefficients directly, while 'recon_mse' matches "
            "the latent reconstruction induced by predicted atoms+coeffs to the ground-truth sparse-code reconstruction, "
            "'gt_atom_recon_mse' matches the latent reconstruction induced by ground-truth atoms + predicted coeffs "
            "to the ground-truth sparse-code reconstruction, and 'gaussian_nll' trains a diagonal-Gaussian coefficient head. "
            "Defaults to 'gaussian_nll' for variational coeffs, 'mse' for deterministic real-valued coeffs, "
            "and 'gt_atom_recon_mse' for quantized coeffs."
        ),
    )
    parser.add_argument(
        "--stage2_coeff_norm_max",
        type=float,
        default=4.0,
        help="Legacy compatibility knob from the removed flattened-token transformer path; ignored by the spatial-depth prior.",
    )
    parser.add_argument(
        "--stage2_coeff_huber_delta",
        type=float,
        default=1.0,
        help="Delta parameter for Huber loss on normalized real-valued coefficients.",
    )
    parser.add_argument(
        "--coeff_depth_weighting",
        type=str,
        default="none",
        choices=["none", "linear", "inverse_rank"],
        help="Per-depth weighting for coeff CE. 'linear' weights earlier (larger-coeff) depths more; "
             "'inverse_rank' uses 1/rank weighting.",
    )
    parser.add_argument(
        "--coeff_focal_gamma",
        type=float,
        default=0.0,
        help="Focal loss gamma for coeff CE. 0 = standard CE; >0 down-weights easy (high-confidence) predictions.",
    )
    parser.add_argument(
        "--stage2_sched_sampling_final_prob",
        type=float,
        default=0.0,
        help="Deprecated compatibility flag. Stage-2 now always trains with pure teacher forcing.",
    )
    # ---- ordinal coefficient regression ----
    parser.add_argument("--ordinal_coeff_weight", type=float, default=0.0,
                        help="Weight for ordinal coefficient regression loss. 0 = disabled.")
    parser.add_argument("--ordinal_coeff_huber_delta", type=float, default=0.5,
                        help="Huber delta for ordinal coefficient regression.")
    parser.add_argument("--ordinal_magnitude_weighted", type=_parse_cli_bool, nargs="?", const=True, default=False,
                        help="Weight ordinal coeff loss by |gt_value|. Large coefficients get more gradient.")
    parser.add_argument("--ordinal_zero_drift_margin", type=float, default=0.0,
                        help="Penalise |pred| > margin when |gt| < threshold. 0 = disabled.")
    parser.add_argument("--ordinal_zero_drift_threshold", type=float, default=0.3,
                        help="GT magnitude below which zero-drift penalty applies.")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--stage2_batch_size", type=int, default=16)
    parser.add_argument("--stage2_warmup_steps", type=int, default=500,
                        help="Linear LR warmup steps for stage-2. 0 = no warmup.")
    parser.add_argument("--stage2_min_lr_ratio", type=float, default=0.01,
                        help="Minimum LR as fraction of peak LR for cosine decay.")
    parser.add_argument("--stage2_weight_decay", type=float, default=0.01,
                        help="AdamW weight decay for stage-2. 0 = plain Adam.")
    parser.add_argument(
        "--stage2_amp",
        type=_parse_cli_bool,
        nargs="?",
        const=True,
        default=True,
        help="Enable CUDA AMP for stage-2 training. Ignored on CPU.",
    )
    parser.add_argument(
        "--stage2_amp_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16"],
        help="Autocast dtype for stage-2 AMP. 'auto' prefers bfloat16 when supported.",
    )
    parser.add_argument(
        "--stage2_autoregressive_coeffs",
        dest="stage2_autoregressive_coeffs",
        action="store_true",
        default=True,
        help="Autoregress the interleaved stage-2 token stream. Required for quantized shared-vocab training.",
    )
    parser.add_argument(
        "--no_stage2_autoregressive_coeffs",
        dest="stage2_autoregressive_coeffs",
        action="store_false",
        help="Use support-only conditioning for the real-valued coefficient path.",
    )
    parser.add_argument("--bottleneck_weight", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--num_hiddens", type=int, default=128)
    parser.add_argument("--ae_num_downsamples", type=int, default=4)
    parser.add_argument("--num_res_layers", type=int, default=2)
    parser.add_argument("--num_res_hiddens", type=int, default=32)
    parser.add_argument("--embedding_dim", type=int, default=16,
                        help="Latent channel depth. Must be > sparsity_level to keep OMP well-conditioned.")
    parser.add_argument("--num_atoms", type=int, default=1024)
    parser.add_argument("--sparsity_level", type=int, default=8)
    parser.add_argument("--n_bins", type=int, default=2048)
    parser.add_argument("--coef_max", type=float, default=20.0)
    parser.add_argument(
        "--quantize_sparse_coeffs",
        type=_parse_cli_bool,
        nargs="?",
        const=True,
        default=True,
        help="Whether to quantize sparse coefficients. Accepts true/false; `--quantize_sparse_coeffs false` enables the real-valued path.",
    )
    parser.add_argument("--coef_quantization", type=str, default="uniform", choices=["uniform", "mu_law"])
    parser.add_argument("--coef_mu", type=float, default=0.0)
    parser.add_argument("--commitment_cost", type=float, default=0.25)
    parser.add_argument(
        "--variational_coeffs",
        type=_parse_cli_bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable a local Gaussian posterior around the deterministic sparse coefficients. Currently supported only when quantize_sparse_coeffs=false.",
    )
    parser.add_argument(
        "--variational_coeff_kl_weight",
        type=float,
        default=1e-3,
        help="Weight on the coefficient-posterior KL term during stage-1. Ignored when --variational_coeffs false.",
    )
    parser.add_argument(
        "--variational_coeff_prior_std",
        type=float,
        default=0.25,
        help="Reference std for the local Gaussian coefficient prior centered on the OMP coefficients.",
    )
    parser.add_argument(
        "--variational_coeff_min_std",
        type=float,
        default=0.01,
        help="Minimum std allowed for the variational coefficient posterior.",
    )
    parser.add_argument("--stage1_coherence_weight", type=float, default=0.0,
                        help="Weight on the dictionary coherence regularizer during stage-1.")
    parser.add_argument("--stage1_coherence_weight_start", type=float, default=float("nan"),
                        help="Initial stage-1 coherence weight before warmup. NaN reuses the final weight.")
    parser.add_argument("--stage1_coherence_warmup_epochs", type=int, default=0,
                        help="Warmup epochs for stage-1 coherence regularization.")
    parser.add_argument("--stage1_coherence_margin", type=float, default=0.0,
                        help="Only penalize absolute atom cosine similarity above this margin.")
    parser.add_argument(
        "--patch_based",
        dest="patch_based",
        action="store_true",
        default=False,
        help="Use patch-based dictionary learning bottleneck.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=4,
        help="Spatial size of each latent patch (patch_based only).",
    )
    parser.add_argument(
        "--patch_stride",
        type=int,
        default=2,
        help="Stride between patches; patch_size//2 gives 50%% overlap (patch_based only).",
    )
    parser.add_argument(
        "--patch_reconstruction",
        type=str,
        default="center_crop",
        choices=["center_crop", "hann", "tile"],
        help=(
            "Patch reconstruction mode: 'center_crop' (center-tile), "
            "'hann' (weighted overlap-add), or 'tile' (direct non-overlap stitching). "
            "When patch_stride == patch_size, overlap-aware modes are normalized to 'tile'."
        ),
    )

    parser.add_argument("--tf_d_model", type=int, default=512)
    parser.add_argument("--tf_heads", type=int, default=8)
    parser.add_argument("--tf_layers", type=int, default=12)
    parser.add_argument(
        "--stage2_arch",
        type=str,
        default="spatial_depth",
        choices=["spatial_depth", "mingpt"],
        help="Stage-2 prior architecture. 'mingpt' is currently supported for quantized sparse coefficients only.",
    )
    parser.add_argument(
        "--tf_global_tokens",
        type=int,
        default=0,
        help="Number of learned global prefix tokens prepended to the spatial stage.",
    )
    parser.add_argument("--tf_ff", type=int, default=1024)
    parser.add_argument("--tf_dropout", type=float, default=0.1)
    parser.add_argument(
        "--token_subset",
        type=int,
        default=0,
        help="Number of stage-1 token grids to encode for stage-2 training (<= 0 uses the full set).",
    )
    parser.add_argument(
        "--rebuild_token_cache",
        action="store_true",
        help="Ignore any existing stage-2 token cache and rebuild it from the stage-1 model.",
    )
    parser.add_argument(
        "--rfid_num_samples",
        type=int,
        default=256,
        help="Number of validation images used for stage-1 reconstruction FID (0 disables it).",
    )
    parser.add_argument(
        "--stage2_fid_num_samples",
        type=int,
        default=0,
        help="Number of validation/generated images used for stage-2 sample FID (0 disables it).",
    )
    parser.add_argument(
        "--stage2_fid_every_epochs",
        type=int,
        default=0,
        help="Compute stage-2 sample FID every N stage-2 epochs (0 disables it).",
    )
    parser.add_argument("--stage2_sample_every_steps", type=int, default=2000)
    parser.add_argument(
        "--stage2_sample_start_step",
        type=int,
        default=0,
        help="Delay stage-2 preview sampling until this global step (0 keeps the old immediate behavior).",
    )
    parser.add_argument("--stage2_sample_batch_size", type=int, default=32)
    parser.add_argument(
        "--stage2_sample_candidate_factor",
        type=int,
        default=4,
        help="Generate this many times more stage-2 candidates and keep the ones closest to stage-1 image stats.",
    )
    parser.add_argument("--stage2_sample_temperature", type=float, default=0.5)
    parser.add_argument("--stage2_sample_top_k", type=int, default=0)
    parser.add_argument(
        "--stage2_sample_coeff_temperature",
        type=float,
        default=float("nan"),
        help="Optional separate coefficient temperature for real-valued stage-2 sampling. NaN falls back to --stage2_sample_temperature.",
    )
    parser.add_argument(
        "--stage2_sample_coeff_mode",
        choices=["gaussian", "mean"],
        default="gaussian",
        help="How to sample real-valued coefficients for stage-2 sample grids.",
    )
    parser.add_argument(
        "--stage2_sample_quality_weight",
        type=float,
        default=1.0,
        help="Quality penalty used when reranking oversampled stage-2 preview candidates against stage-1 reference stats.",
    )
    parser.add_argument(
        "--stage2_sample_brightness_weight",
        type=float,
        default=1.0,
        help="Extra penalty for preview candidates that are darker than the stage-1 reference distribution.",
    )
    parser.add_argument(
        "--stage2_sample_overbright_weight",
        type=float,
        default=1.0,
        help="Extra penalty for preview candidates that are brighter than the stage-1 reference distribution.",
    )
    parser.add_argument(
        "--stage2_sample_reject_dark_z",
        type=float,
        default=1.5,
        help="Hard rejection threshold in reference-standard-deviation units for abnormally dark preview candidates. <= 0 disables rejection.",
    )
    parser.add_argument(
        "--stage2_sample_reject_bright_z",
        type=float,
        default=1.5,
        help="Hard rejection threshold in reference-standard-deviation units for abnormally bright preview candidates. <= 0 disables rejection.",
    )
    parser.add_argument(
        "--stage2_sample_selection_mode",
        choices=["quality_only", "diverse"],
        default="quality_only",
        help=(
            "How to rerank oversampled stage-2 preview candidates. "
            "'quality_only' is safer for color fidelity; 'diverse' can surface outlier hues/textures."
        ),
    )
    parser.add_argument(
        "--stage2_sample_sort_by_quality",
        dest="stage2_sample_sort_by_quality",
        action="store_true",
        default=True,
        help="Sort final stage-2 preview grids by quality so earlier tile indices correspond to safer samples.",
    )
    parser.add_argument(
        "--no_stage2_sample_sort_by_quality",
        dest="stage2_sample_sort_by_quality",
        action="store_false",
        help="Keep the diversity-selection order in stage-2 preview grids.",
    )
    parser.add_argument("--stage2_sample_image_size", type=int, default=128)

    args = parser.parse_args()
    wandb_run = None
    distributed = False

    if args.ae_num_downsamples <= 0:
        raise ValueError(f"ae_num_downsamples must be positive, got {args.ae_num_downsamples}")
    if args.coef_quantization == "mu_law" and args.coef_mu <= 0.0:
        raise ValueError(f"coef_mu must be > 0 when coef_quantization='mu_law', got {args.coef_mu}")
    if args.stage2_sample_temperature <= 0.0:
        raise ValueError("stage2_sample_temperature must be > 0.")
    if args.stage2_sample_start_step < 0:
        raise ValueError("stage2_sample_start_step must be >= 0.")
    if math.isfinite(args.stage2_sample_coeff_temperature) and args.stage2_sample_coeff_temperature <= 0.0:
        raise ValueError("stage2_sample_coeff_temperature must be > 0 when set.")
    if args.stage2_sample_quality_weight < 0.0:
        raise ValueError("stage2_sample_quality_weight must be >= 0.")
    if args.stage2_sample_brightness_weight < 0.0:
        raise ValueError("stage2_sample_brightness_weight must be >= 0.")
    if args.stage2_sample_overbright_weight < 0.0:
        raise ValueError("stage2_sample_overbright_weight must be >= 0.")
    if args.stage2_sample_reject_dark_z < 0.0:
        raise ValueError("stage2_sample_reject_dark_z must be >= 0.")
    if args.stage2_sample_reject_bright_z < 0.0:
        raise ValueError("stage2_sample_reject_bright_z must be >= 0.")
    if args.rfid_num_samples < 0:
        raise ValueError("rfid_num_samples must be >= 0.")
    if args.stage2_fid_num_samples < 0:
        raise ValueError("stage2_fid_num_samples must be >= 0.")
    if args.stage2_fid_every_epochs < 0:
        raise ValueError("stage2_fid_every_epochs must be >= 0.")
    if args.stage1_dict_lr_multiplier <= 0.0:
        raise ValueError("stage1_dict_lr_multiplier must be > 0.")
    if args.stage1_dict_warmup_epochs < 0:
        raise ValueError("stage1_dict_warmup_epochs must be >= 0.")
    if not (0.0 <= args.stage1_dict_min_lr_ratio <= 1.0):
        raise ValueError("stage1_dict_min_lr_ratio must be in [0, 1].")
    if args.stage1_dict_grad_clip < 0.0:
        raise ValueError("stage1_dict_grad_clip must be >= 0.")
    if args.stage1_dict_max_update_norm < 0.0:
        raise ValueError("stage1_dict_max_update_norm must be >= 0.")
    if args.stage1_loss_spike_skip_ratio < 0.0:
        raise ValueError("stage1_loss_spike_skip_ratio must be >= 0.")
    if not (0.0 <= args.stage1_loss_ema_beta < 1.0):
        raise ValueError("stage1_loss_ema_beta must be in [0, 1).")
    if args.stage1_bottleneck_warmup_epochs < 0:
        raise ValueError("stage1_bottleneck_warmup_epochs must be >= 0.")
    if args.stage1_dict_loss_warmup_epochs < 0:
        raise ValueError("stage1_dict_loss_warmup_epochs must be >= 0.")
    if args.stage1_commitment_loss_warmup_epochs < 0:
        raise ValueError("stage1_commitment_loss_warmup_epochs must be >= 0.")
    if math.isfinite(args.stage1_dict_loss_weight) and args.stage1_dict_loss_weight < 0.0:
        raise ValueError("stage1_dict_loss_weight must be >= 0 when set.")
    if math.isfinite(args.stage1_dict_loss_weight_start) and args.stage1_dict_loss_weight_start < 0.0:
        raise ValueError("stage1_dict_loss_weight_start must be >= 0 when set.")
    if math.isfinite(args.stage1_commitment_loss_weight) and args.stage1_commitment_loss_weight < 0.0:
        raise ValueError("stage1_commitment_loss_weight must be >= 0 when set.")
    if math.isfinite(args.stage1_commitment_loss_weight_start) and args.stage1_commitment_loss_weight_start < 0.0:
        raise ValueError("stage1_commitment_loss_weight_start must be >= 0 when set.")
    if args.stage2_coeff_loss_weight < 0.0:
        raise ValueError("stage2_coeff_loss_weight must be >= 0.")
    if args.variational_coeff_kl_weight < 0.0:
        raise ValueError("variational_coeff_kl_weight must be >= 0.")
    if args.variational_coeff_prior_std <= 0.0:
        raise ValueError("variational_coeff_prior_std must be > 0.")
    if args.variational_coeff_min_std <= 0.0:
        raise ValueError("variational_coeff_min_std must be > 0.")
    if args.variational_coeff_min_std > args.variational_coeff_prior_std:
        raise ValueError("variational_coeff_min_std cannot exceed variational_coeff_prior_std.")
    if args.stage2_rq_atom_loss_weight < 0.0:
        raise ValueError("stage2_rq_atom_loss_weight must be >= 0.")
    if args.stage2_rq_coeff_loss_weight < 0.0:
        raise ValueError("stage2_rq_coeff_loss_weight must be >= 0.")
    if args.stage2_coeff_huber_delta <= 0.0:
        raise ValueError("stage2_coeff_huber_delta must be > 0.")
    if not (0.0 <= args.stage2_sched_sampling_final_prob <= 1.0):
        raise ValueError("stage2_sched_sampling_final_prob must be in [0, 1].")
    if args.quantize_sparse_coeffs and (not args.stage2_autoregressive_coeffs):
        raise ValueError(
            "quantized stage-2 training requires stage2_autoregressive_coeffs=true "
            "so the shared atom/coeff token stream stays interleaved."
        )
    if args.tf_global_tokens < 0:
        raise ValueError("tf_global_tokens must be >= 0.")
    if args.stage2_coeff_loss_type is None:
        if args.variational_coeffs:
            args.stage2_coeff_loss_type = "gaussian_nll"
        elif not args.quantize_sparse_coeffs:
            args.stage2_coeff_loss_type = "mse"
        else:
            args.stage2_coeff_loss_type = "gt_atom_recon_mse"
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                "[Config] stage2_coeff_loss_type not set; using "
                f"{args.stage2_coeff_loss_type!r} for "
                f"{'variational' if args.variational_coeffs else ('deterministic real-valued' if not args.quantize_sparse_coeffs else 'quantized')} "
                "stage-2 coefficients."
            )
    resolved_stage2_coeff_loss_type = str(args.stage2_coeff_loss_type).strip().lower()
    args.stage2_coeff_loss_type = resolved_stage2_coeff_loss_type
    if args.variational_coeffs:
        if args.quantize_sparse_coeffs:
            raise ValueError("variational_coeffs requires quantize_sparse_coeffs=false")
        if args.stage2_epochs > 0 and resolved_stage2_coeff_loss_type != "gaussian_nll":
            raise ValueError(
                "variational_coeffs currently requires --stage2_coeff_loss_type gaussian_nll for stage-2 training"
            )
    if (
        args.stage2_epochs > 0
        and (not args.quantize_sparse_coeffs)
        and (not args.variational_coeffs)
        and resolved_stage2_coeff_loss_type == "gt_atom_recon_mse"
        and int(os.environ.get("RANK", "0")) == 0
    ):
        print(
            "[Warn] stage2_coeff_loss_type='gt_atom_recon_mse' on deterministic real-valued coefficients "
            "has been empirically unstable and can collapse into blank or near-white samples; "
            "prefer 'mse' or enable variational_coeffs with 'gaussian_nll'.",
            file=sys.stderr,
        )
    args.stage1_amp_dtype = str(args.stage1_amp_dtype).strip().lower()
    args.stage2_amp_dtype = str(args.stage2_amp_dtype).strip().lower()
    if args.token_subset < 0:
        args.token_subset = 0
    if args.dist_timeout_minutes <= 0:
        raise ValueError("dist_timeout_minutes must be > 0.")
    if args.stage1_source_run is not None:
        args.stage1_source_run = str(args.stage1_source_run).strip()
        if not args.stage1_source_run:
            args.stage1_source_run = None
    if args.stage1_source_ckpt is not None:
        args.stage1_source_ckpt = str(args.stage1_source_ckpt).strip()
        if not args.stage1_source_ckpt:
            args.stage1_source_ckpt = None
    rank0 = int(os.environ.get("RANK", "0")) == 0
    uses_explicit_stage1_dict_weight = math.isfinite(args.stage1_dict_loss_weight)
    uses_explicit_stage1_commitment_weight = math.isfinite(args.stage1_commitment_loss_weight)
    if rank0 and (uses_explicit_stage1_dict_weight or uses_explicit_stage1_commitment_weight):
        print(
            "[Config] explicit stage-1 dict/commitment weights are enabled, so "
            "--bottleneck_weight is only a fallback for any term whose explicit weight is left unset.",
            file=sys.stderr,
        )
    if (
        rank0
        and args.stage1_bottleneck_warmup_epochs > 0
        and float(args.stage1_bottleneck_weight_start) > float(args.bottleneck_weight)
    ):
        print(
            "[Warn] stage1_bottleneck_weight_start > bottleneck_weight, so stage-1 bottleneck pressure "
            "will anneal down over the warmup window instead of ramping up.",
            file=sys.stderr,
        )
    if (
        rank0
        and uses_explicit_stage1_dict_weight
        and math.isfinite(args.stage1_dict_loss_weight_start)
        and args.stage1_dict_loss_warmup_epochs > 0
        and float(args.stage1_dict_loss_weight_start) > float(args.stage1_dict_loss_weight)
    ):
        print(
            "[Warn] stage1_dict_loss_weight_start > stage1_dict_loss_weight, so dictionary-fit pressure "
            "will anneal down over the warmup window instead of ramping up.",
            file=sys.stderr,
        )
    if (
        rank0
        and uses_explicit_stage1_commitment_weight
        and math.isfinite(args.stage1_commitment_loss_weight_start)
        and args.stage1_commitment_loss_warmup_epochs > 0
        and float(args.stage1_commitment_loss_weight_start) > float(args.stage1_commitment_loss_weight)
    ):
        print(
            "[Warn] stage1_commitment_loss_weight_start > stage1_commitment_loss_weight, so encoder commitment "
            "pressure will anneal down over the warmup window instead of ramping up.",
            file=sys.stderr,
        )
    if args.variational_coeffs and args.stage1_epochs <= 0 and args.stage1_source_ckpt is None and args.stage1_source_run is None:
        raise ValueError(
            "variational_coeffs requires either stage1_epochs > 0 or an explicit variational stage-1 source checkpoint/run"
        )
    default_stage1_ckpt = _default_stage1_source_ckpt(bool(args.quantize_sparse_coeffs))
    # Fresh stage-1 training should start from random init unless the caller
    # explicitly provides a checkpoint or run to warm-start from.
    stage1_init_source_ckpt = args.stage1_source_ckpt
    stage1_init_source_run = args.stage1_source_run
    if (
        args.stage1_source_ckpt is None
        and args.stage1_source_run is None
        and args.stage1_epochs <= 0
        and (not args.variational_coeffs)
    ):
        args.stage1_source_run = DEFAULT_STAGE1_SOURCE_RUN
    if args.stage2_source_run is not None:
        args.stage2_source_run = str(args.stage2_source_run).strip()
        if not args.stage2_source_run:
            args.stage2_source_run = None
    if args.stage2_source_token_cache is not None:
        args.stage2_source_token_cache = str(args.stage2_source_token_cache).strip()
        if not args.stage2_source_token_cache:
            args.stage2_source_token_cache = None
    if args.stage2_source_ckpt is not None:
        args.stage2_source_ckpt = str(args.stage2_source_ckpt).strip()
        if not args.stage2_source_ckpt:
            args.stage2_source_ckpt = None
    if (
        args.stage2_source_run is None
        and args.stage2_source_token_cache is None
        and args.stage1_epochs <= 0
        and args.stage1_source_run == DEFAULT_STAGE1_SOURCE_RUN
        and (not args.variational_coeffs)
    ):
        args.stage2_source_run = DEFAULT_STAGE2_SOURCE_RUN
    if args.image_size is None:
        args.image_size = _default_image_size(args.dataset)
    args.image_size = int(args.image_size)
    if args.data_dir is None:
        args.data_dir = str(_default_data_dir(args.dataset))
    if args.wandb_name is None:
        args.wandb_name = _default_run_name(args.dataset, args.image_size, args.quantize_sparse_coeffs)
    if args.out_dir is None:
        args.out_dir = str(_default_out_dir(args.dataset, args.image_size, args.quantize_sparse_coeffs))
    experiment_root = Path(args.out_dir).expanduser().resolve()

    launch_timestamp_env = os.environ.get("PROTO_LAUNCH_TIMESTAMP")
    preinit_wandb = False
    if launch_timestamp_env:
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        distributed = world_size > 1
        is_main_process = (rank == 0)
        launch_timestamp = launch_timestamp_env
        preinit_wandb = distributed
    else:
        distributed, rank, local_rank, world_size = _init_distributed(args.dist_timeout_minutes)
        is_main_process = (rank == 0)
        launch_timestamp = _launch_timestamp()
    run_out_dir = _resolve_run_out_dir(str(experiment_root), launch_timestamp)
    args.out_root = str(experiment_root)
    args.launch_timestamp = launch_timestamp
    args.out_dir = str(run_out_dir)
    args.run_id = _resolve_run_id(getattr(args, "run_id", None), launch_timestamp)
    if args.wandb_dir == "./wandb":
        args.wandb_dir = str(Path(args.out_dir) / "wandb")

    if is_main_process:
        os.makedirs(experiment_root, exist_ok=True)
        os.makedirs(args.out_dir, exist_ok=True)
    stage1_dir = os.path.join(args.out_dir, "stage1")
    stage2_dir = os.path.join(args.out_dir, "stage2")
    args.artifact_tag = _build_run_artifact_tag(args)
    if is_main_process:
        os.makedirs(stage1_dir, exist_ok=True)
        os.makedirs(stage2_dir, exist_ok=True)
        if args.run_id:
            print(f"[Setup] run_id={args.run_id}")
        print(f"[Setup] artifact_tag={args.artifact_tag}")
    if (
        args.stage1_auto_resume_from_latest
        and stage1_init_source_ckpt is None
        and stage1_init_source_run is None
        and args.stage1_epochs > 0
    ):
        auto_resume_path = _find_latest_stage1_autoresume_checkpoint(experiment_root, run_out_dir)
        if auto_resume_path is not None:
            stage1_init_source_ckpt = str(auto_resume_path)
            if is_main_process:
                print(f"[Stage1] auto-resuming from latest checkpoint under {experiment_root}: {auto_resume_path}")

    if preinit_wandb and is_main_process:
        wandb_run = _init_wandb(args)
    if preinit_wandb:
        distributed, rank, local_rank, world_size = _init_distributed(args.dist_timeout_minutes)
        is_main_process = (rank == 0)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")

    if distributed:
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    _barrier()

    if is_main_process:
        print(
            f"[Setup] device={device} world_size={world_size} dataset={args.dataset} "
            f"data_dir={args.data_dir} image_size={args.image_size}"
        )
        print(f"[Setup] experiment_root={experiment_root} run_out_dir={args.out_dir}")
        if wandb_run is None:
            wandb_run = _init_wandb(args)
    if wandb_run is not None:
        _log_wandb(
            wandb_run,
            {
                "setup/device": str(device),
                "setup/dataset": args.dataset,
            },
        )

    def _build_laser() -> LASER:
        return LASER(
            in_channels=3,
            num_hiddens=args.num_hiddens,
            num_downsamples=args.ae_num_downsamples,
            num_residual_layers=args.num_res_layers,
            resolution=args.image_size,
            embedding_dim=args.embedding_dim,
            num_embeddings=args.num_atoms,
            sparsity_level=args.sparsity_level,
            commitment_cost=args.commitment_cost,
            n_bins=args.n_bins,
            coef_max=args.coef_max,
            quantize_sparse_coeffs=args.quantize_sparse_coeffs,
            coef_quantization=args.coef_quantization,
            coef_mu=args.coef_mu,
            out_tanh=True,
            patch_based=args.patch_based,
            patch_size=args.patch_size,
            patch_stride=args.patch_stride,
            patch_reconstruction=args.patch_reconstruction,
            variational_coeffs=args.variational_coeffs,
            variational_coeff_kl_weight=args.variational_coeff_kl_weight,
            variational_coeff_prior_std=args.variational_coeff_prior_std,
            variational_coeff_min_std=args.variational_coeff_min_std,
        )

    def _prepare_stage1_source_checkpoint(run_ref: str) -> Path:
        ready_path = Path(stage1_dir) / "stage1_source.ready"
        error_path = Path(stage1_dir) / "stage1_source.failed"
        path_record = Path(stage1_dir) / "stage1_source.path"
        description = f"stage-1 source checkpoint for W&B run {run_ref}"
        if is_main_process:
            _unlink_if_exists(ready_path)
            _unlink_if_exists(error_path)
            _unlink_if_exists(path_record)
            try:
                source_path = _resolve_stage1_checkpoint_from_wandb_run(
                    run_ref,
                    run_out_dir,
                    quantized=bool(args.quantize_sparse_coeffs),
                )
                _write_atomic_text(path_record, f"{source_path}\n")
                _write_atomic_text(
                    ready_path,
                    f"ready {datetime.datetime.now().isoformat()} {source_path}\n",
                )
            except Exception as exc:
                _write_atomic_text(
                    error_path,
                    f"[Stage1] source checkpoint preparation failed: {type(exc).__name__}: {exc}\n",
                )
                raise
        elif distributed:
            _wait_for_file_signal(
                ready_path,
                error_path,
                timeout_seconds=max(60.0, float(args.dist_timeout_minutes) * 60.0),
                description=description,
            )
        try:
            source_path = Path(path_record.read_text(encoding="utf-8").strip()).expanduser().resolve()
        except OSError as exc:
            raise RuntimeError(f"Could not read prepared source checkpoint path from {path_record}: {exc}") from exc
        if not source_path.exists():
            raise FileNotFoundError(f"Prepared stage-1 source checkpoint does not exist: {source_path}")
        return source_path

    def _load_best_laser_weights(laser_model: LASER) -> Path:
        best_path = Path(stage1_dir) / "ae_best.pt"
        if best_path.exists():
            if is_main_process and args.stage1_epochs > 0:
                print(f"[Stage1] loading trained checkpoint from current run: {best_path}")
        elif args.stage1_source_ckpt is not None:
            best_path = Path(args.stage1_source_ckpt).expanduser().resolve()
            if not best_path.exists():
                raise FileNotFoundError(f"Requested stage-1 source checkpoint does not exist: {best_path}")
            if is_main_process:
                print(f"[Stage1] using local checkpoint from {best_path}")
        elif args.stage1_epochs <= 0 and args.stage1_source_run is not None:
            best_path = _prepare_stage1_source_checkpoint(args.stage1_source_run)
            if is_main_process:
                print(f"[Stage1] using checkpoint from W&B run {args.stage1_source_run}: {best_path}")
        else:
            fallback_path = _find_latest_stage1_checkpoint(experiment_root, run_out_dir)
            if fallback_path is None:
                raise FileNotFoundError(
                    f"Stage-1 checkpoint not found in current run at {best_path} "
                    f"or prior runs under {experiment_root}"
                )
            if is_main_process:
                print(f"[Stage1] reusing prior checkpoint from {fallback_path}")
            best_path = fallback_path
        _load_module_checkpoint(laser_model, best_path)
        return best_path

    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    train_tfm = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normalize,
    ])
    eval_tfm = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    if is_main_process:
        print(f"[Startup] building datasets for {args.dataset} (num_workers={args.num_workers}, token_num_workers={args.token_num_workers})")
    if args.dataset == "cifar10":
        train_set = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=train_tfm)
        val_set = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=eval_tfm)
        stage2_source_set = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=eval_tfm)
    elif args.dataset == "celeba":
        packed_path = _packed_celeba_file(Path(args.data_dir).expanduser().resolve(), args.image_size)
        if packed_path.exists():
            if is_main_process:
                print(f"[Data] using packed CelebA dataset at {packed_path}")
            train_full = PackedRGBImageDataset(root=args.data_dir, image_size=args.image_size, random_horizontal_flip=True)
            val_full = PackedRGBImageDataset(root=args.data_dir, image_size=args.image_size, random_horizontal_flip=False)
            token_full = PackedRGBImageDataset(root=args.data_dir, image_size=args.image_size, random_horizontal_flip=False)
        else:
            if is_main_process:
                print(f"[Data] packed CelebA dataset not found at {packed_path}; falling back to raw image tree")
            train_full = FlatImageDataset(root=args.data_dir, transform=train_tfm)
            val_full = FlatImageDataset(root=args.data_dir, transform=eval_tfm)
            token_full = FlatImageDataset(root=args.data_dir, transform=eval_tfm)
        if len(train_full) < 2:
            raise RuntimeError("CelebA dataset needs at least 2 images for train/val split.")
        val_size = max(1, int(0.05 * len(train_full)))
        train_size = len(train_full) - val_size
        indices = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(args.seed)).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        train_set = Subset(train_full, train_indices)
        val_set = Subset(val_full, val_indices)
        stage2_source_set = Subset(token_full, train_indices)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    if is_main_process:
        print(f"[Startup] dataset split ready: train={len(train_set)} val={len(val_set)} stage2_source={len(stage2_source_set)}")

    train_sampler = DistributedSampler(train_set, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if distributed else None
    val_num_workers = max(0, args.num_workers // 2)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=min(64, args.batch_size),
        shuffle=False,
        sampler=val_sampler,
        num_workers=val_num_workers,
        pin_memory=pin_memory,
        persistent_workers=(val_num_workers > 0),
    )
    rfid_loader = None
    if is_main_process and args.rfid_num_samples > 0:
        rfid_loader = DataLoader(
            val_set,
            batch_size=min(32, min(64, args.batch_size)),
            shuffle=False,
            num_workers=val_num_workers,
            pin_memory=pin_memory,
            persistent_workers=(val_num_workers > 0),
        )
    stage2_fid_loader = None
    if is_main_process and args.stage2_fid_num_samples > 0:
        stage2_fid_loader = DataLoader(
            val_set,
            batch_size=min(32, min(64, args.batch_size)),
            shuffle=False,
            num_workers=val_num_workers,
            pin_memory=pin_memory,
            persistent_workers=(val_num_workers > 0),
        )
    if is_main_process:
        print("[Startup] dataloaders ready")

    laser = _build_laser().to(device)
    if stage1_init_source_ckpt is not None:
        stage1_init_path = Path(stage1_init_source_ckpt).expanduser().resolve()
        if not stage1_init_path.exists():
            raise FileNotFoundError(f"Requested stage-1 init checkpoint does not exist: {stage1_init_path}")
        if is_main_process:
            print(f"[Stage1] warm-starting from local checkpoint: {stage1_init_path}")
        _load_module_checkpoint(laser, stage1_init_path)
    elif stage1_init_source_run is not None:
        stage1_init_path = _prepare_stage1_source_checkpoint(stage1_init_source_run)
        if is_main_process:
            print(f"[Stage1] warm-starting from W&B run {stage1_init_source_run}: {stage1_init_path}")
        _load_module_checkpoint(laser, stage1_init_path)
    laser_stage1 = DDP(laser, device_ids=[local_rank], output_device=local_rank) if distributed else laser
    if is_main_process:
        print(f"[Startup] entering stage1 train loop (epochs={args.stage1_epochs}, batch_size={args.batch_size}, lr={args.stage1_lr})")
    if args.stage1_epochs > 0:
        train_stage1_ae(
            ae=laser_stage1,
            train_loader=train_loader,
            val_loader=val_loader,
            rfid_loader=rfid_loader,
            device=device,
            epochs=args.stage1_epochs,
            lr=args.stage1_lr,
            bottleneck_weight=args.bottleneck_weight,
            grad_clip=args.grad_clip,
            stage1_amp=args.stage1_amp,
            stage1_amp_dtype=args.stage1_amp_dtype,
            out_dir=stage1_dir,
            rfid_num_samples=args.rfid_num_samples,
            lr_schedule=args.stage1_lr_schedule,
            warmup_epochs=args.stage1_warmup_epochs,
            min_lr_ratio=args.stage1_min_lr_ratio,
            dict_optimizer=args.stage1_dict_optimizer,
            dict_lr_multiplier=args.stage1_dict_lr_multiplier,
            dict_lr_schedule=args.stage1_dict_lr_schedule,
            dict_warmup_epochs=args.stage1_dict_warmup_epochs,
            dict_min_lr_ratio=args.stage1_dict_min_lr_ratio,
            dict_grad_clip=args.stage1_dict_grad_clip,
            dict_max_update_norm=args.stage1_dict_max_update_norm,
            loss_spike_skip_ratio=args.stage1_loss_spike_skip_ratio,
            loss_ema_beta=args.stage1_loss_ema_beta,
            bottleneck_weight_start=args.stage1_bottleneck_weight_start,
            bottleneck_warmup_epochs=args.stage1_bottleneck_warmup_epochs,
            dict_loss_weight=args.stage1_dict_loss_weight,
            dict_loss_weight_start=args.stage1_dict_loss_weight_start,
            dict_loss_warmup_epochs=args.stage1_dict_loss_warmup_epochs,
            commitment_loss_weight=args.stage1_commitment_loss_weight,
            commitment_loss_weight_start=args.stage1_commitment_loss_weight_start,
            commitment_loss_warmup_epochs=args.stage1_commitment_loss_warmup_epochs,
            coherence_weight=args.stage1_coherence_weight,
            coherence_weight_start=args.stage1_coherence_weight_start,
            coherence_warmup_epochs=args.stage1_coherence_warmup_epochs,
            coherence_margin=args.stage1_coherence_margin,
            checkpoint_every_steps=args.stage1_checkpoint_every_steps,
            train_sampler=train_sampler,
            is_main_process=is_main_process,
            wandb_run=wandb_run,
            artifact_tag=args.artifact_tag,
        )
    _barrier()

    stage1_best_path = _load_best_laser_weights(laser)
    laser = laser.to(device)

    if args.analyze_spectrum and is_main_process:
        spectrum_loader = DataLoader(
            val_set,
            batch_size=min(64, args.batch_size),
            shuffle=False,
            num_workers=val_num_workers,
            pin_memory=pin_memory,
            persistent_workers=(val_num_workers > 0),
        )
        analyze_patch_spectrum(
            laser,
            spectrum_loader,
            device,
            n_patches=args.spectrum_n_patches,
        )
        if wandb_run is not None:
            _finish_wandb(wandb_run)
        _cleanup_distributed()
        return

    token_cache_path = Path(stage2_dir) / "tokens_cache.pt"
    token_cache_tagged_path = _tagged_artifact_path(token_cache_path, args.artifact_tag)
    token_cache_ready_path = Path(stage2_dir) / "tokens_cache.ready"
    token_cache_error_path = Path(stage2_dir) / "tokens_cache.failed"
    token_subset = None if args.token_subset <= 0 else min(args.token_subset, len(stage2_source_set))
    expected_token_meta = _expected_token_cache_meta(
        args,
        stage2_source_set,
        token_subset,
        laser,
        stage1_best_path,
    )
    _barrier()
    if is_main_process:
        _unlink_if_exists(token_cache_ready_path)
        _unlink_if_exists(token_cache_error_path)
        try:
            cache_ready = False
            current_cache_reason = None
            requested_token_cache_path = None
            requested_token_cache_desc = None
            if not args.rebuild_token_cache:
                if args.stage2_source_token_cache is not None:
                    requested_token_cache_path = Path(args.stage2_source_token_cache).expanduser().resolve()
                    if not requested_token_cache_path.exists():
                        raise FileNotFoundError(
                            f"Requested stage-2 token cache does not exist: {requested_token_cache_path}"
                        )
                    requested_token_cache_desc = f"local path {requested_token_cache_path}"
                elif args.stage2_source_run is not None:
                    requested_token_cache_path = _resolve_stage2_token_cache_from_wandb_run(
                        args.stage2_source_run,
                        run_out_dir,
                    )
                    requested_token_cache_desc = f"W&B run {args.stage2_source_run}"

            if (not cache_ready) and (not args.rebuild_token_cache) and requested_token_cache_path is not None:
                try:
                    token_cache = _load_token_cache(str(requested_token_cache_path))
                    compatible, reason = _token_cache_is_compatible(token_cache, expected_token_meta)
                    if compatible:
                        if requested_token_cache_path.resolve() != token_cache_path.resolve():
                            shutil.copy2(requested_token_cache_path, token_cache_path)
                        _copy_artifact_to_tagged_path(token_cache_path, args.artifact_tag)
                        tokens_flat = token_cache["tokens_flat"]
                        H, W, D = token_cache["shape"]
                        print(
                            f"[Stage2] using token cache from {requested_token_cache_desc}: "
                            f"{tokens_flat.shape} (H={H}, W={W}, D={D}) at {requested_token_cache_path}"
                        )
                        cache_ready = True
                    else:
                        current_cache_reason = (
                            f"requested token cache from {requested_token_cache_desc} incompatible ({reason})"
                        )
                except Exception as exc:
                    current_cache_reason = (
                        f"could not load requested token cache from {requested_token_cache_desc} "
                        f"({type(exc).__name__}: {exc})"
                    )
            if (not cache_ready) and (not args.rebuild_token_cache) and requested_token_cache_path is None:
                try:
                    if token_cache_path.exists():
                        token_cache = _load_token_cache(str(token_cache_path))
                        compatible, reason = _token_cache_is_compatible(token_cache, expected_token_meta)
                        if compatible:
                            tokens_flat = token_cache["tokens_flat"]
                            H, W, D = token_cache["shape"]
                            _copy_artifact_to_tagged_path(token_cache_path, args.artifact_tag)
                            print(
                                f"[Stage2] reusing token cache: {tokens_flat.shape} "
                                f"(H={H}, W={W}, D={D}) from {token_cache_path}"
                            )
                            cache_ready = True
                        else:
                            current_cache_reason = reason
                except Exception as exc:
                    current_cache_reason = f"could not load current token cache ({type(exc).__name__}: {exc})"
            if (not cache_ready) and (not args.rebuild_token_cache) and requested_token_cache_path is None:
                fallback_cache_path = _find_latest_stage2_token_cache(experiment_root, run_out_dir)
                if fallback_cache_path is not None:
                    try:
                        token_cache = _load_token_cache(str(fallback_cache_path))
                        compatible, reason = _token_cache_is_compatible(token_cache, expected_token_meta)
                        if compatible:
                            shutil.copy2(fallback_cache_path, token_cache_path)
                            _copy_artifact_to_tagged_path(token_cache_path, args.artifact_tag)
                            tokens_flat = token_cache["tokens_flat"]
                            H, W, D = token_cache["shape"]
                            print(
                                f"[Stage2] reusing prior token cache: {tokens_flat.shape} "
                                f"(H={H}, W={W}, D={D}) from {fallback_cache_path}"
                            )
                            cache_ready = True
                        elif current_cache_reason is None:
                            current_cache_reason = f"prior token cache incompatible ({reason})"
                    except Exception as exc:
                        if current_cache_reason is None:
                            current_cache_reason = (
                                f"could not load prior token cache ({type(exc).__name__}: {exc})"
                            )
            if (not cache_ready) and (not args.rebuild_token_cache) and current_cache_reason is not None:
                print(f"[Stage2] rebuilding token cache ({current_cache_reason})")
            elif args.rebuild_token_cache:
                print("[Stage2] rebuilding token cache (--rebuild_token_cache)")

            if not cache_ready:
                token_source_loader = DataLoader(
                    stage2_source_set,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.token_num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=(args.token_num_workers > 0),
                )
                tokens_flat, coeffs_flat, H, W, D = precompute_tokens(
                    laser,
                    token_source_loader,
                    device,
                    max_items=token_subset,
                )
                cache = {
                    "tokens_flat": tokens_flat,
                    "shape": (H, W, D),
                    "meta": expected_token_meta,
                }
                if coeffs_flat is not None:
                    cache["coeffs_flat"] = coeffs_flat
                torch.save(cache, str(token_cache_path))
                _copy_artifact_to_tagged_path(token_cache_path, args.artifact_tag)
                print(f"[Stage2] token dataset: {tokens_flat.shape} (H={H}, W={W}, D={D})")
            _write_atomic_text(
                token_cache_ready_path,
                f"ready {datetime.datetime.now().isoformat()} {token_cache_path} tagged={token_cache_tagged_path}\n",
            )
        except Exception as exc:
            _write_atomic_text(
                token_cache_error_path,
                f"[Stage2] token cache preparation failed: {type(exc).__name__}: {exc}\n",
            )
            raise
    else:
        _wait_for_file_signal(
            token_cache_ready_path,
            token_cache_error_path,
            timeout_seconds=max(60.0, float(args.dist_timeout_minutes) * 60.0),
            description=f"stage-2 token cache at {token_cache_path}",
        )
    token_cache = _load_token_cache(str(token_cache_path))
    tokens_flat = token_cache["tokens_flat"]
    coeffs_flat = token_cache.get("coeffs_flat", None)
    H, W, D = token_cache["shape"]
    expected_items = int(expected_token_meta["effective_items"])
    if tokens_flat.size(0) > expected_items:
        tokens_flat = tokens_flat[:expected_items]
        if coeffs_flat is not None:
            coeffs_flat = coeffs_flat[:expected_items]
    if coeffs_flat is not None:
        coeffs_flat_raw = coeffs_flat
        coeffs_flat = _project_real_valued_stage2_coeffs(laser, coeffs_flat.to(torch.float32))
        if is_main_process:
            coeff_delta = (coeffs_flat_raw.to(torch.float32) - coeffs_flat).abs()
            clipped_fraction = float((coeff_delta > 1e-6).float().mean().item())
            if clipped_fraction > 0.0:
                print(
                    "[Stage2] projecting cached real-valued coeffs into bounded stage-2 space "
                    f"(clipped_fraction={clipped_fraction:.4f})"
                )
    real_valued = (coeffs_flat is not None)
    sample_reference_stats = _compute_stage2_sample_reference_stats(
        laser,
        tokens_flat,
        coeffs_flat,
        H,
        W,
        D,
        device,
    )

    from torch.utils.data import TensorDataset
    if real_valued:
        token_dataset = TensorDataset(tokens_flat, coeffs_flat)
    else:
        token_dataset = tokens_flat
    token_sampler = DistributedSampler(token_dataset, shuffle=True) if distributed else None
    token_loader = DataLoader(
        token_dataset,
        batch_size=args.stage2_batch_size,
        shuffle=(token_sampler is None),
        sampler=token_sampler,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=(len(token_dataset) >= args.stage2_batch_size),
    )
    transformer = build_stage2_model(
        laser.bottleneck,
        stage2_arch=args.stage2_arch,
        H=H,
        W=W,
        D=D,
        d_model=args.tf_d_model,
        n_heads=args.tf_heads,
        n_layers=args.tf_layers,
        d_ff=args.tf_ff,
        dropout=args.tf_dropout,
        n_global_spatial_tokens=args.tf_global_tokens,
        real_valued_coeffs=real_valued,
        coeff_max_fallback=args.coef_max,
        autoregressive_coeffs=args.stage2_autoregressive_coeffs,
    ).to(device)
    if args.stage2_source_ckpt is not None:
        stage2_source_ckpt = Path(args.stage2_source_ckpt).expanduser().resolve()
        if not stage2_source_ckpt.exists():
            raise FileNotFoundError(f"Requested stage-2 source checkpoint does not exist: {stage2_source_ckpt}")
        if is_main_process:
            print(f"[Stage2] warm-starting transformer from local checkpoint: {stage2_source_ckpt}")
        _load_module_checkpoint(transformer, stage2_source_ckpt)
    if is_main_process:
        print(
            _describe_stage2_model(
                transformer,
                stage2_arch=args.stage2_arch,
                tf_layers=args.tf_layers,
                tf_global_tokens=args.tf_global_tokens,
            )
        )
    _barrier()

    if args.stage2_epochs <= 0:
        if is_main_process:
            if wandb_run is not None:
                _finish_wandb(wandb_run)
            print(f"Outputs saved to: {args.out_dir}")
        _cleanup_distributed()
        return
    stage2_find_unused_parameters = bool(
        args.stage2_arch == "spatial_depth"
        and (not getattr(transformer, "autoregressive_coeffs", True))
    )
    if distributed and is_main_process and stage2_find_unused_parameters:
        print("[Stage2] enabling DDP unused-parameter detection for this training configuration")
    transformer_stage2 = (
        DDP(
            transformer,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=stage2_find_unused_parameters,
        )
        if distributed
        else transformer
    )

    train_stage2_transformer(
        transformer=transformer_stage2,
        token_loader=token_loader,
        stage2_fid_loader=stage2_fid_loader,
        device=device,
        epochs=args.stage2_epochs,
        lr=args.stage2_lr,
        rq_atom_loss_weight=args.stage2_rq_atom_loss_weight,
        rq_coeff_loss_weight=args.stage2_rq_coeff_loss_weight,
        coeff_loss_weight=args.stage2_coeff_loss_weight,
        coeff_loss_type=args.stage2_coeff_loss_type,
        coeff_huber_delta=args.stage2_coeff_huber_delta,
        coeff_depth_weighting=args.coeff_depth_weighting,
        coeff_focal_gamma=args.coeff_focal_gamma,
        warmup_steps=args.stage2_warmup_steps,
        min_lr_ratio=args.stage2_min_lr_ratio,
        weight_decay=args.stage2_weight_decay,
        sched_sampling_final_prob=args.stage2_sched_sampling_final_prob,
        stage2_amp=args.stage2_amp,
        stage2_amp_dtype=args.stage2_amp_dtype,
        pad_token_id=laser.bottleneck.pad_token_id,
        out_dir=stage2_dir,
        ae_for_decode=laser,
        H=H,
        W=W,
        D=D,
        sample_every_steps=args.stage2_sample_every_steps,
        sample_start_step=args.stage2_sample_start_step,
        sample_batch_size=args.stage2_sample_batch_size,
        sample_candidate_factor=args.stage2_sample_candidate_factor,
        sample_temperature=args.stage2_sample_temperature,
        sample_top_k=(None if args.stage2_sample_top_k <= 0 else args.stage2_sample_top_k),
        sample_coeff_temperature=(
            None if not math.isfinite(args.stage2_sample_coeff_temperature)
            else args.stage2_sample_coeff_temperature
        ),
        sample_coeff_mode=args.stage2_sample_coeff_mode,
        sample_selection_quality_weight=args.stage2_sample_quality_weight,
        sample_brightness_weight=args.stage2_sample_brightness_weight,
        sample_overbright_weight=args.stage2_sample_overbright_weight,
        sample_reject_dark_z=args.stage2_sample_reject_dark_z,
        sample_reject_bright_z=args.stage2_sample_reject_bright_z,
        sample_selection_mode=args.stage2_sample_selection_mode,
        sample_sort_by_quality=args.stage2_sample_sort_by_quality,
        sample_image_size=args.stage2_sample_image_size,
        sample_reference_stats=sample_reference_stats,
        stage2_fid_num_samples=args.stage2_fid_num_samples,
        stage2_fid_every_epochs=args.stage2_fid_every_epochs,
        token_sampler=token_sampler,
        is_main_process=is_main_process,
        wandb_run=wandb_run,
        artifact_tag=args.artifact_tag,
        ordinal_coeff_weight=args.ordinal_coeff_weight,
        ordinal_coeff_huber_delta=args.ordinal_coeff_huber_delta,
        ordinal_magnitude_weighted=args.ordinal_magnitude_weighted,
        ordinal_zero_drift_margin=args.ordinal_zero_drift_margin,
        ordinal_zero_drift_threshold=args.ordinal_zero_drift_threshold,
    )

    if is_main_process:
        if wandb_run is not None:
            _finish_wandb(wandb_run)
        print(f"Outputs saved to: {args.out_dir}")
    _barrier()
    _cleanup_distributed()


if __name__ == "__main__":
    main()
