"""
Train a direct sparse-code prior for LASER.

This replaces the failed "OMP step as VAR scale" formulation with a prior over
the final HxWxD sparse code itself. The model is the same SpatialDepthPrior used
by the working stage-2 path in proto.py, but exposed as a standalone entrypoint
that only depends on an existing stage-1 checkpoint.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

try:
    from spatial_prior import SpatialDepthPrior, build_spatial_depth_prior_config
except ModuleNotFoundError:
    from laser_transformer import SpatialDepthPrior, build_spatial_depth_prior_config
from proto import (
    LASER,
    FlatImageDataset,
    _compute_stage2_sample_reference_stats,
    _load_module_checkpoint,
    precompute_tokens,
    train_stage2_transformer,
)


def _parse_cli_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Could not parse boolean value: {value}")


def _launch_timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _broadcast_optional_string(value: Optional[str], src: int = 0) -> str:
    if not _is_distributed():
        if value is None:
            raise ValueError("value must be provided when distributed training is disabled")
        return str(value)
    obj_list = [value if dist.get_rank() == src else None]
    dist.broadcast_object_list(obj_list, src=src)
    if obj_list[0] is None:
        raise RuntimeError("failed to broadcast launch timestamp")
    return str(obj_list[0])


def _shared_launch_timestamp() -> str:
    value = None
    if not _is_distributed() or dist.get_rank() == 0:
        value = _launch_timestamp()
    return _broadcast_optional_string(value, src=0)


def _init_distributed(timeout_minutes: int) -> Tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0, 1
    if not torch.cuda.is_available():
        raise RuntimeError("DDP training requires CUDA.")

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            timeout=dt.timedelta(minutes=int(timeout_minutes)),
        )
    return True, rank, local_rank, world_size


def _cleanup_distributed():
    if _is_distributed():
        dist.destroy_process_group()


def _barrier():
    if _is_distributed():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def _safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _write_atomic_text(path: Path, text: str) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write(text)
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


def _init_wandb(args, run_dir: Path):
    if wandb is None or str(args.wandb_mode) == "disabled":
        return None
    return wandb.init(
        project=str(args.wandb_project),
        name=str(args.wandb_name or run_dir.name),
        mode=str(args.wandb_mode),
        config=vars(args),
        dir=str(run_dir),
    )


def _build_dataset(dataset: str, data_dir: str, image_size: int, seed: int):
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    eval_tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    if dataset == "cifar10":
        return datasets.CIFAR10(root=data_dir, train=True, download=True, transform=eval_tfm)
    if dataset == "celeba":
        full = FlatImageDataset(root=data_dir, transform=eval_tfm)
        if len(full) < 2:
            raise RuntimeError("CelebA dataset needs at least 2 images for a stable train split.")
        val_size = max(1, int(0.05 * len(full)))
        train_size = len(full) - val_size
        generator = torch.Generator().manual_seed(int(seed))
        train_set, _ = random_split(full, [train_size, val_size], generator=generator)
        return train_set
    raise ValueError(f"Unsupported dataset: {dataset}")


def _build_laser(args) -> LASER:
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
        quantize_sparse_coeffs=bool(args.quantize_sparse_coeffs),
        coef_quantization=args.coef_quantization,
        coef_mu=args.coef_mu,
        out_tanh=True,
        patch_based=args.patch_based,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        patch_reconstruction=args.patch_reconstruction,
    )


def _expected_token_cache_meta(args, stage2_source_set, token_subset: Optional[int], ae: LASER) -> dict:
    effective_items = len(stage2_source_set) if token_subset is None else int(token_subset)
    return {
        "version": 1,
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
        "patch_based": bool(args.patch_based),
        "patch_size": int(args.patch_size),
        "patch_stride": int(args.patch_stride),
        "patch_reconstruction": str(args.patch_reconstruction),
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
            if cache_meta.get(key) != expected_value:
                return False, f"meta mismatch for {key}: cache={cache_meta.get(key)!r}, expected={expected_value!r}"

    return True, "ok"


def _build_spatial_depth_prior(bottleneck, *, H: int, W: int, D: int, args, real_valued_coeffs: bool) -> SpatialDepthPrior:
    cfg = build_spatial_depth_prior_config(
        bottleneck,
        H=H,
        W=W,
        D=D,
        d_model=int(args.tf_d_model),
        n_heads=int(args.tf_heads),
        n_spatial_layers=int(args.tf_layers),
        n_depth_layers=max(1, int(args.tf_layers) // 2),
        d_ff=int(args.tf_ff),
        dropout=float(args.tf_dropout),
        n_global_spatial_tokens=int(args.tf_global_tokens),
        real_valued_coeffs=bool(real_valued_coeffs),
        coeff_max_fallback=float(args.coef_max),
    )
    return SpatialDepthPrior(cfg)


def main():
    parser = argparse.ArgumentParser(
        description="Train a direct sparse-code prior over the final LASER token grid."
    )
    parser.add_argument("--dataset", type=str, default="celeba", choices=["celeba", "cifar10"])
    parser.add_argument("--data_dir", type=str, default="/home/xl598/Projects/data/celeba")
    parser.add_argument("--out_dir", type=str, default="runs/laser_direct_sparse_prior")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--token_num_workers", type=int, default=2)
    parser.add_argument("--cpu_threads", type=int, default=4)
    parser.add_argument("--dist_timeout_minutes", type=int, default=60)
    parser.add_argument("--token_subset", type=int, default=0)
    parser.add_argument("--rebuild_token_cache", action="store_true")
    parser.add_argument("--stage1_source_ckpt", type=str, required=True)
    parser.add_argument(
        "--stage2_source_token_cache",
        type=str,
        default=None,
        help="Optional local tokens_cache.pt to reuse before rebuilding.",
    )
    parser.add_argument(
        "--stage2_source_ckpt",
        type=str,
        default=None,
        help="Optional local SpatialDepthPrior checkpoint to warm-start from.",
    )

    parser.add_argument("--num_hiddens", type=int, default=128)
    parser.add_argument("--ae_num_downsamples", type=int, default=4)
    parser.add_argument("--num_res_layers", type=int, default=2)
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--num_atoms", type=int, default=1024)
    parser.add_argument("--sparsity_level", type=int, default=8)
    parser.add_argument("--n_bins", type=int, default=256)
    parser.add_argument("--coef_max", type=float, default=3.0)
    parser.add_argument("--coef_quantization", type=str, default="uniform", choices=["uniform", "mu_law"])
    parser.add_argument("--coef_mu", type=float, default=0.0)
    parser.add_argument("--commitment_cost", type=float, default=0.25)
    parser.add_argument("--quantize_sparse_coeffs", type=_parse_cli_bool, nargs="?", const=True, default=True)
    parser.add_argument("--patch_based", type=_parse_cli_bool, nargs="?", const=True, default=False)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--patch_stride", type=int, default=2)
    parser.add_argument("--patch_reconstruction", type=str, default="center_crop", choices=["center_crop", "hann"])

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tf_d_model", type=int, default=512)
    parser.add_argument("--tf_heads", type=int, default=8)
    parser.add_argument("--tf_layers", type=int, default=12)
    parser.add_argument("--tf_ff", type=int, default=1024)
    parser.add_argument("--tf_dropout", type=float, default=0.1)
    parser.add_argument("--tf_global_tokens", type=int, default=0)
    parser.add_argument("--stage2_rq_atom_loss_weight", type=float, default=1.0)
    parser.add_argument("--stage2_rq_coeff_loss_weight", type=float, default=1.0)
    parser.add_argument("--stage2_coeff_loss_weight", type=float, default=1.0)
    parser.add_argument(
        "--stage2_coeff_loss_type",
        type=str,
        default="huber",
        choices=["huber", "mse", "recon_mse", "gt_atom_recon_mse"],
    )
    parser.add_argument("--stage2_coeff_huber_delta", type=float, default=1.0)
    parser.add_argument(
        "--stage2_sched_sampling_final_prob",
        type=float,
        default=0.0,
        help="Legacy compatibility knob from the removed flattened-token transformer path; ignored by the spatial-depth prior.",
    )
    parser.add_argument("--amp", type=_parse_cli_bool, nargs="?", const=True, default=True)
    parser.add_argument("--amp_dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16"])

    parser.add_argument("--sample_every_steps", type=int, default=2000)
    parser.add_argument("--sample_batch_size", type=int, default=8)
    parser.add_argument("--sample_candidate_factor", type=int, default=4)
    parser.add_argument("--sample_temperature", type=float, default=1.0)
    parser.add_argument("--sample_top_k", type=int, default=256)
    parser.add_argument("--sample_image_size", type=int, default=128)

    parser.add_argument("--wandb_mode", type=str, default="disabled", choices=["disabled", "offline", "online"])
    parser.add_argument("--wandb_project", type=str, default="laser")
    parser.add_argument("--wandb_name", type=str, default=None)
    args = parser.parse_args()

    if args.lr <= 0.0:
        raise ValueError("lr must be > 0")
    if int(args.batch_size) <= 0:
        raise ValueError("batch_size must be > 0")
    if int(args.cpu_threads) <= 0:
        raise ValueError("cpu_threads must be > 0")
    if int(args.dist_timeout_minutes) <= 0:
        raise ValueError("dist_timeout_minutes must be > 0")
    if int(args.tf_heads) <= 0:
        raise ValueError("tf_heads must be > 0")
    if int(args.tf_layers) <= 0:
        raise ValueError("tf_layers must be > 0")
    if int(args.sample_candidate_factor) <= 0:
        raise ValueError("sample_candidate_factor must be > 0")

    distributed, rank, local_rank, world_size = _init_distributed(int(args.dist_timeout_minutes))
    is_main_process = (rank == 0)
    os.environ["OMP_NUM_THREADS"] = str(int(args.cpu_threads))
    os.environ["MKL_NUM_THREADS"] = str(int(args.cpu_threads))
    torch.set_num_threads(int(args.cpu_threads))
    try:
        torch.set_num_interop_threads(max(1, min(4, int(args.cpu_threads))))
    except RuntimeError:
        pass
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
        torch.backends.cudnn.benchmark = True

    experiment_root = Path(args.out_dir).expanduser().resolve()
    run_dir = experiment_root / _shared_launch_timestamp()
    if is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
    _barrier()
    device = torch.device("cuda", local_rank) if distributed else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if is_main_process:
        print(
            f"[DirectSparsePrior] device={device} world_size={world_size} run_dir={run_dir} "
            f"cpu_threads={torch.get_num_threads()}"
        )
    wandb_run = _init_wandb(args, run_dir) if is_main_process else None

    train_set = _build_dataset(args.dataset, args.data_dir, args.image_size, args.seed)

    ae = _build_laser(args).to(device)
    stage1_ckpt = Path(args.stage1_source_ckpt).expanduser().resolve()
    if not stage1_ckpt.exists():
        _cleanup_distributed()
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {stage1_ckpt}")
    _load_module_checkpoint(ae, stage1_ckpt)
    ae.eval()
    ae.requires_grad_(False)
    if is_main_process:
        coeff_mode = "quantized coeffs" if bool(ae.bottleneck.quantize_sparse_coeffs) else "real-valued coeffs"
        print(
            f"[DirectSparsePrior] loaded stage-1 checkpoint: {stage1_ckpt} "
            f"mode={coeff_mode}"
        )

    experiment_root.mkdir(parents=True, exist_ok=True)
    token_cache_path = experiment_root / "tokens_cache.pt"
    token_cache_ready_path = experiment_root / "tokens_cache.ready"
    token_cache_error_path = experiment_root / "tokens_cache.failed"
    token_subset = None if int(args.token_subset) <= 0 else min(int(args.token_subset), len(train_set))
    expected_token_meta = _expected_token_cache_meta(args, train_set, token_subset, ae)
    _barrier()
    if is_main_process:
        _unlink_if_exists(token_cache_ready_path)
        _unlink_if_exists(token_cache_error_path)
        try:
            cache_ready = False
            cache_reason = None
            requested_token_cache_path = None
            if not args.rebuild_token_cache and args.stage2_source_token_cache:
                requested_token_cache_path = Path(args.stage2_source_token_cache).expanduser().resolve()
                if not requested_token_cache_path.exists():
                    raise FileNotFoundError(
                        f"Requested token cache does not exist: {requested_token_cache_path}"
                    )

            if (not cache_ready) and (not args.rebuild_token_cache) and requested_token_cache_path is not None:
                try:
                    token_cache = _safe_torch_load(requested_token_cache_path)
                    compatible, reason = _token_cache_is_compatible(token_cache, expected_token_meta)
                    if compatible:
                        if requested_token_cache_path != token_cache_path:
                            shutil.copy2(requested_token_cache_path, token_cache_path)
                        tokens_flat = token_cache["tokens_flat"]
                        H, W, D = token_cache["shape"]
                        print(
                            f"[DirectSparsePrior] using external token cache: {tokens_flat.shape} "
                            f"(H={H}, W={W}, D={D}) from {requested_token_cache_path}"
                        )
                        cache_ready = True
                    else:
                        cache_reason = f"requested token cache incompatible ({reason})"
                except Exception as exc:
                    cache_reason = (
                        f"could not load requested token cache "
                        f"({type(exc).__name__}: {exc})"
                    )

            if (not cache_ready) and (not args.rebuild_token_cache) and token_cache_path.exists():
                try:
                    token_cache = _safe_torch_load(token_cache_path)
                    compatible, reason = _token_cache_is_compatible(token_cache, expected_token_meta)
                    if compatible:
                        tokens_flat = token_cache["tokens_flat"]
                        H, W, D = token_cache["shape"]
                        print(
                            f"[DirectSparsePrior] reusing token cache: {tokens_flat.shape} "
                            f"(H={H}, W={W}, D={D}) from {token_cache_path}"
                        )
                        cache_ready = True
                    else:
                        cache_reason = reason
                except Exception as exc:
                    cache_reason = f"could not load current token cache ({type(exc).__name__}: {exc})"

            if (not cache_ready) and cache_reason is not None and (not args.rebuild_token_cache):
                print(f"[DirectSparsePrior] rebuilding token cache ({cache_reason})")
            elif args.rebuild_token_cache:
                print("[DirectSparsePrior] rebuilding token cache (--rebuild_token_cache)")

            if not cache_ready:
                token_source_loader = DataLoader(
                    train_set,
                    batch_size=int(args.batch_size),
                    shuffle=False,
                    num_workers=int(args.token_num_workers),
                    pin_memory=(device.type == "cuda"),
                    persistent_workers=(int(args.token_num_workers) > 0),
                )
                tokens_flat, coeffs_flat, H, W, D = precompute_tokens(
                    ae,
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
                print(
                    f"[DirectSparsePrior] token dataset: {tokens_flat.shape} "
                    f"(H={H}, W={W}, D={D})"
                )
            _write_atomic_text(
                token_cache_ready_path,
                f"ready {dt.datetime.now().isoformat()} {token_cache_path}\n",
            )
        except Exception as exc:
            _write_atomic_text(
                token_cache_error_path,
                f"[DirectSparsePrior] token cache preparation failed: {type(exc).__name__}: {exc}\n",
            )
            raise
    else:
        _wait_for_file_signal(
            token_cache_ready_path,
            token_cache_error_path,
            timeout_seconds=max(60.0, float(args.dist_timeout_minutes) * 60.0),
            description=f"direct sparse-code token cache at {token_cache_path}",
        )
    token_cache = _safe_torch_load(token_cache_path)
    tokens_flat = token_cache["tokens_flat"]
    coeffs_flat = token_cache.get("coeffs_flat", None)
    H, W, D = token_cache["shape"]
    expected_items = int(expected_token_meta["effective_items"])
    if tokens_flat.size(0) > expected_items:
        tokens_flat = tokens_flat[:expected_items]
        if coeffs_flat is not None:
            coeffs_flat = coeffs_flat[:expected_items]
    real_valued_coeffs = (coeffs_flat is not None)
    if is_main_process:
        print(
            f"[DirectSparsePrior] final token grid: items={tokens_flat.size(0)} "
            f"H={int(H)} W={int(W)} D={int(D)} real_valued_coeffs={bool(real_valued_coeffs)}"
        )

    if real_valued_coeffs:
        token_dataset = TensorDataset(tokens_flat, coeffs_flat)
    else:
        token_dataset = tokens_flat
    token_sampler = DistributedSampler(token_dataset, shuffle=True) if distributed else None
    token_loader = DataLoader(
        token_dataset,
        batch_size=int(args.batch_size),
        shuffle=(token_sampler is None),
        sampler=token_sampler,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=(len(token_dataset) >= int(args.batch_size)),
    )

    transformer = _build_spatial_depth_prior(
        ae.bottleneck,
        H=int(H),
        W=int(W),
        D=int(D),
        args=args,
        real_valued_coeffs=bool(real_valued_coeffs),
    ).to(device)
    if args.stage2_source_ckpt is not None:
        stage2_source_ckpt = Path(args.stage2_source_ckpt).expanduser().resolve()
        if not stage2_source_ckpt.exists():
            _cleanup_distributed()
            raise FileNotFoundError(f"Requested stage-2 source checkpoint does not exist: {stage2_source_ckpt}")
        if is_main_process:
            print(f"[DirectSparsePrior] warm-starting prior from {stage2_source_ckpt}")
        _load_module_checkpoint(transformer, stage2_source_ckpt)
    if is_main_process:
        coeff_mode = "real-valued coeffs" if real_valued_coeffs else "quantized sparse coeffs"
        print(
            "[DirectSparsePrior] using SpatialDepthPrior "
            f"({coeff_mode}, spatial_layers={int(args.tf_layers)}, "
            f"depth_layers={max(1, int(args.tf_layers) // 2)}, "
            f"global_tokens={int(args.tf_global_tokens)})"
        )

    if int(args.epochs) <= 0:
        if is_main_process:
            if wandb_run is not None:
                wandb_run.finish()
            print(f"[DirectSparsePrior] no training requested, outputs would be written under {run_dir}")
        _cleanup_distributed()
        return

    sample_reference_stats = None
    if int(args.sample_every_steps) > 0:
        sample_reference_stats = _compute_stage2_sample_reference_stats(
            ae,
            tokens_flat,
            coeffs_flat,
            int(H),
            int(W),
            int(D),
            device,
        )

    if distributed:
        transformer = DDP(
            transformer,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    train_stage2_transformer(
        transformer=transformer,
        token_loader=token_loader,
        device=device,
        epochs=int(args.epochs),
        lr=float(args.lr),
        rq_atom_loss_weight=float(args.stage2_rq_atom_loss_weight),
        rq_coeff_loss_weight=float(args.stage2_rq_coeff_loss_weight),
        coeff_loss_weight=float(args.stage2_coeff_loss_weight),
        coeff_loss_type=str(args.stage2_coeff_loss_type),
        coeff_huber_delta=float(args.stage2_coeff_huber_delta),
        sched_sampling_final_prob=float(args.stage2_sched_sampling_final_prob),
        stage2_amp=bool(args.amp),
        stage2_amp_dtype=str(args.amp_dtype),
        pad_token_id=int(getattr(ae.bottleneck, "pad_token_id", 0)),
        out_dir=str(run_dir),
        ae_for_decode=ae,
        H=int(H),
        W=int(W),
        D=int(D),
        sample_every_steps=int(args.sample_every_steps),
        sample_batch_size=int(args.sample_batch_size),
        sample_candidate_factor=int(args.sample_candidate_factor),
        sample_temperature=float(args.sample_temperature),
        sample_top_k=(None if int(args.sample_top_k) <= 0 else int(args.sample_top_k)),
        sample_image_size=int(args.sample_image_size),
        sample_reference_stats=sample_reference_stats,
        token_sampler=token_sampler,
        is_main_process=is_main_process,
        wandb_run=wandb_run,
    )

    if is_main_process:
        if wandb_run is not None:
            wandb_run.finish()
        print(f"[DirectSparsePrior] outputs saved to: {run_dir}")
    _cleanup_distributed()


if __name__ == "__main__":
    main()
