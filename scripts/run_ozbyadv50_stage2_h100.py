#!/usr/bin/env python3
"""Build token cache and launch ozbyadv50 stage-2 training on one H100."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

STAGE1_CKPT = REPO / "outputs" / "ffhq_ozbyadv50_h100_continue" / "stage1" / "checkpoints" / "run_20260704_201652" / "laser" / "final.ckpt"
DATA_DIR = REPO.parent / "data" / "ffhq"
OUTPUT_DIR = REPO / "outputs" / "ffhq_ozbyadv50_stage2_1000_h100" / "stage2"
TOKEN_CACHE = OUTPUT_DIR / "token_cache" / "ffhq__train__img256__laser_cb256_quantile_p99p5.pt"
WANDB_ID = "ozbyadv50s2e1000"
WANDB_NAME = "ffhq-stage2-1000-k4-a4096-from-ozbyadv50-bilinear-h100"
WANDB_GROUP = "ffhq-ozbyadv50-stage2"


def q(value: object) -> str:
    return shlex.quote(str(value))


def latest_stage2_checkpoint(output_dir: Path) -> Path | None:
    ckpt_root = output_dir / "checkpoints"
    if not ckpt_root.is_dir():
        return None
    candidates = [
        path for path in ckpt_root.rglob("*.ckpt")
        if path.name == "last.ckpt" or path.name.endswith("-last.ckpt")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def validate_stage1_checkpoint(path: Path) -> None:
    from src.checkpoint_io import load_torch_payload

    payload = load_torch_payload(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise SystemExit(f"Not a readable Lightning checkpoint: {path}")
    print(f"Stage-1 checkpoint: {path} (epoch={payload.get('epoch')}, global_step={payload.get('global_step')})")


def build_stage2_args(args: argparse.Namespace) -> list[str]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    token_cache = Path(args.token_cache).expanduser().resolve()
    stage1_ckpt = Path(args.stage1_ckpt).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()

    cmd_args = [
        "stage2",
        f"token_cache_path={token_cache}",
        f"output_dir={output_dir}",
        "seed=42",
        "token_cache.build=true",
        "token_cache.force=false",
        f"token_cache.stage1_checkpoint={stage1_ckpt}",
        f"token_cache.output={token_cache}",
        "token_cache.split=train",
        "token_cache.cache_mode=quantized",
        "token_cache.coeff_vocab_size=256",
        "token_cache.coeff_max=auto",
        "token_cache.coeff_quantization=quantile",
        "token_cache.coeff_calibration_percentile=99.5",
        "token_cache.coeff_mu=0.0",
        f"token_cache.batch_size={int(args.cache_batch_size)}",
        f"token_cache.num_workers={int(args.cache_num_workers)}",
        "token_cache.max_items=0",
        "token_cache.device=auto",
        "data.dataset=ffhq",
        f"data.data_dir={data_dir}",
        "data.image_size=256",
        f"data.num_workers={int(args.stage2_num_workers)}",
        "ar.type=sparse_spatial_depth",
        "ar.autoregressive_coeffs=true",
        "ar.class_conditional=false",
        "ar.vocab_size=null",
        "ar.atom_vocab_size=null",
        "ar.coeff_vocab_size=null",
        "ar.window_sites=0",
        "ar.n_global_spatial_tokens=16",
        "ar.d_model=768",
        "ar.n_heads=12",
        "ar.n_layers=18",
        "ar.d_ff=3072",
        "ar.dropout=0.1",
        f"ar.learning_rate={args.stage2_lr}",
        "ar.weight_decay=0.01",
        "ar.warmup_steps=5000",
        "ar.max_steps=-1",
        "ar.min_lr_ratio=0.08",
        "ar.atom_loss_weight=1.0",
        "ar.coeff_loss_weight=1.0",
        "ar.coeff_depth_weighting=none",
        "ar.coeff_focal_gamma=0.0",
        "ar.atom_label_smoothing=0.0",
        "ar.atom_coverage_weight=0.0",
        "ar.coeff_loss_type=auto",
        "ar.coeff_huber_delta=0.25",
        f"train_ar.max_epochs={int(args.epochs)}",
        f"train_ar.batch_size={int(args.stage2_batch_size)}",
        "train_ar.max_items=0",
        "train_ar.limit_train_batches=1.0",
        "train_ar.limit_val_batches=1.0",
        "train_ar.limit_test_batches=0",
        "train_ar.val_check_interval=1.0",
        "train_ar.validation_split=0.05",
        "train_ar.test_split=0.05",
        "train_ar.log_every_n_steps=20",
        "train_ar.devices=1",
        "train_ar.num_nodes=1",
        "train_ar.strategy=auto",
        "train_ar.precision=bf16-mixed",
        "train_ar.accelerator=gpu",
        "train_ar.deterministic=false",
        "train_ar.accumulate_grad_batches=1",
        "train_ar.gradient_clip_val=1.0",
        "train_ar.checkpoint_save_top_k=1",
        "train_ar.checkpoint_save_last=true",
        "train_ar.checkpoint_keep_recent=3",
        f"train_ar.checkpoint_every_n_epochs={int(args.checkpoint_every_n_epochs)}",
        "+train_ar.checkpoint_upload_to_wandb=true",
        f"+train_ar.checkpoint_upload_every_n_epochs={int(args.checkpoint_upload_every_n_epochs)}",
        "train_ar.sample_every_n_epochs=20",
        "train_ar.sample_every_n_steps=0",
        "train_ar.sample_log_to_wandb=true",
        "train_ar.sample_num_images=8",
        "train_ar.sample_temperature=0.7",
        "train_ar.sample_top_k=0",
        "train_ar.compute_generation_fid=false",
        "train_ar.compute_audio_generation_metrics=false",
        "train_ar.generation_metric_num_samples=32",
        "train_ar.run_test_after_fit=false",
        "train_ar.save_final_samples_after_fit=true",
        "wandb.project=laser",
        f"wandb.name={args.wandb_name}",
        f"wandb.id={args.wandb_id}",
        "wandb.resume=allow",
        f"wandb.group={args.wandb_group}",
        "wandb.tags=[stage2,ffhq,laser,sparse_spatial_depth,unconditional,ozbyadv50,k4-a4096,bilinear,h100,stage2_1000]",
        "wandb.append_timestamp=false",
        f"wandb.save_dir={output_dir / 'wandb'}",
    ]
    resume_ckpt = latest_stage2_checkpoint(output_dir)
    if resume_ckpt is not None:
        cmd_args.append(f"ckpt_path={resume_ckpt}")
        print(f"Resuming existing stage-2 checkpoint: {resume_ckpt}")
    return cmd_args


def command_env(output_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("WANDB_RUN_ID", None)
    env.pop("WANDB_ID", None)
    env.update(
        {
            "WANDB_MODE": "online",
            "WANDB_RESUME": "allow",
            "HYDRA_FULL_ERROR": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", "8"),
            "WANDB_DATA_DIR": env.get("WANDB_DATA_DIR", str(output_dir / "wandb" / "data")),
            "WANDB_CACHE_DIR": env.get("WANDB_CACHE_DIR", str(output_dir / "wandb" / "cache")),
            "WANDB_ARTIFACT_DIR": env.get("WANDB_ARTIFACT_DIR", str(output_dir / "wandb" / "artifacts")),
        }
    )
    return env


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-ckpt", default=str(STAGE1_CKPT))
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--token-cache", default=str(TOKEN_CACHE))
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--stage2-batch-size", type=int, default=32)
    parser.add_argument("--stage2-lr", default="1.25e-4")
    parser.add_argument("--stage2-num-workers", type=int, default=8)
    parser.add_argument("--cache-batch-size", type=int, default=64)
    parser.add_argument("--cache-num-workers", type=int, default=24)
    parser.add_argument("--checkpoint-every-n-epochs", type=int, default=5)
    parser.add_argument("--checkpoint-upload-every-n-epochs", type=int, default=25)
    parser.add_argument("--wandb-id", default=WANDB_ID)
    parser.add_argument("--wandb-name", default=WANDB_NAME)
    parser.add_argument("--wandb-group", default=WANDB_GROUP)
    parser.add_argument("--background", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    token_cache = Path(args.token_cache).expanduser().resolve()
    stage1_ckpt = Path(args.stage1_ckpt).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()

    if not data_dir.is_dir():
        raise SystemExit(f"FFHQ data directory not found: {data_dir}")
    if not stage1_ckpt.is_file():
        raise SystemExit(f"Stage-1 checkpoint not found: {stage1_ckpt}")
    validate_stage1_checkpoint(stage1_ckpt)
    output_dir.mkdir(parents=True, exist_ok=True)
    token_cache.parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "wandb").mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(REPO / "train.py"), *build_stage2_args(args)]
    env = command_env(output_dir)
    print("WANDB_MODE=online")
    print(f"Token cache: {token_cache} ({'exists' if token_cache.is_file() else 'will build'})")
    print("Launching stage 2:")
    print(" ".join(q(part) for part in cmd))
    if args.dry_run:
        return 0
    if args.background:
        log_path = output_dir / "stage2_h100.log"
        with log_path.open("ab") as log:
            proc = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"Started background stage-2 pid={proc.pid}")
        print(f"Log: {log_path}")
        return 0
    return subprocess.run(cmd, cwd=str(REPO), env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
