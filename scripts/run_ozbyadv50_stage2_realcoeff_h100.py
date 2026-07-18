#!/usr/bin/env python3
"""Build/run FFHQ stage-2 with real-valued sparse coefficients on one H100."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Iterable

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

STAGE1_CKPT = (
    REPO
    / "outputs"
    / "ffhq_ozbyadv50_h100_continue"
    / "stage1"
    / "checkpoints"
    / "run_20260704_201652"
    / "laser"
    / "final.ckpt"
)
DATA_DIR = REPO.parent / "data" / "ffhq"
OUTPUT_DIR = REPO / "outputs" / "ffhq_ozbyadv50_stage2_realcoeff_h100" / "stage2"
TOKEN_CACHE = OUTPUT_DIR / "token_cache" / "ffhq__train__img256__laser_real.pt"


def q(value: object) -> str:
    return shlex.quote(str(value))


def validate_stage1_checkpoint(path: Path) -> None:
    from src.checkpoint_io import load_torch_payload

    payload = load_torch_payload(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise SystemExit(f"Not a readable Lightning checkpoint: {path}")
    print(f"Stage-1 checkpoint: {path} (epoch={payload.get('epoch')}, global_step={payload.get('global_step')})")


def command_env(output_dir: Path, *, wandb_id: str = "", wandb_resume: str = "never") -> dict[str, str]:
    env = os.environ.copy()
    env.pop("WANDB_RUN_ID", None)
    env.pop("WANDB_ID", None)
    env.update(
        {
            "WANDB_MODE": "online",
            "WANDB_RESUME": str(wandb_resume or "never"),
            "HYDRA_FULL_ERROR": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", "8"),
            "WANDB_DATA_DIR": env.get("WANDB_DATA_DIR", str(output_dir / "wandb" / "data")),
            "WANDB_CACHE_DIR": env.get("WANDB_CACHE_DIR", str(output_dir / "wandb" / "cache")),
            "WANDB_ARTIFACT_DIR": env.get("WANDB_ARTIFACT_DIR", str(output_dir / "wandb" / "artifacts")),
        }
    )
    if wandb_id:
        env["WANDB_RUN_ID"] = str(wandb_id)
    return env


def build_cache_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "tools" / "build_token_cache.py"),
        "--stage1_checkpoint",
        str(Path(args.stage1_ckpt).expanduser().resolve()),
        "--dataset",
        "ffhq",
        "--data_dir",
        str(Path(args.data_dir).expanduser().resolve()),
        "--split",
        "train",
        "--cache_mode",
        "real_valued",
        "--image_size",
        "256",
        "--batch_size",
        str(int(args.cache_batch_size)),
        "--num_workers",
        str(int(args.cache_num_workers)),
        "--mean",
        "0.5",
        "0.5",
        "0.5",
        "--std",
        "0.5",
        "0.5",
        "0.5",
        "--output",
        str(Path(args.token_cache).expanduser().resolve()),
        "--max_items",
        str(int(args.cache_max_items)),
        "--device",
        str(args.cache_device),
    ]
    return cmd


def stage2_cmd(args: argparse.Namespace, *, stamp: str) -> list[str]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    token_cache = Path(args.token_cache).expanduser().resolve()
    stage1_ckpt = Path(args.stage1_ckpt).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    ar_suffix = "ar" if args.autoregressive_coeffs else "atomonly"
    wandb_id = args.wandb_id or f"ozbyadv50-realcoeff-{ar_suffix}-{stamp}"
    wandb_name = args.wandb_name or f"ffhq-stage2-realcoeff-{ar_suffix}-ozbyadv50-{stamp}"
    wandb_group = args.wandb_group or f"ffhq-ozbyadv50-stage2-realcoeff-{ar_suffix}"

    cmd = [
        sys.executable,
        str(REPO / "train.py"),
        "stage2",
        f"token_cache_path={token_cache}",
        f"output_dir={output_dir}",
        "seed=42",
        "token_cache.build=true",
        "token_cache.force=false",
        f"token_cache.stage1_checkpoint={stage1_ckpt}",
        f"token_cache.output={token_cache}",
        "token_cache.split=train",
        "token_cache.cache_mode=real_valued",
        f"token_cache.batch_size={int(args.cache_batch_size)}",
        f"token_cache.num_workers={int(args.cache_num_workers)}",
        "token_cache.max_items=0",
        "token_cache.device=auto",
        "data.dataset=ffhq",
        f"data.data_dir={data_dir}",
        "data.image_size=256",
        f"data.num_workers={int(args.stage2_num_workers)}",
        "ar.type=sparse_spatial_depth",
        f"ar.autoregressive_coeffs={str(bool(args.autoregressive_coeffs)).lower()}",
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
        f"ar.coeff_loss_weight={args.coeff_loss_weight}",
        f"ar.atom_label_smoothing={args.atom_label_smoothing}",
        f"ar.atom_coverage_weight={args.atom_coverage_weight}",
        "ar.coeff_loss_type=huber",
        f"ar.coeff_huber_delta={args.coeff_huber_delta}",
        f"ar.coeff_head_hidden_mult={args.coeff_head_hidden_mult}",
        f"ar.coeff_head_depth={int(args.coeff_head_depth)}",
        f"ar.coeff_head_dropout={args.coeff_head_dropout}",
        f"ar.sample_coeff_mode={args.sample_coeff_mode}",
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
        f"train_ar.checkpoint_save_top_k={int(args.checkpoint_save_top_k)}",
        "train_ar.checkpoint_save_last=true",
        "train_ar.checkpoint_keep_recent=3",
        f"train_ar.checkpoint_every_n_epochs={int(args.checkpoint_every_n_epochs)}",
        f"train_ar.checkpoint_monitor={args.checkpoint_monitor}",
        f"train_ar.checkpoint_mode={args.checkpoint_mode}",
        f"train_ar.generation_fid_every_n_epochs={int(args.generation_fid_every_n_epochs)}",
        f"train_ar.generation_fid_num_samples={int(args.generation_fid_num_samples)}",
        "+train_ar.checkpoint_upload_to_wandb=true",
        f"+train_ar.checkpoint_upload_every_n_epochs={int(args.checkpoint_upload_every_n_epochs)}",
        f"train_ar.sample_every_n_epochs={int(args.sample_every_n_epochs)}",
        "train_ar.sample_every_n_steps=0",
        "train_ar.sample_log_to_wandb=true",
        f"train_ar.sample_num_images={int(args.sample_num_images)}",
        f"train_ar.sample_temperature={args.sample_temperature}",
        f"train_ar.sample_top_k={int(args.sample_top_k)}",
        f"train_ar.sample_coeff_mode={args.sample_coeff_mode}",
        f"train_ar.compute_generation_fid={str(bool(args.compute_generation_fid)).lower()}",
        "train_ar.compute_audio_generation_metrics=false",
        f"train_ar.generation_metric_num_samples={int(args.generation_metric_num_samples)}",
        "train_ar.run_test_after_fit=false",
        "train_ar.save_final_samples_after_fit=true",
        "wandb.project=laser",
        f"wandb.name={wandb_name}",
        f"wandb.id={wandb_id}",
        f"wandb.resume={args.wandb_resume}",
        f"wandb.group={wandb_group}",
        (
            "wandb.tags=[stage2,ffhq,laser,sparse_spatial_depth,real_coeffs,"
            f"{ar_suffix},anti_collapse,ozbyadv50,h100]"
        ),
        "wandb.append_timestamp=false",
        f"wandb.save_dir={output_dir / 'wandb'}",
    ]
    ckpt = str(args.ckpt or "").strip()
    if ckpt:
        cmd.append(f"ckpt_path={Path(ckpt).expanduser().resolve()}")
    return cmd


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
    parser.add_argument("--cache-max-items", type=int, default=0)
    parser.add_argument("--cache-device", default="auto")
    parser.add_argument("--coeff-loss-weight", default="1.0")
    parser.add_argument("--coeff-huber-delta", default="0.25")
    parser.add_argument("--coeff-head-hidden-mult", default="2.0")
    parser.add_argument("--coeff-head-depth", type=int, default=3)
    parser.add_argument("--coeff-head-dropout", default="0.05")
    parser.add_argument("--atom-coverage-weight", default="0.0")
    parser.add_argument("--atom-label-smoothing", default="0.0")
    parser.add_argument("--autoregressive-coeffs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--checkpoint-every-n-epochs", type=int, default=5)
    parser.add_argument("--checkpoint-upload-every-n-epochs", type=int, default=25)
    parser.add_argument("--checkpoint-save-top-k", type=int, default=1)
    parser.add_argument("--checkpoint-monitor", default="val/loss")
    parser.add_argument("--checkpoint-mode", default="min")
    parser.add_argument("--generation-fid-every-n-epochs", type=int, default=0)
    parser.add_argument("--generation-fid-num-samples", type=int, default=0)
    parser.add_argument("--generation-metric-num-samples", type=int, default=32)
    parser.add_argument("--compute-generation-fid", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sample-every-n-epochs", type=int, default=20)
    parser.add_argument("--sample-num-images", type=int, default=8)
    parser.add_argument("--sample-temperature", default="0.7")
    parser.add_argument("--sample-top-k", type=int, default=0)
    parser.add_argument("--sample-coeff-mode", choices=["mean", "gaussian"], default="mean")
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--wandb-id", default="")
    parser.add_argument("--wandb-name", default="")
    parser.add_argument("--wandb-group", default="")
    parser.add_argument("--wandb-resume", default="never")
    parser.add_argument("--background", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--build-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    token_cache = Path(args.token_cache).expanduser().resolve()
    stage1_ckpt = Path(args.stage1_ckpt).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if not data_dir.is_dir():
        raise SystemExit(f"FFHQ data directory not found: {data_dir}")
    if not stage1_ckpt.is_file():
        raise SystemExit(f"Stage-1 checkpoint not found: {stage1_ckpt}")
    validate_stage1_checkpoint(stage1_ckpt)
    output_dir.mkdir(parents=True, exist_ok=True)
    token_cache.parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "wandb").mkdir(parents=True, exist_ok=True)

    cache_cmd = build_cache_cmd(args)
    train_cmd = stage2_cmd(args, stamp=stamp)
    print("Real-valued token cache:", token_cache)
    if args.build_cache and not token_cache.is_file():
        print("Building real-valued token cache:")
        print(" ".join(q(part) for part in cache_cmd))
        if not args.dry_run:
            subprocess.run(cache_cmd, cwd=str(REPO), check=True)
    else:
        print(f"Using existing real-valued token cache: {token_cache} ({'exists' if token_cache.is_file() else 'missing'})")

    print("Launching real-valued stage 2:")
    print(" ".join(q(part) for part in train_cmd))
    if args.dry_run:
        return 0

    env = command_env(output_dir, wandb_id=args.wandb_id, wandb_resume=args.wandb_resume)
    if args.background:
        log_path = output_dir / f"stage2_realcoeff_h100_{stamp}.log"
        with log_path.open("ab") as log:
            proc = subprocess.Popen(
                train_cmd,
                cwd=str(REPO),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(f"Started background real-valued stage-2 pid={proc.pid}")
        print(f"Log: {log_path}")
        return 0
    return subprocess.run(train_cmd, cwd=str(REPO), env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
