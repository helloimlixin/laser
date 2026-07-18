#!/usr/bin/env python3
"""Launch a full-sequence GPT stage-2 prior for the ozbyadv50 FFHQ cache on one H100."""

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
SPATIAL_OUTPUT_DIR = REPO / "outputs" / "ffhq_ozbyadv50_stage2_1000_h100" / "stage2"
TOKEN_CACHE = SPATIAL_OUTPUT_DIR / "token_cache" / "ffhq__train__img256__laser_cb256_quantile_p99p5.pt"
OUTPUT_DIR = REPO / "outputs" / "ffhq_ozbyadv50_stage2_gpt_h100" / "stage2"


def q(value: object) -> str:
    return shlex.quote(str(value))


def validate_stage1_checkpoint(path: Path) -> None:
    from src.checkpoint_io import load_torch_payload

    payload = load_torch_payload(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise SystemExit(f"Not a readable Lightning checkpoint: {path}")
    print(f"Stage-1 checkpoint: {path} (epoch={payload.get('epoch')}, global_step={payload.get('global_step')})")


def command_env(output_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("WANDB_RUN_ID", None)
    env.pop("WANDB_ID", None)
    env.update(
        {
            "WANDB_MODE": "online",
            "WANDB_RESUME": "never",
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


def stage2_cmd(args: argparse.Namespace, *, stamp: str) -> list[str]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    token_cache = Path(args.token_cache).expanduser().resolve()
    stage1_ckpt = Path(args.stage1_ckpt).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    wandb_id = args.wandb_id or f"ozbyadv50-gpt-{stamp}"
    wandb_name = args.wandb_name or f"ffhq-stage2-gpt-ozbyadv50-h100-{stamp}"

    return [
        sys.executable,
        str(REPO / "train.py"),
        "stage2",
        f"token_cache_path={token_cache}",
        f"output_dir={output_dir}",
        "seed=42",
        "token_cache.build=false",
        "token_cache.force=false",
        f"token_cache.stage1_checkpoint={stage1_ckpt}",
        f"token_cache.output={token_cache}",
        "token_cache.split=train",
        "token_cache.cache_mode=quantized",
        "data.dataset=ffhq",
        f"data.data_dir={data_dir}",
        "data.image_size=256",
        f"data.num_workers={int(args.stage2_num_workers)}",
        "ar.type=gpt",
        "ar.autoregressive_coeffs=true",
        "ar.class_conditional=false",
        "ar.vocab_size=null",
        "ar.atom_vocab_size=null",
        "ar.coeff_vocab_size=null",
        f"ar.window_sites={int(args.window_sites)}",
        f"ar.n_global_spatial_tokens={int(args.global_tokens)}",
        f"ar.d_model={int(args.d_model)}",
        f"ar.n_heads={int(args.n_heads)}",
        f"ar.n_layers={int(args.n_layers)}",
        f"ar.d_ff={int(args.d_ff)}",
        f"ar.dropout={args.dropout}",
        f"ar.learning_rate={args.stage2_lr}",
        "ar.weight_decay=0.01",
        f"ar.warmup_steps={int(args.warmup_steps)}",
        "ar.max_steps=-1",
        "ar.min_lr_ratio=0.08",
        "ar.atom_loss_weight=1.0",
        "ar.coeff_loss_weight=1.0",
        "ar.coeff_depth_weighting=none",
        "ar.coeff_focal_gamma=0.0",
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
        f"train_ar.accumulate_grad_batches={int(args.accumulate_grad_batches)}",
        "train_ar.gradient_clip_val=1.0",
        "train_ar.checkpoint_save_top_k=1",
        "train_ar.checkpoint_save_last=true",
        "train_ar.checkpoint_keep_recent=3",
        f"train_ar.checkpoint_every_n_epochs={int(args.checkpoint_every_n_epochs)}",
        "+train_ar.checkpoint_upload_to_wandb=true",
        f"+train_ar.checkpoint_upload_every_n_epochs={int(args.checkpoint_upload_every_n_epochs)}",
        "train_ar.sample_every_n_epochs=10",
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
        f"wandb.name={wandb_name}",
        f"wandb.id={wandb_id}",
        "wandb.resume=never",
        "wandb.group=ffhq-ozbyadv50-stage2-gpt",
        "wandb.tags=[stage2,ffhq,laser,gpt,full_sequence,unconditional,ozbyadv50,h100]",
        "wandb.append_timestamp=false",
        f"wandb.save_dir={output_dir / 'wandb'}",
    ]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-ckpt", default=str(STAGE1_CKPT))
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--token-cache", default=str(TOKEN_CACHE))
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--stage2-batch-size", type=int, default=16)
    parser.add_argument("--accumulate-grad-batches", type=int, default=2)
    parser.add_argument("--stage2-lr", default="8.0e-5")
    parser.add_argument("--warmup-steps", type=int, default=5000)
    parser.add_argument("--stage2-num-workers", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--n-layers", type=int, default=24)
    parser.add_argument("--d-ff", type=int, default=4096)
    parser.add_argument("--dropout", default="0.1")
    parser.add_argument("--window-sites", type=int, default=0, help="0 means full-sequence causal attention over all 512 tokens.")
    parser.add_argument("--global-tokens", type=int, default=16)
    parser.add_argument("--checkpoint-every-n-epochs", type=int, default=5)
    parser.add_argument("--checkpoint-upload-every-n-epochs", type=int, default=25)
    parser.add_argument("--wandb-id", default="")
    parser.add_argument("--wandb-name", default="")
    parser.add_argument("--background", action=argparse.BooleanOptionalAction, default=True)
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
    if not token_cache.is_file():
        raise SystemExit(f"Token cache not found: {token_cache}")
    validate_stage1_checkpoint(stage1_ckpt)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "wandb").mkdir(parents=True, exist_ok=True)

    cmd = stage2_cmd(args, stamp=stamp)
    print(f"Token cache: {token_cache}")
    print("Launching GPT stage 2:")
    print(" ".join(q(part) for part in cmd))
    if args.dry_run:
        return 0

    env = command_env(output_dir)
    if args.background:
        log_path = output_dir / f"stage2_gpt_h100_{stamp}.log"
        with log_path.open("ab") as log:
            proc = subprocess.Popen(
                cmd,
                cwd=str(REPO),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(f"Started background GPT stage-2 pid={proc.pid}")
        print(f"Log: {log_path}")
        return 0
    return subprocess.run(cmd, cwd=str(REPO), env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
