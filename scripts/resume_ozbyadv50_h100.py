#!/usr/bin/env python3
"""Resume W&B run ozbyadv50 on a single H100 80GB node.

The crashed W&B run did not upload checkpoints. This helper first tries to
recover a checkpoint from W&B artifacts/files, then falls back to a user-supplied
local checkpoint path via --ckpt. It keeps the original model architecture and
uses H100-oriented single-GPU training settings.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable

RUN_PATH = "helloimlixin-rutgers/laser/ozbyadv50"
RUN_ID = "ozbyadv50"
RUN_NAME = "ffhq-stage1-adv50-k4-a4096-from-ozby19jo-bilinear"
RUN_GROUP = "ffhq-ozby19jo-adv50"

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
DEFAULT_DATA_DIR = REPO.parent / "data" / "ffhq"
DEFAULT_OUTPUT_DIR = REPO / "outputs" / "ffhq_ozbyadv50_h100_continue" / "stage1"
DEFAULT_DOWNLOAD_DIR = DEFAULT_OUTPUT_DIR / "downloaded_checkpoints"
CKPT_SUFFIXES = (".ckpt", ".pt", ".pth")


def q(value: object) -> str:
    return shlex.quote(str(value))


def is_checkpoint_name(name: str) -> bool:
    lower = str(name).lower()
    return lower.endswith(CKPT_SUFFIXES) or "checkpoint" in lower


def checkpoint_sort_key(path: Path) -> tuple[int, float, str]:
    name = path.name.lower()
    priority = 0
    if name == "last.ckpt" or name.endswith("-last.ckpt"):
        priority = 3
    elif name == "final.ckpt" or name.endswith("-final.ckpt"):
        priority = 2
    elif name.endswith(".ckpt"):
        priority = 1
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return priority, mtime, str(path)


def best_local_checkpoint(root: Path) -> Path | None:
    if not root.exists():
        return None
    candidates = [path for path in root.rglob("*") if path.is_file() and is_checkpoint_name(path.name)]
    if not candidates:
        return None
    return sorted(candidates, key=checkpoint_sort_key, reverse=True)[0]


def download_run_file(run, name: str, root: Path) -> Path | None:
    try:
        downloaded = run.file(name).download(root=str(root), replace=True)
    except Exception:
        return None
    return Path(downloaded.name)


def infer_original_checkpoint_hint(metadata_path: Path | None) -> list[str]:
    if metadata_path is None or not metadata_path.is_file():
        return []
    try:
        metadata = json.loads(metadata_path.read_text())
    except Exception:
        return []
    args = list(metadata.get("args") or [])
    output_dir = ""
    init_ckpt = ""
    for item in args:
        if str(item).startswith("output_dir="):
            output_dir = str(item).split("=", 1)[1]
        elif str(item).startswith("init_ckpt_path="):
            init_ckpt = str(item).split("=", 1)[1]
    hints = []
    if output_dir:
        hints.append(f"original output_dir: {output_dir}")
        hints.append(f"likely local checkpoint root: {output_dir}/checkpoints")
    if init_ckpt:
        hints.append(f"original init_ckpt_path: {init_ckpt}")
    return hints


def try_download_checkpoint(run_path: str, download_dir: Path) -> Path | None:
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise SystemExit("wandb is not installed. Run `python -m pip install --user wandb` first.") from exc

    download_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=90)
    run = api.run(run_path)

    metadata_root = download_dir / "run_files"
    metadata_root.mkdir(parents=True, exist_ok=True)
    metadata_path = download_run_file(run, "wandb-metadata.json", metadata_root)

    artifact_hits = []
    for artifact in run.logged_artifacts():
        try:
            files = list(artifact.files())
        except Exception:
            continue
        if not any(is_checkpoint_name(file.name) for file in files):
            continue
        created = str(getattr(artifact, "created_at", "") or "")
        artifact_hits.append((created, artifact))

    for _, artifact in sorted(artifact_hits, key=lambda item: item[0], reverse=True):
        target = download_dir / "artifacts" / artifact.name.replace(":", "_").replace("/", "_")
        target.mkdir(parents=True, exist_ok=True)
        artifact.download(root=str(target))
        ckpt = best_local_checkpoint(target)
        if ckpt is not None:
            print(f"Downloaded checkpoint artifact {artifact.name}: {ckpt}")
            return ckpt

    checkpoint_files = []
    for file in run.files(per_page=10000):
        if is_checkpoint_name(file.name):
            checkpoint_files.append(file)
    if checkpoint_files:
        target = download_dir / "run_checkpoint_files"
        target.mkdir(parents=True, exist_ok=True)
        for file in checkpoint_files:
            file.download(root=str(target), replace=True)
        ckpt = best_local_checkpoint(target)
        if ckpt is not None:
            print(f"Downloaded checkpoint file from run files: {ckpt}")
            return ckpt

    print(f"No checkpoint artifact or checkpoint file is attached to {run_path}.", file=sys.stderr)
    print(f"Run state: {run.state}; summary epoch={dict(run.summary).get('epoch')}; "
          f"trainer/global_step={dict(run.summary).get('trainer/global_step')}", file=sys.stderr)
    for hint in infer_original_checkpoint_hint(metadata_path):
        print(hint, file=sys.stderr)
    return None




def validate_torch_checkpoint(path: Path) -> None:
    """Fail before launching if the checkpoint is not a readable torch payload."""
    try:
        from src.checkpoint_io import load_torch_payload

        payload = load_torch_payload(path, map_location="cpu")
    except Exception as exc:
        details = str(exc).splitlines()[0]
        try:
            from zipfile import ZipFile

            with ZipFile(path) as archive:
                negative = [info.header_offset for info in archive.infolist() if info.header_offset < 0]
            if negative:
                missing = -min(negative)
                details += f"; zip archive appears to be missing about {missing} leading bytes"
        except Exception:
            pass
        raise SystemExit(f"Checkpoint is not readable by torch.load: {path}\n{details}") from exc
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise SystemExit(f"Checkpoint does not look like a Lightning checkpoint with state_dict: {path}")
    print(f"Validated checkpoint: {path} (epoch={payload.get('epoch')}, global_step={payload.get('global_step')})")


def build_stage1_args(args: argparse.Namespace, ckpt_path: Path) -> list[str]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    return [
        "stage1",
        "seed=42",
        f"output_dir={output_dir}",
        f"ckpt_path={ckpt_path.expanduser().resolve()}",
        "model=laser_image_nonpatch_d5",
        "data=ffhq",
        "data.dataset=ffhq",
        f"data.data_dir={data_dir}",
        "data.image_size=256",
        "data.train_crop_size=null",
        f"data.batch_size={int(args.batch_size)}",
        f"data.eval_batch_size={int(args.eval_batch_size)}",
        f"data.num_workers={int(args.num_workers)}",
        "data.pin_memory=true",
        "data.prefetch_factor=6",
        "data.augment=true",
        "train.accelerator=gpu",
        "train.devices=1",
        "train.num_nodes=1",
        "train.strategy=auto",
        f"train.precision={args.precision}",
        f"train.max_epochs={int(args.max_epochs)}",
        "train.max_steps=-1",
        "train.limit_train_batches=1.0",
        "train.limit_val_batches=1.0",
        "train.limit_test_batches=0",
        "train.val_check_interval=1.0",
        "train.run_test_after_fit=false",
        "train.compute_rfid_after_fit=false",
        "train.rfid_split=val",
        f"train.rfid_batch_size={int(args.rfid_batch_size)}",
        f"train.rfid_num_workers={int(args.num_workers)}",
        "train.rfid_max_samples=0",
        "train.rfid_device=auto",
        "train.rfid_feature=2048",
        f"train.learning_rate={args.learning_rate}",
        "train.beta=0.5",
        "train.beta2=0.9",
        "train.warmup_steps=0",
        "train.min_lr_ratio=1.0",
        "train.accumulate_grad_batches=1",
        "train.gradient_clip_val=1.0",
        "train.log_every_n_steps=20",
        "train.deterministic=false",
        "model.type=laser",
        "model.backbone=ddpm",
        "model.num_downsamples=5",
        "model.channel_multipliers=[1,1,2,2,4,4]",
        "model.backbone_latent_channels=512",
        "model.attn_resolutions=[8,16]",
        "model.decoder_extra_residual_layers=2",
        "+model.decoder_upsample_mode=bilinear",
        "model.use_mid_attention=true",
        "model.in_channels=3",
        "model.dropout=0.0",
        "model.num_hiddens=128",
        "model.num_residual_blocks=3",
        "model.num_residual_hiddens=96",
        "model.bottleneck_type=dictionary",
        "model.num_embeddings=4096",
        "model.embedding_dim=128",
        "model.sparsity_level=4",
        "model.commitment_cost=0.25",
        "model.bottleneck_loss_weight=0.75",
        "model.dictionary_loss_weight=null",
        f"model.dict_learning_rate={args.dict_learning_rate}",
        "model.coef_max=null",
        "model.data_init_from_first_batch=false",
        "model.patch_based=false",
        "model.patch_size=1",
        "model.patch_stride=1",
        "model.patch_reconstruction=tile",
        "model.sparsity_reg_weight=0.0",
        "model.recon_mse_weight=1.0",
        "model.recon_l1_weight=0.0",
        "model.recon_edge_weight=0.0",
        "model.perceptual_weight=0.05",
        "model.perceptual_start_step=5000",
        "model.perceptual_warmup_steps=10000",
        "model.adversarial_weight=0.75",
        "model.adversarial_start_step=0",
        "model.adversarial_warmup_steps=0",
        "model.disc_start_step=0",
        f"model.disc_learning_rate={args.disc_learning_rate}",
        "model.discriminator_beta1=0.5",
        "model.discriminator_beta2=0.9",
        "model.disc_channels=64",
        "model.disc_num_layers=3",
        "model.disc_norm=group",
        "model.disc_spectral=false",
        "model.disc_loss=hinge",
        "model.use_adaptive_disc_weight=true",
        "model.disc_factor=1.0",
        "model.disc_weight_max=10000.0",
        "model.compute_fid=true",
        "model.fid_feature=2048",
        "model.log_images_every_n_steps=250",
        "model.diag_log_interval=250",
        "model.enable_val_latent_visuals=true",
        "model.codebook_visual_max_vectors=4096",
        "wandb.project=laser",
        f"wandb.name={RUN_NAME}",
        f"wandb.group={RUN_GROUP}",
        "wandb.tags=[stage1_adv,ffhq,laser,dictionary,adversarial,ozby19jo,adv50,bilinear,h100,continue]",
        "wandb.append_timestamp=false",
        f"wandb.save_dir={output_dir / 'wandb'}",
        f"wandb.id={RUN_ID}",
        "wandb.resume=must",
        "checkpoint.upload_to_wandb=true",
        "checkpoint.upload_every_n_epochs=1",
    ]


def command_env(output_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "WANDB_MODE": "online",
            "WANDB_RUN_ID": RUN_ID,
            "WANDB_RESUME": "must",
            "WANDB_PROJECT": "laser",
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


def run_command(cmd: list[str], args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "wandb").mkdir(parents=True, exist_ok=True)
    env = command_env(output_dir)
    print("WANDB_MODE=online")
    print("Launching:")
    print(" ".join(q(part) for part in cmd))
    if args.dry_run:
        return 0
    if args.background:
        log_path = output_dir / "resume_h100.log"
        with log_path.open("ab") as log:
            proc = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"Started background training pid={proc.pid}")
        print(f"Log: {log_path}")
        return 0
    return subprocess.run(cmd, cwd=str(REPO), env=env).returncode


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-path", default=RUN_PATH)
    parser.add_argument("--ckpt", default=None, help="Local checkpoint path. If omitted, try W&B artifact/run-file download.")
    parser.add_argument("--download-dir", default=str(DEFAULT_DOWNLOAD_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--batch-size", type=int, default=24, help="Single-H100 per-step batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--rfid-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", default="3.0e-5", help="Scaled from 4e-5 at effective batch 32 to batch 24.")
    parser.add_argument("--dict-learning-rate", default="1.875e-4", help="Scaled from 2.5e-4 at effective batch 32 to batch 24.")
    parser.add_argument("--disc-learning-rate", default="3.0e-5", help="Scaled discriminator LR for the single-H100 effective batch.")
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--background", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"FFHQ data directory not found: {data_dir}")

    if args.ckpt:
        ckpt = Path(args.ckpt).expanduser().resolve()
    else:
        ckpt = try_download_checkpoint(args.run_path, Path(args.download_dir).expanduser().resolve())
        if ckpt is None:
            raise SystemExit(
                "Cannot continue ozbyadv50 because W&B has no uploaded checkpoint for this run. "
                "Re-run with --ckpt /path/to/last.ckpt if the original scratch checkpoint is available, "
                "or upload that checkpoint to W&B and run this helper again."
            )
    if not ckpt.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt}")
    validate_torch_checkpoint(ckpt)

    cmd = [sys.executable, str(REPO / "train.py"), *build_stage1_args(args, ckpt)]
    return run_command(cmd, args)


if __name__ == "__main__":
    raise SystemExit(main())
