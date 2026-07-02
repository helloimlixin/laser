#!/usr/bin/env python3
"""Submit CelebA-HQ simple-backbone stage1_adv -> cache -> stage2 continuations."""

import argparse
import fnmatch
import os
import shlex
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


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


def user():
    return os.environ.get("USER", "unknown")


def scratch_path(*parts):
    return str(Path("/scratch") / user() / Path(*parts))


def q(value):
    return shlex.quote(str(value))


def snapshot_ignore(repo):
    repo = repo.resolve()

    def ignore(current_dir, names):
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


def snapshot_repo(repo, snapshot_root, stem):
    snapshot_root.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_root / stem
    if snapshot_path.exists():
        shutil.rmtree(str(snapshot_path))
    shutil.copytree(str(repo), str(snapshot_path), ignore=snapshot_ignore(repo))
    return snapshot_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--snapshot-root", default=scratch_path("submission_snapshots"))
    parser.add_argument(
        "--run-root-base",
        default=scratch_path("runs", "celebahq_simple_adv_stage2"),
    )
    parser.add_argument("--data-dir", default=scratch_path("Projects", "data", "celeba_hq"))
    parser.add_argument("--partition", default="gpu-redhat")
    parser.add_argument("--constraint", default="adalovelace")
    parser.add_argument("--time-limit", default="3-00:00:00")
    parser.add_argument("--cpus-per-task", type=int, default=24)
    parser.add_argument("--mem-mb", type=int, default=240000)
    parser.add_argument("--gpus", type=int, default=3)
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument("--stage1-adv-epochs", type=int, default=25)
    parser.add_argument("--stage2-epochs", type=int, default=300)
    parser.add_argument("--stage2-max-steps", type=int, default=150000)
    parser.add_argument("--stage2-batch-size", type=int, default=16)
    parser.add_argument("--stage2-lr", default="2.5e-4")
    parser.add_argument("--cache-batch-size", type=int, default=64)
    parser.add_argument("--cache-num-workers", type=int, default=8)
    parser.add_argument("--image", default="docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime")
    parser.add_argument("--pydeps", default=scratch_path(".pydeps", "laser_src_py311"))
    parser.add_argument("--wandb-project", default="laser")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.gpus < 1:
        raise SystemExit("--gpus must be >= 1")
    if args.stage2_batch_size < 1:
        raise SystemExit("--stage2-batch-size must be >= 1")
    return args


def default_variants():
    root = Path("/scratch") / user() / "runs" / "celebahq_stage1_simple_sweep" / "celebahq-full-stage1-sweep-simple-20260625_023236"
    return [
        {
            "label": "k2-a8192",
            "wandb_id": "ab5i0nam",
            "sparsity": 2,
            "num_embeddings": 8192,
            "stage1_ckpt": root / "k2-a8192" / "stage1" / "checkpoints" / "run_20260625_083339" / "laser" / "final.ckpt",
        },
        {
            "label": "k2-a16384",
            "wandb_id": "h7zjvmx4",
            "sparsity": 2,
            "num_embeddings": 16384,
            "stage1_ckpt": root / "k2-a16384" / "stage1" / "checkpoints" / "run_20260625_091051" / "laser" / "final.ckpt",
        },
    ]


def write_variants(path, variants, run_root):
    lines = []
    for idx, variant in enumerate(variants):
        out_dir = run_root / variant["label"]
        lines.append(
            "\t".join(
                [
                    str(idx),
                    variant["label"],
                    variant["wandb_id"],
                    str(variant["sparsity"]),
                    str(variant["num_embeddings"]),
                    str(variant["stage1_ckpt"]),
                    str(out_dir),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run_script(path, args, snapshot, run_root, variants_path, group):
    data_dir = Path(args.data_dir).expanduser().resolve()
    script = """#!/bin/bash
set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
VARIANT_LINE="$(awk -v id="$TASK_ID" 'BEGIN { FS="\\t" } $1 == id { print; found=1 } END { if (!found) exit 2 }' {variants_path})"
IFS=$'\\t' read -r TASK_ID LABEL SOURCE_WANDB_ID SPARSITY NUM_EMBEDDINGS STAGE1_CKPT RUN_DIR <<< "$VARIANT_LINE"

STAGE1_ADV_DIR="$RUN_DIR/stage1_adv"
STAGE2_DIR="$RUN_DIR/stage2"
CACHE_PATH="$STAGE2_DIR/token_cache.pt"

mkdir -p "$STAGE1_ADV_DIR" "$STAGE2_DIR"
echo "=== CelebA-HQ simple adv+stage2 continuation ==="
echo "task_id=$TASK_ID label=$LABEL source_wandb_id=$SOURCE_WANDB_ID"
echo "sparsity=$SPARSITY num_embeddings=$NUM_EMBEDDINGS"
echo "stage1_ckpt=$STAGE1_CKPT"
echo "run_dir=$RUN_DIR"
echo "slurm_job_id=${SLURM_JOB_ID:-unknown} array_task=${SLURM_ARRAY_TASK_ID:-none}"
echo "=== GPU inventory ==="
nvidia-smi
echo ""

export PYTHONUSERBASE={pydeps}
export PATH="$PYTHONUSERBASE/bin:$PATH"
export PYTHONPATH="$PYTHONUSERBASE/lib/python3.11/site-packages:$PYTHONUSERBASE/lib/python3.12/site-packages:{snapshot}${{PYTHONPATH:+:$PYTHONPATH}}"
export WANDB_MODE="${{WANDB_MODE:-online}}"
export LASER_DISABLE_WANDB_MEDIA="${{LASER_DISABLE_WANDB_MEDIA:-0}}"
export WANDB_DATA_DIR="${{WANDB_DATA_DIR:-$RUN_DIR/wandb/data}}"
export WANDB_CACHE_DIR="${{WANDB_CACHE_DIR:-$RUN_DIR/wandb/cache}}"
export WANDB_ARTIFACT_DIR="${{WANDB_ARTIFACT_DIR:-$RUN_DIR/wandb/artifacts}}"
export XDG_CACHE_HOME="${{XDG_CACHE_HOME:-/scratch/{user}/.cache}}"
export TORCH_HOME="${{TORCH_HOME:-$XDG_CACHE_HOME/torch}}"
export PIP_CACHE_DIR="${{PIP_CACHE_DIR:-$XDG_CACHE_HOME/pip}}"
export WANDB_CONFIG_DIR="${{WANDB_CONFIG_DIR:-/scratch/{user}/.config/wandb}}"
export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-4}}"
export PYTHONUNBUFFERED=1
export TMPDIR="/tmp/laser_celebahq_adv_s2_${{SLURM_JOB_ID:-$$}}_${{TASK_ID}}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
mkdir -p "$PYTHONUSERBASE" "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$WANDB_ARTIFACT_DIR" "$WANDB_CONFIG_DIR"

PYTHON_BIN="${{PYTHON_BIN:-$(command -v python3 || command -v python || true)}}"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "python3/python not found" >&2
  exit 127
fi

if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
  echo "ERROR: $PYTHON_BIN is too old; LASER requires Python >= 3.10." >&2
  exit 2
fi

if command -v flock >/dev/null 2>&1; then
  (
    flock 9
    "$PYTHON_BIN" -m pip install --user --quiet scipy wandb lightning omegaconf hydra-core rich 'torchmetrics[image]' torch-fidelity matplotlib lpips 2>/dev/null || true
  ) 9>"$PYTHONUSERBASE/.install.lock"
else
  "$PYTHON_BIN" -m pip install --user --quiet scipy wandb lightning omegaconf hydra-core rich 'torchmetrics[image]' torch-fidelity matplotlib lpips 2>/dev/null || true
fi

cd {snapshot}

if [[ ! -f "$STAGE1_CKPT" ]]; then
  echo "Missing stage-1 checkpoint: $STAGE1_CKPT" >&2
  exit 1
fi

ADV_CKPT="$(find "$STAGE1_ADV_DIR/checkpoints" -path '*/laser/final.ckpt' -type f 2>/dev/null | sort | tail -1 || true)"
if [[ -n "$ADV_CKPT" && -f "$ADV_CKPT" ]]; then
  echo "=== Reusing existing stage1_adv checkpoint: $ADV_CKPT ==="
else
  ADV_ARGS=(
    stage1
    seed=42
    output_dir="$STAGE1_ADV_DIR"
    init_ckpt_path="$STAGE1_CKPT"
    model=laser_image_nonpatch_d5
    data=celebahq
    data.data_dir={data_dir}
    data.batch_size=11
    data.eval_batch_size=11
    data.num_workers=8
    data.pin_memory=true
    data.prefetch_factor=4
    data.image_size=256
    data.train_crop_size=null
    data.augment=true
    train.accelerator=gpu
    train.devices={gpus}
    train.num_nodes=1
    train.strategy=ddp
    train.precision=bf16-mixed
    train.max_epochs={stage1_adv_epochs}
    train.max_steps=-1
    train.limit_train_batches=1.0
    train.limit_val_batches=1.0
    train.limit_test_batches=0
    train.val_check_interval=1.0
    train.run_test_after_fit=false
    train.compute_rfid_after_fit=true
    train.rfid_split=val
    train.rfid_batch_size=32
    train.rfid_num_workers=8
    train.rfid_max_samples=0
    train.rfid_device=auto
    train.rfid_feature=2048
    train.learning_rate=4.0e-5
    train.beta=0.5
    train.beta2=0.9
    train.warmup_steps=0
    train.min_lr_ratio=1.0
    train.accumulate_grad_batches=1
    train.gradient_clip_val=1.0
    train.log_every_n_steps=20
    train.deterministic=false
    model.bottleneck_type=dictionary
    model.dropout=0.0
    model.embedding_dim=128
    model.commitment_cost=0.25
    model.bottleneck_loss_weight=0.75
    model.dictionary_loss_weight=null
    model.dict_learning_rate=2.5e-4
    model.coef_max=null
    model.patch_based=false
    model.patch_size=1
    model.patch_stride=1
    model.patch_reconstruction=tile
    model.data_init_from_first_batch=false
    model.recon_mse_weight=1.0
    model.recon_l1_weight=0.0
    model.recon_edge_weight=0.0
    model.perceptual_weight=1.0
    model.perceptual_start_step=0
    model.perceptual_warmup_steps=0
    model.adversarial_weight=0.75
    model.adversarial_start_step=0
    model.adversarial_warmup_steps=0
    model.disc_start_step=0
    model.disc_learning_rate=null
    model.discriminator_beta1=0.5
    model.discriminator_beta2=0.9
    model.disc_channels=64
    model.disc_num_layers=3
    model.disc_norm=group
    model.disc_spectral=false
    model.disc_loss=hinge
    model.use_adaptive_disc_weight=true
    model.disc_factor=1.0
    model.compute_fid=true
    model.log_images_every_n_steps=250
    model.diag_log_interval=250
    model.enable_val_latent_visuals=true
    model.codebook_visual_max_vectors=4096
    model.backbone=simple
    model.num_embeddings="$NUM_EMBEDDINGS"
    model.sparsity_level="$SPARSITY"
    wandb.project={wandb_project}
    wandb.name="celebahq-stage1-adv-$LABEL-from-$SOURCE_WANDB_ID"
    wandb.group="{group}"
    'wandb.tags=[stage1_adv,celebahq,laser,dictionary,adversarial,simple,continuation]'
    wandb.append_timestamp=false
    wandb.save_dir="$STAGE1_ADV_DIR/wandb"
    checkpoint.upload_to_wandb=true
  )
  echo "=== Stage 1 adversarial fine-tune ==="
  printf 'Launching:'
  printf ' %q' "$PYTHON_BIN" train.py "${{ADV_ARGS[@]}}"
  printf '\\n'
  "$PYTHON_BIN" train.py "${{ADV_ARGS[@]}}"
  ADV_CKPT="$(find "$STAGE1_ADV_DIR/checkpoints" -path '*/laser/final.ckpt' -type f | sort | tail -1)"
fi

if [[ -z "$ADV_CKPT" || ! -f "$ADV_CKPT" ]]; then
  echo "Could not resolve stage1_adv final checkpoint under $STAGE1_ADV_DIR" >&2
  exit 1
fi

if [[ -f "$CACHE_PATH" ]]; then
  echo "=== Reusing token cache: $CACHE_PATH ==="
else
  echo "=== Building token cache from adversarial checkpoint ==="
  "$PYTHON_BIN" cache.py \\
    --stage1-checkpoint "$ADV_CKPT" \\
    --output-path "$CACHE_PATH" \\
    --dataset celebahq \\
    --data-dir {data_dir} \\
    --split train \\
    --image-size 256 \\
    --batch-size {cache_batch_size} \\
    --num-workers {cache_num_workers} \\
    --seed 42 \\
    --max-items 0 \\
    --model-type laser \\
    --coeff-bins 256 \\
    --coeff-max auto_p99.5 \\
    --coeff-quantization uniform
fi

STAGE2_ARGS=(
  stage2
  token_cache_path="$CACHE_PATH"
  output_dir="$STAGE2_DIR"
  seed=42
  data.dataset=celebahq
  data.data_dir={data_dir}
  data.image_size=256
  data.num_workers=8
  ar.type=sparse_spatial_depth
  ar.autoregressive_coeffs=true
  ar.class_conditional=false
  ar.max_steps={stage2_max_steps}
  ar.d_model=768
  ar.n_heads=12
  ar.n_layers=18
  ar.d_ff=3072
  ar.n_global_spatial_tokens=16
  ar.dropout=0.1
  ar.learning_rate={stage2_lr}
  ar.weight_decay=0.01
  ar.warmup_steps=5000
  ar.min_lr_ratio=0.08
  ar.coeff_loss_type=auto
  ar.coeff_loss_weight=1.0
  ar.coeff_huber_delta=0.25
  train_ar.max_epochs={stage2_epochs}
  train_ar.batch_size={stage2_batch_size}
  train_ar.max_items=0
  train_ar.limit_train_batches=1.0
  train_ar.limit_val_batches=1.0
  train_ar.limit_test_batches=0
  train_ar.val_check_interval=1.0
  train_ar.log_every_n_steps=20
  train_ar.devices={gpus}
  train_ar.num_nodes=1
  train_ar.strategy=ddp
  train_ar.precision=bf16-mixed
  train_ar.accelerator=gpu
  train_ar.deterministic=false
  train_ar.gradient_clip_val=1.0
  train_ar.checkpoint_save_top_k=1
  train_ar.checkpoint_save_last=true
  train_ar.checkpoint_keep_recent=3
  train_ar.checkpoint_every_n_epochs=5
  train_ar.sample_every_n_epochs=1
  train_ar.sample_every_n_steps=1000
  train_ar.sample_log_to_wandb=true
  train_ar.sample_num_images=16
  train_ar.sample_temperature=0.8
  train_ar.sample_top_k=0
  train_ar.compute_generation_fid=true
  train_ar.generation_metric_num_samples=5000
  train_ar.run_test_after_fit=false
  train_ar.save_final_samples_after_fit=true
  train_ar.checkpoint_upload_to_wandb=true
  wandb.project={wandb_project}
  wandb.name="celebahq-stage2-$LABEL-from-$SOURCE_WANDB_ID"
  wandb.group="{group}"
  'wandb.tags=[stage2,celebahq,laser,sparse_spatial_depth,unconditional,simple,continuation]'
  wandb.append_timestamp=false
  wandb.save_dir="$STAGE2_DIR/wandb"
)

S2_CKPT="$(find "$STAGE2_DIR/checkpoints" -path '*/last.ckpt' -type f 2>/dev/null | sort | tail -1 || true)"
if [[ -n "$S2_CKPT" && -f "$S2_CKPT" ]]; then
  STAGE2_ARGS+=("ckpt_path=$S2_CKPT")
  echo "=== Resuming stage 2 from $S2_CKPT ==="
else
  echo "=== Starting fresh stage 2 ==="
fi

printf 'Launching:'
printf ' %q' "$PYTHON_BIN" train.py "${{STAGE2_ARGS[@]}}"
printf '\\n'
exec "$PYTHON_BIN" train.py "${{STAGE2_ARGS[@]}}"
"""
    replacements = {
        "{variants_path}": q(variants_path),
        "{pydeps}": q(args.pydeps),
        "{snapshot}": q(snapshot),
        "{user}": user(),
        "{data_dir}": q(data_dir),
        "{gpus}": str(int(args.gpus)),
        "{stage1_adv_epochs}": str(int(args.stage1_adv_epochs)),
        "{wandb_project}": q(args.wandb_project),
        "{group}": group,
        "{cache_batch_size}": str(int(args.cache_batch_size)),
        "{cache_num_workers}": str(int(args.cache_num_workers)),
        "{stage2_max_steps}": str(int(args.stage2_max_steps)),
        "{stage2_lr}": str(args.stage2_lr),
        "{stage2_epochs}": str(int(args.stage2_epochs)),
        "{stage2_batch_size}": str(int(args.stage2_batch_size)),
    }
    for needle, value in replacements.items():
        script = script.replace(needle, value)
    script = script.replace("${{", "${").replace("}}", "}")
    path.write_text(script, encoding="utf-8")
    os.chmod(str(path), 0o755)


def write_sbatch(path, args, snapshot, run_root, run_script):
    script = """#!/bin/bash
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

echo "container_bin=$CONTAINER_BIN"
echo "image={image}"
nvidia-smi || true

if [[ -n "$CONTAINER_BIN" ]]; then
  "$CONTAINER_BIN" exec --nv \\
    --bind {snapshot} \\
    --bind /scratch/{user} \\
    --bind {data_dir} \\
    --bind {run_root} \\
    --bind /dev/shm \\
    {image} \\
    bash {run_script}
else
  echo "Warning: singularity/apptainer not found; running bare" >&2
  bash {run_script}
fi
""".format(
        image=q(args.image),
        snapshot=q(snapshot),
        user=user(),
        data_dir=q(Path(args.data_dir).expanduser().resolve()),
        run_root=q(run_root),
        run_script=q(run_script),
    )
    path.write_text(script, encoding="utf-8")
    os.chmod(str(path), 0o755)


def main():
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError("Data directory not found: " + str(data_dir))

    variants = default_variants()
    for variant in variants:
        if not Path(variant["stage1_ckpt"]).is_file():
            raise FileNotFoundError("Missing stage-1 checkpoint: " + str(variant["stage1_ckpt"]))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    group = "celebahq-simple-adv-stage2-" + stamp
    snapshot = snapshot_repo(
        repo,
        Path(args.snapshot_root).expanduser().resolve(),
        "laser_celebahq_simple_adv_stage2_" + stamp,
    )
    run_root = Path(args.run_root_base).expanduser().resolve() / group
    slurm_dir = run_root / "slurm"
    run_root.mkdir(parents=True, exist_ok=True)
    slurm_dir.mkdir(parents=True, exist_ok=True)

    variants_path = run_root / "variants.tsv"
    run_script = run_root / "run_variant.sh"
    sbatch_script = run_root / "sbatch_array.sh"
    write_variants(variants_path, variants, run_root)
    write_run_script(run_script, args, snapshot, run_root, variants_path, group)
    write_sbatch(sbatch_script, args, snapshot, run_root, run_script)

    log_base = slurm_dir / "celebahq_simple_adv_stage2_%A_%a"
    array_spec = "0-{}".format(len(variants) - 1)
    if int(args.max_concurrent) > 0:
        array_spec += "%{}".format(int(args.max_concurrent))
    cmd = [
        "sbatch",
        "--partition=" + str(args.partition),
        "--job-name=chq-simple-adv-s2",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=" + str(int(args.cpus_per_task)),
        "--gres=gpu:" + str(int(args.gpus)),
        "--mem=" + str(int(args.mem_mb)),
        "--time=" + str(args.time_limit),
        "--chdir=" + str(snapshot),
        "--array=" + array_spec,
        "--output=" + str(log_base) + ".out",
        "--error=" + str(log_base) + ".err",
        "--requeue",
    ]
    if str(args.constraint).strip():
        cmd.append("--constraint=" + str(args.constraint))
    cmd.append(str(sbatch_script))

    print("Snapshot:", snapshot)
    print("Run root:", run_root)
    print("Variants:", variants_path)
    for idx, variant in enumerate(variants):
        print("[task {}] {} {} ckpt={}".format(idx, variant["label"], variant["wandb_id"], variant["stage1_ckpt"]))
    if args.dry_run:
        print("Dry run sbatch:", " ".join(q(part) for part in cmd))
        return 0

    proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    job_id = (proc.stdout or proc.stderr).strip().split()[-1]
    print("Submitted array job:", job_id)
    for idx, variant in enumerate(variants):
        print(
            "[task {idx}] {label}: stdout={out} stderr={err}".format(
                idx=idx,
                label=variant["label"],
                out=slurm_dir / ("celebahq_simple_adv_stage2_" + job_id + "_" + str(idx) + ".out"),
                err=slurm_dir / ("celebahq_simple_adv_stage2_" + job_id + "_" + str(idx) + ".err"),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
