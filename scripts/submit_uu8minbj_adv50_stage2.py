#!/usr/bin/env python3
"""Submit FFHQ uu8minbj stage1_adv50 -> quantile cache -> stage2-200."""

import argparse
import fnmatch
import os
import shlex
import shutil
import subprocess
import textwrap
from datetime import datetime
from pathlib import Path
from typing import List, Set, Tuple


USER = os.environ.get("USER", "xl598")
REPO = Path("/scratch/xl598/Projects/laser")
SNAPSHOT_ROOT = Path("/scratch/xl598/submission_snapshots")
RUN_ROOT_BASE = Path("/scratch/xl598/runs/ffhq_uu8minbj_adv50_stage2")
DATA_DIR = Path("/scratch/xl598/Projects/data/ffhq")
INIT_CKPT = Path(
    "/scratch/xl598/runs/ffhq_stage1_sweep_lpips005/"
    "ffhq-laser-paper-stage1-sweep-20260628_022906/k2-a4096/stage1/"
    "checkpoints/run_slurm57712518_0/laser/uu8minbj-final.ckpt"
)
PYDEPS = Path("/scratch/xl598/.pydeps/laser_src_py311")
IMAGE = "docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime"
VGG16_WEIGHTS = Path("/scratch/xl598/.cache/torch/hub/checkpoints/vgg16-397923af.pth")

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


def q(value: object) -> str:
    return shlex.quote(str(value))


def bash_array(items: List[str]) -> str:
    return "\n".join(f"  {q(item)}" for item in items)


def script_body(raw: str) -> str:
    lines = textwrap.dedent(raw).lstrip("\n").splitlines()
    first = next((line for line in lines if line.strip()), "")
    leading = len(first) - len(first.lstrip(" "))
    prefix = " " * leading
    cleaned = []
    for line in lines:
        if leading and line.startswith(prefix):
            line = line[leading:]
        cleaned.append(line.rstrip())
    return "\n".join(cleaned).rstrip() + "\n"


def snapshot_ignore(repo: Path):
    repo = repo.resolve()

    def ignore(current_dir: str, names: List[str]) -> Set[str]:
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


def snapshot_repo(repo: Path, snapshot: Path) -> None:
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if snapshot.exists():
        raise FileExistsError(f"snapshot already exists: {snapshot}")
    shutil.copytree(repo, snapshot, ignore=snapshot_ignore(repo))


def stage1_adv_args(args: argparse.Namespace, output_dir: Path) -> List[str]:
    return [
        "stage1",
        "seed=42",
        f"output_dir={output_dir}",
        "model=laser_image_nonpatch_d5",
        "data=ffhq",
        f"data.data_dir={args.data_dir}",
        "data.batch_size=8",
        "data.eval_batch_size=8",
        "data.num_workers=8",
        "data.pin_memory=true",
        "data.prefetch_factor=6",
        "data.image_size=256",
        "data.train_crop_size=null",
        "data.augment=true",
        "train.accelerator=gpu",
        f"train.devices={int(args.gpus)}",
        "train.num_nodes=1",
        "train.strategy=ddp",
        "train.precision=bf16-mixed",
        f"train.max_epochs={int(args.stage1_adv_epochs)}",
        "train.max_steps=-1",
        "train.limit_train_batches=1.0",
        "train.limit_val_batches=1.0",
        "train.limit_test_batches=0",
        "train.val_check_interval=1.0",
        "train.run_test_after_fit=false",
        "train.compute_rfid_after_fit=true",
        "train.rfid_split=val",
        "train.rfid_batch_size=32",
        "train.rfid_num_workers=8",
        "train.rfid_max_samples=0",
        "train.rfid_device=auto",
        "train.rfid_feature=2048",
        "train.learning_rate=4.0e-5",
        "train.beta=0.5",
        "train.beta2=0.9",
        "train.warmup_steps=0",
        "train.min_lr_ratio=1.0",
        "train.accumulate_grad_batches=1",
        "train.gradient_clip_val=1.0",
        "train.log_every_n_steps=20",
        "train.deterministic=false",
        "model.bottleneck_type=dictionary",
        "model.dropout=0.0",
        "model.embedding_dim=128",
        "model.commitment_cost=0.25",
        "model.bottleneck_loss_weight=0.75",
        "model.dictionary_loss_weight=null",
        "model.dict_learning_rate=2.5e-4",
        "model.coef_max=null",
        "model.patch_based=false",
        "model.patch_size=1",
        "model.patch_stride=1",
        "model.patch_reconstruction=tile",
        "model.data_init_from_first_batch=false",
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
        "model.disc_learning_rate=null",
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
        "model.log_images_every_n_steps=250",
        "model.diag_log_interval=250",
        "model.enable_val_latent_visuals=true",
        "model.codebook_visual_max_vectors=4096",
        "model.num_embeddings=4096",
        "model.sparsity_level=2",
        "wandb.project=laser",
        f"wandb.name={args.adv_wandb_name}",
        f"+wandb.id={args.adv_wandb_id}",
        "+wandb.resume=allow",
        f"wandb.group={args.wandb_group}",
        "wandb.tags=[stage1_adv,ffhq,laser,dictionary,adversarial,uu8minbj,adv50]",
        "wandb.append_timestamp=false",
        f"wandb.save_dir={output_dir / 'wandb'}",
        "checkpoint.upload_to_wandb=true",
    ]


def stage2_args(args: argparse.Namespace, output_dir: Path, cache_path: Path) -> List[str]:
    return [
        "stage2",
        f"token_cache_path={cache_path}",
        f"output_dir={output_dir}",
        "seed=42",
        "data.dataset=ffhq",
        f"data.data_dir={args.data_dir}",
        "data.image_size=256",
        "data.num_workers=8",
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
        f"train_ar.max_epochs={int(args.stage2_epochs)}",
        f"train_ar.batch_size={int(args.stage2_batch_size)}",
        "train_ar.max_items=0",
        "train_ar.limit_train_batches=1.0",
        "train_ar.limit_val_batches=1.0",
        "train_ar.limit_test_batches=0",
        "train_ar.val_check_interval=1.0",
        "train_ar.validation_split=0.05",
        "train_ar.test_split=0.05",
        "train_ar.log_every_n_steps=20",
        f"train_ar.devices={int(args.gpus)}",
        "train_ar.num_nodes=1",
        "train_ar.strategy=ddp",
        "train_ar.precision=bf16-mixed",
        "train_ar.accelerator=gpu",
        "train_ar.deterministic=false",
        "train_ar.accumulate_grad_batches=1",
        "train_ar.gradient_clip_val=1.0",
        "train_ar.checkpoint_save_top_k=1",
        "train_ar.checkpoint_save_last=true",
        "train_ar.checkpoint_keep_recent=3",
        "train_ar.checkpoint_every_n_epochs=5",
        "+train_ar.checkpoint_upload_to_wandb=true",
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
        f"wandb.name={args.stage2_wandb_name}",
        f"+wandb.id={args.stage2_wandb_id}",
        "+wandb.resume=allow",
        f"wandb.group={args.wandb_group}",
        "wandb.tags=[stage2,ffhq,laser,sparse_spatial_depth,unconditional,uu8minbj,stage2_200]",
        "wandb.append_timestamp=false",
        f"wandb.save_dir={output_dir / 'wandb'}",
    ]


def write_job_files(args: argparse.Namespace, snapshot: Path, job_root: Path) -> Tuple[Path, Path]:
    stage1_adv_dir = job_root / "stage1_adv50"
    stage2_dir = job_root / "stage2"
    cache_path = stage2_dir / "token_cache" / "ffhq__train__img256__laser_cb256_quantile_p99p5.pt"
    slurm_dir = job_root / "slurm"
    for path in (stage1_adv_dir, stage2_dir, cache_path.parent, slurm_dir):
        path.mkdir(parents=True, exist_ok=True)

    run_script = job_root / "run_uu8minbj_adv50_stage2.sh"
    sbatch_script = job_root / "sbatch_uu8minbj_adv50_stage2.sh"
    log_base = slurm_dir / "uu8minbj_adv50_stage2"
    stage1_lines = bash_array(stage1_adv_args(args, stage1_adv_dir))
    stage2_lines = bash_array(stage2_args(args, stage2_dir, cache_path))

    run_script.write_text(
        script_body(
            f"""\
            #!/bin/bash
            set -euo pipefail

            INIT_CKPT={q(args.init_ckpt)}
            STAGE1_ADV_DIR={q(stage1_adv_dir)}
            STAGE2_DIR={q(stage2_dir)}
            CACHE_PATH={q(cache_path)}

            export PYTHONUSERBASE={q(args.pydeps)}
            export PATH="$PYTHONUSERBASE/bin:$PATH"
            export PYTHONPATH="$PYTHONUSERBASE/lib/python3.11/site-packages:$PYTHONUSERBASE/lib/python3.12/site-packages:{q(snapshot)}${{PYTHONPATH:+:$PYTHONPATH}}"
            export WANDB_MODE="${{WANDB_MODE:-online}}"
            export WANDB_RESUME="${{WANDB_RESUME:-allow}}"
            export LASER_DISABLE_WANDB_MEDIA="${{LASER_DISABLE_WANDB_MEDIA:-0}}"
            export WANDB_DATA_DIR="${{WANDB_DATA_DIR:-{job_root}/wandb/data}}"
            export WANDB_CACHE_DIR="${{WANDB_CACHE_DIR:-{job_root}/wandb/cache}}"
            export WANDB_ARTIFACT_DIR="${{WANDB_ARTIFACT_DIR:-{job_root}/wandb/artifacts}}"
            export WANDB_CONFIG_DIR="${{WANDB_CONFIG_DIR:-/scratch/{USER}/.config/wandb}}"
            export XDG_CACHE_HOME="${{XDG_CACHE_HOME:-/scratch/{USER}/.cache}}"
            export TORCH_HOME="${{TORCH_HOME:-$XDG_CACHE_HOME/torch}}"
            export PIP_CACHE_DIR="${{PIP_CACHE_DIR:-$XDG_CACHE_HOME/pip}}"
            export MPLCONFIGDIR="${{MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}}"
            export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
            export HYDRA_FULL_ERROR=1
            export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-4}}"
            export PYTHONUNBUFFERED=1
            export TMPDIR="/tmp/laser_uu8_adv_s2_${{SLURM_JOB_ID:-$$}}"
            export TEMP="$TMPDIR"
            export TMP="$TMPDIR"
            if [[ -z "${{LASER_VGG16_WEIGHTS:-}}" && -f {q(VGG16_WEIGHTS)} ]]; then
              export LASER_VGG16_WEIGHTS={q(VGG16_WEIGHTS)}
            fi

            mkdir -p "$PYTHONUSERBASE" "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" \\
              "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$WANDB_ARTIFACT_DIR" "$WANDB_CONFIG_DIR" "$MPLCONFIGDIR" \\
              "$STAGE1_ADV_DIR/wandb" "$STAGE2_DIR/wandb" "$(dirname "$CACHE_PATH")"

            PYTHON_BIN="${{PYTHON_BIN:-$(command -v python3 || command -v python || true)}}"
            if [[ -z "$PYTHON_BIN" ]]; then
              echo "python3/python not found" >&2
              exit 127
            fi
            if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
              echo "ERROR: $PYTHON_BIN is too old; LASER requires Python >= 3.10." >&2
              exit 2
            fi

            if command -v nvidia-smi >/dev/null 2>&1; then
              nvidia-smi || true
            fi

            if command -v flock >/dev/null 2>&1; then
              (
                flock 9
                "$PYTHON_BIN" -m pip install --user --quiet scipy wandb lightning omegaconf hydra-core rich 'torchmetrics[image]' torch-fidelity matplotlib lpips tqdm 2>/dev/null || true
              ) 9>"$PYTHONUSERBASE/.install.lock"
            else
              "$PYTHON_BIN" -m pip install --user --quiet scipy wandb lightning omegaconf hydra-core rich 'torchmetrics[image]' torch-fidelity matplotlib lpips tqdm 2>/dev/null || true
            fi

            if [[ ! -f "$INIT_CKPT" ]]; then
              echo "Missing init checkpoint: $INIT_CKPT" >&2
              exit 1
            fi

            cd {q(snapshot)}

            find_latest() {{
              local root="$1"
              shift
              if [[ ! -d "$root" ]]; then
                return 0
              fi
              find "$root" "$@" -type f -printf '%T@ %p\\n' 2>/dev/null | sort -nr | awk 'NR == 1 {{ sub(/^[^ ]+ /, ""); print; }}'
            }}

            ADV_FINAL="$(find_latest "$STAGE1_ADV_DIR/checkpoints" \\( -name '*-final.ckpt' -o -name 'final.ckpt' \\))"
            if [[ -n "$ADV_FINAL" && -f "$ADV_FINAL" ]]; then
              echo "=== Reusing completed stage1_adv checkpoint: $ADV_FINAL ==="
            else
              ADV_RESUME="$(find_latest "$STAGE1_ADV_DIR/checkpoints" \\( -name '*-last.ckpt' -o -name 'last.ckpt' \\))"
              ADV_ARGS=(
            {stage1_lines}
              )
              if [[ -n "$ADV_RESUME" && -f "$ADV_RESUME" ]]; then
                echo "=== Resuming stage1_adv from: $ADV_RESUME ==="
                ADV_ARGS+=("ckpt_path=$ADV_RESUME")
              else
                echo "=== Initializing stage1_adv from: $INIT_CKPT ==="
                ADV_ARGS+=("init_ckpt_path=$INIT_CKPT")
              fi
              unset WANDB_RUN_ID WANDB_ID
              printf 'Launching stage1_adv:'
              printf ' %q' "$PYTHON_BIN" train.py "${{ADV_ARGS[@]}}"
              printf '\\n'
              "$PYTHON_BIN" train.py "${{ADV_ARGS[@]}}"
              ADV_FINAL="$(find_latest "$STAGE1_ADV_DIR/checkpoints" \\( -name '*-final.ckpt' -o -name 'final.ckpt' \\))"
            fi

            if [[ -z "$ADV_FINAL" || ! -f "$ADV_FINAL" ]]; then
              echo "Could not resolve stage1_adv final checkpoint under $STAGE1_ADV_DIR" >&2
              exit 1
            fi

            if [[ -f "$CACHE_PATH" ]]; then
              echo "=== Reusing token cache: $CACHE_PATH ==="
            else
              echo "=== Building FFHQ quantile token cache from: $ADV_FINAL ==="
              "$PYTHON_BIN" scripts/tools/build_token_cache.py \\
                --stage1_checkpoint "$ADV_FINAL" \\
                --dataset ffhq \\
                --data_dir {q(args.data_dir)} \\
                --split train \\
                --cache_mode quantized \\
                --image_size 256 \\
                --batch_size {int(args.cache_batch_size)} \\
                --num_workers {int(args.cache_num_workers)} \\
                --mean 0.5 0.5 0.5 \\
                --std 0.5 0.5 0.5 \\
                --coeff_vocab_size 256 \\
                --coeff_quantization quantile \\
                --coeff_calibration_percentile 99.5 \\
                --output "$CACHE_PATH" \\
                --max_items 0 \\
                --device auto
            fi

            STAGE2_RESUME="$(find_latest "$STAGE2_DIR/checkpoints" \\( -name '*-last.ckpt' -o -name 'last.ckpt' \\))"
            STAGE2_ARGS=(
            {stage2_lines}
            )
            if [[ -n "$STAGE2_RESUME" && -f "$STAGE2_RESUME" ]]; then
              echo "=== Resuming stage2 from: $STAGE2_RESUME ==="
              STAGE2_ARGS+=("ckpt_path=$STAGE2_RESUME")
            else
              echo "=== Starting fresh stage2 for {int(args.stage2_epochs)} epochs ==="
            fi
            unset WANDB_RUN_ID WANDB_ID
            printf 'Launching stage2:'
            printf ' %q' "$PYTHON_BIN" train.py "${{STAGE2_ARGS[@]}}"
            printf '\\n'
            exec "$PYTHON_BIN" train.py "${{STAGE2_ARGS[@]}}"
            """
        ),
        encoding="utf-8",
    )
    os.chmod(run_script, 0o755)

    constraint_line = f"#SBATCH --constraint={args.constraint}\n" if str(args.constraint or "").strip() else ""
    exclude_line = f"#SBATCH --exclude={args.exclude}\n" if str(args.exclude or "").strip() else ""
    sbatch_script.write_text(
        script_body(
            f"""\
            #!/bin/bash
            #SBATCH --job-name=ffhq-uu8-adv-s2
            #SBATCH --partition={args.partition}
            #SBATCH --nodes=1
            #SBATCH --ntasks=1
            #SBATCH --ntasks-per-node=1
            #SBATCH --gres=gpu:{args.gpus}
            #SBATCH --cpus-per-task={args.cpus_per_task}
            #SBATCH --mem={args.mem_mb}
            #SBATCH --time={args.time_limit}
            {constraint_line}{exclude_line}#SBATCH --output={log_base}_%j.out
            #SBATCH --error={log_base}_%j.err
            #SBATCH --requeue

            set -euo pipefail

            if ! command -v module >/dev/null 2>&1; then
              if [[ -f /usr/share/lmod/lmod/init/bash ]]; then
                set +u; source /usr/share/lmod/lmod/init/bash; set -u
              elif [[ -f /usr/share/Modules/init/bash ]]; then
                set +u; source /usr/share/Modules/init/bash; set -u
              elif [[ -f /etc/profile.d/modules.sh ]]; then
                set +u; source /etc/profile.d/modules.sh; set -u
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

            export APPTAINER_CACHEDIR="${{APPTAINER_CACHEDIR:-/cache/home/{USER}/.apptainer/cache}}"
            export SINGULARITY_CACHEDIR="${{SINGULARITY_CACHEDIR:-$APPTAINER_CACHEDIR}}"
            export APPTAINER_TMPDIR="${{APPTAINER_TMPDIR:-/scratch/{USER}/.apptainer/tmp_${{SLURM_JOB_ID:-manual}}}}"
            export SINGULARITY_TMPDIR="${{SINGULARITY_TMPDIR:-$APPTAINER_TMPDIR}}"
            mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

            IMAGE="${{IMAGE:-{args.image}}}"
            echo "container_bin=$CONTAINER_BIN"
            echo "image=$IMAGE"
            echo "snapshot={snapshot}"
            echo "job_root={job_root}"

            if [[ -n "$CONTAINER_BIN" ]]; then
              srun "$CONTAINER_BIN" exec --nv \\
                --bind /cache/home/{USER} \\
                --bind {q(snapshot)} \\
                --bind /scratch/{USER} \\
                --bind {q(args.data_dir)} \\
                --bind {q(job_root)} \\
                --bind {q(Path(args.init_ckpt).parent)} \\
                --bind /dev/shm \\
                "$IMAGE" \\
                bash {q(run_script)}
            else
              echo "Warning: singularity/apptainer not found; running bare" >&2
              srun bash {q(run_script)}
            fi
            """
        ),
        encoding="utf-8",
    )
    os.chmod(sbatch_script, 0o755)
    return run_script, sbatch_script


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(REPO))
    parser.add_argument("--snapshot-root", default=str(SNAPSHOT_ROOT))
    parser.add_argument("--run-root-base", default=str(RUN_ROOT_BASE))
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--init-ckpt", default=str(INIT_CKPT))
    parser.add_argument("--pydeps", default=str(PYDEPS))
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--partition", default="gpu-redhat")
    parser.add_argument("--constraint", default="adalovelace")
    parser.add_argument("--exclude", default="gpu029")
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--cpus-per-task", type=int, default=32)
    parser.add_argument("--mem-mb", type=int, default=240000)
    parser.add_argument("--time-limit", default="3-00:00:00")
    parser.add_argument("--stage1-adv-epochs", type=int, default=50)
    parser.add_argument("--stage2-epochs", type=int, default=200)
    parser.add_argument("--stage2-batch-size", type=int, default=16)
    parser.add_argument("--stage2-lr", default="2.5e-4")
    parser.add_argument("--cache-batch-size", type=int, default=64)
    parser.add_argument("--cache-num-workers", type=int, default=8)
    parser.add_argument("--wandb-group", default="")
    parser.add_argument("--adv-wandb-id", default="")
    parser.add_argument("--stage2-wandb-id", default="")
    parser.add_argument("--adv-wandb-name", default="ffhq-stage1-adv50-k2-a4096-from-uu8minbj")
    parser.add_argument("--stage2-wandb-name", default="ffhq-stage2-200-k2-a4096-from-uu8minbj-adv50")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.gpus <= 0:
        raise SystemExit("--gpus must be positive")
    if args.stage1_adv_epochs <= 0:
        raise SystemExit("--stage1-adv-epochs must be positive")
    if args.stage2_epochs <= 0:
        raise SystemExit("--stage2-epochs must be positive")
    if args.stage2_batch_size <= 0:
        raise SystemExit("--stage2-batch-size must be positive")
    return args


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    init_ckpt = Path(args.init_ckpt).expanduser().resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"repo not found: {repo}")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"FFHQ data dir not found: {data_dir}")
    if not init_ckpt.is_file():
        raise FileNotFoundError(f"init checkpoint not found: {init_ckpt}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    group = str(args.wandb_group or f"ffhq-uu8minbj-adv50-stage2-{stamp}")
    run_token = stamp.replace("_", "")
    args.data_dir = str(data_dir)
    args.init_ckpt = str(init_ckpt)
    args.pydeps = str(Path(args.pydeps).expanduser().resolve())
    args.image = str(args.image)
    args.wandb_group = group
    args.adv_wandb_id = str(args.adv_wandb_id or f"uu8adv{run_token[-6:]}")
    args.stage2_wandb_id = str(args.stage2_wandb_id or f"uu8s2{run_token[-6:]}")

    snapshot = Path(args.snapshot_root).expanduser().resolve() / f"laser_uu8minbj_adv50_stage2_{stamp}"
    job_root = Path(args.run_root_base).expanduser().resolve() / group
    job_root.mkdir(parents=True, exist_ok=True)
    snapshot_repo(repo, snapshot)
    _, sbatch_script = write_job_files(args, snapshot, job_root)

    if args.dry_run:
        job_id = "dry-run"
        print("Dry run: not submitting.")
    else:
        result = subprocess.run(
            ["sbatch", str(sbatch_script)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        print(result.stdout.strip())
        job_id = (result.stdout or result.stderr).strip().split()[-1]

    print(f"Snapshot: {snapshot}")
    print(f"Job root: {job_root}")
    print(f"SBATCH: {sbatch_script}")
    print(f"Init checkpoint: {init_ckpt}")
    print(f"Stage1 adv epochs: {args.stage1_adv_epochs}")
    print(f"Stage2 epochs: {args.stage2_epochs}")
    print(f"Stage2 batch size per process: {args.stage2_batch_size}")
    print(f"W&B group: {group}")
    print(f"Stage1 adv W&B: helloimlixin-rutgers/laser/{args.adv_wandb_id} ({args.adv_wandb_name})")
    print(f"Stage2 W&B: helloimlixin-rutgers/laser/{args.stage2_wandb_id} ({args.stage2_wandb_name})")
    if job_id != "dry-run":
        print(f"stdout: {job_root / 'slurm' / ('uu8minbj_adv50_stage2_' + job_id + '.out')}")
        print(f"stderr: {job_root / 'slurm' / ('uu8minbj_adv50_stage2_' + job_id + '.err')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
