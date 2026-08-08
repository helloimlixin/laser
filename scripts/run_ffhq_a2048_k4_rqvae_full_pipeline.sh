#!/usr/bin/env bash
# End-to-end FFHQ LASER a2048/k4 pipeline using KakaoBrain's RQ-VAE/RQ-Transformer
# geometry, losses, optimizer settings, official 60k/10k split, and augmentations.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xl598/anaconda3/envs/laser/bin/python}"
DATA_ROOT="${FFHQ_ROOT:-/home/xl598/Projects/data/ffhq}"
STAMP="${STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/outputs/ffhq-a2048-k4-rqvae-full-$STAMP}"
STAGE1_OUT="$RUN_ROOT/stage1"
RFID_RESULT="$RUN_ROOT/stage1/rfid_ffhq_70000.json"
TOKEN_CACHE="$RUN_ROOT/token_cache/ffhq_full_a2048_k4_compound.pt"
TOKEN_REPORT="${TOKEN_CACHE%.pt}.validation.json"
STAGE2_OUT="$RUN_ROOT/stage2-compound-official-rqtransformer-350M"
TRAIN_LIST="$ROOT/third_party/rq-vae-transformer/rqvae/img_datasets/assets/ffhqtrain.txt"
VAL_LIST="$ROOT/third_party/rq-vae-transformer/rqvae/img_datasets/assets/ffhqvalidation.txt"
EXPECTED_FFHQ_IMAGES=70000

PIPELINE_CUDA_VISIBLE_DEVICES="${PIPELINE_CUDA_VISIBLE_DEVICES:-1}"
NPROC="${NPROC:-1}"
STAGE1_MICROBATCH="${STAGE1_MICROBATCH:-8}"
STAGE1_EVAL_BATCH="${STAGE1_EVAL_BATCH:-8}"
STAGE1_PRECISION="${STAGE1_PRECISION:-32}"
STAGE2_MICROBATCH="${STAGE2_MICROBATCH:-2}"
STAGE2_FID_BATCH="${STAGE2_FID_BATCH:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TASK_TMPDIR="${TASK_TMPDIR:-/tmp/laser_ffhq_k4_$STAMP}"

for required in "$PYTHON_BIN" "$DATA_ROOT" "$TRAIN_LIST" "$VAL_LIST"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 1
  fi
done

image_count="$(find "$DATA_ROOT" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' \) | wc -l)"
if [[ "$image_count" -ne "$EXPECTED_FFHQ_IMAGES" ]]; then
  echo "Expected a complete ${EXPECTED_FFHQ_IMAGES}-image FFHQ corpus; found $image_count" >&2
  exit 1
fi
if (( 128 % (STAGE1_MICROBATCH * NPROC) != 0 )); then
  echo "Stage-1 microbatch * world size must divide the official total batch 128" >&2
  exit 1
fi
if (( 128 % (STAGE2_MICROBATCH * NPROC) != 0 )); then
  echo "Stage-2 microbatch * world size must divide the official total batch 128" >&2
  exit 1
fi
STAGE1_ACCUMULATE=$((128 / (STAGE1_MICROBATCH * NPROC)))

mkdir -p "$STAGE1_OUT" "$RUN_ROOT/token_cache" "$STAGE2_OUT" \
  "$RUN_ROOT/logs" "$ROOT/.cache/wandb" "$ROOT/.local/share/wandb" "$ROOT/wandb" \
  "$TASK_TMPDIR"
printf '%s\n' "$$" > "$RUN_ROOT/pipeline.pid"
if [[ ! -s "$RUN_ROOT/status.tsv" ]]; then
  printf 'time_utc\tphase\tstate\tdetail\n' > "$RUN_ROOT/status.tsv"
fi

export CUDA_VISIBLE_DEVICES="$PIPELINE_CUDA_VISIBLE_DEVICES"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RESUME=allow
export WANDB_CACHE_DIR="$ROOT/.cache/wandb"
export WANDB_DATA_DIR="$ROOT/.local/share/wandb"
export WANDB_DIR="$ROOT/wandb"
export XDG_CACHE_HOME="$ROOT/.cache"
export TMPDIR="$TASK_TMPDIR"
export TMP="$TASK_TMPDIR"
export TEMP="$TASK_TMPDIR"
export LASER_VGG16_WEIGHTS="${LASER_VGG16_WEIGHTS:-/home/xl598/.cache/torch/hub/checkpoints/vgg16-397923af.pth}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

status() {
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "${3:-}" \
    >> "$RUN_ROOT/status.tsv"
}

latest_checkpoint() {
  find "$1" -type f -name "$2" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | awk 'NR == 1 { sub(/^[^ ]+ /, ""); print; }'
}

active_phase=driver
on_exit() {
  exit_code="$?"
  if (( exit_code == 0 )); then
    status "$active_phase" complete "pipeline exit=0"
  else
    status "$active_phase" failed "pipeline exit=$exit_code"
  fi
}
trap on_exit EXIT

printf '%s\n' \
  "run_root=$RUN_ROOT" \
  "data_root=$DATA_ROOT" \
  "cuda_visible_devices=$PIPELINE_CUDA_VISIBLE_DEVICES" \
  "world_size=$NPROC" \
  "stage1_microbatch=$STAGE1_MICROBATCH" \
  "stage1_accumulation=$STAGE1_ACCUMULATE" \
  "stage1_effective_batch=128" \
  "stage1_precision=$STAGE1_PRECISION" \
  "stage2_microbatch=$STAGE2_MICROBATCH" \
  "stage2_effective_batch=128" \
  "restart=RUN_ROOT=$RUN_ROOT STAMP=$STAMP PIPELINE_CUDA_VISIBLE_DEVICES=$PIPELINE_CUDA_VISIBLE_DEVICES NPROC=$NPROC $0" \
  > "$RUN_ROOT/run.info"

active_phase=stage1
stage1_checkpoint="$(latest_checkpoint "$STAGE1_OUT" final.ckpt)"
if [[ ! -f "$STAGE1_OUT/.phase_complete" ]]; then
  resume_args=()
  stage1_last="$(latest_checkpoint "$STAGE1_OUT" last.ckpt)"
  if [[ -n "$stage1_last" ]]; then
    resume_args+=("ckpt_path=$stage1_last")
  fi
  status "$active_phase" starting \
    "official FFHQ stage1; a2048 k4; microbatch=$STAGE1_MICROBATCH accumulation=$STAGE1_ACCUMULATE"
  "$PYTHON_BIN" "$ROOT/train.py" stage1 \
    model=laser_ffhq_rqvae_a2048_k4 \
    data=ffhq \
    seed=42 \
    "data.data_dir=$DATA_ROOT" \
    data.image_size=256 \
    "data.batch_size=$STAGE1_MICROBATCH" \
    "data.eval_batch_size=$STAGE1_EVAL_BATCH" \
    "data.num_workers=$NUM_WORKERS" \
    data.train_crop_size=null \
    data.augment=true \
    "+data.train_list_file=$TRAIN_LIST" \
    "+data.val_list_file=$VAL_LIST" \
    '+data.train_random_resized_crop_scale=[0.75,1.0]' \
    train.accelerator=gpu \
    train.num_nodes=1 \
    "train.devices=$NPROC" \
    "train.strategy=$([[ "$NPROC" -gt 1 ]] && echo ddp || echo auto)" \
    "train.precision=$STAGE1_PRECISION" \
    train.max_epochs=150 \
    train.max_steps=-1 \
    train.learning_rate=4.0e-5 \
    train.beta=0.5 \
    train.beta2=0.9 \
    train.warmup_steps=2345 \
    train.min_lr_ratio=1.0 \
    "train.accumulate_grad_batches=$STAGE1_ACCUMULATE" \
    train.gradient_clip_val=0.0 \
    train.deterministic=false \
    train.log_every_n_steps=25 \
    train.limit_train_batches=1.0 \
    train.limit_val_batches=1.0 \
    train.limit_test_batches=0 \
    train.val_check_interval=1.0 \
    train.run_test_after_fit=false \
    train.compute_rfid_after_fit=false \
    checkpoint.monitor=val/rfid \
    checkpoint.mode=min \
    checkpoint.save_top_k=3 \
    checkpoint.save_last=true \
    checkpoint.every_n_epochs=5 \
    checkpoint.upload_to_wandb=false \
    wandb.project=laser \
    wandb.append_timestamp=false \
    "wandb.id=ffhqa2048k4rqvae$STAMP" \
    wandb.resume=allow \
    "wandb.name=ffhq-a2048-k4-rqvae-stage1-$STAMP" \
    "wandb.group=ffhq-a2048-k4-rqvae-full-$STAMP" \
    'wandb.tags=[stage1,ffhq,rqvae_config,dictionary,a2048,k4,8x8x4,lpips1,gan,patchgan2,effective_batch128]' \
    "wandb.save_dir=$STAGE1_OUT/wandb" \
    "output_dir=$STAGE1_OUT" \
    "hydra.run.dir=$STAGE1_OUT/hydra" \
    "${resume_args[@]}" \
    2>&1 | tee -a "$RUN_ROOT/logs/stage1.log"
  stage1_checkpoint="$(latest_checkpoint "$STAGE1_OUT" final.ckpt)"
  if [[ -z "$stage1_checkpoint" ]]; then
    echo "Stage 1 completed without final.ckpt" >&2
    exit 1
  fi
  touch "$STAGE1_OUT/.phase_complete"
  status "$active_phase" complete "checkpoint=$stage1_checkpoint"
fi
if [[ -z "$stage1_checkpoint" ]]; then
  echo "Stage-1 completion marker exists but final.ckpt is missing" >&2
  exit 1
fi

active_phase=rfid
rfid_valid=0
if [[ -f "$RFID_RESULT" ]]; then
  rfid_valid="$($PYTHON_BIN -c 'import json,sys; p=json.load(open(sys.argv[1])); print(int(p.get("dataset") == "ffhq" and p.get("num_images") == 70000 and isinstance(p.get("rfid"), (int, float))))' "$RFID_RESULT")"
fi
if [[ "$rfid_valid" -ne 1 ]]; then
  status "$active_phase" starting "native full-70000 reconstruction FID"
  torchrun --standalone --nproc_per_node="$NPROC" \
    "$ROOT/scripts/evaluate_upstream_laser_rfid.py" \
    --checkpoint "$stage1_checkpoint" \
    --data "$DATA_ROOT" \
    --output "$RFID_RESULT" \
    --dataset ffhq \
    --num-images 70000 \
    --num-atoms 2048 \
    --sparsity-level 4 \
    --coeff-vocab-size 1024 \
    --batch-size 4 \
    --backend native \
    2>&1 | tee -a "$RUN_ROOT/logs/rfid.log"
fi
$PYTHON_BIN -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["dataset"] == "ffhq" and p["num_images"] == 70000 and isinstance(p["rfid"], (int, float)), p' "$RFID_RESULT"
status "$active_phase" complete "result=$RFID_RESULT"

active_phase=token_cache
cache_valid=0
if [[ -f "$TOKEN_CACHE" && -f "$TOKEN_REPORT" ]]; then
  cache_valid="$($PYTHON_BIN -c 'import json,sys; p=json.load(open(sys.argv[1])); print(int(p.get("passed") is True and p.get("items") == 70000 and p.get("compound_sequence_length") == 256))' "$TOKEN_REPORT")"
fi
if [[ "$cache_valid" -ne 1 ]]; then
  status "$active_phase" starting "full-70000 compound a2048/k4 cache"
  torchrun --standalone --nproc_per_node="$NPROC" \
    "$ROOT/scripts/tools/build_official_imagenet_token_cache.py" \
    --checkpoint "$stage1_checkpoint" \
    --data "$DATA_ROOT" \
    --output "$TOKEN_CACHE" \
    --dataset ffhq \
    --batch-size 16 \
    --num-workers "$NUM_WORKERS" \
    --num-atoms 2048 \
    --sparsity-level 4 \
    --coeff-vocab-size 1024 \
    --coeff-max 3 \
    --auto-coeff-scales-percentile 100 \
    --verify-samples 64 \
    --compound \
    2>&1 | tee -a "$RUN_ROOT/logs/token_cache.log"
fi
$PYTHON_BIN -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["passed"] and p["items"] == 70000 and p["compound_sequence_length"] == 256, p' "$TOKEN_REPORT"
status "$active_phase" complete "cache=$TOKEN_CACHE"

active_phase=stage2
status "$active_phase" starting "official FFHQ RQ-Transformer 350M; 8x8x4 compound events"
printf '%s\n' "$$" > "$STAGE2_OUT/launcher.pid"
exec torchrun --standalone --nproc_per_node="$NPROC" \
  "$ROOT/scripts/train_official_rqtransformer_laser_stage2.py" \
  --checkpoint "$stage1_checkpoint" \
  --data "$DATA_ROOT" \
  --dataset ffhq \
  --model-preset ffhq-350m \
  --token-cache "$TOKEN_CACHE" \
  --output "$STAGE2_OUT" \
  --distributed-backend ddp \
  --epochs 200 \
  --batch-size "$STAGE2_MICROBATCH" \
  --total-batch-size 128 \
  --num-atoms 2048 \
  --sparsity-level 4 \
  --coeff-vocab-size 1024 \
  --coeff-max 3 \
  --compound-tokens \
  --compound-micro-transformer-layers 2 \
  --compound-depth-specific-coeff-heads \
  --compound-distribution-geometry \
  --geometry-top-k 4 \
  --atom-loss-weight 1.5 \
  --geometry-loss-weight 0.05 \
  --geometry-start-epoch 2 \
  --geometry-warmup-epochs 3 \
  --lr 0.0005 \
  --lr-schedule constant \
  --atom-temperature 1.0 \
  --atom-top-k 250 \
  --atom-top-p 1.0 \
  --coeff-temperature 1.0 \
  --coeff-top-k 250 \
  --coeff-top-p 1.0 \
  --fid-num-samples 50000 \
  --fid-batch-size "$STAGE2_FID_BATCH" \
  --fid-every 5 \
  --save-ckpt-freq 10 \
  --save-step-freq 250 \
  --sample-grid-every 1000 \
  --sample-grid-size 64 \
  --sample-grid-batch-size "$STAGE2_FID_BATCH" \
  --sample-grid-sweep \
  --sample-grid-on-start \
  --upload-token-cache \
  --resume \
  --wandb-entity helloimlixin-rutgers \
  --wandb-project laser \
  --wandb-id "ffhqcompoundrqt350a2048k4$STAMP" \
  --wandb-name "ffhq-a2048-k4-compound-official-rqtransformer-350M-$STAMP" \
  2>&1 | tee -a "$RUN_ROOT/logs/stage2.log"
