# Stage1 and VAR Multinode Runbook

## Current Recommended Launch

Use these saved launchers:

```bash
/cache/home/xl598/Projects/laser/scratch/scripts/launch_stage1_balanced.sh
/cache/home/xl598/Projects/laser/scratch/scripts/launch_stage1_nonquantized.sh
```

`launch_stage1_balanced.sh` is the conservative reference-style quantized baseline.

`launch_stage1_nonquantized.sh` launches the matching nonquantized sparse-coefficient comparison run.


## VAR Stage2 Launch

Use this launcher for the sparsity-stage VAR path:

```bash
/cache/home/xl598/Projects/laser/scratch/scripts/launch_var_rqsd_multinode_slurm.sh
```

It boots a stage1 run through `proto.py`, parses the resulting run directory, then launches `var_stage2.py` from `stage1/ae_best.pt`.

The corresponding Slurm payload script is:

```bash
/cache/home/xl598/Projects/laser/scratch/scripts/run_var_rqsd_multinode_job.sh
```

## Current Recommended Configuration

### Cluster layout

- Partition: `gpu-redhat`
- Nodes: `8`
- GPUs per node: `3`
- Total GPUs: `24`
- CPUs per task: `8`
- Memory: `500000` MB
- Time limit: `3-00:00:00`

### Training schedule

- `stage1_epochs=100`
- `stage2_epochs=100`
- `stage1_lr=2e-4`
- `stage2_lr=1e-3`
- `stage1_lr_schedule=cosine`
- `stage1_warmup_epochs=2`
- `stage1_min_lr_ratio=0.1`
- `stage1_dict_optimizer=separate_sgd`
- `stage1_dict_lr_multiplier=0.25`
- `stage1_dict_lr_schedule=cosine`
- `stage1_dict_warmup_epochs=2`
- `stage1_dict_min_lr_ratio=0.05`
- `stage1_dict_grad_clip=0.05`
- `stage1_dict_max_update_norm=0.0`
- `stage1_loss_spike_skip_ratio=0.0`
- `stage1_loss_ema_beta=0.98`
- `stage1_bottleneck_weight_start=1.0`
- `stage1_bottleneck_warmup_epochs=0`
- `grad_clip=1.0`
- `stage2_warmup_steps=500`
- `stage2_min_lr_ratio=0.01`
- `stage2_weight_decay=0.01`
- `batch_size=32`
- `stage2_batch_size=32`
- `num_workers=12`
- `token_num_workers=0`

## Why This Setup

This configuration is the current recommended conservative path after the faster high-LR variants proved worse than the stronger historical run.

### What failed

- `stage1_lr=1.92e-2` with large batch: numerically finite but optimization was clearly unstable. The bottleneck term `b` exploded early.
- `batch_size=8`: gave more steps per epoch but made the run about 4x slower wall-clock than the original `batch_size=32` setup.
- Early versions of the new stage1 skip and rollback guards were rank-local and caused a DDP collective mismatch crash.

### What was fixed in code

The current codebase already contains these fixes:

### Quantization control

The shared multinode launcher now supports explicit sparse-coefficient mode selection through `QUANTIZE_SPARSE_COEFFS=true|false`.

- `launch_stage1_balanced.sh` keeps `QUANTIZE_SPARSE_COEFFS=true` by default
- `launch_stage1_nonquantized.sh` sets `QUANTIZE_SPARSE_COEFFS=false`

- Extra numerical sanitization in stage1 and bottleneck paths.
- Dictionary-specific LR scheduling in the default `shared_adam` path.
- Separate dictionary gradient clipping and tangent-space projection.
- Loss-spike skip guard.
- Dictionary update rollback guard.
- DDP-safe synchronization of skip and rollback decisions across ranks.
- Cleaner multinode launcher behavior on `gpu-redhat`.

## Important Files

- Main trainer: `/cache/home/xl598/Projects/laser/scratch/proto.py`
- Lightning path: `/cache/home/xl598/Projects/laser/scratch/proto_lightning.py`
- Shared multinode launcher: `/cache/home/xl598/Projects/laser/scratch/scripts/launch_proto_rqsd_multinode_slurm.sh`
- Saved quantized launcher: `/cache/home/xl598/Projects/laser/scratch/scripts/launch_stage1_balanced.sh`
- Saved nonquantized launcher: `/cache/home/xl598/Projects/laser/scratch/scripts/launch_stage1_nonquantized.sh`

- VAR model: `/cache/home/xl598/Projects/laser/scratch/var.py`
- VAR stage2 trainer: `/cache/home/xl598/Projects/laser/scratch/var_stage2.py`
- VAR launch wrapper: `/cache/home/xl598/Projects/laser/scratch/scripts/launch_var_rqsd_multinode_slurm.sh`
- VAR Slurm payload: `/cache/home/xl598/Projects/laser/scratch/scripts/run_var_rqsd_multinode_job.sh`



## VAR Stage2 Workflow

### Quick debug example

This is the current small-cluster debug shape for the VAR path:

```bash
cd /cache/home/xl598/Projects/laser/scratch
JOB_NAME=laser-var-debug5-3g \
PARTITION=gpu-redhat \
NODES=1 \
GPUS_PER_NODE=3 \
CPUS_PER_TASK=4 \
MEM_MB=128000 \
TIME_LIMIT=03:00:00 \
OUT_DIR=/scratch/$USER/runs/laser_var_debug5_3g_celeba128_quantized \
STAGE1_EPOCHS=5 \
STAGE2_EPOCHS=5 \
BATCH_SIZE=16 \
STAGE2_BATCH_SIZE=16 \
NUM_WORKERS=4 \
TOKEN_NUM_WORKERS=2 \
TOKEN_SUBSET=128 \
STAGE2_SAMPLE_EVERY_STEPS=0 \
WANDB_NAME=laser_var_debug5_3g_bs16 \
LOG_PREFIX=laser_var_debug5_3g \
./scripts/launch_var_rqsd_multinode_slurm.sh
```

### Behavior differences that matter

- The launcher always runs stage1 first with `proto.py`, then runs `var_stage2.py` from the resulting `run_dir`.
- `var_stage2.py` disables sparse-slot canonicalization so the VAR stages follow greedy OMP slot order.
- VAR outputs land under `<run_dir>/stage2_var/`.
- The stage2 token cache is stored at `<run_dir>/stage2_var/tokens_cache_greedy.pt`.
- `TOKEN_SUBSET` only limits stage2 token-cache precompute and stage2 training items. It does not shrink stage1; stage1 still sees the full training split.
- W&B creates separate runs named `${WANDB_NAME}_stage1` and `${WANDB_NAME}_var`.

### Monitoring

Replace `<jobid>` with the value returned by `sbatch`.

```bash
squeue -j <jobid>
tail -f /cache/home/xl598/Projects/laser/scratch/laser_var_debug5_3g_<jobid>.out
tail -f /cache/home/xl598/Projects/laser/scratch/laser_var_debug5_3g_<jobid>.err
```

## How To Start Next Time

### 1. Launch the saved setup

Quantized baseline:

```bash
cd /cache/home/xl598/Projects/laser/scratch
./scripts/launch_stage1_balanced.sh
```

Nonquantized comparison:

```bash
cd /cache/home/xl598/Projects/laser/scratch
./scripts/launch_stage1_nonquantized.sh
```

### 2. Check Slurm status

Replace `<jobid>` with the job printed by `sbatch`.

```bash
squeue -j <jobid> -o '%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R'
sacct -j <jobid> --format=JobID,JobName,Partition,State,ExitCode,Elapsed,NodeList -P
```

### 3. Check logs

```bash
tail -n 200 /cache/home/xl598/Projects/laser/scratch/proto_rqsd_rh_24g_mn_<jobid>.out
tail -n 200 /cache/home/xl598/Projects/laser/scratch/proto_rqsd_rh_24g_mn_<jobid>.err
```

### 4. Verify W&B

Look for lines like:

- `wandb: Syncing run ...`
- `wandb: View run at https://wandb.ai/...`

The API key is expected at:

```text
/scratch/$USER/.secrets/wandb_api_key
```

## What To Look For Early In Stage1

The first useful checks are:

- Did NCCL initialize on all 24 ranks?
- Did W&B initialize successfully?
- Did stage1 actually enter the train loop?
- Did the run avoid DDP collective mismatch errors?
- Did the new guards print warnings?

### Good signs

- No `RuntimeError: Detected mismatch between collectives on ranks`
- No `non-finite loss` warnings
- No `non-finite gradients` warnings
- No immediate huge jump in bottleneck loss `b`

### Bad signs

- Any DDP collective mismatch
- `b` jumping very early into multi-unit or double-digit territory
- Repeated dictionary rollback warnings
- Repeated loss-spike skip warnings

## Monitoring Checklist

Use these checkpoints:

### After startup

Confirm:

- NCCL init complete on all ranks
- dataloaders ready
- stage1 train loop entered

### After 50-100 stage1 steps

Check:

- rough throughput
- whether `b` is settling or exploding
- whether guard warnings are firing

### End of epoch 1

Check:

- train loss trend
- validation loss
- PSNR
- SSIM

### Epochs 2-3

Check:

- whether finite spikes reappear later
- whether `b` grows suddenly when optimizer state accumulates

## Recent Run History

| Job ID | Key config | Outcome | Notes |
| --- | --- | --- | --- |
| `50378721` | `stage1_lr=8e-3`, `batch_size=32` | Unstable | Early stage1 was noticeably noisier and bottleneck loss stayed elevated. |
| `50378729` | `stage1_lr=1.92e-2`, `batch_size=32` | Failed by instability | Bottleneck term exploded early; nominal 4x LR was not viable. |
| `50378738` | `stage1_lr=1.92e-2`, bottleneck ramp, `batch_size=32` | Failed by instability | Bottleneck ramp reduced scalar loss contribution but did not stop underlying sparse-code blow-up. |
| `50378745` | `stage1_lr=1.2e-2`, `batch_size=8`, long warmup | Replaced | Much slower wall-clock because epoch length increased about 4x. |
| `50378746` | `stage1_lr=1.2e-2`, `batch_size=8`, shorter warmup | Failed by DDP mismatch | Training looked better, but rank-local guard decisions caused collective mismatch. |
| `50378763` | Same recipe as `50378746` with DDP-safe guard fix | Replaced | Relaunched to verify synchronized skip/rollback logic. |
| `50378775` | `stage1_lr=1.0e-2`, `batch_size=16` | Unstable later | Improved speed-stability tradeoff versus `bs=8`, but later triggered repeated spike skips and dictionary rollbacks. |
| `50379248` | `stage1_lr=8.0e-3`, `batch_size=32` | Replaced | Early stage1 looked calmer than the `bs=16`, `1e-2` run, but this was still a fast-path experiment rather than the proven conservative regime. |
| `50379271` | `stage1_lr=2e-4`, `stage1_dict_optimizer=separate_sgd`, `batch_size=32`, quantized | Current quantized reference run | Launched to match the stronger historical `dict_safe` recipe as closely as possible on 24 GPUs. |
| `50379293` | `stage1_lr=2e-4`, `stage1_dict_optimizer=separate_sgd`, `batch_size=32`, nonquantized | Current nonquantized comparison run | Same conservative recipe, reshaped to `4 x 3` on `gpu-redhat` for a faster start. |

## Suggested Decision Rules

### If the run is stable but too slow

Prefer this order:

1. Keep `batch_size=32` and the conservative stage1 recipe as the default
2. If you want to speed up later, change one axis at a time
3. Prefer changing optimizer mode or LR only after you have a clean baseline

Do not jump back to high-LR fast-path settings unless you explicitly want to re-enter that tradeoff.

### If the run is unstable again

Prefer this order:

1. First confirm whether the conservative reference-style recipe is still underperforming
2. If it is, compare exact metrics and topology against the historical run before tuning further
3. Only then adjust one variable at a time
4. Start with topology or global-batch differences before touching LR

### If throughput is good and stage1 is stable

Keep this setup and let it run through epoch 1 and epoch 2 before changing anything else.

## Commands Worth Reusing

### Launch

Quantized baseline:

```bash
./scripts/launch_stage1_balanced.sh
```

Nonquantized comparison:

```bash
./scripts/launch_stage1_nonquantized.sh
```


VAR stage2 path:

```bash
./scripts/launch_var_rqsd_multinode_slurm.sh
```

### Cancel

```bash
scancel <jobid>
```

### Watch logs

```bash
tail -f /cache/home/xl598/Projects/laser/scratch/proto_rqsd_rh_24g_mn_<jobid>.err
tail -f /cache/home/xl598/Projects/laser/scratch/proto_rqsd_rh_24g_mn_<jobid>.out
```

## Current Active Jobs At Time Of Writing

Quantized reference run:

- Job ID: `50379271`
- W&B run: `eh94j99b`
- W&B URL: `https://wandb.ai/helloimlixin-rutgers/laser-scratch/runs/eh94j99b`

Nonquantized comparison run:

- Job ID: `50379293`
- W&B run: `l0vius3u`
- W&B URL: `https://wandb.ai/helloimlixin-rutgers/laser-scratch/runs/l0vius3u`

Treat this as a snapshot, not a permanent truth. Next time you resume, first confirm whether these jobs are still running and whether their early stage1 behavior stayed healthy.
