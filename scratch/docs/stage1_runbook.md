# Stage1 Launch Guide

This page describes the short launchers that are meant to be used directly. The longer `launch_*` scripts are still available, but they are treated as implementation details unless you are changing cluster behavior.

## Supported Entry Points

Run from `/cache/home/xl598/Projects/laser/scratch`.

```bash
./scripts/pack128.sh
./scripts/fast100.sh
./scripts/patch100.sh
./scripts/ref.sh
```

- `pack128.sh` submits the CelebA packing job that writes `celeba_128x128_rgb_uint8.npy` and `.json`.
- `fast100.sh` launches the current fast baseline: `laser.py`, 100 epochs for both stages, 3 GPUs, lightweight AE, no patch bottleneck.
- `patch100.sh` launches the same 100/100 baseline with overlapping patch dictionary learning enabled.
- `ref.sh` launches the historical `415ephb2`-style small-cluster reference recipe.

## Baseline Settings

`fast100.sh` and `patch100.sh` share the same training defaults:

- `stage1_epochs=100`
- `stage2_epochs=100`
- `stage1_lr=2e-4`
- `stage2_lr=1e-3`
- `batch_size=32`
- `stage2_batch_size=32`
- `num_workers=12`
- `token_num_workers=4`
- `token_subset=98304`
- `num_hiddens=64`
- `num_res_hiddens=32`
- `num_res_layers=1`
- `wandb_project=laser-dl`

`fast100.sh` adds:

- `ae_num_downsamples=4`
- `patch_based=false`
- `stage2_coeff_loss_type=gt_atom_recon_mse`

`patch100.sh` adds:

- `ae_num_downsamples=2`
- `num_atoms=3072`
- `sparsity_level=16`
- `patch_based=true`
- `patch_size=8`
- `patch_stride=4`
- `patch_reconstruction=hann`
- `quantize_sparse_coeffs=false`
- `coef_max=8.0`
- `stage1_warmup_epochs=1`
- `stage1_dict_lr_multiplier=1.0`
- `stage1_dict_warmup_epochs=0`
- `stage1_dict_grad_clip=1.0`
- `commitment_cost=1.0`
- `bottleneck_weight=0.05`
- `stage1_bottleneck_weight_start=0.25`
- `stage1_bottleneck_warmup_epochs=1`
- `stage1_dict_optimizer=shared_adam`
- `stage1_dict_loss_weight=0.01`
- `stage1_dict_loss_weight_start=0.05`
- `stage1_dict_loss_warmup_epochs=12`
- `stage1_commitment_loss_weight=0.10`
- `stage1_commitment_loss_weight_start=0.25`
- `stage1_commitment_loss_warmup_epochs=12`
- `stage1_coherence_weight=1.0`
- `stage1_coherence_weight_start=0.0`
- `stage1_coherence_warmup_epochs=12`
- `stage1_coherence_margin=0.0`
- `stage2_coeff_loss_type=mse`

## Dataset Prep

The launchers default to:

```text
DATA_DIR=/cache/home/xl598/Projects/data/celeba
```

For the fast path, that directory needs the packed files:

```text
celeba_128x128_rgb_uint8.npy
celeba_128x128_rgb_uint8.json
```

Create them with:

```bash
./scripts/pack128.sh
```

The pack job writes to:

```text
/scratch/$USER/datasets/celeba_packed_128
```

If you keep the packed files somewhere else, either point `DATA_DIR` at a directory that contains them or copy the files into `/cache/home/xl598/Projects/data/celeba`.

Use real files in `DATA_DIR` for the most reliable behavior. Symlinks to packed files have been inconsistent across partitions.

## Launch

Fast baseline:

```bash
./scripts/fast100.sh
```

Patch baseline:

```bash
./scripts/patch100.sh
```

Historical reference:

```bash
./scripts/ref.sh
```

You can still override launcher settings with environment variables. Common examples:

```bash
PARTITION=gpu ./scripts/fast100.sh
PARTITION=cgpu ./scripts/patch100.sh
WANDB_NAME=fast100-v2 OUT_DIR=/scratch/$USER/runs/laser_fast100_v2 ./scripts/fast100.sh
```

## Check Resources

Before submitting:

```bash
sinfo -s
```

After submitting:

```bash
squeue -u $USER
```

To inspect one job:

```bash
scontrol show job <jobid>
sacct -j <jobid> --format=JobID,JobName,Partition,State,ExitCode,Elapsed,NodeList -P
```

## Early Log Checks

Good startup lines:

- `[Data] using packed CelebA dataset ...`
- `[Startup] dataloaders ready`
- `[Startup] entering stage1 train loop ...`

Bad startup lines:

- `packed CelebA dataset not found ... falling back to raw image tree`
- `FileNotFoundError: Dataset directory not found`
- `RuntimeError: Detected mismatch between collectives on ranks`

Watch logs with:

```bash
tail -f /home/xl598/Projects/laser/scratch/fast100_<jobid>.out
tail -f /home/xl598/Projects/laser/scratch/patch100_<jobid>.out
```

## Advanced Scripts

Use these only when you need to change the actual launch mechanism:

- `scripts/launch_proto_rqsd_multinode_slurm.sh`
- `scripts/launch_var_rqsd_multinode_slurm.sh`
- `scripts/run_var_rqsd_multinode_job.sh`
- `scripts/launch_stage1_balanced.sh`
- `scripts/launch_stage1_nonquantized.sh`
