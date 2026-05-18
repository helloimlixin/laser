# Recent Conversion Summary

Date: 2026-04-10

This note records the recent conversion from the weaker maintained stage-1 setup to the current stronger VQGAN-style CelebA-HQ `256 x 256` launch path, plus the current run state after the launcher fixes.

## What Changed

### 1. Stronger stage-1 backbone

The maintained LASER stage-1 path was upgraded from a weak capped-width VQGAN/DDPM-style backbone to an explicit per-level width schedule.

Key code changes:

- `src/models/laser.py`
  - added `channel_multipliers`
  - VQGAN-style encoder/decoder now use the explicit width schedule when provided
  - `num_downsamples` is derived from the schedule length when that schedule is explicit
- `train.py`
  - forwards `channel_multipliers` into the maintained LASER model
- `extract_token_cache.py`
  - stores `channel_multipliers` in cache metadata
- `src/stage2_compat.py`
  - reconstructs the stage-1 backbone using the stored or inferred channel schedule
- `README.md`
  - documents the stronger backbone path and the explicit width schedule knob

The main practical effect is that we can keep high-resolution stages affordable while widening the deeper `16 x 16` and `8 x 8` blocks where reconstruction quality and global structure actually depend on capacity.

### 2. Stronger launch defaults

The current CelebA-HQ sweep script is:

- `scripts/sweep_celebahq256_vqgan_cap.sh`

Important launch defaults now include:

- `3` GPUs per job when requested
- explicit `channel_multipliers`
- VQGAN/DDPM-style attention
- dense logging and visualization
- stage-2 sampling every epoch

Current stronger schedules:

- `256 -> 8 x 8`: `channel_multipliers=[1,1,2,2,4,4]`
- `256 -> 16 x 16`: `channel_multipliers=[1,1,2,4,4]`

### 3. Launcher reliability fixes

The initial multi-GPU conversion exposed several launcher issues:

- missing `configs/wandb/default.yaml` in some snapshots
- invalid `model.resolution` Hydra override in the first `256 x 256` sweep
- fragile `sbatch --wrap` quoting for large containerized jobs
- `cgpu` behaving unreliably for shared logs and scratch-visible outputs

The maintained sweep script now:

- prefers `python3`
- makes `nvidia-smi` non-fatal
- supports `apptainer` as well as `singularity`
- submits a real per-case batch script instead of a giant inline `--wrap`

## Resource Decision

We moved away from the earlier `1`-GPU memory-constrained runs and relaunched on `gpu-redhat` with:

- `3` GPUs per job
- `CPUS=12`
- `MEM_MB=128000`
- `TIME_LIMIT=72:00:00`

`cgpu` looked attractive because it was idle, but it proved unreliable for the current sweep path. The stable partition for these launches is currently `gpu-redhat`.

## Embedding-Dim Shift

The patch runs were initially too narrow:

- many patch cases were launched with `embedding_dim=4`

That is likely too low for CelebA-HQ `256 x 256`, especially once the encoder/decoder got stronger.

The current prioritized patch embedding-dim comparison is:

- `d=4` as the direct control
- `d=8` as the main likely sweet spot
- `d=16` as the higher-cost upper bound

Relevant patch cases:

- `p4s4_k16_d4k`
- `p4s4_k16_d4k_e8`
- `p4s4_k16_d4k_e16`
- `p2s2_k16_d4k_e8`

## Current Snapshots and Run Roots

Working snapshot for the current stable multi-GPU path:

- `/cache/home/xl598/submission_snapshots/laser_chq256_vqgan_strong3gpu_sbatchscript_20260410_021834`

Current run roots:

- `8 x 8` smoke: `/scratch/xl598/runs/celebahq256_vqgan8x8_strong3gpu_sbatchscript_smoke_redhat_20260410_021834`
- `8 x 8` main low-`d` patch set: `/scratch/xl598/runs/celebahq256_vqgan8x8_strong3gpu_sbatchscript_redhat_20260410_021834`
- `8 x 8` higher-`d` patch set: `/scratch/xl598/runs/celebahq256_vqgan8x8_strong3gpu_embdim_20260410_021834`
- `16 x 16` higher-`d` smoke: `/scratch/xl598/runs/celebahq256_vqgan16x16_strong3gpu_embdim_smoke_20260410_021834`

## Current Job State

As of this note, the most important jobs are:

Running:

- `50938053`: `8 x 8` `p4s4_k16_d4k` on `gpu-redhat`
- `50938111`: `8 x 8` `p4s4_k16_d4k_e8` on `gpu-redhat`
- `50938112`: `8 x 8` `p4s4_k16_d4k_e16` on `gpu-redhat`

Pending:

- `50938113`: `8 x 8` `p2s2_k16_d4k_e8`
- `50938115`: `16 x 16` smoke `p2s2_k16_d4k_e8`

Lower-priority pending nonpatch and low-value low-`d` jobs were cancelled to free slots for the `e8` and `e16` runs.

## Practical Read

The current conversion has three main outcomes:

1. The maintained stage-1 model is no longer bottlenecked by an overly weak encoder/decoder schedule.
2. The multi-GPU launch path is now stable on `gpu-redhat`.
3. The next useful decision is whether patch runs should standardize on `embedding_dim=8` instead of `4`.

If the current `p4s4` `e8` and `e16` jobs train cleanly, the likely next move is:

- promote `d=8` to the default patch embedding dim
- keep `d=4` only as a comparison baseline
- expand the `16 x 16` higher-`d` patch sweep after the smoke run confirms stability
