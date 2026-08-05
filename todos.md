# ImageNet compound RQ-Transformer lab migration runbook

Updated: 2026-08-04 UTC

## Active local pivot: CelebA-HQ compound RQ-Transformer

- W&B run: `helloimlixin-rutgers/laser/celebahq-compound-rqt350-a2048k2-20260804`
- Run name: `celebahq-a2048-k2-compound-v5b-official-rqtransformer-350M`
- State: running on 2 x RTX 4000 Ada (20 GB)
- Launcher: `scripts/launch_celebahq_a2048_k2_compound_official_stage2.sh`
- Output: `outputs/celebahq-a2048-k2-rqvae-strict-20260720-145706/stage2-compound-v5b-official-rqtransformer-350M`
- Log: `runs/celebahq_a2048_k2_compound_production.log`
- Stage-1 source: `helloimlixin-rutgers/laser/celebahq-a2048-k2-rqvae-strict-20260720-145706`
- Stage-1 checkpoint: best rFID slot 1, epoch 150, rFID 18.4206; downloaded
  checkpoint MD5 `7b098c140b8b653ca263d9d12f9cbb22` matches W&B.
- Dataset: `/home/xl598/Projects/data/celeba_hq` (28,000 train, 2,000 val).
- Token cache: 28,000 rows of 8x8x2 compound components, with per-depth
  coefficient scales `[45.83333206, 10.94791698]`; validation passed with exact
  atom agreement and maximum normalized coefficient error 0.00158.
- Token-cache artifact: `celebahq-compound-rqt350-a2048k2-20260804-token-cache:v0`
  (`latest`, `training-cache`).

The stage-2 body/head geometry and optimizer settings match KakaoBrain's
original FFHQ 350M configuration: embedding width 1024, 24 body layers, four
depth-head layers, 16 attention heads, AdamW at fixed LR 5e-4, weight decay
1e-4, betas (0.9, 0.95), global batch 128, 200 epochs, and top-k 250 sampling.
The compound v5b change uses two `(atom, coefficient)` events at every 8x8
location, a two-layer pair-local micro-transformer, depth-specific coefficient
heads, atom loss weight 1.5, and distribution-geometry weight 0.05 warmed from
epochs 2 through 5. The resulting model has 383,477,760 parameters.

Validation on the lab GPUs:

- One production-shaped DDP optimizer step passed at 7.55 GiB allocated and
  7.61 GiB reserved on both GPUs.
- Batch-8 compound autoregressive generation and stage-1 decode passed at
  12.06 GiB allocated and 12.36 GiB reserved on both GPUs.
- FID follows the original FFHQ protocol: 50,000 generated images against all
  training images, evaluated every 50 epochs.
- Preview sampling writes four separate, borderless 8x8 mosaics every 1,000
  optimizer steps. Each filename and W&B panel identifies exactly one sampling
  setting; the images themselves contain no title, subtitle, label, margin, or
  padding. The four settings compare original top-k 250, a colder top-k 250,
  atom/coefficient nucleus sampling, and a hybrid top-k/top-p setting.
- Every scheduled full checkpoint upload contains `last.pt`, all retained
  top-three FID checkpoints, and all retained top-three Inception Score
  checkpoints. Step-250 atomic recovery saves remain local so artifact upload
  does not stall training every few minutes.

## Current RunPod job

- W&B run: `helloimlixin-rutgers/laser/c5cos10r`
- Run name: `imagenet-rqtransformer-laser-compound-v5b-original-cosine-from-epoch10`
- State when migration work began: crashed after the four-H100 RunPod job stopped
- Launcher: `scripts/launch_compound_v5b_original_cosine_scratch.sh`
- Trainer: `scripts/train_official_rqtransformer_laser_stage2.py`
- Output: `outputs/swgbasnb_compound_v5b_original_cosine_from_epoch10/stage2`
- Recovery checkpoint: `outputs/swgbasnb_compound_v5b_original_cosine_from_epoch10/stage2/checkpoints/last.pt`
- Stage-1 checkpoint: `outputs/imagenet_x3h5cl0h_stage2/stage1_checkpoint/best_rfid_slot3_model.pt`
- Token cache: `outputs/swgbasnb_compound_pairs_from_scratch/token_cache/imagenet_train_compound_pairs.pt`
- ImageNet root: `/workspace/Projects/data/imagenet`

The job resumed at epoch 10/global step 6,250. The migration snapshot uploaded
while writing this runbook is the completed epoch-18 checkpoint at global step
11,250. It is approximately 17.4 GB and contains model, optimizer, scheduler,
epoch/global-step cursor, and best-metric history.

The final W&B summary reached global step 11,750 with LR
`0.0004576489747989561`, but no artifact newer than the step-11,250 immutable
snapshot was committed before the crash. A continuation from `:v1` must replay
those 500 unartifacted optimizer steps. The checkpointed LR at step 11,250 is
`0.00046108198137550626` and was reproduced exactly by the lab smoke tests.

Latest completed evaluation at the time of migration preparation:

```text
epoch 15: FID 31.9039, Inception Score 32.2789 +/- 0.5079
```

## Active RunPod settings

```yaml
hardware:
  gpus: 4
  gpu_type: H100 80 GB
  distributed_backend: DDP

model:
  architecture: compound-v4-micro2-rqtransformer-1400M
  parameter_count: 1_451_676_160
  num_atoms: 16384
  coeff_vocab_size: 2048
  coeff_max: 20
  coeff_scale: 6.4
  compound_tokens: true
  compound_micro_transformer_layers: 2
  compound_depth_specific_coeff_heads: true

objective:
  atom_loss_weight: 1.5
  compound_distribution_geometry: true
  geometry_top_k: 4
  geometry_loss_weight: 0.05
  geometry_start_epoch: 2
  geometry_warmup_epochs: 3

optimizer:
  type: fused AdamW
  lr: 0.0005
  weight_decay: 0.0001
  betas: [0.9, 0.95]
  lr_schedule: cosine
  lr_schedule_epochs: 100
  min_lr: 0
  total_batch_size: 2048

training:
  epochs: 100
  batch_size_per_gpu: 64
  gradient_accumulation: 8
  precision: bfloat16 autocast with FP32 parameters and optimizer
  save_ckpt_freq_epochs: 2
  save_step_freq: 250
  sample_grid_every_optimizer_steps: 500

evaluation:
  fid_every_epochs: 5
  fid_num_samples: 50000
  fid_batch_size_per_gpu: 96
  atom_temperature: 0.90
  atom_top_p: 0.92
  coeff_temperature: 1.00
  coeff_top_p: 0.85
```

The effective batch is unchanged by accumulation:

```text
4 GPUs * 64 samples/GPU * 8 microbatches = 2048 samples/update
```

## Required lab configuration: 2 x RTX 4000 Ada (20 GB each)

Do not launch the current DDP trainer unchanged on this machine. The checkpoint
contains 1.452B FP32 parameters (5.81 GB). Replicated FP32 parameters, gradients,
and two Adam moments have a theoretical lower bound of about 23.2 GB per GPU
before activations, CUDA workspaces, DDP buckets, or the frozen stage-1 model.
Reducing only `--batch-size` therefore cannot fit a 20 GB GPU.

Preserve the architecture, optimizer recipe, global batch, and cosine schedule,
but use two-way FSDP `FULL_SHARD` for parameters, gradients, and optimizer state.
The initial batch-4 profile passed training but FID-style generation reserved
19.15 GB. Per the fallback rule below, use this validated profile:

```yaml
hardware:
  gpus: 2
  gpu_type: RTX 4000 Ada 20 GB
  distributed_backend: FSDP FULL_SHARD

training:
  batch_size_per_gpu: 2
  total_batch_size: 2048
  gradient_accumulation: 512
  precision: bfloat16 autocast
  sample_grid_every_optimizer_steps: 0

evaluation:
  fid_batch_size_per_gpu: 2
  fid_num_samples: 50000
  fid_every_epochs: 5
```

```text
2 GPUs * 2 samples/GPU * 512 microbatches = 2048 samples/update
```

Keep `--lr 0.0005`, `--lr-schedule cosine`, `--lr-schedule-epochs 100`, and
`--min-lr 0`. Keeping the same global batch and optimizer-step count preserves
the intended schedule. Disable the 64-image training preview initially because
its generation batch is independent of `--batch-size`; re-enable it only after
adding a configurable, tested preview batch size.

The production launcher implementing this command is
`scripts/launch_compound_v5b_lab_fsdp.sh`:

```bash
torchrun --standalone --nproc_per_node=2 \
  scripts/train_official_rqtransformer_laser_stage2.py \
  --distributed-backend fsdp \
  --checkpoint outputs/imagenet_x3h5cl0h_stage2/stage1_checkpoint/best_rfid_slot3_model.pt \
  --token-cache outputs/swgbasnb_compound_pairs_from_scratch/token_cache/imagenet_train_compound_pairs.pt \
  --compound-tokens \
  --compound-micro-transformer-layers 2 \
  --compound-depth-specific-coeff-heads \
  --compound-distribution-geometry \
  --geometry-top-k 4 \
  --atom-loss-weight 1.5 \
  --geometry-loss-weight 0.05 \
  --geometry-start-epoch 2 \
  --geometry-warmup-epochs 3 \
  --data /path/to/imagenet \
  --output outputs/swgbasnb_compound_v5b_original_cosine_from_epoch10/stage2 \
  --checkpoint-dir outputs/swgbasnb_compound_v5b_original_cosine_from_epoch10/stage2/checkpoints \
  --epochs 100 \
  --batch-size 2 \
  --total-batch-size 2048 \
  --num-atoms 16384 \
  --coeff-vocab-size 2048 \
  --coeff-max 20 \
  --coeff-scale 6.4 \
  --lr 0.0005 \
  --lr-schedule cosine \
  --lr-schedule-epochs 100 \
  --min-lr 0 \
  --fid-num-samples 50000 \
  --fid-batch-size 2 \
  --fid-every 5 \
  --save-ckpt-freq 2 \
  --save-step-freq 250 \
  --sample-grid-every 0 \
  --atom-temperature 0.90 \
  --atom-top-p 0.92 \
  --coeff-temperature 1.00 \
  --coeff-top-p 0.85 \
  --upload-checkpoints \
  --resume \
  --resume-checkpoint outputs/swgbasnb_compound_v5b_original_cosine_from_epoch10/stage2/checkpoints/last.pt \
  --wandb-id c5cos10r \
  --wandb-name imagenet-rqtransformer-laser-compound-v5b-original-cosine-from-epoch10
```

The command is now runnable once ImageNet `val/` is present on the lab host.

## Lab migration implementation and validation

- Added transformer-block-granularity FSDP `FULL_SHARD`; the frozen stage-1
  auxiliary model remains unwrapped.
- Added FSDP-safe clipping, accumulation, cached sampling, rank-zero full
  checkpoint save, and DDP/FSDP optimizer-state conversion. FSDP deliberately
  reduce-scatters every microbatch because `no_sync()` retains full gradients.
- FID sampling temporarily offloads sharded Adam state to 125 GB host RAM while
  full parameters are summoned, then restores the optimizer state to each GPU.
- Downloaded and manifest-verified all three immutable migration files, then
  hard-linked them into the exact paths listed above.
- Unit tests: `10 passed` for the focused scheduler/checkpoint/objective suite.
- Two-rank toy test: legacy DDP optimizer load, FSDP update, rank-zero full save,
  and ordinary AdamW reload all passed.
- Full 1.452B smoke measurements (allocated/reserved on both ranks):
  - training batch 1: `11.75/18.38 GiB`
  - training batch 4: `11.81/18.20 GiB`
  - training batch 2: `11.77/18.60 GiB`
  - generation batch 4 before optimizer offload: `18.23/19.15 GiB`
  - generation batch 2 after optimizer offload: `16.81/17.07 GiB`
- A full 17.42 GB FSDP recovery save at step 11,251 was loaded successfully by
  an ordinary unwrapped model and AdamW with all 829 states. The smoke-only
  checkpoint was deleted after validation; the migration checkpoint is intact.
- Smoke logs are under `runs/c5cos10r_fsdp_smoke*.log`.
- Current blocker (2026-08-04): no ImageNet tree exists at
  `/home/xl598/Projects/data/imagenet` or the usual lab mount locations. The
  production launcher has not been started because epoch-20 FID cannot run
  without ImageNet validation images. Standard ImageNet access/copy/mount is
  required; the token cache means training images are not read during stage 2,
  but the validation set is still required for the established FID protocol.

## FSDP implementation checklist (completed)

- [x] Add FSDP `FULL_SHARD` wrapping at transformer-block granularity. Do not wrap
  the frozen stage-1 auxiliary model.
- [x] Update `unwrap()`, gradient-accumulation `no_sync()`, gradient clipping, and
  rank-zero save paths for FSDP.
- [x] Load the existing full DDP model and AdamW state into FSDP on resume. Preserve
  `epoch`, `batch_idx`, `global_step`, scheduler state, and best-metric history.
- [x] Save a full, CPU-offloaded, rank-zero-compatible checkpoint so future DDP/FSDP
  launches can read the same format. Avoid materializing it on every rank.
- [x] Run a two-GPU smoke test with batch 1, then batch 4. Record peak allocated and
  reserved memory on both GPUs. Fall back to batch 2 and accumulation 512 if
  either rank approaches 19 GB or FID generation OOMs.
- [x] Verify the lab host has enough RAM and at least 55 GB of free local disk for
  one 17.4 GB checkpoint, one atomic-save temporary, and W&B staging.
- [x] Test native NCCL over PCIe first. Set `NCCL_P2P_DISABLE=1` only if the host's
  peer mappings or IOMMU configuration cause NCCL failures.

Expect a large throughput reduction: the lab job performs 32 times as many
training microbatches per optimizer step as the active RunPod job, before the
additional H100-versus-RTX performance difference.

## W&B migration artifacts

- Existing evaluated checkpoint artifact: `c5cos10r-checkpoint:v0`
  (`best`, `epoch-15`), containing the epoch-15 recovery/best checkpoints and
  inherited best checkpoints from the source run.
- Migration checkpoint artifact: `c5cos10r-checkpoint:v1` (committed, 18.68 GB),
  containing `stage2/last.pt` from epoch 18/global step 11,250 and
  `stage1/best_rfid_slot3_model.pt`. It has aliases `migration-epoch-18` and
  `migration-step-11250`.
- Token-cache artifact: `c5cos10r-token-cache:v0` (committed, 658.5 MB),
  containing `imagenet_train_compound_pairs.pt`. It has alias `migration`.

Download the immutable `:v1`/`:v0` versions or their `migration-*` aliases,
not the moving `latest` checkpoint alias: the active job will move `latest` on
its next FID upload. W&B validates each downloaded file against its artifact
manifest. Place the files at the relative paths used in the target command.
ImageNet itself is not uploaded and must be copied or mounted with standard
`train/` and `val/` directories.

## Cutover checklist

1. Let the RunPod job finish its current optimizer step and save a fresh
   recovery checkpoint. Do not copy a `.tmp` file.
2. Upload the final recovery checkpoint as a new `c5cos10r-checkpoint` artifact
   version and record its epoch/global step here if it is newer than step 11,250.
3. Download the migration artifacts and ImageNet onto the lab machine.
4. Implement and smoke-test the FSDP checklist before terminating RunPod.
5. Resume with W&B ID `c5cos10r`, confirm the printed epoch/batch/global-step
   cursor, and compare the first resumed LR with the last W&B LR.
6. Confirm both GPUs remain below 19 GB during training and FID generation.
7. Only after the lab resume has completed at least one optimizer step and a
   recovery save should the RunPod be shut down.

## Operational cautions

- Never delete the newest validated `last.pt` until its W&B artifact and lab
  download have both been verified.
- The trainer atomically replaces `last.pt`; an interrupted write should leave
  the previous checkpoint intact and may leave `last.pt.tmp`.
- W&B checkpoint upload occurs automatically only on FID epochs. Step-frequency
  recovery saves are local unless explicitly uploaded for migration.
- The stage-1 checkpoint contains legacy OmegaConf metadata. Use the trainer's
  compatibility loader rather than importing the old third-party package.
- Changing the per-GPU batch is resume-compatible. Changing the model
  architecture, global batch, optimizer-step count, or cosine horizon is not a
  transparent continuation of `c5cos10r`.
