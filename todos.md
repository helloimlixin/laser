# ImageNet RQ-Transformer migration runbook

Updated: 2026-07-27 UTC

## Migration status

- The four-GPU training run was intentionally stopped on 2026-07-27 UTC.
- No training or `torchrun` process should be active on the source machine.
- The latest completed handoff checkpoint is `checkpoints/last.pt` from epoch
  30, global step 11,513.
- Latest recorded 50K FID: 29.0004 at epoch 30.
- The launcher now defaults to eight local GPU processes and native NCCL peer
  transport. Paths and process count can be overridden through environment
  variables documented below.

Checkpoint SHA-256 checksums for transfer verification:

```text
5186e9498777f9f91069cfbdb0bc8af39d84524877b67b43a9235e5e98aa07fa  outputs/imagenet_x3h5cl0h_stage2_a16384_k2_c2048_m20/stage2/checkpoints/last.pt
e7644121b6b4644ac9405bae56881fd59d5f8dd32f91473ae965f30b1aac1606  outputs/imagenet_x3h5cl0h_stage2/stage1_checkpoint/best_rfid_slot3_model.pt
```

After copying, verify with `sha256sum` before launching.

## Current run

- W&B project: `laser`
- W&B run ID: `swgbasnb`
- W&B run name: `imagenet-official-rqtransformer-laser-a16384-k2-c2048-m20`
- Output directory: `outputs/imagenet_x3h5cl0h_stage2_a16384_k2_c2048_m20/stage2`
- Resume checkpoint: `outputs/imagenet_x3h5cl0h_stage2_a16384_k2_c2048_m20/stage2/checkpoints/last.pt`
- Stage-1 checkpoint: `outputs/imagenet_x3h5cl0h_stage2/stage1_checkpoint/best_rfid_slot3_model.pt`
- ImageNet root: `/workspace/Projects/data/imagenet`
- Launcher: `scripts/launch_x3h5cl0h_official_stage2.sh`
- Trainer: `scripts/train_official_rqtransformer_laser_stage2.py`
- Current checkpoint size: approximately 16.7 GB
- Resume is enabled by default. The trainer restores the model, AdamW optimizer,
  epoch, batch cursor, global step, and best-FID history.

## Architecture

```yaml
dataset:
  type: imagenet
  vocab_size: 16384
  transforms:
    type: imagenet256x256

arch:
  type: rq-transformer
  block_size: [8, 8, 4]
  embed_dim: 1536
  input_embed_dim: 256
  shared_tok_emb: true
  shared_cls_emb: true
  input_emb_vqvae: true
  head_emb_vqvae: true
  cumsum_depth_ctx: true
  vocab_size_cond: 1000
  block_size_cond: 1
  body:
    n_layer: 42
    block:
      n_head: 24
  head:
    n_layer: 6
    block:
      n_head: 24
```

LASER adaptation details:

- Dictionary atoms: 16,384
- Coefficient vocabulary: 2,048
- Combined classifier vocabulary: 18,432
- Coefficient maximum: 20
- Coefficient scale: 6.4
- Code layout alternates atom and coefficient depths.
- Stage-1 auxiliary embeddings use the frozen dictionary checkpoint above.

## Active training settings

```yaml
loss:
  type: soft_target_cross_entropy
  stochastic_codes: true
  temp: 0.5

optimizer:
  type: adamW
  init_lr: 0.0005
  weight_decay: 0.0001
  betas: [0.9, 0.95]
  max_gn: 1.0

experiment:
  amp: true
  batch_size_per_gpu: 8
  total_batch_size: 2048
  epochs: 100
  save_ckpt_freq: 2
  fid_every: 5
  fid_num_samples: 50000
  fid_batch_size: 96
  sample_grid_every_optimizer_steps: 100
  sample:
    top_k: 16384
    top_p: 0.92
```

Additional behavior:

- Full local checkpoints are saved atomically every two epochs.
- W&B checkpoint artifact upload is enabled.
- One artifact version is uploaded per checkpoint event under
  `swgbasnb-checkpoint`.
- Every uploaded version receives `latest`. A qualifying FID checkpoint also
  receives `best` and `epoch-N` aliases on the same artifact version. Do not
  restore the old duplicate `last` plus `best` uploads; that exhausted disk quota.
- 50K FID runs every five epochs.
- Class-grid previews run every 100 optimizer steps and do not write a full
  recovery checkpoint.

## Launcher environment

```bash
cd /workspace/Projects/laser
bash scripts/launch_x3h5cl0h_official_stage2.sh
```

The migration-ready launcher accepts:

```bash
NPROC_PER_NODE=8
IMAGENET_ROOT=/workspace/Projects/data/imagenet
STAGE1_CHECKPOINT=/workspace/Projects/laser/outputs/imagenet_x3h5cl0h_stage2/stage1_checkpoint/best_rfid_slot3_model.pt
OUTPUT_DIR=/workspace/Projects/laser/outputs/imagenet_x3h5cl0h_stage2_a16384_k2_c2048_m20/stage2
```

For a temporary four-GPU launch, set `NPROC_PER_NODE=4`. With four GPUs,
batch 8 per GPU, and total batch 2048, gradient accumulation is:

```text
2048 / (4 * 8) = 64 microbatches per optimizer step
```

## Switching to eight H100s

1. Copy the repository, stage-1 checkpoint, and latest stage-2 checkpoint to the
   same relative paths on the new machine.
2. Copy or recreate the ImageNet directory with `train/` and `val/` folders.
3. Ensure at least 40 GB of free quota beyond existing checkpoints: an atomic
   local save and W&B artifact staging may each temporarily require about 17 GB.
4. Authenticate W&B and keep the same run ID, `swgbasnb`.
5. The launcher now defaults to:

   ```bash
   torchrun --standalone --nproc_per_node=8
   ```

6. Keep `--batch-size 8`, `--total-batch-size 2048`, and `--lr 0.0005` to
   preserve the current optimization recipe. Accumulation becomes 32:

   ```text
   2048 / (8 * 8) = 32 microbatches per optimizer step
   ```

7. Native NCCL P2P/shared-memory transport is enabled by default. Only on a
   host with broken CUDA peer mappings should these be exported manually:

   ```bash
   export NCCL_P2P_DISABLE=1
   export NCCL_SHM_DISABLE=1
   ```

8. Launch inside a persistent session:

   ```bash
   tmux new-session -s imagenet_stage2
   cd /workspace/Projects/laser
   bash scripts/launch_x3h5cl0h_official_stage2.sh
   ```

9. Verify all ranks and the resume cursor:

   ```bash
   nvidia-smi
   ps -ef | rg 'torchrun|train_official_rqtransformer'
   ```

   Expected output includes a line similar to:

   ```text
   Resumed from .../checkpoints/last.pt: epoch=..., batch=..., step=...
   ```

## No-gradient-accumulation alternative

This is **not** the active configuration. On eight GPUs with batch 8, disabling
accumulation requires `--total-batch-size 64`. A linear LR scaling from the
active recipe gives approximately:

```text
0.0005 * 64 / 2048 = 0.000015625
```

That would create roughly 20,000 optimizer updates per ImageNet epoch instead
of about 625. It materially changes optimization and is not expected to reduce
the amount of image compute per epoch. Use it only as a deliberate new recipe,
not as a transparent resume of the current run.

## Operational cautions

- Never delete `checkpoints/last.pt` unless a replacement has been validated.
- An interrupted atomic save may leave `checkpoints/last.pt.tmp`; the previous
  `last.pt` remains valid. Remove only the incomplete `.tmp` after confirming no
  training process is writing it.
- Interrupted W&B uploads may leave large files under
  `/workspace/.local/share/wandb/artifacts/staging`. Confirm the uploader is dead
  before removing an orphaned staging file.
- The stage-1 checkpoint contains legacy OmegaConf metadata. The trainer uses a
  weights-only compatibility loader and does not require the old third-party
  package on `PYTHONPATH`.
- The current machine disables NCCL P2P and shared memory because its GPU peer
  mappings are unavailable. Do not carry those overrides onto a healthy
  NVLink/NVSwitch machine without testing native NCCL first.
