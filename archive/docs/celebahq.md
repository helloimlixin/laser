# CelebA-HQ compound LASER RQ-Transformer

Updated: 2026-08-04 (America/New_York)

## Active run

- W&B: [celebahq-compound-rqt350-a2048k2-20260804](https://wandb.ai/helloimlixin-rutgers/laser/runs/celebahq-compound-rqt350-a2048k2-20260804)
- Name: `celebahq-a2048-k2-compound-v5b-official-rqtransformer-350M`
- State at this snapshot: running, optimizer step 2,470, epoch 11
- Latest loss: 5.84379
- Throughput: 129.19 images/second
- Learning rate: 0.0005
- Launcher: `scripts/launch_celebahq_a2048_k2_compound_official_stage2.sh`
- Trainer: `scripts/train_official_rqtransformer_laser_stage2.py`
- Production log: `runs/celebahq_a2048_k2_compound_production.log`
- Output: `outputs/celebahq-a2048-k2-rqvae-strict-20260720-145706/stage2-compound-v5b-official-rqtransformer-350M`

## Dataset

- Root: `/home/xl598/Projects/data/celeba_hq`
- Training images: 28,000
- Validation images: 2,000
- Layout: ImageFolder-compatible `train/{female,male}` and `val/{female,male}`
- Stage 2 is unconditional. Dataset class labels are replaced with condition ID 0.
- Cached training transform: resize to 256, then 256x256 center crop and normalization to `[-1, 1]`

## Stage-1 checkpoint

- Source W&B run: `helloimlixin-rutgers/laser/celebahq-a2048-k2-rqvae-strict-20260720-145706`
- Selected file: `best_rfid_slot1_model.pt`
- Selected epoch: 150
- Reconstruction FID: 18.42058
- Local path: `outputs/celebahq-a2048-k2-rqvae-strict-20260720-145706/stage1_checkpoint/best_rfid_slot1_model.pt`
- Size: 1,211,986,838 bytes
- Verified MD5: `7b098c140b8b653ca263d9d12f9cbb22`

Stage-1 architecture:

- Strict original FFHQ 256 RQ-VAE encoder/decoder geometry
- Latent shape: `8x8x256`
- Sparse code shape: `8x8x2`
- Dictionary atoms: 2,048
- Sparsity: two OMP atoms per spatial location
- Embedding dimension: 256
- Shared dictionary across depths
- Attention resolution: 16
- Encoder/decoder base channels: 128
- Channel multipliers: `[1, 1, 2, 2, 4, 4]`
- Residual blocks per level: 2
- Stage-1 precision: FP32
- Stage-1 global batch: 128
- Stage-1 epochs: 150
- Adam learning rate: 4e-5
- Adam betas: `(0.5, 0.9)`
- Hinge discriminator loss, vanilla generator loss
- Discriminator weight: 0.75
- Perceptual weight: 1.0
- Discriminator active from epoch 0

## Compound token cache

- Path: `outputs/celebahq-a2048-k2-rqvae-strict-20260720-145706/token_cache/celebahq_train_compound_pairs.pt`
- Rows: 28,000
- Tensor shape per image: `8x8x2`
- Atom dtype: int16
- Coefficient dtype: float16
- Coefficient vocabulary: 1,024 uniform bins over `[-3, 3]`
- Per-depth physical coefficient scales: `[45.83333206, 10.94791698]`
- Scale calibration: full-dataset absolute maximum maps to coefficient magnitude 3
- Compound sequence length: 128
- Scalar interleaved baseline length: 256
- Atom verification agreement: 100%
- Maximum normalized coefficient cache error: 0.001578
- Physical coefficient quantization MAE: 0.04209
- Quantized reconstruction PSNR: 14.97394 dB
- Duplicate atom rate within a pair: 0
- Validation result: passed

W&B cache artifact:

- Collection: `celebahq-compound-rqt350-a2048k2-20260804-token-cache`
- Current version: `v0`
- Aliases: `latest`, `training-cache`
- Artifact file size: 14,394,542 bytes

## Stage-2 model

The global architecture follows the original FFHQ 350M RQ-Transformer preset,
with the compound LASER pair model added locally at each depth event.

- Parameter count: 383,477,760
- Spatial grid: `8x8` (64 sites)
- Compound depth: 2 events per site
- Autoregressive latent length: `8 x 8 x 2 = 128`
- Unconditional condition positions: 1, always condition ID 0
- Transformer embedding width: 1,024
- Input/latent embedding width: 256
- Spatial body: 24 layers, 16 attention heads
- Local depth head: 4 layers, 16 attention heads
- Atom vocabulary: 2,048
- Coefficient vocabulary: 1,024
- Shared token embeddings and cumulative depth context enabled
- Two-layer pair-local coefficient micro-transformer
- Separate coefficient classifier for each of the two sparse depths

Each autoregressive event factors as:

```text
p(atom, coefficient | history)
  = p(atom | history) * p(coefficient | history, atom)
```

The model therefore advances global autoregressive history 128 times. Each
event makes one atom decision and one coefficient decision. The two physical
contributions at a spatial site are summed to reconstruct an `8x8x256` latent
for the frozen stage-1 decoder.

## Training objective

- Compound atom/coefficient classification: enabled
- Atom NLL weight: 1.5
- Coefficient cross-entropy weight: 1.0
- Distribution-derived physical geometry loss: enabled
- Geometry candidate top-k: 4 atoms plus the target atom
- Target geometry weight: 0.05
- Geometry begins after epoch 2
- Geometry warmup: epochs 2 through 5
- Geometry compares both pair-level and summed spatial physical contributions
- Stage-1 encoder, decoder, and dictionary remain frozen

## Optimization and hardware

- Hardware: 2 x NVIDIA RTX 4000 Ada, 20 GB each
- Distributed backend: DDP
- Per-GPU microbatch: 8
- Gradient accumulation: 8 microbatches
- Effective global batch: `2 GPUs x 8 x 8 = 128`
- Epochs: 200
- Optimizer: AdamW
- Learning rate: fixed 0.0005
- Weight decay: 0.0001
- Betas: `(0.9, 0.95)`
- Gradient clipping: 1.0
- Training precision: BF16 autocast with FP32 parameters and optimizer state
- Peak after resumed optimizer step: 9.28 GiB allocated, 9.39 GiB reserved per GPU
- Resume: enabled from `checkpoints/last.pt`

## Preview sampling

Preview sampling runs once on launch/resume and every 1,000 optimizer steps.
At the current throughput, 1,000 steps are approximately 4.6 epochs or 16
minutes. Every preview event produces four independent 8x8 grids (64 samples
per setting), generated in batches of eight.

The PNGs are raw 2048x2048 mosaics. They contain no titles, subtitles, labels,
margins, borders, or padding. Every file and W&B panel represents exactly one
sampling setting.

| Setting/file suffix | Atom sampling | Coefficient sampling |
|---|---|---|
| `at1_k250_p1__ct1_k250_p1` | temperature 1.0, top-k 250, top-p 1.0 | temperature 1.0, top-k 250, top-p 1.0 |
| `at0.85_k250_p1__ct0.85_k250_p1` | temperature 0.85, top-k 250, top-p 1.0 | temperature 0.85, top-k 250, top-p 1.0 |
| `at0.9_k0_p0.92__ct1_k0_p0.85` | temperature 0.9, no top-k, top-p 0.92 | temperature 1.0, no top-k, top-p 0.85 |
| `at0.95_k250_p0.95__ct0.9_k250_p0.95` | temperature 0.95, top-k 250, top-p 0.95 | temperature 0.9, top-k 250, top-p 0.95 |

The main FID/IS generation protocol uses the original setting: temperature
1.0, top-k 250, and top-p 1.0 for both atom and coefficient predictions.

## Evaluation

- Metrics: FID, Inception Score, and Inception Score standard deviation
- Cadence: every 50 epochs
- Generated samples per evaluation: 50,000
- Generation batch per GPU: 8
- Real FID reference: all 28,000 training images
- Sampling setting: temperature 1.0, top-k 250, top-p 1.0

This matches the original FFHQ evaluation convention of comparing 50,000
generated images against the training-image distribution.

## Checkpoints and W&B artifacts

Local retention:

- Atomic `last.pt` recovery save every 250 optimizer steps
- Scheduled full checkpoint every 10 epochs
- Full checkpoint at every FID/IS evaluation and at the final epoch
- Keep the three checkpoints with lowest FID
- Keep the three checkpoints with highest Inception Score
- Always keep the latest `last.pt`

W&B retention:

- Checkpoint collection: `celebahq-compound-rqt350-a2048k2-20260804-checkpoint`
- Current uploaded version at this snapshot: `latest` containing the epoch-10 `last.pt`
- Every scheduled full artifact version contains `last.pt`, all locally retained
  top-three FID checkpoints, and all locally retained top-three Inception Score
  checkpoints.
- New metric winners receive `best-fid`/`fid-epoch-N` or
  `best-is`/`is-epoch-N` aliases.
- Step-250 atomic recovery snapshots stay local; uploading a changing 4.3 GB
  artifact every few minutes would stall training. The latest recovery state is
  included in every scheduled full upload.

No FID or Inception best checkpoint exists before the first epoch-50 evaluation,
so the current artifact contains only `last.pt`. The top-three sets will begin
populating at epoch 50.

## Launch and resume

Run or resume with:

```bash
nohup setsid scripts/launch_celebahq_a2048_k2_compound_official_stage2.sh \
  >> runs/celebahq_a2048_k2_compound_production.log 2>&1 < /dev/null &
```

The launcher validates the dataset, stage-1 checkpoint, and token cache before
starting. The trainer also validates cache dataset identity, atom vocabulary,
coefficient vocabulary, coefficient range, and calibrated coefficient scales.

## Validation completed

- Stage-1 checkpoint MD5 matches the W&B source file.
- Token cache structural and numerical validation passed.
- One production-shaped DDP optimizer step passed.
- Batch-8 compound autoregressive generation and stage-1 decoding passed.
- Focused compound-objective, model-preset, scheduler, and checkpoint tests:
  11 passed.
