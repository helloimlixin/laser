# Compound stage two with the selected Church tokenizer, September 21, 2026

The user requested stage-two compound-token training with the best stage-one
checkpoint. This is a fresh 90-epoch transformer run using the selected
three-epoch LR 1e-5 Church tokenizer and its complete prebuilt compound cache.

[Online run](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-laser-ft3best-compound-nogeom90-h200x8-20260921).

## Selected stage one and cache

Stage one started from the ImageNet rFID 4.21 LASER checkpoint and finetuned
all components for three Church epochs. Epoch three was selected with full
126227-image reconstruction FID 2.6393, matched 4096-image reconstruction FID
6.0640, and held-out 300-image LPIPS 0.24729. This follows the completed
[stage-one comparison](church-stage1-improve-2026-09-20.md).

- Tokenizer SHA256: `762c51a10267ed6fa55709ff0d6cf997940d21056b16c7a0a0399b3ebf93868d`.
- Compound cache SHA256: `c99a079ce660013925e9a8514a12e86e070dd29c0f7c4e10cbb4e48e304f8d21`.
- Cache: all 126227 training images, each represented by 8×8×4 atom indices and normalized coefficients.
- Coefficient scales: `[5.3782935, 2.6181552, 1.7180322, 1.0875155]`, fitted to this tokenizer.
- Cache precision: FP32 encoder and OMP, TF32 disabled; int16 atom indices and FP16 coefficient storage.

The cache was validated against fresh encoding, with exact support agreement
on the probe. Stage-two training reads resident sparse components and performs
no image encoding or OMP in its training loop. The tokenizer remains frozen.
The run links the previously committed selected-model and cache W&B artifacts
as inputs and verifies their remote manifest sizes and MD5 values. It separately
uploads the exact stage-two request, source manifest, and stage-one provenance.

## Training configuration

The completed one-epoch-tokenizer compound run provides the baseline recipe;
the selected tokenizer and its matching coefficient scales/cache are replaced.
The transformer initializes from scratch, with fresh optimizer and scheduler.
Geometry loss and the geometry candidate forward path remain disabled, following
the user's earlier instruction.

| Setting | Value |
|---|---|
| Architecture | Church 350M RQTransformer preset; full compound-pair autoregression |
| Compound heads | Two micro-transformer layers; depth-specific coefficient heads |
| Vocabulary | 16384 atoms; 2048 coefficient bins; four sparse depths |
| Loss | `(1.5 * atom hard CE + coefficient soft CE) / 2.5` |
| GPUs / batch | Eight H200s; 192 per GPU; global 1536; no accumulation |
| Duration | 90 epochs; 82 updates per epoch; 7380 optimizer updates |
| Optimizer | Fused AdamW, betas (0.9, 0.95), weight decay 1e-4, gradient clip 1 |
| Learning rate | 5e-4 to zero over 90 epochs, cosine, no warmup |
| Precision | BF16 training; FP32 short depth attention and spatial FlashAttention |
| Preview | 64 generated images every 500 optimizer updates |
| Sampling | Atom top-k 250; coefficient top-p 0.85; temperatures 1 |
| FID | 50000 generated images at epochs 1, 10, 20, ..., 90 |
| FID reference | Same original RQ-VAE Church training statistics as the baseline |
| Recovery | Full checkpoint every epoch and every 500 steps |
| Online retention | Full last and best-FID checkpoints; verified immutable uploads |

The previously measured batch-size benchmark favored 192 per H200: batch 256
used substantially more memory for less than one percent extra throughput.
The new run retains this measured batch size and a fresh 90-epoch LR schedule.

Production verification measured approximately **5035–5042 training images per
second** over updates 40–80, with finite loss and zero geometry weight. W&B
confirmed the selected tokenizer/cache input artifacts, all eight GPUs, global
batch 1536, and the requested sampling interval. The first 50k-image FID
evaluation began after epoch one.

## Execution and recovery

Experiment root: `outputs/church-compound-ft3best-scratch90-20260921`.
The detached supervisor first runs an eight-GPU preflight with 12 actual
optimizer updates, sampling, finite full-state checks, and checkpoint saving.
Preflight and production use separate directories; production starts fresh.
The supervisor verifies the frozen source manifest and cache receipt before
launching training, and supports recovery from this run's own last checkpoint.

The preflight completed successfully at step 12 and saved its generated preview.
Source verification covers 382 execution/provenance files. Copied W&B output
logs were excluded from the restart manifest after startup; every execution
source hash remained unchanged. `source-manifest-scope.json` records this
filtering and retains the original launch manifest for provenance.

- `supervise.py`: detached preflight/training supervision, lock, and recovery.
- `status.json`: live phase and process IDs.
- `preflight/complete.json`: successful training/sampling preflight receipt.
- `train/request.json`: exact resolved experiment and configuration.
- `train/metrics.jsonl`: training throughput, objective terms, and evaluation.
- `train/token_cache_artifact.json`: verified input artifacts and provenance upload.
- `train/checkpoint-upload.json`: latest verified full last/best checkpoint artifact.
- `train/complete.json`: final completion receipt, written only after training and uploads finish.

Training completed all 90 epochs and 7380 optimizer updates. Best generation
FID was **11.8516475 at epoch 50**; final epoch-90 FID was **12.0336509**.
The full last and best-FID checkpoints were verified online in
`helloimlixin-rutgers/laser/church-laser-ft3best-compound-nogeom90-h200x8-20260921-selected-checkpoints:v8`.
The final checkpoint contains eight rank RNG states, optimizer step 7380,
finite model weights, and terminal learning rate zero.

The previous compound run completed 90 epochs with best generation FID 17.2044
using its original one-epoch tokenizer. The new tokenizer therefore improved
best measured generation FID under the matched compound recipe. Generated
architecture still has visible distortions; improved FID does not establish
that those errors are resolved. A separate sampler comparison uses the frozen
epoch-50 checkpoint; see `church-compound-sampling-2026-09-21.md`.
