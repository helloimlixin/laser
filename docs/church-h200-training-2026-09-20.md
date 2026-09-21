# Church training request, September 20, 2026

Both requested experiments finished training on all eight H200 GPUs. The compound continuation reached step 51,704 and best FID 11.2019. The integer experiment completed 90 epochs / 5,580 optimizer updates, with best FID 16.4650 (epoch 70) and final FID 17.1315. Its final best/last artifact was verified online before W&B finished. The old queue reports a failure because a final status callback attempted to log after `run.finish()`; training and checkpoint uploads had already completed. The subsequent user request starts a [fresh compound-token comparison](church-compound-ft1-2026-09-20.md) using the completed one-epoch tokenizer.

- Supervisor: `outputs/church-request-20260920/supervise.py`
- Live queue state: `outputs/church-request-20260920/queue-status.json`
- Launch receipt: `outputs/church-request-20260920/launch.json`
- Configuration plan: `outputs/church-request-20260920/queue-plan.json`
- Source hashes: `outputs/church-request-20260920/launch-source-manifest.json`
- Log: `outputs/church-request-20260920/supervisor.log`

## Compound continuation

Online run: <https://wandb.ai/helloimlixin-rutgers/laser/runs/church-laser-rfid421-ft3-compound-scratch90-20260918>

The online `last.pt` was recovered at step 47,437, epoch 48, old batch cursor 218. Forty additional epochs mean **5,048,320 additional training images**, equivalent to 39,440 updates at the old global batch of 128. The equivalent progress target is **86,877** old-batch updates. The recovered run had not reached its originally configured 90 epochs.

The production layout uses **192 images per GPU, global batch 1,536**, and no accumulation. Model weights and Adam state are restored. A six-update transition at batch 128 aligns the saved data cursor exactly before changing batch size. The learning-rate schedule advances by images seen on the original 88,740-update cosine horizon. Adam's learning rate scales by the square root of the batch ratio (sqrt(12)), with a one-epoch ramp from the restored learning rate. This scaling is implemented in `compound-runtime/data_progress_schedule.py` and survives checkpoint resume. The final image budget rounds up to a full new batch, adding at most 1,535 images; the concrete stop step is recorded in `compound-train/request.json`.

Profiling found that FlashAttention on the model's two- and four-token coefficient/depth sequences used about 65% of GPU time. Those short sequences now use dense attention with FP32 scores; the spatial transformer retains FlashAttention and cached generation retains its original path. Forward and gradient parity checks passed. The per-GPU batch-192 benchmark measured **259.9 images/second**, compared with **115.7** for the original batch-16 path. Batch 256 gained less than 1% while increasing allocated memory from 93.9 to 123.3 GiB. The live eight-GPU run then measured **2,030.95 images/second**, compared with **825.47** before optimization (about **2.46x**). GPU-synchronizing diagnostics now run only on rank zero at their ten-update logging cadence.

The change from four GPUs to eight requires new random streams. The first transition seeds each rank with `seed + restored_global_step * 8 + rank`; later eight-rank resumes restore every saved stream. This is a continuation of the saved training state, not a numerically identical four-GPU trajectory.

The original full token cache is reused byte-for-byte and loaded into RAM. Its SHA256 is `722aff4b6cdc1c62ae24d67bd620979b03cca6345104518c0fe67c1e7bd4824e`. The frozen tokenizer's tensors exactly match the recovered source stage-1 best checkpoint. Its newly exported file has different serialization metadata. Newly encoding images on H200 does not reproduce every original cached support bit-for-bit, so the original cached training codes are preserved. Verification details are in `compound-recovery/asset-verification.json`.

Samples and full recovery checkpoints occur every **500 optimizer updates**. FID uses the original 50,000-sample protocol every ten epochs and the original real-image reference. The existing best FID of 11.569545293216208 is retained until improved. Stable online files `last.pt` and `best-fid-01.pt` upload in the background from immutable hardlinks, with MD5 and size verification. The queue keeps one active transfer and the latest pending snapshot, preventing network throughput from stopping the GPUs or accumulating unbounded files. Final completion waits for the final upload to verify. FID generation uses batch **512 per GPU**, validated through a full 50,000-image evaluation at epoch 50, which achieved **FID 11.3565**; preview decoding uses batch 64.

## Integer-token experiment

Planned stage-1 ID: `church-laser-rfid421-ft1-yaml-h200x8-20260920`.

Planned stage-2 ID: `church-laser-rfid421-ft1-integer-scratch90-h200x8-20260920`.

The frozen source and LASER integer adapter follow the prior `church-laser-ft3ep-scratch-adaptive-lr-20260916` experiment. Stage 1 starts from the ImageNet rFID **4.210914** checkpoint and trains for exactly **one epoch**, 987 optimizer updates. The supplied [Church stage-1 YAML](https://github.com/kakaobrain/rq-vae-transformer/blob/main/configs/lsun-church/stage1/church256-rqvae-8x8x4.yaml) specifies Adam learning rate **4e-5**, betas `(0.5, 0.9)`, and the GAN/perceptual settings used here. LASER's sparse bottleneck is retained. Eight GPUs use batch 16 each, total 128. Paired-rank SyncBatchNorm preserves the reference discriminator's batch-32 normalization scope.

After stage 1, all 126,227 training images are encoded once. Preparation builds a complete FP32 latent cache, fits the 32,769-word adaptive integer vocabulary, calibrates the target temperature, and builds a uint16 integer-map cache. Each transformer rank loads the latent cache into RAM. Stochastic integer codes and soft targets are recomputed from cached latents on each visit, preserving the original RQTransformer training recipe; the hard integer map is retained for diagnostics. The frozen image encoder does not run during transformer training.

The stage-1 audit verified changes in all 162 encoder tensors, all 220 decoder tensors, both projection layers, the sparse dictionary, and all 14 floating-point discriminator state tensors. The dictionary uses LASER's alternating residual update after every generator update; its lack of an Adam gradient does not mean it is frozen. Production asserts that every requested component updates before accepting the finetuned tokenizer.

Stage 2 starts from random transformer weights with a fresh optimizer and trains **90 epochs**. All eight GPUs use batch 256 each, global batch **2,048**, no accumulation. AdamW starts at **5e-4**, betas `(0.9, 0.95)`, weight decay `1e-4`, with the 90-epoch cosine schedule. There are no FID-triggered learning-rate reductions. FP16 autocast uses FP32 codebook geometry and cross entropy.

The new transformer samples 100 images every **500 successful optimizer updates**. FID uses 50,000 generated images at epoch one and every ten epochs. Generation/decode batches are 64 per GPU. The latest full state and the single best-FID full state are committed in the background as online W&B artifacts, with `last`, `latest`, and `best-fid` aliases. Frozen tokenizer, integer vocabulary, calibration, and config are uploaded as a separate recovery artifact.

## Validation and operation

- Removed 45 obsolete diagnostic checkpoints, recovering **172.971 GiB**, plus **28.832 GiB** from completed diagnostics and abandoned upload snapshots (**201.803 GiB total**); production best/last checkpoints and data were preserved. Inventory: `outputs/church-request-20260920/cleanup.json`.
- Compound objective/upload tests: 29 passed. Continuation/resume helper tests: 20 passed.
- Eight-GPU compound preflight resumed the real checkpoint for two updates, reached step 47,439, saved eight RNG states, preserved scheduler state, produced samples, and passed finite-weight checks.
- Eight-GPU stage-1 and complete preparation preflights passed.
- Eight-GPU integer preflight and resume reached three successful updates, recovered optimizer/scheduler/RNG state, sampled on all eight ranks, and produced a 100-image grid. FP16 loss scaling skipped one overflowing update and recovered normally; checkpoint weights are finite.
- Before the queued 90-epoch run, another short offline preflight validates the actual one-epoch tokenizer and complete production cache. Its weights are discarded; production starts from scratch.

The supervisor checks source hashes before every phase and stops on errors. It only advances after completion receipts and checkpoint-upload receipts validate. It survives terminal disconnection but requires this machine to remain running. It can resume a partially completed compound or integer transformer from `last.pt`; incomplete stage-1/cache preparation requires review before restarting.

To inspect progress, read the queue state and its phase log. To restart a stopped queue after reviewing an error, run `python outputs/church-request-20260920/supervise.py` from the repository. A file lock prevents duplicate supervisors. The W&B key is read from a private file outside the repository; it is not embedded in launch commands or this document.
