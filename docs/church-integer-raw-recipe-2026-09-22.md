This run was superseded by the user's fresh compound-token request. The
single-GPU continuation stopped cleanly at epoch 54 / step 3,348; full latest
and best-FID checkpoints are verified online in selected-checkpoints artifact
v35. The replacement has full event history in both prediction decoders and is
documented in [the fresh compound run](church-compound-full-history-2026-09-22.md).

The user originally selected the **32,769-token integer model**, with coefficients
in raw latent units and no coefficient clipping. Stage 2 starts from random
weights and a fresh optimizer. The selected three-epoch Church LASER encoder,
decoder, dictionary, and fitted coefficient levels remain frozen.

Run: `church-laser-integer32769-raw-noclip-scratch300-h200x5-20260922`.
Output: `outputs/church-integer-raw-rqrecipe300-20260922/`.
Online: [W&B run](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-laser-integer32769-raw-noclip-scratch300-h200x5-20260922).

Each of the four residual depths predicts one integer: zero, or
`1 + 2 * atom_id + coefficient_level_id`. The vocabulary contains 16,384 atoms
with two fitted signed levels each, plus zero. It is the previously selected
adaptive scaled-atom residual quantizer, not the compound model with separate
atom and coefficient heads. A level multiplies its dictionary atom directly,
without per-depth scaling or coefficient clamping. The fitted levels range from
-13.903696 to 14.989018 in raw units. Finite-level quantization is lossy; encoding
integer IDs does not preserve arbitrary FP32 coefficient values exactly.

The exact reused codebook SHA256 is
`812c10ba93167663cb56c19da0c1673f6aa88feb0bc264cf9d1f2e2f13d4b1cb`.
Its dictionary and levels are checked against the original frozen tokenizer's
full state hash. A complete integer-vocabulary embedding check verifies that
every ID decodes to the corresponding raw coefficient times dictionary atom.
The full 126,227-image FP32 encoder cache is reused. There is no image encoder
execution in the training loop. All eight saved validation shards are combined
and redistributed across the current five ranks; the complete official
300-image validation population is retained.

Soft geometric targets over all 32,769 codewords and stochastic integer histories
are recomputed on each visit, using the original calibrated temperature 0.125.
Target geometry uses FP32 with TF32 disabled. This is not the compound model's
hard atom objective and separate scalar coefficient objective. The existing hard
integer-map cache is for reconstruction diagnostics, not repeated fixed targets.

The optimization recipe remains 300 epochs, global batch 2,048, AdamW with
LR 5e-4, betas (0.9, 0.95), weight decay 1e-4, and cosine decay to zero without
warmup. Residual dropout is 0.1, with zero attention/embedding dropout. The
released normal-0.02 initializer is applied. Optimizer gradient clipping at
norm 1 remains part of that recipe; coefficient clipping is disabled.

An exact global batch sampler partitions each shuffled epoch without duplicates
or omissions. Five ranks receive 410/410/410/409/409 images per full update,
optionally split into two microbatches according to measured throughput. The
last update has 1,299 images, giving exactly 62 updates per epoch and 18,600 total.
Local mean losses are weighted by `world_size * local_count / global_count`
before DDP's rank-averaged gradient reduction, including accumulated microbatches.

The new runtime uses BF16 transformer operations, fused AdamW, SDPA spatial
attention, and FP32 dense attention for four-token depth sequences. Soft CE
recomputes bounded FP32 row chunks in backward without changing the target
distribution or truncating its vocabulary. The target-construction chunk size
changes random-draw assignment relative to the old integer run; this fresh run
has its own fixed, checkpointed random streams. Training throughput and FID batch
sizes are selected by five-GPU benchmarks before production starts.

FID uses the unedited released RQ-VAE evaluator and 50,000 generated images every
epoch. The reference is all 126,227 Church training images with the same RGB,
bilinear short-side resize to 256, center crop to 256, and [-1,1] normalization
used for the encoder cache. The common data-protocol SHA256 is
`61e137beb3b1ec7029e0e67e964a757dee6633f8d6bd061d93f648790e17e875`.
Sampling follows the integer version: top-k 1,400, top-p 1, temperature 1.
The first 64 rank-zero FID samples form an 8-by-8 preview grid each epoch.
Decoder and Inception run in FP32 without TF32. Official Inception preprocessing
is applied to both reference and generated images. Evaluation restores training RNG.

Full latest and best-FID checkpoints include model, AdamW, scheduler, five-rank
RNG state, codebook/tokenizer hashes, and configuration. They are saved on local
`/tmp` storage and asynchronously uploaded to W&B with remote size/MD5 verification.
The upload queue retains one active transfer and the newest pending snapshot;
normal completion drains uploads. Local progress can precede the online checkpoint.

The replaced compound run stopped at a fully saved epoch boundary: epoch 187,
step 11,594, latest FID 13.12659 and best FID 9.95524 at epoch 57. Full latest/best
states are preserved in `compound-handoff/` under the new local storage directory,
and separately published to the original W&B run. They are not initialization
weights for the new integer model.

See `math-validation.json` for attention forward/gradient/cached equivalence,
the benchmark directories for full-state reload and production-architecture
sampling checks on every GPU, `validation.json` for focused tests,
`benchmark-selection.json` for throughput, `fid-benchmark.json` for generation
layout, and `train/initialization.json` for fresh initialization evidence.

The selected layout is one full local batch per optimizer update: **2,706.1
images/second**, versus 2,624.2 with two accumulated microbatches. Peak allocated
memory was 80.4 GiB. The fastest tested generation batch was 2,048 per GPU;
10,240 generated images with decoding/Inception took 22.18 seconds in the
benchmark. Thirteen focused tests pass. All five ranks restored the benchmark
model and AdamW tensors bit for bit, with identical scheduler and Python, NumPy,
CPU Torch, and CUDA Torch RNG streams. Production initializes independently:
386,882,561 parameters, zero optimizer-state entries, initial weights SHA256
`ca0c5887b5e5b760a1a743e4742fa5184fc4e4558d35376ca18869eb9469084c`.

Production completed its first epoch at step 62 with FID50k 156.84611; epoch 2
scored 136.24288. At launch verification, four epochs had completed. The first
full checkpoint artifact, `selected-checkpoints:v0`, is committed and independently
verified online: `last.pt` and `best-fid-01.pt` are each 4,643,146,466 bytes with
matching local/remote MD5 digests. The first transfer took about ten minutes;
newer latest/best snapshots continue uploading asynchronously while training runs.
The epoch-1 grid was independently downloaded from W&B and matched byte for byte.
The exact full-training FID statistics and data-protocol record are also uploaded
and hash-verified. `launch-complete.json` collects these verification receipts.

## Continuation after the machine restart

On September 22, the user requested continuation of this same run to epoch 300.
The original run had stopped cleanly at epoch **37**, optimizer step **2,294**,
with FID50k **12.2526825**. Its best FID was **11.8709209** at epoch **36**.
Both full checkpoints were recovered from the committed W&B artifact
`church-laser-integer32769-raw-noclip-scratch300-h200x5-20260922-selected-checkpoints:v17`.
Each is 4,643,146,466 bytes; both downloads match the remote MD5 manifest.

The replacement machine exposes one H200. The continuation executes the five
original rank batches sequentially, retaining each rank's saved Python, NumPy,
CPU Torch and CUDA Torch random state. Per-microbatch losses retain the original
`5 * local_count / global_count` weighting through BF16 backpropagation. The
accumulated FP32 parameter gradients are then divided by five before one
gradient clipping and AdamW update. Global batch 2,048, the final batch of 1,299, all 126,227
training images, 62 updates per epoch, and the 18,600-step cosine schedule are
preserved. There is no optimizer or learning-rate reset. Floating-point gradient
summation order differs from NCCL, so continuation is not claimed to reproduce
the five-GPU trajectory bit for bit.

FID still evaluates 50,000 generated images each epoch against the same verified
reference, running the five original seed streams of 10,000 images sequentially.
The sampling settings, generation batch size, frozen tokenizer, full-vocabulary
soft targets, and raw coefficient representation are unchanged.

The continuation driver is
`outputs/church-integer-raw-rqrecipe300-20260922/resume_single_gpu.py`.
Its supervisor, logs, receipts, samples, and durable checkpoints are in
`outputs/church-integer-raw-rqrecipe300-20260922/continuation-h200x1-20260922-fixed/`.
`serial-rank-validation.json` records full-epoch coverage, exact original rank
batches, independent CUDA dropout streams, and exact BF16-backward/FP32-gradient
and fused-AdamW equivalence checks for both full and final batches. Startup verifies every data hash and the restored
model, optimizer tensors, scheduler and checkpoint metadata.

The detached supervisor targets epoch 300 and retries failed workers from full
epoch checkpoints. Each epoch queues its latest and best-FID full states into
the same W&B `selected-checkpoints` artifact collection, with `latest`, `last`
and `best-fid` aliases and remote size/MD5 verification. Completion is recorded
only after epoch 300, step 18,600, and successful final checkpoint upload.
`status.json`, `supervisor-status.json`, and `checkpoint-upload.json` distinguish
current training progress from the latest checkpoint verified online.

The initial single-GPU attempt divided the loss by five before BF16 backward.
Although mathematically equivalent in real arithmetic, a CUDA audit measured
0.354% relative gradient error from BF16 rounding. Its epoch-38 FID was 17.27108.
That attempt is retained under `continuation-h200x1-20260922/` and marked
superseded. The corrected continuation reloads epoch 37 and delays division until
FP32 gradients are accumulated; the CUDA comparison now has zero gradient and
optimizer-update discrepancy. This establishes the precision correction, not
that it explains the entire observed FID change. Replaying the old epoch-37 first
FID batch also reproduces its original grid within one pixel intensity level
(mean absolute difference 0.0000743 on the 0–255 scale).
The subsequent full 50,000-image replay scored **12.252683786**, versus the
original **12.252682509**, an absolute difference of **0.000001277**. This verifies
the single-GPU sampling and released FID evaluator against the original result;
the audit is recorded in `baseline-fid-audit.json` and the W&B run summary.
The corrected epoch 38 scored **16.0398914**, so it did not replace the epoch-36
best of **11.8709209**. The precision correction improved the observed result
relative to the superseded attempt but did not restore the previous FID; the
original 300-epoch schedule continues with best-checkpoint retention.
The corrected latest and best states are committed in W&B artifact
`church-laser-integer32769-raw-noclip-scratch300-h200x5-20260922-selected-checkpoints:v19`,
with remote size/MD5 verification. It contains epoch-38 `last.pt` (4,643,147,170
bytes) and epoch-36 `best-fid-01.pt` (4,643,146,466 bytes). The `latest`, `last`,
and `best-fid` aliases point to this bundle. The first attempt's `v18` is marked
superseded and retained for diagnosis. Training advanced into epoch 39.

The separate `monitor.py` watchdog checks progress and uploads every 30 seconds.
It requests recovery after 15 minutes without progress or 45 minutes without a
pending checkpoint upload completing, allows an epoch boundary for graceful
shutdown, and restarts an unexpectedly lost supervisor. Interventions are bounded
to avoid an uncontrolled restart loop. Its live record is `monitor-status.json`;
it exits successfully only after epoch 300 and the final verified online upload.
It also publishes a heartbeat, current epoch, online-checkpoint epoch, and worker
health to the W&B run file `monitor-status.json` every five minutes, mirrored in
summary fields under `monitor/`. The dedicated file persists independently of
the training client's summary updates. The full baseline audit is also stored
in the W&B run file `baseline-fid-audit.json`.
