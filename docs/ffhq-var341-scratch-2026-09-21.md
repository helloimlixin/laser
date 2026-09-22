FFHQ-256 trains a fresh tokenizer and a fresh compound VAR prior. Production
does not load CelebA-HQ weights or preflight weights. The frozen runtime and
its source hashes are in `outputs/ffhq256-var341-scratch-20260921/`; the
detached supervisor writes its current stage to `pipeline-status.json`.

All 70,000 original FFHQ images passed official MD5, file-size, and ZIP CRC
checks. Images are resized from 1024 to 256 with PIL LANCZOS and stored as
lossless PNG bytes in Hugging Face Arrow files. The official split is
60,000 training images and 10,000 validation images, with disjoint recorded
image IDs. Training uses seeded horizontal flips. All labels are zero:
this is an unconditional model with one class, no conditioning dropout,
and CFG 0 (the implementation's unguided setting).

The model has scales `(1, 2, 4, 8, 16)`, totaling 341 spatial sites, two
atom/coefficient pairs per site, 4,096 atoms, 257 coefficient bins, 32 latent
channels, and tokenizer width 160. Scratch initialization uses the native
evenly spaced residual-convolution coordinates. Fixed pretrained LPIPS and
Inception networks supply loss and evaluation features.

The supervised stages are:

1. Six tokenizer updates on three H200s, exercising adversarial backward,
   both optimizers, finite-gradient checks, and three-rank RNG serialization.
2. Fifty tokenizer epochs, batch 64 per GPU (192 total), LR 1e-4 for the
   backbone and dictionary, 200 warmup updates, fresh discriminator, and
   adversarial loss after 1,000 updates. The existing L1, LPIPS, quantization,
   and adaptive adversarial losses are retained. Every epoch evaluates
   reconstruction FID against all 10,000 validation images. The best
   tokenizer must pass the reconstruction-FID threshold of 50.
3. Stochastic-support and physical-coefficient calibration, retaining the
   +5% latent-MSE and +0.005 LPIPS acceptance limits.
4. A new cache of all 60,000 training images, both flip views, and 16 complete
   stochastic trajectories per view, plus deterministic validation codes.
5. A 100-update prior overfit check, production-batch updates, distributed
   sampling/FID, and optimizer/RNG checkpoint checks.
6. Fifty fresh VAR-d16 prior epochs, batch 128 per GPU (384 total), LR 3e-4,
   one warmup epoch, final LR 3e-5, and atom-loss weight 1.5. Sampling uses
   top-k 250, top-p 1, and 64-image grids arranged in eight rows and columns.
7. Best-generation-FID checkpoint evaluation with 2,000 and 50,000 generated
   images, each against the same 10,000 held-out real images. Periodic
   training FID uses 2,000 generated images. These are diagnostic estimates,
   not the official ADM FID50k protocol.

The [tokenizer run](https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhq256-var341-tokenizer-scratch-20260921)
starts first. The [prior run](https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhq256-var341-stochastic-compound-scratch-20260921)
starts automatically after calibration, caching, and preflight pass.

Large files use local `/tmp` storage because the persistent workspace is
near its quota. Exact prepared images, the token cache, source/configuration
provenance, and the selected tokenizer are published as W&B artifacts.
Tokenizer last/best-rFID and prior last/best-FID checkpoints are uploaded
online asynchronously from immutable snapshots, including optimizer and RNG
state. Tokenizer publication runs after epoch 1, every five epochs, and at
completion or graceful interruption. Confirmed upload receipts record file
sizes and digests. Stage handoff verifies committed online checkpoints;
missing local state on resume must be restored from the matching artifact.

The supervisor writes `complete.json` only after both training stages,
calibration, caching, final evaluation, and online checkpoint verification
finish. Failures stop the pipeline and record the failing stage. Focused
verification in the frozen runtime passed 28 tests, covering FFHQ checksum
and split handling, unconditional sampling, exact checkpoint serialization,
upload snapshots, resume contracts, cache consistency, and scratch models.

Launch verification: the six-update three-GPU tokenizer preflight passed,
including both optimizers and three saved RNG states. Production independently
initialized the same scratch weights, then advanced beyond 500 updates at
approximately 208 images/second. W&B reported the tokenizer run as running
with the intended FFHQ dataset, scratch initialization, five scales, 64-sample
previews, eight grid columns, and checkpoint uploads enabled. The first full
10,000-image reconstruction evaluation improved from 346.30 before training
to 163.09 after epoch 1. This is reconstruction FID, not generation FID.
The provenance artifact is committed; the initial checkpoint and image-data
uploads were still pending at the end of launch monitoring. Their committed
state must be established by upload receipts and the supervisor's online
verification, not inferred from the fact that they were queued.

The first last/best tokenizer bundle subsequently committed as
`ffhq256-var341-tokenizer-scratch-20260921-checkpoints:v0`.

On request to accelerate training, the pipeline was checkpointed after epoch
3, step 936. Three-GPU benchmarks used isolated copies of that full state,
16 updates per case, production batch 64 per GPU, and adversarial training
enabled. Median throughput over updates 6–16 was 193.15 images/second for
the original execution settings, 196.95 with cuDNN autotuning, 158.66 with
autotuning and channels-last throughout the model, and 199.74 with autotuning
and channels-last only for the frozen LPIPS network. The last option was
selected, a 3.41% throughput improvement. Full-model channels-last was
rejected. TF32 remains disabled and the existing BF16 autocast policy is
retained. Different convolution algorithms can change floating-point rounding.
These are execution changes; model dimensions, optimizer settings, batches,
data, training budgets, and evaluation frequency remain unchanged.

Evaluation now caches the globally reduced Inception moments of the fixed
real validation images within each worker process, keyed by subset size.
Every epoch still reconstructs and evaluates all 10,000 validation images.
A three-rank, 192-image check produced identical FID before and after cache
reuse (absolute difference 0), while halving Inception calls from two to one
per rank. The reference cache is rebuilt after a process restart. Frozen-runtime
regression verification again passed 28 tests. Benchmark results, numerical
checks, original source/configuration snapshots, and the selection receipt
are recorded under `speed-tuning/`. Production resumed from step 936 with
its optimizer and RNG state; benchmark updates were not transferred.

Stall recovery on 2026-09-21: updates stopped after step 1280 at 18:53 UTC.
Rank 0 had no data workers or NCCL watchdog threads remaining and was blocked
in cleanup; ranks 1 and 2 were waiting with low GPU power. Empty temporary
status and failure-receipt files point to an output-filesystem write failure.
The original exception could not be recovered, so its precise filesystem
error is not confirmed. Later write probes succeeded. The latest complete
checkpoint was epoch 4, step 1248, with finite model weights, both optimizer
states, and three RNG states. Recovery necessarily replays work since that
checkpoint. The best checkpoint remained epoch 3, reconstruction FID 50.55.

Live run files now reside in
`/tmp/laser-var-runs/ffhq256-var341-scratch-20260921/`; the original output
path is a symlink to this directory. Its former contents were retained in
`outputs/ffhq256-var341-scratch-20260921-workspace-backup/`, which also receives
best-effort metadata mirrors every minute. Mirror failures are recorded
locally and cannot block GPU workers. W&B remains the online checkpoint
backup. Updated source/configuration provenance is republished on resume.

Fatal worker errors now write an independent local diagnostic and exit
immediately, allowing torchrun to stop peers instead of blocking inside
collective teardown. The supervisor permits three attempts per stage and
checks training progress every five seconds. Ten minutes without progress
triggers bounded termination and a checkpoint-based retry; explicit final
checkpoint-upload waits have a one-hour allowance. Two worker-failure tests
and three supervisor tests cover failed filesystem writes, bypassing
collective teardown, retrying a failed worker, stale-progress detection,
and nonfatal workspace-mirror failures. The existing 28 regression tests
also passed after the recovery changes.

Stage 2 extension on 2026-09-22: the original pipeline completed all 50
tokenizer epochs and 50 prior epochs, including online artifact verification
and final evaluation. The selected tokenizer is epoch 47 (10,000-image
reconstruction FID 4.84459). Prior sampled FID improved from 41.0691 at epoch
45 to 37.8654 at epoch 50, using 2,000 generated images against the fixed
10,000-image validation reference. The epoch-50 checkpoint's 50,000-sample
evaluation was 30.03312 against that same held-out reference. Validation NLL
had flattened, so improvements in generation must be measured rather than
inferred from training loss.

The requested continuation extends only stage 2 to epoch 75. It resumed from
epoch 50/update 7800 with the original AdamW moments, all three RNG states,
selected tokenizer, and stochastic token cache. The learning-rate schedule
retains its original 50-epoch duration and holds the final rate at 0.00003
through the additional 25 epochs. An explicit epoch-extension flag permits
the increased budget while retaining checks against changed architecture,
data, objective, cache, and tokenizer. New checkpoints store the original
schedule duration so that subsequent recovery cannot restart the decay.
Global batch remains 384 on three H200s; previews remain 64 images in an
8-by-8 grid. Best-FID selection and online last/best checkpoint uploads
continue. Live verification reached update 8280 at about 1,012 images/second
with LR 0.00003. W&B confirmed the run is running with a 75-epoch budget;
the resumed epoch-50 bundle committed as version 7.

Original metadata, results, and source/configuration versions are archived
under `extensions/epoch050-to075-20260922/` in the run directory. Immutable
hard links to the original epoch-50 checkpoints are under
`/tmp/laser-var-checkpoints/ffhq256-var341-scratch-20260921/history/epoch050/`.
The original online bundle is checkpoint artifact version 6.

Final evaluation retains the held-out diagnostics and adds two evaluations
using RQ-Transformer's released `ffhq_256_train.npz`: the best checkpoint
after the extension and the preserved epoch-50 baseline, each generating
50,000 images with the same seeds and sampling settings. This additional
evaluation uses the released Inception implementation, an FP32 decoder,
continuous float pixels in [0,1], and float64 streaming feature moments.
It does not round generated pixels to uint8. The published statistics archive
passed its publisher MD5; the selected file's SHA256 is checked before use.
The final W&B evaluation artifact includes the reference statistics and
evaluation source, alongside results and grids.

The released RQ-Transformer FFHQ score is 10.38, as documented at
https://github.com/kakaobrain/rq-vae-transformer. This remains a qualified
comparison: LASER uses the official contiguous 60,000/10,000 split, whereas
RQ uses its supplied shuffled split (51,397 training images overlap).
Architectures and training budgets also differ. A lower score against the
same released statistics would not establish a fully controlled model
comparison. No improvement or benchmark win is assumed in advance.

Validation for this extension passed 25 focused resume, compound model,
pipeline, checkpoint, and failure-handling tests. After adding the reference
evaluation, 13 selected evaluation/configuration/resume tests passed, including
continuous pixel preservation, reference checksum rejection, and rejection of
incomplete distributed sample counts. The final evaluator also passed Python
compilation.

The first extension check at epoch 55 scored 39.7381 on the unchanged
2,000-generated/10,000-held-out diagnostic, worse than epoch 50 (37.8654).
The epoch-50 best-FID checkpoint therefore remains selected. Training
continues to the authorized epoch-75 budget; this first check does not
establish an improvement. The extension passed update 8600 at approximately
1,012 images/second, with all three GPUs active.
