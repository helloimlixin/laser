# Fresh LASER Church training with RQ-Transformer recipe

On September 22, the user selected the earlier 32,769-token integer model.
This compound run was stopped and preserved at epoch 187 / step 11,594, with
best FID 9.95524 at epoch 57. Its full final and best states are verified online.
The active replacement is documented in
[the raw-level integer run](church-integer-raw-recipe-2026-09-22.md).

The September 21 request supersedes the epoch-60 continuation and the queued
augmentation/dropout interventions. This experiment initializes a new LASER
stage-two transformer and AdamW optimizer. The selected LASER tokenizer remains
frozen. No previous transformer weights, optimizer state, or best FID are inherited.

Run: `helloimlixin-rutgers/laser/church-laser-scratch-rqrecipe300-b2048-h200x5-20260921`.
Launch directory: `outputs/church-stage2-released-recipe-20260921`.

| Setting | New run |
|---|---|
| Training data | All 126,227 LSUN Church training images, once each epoch |
| Monitoring | Separate official 300-image validation split; no images removed from training |
| Duration | 300 epochs, 62 updates per epoch, 18,600 updates total |
| Global batch | 2,048; smaller final batch of 1,299 images |
| Optimizer | Fused AdamW, betas (0.9, 0.95), weight decay 1e-4 |
| Learning rate | 5e-4, cosine to zero over all 300 epochs, no warmup |
| Gradient clipping | Global norm 1.0 |
| Dropout | Residual 0.1; attention and embedding 0 |
| Initialization | Released `Stage2Model._init_weights` helper: normal std 0.02, zero linear bias, unit LayerNorm |
| Preprocessing | PIL RGB, bilinear short-side resize to 256, center crop 256, normalize to [-1,1] |
| Image augmentation | None |
| FID | 50,000 generated images every epoch, full training population as real reference |
| Checkpoints | Full latest and best FID states, including optimizer, scheduler and all five ranks' RNG; online W&B artifacts |

The paper specifies batch 2,048; the public example Church YAML specifies 256.
This run follows the paper batch from the user's quotation. The released stage-two
training loop is unavailable, so the original initialization call site and optimizer
parameter grouping cannot be independently confirmed. The released initialization
helper is applied to the fresh LASER model; AdamW weight decay applies to all its
trainable parameters. This is a LASER adaptation of the recipe, not an identical
RQ-code architecture.

LASER-specific choices remain explicit: 404,738,048 trainable parameters, 16,384
atom IDs, 2,048 coefficient bins, four pairs per spatial location, full pair
autoregression, physical coefficients without clipping, stochastic OMP support
bank with 16 variants, coefficient target temperature 0.125, atom/coefficient
loss weights 1.5/1.0, and uniform depth weighting. Generation uses atom top-k 250
and unrestricted coefficient sampling, both at temperature 1. These are not
claimed to be the original RQ-Transformer's code distribution or top-k 1400 sampler.

## Exact batching and throughput

`src/training/exact_global_batch.py` partitions one global permutation into batches,
then splits each batch across the five ranks. With no accumulation the full-batch
sizes are 410, 410, 410, 409, 409. Each rank's mean loss is weighted by
`world_size * local_images / global_images` before DDP reduction. With accumulation,
the same image weighting applies across all microbatches. This computes the global
mean gradient despite unequal rank sizes. No padding duplicates or dropped images
are needed. The sampler cursor resumes only at completed optimizer boundaries.

The isolated runtime patches the frozen trainer to use this sampler. Tests verify
full Church coverage, exact batch counts, global-gradient equivalence for full and
partial batches, deterministic resume ordering and RNG isolation. Five-GPU trials
compare accumulation 1 and 2, using fresh disposable models with the final initializer.
The production model is initialized again independently; benchmark weights are unused.
`benchmark-selection.json` records the measured choice. BF16 transformer operations,
FlashAttention for spatial attention, exact dense attention for short depth sequences,
fused AdamW, pinned cached-data loading and asynchronous checkpoint uploads improve
throughput. `fid-benchmark.json` records evaluation batch-size trials.

The selected layout uses one batch per rank per update: 3,191.6 images/second
versus 3,087.1 with two microbatches, at 100.7 GiB peak allocated memory.
Sampling batch 2,048 per GPU was fastest among 512, 1,024 and 2,048. The first
production FID50k completed in 66.7 seconds including held-out monitoring, and
epoch 1 produced FID 145.8415. This is an early scratch-training result, not a
comparison to a converged model. All five GPUs reached 100% utilization during
generation. The production job continues under the detached supervisor.

At initial verification, six epochs had completed and FID improved to 105.2654.
The first full checkpoint artifact, `selected-checkpoints:v0`, was independently
verified online: both `last.pt` and `best-fid-01.pt` were 4,857,570,698 bytes and
matched their local MD5 digests. Later latest/best snapshots continue uploading
asynchronously; network throughput means online checkpoint state can lag training.

Visual sample grids were enabled after the user's follow-up. A checkpointed
restart restored epoch 16, step 992, the AdamW state, the cosine position
(LR 0.0004964990092676253), and all five rank RNG streams. Every subsequent
epoch logs an 8-by-8 grid under `samples/fid_grid` in the same W&B run and saves
it in `train/samples/`. These are the first 64 rank-zero images already generated
for FID50k; no extra sampling, altered temperature, changed random draws, or
selection of favorable images is involved. The image-capture check confirmed
unchanged metric inputs and RNG. The original launch source is preserved in
`before-preview-logging/`; the updated wrapper and its manifest are recorded
in the `preview-logging` W&B code artifact.

## Official FID implementation

The upstream `fid.py` and `inception.py` are copied without edits from commit
`341395e562ac347f5eb62db9f5f08b9f2cc42a60`. The evaluation calls upstream
`get_inception_model`, `mean_covar_numpy`, and `frechet_distance` directly.
Only a runtime compatibility adapter handles SciPy's removed `sqrtm(disp=...)`
argument; the equations and safeguards are unchanged. Inception uses full FP32
with autocast and TF32 disabled. Distributed workers extract features, which are
gathered before the original NumPy mean/covariance calculation on rank zero.

The real reference was generated by the official `compute_statistics_dataset`
using all 126,227 unique training images without padding or dropping. Its SHA256
is `ad3b5a341831d877110afdff37d6fff4be855612053e3eb9e2e7119f5593d364`.
Source hashes and reference provenance are in `official-fid-provenance.json`.
Evaluation isolates its RNG from training. Changing sampling batch layout changes
the mapping of random draws, so this new run's stream is fixed after benchmarking.

The subsequent [FID/cache transform audit](../outputs/church-stage2-released-recipe-20260921/fid-cache-transform-audit.json)
verified the active physical token cache, its native encoder-cache ancestry, and
the real-reference statistics against the same data-protocol hash. Both actual
frozen transform factories use the released LSUN branch: RGB conversion, bilinear
short-side resize to 256, center crop to 256 by 256, tensor conversion, and
normalization to [-1, 1]. There is no random crop or horizontal flip. The complete
ordered population of 126,227 unique training-image keys matches, and all 36 fresh
pixel probes match both transform paths and their historical hashes bit for bit.
The 300 validation images are used only for loss monitoring. Official FID converts
real and generated images to [0, 1] before the common Inception resize to 299 by
299. The audit passed without rebuilding the reference or restarting training.

Working full checkpoints use local `/tmp/laser-church-released-scratch-20260921`
storage because workspace quota previously caused checkpoint-write failures.
W&B receives immutable full-state copies with remote size and digest verification.
The bounded uploader keeps one active upload plus the newest pending snapshot and
drains before normal completion. A host loss before a new upload commits can lose
the latest local progress; the previous committed artifact remains recoverable.

Sources: [paper appendix A.3](https://arxiv.org/html/2203.01941#A3),
[released Church config](https://github.com/kakaobrain/rq-vae-transformer/blob/main/configs/lsun-church/stage2/lsun-church256-sqgan-8x8x4-350M-simp.yaml),
[official FID source](https://github.com/kakaobrain/rq-vae-transformer/blob/341395e562ac347f5eb62db9f5f08b9f2cc42a60/rqvae/metrics/fid.py).
