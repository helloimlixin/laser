# Church full-vocabulary OMP experiment

The 16-trajectory bank is replaced by fresh stochastic OMP from frozen encoder
latents on every training visit. Each of four support decisions evaluates all
16,384 dictionary atoms, excluding atoms already selected at that site. Its
target is the full FP32 distribution proportional to
`exp(residual_correlation² / 0.0625)`. There is no training top-k, support-bank
lookup, or cap on the number of distinct trajectories across visits.

OMP still jointly refits the four physical dictionary coefficients. Independent
coefficient IDs are sampled from the existing complete 2,048-bin normalized
kernel at temperature 0.03125; the same distributions provide soft labels. The
frozen Church tokenizer, dictionary, coefficient scales, and image decoder are
unchanged. Encoder latents use all 126,227 original training images, original
LMDB key order, Resize256/CenterCrop256, FP32 encoding, and TF32 disabled.

The model now predicts `a0,a1,a2,a3,c0,c1,c2,c3` within each spatial site. This is
necessary to use OMP selection probabilities as exact causal atom targets:
previously emitted final-refit coefficients would contain information about
later support choices, so the OMP support-only distribution would not equal the
teacher conditional on that pair history. The old finite-bank posterior was
consistent with its own pair order; it was an approximation to a richer teacher,
not a demonstrated causal-attention bug.

The 24-layer spatial transformer and four-layer depth transformer are retained,
with eight depth events instead of four. Four extra positional vectors bring
the parameter count to 404,742,144. Completed spatial sites still use the same
learned physical pair embeddings. The coefficient micro-transformer and four
coefficient classifiers are retained. Training and generation use the same
supports-first order. Atom masks exclude used supports in both paths.

The experiment starts from fresh stage-2 weights. Its shared backbone matches
the earlier seed-0 initializer; the extra positions are newly initialized. It
keeps global batch 2,048, 62 optimizer updates per epoch, 300 epochs / 18,600
updates, AdamW LR 0.0005 cosine to zero, no warmup, betas (0.9,0.95), weight decay
0.0001, gradient clipping 1, residual dropout 0.2, and atom loss weight 1.5.
It is a controlled teacher-plus-factorization experiment, not the same
architecture or random training trajectory as the bank baseline.

Official FID50k uses the previously verified FP32 feature pipeline and reference
SHA256 `809489d8316b9e6eb9dc3bc021b6d602f4b6d816cc80621c6b9c189a9253a7f6`.
Evaluation runs at epoch 1 then every five epochs to reduce evaluation overhead.
The generation settings remain atom temperature 1 / top-k 700 / top-p 1 and
coefficient temperature 0.9 / top-p 0.85. Full recovery checkpoints are saved
every 62 updates, with last/best snapshots uploaded and remotely verified.
The fixed-history coefficient monitor now computes soft-target KL; the old
monitor's `coeff_kl` was actually hard coefficient NLL and is not comparable.

Validation before launch:

- 17 CPU tests passed in the frozen Python-3.12-compatible runtime. They check
  full-vocabulary probabilities against independent residual calculations,
  final least-squares coefficients, RNG replay and fresh draws, the eight-event
  causal order, cached/dense prediction agreement with unknown future tokens,
  soft-loss gradients, and generation with distinct supports.
- On 64 real Church images, 16.85% of freshly sampled complete trajectories were
  absent from that site's old 16-slot bank. Continuous latent reconstruction
  error was 2.44313, versus 2.44328 for the old bank audit and 2.39388 for greedy
  OMP. These are exploratory probe measurements, not FID improvements.
- Full-teacher support entropies were [0.0360,0.1292,0.3783,1.2279] nats.
  The first choice remains sharp at this OMP temperature. Differences from the
  old causal bank entropy reflect both finite sampling and changed conditioning.
- The largest normal-equation residual on the probe was 1.06e-5. Coefficient
  target entropy remained 5.17187 nats, matching the unchanged kernel.

Implementation: `src/training/full_vocab_omp.py`, the optional online-OMP branch
of `src/training/rqtransformer.py`, and optional probability output from
`src/stochastic_compound.py`. Existing training modes retain their defaults.

Experiment files, frozen runtime, plan, measurements, and launch/resume entry
point are in `outputs/church-fullvocab-omp-20260927/`. Run `launch.py --resume`
there to resume its own checkpoint. A standalone checkpoint loader must attach
`SupportsFirstOMPTransformer` before loading this experiment's state dict; the
saved `online_omp_temperature` and `compound_event_order` identify the format.

The previous lower-noise run and checkpoints are retained under
`outputs/church-stochomp-fresh-t003125-20260927/`; its checkpoint was inspected at
update 1,364 before stopping the launcher. No previous run's files were deleted.
Whether this change beats the matched 9.6012-FID baseline remains an empirical
question; removing the finite bank does not establish better FID by itself.

## Throughput and initial launch

On five idle H100 NVL GPUs, a 137-image microbatch measured 972 images/s and
51.34 GiB peak allocated memory; a 205-image microbatch measured 1,164 images/s
and 72.68 GiB. Both completed training, generation, and image decoding. The
205-image setting was selected with two accumulation steps. Actual accumulated
training reached approximately 1,299 images/s at updates 20–40. These measurements
exclude FID and checkpoint time; the online teacher and eight-event head do more
work than the old bank-based four-event model.

Checkpoint upload staging now links immutable local snapshots, avoiding repeated
multi-gigabyte reads from workspace storage. Persistent full checkpoints still
complete before the upload is queued. Local files use atomic replacement, so
active upload links retain their original bytes. A separate check verified
immutability across replacement and hydration of a missing local best snapshot
on resume. Last and best states remain stored in the workspace and uploaded.

The full latent cache SHA256 is
`bb70e944e531d75a95b2bbadb3092af996ffd7d78488da82e9a572fb291a9fdd`.
Both local and persistent copies were verified. The 64-image independent probe
aligned to the same source rows with maximum encoder-latent difference 3.34e-6.
All five ranks verified fresh optimizer/scheduler state and matching coefficient
kernels during startup.

[Live W&B run](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-fullvocab-omp-t003125-b2048-h100x5-20260927).

The first epoch completed all 62 updates. Its full checkpoint validated finite
model parameters and all 517 optimizer states, scheduler step 62, all five rank
RNG states, and the eight-position head. The first FID50k evaluation was still
running at this initial report.

## FID decode OOM repair

The first attempt ran out of CUDA memory during the epoch-1 FID decoder. Token
generation used 1,000 images per rank, but the active trainer had lost the prior
64-image decoder chunking when the online-OMP changes were integrated. The
decoder attempted a 31.25-GiB allocation. The earlier eight-image generation
smoke test did not cover this production-batch path.

The trainer now decodes at most 64 images per call and streams each chunk to
the metric accumulator, preserving the 1,000-image AR generation batch, sample
order, FP32 pixels/features, 500-image feature batches, and official FID50k
reference. It also restores the prior workspace-compatible checkpoint copy
fallback. Three regression tests passed: compound and sparse generation with
a 1,000-image batch plus a remainder, and snapshot fallback without filesystem
metadata copying.

Training resumes the same run from the verified update-62 checkpoint, retaining
weights, optimizer, schedule, and all-rank RNG states. No optimizer updates were
lost. Full FID completion and resumed training are being checked. The failed
source manifest/log and repair record are retained in `oom-repair/`; the updated
training source was republished as a new provenance artifact version.

The repaired production path completed all 50,000 samples in 195.89
seconds. Epoch-1 FID was 159.8085; this early score does not establish final
quality. Peak allocated/reserved GPU memory was 46.10/51.85 GiB on every rank.
Both persistent last and best checkpoints were saved. Training resumed through
update 100 with finite loss/gradients and warm throughput of
approximately 1281 images/s. Training microbatch 205,
global batch 2,048, optimizer, learning-rate schedule, and noise stayed fixed.
