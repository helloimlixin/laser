This trial uses the frozen CelebA-HQ tokenizer from
[the original LASER VAR experiment](https://wandb.ai/helloimlixin-rutgers/laser/runs/celebahq256-var-laser-poc-20260916)
and trains a new compound-pair VAR prior. The requested references are
[the repaired CelebA-HQ VAR run](https://wandb.ai/helloimlixin-rutgers/laser/runs/celebahq256-var-laser-fix-20260917)
and [the Church stochastic compound continuation](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-laser-stochastic-best70-lr2e5-30ep-h200x5-20260921).

The new run is
[celebahq256-var-stochastic-compound-20260921](https://wandb.ai/helloimlixin-rutgers/laser/runs/celebahq256-var-stochastic-compound-20260921).
The experiment directory is `outputs/celebahq256-var-compound-20260921/`.

## Plateau diagnosis

The reference run's validation joint NLL reached its minimum, **9.855763**, at
epoch 29, and worsened to **9.946092** at epoch 39. At the end, training atom
NLL was 5.9469 versus validation 6.6423, and training coefficient NLL was
3.1174 versus validation 3.3038. This supports a generalization plateau;
it does not identify a single proven architectural cause.

Its FID series mixes sample counts. Epochs 20, 30, and 39 used 128 generated
and 128 real images and scored 102.29, 103.69, and 109.10. Epoch 25 used
2,000 images and scored **52.35**. These estimates must not be compared as
one consistent learning curve. The old 128-image score is not a FID50k result.

The local source predates the reference run's advertised
`context_atom_bilinear_v1` coefficient fix. Its additive linear coefficient
head has no interaction between the current atom and contextual features.
The reference configuration explicitly says that issue was already repaired;
it is therefore **not established as the cause of that run's plateau**.
The reference run's W&B files contain no trained checkpoint or full VAR source,
and its saved Git diff does not include the untracked VAR implementation.
Historical diagnosis uses the online configuration/history and the locally
available tokenizer, rather than claiming a replay of the missing prior.

Two concrete runtime defects were also found and fixed: real-image FID
statistics used distributed collectives inside a rank-zero-only branch, which
would hang multiple GPUs; and NumPy evaluation indices were passed directly
to a Hugging Face dataset that requires Python integers. Neither explains the
earlier single-GPU run's generation quality.

## Compound formulation and attempted remedy

Each event follows
`p(atom_d | past_pairs, past_scales) * p(coeff_d | atom_d, past_pairs, past_scales)`.
The original VAR-d16 spatial body retains its 680 positions, ten scales, and
two sparse pairs per site. It does not flatten sparse depth into the spatial
sequence. A local width-256, two-layer causal head consumes shifted completed
pairs. Each pair retains its atom vector, learned coefficient-ID embedding,
and physical contribution. Two short attention layers refine coefficient
predictions using the selected atom; coefficient classifiers are depth-specific.
Short attention is computed in FP32 during BF16 training.

The training objective follows the Church weighting:
`(1.5 * atom CE + coefficient soft CE) / 2.5`.
Validation reports separate hard atom/coefficient NLLs on fixed deterministic
targets so the training soft objective is not mislabeled as joint NLL.
The new stochastic targets and dropout are an attempt to improve generalization;
improved generation quality still requires the actual training/evaluation run.

The CelebA-HQ tokenizer's existing 4,096 atoms, 257 asinh coefficient bins per
scale, encoder, decoder, dictionary, and residual convolutions stay frozen.
The Church tokenizer's four-depth 8x8 layout, 2,048 Lloyd–Max centers, and
physical temperatures are not transferable unchanged to this different codec.
This is a transfer of its compound formulation and stochastic supervision,
not an exact reproduction of its architecture or discretization.

Supports were initially sampled online from squared residual correlations, followed by
least-squares coefficient refitting. Coefficient targets use a softmax over
squared physical distances, and teacher coefficient IDs are sampled from those
same targets. Later spatial scales are **recomputed from the actually sampled
earlier scales**. Independently mixing Church-style per-site cached variants
across VAR scales would invalidate these residual targets. At the user's
request, production now uses a precomputed bank of complete multiscale
trajectories, described below.

Calibration used 64 fixed training and 64 fixed validation images. The selected
temperatures change 87.36% of validation spatial supports while increasing
latent reconstruction MSE by only **0.421%** and LPIPS by **0.001343**. The next
larger candidate increased latent MSE by 5.10% and was rejected by the 5% gate.
The unchanged tokenizer baseline on this subset has LPIPS 0.18652 and
PSNR 22.785 dB. Exact temperatures, image indices, and checkpoint SHA256 are
recorded in `audit/calibration.json`.

## Training and checks

Production starts its prior from random initialization and uses the source
tokenizer at epoch 50 / update 10,900. Preflight weights are isolated. Training
initially used two H200 GPUs, batch 64 per GPU, accumulation 2, and effective
batch 256. At the user's request to use all GPUs, the same run resumes with
three H200s, batch 128 per GPU, no accumulation, and effective batch 384.
The budget is 50 epochs with peak LR 3e-4, one warmup epoch, and the original
linear decay to 3e-5, anchored to completed epoch fraction through the layout
change. Atom sampling uses top-k 250 and top-p 1; coefficient top-p is 1;
CFG remains 1.5.

Every scheduled FID uses **2,000 generated and all 2,000 validation images**,
at epoch 1 and every five epochs. Evaluation restores training RNG afterward.
This is a consistently sized diagnostic, not a published FID50k comparison.
The full latest, best-validation, and best-FID checkpoints retain optimizer
state and all rank RNG states. Checkpoints are uploaded at FID evaluations
and completion. Resume checks tokenizer identity and training configuration;
an explicit execution flag permits batch/GPU-layout changes while rejecting
changes to the model, objective, data, or learning-rate recipe.

The focused regression suite passes 24 tests, including compound causality,
atom/context interactions, preservation of the original deterministic codec,
stochastic trajectory roundtrips, full soft-objective gradients, and cached
versus teacher-forced logits. Two unrelated archived ImageNet CLI assertions
fail identically when loading the unmodified HEAD CLI; neither recipe was
changed for this experiment.

The full-model preflight first overfits sixteen images for 100 updates, then
exercises stochastic targets, image encoding, the production microbatch and
gradient accumulation, validation, distributed generation/FID, and full-state
checkpoint serialization. `preflight/complete.json` must exist before the
production driver starts. `runtime/`, `source-manifest.json`, and
`requirements-runtime.txt` preserve the executed source and environment.
`train/complete.json` is written only after the full training budget and final
checkpoint upload finish; a launch alone is not completion.

Launch verification succeeded. The production run is online and advanced past
update 40 at approximately 198 images/second, with finite gradients and about
55.4 GiB peak allocated memory per GPU. Objective decreased from 7.228 at the
first update to 6.924 at update 40 during warmup. The first full checkpoint
contains 274 optimizer states and both rank RNG states; the source/configuration
artifact is committed in W&B. `launch-verification.json` records the online
and local checks. The final preflight completed 102 updates, reducing the
five-update-average overfit objective from 7.0069 to 0.3575 before the two
stochastic production-path updates. No claim of improved generation FID is
made before production evaluation.

## Throughput investigation and upload correction

The initial driver mistakenly waited for checkpoint artifact uploads on rank
zero while rank one waited at a distributed barrier. After the epoch-1 and
epoch-5 FID evaluations, the delays before the next training log were 821 and
730 seconds. These network waits consumed over 25 minutes of the first 56
minutes. Rank one's 100% GPU utilization during such a wait is not productive
model computation; its memory utilization and power were correspondingly low.

An independent benchmark on the third H200 measured, per batch of 64 images:

| Component | Seconds | Approximate share |
| --- | ---: | ---: |
| Frozen image encoder | 0.111 | 17% |
| Stochastic multiscale OMP | 0.065 | 10% |
| Prior forward/backward | 0.463 | 72% |
| Optimizer | 0.005 | 1% |

Active production throughput was stable around 198 images/second across two
GPUs, consistent with that benchmark. Increasing the per-GPU batch from 64 to
128 improved standalone throughput by only 3.5% and raised peak memory from
53.2 to 101.6 GiB. Batch 160 reached 103.7 images/second and 126 GiB in isolation,
but failed the three-rank memory check due to additional memory use and allocator
fragmentation. Batch 128 was selected for the all-GPU run, with expandable
allocator segments. This changes the global batch from 256 to 384.

The corrected uploader has separate snapshot and network workers. Submission
opens handles to the atomically saved checkpoint inodes and immediately returns.
The snapshot worker copies immutable files to local `/tmp` storage; the network
worker uploads them while training continues. There is one active network
transfer and one latest pending snapshot, with superseded pending transfers
coalesced. Upload failures are propagated, and completion drains the final
transfer. Local staging avoids adding large hardlinks under the shared
workspace quota. Tests verify snapshot immutability, coalescing, cleanup, and
failure propagation; nine focused integration tests passed in frozen source.

The original runtime and manifest were archived under
`restarts/async-upload-20260921/`. The upload-blocked epoch-10 checkpoint at
step 1,090 was verified with all 274 optimizer states and both RNG states,
and copied to local backup before restarting the same run. No model, loss,
data, sampling, batch, or LR-schedule settings changed. At that point the
2,000-sample FID was 45.3004. `throughput-profile.json` records the benchmark.

## All-GPU restart and comparison with Church

The user then requested all GPUs and maximum speed. A later workspace write
failure left rank zero outside the training loop and rank one waiting in a
collective. The last valid saved state was epoch 11, batch 2, step 1,200;
logged but unsaved progress through step 1,260 is replayed. All 274 optimizer
states and the saved model were retained. Mapping the partial epoch to the
new batch replays 256 images instead of dropping data. The third rank gets
an independent deterministic RNG stream; existing rank RNG states are restored.
The LR schedule is anchored to the saved fractional epoch and the original
109 updates/epoch, so changing the number of updates per epoch does not cause
an LR jump or premature decay. Three regression tests cover migration,
schedule continuity, ordinary resumption, and rejection of objective changes.

Training checkpoints now write atomically to local
`/tmp/laser-var-checkpoints/celebahq256-var-compound-20260921/train/`.
`checkpoint-mirror.py` copies immutable generations asynchronously to the
original durable `train/` directory, retaining the previous complete copy on
quota errors and retrying. Its receipt is `mirror-status.json` in the local
checkpoint directory. W&B checkpoint uploads remain asynchronous. Both
background paths can lag local training; local storage alone is ephemeral.
W&B's active log directory is also local. Redundant preflight weights were
moved to local archival storage, releasing 3.6 GiB of shared quota.

The requested comparison run,
[Church rawcoeff90](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-laser-ft3best-stochastic-rawcoeff90-h200x5-20260921),
reports approximately 3,025 training images/second on five H200s with batch
192 per GPU and a prebuilt token cache. Its tokenizer/prior layout is 8x8x4:
64 spatial positions and 256 compound pairs per image. This VAR uses 680
spatial positions and 1,360 compound pairs: 10.625 times the spatial positions
and 5.3125 times the pair targets. It also encodes images and samples complete
multiscale trajectories online (about 27% of the measured training step).
The two configurations therefore have substantially different training work
despite similar parameter-count classes. VAR's reduction in sequential
generation steps does not imply that training this longer spatial sequence
will outrun a teacher-forced 64-position Church prior.

The resumed three-GPU run measured 305.1–305.4 images/second at updates
1,220, 1,240, and 1,260, versus approximately 197 previously: a 55% increase.
Peak allocated memory was 102.7 GiB per GPU. During active training, all
three devices measured 100% utilization and roughly 590–605 W. The full
epoch-12 checkpoint and its durable mirror were verified with three RNG
states, 274 optimizer states, and the original schedule reference. See
`all-gpu-verification.json` for the checkpoint and throughput receipt.

## Precomputed tokens and short-attention optimization

The second speed change resumes the saved epoch-22, batch-12, step-2,004 state.
The checkpoint is backed up as `/tmp/laser-var-pre-cache-checkpoint.pt`; no
logged training updates are discarded for this transition. The optimizer,
global batch 384, tokenizer, scale schedule, loss, and LR schedule are retained.

`scripts/tools/prepare_var_compound_cache.py` builds complete stochastic
trajectories using all three H200s. The train bank contains every one of the
28,000 images, both horizontal-flip views, and 16 trajectories per view.
Validation stores deterministic trajectories for all 2,000 images. Atom and
coefficient IDs are uint16; physical least-squares coefficients are FP32, so
the original calibrated soft targets can be restored without quantization.
Training selects one complete trajectory per image visit. View choices match
the original seeded horizontal-flip policy, and trajectory choices are
deterministic functions of seed, epoch, and image index for reliable resume.
The finite 16-variant bank replaces fresh online resampling; this change is
explicitly recorded in cache provenance.

The cache lives in
`/tmp/laser-var-token-cache/celebahq256-var-compound-20260921/`. Its manifest
records tokenizer and calibration identity, dataset fingerprints, shapes,
sampling seeds, and SHA256 for every array. Training verifies those files
before using the cache and stores the manifest hash in each full checkpoint.
The cache is submitted to W&B as a separate dataset artifact asynchronously;
local storage is ephemeral until that upload commits. Each build rank verifies
exact roundtrips of cached IDs, contexts, reconstructed latent maps, and soft
target probabilities, and checks that different variants change supports.
Training uses cached IDs and values; it no longer decodes images, runs the
image encoder, or runs OMP for each optimizer update. It reconstructs teacher
contexts from the cached codes using the frozen residual modules.

Profiling also found generic tiled attention kernels in the two-token causal
blocks. For two tokens, the first output is exactly the first value, while the
second is a sigmoid-weighted mixture of both values. The implementation now
uses this equivalent FP32 expression, keeping all weights and parameter names.
Longer local sequences retain the standard attention fallback. Tests compare
outputs and gradients to reference attention, including one-, two-, and
four-token sequences. An alternating full-prior batch-32 comparison measured
an 18.4% improvement from this attention change; the cache builder was also
active, so this is a relative benchmark, not production throughput.

The 680 positions are the sum of the ten grid areas. Compared with Church's
64 positions, a 16x16 final grid first accounts for a factor of four, and the
multiscale pyramid adds a factor of 2.65625. Reducing to `(1,2,4,8,13,16)` gives
510 positions; `(1,2,4,8,16)` gives 341. A fixed 64-image validation audit
re-encoded residuals at these schedules, preserving each retained scale's
original residual kernel and coefficient grid:

| Positions | LPIPS | Relative latent MSE | PSNR |
| --- | ---: | ---: | ---: |
| 680 | 0.18652 | 1.000 | 22.785 |
| 510 | 0.18921 | 1.099 | 22.760 |
| 341 | 0.19617 | 1.377 | 22.564 |

Both reduced schedules exceed the existing 5% latent-MSE reconstruction gate;
the 341-position schedule also exceeds the +0.005 LPIPS gate. The current
run therefore keeps 680 positions. A fewer-scale experiment would need
tokenizer adaptation and evaluation as a separate training configuration.
`reduced-scale-audit.json`, `kernel-profile.json`, `short-attention-benchmark.json`,
and the preserved scripts under `speed-audit/` contain the evidence.

The completed cache occupies 9,770,481,024 bytes and was built in 332 seconds.
After resumption, production measured 485.6 and 484.4 images/second at updates
2,020 and 2,040, versus 305 previously: approximately 59% faster. All three
GPUs measured 100% utilization and 654–671 W during active training, with
100.4 GiB peak allocated memory. `cache-training-verification.json` records
the resumed full checkpoint, its cache identity, and durable mirror check.

A paired 341-versus-680-position benchmark subsequently measured a 1.89x
step-time improvement with cached tokens, context/target reconstruction,
forward/backward, gradient clipping, and fused AdamW included. It used batch
16 on one H200 in spare memory while production continued, with alternating
measurement order. Applying that ratio to the observed 484.8 images/second
production baseline gives approximately 916 images/second across three H200s.
This is a throughput estimate, not a three-GPU 341-position measurement;
production batch size and distributed overhead can change the ratio. Exact
timings are in `341-position-throughput.json`, and its script is preserved as
`speed-audit/scale-throughput.py`. Production remains on the 680-position model.

Sampling was measured separately from training on one H200 while the live
three-GPU training job continued. Both implementations used KV caching and
their run's sampling configuration: VAR used BF16 and CFG 1.5; Church RQ used
its original AMP sampling and FP32 decoder with TF32 enabled. The RQ model
was loaded from the referenced rawcoeff90 run's checkpoint and frozen runtime,
including its short-attention patch. VAR used the current CelebA checkpoint.
The reduced VAR retained the appropriate positional embeddings, coefficient
grids, and residual kernels; its generated quality has not been validated.

Each result is the median of three samples after a warmup. Decoder batches
were capped at 16 for all models to fit alongside training. FID, feature
extraction, and checkpoint I/O were excluded.

| Model | Batch-1 latency, decoded | Batch-16 decoded images/s | Batch-64 prior images/s | Batch-64 decoded images/s |
| --- | ---: | ---: | ---: | ---: |
| Church compound RQ, 64 positions | 2.236 s | 6.70 | 26.77 | 23.70 |
| CelebA compound VAR, 680 positions | 0.157 s | 42.53 | 90.31 | 56.08 |
| Reduced compound VAR, 341 positions | 0.084 s | 68.31 | 195.77 | 85.02 |

At matched batch 64, decoded sampling is 2.37x faster for current VAR and
3.59x faster for the reduced-scale timing experiment than for this RQ run.
The advantage is larger at batch 1: 14.2x and 26.6x respectively. These are
comparisons of the configured implementations, not controlled architecture
ablations: the datasets, tokenizer, parameter counts (313M versus 405M),
coefficient vocabularies, sparsity (2 versus 4), and guidance differ. Shared
GPU contention also means these are not dedicated-GPU peak throughput.

The current VAR makes 10 sequential spatial-backbone passes, reduced VAR
makes 5, and RQ makes 64. VAR predicts a whole scale's sites together. Counting
atom and coefficient decisions separately gives 40, 20, and 512 sequential
categorical sampling rounds, respectively. A precomputed training-token
cache does not replace these generation decisions; KV caching helps both
implementations. Church's logged roughly 640 images/s FID pipeline used five
GPUs and batch 512 per GPU and is not directly comparable to this batch-64
single-GPU audit. Details are in `sampling-var.json`, `sampling-rq.json`, and
the scripts `speed-audit/sampling-var.py` and `speed-audit/sampling-rq.py`.
