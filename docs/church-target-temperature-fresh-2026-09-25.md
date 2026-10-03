# Fresh Church transformer training, 2026-09-25

Two new transformer runs start from identical random weights, without any earlier
stage-2 checkpoint, optimizer state, or scheduler state. The existing frozen
tokenizer is retained. This implements the request to train from scratch with the
current fixes and the lower coefficient target temperature.

| Run | Coefficient target temperature | GPUs | Initial microbatch × accumulation × ranks |
|---|---:|---:|---:|
| [Treatment](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-stochomp-fresh-t00625-k700ct09-b2048-20260925) | 0.0625 | 2 H200 | 256 × 4 × 2 |
| [Control](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-stochomp-fresh-t025-k700ct09-control-b2048-20260925) | 0.25 | 1 H200 | 256 × 8 × 1 |

Both use global batch 2,048, 126,227 training images, 62 updates per epoch,
300 epochs, and a fresh 18,600-update cosine schedule from learning rate 0.0005
to zero. Both use the previously selected stochastic-model sampler: atom top-k
700, atom temperature 1, coefficient top-p 0.85, and coefficient sampling
temperature 0.9. Training target temperature and sampling temperature are
different settings. There is no latent rescaling or covariance transport.

The earlier run was checkpointed and stopped at epoch 65, optimizer step 4,030.
Its full local snapshot is
`/mnt/laser-church/target-temperature-fresh-20260925/original-stopped-epoch65.pt`.
Neither new run loads it. Earlier comparisons and their limitations are in
[the continuation report](church-stochomp-h200-continuation-2026-09-25.md).
The matched-reference RQTransformer target remains FID50k 8.001728; these fresh
runs have not established an improvement over it.

## Initialization and checks

The complete initial state SHA256 is identical in both runs:
`13c80eae970db877341dd190c4b0adbe7e4e6c37c0273186d6cfeaf9aaf1ffe0`.
Every rank records an empty optimizer and scheduler step zero before training.
The initialization uses seed zero and the model's normal initialization, with
404,738,048 parameters. The tokenizer SHA256 remains
`762c51a10267ed6fa55709ff0d6cf997940d21056b16c7a0a0399b3ebf93868d`.

The retained joint model predicts each atom from preceding pairs and its
coefficient from the preceding pairs plus the current atom. Full pair
autoregression, two coefficient micro-transformer layers, and depth-specific
coefficient heads are enabled. The prelaunch audit checked the relevant causal
and numerical behavior; it is not proof that every possible implementation bug
has been eliminated.

- The compound autoregression and stochastic-cache test suites passed all
  11 tests.
- Exact finite-bank enumeration at target temperatures 0.25 and 0.0625 matched
  24 conditional distributions per temperature, with maximum errors below
  8e-9. Sparse and dense losses and gradients agree; zero-mass masked targets
  produce finite losses.
- Current and future coefficients do not leak into earlier predictions. Six
  causal tests also passed with the production attention implementation enabled.
  A four-depth, two-micro-layer cached-sampling check matched teacher-forced
  logits within 2.39e-7.
- Both distributed layouts cover all 126,227 images exactly once per epoch,
  including the final partial batch. Simulated accumulated gradients match the
  global-batch mean within 1.2e-16.

## Teacher bank

Reducing coefficient target temperature requires new normalization constants.
All 517,025,792 coefficient log-partitions were recomputed over the actual 2,048
bins at temperature 0.0625. Atom supports, continuous normalized coefficients,
and labels are bitwise unchanged. Independent FP64 checks of 4,096 entries
found maximum absolute partition error 5.24e-7.

The new bank SHA256 is
`8ec80c5b10ea4fb14d8d7bac0a41827cc0c58beaf0d2e6f9ef5652a3d46e6976`.
It is committed in W&B as
`helloimlixin-rutgers/laser/church-stochomp-normt00625-soft-bank-20260925:v0`,
with remote file size and digest verified. The control uses the authenticated
original 0.25 bank. Conditional soft atom targets use the corresponding bank
and coefficient temperature in each arm.

## Monitoring and interpretation

Live files are under
`/mnt/laser-church/target-temperature-fresh-20260925/`, with separate `t00625`
and `t025-control` training and checkpoint directories. `status.json` records
startup validation; each arm has `train.log`, `train/metrics.jsonl`, and full
`train/checkpoints/last.pt` and `best.pt` checkpoints. Detached training processes
continue after the startup verifier exits. Each run logs generated samples,
held-out likelihood diagnostics, official FID50k, and selected checkpoints to W&B.

FID uses the unchanged official FP32 Inception evaluator and shared real
reference. Compare the arms at equal epochs or optimizer updates. Their different
rank layouts mean neither training random draws nor online FID sample sets are
bitwise paired. A final comparison should evaluate selected checkpoints with
identical one-GPU batches and independent paired seeds; that confirmation has
not yet been run. Cross-entropy and target entropy also differ with the teacher
temperature, so lower raw loss alone is not evidence of better generation.

Reproduction scripts, plans, audit records, and initialization proofs are copied
to `outputs/church-target-temperature-fresh-20260925/`. Both complete runtime
source archives include the active checkpoint and held-out-monitor helpers. The
control's first training-provenance archive preceded inclusion of those two
helpers; the combined validation archive supplies the complete sources.

## Startup verification result

Both first-epoch checkpoints passed validation: epoch 1, optimizer step 62,
517 optimizer parameter states at step 62, scheduler step 62, finite model
weights, correct target temperatures and sampling settings, and the expected
number of saved rank RNG states. Both trainers continue independently.
The combined source and validation artifact is published by the
[validation run](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-stochomp-fresh-targettemp-validation-20260925).

## GPU throughput tuning

The user requested use of all GPU capacity and larger batches for throughput.
All three H200s remain assigned: two to the treatment and one to the control.
Both runs resumed their own full checkpoints; neither was reinitialized.
Global batch 2,048, optimizer state, learning-rate schedule and target settings
are retained. The saved rank random streams are restored, but changing
microbatch grouping changes subsequent random draws and floating-point order.

| Arm | Training batch/GPU | Accumulation | Generation batch/GPU | Training images/s, before → after | Generation images/s, before → after |
|---|---:|---:|---:|---:|---:|
| t00625 | 512 | 2 | 4096 | 1339.2 → 1376.0 | 368.7 → 411.1 |
| t025-control | 512 | 4 | 4096 | 677.0 → 693.4 | 184.7 → 204.3 |

Training timings use 18 optimizer steps per candidate, discarding the first five
steps from the mean. Generation timings include autoregressive sampling,
decoder work and official FP32 Inception extraction; the slower rank determines
aggregate throughput. These are short local benchmarks, not end-to-end epoch
speedups. The 8,192-image generation test exceeded memory on at least one rank
and was excluded from the treatment selection.

The control generation benchmark runs in a separate process loading its
saved checkpoint; the treatment benchmark shares its warmed training
process. Both compare batch sizes within their own fixed model and setup.

All 62 global batches contain the same image sets before and after the
accumulation change. Simulated full and final-batch mean gradients agree
within 5.6e-17. Unsafe mid-epoch and effective-batch-size migrations are rejected.
The official FID implementation, Inception feature batches of 64, reference,
50,000-image count and sampler settings are retained. Larger generation
batches change random-number assignment, so final model comparisons still
require common evaluation batches and paired seeds.

t00625 completed a full epoch after resizing: epoch 7, step 434. Optimizer/scheduler continuity, finite weights, expected rank RNG states, and both batch settings passed checkpoint validation.

t025-control completed a full epoch after resizing: epoch 6, step 372. Optimizer/scheduler continuity, finite weights, expected rank RNG states, and both batch settings passed checkpoint validation.

Tuning scripts, benchmark records and source archives are in
`outputs/church-target-temperature-fresh-20260925/throughput/` and the
[throughput validation run](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-stochomp-fresh-throughput-20260925).
