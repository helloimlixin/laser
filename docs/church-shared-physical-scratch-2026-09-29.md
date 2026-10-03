# LSUN Church stage two from scratch with shared physical coefficients

The user clarified that the new representation should be trained from scratch.
This experiment initializes every stage-two parameter anew and creates a fresh
AdamW optimizer and cosine schedule. It retains the original compound RQ
architecture: 404,738,048 parameters across 517 parameter tensors. The Church
stage-one tokenizer, dictionary and decoder remain frozen.

All four sparse depths use exactly the same 2,048 physical coefficient centers
over [-22.9870662689209, 22.9870662689209], with unit scales. The original
16-variant bank retains its atom supports and continuous physical coefficients.
No support or coefficient refitting is introduced. Details and validation of the
representation are in [the shared-grid note](church-shared-physical-grid-2026-09-29.md).

Training uses the existing fresh paired procedural teacher. It conditions on the
complete continuous sparse vector, draws an atom/sign and its conditional
coefficient, and commits complete pairs in the final ascending sweep. It uses
the full eligible atom vocabulary and all shared coefficient bins. Its physical
temperature stays at 0.4204482076268573, with one warmup sweep. This finite
procedure is not claimed to produce equilibrium Gibbs samples.

The run uses all eight H100 80 GB GPUs, microbatch 128 and accumulation two, for
global batch 2,048. The final update in each epoch contains 1,299 images with
sample-weighted gradients, so every one of the 126,227 images is visited exactly
once per epoch. The requested 300 epochs are 18,600 updates.

The fresh Church recipe uses AdamW with learning rate 0.0005, betas (0.9,0.95),
epsilon 1e-8, weight decay 1e-4, gradient clipping 1, and residual dropout 0.2.
The learning rate follows cosine decay from update zero through 18,600, with
zero warmup, matching the authenticated earlier fresh recipe. No continuation
learning-rate multiplier, old Adam moments, or checkpoint initialization is used.
The atom/coefficient cross-entropy weight ratio remains 1.5:1. No geometry,
regression, contrastive, or reward loss is added. Model computation uses the
original outer BF16 autocast; teacher probabilities and losses remain FP32.

The audited conditional ancestral sampling adapter preserves the RQ architecture,
cache and event order. Generation uses all eligible atoms and all coefficient
bins at temperature 1, without top-k or nucleus truncation. These numerical
settings are our explicit policy, not a claim about undocumented DCTransformer
settings. Best-checkpoint selection uses only this one fixed generation policy.
The old checkpoint's matched untruncated control FID is 11.169160294; its 9.493208036
score used different generation filters and is not a matched-policy comparison.

Sixty-four preview images are sampled every 200 updates. Official Church 50k FID,
including separate mean and covariance terms, is evaluated every 10 epochs
(620 updates), using the same eight logical seeds and FP32 decoder/Inception
protocol as the control. There is no random-model FID evaluation before training.

Local full checkpoints are saved every 200 updates and around FID boundaries.
Full LAST states are uploaded to W&B and durable storage at step 200, every FID
boundary and completion. A BEST-FID state is selected and uploaded only after a
measured FID exists for this run. Checkpoints include model, optimizer, schedule,
data cursor, independent per-rank model/teacher RNG streams, codec identity and
source provenance. Recovery accepts only this run's own compatible checkpoints.

The isolated runtime is `/tmp/laser-church-shared-physical-scratch-20260929`.
Durable records are in
[the output directory](../outputs/church-shared-physical-scratch-20260929/).
Training and previews are recorded on
[W&B](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-shared-physical-scratch-20260929).
