# Fresh relaunch with balanced coefficient conditioning

Retired on September 23 at completed epoch 3 / update 186, best FID50k 148.17194,
after the user selected explicit dictionary-vector attention conditioning.
The evaluated latest/best states are pinned under this run's `retired/` directory
and verified online in selected-checkpoints artifact v4 (5,375,859,554 bytes
per full state, matching remote digests). The fresh successor is
[the dictionary-attention run](church-dictionary-attention-relaunch-2026-09-23.md).

Run: [church-laser-compound-balanced-fullar-scratch300-h200x1-20260923](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-laser-compound-balanced-fullar-scratch300-h200x1-20260923).
Local directory: `outputs/church-compound-balanced-fullar-scratch300-h200x1-20260923`.

This is a fresh 300-epoch training run. The stage-2 model, optimizer, scheduler
and training random state start from seed 20260923, with no trained stage-2
checkpoint or disposable trial weights loaded. The frozen stage-1 tokenizer and
prebuilt full-training token cache are reused.

## Correction

The previous full-history model's history activations overwhelmed its current
atom embedding. The [trained-checkpoint diagnosis](church-full-history-conditioning-diagnosis-2026-09-22.md)
measured a roughly 2,000-fold RMS difference and negligible effect from shuffling
the coefficient decoder's current-atom input.

The corrected coefficient decoder independently applies learned LayerNorm to
four hidden feature fields before combining them:

`0.5 * (LN(atom-decoder history) + LN(current atom embedding)`
`       + LN(previous pair embedding) + LN(local physical prefix projection))`.

These are hidden-feature normalizations. Raw sparse coefficients, their physical
units, bin centers, soft targets, and decoded latent construction are unchanged.
There is no coefficient clipping or depth normalization. The finite 2,048-bin
vocabulary remains a quantization of continuous coefficients.

The model has 447,949,824 trainable parameters. The atom decoder has 24 layers,
the coefficient decoder has 8, and both use width 1,024 and 16 attention heads.
Both use causal attention over the complete 256-event sequence.
Each of the 256 atom predictions is followed by its conditional coefficient
prediction; there is no spatial pooling, history-window truncation or cache
reset between sites. The correction adds 8,192 LayerNorm parameters.

The design adapts the stacked conditional factorization from
[DCTransformer](https://proceedings.mlr.press/v139/nash21a/nash21a.pdf).
It remains an adaptation, not a reproduction of the full architecture or
optimization recipe. This relaunch isolates the tested conditioning correction;
it retains the matched RQ-style optimizer recipe rather than simultaneously
adding the paper's ReZero gates or warmup.

## Cache, training and evaluation

The complete cache was built before launch and is loaded into RAM: 126,227
training images, 8×8 sites, 16 complete stochastic OMP alternatives per site,
and four atom/raw-FP32-coefficient pairs per alternative. Training performs no
image loading, encoder inference or OMP solving. GPU selection keeps each
trajectory's atoms and coefficients together. The immutable cache SHA-256 is
`437bf76107dd3da5661db6c766b474be43304f37705d6c8431f2e34ae3afa9fa`.

Global batch is exactly 2,048, including a correctly weighted final 1,299-image
batch. All images are visited once per epoch: 62 updates per epoch and 18,600
total. AdamW uses LR 5e-4, betas (0.9, 0.95), weight decay 1e-4, cosine decay to
zero, no warmup, residual dropout 0.1, and gradient norm limit 1. Atom and
coefficient objective weights are 1.5 and 1.0. Coefficient soft-target
temperature is 0.125 in raw physical units.

Official RQ FID50k runs each epoch against the same full 126,227-image reference.
Sampling uses atom temperature 1 / top-k 250 and coefficient temperature 1 /
all bins, with top-p 1. The first 64 generated images are logged without
selection. Frozen decoding and Inception evaluation use FP32 with TF32 disabled.

## Conditioning checks and verification

Each epoch logs validation loss with standard inputs, shuffled current-atom
conditioning and shuffled completed-pair history on a fixed 128-image probe.
It also logs each field's RMS before and after normalization. This distinguishes
an available causal path from effective learned use of the inputs. A separate
fixed 300-image training/validation probe retains the previous head metrics.
Diagnostics restore model modes, predictions and RNG state exactly.

Fourteen focused tests passed. The production-size GPU preflight also passed
causality checks, dense/cached agreement at all 256 events, gradient/update
checks for both decoders and every classifier, raw-coefficient codec checks,
exact full-data batch coverage and a 2,048-image sampling batch. After two
disposable global-batch updates, the history/current-atom RMS ratio was 978
before field normalization and 1.013 after it. Preflight maximum FP32 cached
logit error was 5.72e-6; maximum BF16 distribution KL was 1.89e-5.

The earlier matched fresh pilot modestly improved coefficient validation KL
from 2.013 to 1.940 after 64 small-batch updates. That pilot establishes neither
competitive FID nor the eventual quality of this fresh full-budget run.

## Recovery, publication and predecessor

The detached supervisor resumes from complete model, AdamW, scheduler and
Python/NumPy/CPU/CUDA RNG states. A pre-evaluation checkpoint explicitly records
pending evaluation, so a restart does not repeat an already completed epoch.
The monitor checks progress and uploads, provides bounded crash/stall recovery,
and requires epoch 300 plus a verified final checkpoint upload for completion.

Latest and best-FID checkpoints are published as W&B artifacts with remote
digest and size verification. Execution sources, configuration, validation
evidence and input-artifact lineage are published with the run.

The preceding full-history run was retired at its last fully saved epoch 26 /
step 1,612. Its best checkpoint was epoch 24, FID50k 95.6076. Both full states
were pinned locally before the old worker was stopped; online retirement
receipts are stored under that run's `retired/` directory. No state from that
model initializes this relaunch.

## Verified launch

The online configuration, committed source artifact v0 and both frozen input
artifact links were checked independently through the W&B API. The published
fresh-state receipt confirms zero optimizer entries, scheduler step zero and
no loaded stage-2 checkpoint. Production reached step 34 of epoch 1 with finite
losses and gradients, at approximately 470 images/second. The supervisor and
monitor are both alive. The old latest/best artifact v50 was also independently
verified by remote digests and file sizes.

Startup initially lacked the local frozen `official-metrics` files. They were
restored byte-for-byte from the previous evaluator and import-checked before
the successful retry; no optimizer updates preceded that correction. The
running source snapshot includes those evaluator files. Initial FID50k and
new-run checkpoint selection remain pending at this launch verification.

The first epoch subsequently completed all 62 optimizer updates. Its full
pre-evaluation checkpoint was checked for all 447,949,824 parameters, all new
normalization weights, optimizer/scheduler step 62 and complete RNG states.
The first production conditioning probe measured history/current-atom RMS
ratio 1.02075 after normalization. Shuffling current-atom conditioning increased
mean coefficient KL by 0.02811 nats (0.09639 at depth 0); shuffling pair history
also increased both atom and coefficient losses. FID50k and the first checkpoint
upload are in progress. These checks establish active conditioning, not a
generation-quality improvement.
