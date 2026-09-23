# Fresh Church run with dictionary-vector coefficient conditioning

Run: [church-laser-compound-dictattn-fullar-scratch300-h200x1-20260923](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-laser-compound-dictattn-fullar-scratch300-h200x1-20260923).
Directory: `outputs/church-compound-dictattn-fullar-scratch300-h200x1-20260923`.

The user selected the explicit dictionary-vector conditioning identified in
[the FFHQ 8.17 audit](ffhq817-transfer-audit-2026-09-23.md). The new model has
473,408,512 parameters: 24 full-history atom layers, 8 full-history coefficient
layers, and a two-layer attention module that combines history with the current
atom separately at every event. Width remains 1,024 with 16 attention heads.

For each event, independently normalized history, previous-pair embedding, and
local physical prefix form the history token. The current-atom token combines
the normalized learned atom-ID embedding with an independently normalized
linear projection of the frozen 256-dimensional dictionary vector. Each sum
is divided by the square root of its number of terms. Learned role embeddings
distinguish the two attention positions. Two causal blocks process
`[history, current atom]`; their atom-position output is combined with the
history residual and supplied to the full-history coefficient decoder.

The event-local fusion cannot attend to other events. Both main decoders still
attend to every previous individual compound event, without spatial pooling or
cache resets between sites. The current coefficient and all future events are
excluded from the current prediction. The factorization remains
`product_t p(atom_t | pairs_<t) p(coeff_t | atom_t, pairs_<t)`.

These normalizations act on hidden features. The full prebuilt cache still
stores raw FP32 coefficients without clipping or depth scaling. Frozen stage 1,
the 16 cached OMP alternatives/site, 2,048 physical coefficient centers,
temperature .125, and physical sparse reconstruction remain unchanged.

Training starts with random stage-2 weights, empty AdamW state, and seed
20260923. No predecessor or pilot weights initialize production. The duration
is 300 epochs, exact global batch 2,048, 62 updates per epoch, and 18,600 total
updates. AdamW uses LR 5e-4, betas (.9,.95), weight decay 1e-4, cosine decay to
zero, no warmup, dropout .1, and gradient norm limit 1. Classification weights
are atom 1.5 and coefficient 1; auxiliary geometry weight remains zero.

The coefficient training log now records cross-entropy, target entropy, and
their difference (KL) separately. Every epoch also records the 300-image
train/validation probe and a fixed 128-image conditioning probe. The latter
shuffles both current-atom fields together, dictionary geometry alone, or
completed-pair history. It restores model modes and RNG state after evaluation.

Official Church FID50k runs each epoch with atom top-k 250, coefficient top-p 1,
and both temperatures 1. Frozen decoding and Inception remain FP32. The first
64 generated images are logged without selection. Latest and best-FID full
checkpoints include optimizer, schedule and RNG states, with online artifact
digest/size verification. Detached supervision and monitoring provide bounded
crash/stall recovery through epoch 300.

Verification and launch results are recorded below after completion. Unit tests
cover full-sequence causality, dense/cached equality across 256 events, all-field
gradient flow, independent dictionary conditioning, hidden-scale robustness,
and exact model/optimizer/RNG continuation. A production-size GPU preflight,
microbatch benchmark, and matched fresh control/candidate pilot precede launch.
The pilot is an early-learning diagnostic and does not establish improved FID.

The first production-size backward preflight exposed a CUDA illegal-memory
access in BF16 fused scaled-dot-product attention over 32,768 independent
two-token sequences. The failure was reproduced in an isolated attention call.
The field-fusion blocks now compute exact two-position causal softmax attention
directly: the first position sees only its own value, and the second position's
two-way softmax is a sigmoid of the score difference. Scores and weighted sums
use FP32. Main full-history attention continues using its existing kernel.
The replacement matches reference attention outputs and gradients in FP32 and
passes the previously failing GPU batch. Twenty-four focused tests pass,
including this regression. No production optimization preceded the correction.

## Completed prelaunch measurements

Production-size GPU verification passed with 473,408,512 parameters. Maximum
FP32 cached/dense logit difference across all 256 events is 6.32e-6; maximum
BF16 distribution KL is 1.84e-5. Future/current-coefficient leakage checks are
exactly zero. Every trainable parameter has finite nonzero gradients in two
disposable full-global-batch updates. Sampling 2,048 images fits in 67.57 GiB.

The fastest tested training configuration is microbatch 256, accumulating eight
microbatches for global batch 2,048: 413.0 images/s and 108.63 GiB peak memory.
All 126,227 training images are covered, including the correctly weighted final
1,299-image global batch.

A matched pilot used identical shared initial weights, the same sampled targets
and cached trajectories, 64 updates, batch 256, and 16,384 image presentations
per arm. Extra fusion dropout makes later dropout streams architecture-specific.
On the fixed 128-image validation probe:

| Metric | Normalized-sum control | Dictionary-attention candidate |
|---|---:|---:|
| Mean coefficient KL | 1.93231 | 1.90940 |
| Mean atom NLL | 9.50541 | 9.48873 |
| Current-atom shuffle penalty, mean coefficient KL | .02310 | .14177 |
| Dictionary-only shuffle penalty | 0 | .07280 |

The coefficient improvement is concentrated at depth zero (KL 2.71078 to
2.50886). Later-depth KL is slightly worse in this short pilot, and current-atom
shuffle penalties remain small at those depths. This supports testing the new
conditioning mechanism in production; it is not evidence that all depth issues
are solved or that FID will improve. All pilot, preflight, and benchmark weights
and optimizer states are discarded.

The predecessor completed epoch 3 / update 186 with best FID50k 148.17194.
Its evaluated latest and best full states were pinned before the worker exited.
Remaining uploads were handed to a CPU process to release the GPU. The new
trainer releases GPU objects before draining its final uploads.

## Online launch verification

Production started successfully on its first attempt. The independent W&B API
check confirmed both conditioning flags, 473,408,512 parameters, the 300-epoch
budget, and the frozen cache/tokenizer hashes. Source artifact v0 is committed;
remote digests and file sizes match the fresh-state receipt, configuration,
trainer, monitor, preflight, pilot report and source archive. Both frozen input
artifacts are linked to the run.

The fresh-state receipt records zero optimizer entries, scheduler step zero,
and no stage-2 checkpoint loaded. Production reached update 13 of epoch one
with finite losses and gradients at approximately 413 images/s. Training CE,
target entropy and coefficient KL are all logged separately. The worker,
supervisor and monitor are alive. The first new checkpoint and FID50k are
pending at this verification; no generation-quality improvement is claimed.

The predecessor's final evaluated latest and best states are now verified
online in its selected-checkpoints artifact v4, with matching digests and
5,375,859,554 bytes per full state. Its W&B run is finished and points to this
fresh successor. This retirement verification is attached to the new run's
`online-launch-verification.json` file.

Production subsequently completed all 62 updates of epoch one. Its first full
checkpoint is 5,681,398,586 bytes and was independently checked for every
473,408,512 model parameter, the dictionary projection and both fusion layers,
optimizer/scheduler step 62, and Python/NumPy/CPU/CUDA RNG states. It records
pending evaluation, allowing exact continuation through the first FID50k.
The checkpoint upload and evaluation proceed under the detached supervisor.
