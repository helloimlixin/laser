# Fresh Church stage 2 with joint sparse-vector targets

This run starts a new, randomly initialized stage-2 model, optimizer and schedule.
It does not inherit the earlier Church continuation or any FID-reward updates.
Its architecture and optimizer derive from the recovered
[FFHQ 8.174 recipe](../recovered/ffhqcmp0804205803/README.md); the target law below
is an explicit adaptation rather than a claim that FFHQ used joint soft targets.

## Fixed representation

The tokenizer, decoder and normalized dictionary remain frozen. Training uses
the original deterministic Church cache: 126,227 images, four support/coefficient
pairs per 8-by-8 spatial grid, FP32 coefficients and 2,048 bins on normalized
[-3, 3]. Existing physical scales are retained:
`[7.662353992462158, 4.1580352783203125, 2.633323907852173, 1.6512117385864258]`.
There is no new clipping, coefficient fitting, OMP-bank sampling or tokenizer update.

For each site, the target is the complete cached physical vector
`z = sum_d scale[d] * coefficient[d] * dictionary[:, atom[d]]`.
An atom and its coefficient always travel together in the candidate pool.
The pool has six pairs per depth: the cached atom's nearest bin and two bin
draws, plus three bin draws for one alternative atom. Alternative atom proposals
marginalize the two signs of the cached binned coefficient, so equivalent
positive and negative atom/coefficient representations receive equivalent
proposal scores. Every eligible atom and coefficient bin has proposal support.
Coefficients are sampled from the fixed bins; none is solved or refitted.

All valid combinations of these four candidate lists are scored by their
complete-vector squared error, including cross terms. Stochastic histories and
prefix-conditioned atom/coefficient soft labels come from that same normalized
distribution. Duplicate pairs are deduplicated and repeated atom supports are
forbidden. This is exact within a randomly proposed finite pool, not an exact
Gibbs distribution over the unrestricted vocabulary.

The training objective is equal-weight joint atom/coefficient negative log
likelihood. The archived marginal-product geometry surrogate, contrastive losses
and FID/covariance rewards are absent. FID is used for evaluation and checkpoint
selection only.

## Noise calibration

The temperature is calibrated using 128 fixed training images, with a separate
128-image training subset for checking. The 1,024 stage-2 validation images and
FID reference are excluded from calibration. Temperature is in squared physical
vector units; neither FFHQ's normalized temperature 0.5 nor the previous physical
temperature 2 is copied.

Before selecting a temperature, the added expected vector-error ceiling is set
to the smaller of 10% of frozen encoder reconstruction error and 1% of cached
vector energy. Actual nearest-bin error, per-depth bin spacing, support changes,
entropy and decoded reconstruction changes are reported separately. This is a
conservative noise-budget heuristic, not evidence that a particular FID will be
reached. Calibration does not change the representation or fit coefficients.

The selected physical temperature is **0.33584674003704434**. Calibration added
expected vector error is 0.237833 against the registered 0.242614 ceiling; the
independent check is 0.239077. The latter is 10.193% of its own slightly smaller
encoder residual, so the 10% rule is not claimed as an exact bound for every
subset. No retuning used those check results. Support changes are about 0.61%
of pair slots (about 1.97% at depth four). The decoder comparison has 36.06 dB
PSNR relative to continuous cached-vector reconstructions; this is not image
quality versus ground truth and is not FID.

## Training and artifacts

The recovered width-1,024, 24-spatial/4-depth-layer architecture retains its
physical pair embeddings, learned identity adapter and two-layer conditional
coefficient head. Residual dropout is 0.1. AdamW starts empty at learning rate
5e-4, betas (0.9, 0.95), weight decay 1e-4 and gradient clipping 1.0. Cosine decay
starts at update zero with no warmup, matching the FFHQ recipe.

Eight GPUs each train on 32 images: global batch 256, without accumulation.
This doubles FFHQ's batch while retaining 97,800 optimizer updates over 200
Church epochs. Training contains 125,203 images; the final incomplete batch is
dropped each shuffled epoch, as in FFHQ. The remaining 1,024 images are held out
from this fresh stage-2 model; the pre-existing tokenizer predates this split.
The recovered forward path uses FP32, despite the historical outer BF16 context.

Two sample grids are logged every 200 updates: the FFHQ sampler (atom top-k 250,
coefficient top-p 0.85, both temperatures 1) and a full-vocabulary sampler. Official
50k FID uses the FFHQ sampler after the first epoch, every ten epochs and at the
end. Validation joint loss is measured each epoch. Full last/best checkpoints
include optimizer, cosine position, eight RNG streams and data cursor. The first
checkpoint is at step 200, then every 1,000 steps and each FID/final boundary.
Older local snapshots are pruned only after verified remote publication; latest,
best and pending uploads are retained.

Runtime, calibration, verification and launch records are preserved in
`outputs/church-ffhq-joint-scratch-20260928/`. The concrete `plan.json` and
`calibration.json` in that directory are authoritative for the launched run.
