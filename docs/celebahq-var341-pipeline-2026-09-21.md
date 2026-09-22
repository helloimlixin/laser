The user authorized the complete 341-position CelebA-HQ LASER VAR pipeline on
2026-09-21. Its experiment directory is
`outputs/celebahq256-var341-compound-20260921/`. The baseline 680-position run
finished its existing 50-epoch budget before this pipeline took the GPUs.
The detached runner records its current phase in `pipeline-status.json` and
does not write `complete.json` until training, final evaluation, and durable
checkpoint mirroring finish.

The five scales are `(1, 2, 4, 8, 16)`, totaling 341 spatial sites, with two
compound atom/coefficient pairs at each site. The source tokenizer uses ten
scales. Transfer selects its coefficient ranges at indices `(0, 1, 3, 6, 9)`
and preserves the corresponding residual-convolution coordinates
`(0, 1/9, 3/9, 6/9, 1)`. Using the new uniformly spaced scale indices to select
old convolution weights would change the retained scales' behavior.

The earlier frozen-tokenizer audit found an LPIPS increase from 0.18652 to
0.19617 and a 37.7% latent reconstruction MSE increase after reducing scales.
Consequently this experiment adapts the tokenizer before building a prior.
Its encoder, decoder, dictionary, and residual convolutions initialize from
`celebahq256-var-laser-poc-20260916/tokenizer-last.pt`. The source discriminator
drove the adaptive GAN weight to its cap during the initial smoke check.
This experiment instead initializes a fresh discriminator and waits 100
generator updates before activating adversarial training. Both optimizers
start fresh, with LR 2e-5 and 50 warmup updates. Training uses BF16, the existing L1 + LPIPS + quantization
loss, adaptive adversarial weighting, training-only coefficient-range updates,
batch 64 per GPU, and three H200s. The adaptation budget is ten epochs.

Reconstruction FID uses all 2,000 validation images each epoch; the best of
these adapted checkpoints supplies the frozen tokenizer for subsequent stages.
The selected tokenizer is re-evaluated and must pass the existing maximum
reconstruction-FID threshold of 50. Its complete weight/configuration identity
and checkpoint selection are recorded. Latent MSE before and after encoder
adaptation is not an invariant measure of reconstruction quality; image-space
metrics and the per-tokenizer stochastic-calibration checks are also retained.

The automatic stages are:

1. Three-rank tokenizer preflight, including adversarial backward, both
   optimizers, finite-gradient checks, and three-rank RNG serialization.
2. Ten-epoch tokenizer adaptation and best-checkpoint selection.
3. Recalibration of stochastic supports and physical coefficient soft targets
   against this tokenizer's deterministic reconstruction. The existing +5%
   latent MSE and +0.005 LPIPS stochastic-target limits remain enforced.
4. A new precomputed bank of complete five-scale trajectories: all 28,000
   training images, both horizontal-flip views, and 16 variants per view;
   deterministic codes for all 2,000 validation images. Every scale's residual
   follows the actually sampled earlier scales. Three GPUs build the bank.
   Exact cached contexts, latent reconstruction, and soft-target roundtrips
   are checked on every rank. The old 680-position cache is not reused.
5. Full compound-prior preflight: 100 fixed-batch overfit updates, production
   cached-batch updates, validation, distributed sampling/FID, and complete
   optimizer/RNG checkpoint validation.
6. A fresh 50-epoch VAR-d16 compound prior, using the adapted frozen tokenizer
   and the new cache. Batch 128 per GPU, effective batch 384, LR 3e-4, one
   warmup epoch, and decay to 3e-5 retain the baseline training recipe. Atom
   loss weight is 1.5; sampling uses top-k 250, top-p 1, and CFG 1.5.
7. Evaluation of the best diagnostic-FID checkpoint with all 2,000 validation
   images and separate 2,000- and 50,000-generated-image estimates. The real
   reference is still the 2,000-image held-out split; the latter is not an
   official ADM FID50k result. Evaluation uses all three GPUs and batch 64.

The runs are [tokenizer adaptation](https://wandb.ai/helloimlixin-rutgers/laser/runs/celebahq256-var341-tokenizer-20260921)
and [compound prior](https://wandb.ai/helloimlixin-rutgers/laser/runs/celebahq256-var341-stochastic-compound-20260921).
The prior run starts automatically after the tokenizer, cache, and preflight
stages pass. Final evaluation has the same prior ID with `-eval` appended.

Checkpoints are written atomically under
`/tmp/laser-var-checkpoints/celebahq256-var341-compound-20260921/` and copied to
durable storage by an independent CPU mirror. A failed copy retains the last
complete durable generation and retries; its status is `mirror-status.json`.
Prior W&B uploads are asynchronous. The selected tokenizer is published as
a separate model artifact because it is required to decode prior samples.
The cache is also logged as a W&B dataset artifact. Local temporary storage
is ephemeral until a mirror or upload completes.

Executed source is preserved in `runtime/`, with hashes in
`source-manifest.json`. `scripts/tools/run_var341_pipeline.py` owns stage
handoffs, failure receipts, and resumption. Preflight weights are isolated
from production. Initial focused verification passed 31 tests, including
scale-transfer rejection cases, preservation of retained kernels and ranges,
strict checkpoint reloads, cached-code roundtrips, compound causality, and
gradient checks. These checks establish implementation consistency; improved
generation quality still depends on the completed training and evaluation.

Launch verification: the six-update, batch-64, three-rank tokenizer preflight
passed with finite generator/discriminator gradients and complete optimizer
and RNG state. Its steady throughput was approximately 194 images/second,
with 120.2 GiB peak allocated memory per H200. Production then started from
the transferred tokenizer independently of those preflight updates. It
reached update 80 at roughly 207–208 images/second during the adversarial
warmup, with all three GPUs active at approximately 640–655 W. The initial
341-position reconstruction FID on all 2,000 held-out images was 12.291,
versus the source tokenizer's recorded 12.164. These reconstruction metrics
are distinct from generation FID and from the earlier 64-image LPIPS audit.
The online tokenizer run, five-scale configuration, and batch size were
verified through the W&B API. The frozen runtime passed the same 31 tests.

The baseline's final artifact drain initially retained idle GPU contexts after
epoch 50. All three final/best checkpoints were verified byte-for-byte against
their durable copies, including optimizer states and three-rank RNG state.
Only then were its GPU workers stopped. Its final W&B publication continues
in a CPU-only process, with receipts `gpu-handoff.json` and
`cpu-finalization.json` in the baseline directory. No training updates were
lost. The new pipeline's handoff distinguishes a completed training budget
with verified checkpoints from completed network publication; neither is
silently substituted for the other.
