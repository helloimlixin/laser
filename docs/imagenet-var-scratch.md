# ImageNet VAR: matched VQ versus LASER from scratch

**Resumed with user authorization.** After reviewing the unavailable original
stage-1 recipe, the user chose to continue the earlier matched ImageNet-only
scratch experiment. Both arms resume their saved checkpoints with the same
frozen source, data, hyperparameters and W&B run IDs. Their 100-epoch tokenizer
schedule is a shared experimental choice; it is not the published VAR recipe.
The [stage-1 source audit](var-stage1-protocol-audit.md) records that distinction.

The original pretrained adaptation has also been stopped. This experiment
trains **both tokenizers and both VAR priors from random initialization**.
Artifacts are in `outputs/imagenet-var-scratch-20260914/{vq,laser}`.

| Control | VQ | LASER |
| --- | --- | --- |
| Trainable initialization | Random | Random |
| Tokenizer backbone | Released VAR VAE, width 160, latent width 32 | Identical |
| Spatial downsampling | 16 at image size 256 | Identical |
| Scale sizes | 1,2,3,4,5,6,8,10,13,16 | Identical |
| Transformer positions | 680 | 680 |
| Dictionary atoms | 4096 | 4096 |
| Representation per site | One VQ index | Two atom/coefficient pairs |
| Prior | Official VAR-d16 body | Same body, additional sparse-code heads |
| Prior parameters | 310,283,520 | 311,665,153 |
| GPUs | H200 0,1 | H200 2,3 |

## Initialization and data controls

The scratch factory rejects a non-null `pretrained_vae`. Tests prohibit
`torch.load` during tokenizer construction. Both arms use seed 0 and start with
the same encoder, decoder, quantization convolutions, and residual convolutions.
The initial VQ embedding and LASER dictionary contain the same random unit
directions. Unit normalization is our shared initialization choice; it is not a
claim to reproduce an unpublished tokenizer initialization recipe.

The complete random tokenizer state is saved in `initial-tokenizer.pt` before
training. `provenance.json` records initialization and shared-backbone SHA-256
hashes. The discriminator and prior have separate initialization receipts.
The prior body and shared atom prediction head start identically; additional
LASER coefficient/depth heads are also random.

Training uses the full ImageNet train manifest (1,281,167 images, 1000 classes)
with fresh deterministic 288px Lanczos resize/256px random crops each epoch and
no horizontal flips. Both arms use the same distributed sampler, augmentation
seed, effective batch, and incomplete-batch handling. `tokenizer-first-batch.json`
records the first input pixels and labels and the rank-zero epoch sample-order
hash. Validation uses the same fixed subsets and center crops.

The frozen LPIPS/VGG perceptual loss and Inception evaluation networks use
pretrained weights in both arms. They are shared loss/metric networks; no
trainable tokenizer, discriminator, or generation model is initialized from a
pretrained checkpoint.

## Matched ImageNet training budget

Both tokenizers train for 100 ImageNet epochs, global batch 128 (32 images per
GPU, two microbatches), with AdamW, LR 1e-4, 500-update warmup, L1 + LPIPS +
bottleneck loss, and adaptive adversarial loss starting at update 10,000.
The discriminator performs one accumulated update per tokenizer update.
VQ retains the released nearest-neighbor quantizer and its loss. LASER uses this
repository's OMP dictionary learning with two atoms and 257 coefficient levels.
Its coefficient ranges track training coefficients by an EMA and freeze during
validation and prior training.

After tokenizer training, each arm automatically trains a fresh VAR-d16 prior
for 200 epochs, global batch 768, AdamW LR 3e-4, betas (0.9,0.95), weight decay
0.05, four warmup epochs, and the released `lin0` schedule. These are the
[released VAR-d16 prior settings](https://github.com/FoundationVision/VAR).
Both arms receive the same number of updates/images; runtime is measured
separately. This is a substantial multi-day run, not a one-epoch adaptation.

## What the comparison establishes

Factor 16 preserves the established VAR spatial layout and transformer length.
It controls spatial compression without claiming that we have optimized the
downsampling factor. LASER carries more discrete information: fixed-width codes
are 28,560 bits/image versus VQ's 8,160 (3.5 times as much), excluding shared
model parameters. Therefore this initial experiment compares methods at the
same spatial layout and nearly the same prior size, **not at equal bitrate**.

The published VAR tokenizer was trained on about 8 million downloaded images
from full OpenImages v4, as clarified by an
[author](https://github.com/FoundationVision/VAR/issues/145#issuecomment-2719768814).
The examined public sources do not establish the complete recipe for that
multi-scale checkpoint. Our ImageNet-only scratch
VQ control is the direct ablation baseline. Published VAR-d16 FID 3.55 is an
external reference, not an assumed score for this control or a current overall
SOTA threshold. See the [VAR paper](https://arxiv.org/html/2404.02905v2) and
[released model table](https://github.com/FoundationVision/VAR).

Report paired reconstruction FID/PSNR during tokenizer training, and generated
FID, sFID, Inception Score, precision, and recall from 50,000 class-balanced
samples using the official ADM evaluator during prior training. Both use the
same CFG 1.5, top-k 900, top-p 0.96, and no rejection sampling. Small preview
FID values are diagnostics. Raw VQ and LASER cross-entropies have different
targets and must not be compared as equivalent likelihoods.

## Runs and restart

- [VQ scratch W&B](https://wandb.ai/helloimlixin-rutgers/laser/runs/imagenet-var-vq-scratch-f16-d16-20260914)
- [LASER scratch W&B](https://wandb.ai/helloimlixin-rutgers/laser/runs/imagenet-var-laser-scratch-f16-d16-20260914)

`protocol-audit-stop-request.json` and each arm's `paused-protocol-audit.json`
are historical records of the protocol review. The user subsequently authorized
resuming this custom ImageNet recipe. VQ resumed at update 878 and LASER at 817;
both continue toward the same epoch/update budgets. Original random
initialization, model/optimizer states and RNG states are preserved.

`launch.json` at the experiment root records both exact commands, process IDs,
GPU assignments, and the frozen source snapshot. Restart an exited arm with
its recorded command and environment; checkpoint resume is enabled. Do not
launch a second copy while its current process is alive. A changed training
contract requires a separate output directory. Smoke-test checkpoints are in
separate `smoke-*` directories and are never used to initialize production.
`launch-history/` preserves earlier launch records; `launch.json` identifies the
latest processes. Per-arm `resume-*.json` files record the checkpoint hashes and
positions used for each restart.

The implementation tests cover checkpoint-free initialization, identical shared
weights, original VQ prior forward behavior, finite gradients, sparse-code
causality and roundtrips, coefficient-range freezing, and matching recipes.
Two-GPU smoke runs additionally exercise tokenizer/prior optimization,
adversarial accumulation, sampling, and prior checkpoint resume for each arm.
