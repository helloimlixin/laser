# Church stage-one recipe and loss audit, September 20, 2026

The current one-epoch Church finetune uses the GitHub example YAML's update budget and
GAN/backbone loss weights. Its saved Adam states contain **987 generator updates
and 987 discriminator updates**. Eight H200 GPUs did not reduce this budget:
8 × 16 preserves the example's 4 × 32 global batch of 128.

**Correction after inspecting the released checkpoint archive:** the official
Church checkpoint's bundled configuration specifies **3 epochs and fixed LR
4e-6**, whereas the GitHub example YAML specifies **1 epoch and fixed LR 4e-5**.
The archive MD5 and bundled configuration bytes were verified. Thus the current
one-epoch run matches the example YAML, not the settings bundled with the paper's
released checkpoint. The released model contains no saved optimizer counters;
2961 G/D updates is the budget implied by its config on our dataset, not a
counter read from that model. See [the follow-up experiment](church-stage1-improve-2026-09-20.md).

The current tokenizer has worse matched reconstruction FID than the earlier
three-epoch tokenizer. Conversion to compound tokens contributes negligible
additional reconstruction error. This supports investigating stage-one quality,
but does not establish insufficient discriminator updates or a missing loss as
the cause. The older tokenizer also used different optimization settings.

## GitHub example recipe and actual counters

Primary sources:

- [Training instructions and four-GPU commands](https://github.com/kakaobrain/rq-vae-transformer#training-and-evaluation-of-rq-vae)
- [Church YAML](https://github.com/kakaobrain/rq-vae-transformer/blob/main/configs/lsun-church/stage1/church256-rqvae-8x8x4.yaml)
- [ImageNet YAML](https://github.com/kakaobrain/rq-vae-transformer/blob/main/configs/imagenet256/stage1/in256-rqvae-8x8x4.yaml)
- [Generator/discriminator update loop](https://github.com/kakaobrain/rq-vae-transformer/blob/main/rqvae/trainers/trainer_rqvae.py)
- [Scheduler](https://github.com/kakaobrain/rq-vae-transformer/blob/main/rqvae/optimizer/scheduler.py)

| Setting | GitHub Church example YAML | Current Church finetune |
|---|---|---|
| GPU batch layout | 4 × 32 | 8 × 16 |
| Global batch | 128 | 128 |
| Duration | 1 epoch | 1 epoch |
| G and D updates | 987 each for our 126,227 images | 987 each, verified in optimizer states |
| G:D update ratio | 1:1 | 1:1 |
| Discriminator starts | Epoch 0 | Epoch 0 |
| G/D optimizer | Adam, betas (0.5, 0.9), no weight decay | Same |
| G/D learning rate | Fixed 4e-5; no warmup | Same |
| Discriminator | PatchGAN, ndf 64, 2 layers, BatchNorm | Same; paired ranks preserve a 32-image BN scope |
| Initialization | Pretrained model and discriminator, fresh optimizers | Same |

The ImageNet source checkpoint contains **100,100 updates in both optimizers**,
10 epochs at 10,010 updates per epoch. This matches the released default
10-epoch ImageNet update count at global batch 128. Both warm up for 0.5 epoch.
However, our pretraining decays from 4e-5 to zero afterward, while the released
configuration holds 4e-5 after warmup. Our pretraining discriminator used
64-image BatchNorm groups instead of the released 32-image local batches.
Matching update counts does not make these training recipes identical.

The earlier three-epoch Church checkpoint records Lightning `global_step=5922`,
but its individual optimizer counters are **2961 G and 2961 D**. That global
step counts both optimizers. Comparing it directly with the native trainer's
987 would incorrectly suggest a sixfold rather than threefold update budget.
It used LR **4e-6**, gradient-based dictionary updates, and a different LASER
training driver. Therefore the reconstruction comparison is not a controlled
ablation of epoch count alone.

## Losses and gradient verification

The current generator objective is:

`mean pixel MSE + 0.25 * mean-prefix commitment + LPIPS + 0.75 * adaptive_weight * (-mean D(reconstruction))`.

The discriminator objective is:

`0.75 * 0.5 * (mean relu(1 - D(real)) + mean relu(1 + D(detached reconstruction)))`.

The adaptive weight is the detached ratio of the decoder output-layer gradient
norms of MSE + LPIPS and the generator adversarial loss, with denominator epsilon
1e-4 and clamp [0, 10000]. Its function matches upstream structurally. LPIPS uses
the pretrained VGG weights and the released input scaling on [-1, 1] images.
Perceptual weights are frozen while gradients reach the reconstruction.

The sparse commitment is the mean MSE over all four OMP prefix reconstructions,
with commitment cost 1 and stop-gradient on the reconstructed targets. This
retains RQ-VAE's average over depth, adapted to sparse coefficients and OMP.
The dictionary updates after each generator step by the alternating residual
method. It is not frozen merely because its parameter has no Adam gradient.
The dictionary fitting diagnostic is not accidentally added as a second
commitment term in this mode. RQ-VAE instead updates its codebook by EMA.

A two-image FP32 forward/backward probe on a private copy of the final checkpoint
verified the loss values, mean reductions, commitment gradient, nonzero gradients
in encoder/decoder/projections, nonzero discriminator gradients, and isolation
of generator parameters from the discriminator backward pass. The alternating
dictionary step changed its weights. The probe used 2.45 GiB and did not change
any saved or production weights. Its local two-image BatchNorm only verifies
gradient connectivity, not distributed BatchNorm equivalence.

All 19 logged training samples have nonzero adaptive weights, ranging from
0.19336 to 1.89585. The saved optimizer counters and before/after parameter
hashes independently verify that all requested components updated.

**Logging caveat:** `loss_total` excludes the generator adversarial term, following
upstream's logging convention. The actual backward objective includes it. The
19 logged values match MSE + 0.25 commitment + LPIPS within 1.68e-8. Validation
MSE also uses upstream's channel-scaled aggregation, so its displayed value is
not directly the same reduction as training MSE.

The original ImageNet trainer and dictionary source were recovered from its W&B
diff and matched against the source SHA256 values in the checkpoint. The
generator/discriminator training loop and GAN function match the Church source
structurally. Dictionary differences concern visualization state; the fitting
objective and alternating update are unchanged.

The active stage-two compound objective is `(1.5 * atom hard CE + coefficient
soft CE) / 2.5`, averaged across positions and depths, with geometry disabled.
An independent CE calculation and backward check passed; 115 logged training
samples matched this expression within 1.24e-6. This is a compound-token
adaptation, not an unchanged RQ discrete-token objective.

## Matched reconstruction screen

All methods reconstructed the same first 4096 Church training images with the
released resize/center-crop transform, FP32 and TF32 disabled, and the released
FID Inception features. Each reconstruction distribution was compared with the
same 4096 originals. These are **subset screening scores**, not full-population
rFID or generated-image FID. The current native full-population stage-one log
reports rFID 5.2212; it should not be compared numerically with this subset table.

| Reconstruction | Matched subset rFID ↓ | Mean per-image PSNR ↑ |
|---|---:|---:|
| ImageNet source, before Church finetuning | 17.4621 | 19.2050 dB |
| Current one-epoch native LASER | 8.9197 | 19.9777 dB |
| Current one-epoch compound cache | 8.9183 | 19.9772 dB |
| Earlier three-epoch native LASER | 6.5315 | 18.9562 dB |
| Earlier three-epoch compound cache | 6.5386 | 18.9491 dB |

The one-epoch finetune substantially improves on the ImageNet initialization.
The earlier three-epoch tokenizer improves distributional reconstruction FID,
while the one-epoch tokenizer has better pixel PSNR. Compound quantization is
nearly neutral for both. The optimized native inference path agrees with the
historical quantizer to latent NMSE 6.75e-14 on the four-image implementation probe.

The GPU comparison took 38.5 seconds and the CPU FID calculation 11.9 seconds.
The geometry-free stage-two run resumed at saved step 656 on all eight GPUs.
Its tokenizer, loss, optimizer and requested 90-epoch duration were not changed.
Following the request to improve stage one and inspect the released Church
checkpoint, two three-epoch trials were launched at fixed LR 4e-6 and 1e-5,
retaining global batch 128 and the audited loss. Each evaluates at 987, 1974,
and 2961 G/D updates. This compares durations within each trial and learning
rates between trials; neither is an epoch-only ablation against the earlier
one-epoch LR 4e-5 run.

Artifacts are in `outputs/church-stage1-audit-20260920`:

- `result.json` and `comparison.png`: reconstruction scores and aligned images.
- `optimizer-audit.json`, `imagenet-optimizer-audit.json`, and
  `ft3-optimizer-audit.json`: actual saved optimizer step counters.
- `loss-probe.json` and `logged-loss-audit.json`: loss/gradient checks.
- `stage2-loss-audit.json`: compound objective checks.
- `pretraining-source/recovery.json`: verified historical source recovery.
- `upstream/sources.json`: primary-source URLs and downloaded source hashes.
