# FFHQ commitment correction and quality audit, 2026-09-26

The [existing run](https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhq-a2048-k4-rqrecipe150-20260924) resumes from optimizer step **35,000**, epoch index 74, batch cursor 294, with the commitment coefficient applied exactly once. Production resumes from the original saved training state. Evaluated pilot weights are retained separately as reconstruction candidates, with their full-validation results; they are not substituted for production optimizer state.

## Objective correction

The previous configuration multiplied an inner commitment coefficient 0.25 by an outer latent coefficient 0.25, making the effective encoder commitment weight **0.0625**. Set `arch.hparams.commitment_cost: 1.0` and retain `arch.hparams.latent_loss_weight: 0.25`:

`L_latent = 0.25 * (L_dictionary + MSE(stop_gradient(z_quantized), z_encoder))`.

The dictionary coefficient stays **0.25**. The actual quantizer gradient test verifies the encoder commitment gradient is exactly four times the old one, agrees with the closed-form gradient, and leaves the dictionary gradient unchanged. The launcher asserts both coefficients on every restart. This changes the objective prospectively; it does not rewrite prior training history. The first resumed epoch's accumulated training losses include both objective versions, so its aggregate total loss is not directly comparable with a complete corrected epoch.

The original RQ-VAE trainer applies the outer latent coefficient to an unweighted quantizer commitment loss. LASER additionally has a learned-dictionary loss, which this correction preserves. Generic dictionary defaults and unrelated experiments are unchanged.

## Matched parameter screening

Every candidate loads the same model, discriminator, both Adam states and scheduler clocks from the protected source. Each uses the same 10,240 training images, identical per-image seeded crops/flips, global batch 128, and 80 optimizer updates. Evaluation uses 1,024 evenly spaced official validation images, FP32 with TF32 disabled, original RQ-VAE Inception/rFID, NVIDIA FLIP, frozen VGG LPIPS, and RGB [0,1] MSE. All four metrics are lower-is-better. Subset rFID is sample-size dependent and must not be compared directly with the run's 10,000-image scores.

| Candidate (80 updates) | Subset rFID | FLIP | LPIPS | MSE [0,1] |
|---|---:|---:|---:|---:|
| control | 29.1703 | 0.239435 | 0.234095 | 0.005536 |
| corrected | 29.1577 | 0.208859 | 0.223733 | 0.004698 |
| micro32 | 29.4762 | 0.227952 | 0.231420 | 0.005435 |
| low_gan | 29.5696 | 0.210458 | 0.225422 | 0.005053 |
| strong_d | 36.9004 | 0.260934 | 0.239605 | 0.005834 |
| low_lr | 28.2234 | 0.207470 | 0.221344 | 0.004761 |
| progressive | 29.9791 | 0.224263 | 0.228081 | 0.005187 |

`control` has the old double-weighted commitment; all other candidates fix it. `corrected` retains batch 64/GPU and accumulation 1. `micro32` uses batch 32/GPU and accumulation 2. The remaining candidates use that microbatch-32 configuration: `low_gan` changes GAN weight 0.75 to 0.25; `strong_d` changes discriminator LR 4e-5 to 2e-4; `low_lr` changes model and dictionary LR 4e-5 to 2e-5; `progressive` enables prefix-averaged bottleneck losses. Model/dictionary and discriminator LR otherwise remain 4e-5.

Discriminator parameter/optimizer identity checks passed. A separate fixed-reconstruction diagnostic demonstrated discriminator loss reduction and improved held-out margins at the existing LR; its near-one live hinge loss is not evidence of a disconnected optimizer. Higher discriminator LR worsened the paired reconstruction metrics. All training trials passed finite-parameter, optimizer/scheduler-counter, and exact cross-rank dictionary agreement checks.

## Full held-out confirmation

Two longer paired trials each start again from step 35,000 and run 200 updates on the same 25,600 images with identical augmentations. Their weights are evaluated on **all 10,000 validation images** with the production reference statistics and FP32 protocol:

| Candidate (same 200 updates) | Full 10,000 rFID | FLIP | LPIPS | MSE [0,1] |
|---|---:|---:|---:|---:|
| corrected | 8.9999 | 0.224725 | 0.227996 | 0.005017 |
| low_lr | 9.0306 | 0.201481 | 0.219733 | 0.004310 |

A visual comparison of the first eight official validation images is saved as `reconstruction-comparison.png`; each triplet is original, corrected LR4e-5, corrected LR2e-5. Both reconstructions retain face layout and appearance, with visible fine-detail smoothing; visual inspection is supplementary to the full-set metrics.

The commitment correction is required for objective correctness. The selected continuation is **low_lr**. The initial strict screening gate required rFID to improve by more than 0.1, FLIP and LPIPS not to increase, and MSE not to increase by more than 2%. That gate passed: **False**. The full comparison instead revealed a quality tradeoff: rFID 9.00 versus 9.03, with materially lower FLIP, LPIPS and pixel error at the lower LR. Final selection rationale: **Choose materially better fidelity (FLIP -10.3%, LPIPS -3.6%, MSE -14.1%) for a measured rFID increase of 0.03075 (9.00 to 9.03). This is a small explicit rFID/fidelity tradeoff, not a strict rFID win.**. This is an explicit multi-metric choice, not a claim that lower LR won on rFID. Other GAN/discriminator and progressive-loss changes are rejected. These are short, single-seed trials, not proof of a global optimum or a guarantee of long-term quality. Full production validation now logs rFID, NVIDIA FLIP, LPIPS and MSE every epoch, together with the existing reconstruction and coefficient previews.

## Retained recipe and continuity

- Official FFHQ split: **60,000 training / 10,000 validation images**, verified image IDs and RGB256 inputs; official crop/flip augmentation.
- Two H200 GPUs, batch **32/GPU**, accumulation **2**, global batch 128; discriminator BatchNorm groups of 32.
- Model and dictionary Adam LR **2e-05**, discriminator LR **4e-5**, betas (0.5, 0.9), no weight decay; original completed five-epoch warm-up and constant LR schedule; target 150 epochs.
- GAN weight **0.75**, LPIPS weight **1**, two-layer discriminator, 64 base channels; progressive bottleneck loss disabled.
- Training TF32 convolutions retained; FP32 sparse solver, parameters, optimizers and validation, with matmul TF32 disabled.

Protected source checkpoint ID: `089b6b7bdf0e4ce8a02ec9eb12fbf565`. Both optimizer steps are 35,000, with 18,816 images consumed per rank in the current epoch. Both complete optimizer states, scheduler clocks, RNG states and partial-epoch metrics are restored. If the batch changes, the microbatch cursor and metric accumulator are rebased to preserve the consumed-image count; optimizer-step progress is unchanged. A lower constant LR rebases only scheduler LR levels, preserving the existing Adam moments and all scheduler step clocks. Resume assertions verify all optimizer counters and both scheduler state dictionaries against the saved source. Recovery checkpointing continues every 200 steps, plus epoch checkpoints and versioned W&B artifacts.

The full source (including optimizer/RNG state) is uploaded to the same W&B run as `quality-source-step35000.pt`. The former best-three checkpoints are separately hardlinked under the audit directory, including epoch 67 rFID **9.6765**, epoch 66 **9.6842**, and epoch 50 **9.8327**. Existing versioned W&B checkpoint artifacts remain available. The two fully evaluated trial snapshots are also uploaded as `quality-candidate-*-step35200-weights.pt`; these include model/discriminator weights, but not optimizer states. Keeping these anchors protects the ability to recover; it does not guarantee that future metrics cannot fluctuate.

Runtime: `/tmp/laser-ffhq-k4-rqrecipe150-20260924`; supervisor `quality_supervisor.py`; status `quality-supervisor-status.json`; logs `quality-training-attempt*.log`; restore receipt `quality-resume-audit.json`. Audit sources and results are under `/tmp/laser-ffhq-quality-20260926`, with credential-free copies and the full runtime archive under `outputs/ffhq-k4-rqrecipe150-quality-20260926`.

## Production verification

Production resumed with both optimizer counters and scheduler clocks at 35,000. A new checkpoint at step 35,200 contains the corrected coefficient and selected configuration; all model/discriminator parameters are finite and both optimizer/scheduler clocks agree. The first full production validation (epoch 75, 10,000 images) recorded rFID 10.5274, FLIP 0.220825, LPIPS 0.227289, and RGB [0,1] MSE 0.005229.
