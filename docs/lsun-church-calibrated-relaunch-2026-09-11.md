# Church stage 2: calibrated targets and fresh image views

**Current status, September 11:** both calibrated runs are paused with recoverable checkpoints. The hard-target run paused earlier at step 55,001; the soft-target run paused at step **58,272**, epoch **59.5224**, to make room for the [matched coefficient-ordering pilot](lsun-church-pattern-order-2026-09-11.md). Its saved optimizer and data-stream state were verified. The experiment description below records the original relaunch.

The two earlier FFHQ-transfer runs remain paused with their last and best checkpoints intact. This relaunch addresses excessive coefficient-target noise and the growing training/holdout gap. It does not assume that the changes will improve unconditional image quality.

Both new experiments use the **218,802,176-parameter** architecture: width 768, 20 spatial layers, six within-site depth layers, and two atom-conditioned micro-transformer layers. Each starts from the same random initialization and uses the same image order, augmentation policy, optimizer settings, and evaluation settings. One uses calibrated physical soft targets; the other uses deterministic nearest-bin targets. Holding capacity fixed makes this a coefficient-target comparison.

## Target calibration

The previous normalized-space target injected physical noise with a standard deviation of roughly half the coefficient RMS. The new soft target measures distance in **physical coefficient units**, with one common physical standard deviation across depths. The bin vocabulary, dictionary, scales, and 8×8×4 compound representation are retained.

The calibration used 256 randomly selected **training images only**, sampled from the full untruncated target distribution. Before evaluating candidates, the acceptance limits were LPIPS mean plus two standard errors ≤ 0.01 and relative latent MSE ≤ 0.005. The widest candidate satisfying both limits was selected.

| Physical standard deviation | Target temperature | LPIPS | LPIPS upper bound | Relative latent MSE | Pass |
|---:|---:|---:|---:|---:|:---:|
| 0.5 | 0.5 | 0.03147 | 0.03222 | 0.01246 | No |
| 0.25 | 0.125 | 0.01005 | 0.01032 | 0.00313 | No |
| **0.125** | **0.03125** | **0.00285** | **0.00293** | **0.00077** | **Yes** |

The selected target's reconstruction PSNR is 40.71 dB relative to the continuous tokenizer reconstruction. This is a decoder-distortion calibration, not a generation FID or generalization result. The hard-target arm samples no coefficient-label noise.

[Calibration measurements](../outputs/church-calibrated-20260911/calibration/calibration.json) · [Clean/sampled reconstruction pairs](../outputs/church-calibrated-20260911/calibration/sigma-0.125.png)

## Training changes

- Fresh pixel-space augmentation each epoch: resize the shorter side to 256, random 256×256 crop, horizontal flip with probability 0.5. Every view is re-encoded through the frozen encoder and FP32 OMP. Latent tokens are not flipped or cropped directly.
- The 125,203 training image keys and 1,024 held-out keys remain disjoint. The 300 official validation images remain separate. Holdout and validation use the existing deterministic center-crop cache. The tokenizer previously saw the original training population.
- The existing one-epoch Church finetune from the ImageNet rFID 4.2109 checkpoint remains frozen. Its checkpoint SHA-256 is `93f19adc8ca2cfae0cf629d48d2d6f5f7039440b1a48ad1c7ba43cf17932a388`.
- AdamW, betas (0.9, 0.95), matrix weight decay 0.05; bias, normalization, and positional parameters have zero decay. Residual dropout increases from 0.1 to 0.15. Gradient norm is clipped at 1.
- Effective batch 128, accumulated as two microbatches of 64. BF16 model/encoder, FP32 OMP and decoder. Training records coefficient range violations and aborts if more than 1% of a microbatch exceeds the retained bin range.
- One-epoch warmup to LR 1e-4, exponential decay to 2e-5 at epoch 10, then cosine decay to 1e-6 at epoch 200. Absolute image-epoch progress determines the rate on resume. Two hundred epochs is a maximum, subject to early stopping.
- Full compound-pair history and 64-site spatial context remain. Atom loss weight 1.5 and distribution-geometry weight 0.05 remain, with geometry delayed two epochs and ramped over three.

Augmentation is deterministic for an image index, epoch, and seed. DataLoader workers may prefetch only from a copy of the shuffled epoch stream; the trainer commits positions after receiving the corresponding images. This preserves both the data sequence and the augmented views on resume, independent of worker scheduling.

## Validation and stopping

Epoch 1 and every two epochs measure a fixed-seed stochastic-context holdout objective matching the target construction used in training. Clean-context holdout metrics are logged separately. Center-crop training-probe and official validation metrics also track atom NLL, coefficient KL/NLL, sign accuracy, coefficient MAE, and depth-specific losses.

The stopping score is `(1.5 * atom NLL + coefficient KL) / 2.5`. Each arm stops after three consecutive checks without at least 0.01 improvement, no earlier than epoch 8. This score is for within-arm stopping; the hard and soft target entropies differ, so raw scores are not directly comparable across arms. Clean-context likelihood and generated-image metrics provide additional comparisons.

Each evaluation generates a fixed-seed **FID-4096** screen and sample grid, using atom top-k 2,048, coefficient nucleus p=0.5, and temperature 1. The tighter coefficient sampling helped the old frozen checkpoint; it is not yet validated as optimal for these newly trained priors. Original RQ-VAE Inception and Church reference statistics are used.

Full FID-50,000 runs every 25 epochs if training continues. After early stopping or reaching the epoch limit, an independent-seed FID-50,000 confirms the best screening checkpoint. The best holdout checkpoint is retained separately. Small-sample screening FID is not compared directly with historical FID-50,000.

Full model/optimizer/data-order checkpoints are saved every 250 optimizer steps and at epoch boundaries. A pending evaluation is recorded before evaluation starts so resume cannot silently skip it. SIGTERM/SIGINT save and pause. Source and configuration hashes are checked on resume.

## Verification

**35 tests passed**, covering physical target width, deterministic hard targets, image augmentation, prefetch/resume alignment, early-stopping state, parameter grouping, compound causality, cached generation, and geometry objectives.

Actual GPU smoke runs exercised training, image decoding, and FID for both target modes. A four-step augmented run resumed to step six produced bitwise-identical model weights, optimizer state, data stream, monitor state, and pending-evaluation state to uninterrupted six-step training.

A separate test on a 16-image population with an effectively zero learning rate exercised epoch boundaries, automatic stopping, restoration of the selected checkpoint, and independent final generation. It stopped as expected at epoch 2 and selected epoch 1. Its 64-image FIDs are plumbing checks with random weights, not quality measurements.

[Verification record](../outputs/church-calibrated-20260911/verification.json) · [Exact resume comparison](../outputs/church-calibrated-20260911/resume-verification.json)

## Runs and files

- [Calibrated soft-target run](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-calibrated-soft-aug219m-20260911)
- [Nearest-bin hard-target run](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-calibrated-hard-aug219m-20260911)
- Outputs: `outputs/church-calibrated-20260911/{soft,hard}`.
- Launcher: `scripts/launch_church_calibrated.py`; resume with `--resume` and the existing private W&B credential.
- Source snapshot: `outputs/church-calibrated-20260911/source-snapshot`.

Both runs launched on September 11 at 04:19 UTC, on separate H200 GPUs (soft PID 28989, hard PID 28990). W&B independently reported both as running. At the launch check, both had completed 60 optimizer steps with decreasing training loss, finite gradients, and no logged coefficient range violations. This verifies startup only; the first generated-image evaluation is scheduled after epoch 1. [Launch health record](../outputs/church-calibrated-20260911/launch-health.json).

The relaunch combines target correction, augmentation, stronger regularization, earlier LR decay, and stopping safeguards. It is not an isolated ablation of each change against the paused runs. Improved generated-image quality remains to be established by the new evaluations.
