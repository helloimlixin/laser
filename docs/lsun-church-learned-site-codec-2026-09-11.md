# Learned compression of complete sparse latent sites

This pilot trains a **7,869,952-parameter compressor** that emits an 8×8 grid of categorical IDs from a 16,384-token vocabulary. Each ID independently determines **all four dictionary atom IDs and all four signed coefficient-bin IDs** at its location. There is no source-support input or continuous residual channel at decoding time.

The comparison is the earlier 16,384-entry direct codebook, whose validation reconstruction LPIPS was 0.35116 and PSNR was 16.891 dB relative to continuous LASER reconstructions. A better compressor must reduce distortion before training a new autoregressive prior is justified. Generation quality and token predictability remain separate questions.

## Learned representation

The encoder applies three convolutional residual blocks to the normalized 8×8 latent grid. Its output is a residual adjustment to the original physical latent. Each adjusted site chooses its nearest learned 256-dimensional codeword. The residual output starts at zero, and codewords initialize from the previous 16,384-entry complete sparse codebook.

Every selected codeword is projected through four steps of OMP with distinct atoms in the frozen LASER dictionary, followed by nearest-bin coefficient quantization using the retained depth scales. Consequently the actual forward image is always decoded from four integer atom IDs and four integer coefficient IDs per site. The OMP solve uses FP32 with a 1e-6 ridge; coefficient signs remain encoded in the signed bins.

The integer assignment and OMP/bin projection have an identity gradient surrogate. Reconstruction gradients update both the encoder and selected codeword vectors. Separate commitment, codeword-alignment, and projection penalties help keep these surrogate paths aligned with their discrete outputs. This is an experimental, biased gradient estimator; the hard forward and exported-codebook checks are the evidence that continuous information is not bypassing compression.

The learned encoder may use spatial context when assigning IDs. Decoding an individual ID is pointwise and independent of its neighbors. At completion, all learned codewords are exported as a finite table of complete sparse codes, so generation can decode IDs using table lookup.

This uses the learned discrete-bottleneck idea of [VQ-VAE](https://arxiv.org/abs/1711.00937), adapted to a frozen sparse dictionary and image decoder. It is not a reproduction of that paper's architecture or a claim of its reported performance.

## Training and evaluation

- Frozen stage 1: the existing one-epoch Church finetune from ImageNet rFID 4.2109. Its encoder, dictionary, coefficient bins/scales, and image decoder receive no optimizer updates. SHA-256: `93f19adc8ca2cfae0cf629d48d2d6f5f7039440b1a48ad1c7ba43cf17932a388`.
- Training uses the existing center-crop continuous sparse cache, with 125,075 training images. There is no image augmentation in this pilot. The same 128 training-calibration images reserved in the earlier codebook experiment are excluded from optimizer updates.
- Targets are continuous frozen-LASER reconstructions, rather than original photographs. The image decoder is frozen but retains gradients with respect to its input; LPIPS parameters are frozen as well.
- Objective: `LPIPS + 0.5 image L1 + 0.1 relative latent MSE + 0.25 commitment + 0.1 alignment + 0.1 projection`.
- AdamW, betas (0.9, 0.95); encoder weight decay 0.01 and codeword decay zero. Encoder LR peaks at 1e-4 and codeword LR at 5e-4 after a 100-step warmup, then each follows cosine decay to one tenth of its peak. Gradient norm is clipped at 1.
- Effective batch 32, microbatch 4. The encoder uses BF16; assignments, OMP, sparse reconstruction, and image decoding use FP32.
- Maximum 6,000 optimizer steps, approximately 1.54 passes through the training population. Calibration LPIPS is checked every 250 steps. Four checks without a cumulative improvement of 0.001 trigger stopping after at least 2,000 steps. The minimum measured calibration LPIPS determines the retained checkpoint.
- The same evaluations also measure a fixed 256-image subset of the separate holdout and all 300 official validation images. PSNR, relative latent MSE, code usage, and marginal token entropy are recorded alongside LPIPS. These marginal statistics do not establish autoregressive learnability.
- The existing acceptance thresholds remain: training-calibration LPIPS mean plus two standard errors ≤ 0.01 and relative latent MSE ≤ 0.005. These criteria concern added compression distortion, not unconditional image quality.

Checkpoint files retain optimizer state, the exact shuffled data stream, stopping state, pending evaluation, configuration, and source hashes. Periodic saves occur every 100 steps; SIGTERM/SIGINT pause after the current optimizer step. At termination, the best checkpoint is restored and a standalone `best-codebook.pt` is exported alongside `best.pt`.

## Verification and run

Ten focused tests pass, including full sparse decoding from IDs alone, distinct OMP atoms, signed-bin reconstruction, exact agreement of hard forward and exported codebook lookup, image gradients through frozen decoder weights, and CPU optimizer-resume reproduction. Actual GPU smoke runs exercise the full decoder/LPIPS backward path, validation, checkpoint selection, and sparse-table export. The original stage-1 parameter and buffer versions are checked during training.

Deterministic CUDA algorithms, a fixed cuBLAS workspace setting, and deterministic FP32 decoder convolutions make the GPU resume check bitwise reproducible. A four-step run resumed to step six exactly matched uninterrupted six-step training in model weights, optimizer tensors, data stream, evaluation token grids, and stopping state. A separate negligible-LR smoke run exercised early stopping at step two and exported the selected step-zero codebook. [Verification record](../outputs/church-learned-site-codec-20260911/verification.json).

The production pilot launched at 06:00 UTC on September 11 on GPU 0 and completed normally after 4,500 optimizer steps (3.43 hours, approximately 1.15 training passes). Its stopping rule triggered after four checks without a cumulative calibration LPIPS improvement of 0.001. The selected checkpoint is step 4,500; `best.pt`, `last.pt`, and the complete sparse lookup table `best-codebook.pt` are retained. The process has exited.

Before optimizer updates, the actual learned-codec path measured calibration LPIPS 0.35024, holdout LPIPS 0.34996, and validation LPIPS **0.35220** (validation PSNR **16.880 dB**). These initial results include the codeword OMP projection and deterministic decoder settings. They are the direct starting point for this run's improvement measurements. W&B confirmed the run was active, and initial training had finite gradients with zero coefficient clipping. [Launch health record](../outputs/church-learned-site-codec-20260911/launch-health.json).

- [W&B pilot](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-learned-complete16k-20260911).
- Outputs: `outputs/church-learned-site-codec-20260911/train`.
- Launcher: `scripts/launch_church_learned_site_codec.py`.
- Trainer: `scripts/train_church_learned_site_codec.py`.
- Codec: `src/learned_sparse_site_codec.py`.

## Completed result: compression remains inadequate

| Selected checkpoint split | LPIPS ↓ | PSNR dB ↑ | Relative latent MSE ↓ |
| --- | ---: | ---: | ---: |
| Calibration, 128 images | 0.31799 | 16.915 | 0.42280 |
| Holdout, 256 images | 0.31866 | 16.925 | 0.41710 |
| Validation, 300 images | 0.32125 | 16.819 | 0.41779 |

The pilot fails the existing compression acceptance criteria by a large margin. These are reconstruction metrics relative to continuous frozen-LASER reconstructions, not unconditional generation FID. No autoregressive prior has been trained on this vocabulary.

Validation LPIPS improved from 0.35220 to 0.32714 in the first 250 steps, then only to 0.32125 over the next 4,250 steps. Validation relative latent MSE increased from 0.36746 to 0.41779; PSNR did not improve. The modest perceptual improvement therefore did not recover the lost latent information. Training and evaluation LPIPS stalled at similar levels. Validation used 10,095 of 16,384 codewords with marginal entropy 13.020 bits, so widespread codebook collapse is not evident. These statistics do not establish token predictability.

The commitment/alignment mismatch increased from a mean of 0.691 in the training window ending at step 500 to 0.924 in the window ending at step 4,500. Projection mismatch also increased. Gradient norms remained finite and below the clipping threshold. The identity surrogate has not kept the encoder queries, dense codewords, and discrete reconstructions aligned well enough; this observation does not establish a global limit on learned tokenizers or prove a particular optimizer fix would work.

- [Training curves](../outputs/church-learned-site-codec-20260911/plateau-curves.png), with a [PDF version](../outputs/church-learned-site-codec-20260911/plateau-curves.pdf).
- [Completed training result](../outputs/church-learned-site-codec-20260911/train/results.json).
- [Training-window summary](../outputs/church-learned-site-codec-20260911/plateau-training-summary.json).

## Frozen plateau diagnosis

The diagnostic reloads the selected checkpoint and exported codebook, checks that atom/bin lookup reproduces every exported latent, and reproduces all saved calibration, holdout, and validation baseline metrics within 1e-5. All model weights remain frozen.

| Validation ablation, same 300 images | LPIPS ↓ | Relative latent MSE ↓ |
| --- | ---: | ---: |
| Actual learned assignment and complete sparse lookup | 0.32125 | 0.41779 |
| Nearest projected codeword in latent L2 | 0.34413 | 0.35731 |
| Learned assignment, dense codeword before sparse projection | 0.31723 | 0.41229 |
| Nearest dense codeword in latent L2 | 0.34050 | 0.35725 |

The learned assignments provide a real perceptual benefit over latent-nearest assignment, despite worse latent MSE. Removing OMP and coefficient quantization from the selected codewords only improves LPIPS by 0.00402 (paired SE 0.00021), leaving substantial distortion. This diagnostic bypass is not a complete sparse-code representation, and does not test training a different dense tokenizer from scratch.

A further restricted assignment search uses 32 known calibration targets and keeps the entire learned codebook fixed. Each site chooses among its current ID and its 32 nearest projected codewords. We optimize only candidate logits for 64 iterations using a categorical straight-through surrogate, assess hard IDs on every forward, and retain each image's best measured hard reconstruction. LPIPS changes from 0.30407 to 0.29133 (paired improvement 0.01274, SE 0.00145), while relative latent MSE increases from 0.41607 to 0.46376. The final selected IDs independently reproduce the retained metrics. The search uses the target image and is not a generator, a global assignment optimum, or a proof of a fundamental capacity bound.

- [Diagnostic results](../outputs/church-learned-site-codec-20260911/plateau-diagnosis/results.json).
- [Assignment-search images](../outputs/church-learned-site-codec-20260911/plateau-diagnosis/inversion.png): continuous reference, learned tokens, best searched hard tokens.
- Diagnostic script: `scripts/diagnose_church_site_codec_plateau.py`.

The user requested one integer for the complete sparse site; the 14-bit vocabulary was an implementation choice, not a user requirement. It reduces the nominal fixed-width code from 100 bits per site to 14 bits, excluding shared codebook storage. A larger integer can contain multiple internal fields without requiring a materialized vocabulary of every combination, but a predictor must still model those fields. A bounded follow-up measures 27-, 45-, and 63-bit residual-code integers, alongside a 63-bit integer preserving exact support plus a coefficient pattern, before considering further neural training.

That follow-up has completed: the best new representation, exact support plus a coefficient pattern in 63 bits, achieves validation LPIPS 0.03568 and relative latent MSE 0.01480. It is much better than the 14-bit codeword but still fails the existing calibration gate. [Complete larger-bit results and interpretation](lsun-church-site-integer-plateau-2026-09-11.md).
