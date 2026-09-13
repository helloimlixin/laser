# Eight-level scaled-atom RQ stage 2 on LSUN Church

Launched a fresh RQTransformer to model the eight-level tokenizer that preserved
reconstruction quality in the frozen study. Every token jointly selects a
dictionary atom and a signed coefficient. Four residual selections at each 8×8
latent location use fixed earlier contributions, matching RQ semantics.

Run: https://wandb.ai/helloimlixin-rutgers/laser/runs/church-scaled-rq8-scratch-20260913

Output: `outputs/church-scaled-atom-stage2-20260913/train`.
Launch receipt, source snapshot, verification, calibration, and latent-cache
provenance are under `outputs/church-scaled-atom-stage2-20260913`.

## Model and tokenizer

- Released Church RQTransformer architecture: 24 spatial layers, four depth
  layers, width 1024, 16 heads. The classifier has 131,073 outputs. Total model
  size: 487,644,161 parameters.
- Fresh seed-zero initialization using the released initializer; no pretrained
  stage-2 weights or preflight checkpoint are loaded. Both ranks have identical
  initial weights and empty optimizer states.
- Initial weights SHA256:
  `a3f2db613ecb2210428f537ff9676d5d6676dbbe6533475c5a5f92df934ae256`.
- Frozen sparse Church checkpoint:
  `outputs/lsun-church-sign-pattern-20260910/assets/best_rfid_slot1_model.pt`, SHA256
  `93f19adc8ca2cfae0cf629d48d2d6f5f7039440b1a48ad1c7ba43cf17932a388`.
- Frozen expanded codebook:
  `outputs/church-scaled-atom-rq-20260912/sweep/scaled-atom-codebooks.pt`, SHA256
  `275bc44c7eed0b1b7f14308e49b1031ec8b4b0258a2dcb91ac2bc69ec6b8a1f1`.
- Nonzero levels: approximately `±[1.97670, 3.92997, 6.52144, 9.65730]`, shared
  across all depths; token zero denotes the unique zero vector. Token
  `1 + atom * 8 + bin` denotes `level[bin] * dictionary[:, atom]`.
- Fixed physical codebook vectors feed both the original spatial and depth
  conditioning paths. Support and coefficient both influence future predictions.

## Training targets and temperature

Blindly copying the original soft-target temperature would produce much more
distortion for this sparse tokenizer. Calibration on 128 Church training images
(indices 62,048–62,175) measured final latent reconstruction MSE after stochastic
codeword selection relative to hard quantization:

| Tokenizer / temperature | Sampled / hard latent MSE |
| --- | ---: |
| Original Church RQ tokenizer, 0.5 | 1.0153 |
| Expanded sparse RQ, 0.5 | 1.6210 |
| Expanded sparse RQ, 0.25 | 1.1563 |
| Expanded sparse RQ, **0.125** | **1.0292** |

The selected temperature is **0.125**: the largest tested temperature no greater
than 0.5 that stays within two percentage points of the original control's
sampled/hard MSE ratio. This is a calibration on training images, separate from
the coefficient-level fitting images and official validation images. It is not
a claim that the two latent spaces have identical noise distributions.

Targets use the full vocabulary, with each depth conditioned on its sampled
prefix. There is no additive physical coefficient noise. The temperature above
controls training targets; generation retains the released temperature 1.0,
top-k 250, top-p 1.0 sampler.

Intermediate target computation and FP32 soft cross entropy use chunks of 128
latent rows. The implementation preserves all probability mass and recomputes
the CE gradient in chunks. It does not truncate training targets to top-k.

## Optimization and evaluation

- Two H200 GPUs, microbatch 128 per GPU, effective batch 256, one microbatch per
  optimizer update. This was approximately 10% faster than microbatch 32 with
  four accumulation steps in the initial benchmark. Peak allocated memory was
  56.10 GiB per GPU during the training preflight.
- All 126,227 Church training images cached from the correct encoder in FP32,
  with the released resize/center-crop transform. CUDA matmul and cuDNN TF32 are
  disabled. Stochastic targets are regenerated each visit. Official validation
  uses all 300 images.
- Released AdamW setup: initial LR 0.0005, betas (0.9, 0.95), weight decay 0.0001,
  gradient clipping at 1.0, 300-epoch cosine schedule, no warmup.
- FP16 model autocast with GradScaler; FP32 tokenizer geometry and CE.
- A downward-only LR multiplier follows the user's request to react to FID
  regression or convergence. It halves after two consecutive fixed-seed 4,096
  sample FIDs exceed the best by 0.25, or after three checks without at least
  0.1 improvement, with at least ten epochs between reductions. Comparisons use
  only the same 4,096-sample protocol, never a mixture of 4k and 50k FIDs.
- The multiplier is applied once to the underlying cosine schedule; a test
  verifies that PyTorch's recursive cosine update does not compound it at each
  optimizer step.
- Occasional GradScaler skips are recorded. Eight consecutive skips trigger a
  saved failure; the new run has no cumulative 25-skip termination limit.
- Validation and 4,096-sample generation FID after epoch one, then every five
  epochs. Additional 50,000-sample FID every 50 epochs.
- Full model/optimizer/scheduler/scaler/two-rank RNG checkpoints at update 25
  and every epoch; model-only archives every ten epochs. Graceful termination
  saves the current state. Failure records prevent a stale training status from
  being the only evidence of a stopped process.

## Verification

Eleven tests passed across the tokenizer, new training helpers, and original
gradient-accumulation checks. They cover expanded-codebook parity, stochastic RQ
targets, support/coefficient conditioning and causal masking, exact chunked CE
loss/gradient, LR response, and partial-batch weighting.

The full 488M model completed three two-GPU preflight updates at the production
batch size with finite loss/gradients and zero AMP skips. The checkpoint reloaded
strictly; every model and optimizer tensor was finite, all 460 optimizer entries
and both RNG states were present. On the reloaded full model, chunked CE and the
released dense CE both measured 11.7160034. The released cached sampler produced
valid code IDs and finite decoded images. Those images are integration checks
after three updates, not evidence of generation quality.

The launcher verifies and snapshots 84 source files, including the pinned
released architecture at commit `341395e562ac347f5eb62db9f5f08b9f2cc42a60`.
The released repository did not include its stage-2 training loop; this run uses
the local verified driver with its model, optimizer, scheduler, and sampler.

The previous original-RQ baseline had already stopped around epoch 152 due to its
cumulative AMP-skip guard. Its epoch-150 50k generation FID was 15.8196. This new
launch neither resumed that model nor loaded its stage-2 weights.
