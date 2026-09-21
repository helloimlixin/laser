# Church integer stage 2 with the selected stage-1 checkpoint

Fresh 90-epoch training requested on 2026-09-21. The experiment is isolated in
`outputs/church-integer-ft3best-scratch90-20260921`; `supervise.py` builds the
cache, runs a full-batch preflight and an actual checkpoint resume, and then
starts production training from random transformer weights. Status and logs
remain in this directory after the interactive session ends.

W&B run: `helloimlixin-rutgers/laser/church-laser-ft3best-integer-scratch90-h200x8-20260921`.

## Frozen stage 1 and integer conversion

- Selected tokenizer: `outputs/church-stage1-improve-20260920/selected/tokenizer.pt`.
- SHA-256: `762c51a10267ed6fa55709ff0d6cf997940d21056b16c7a0a0399b3ebf93868d`.
- Full-model Church fine-tuning: three epochs at LR 1e-5; selected full
  reconstruction FID 2.6393383730869004. This score describes the original sparse
  bottleneck; the integer conversion has a separate reconstruction screen.
- Reuse the earlier adaptive scaled-atom integer RQ recipe: one zero token plus
  16,384 atoms times two fitted signed coefficient levels, yielding 32,769 tokens.
  Each image has an 8 x 8 x 4 code grid and a shared vocabulary.
- Fit atom-specific levels on 4,096 cached training latents, eight passes,
  prior weight four. Recalibrate the scalar target temperature against the
  original RQVAE stochastic/hard residual distortion control.
- Prebuild FP32 encoder latents for all 126,227 training images, all 300
  validation images, and a deterministic uint16 integer-code map. Training
  keeps the latent cache in host RAM and recomputes stochastic integer contexts
  and soft targets on each visit. The frozen encoder is never run in the
  training loop. Fixed hard targets would change the recovered recipe.

## Training and evaluation

- 386,882,561-parameter RQTransformer, fresh seed-zero initialization.
- Eight H200 GPUs; 256 images per GPU, global batch 2,048, no accumulation.
- 90 epochs, 62 updates per epoch; 5,580 updates if AMP skips none.
- AdamW: LR 5e-4, betas 0.9/0.95, weight decay 1e-4, gradient clipping 1.
  Published cosine schedule over the requested 90 epochs; no FID-triggered
  LR reductions. This preserves the September 20 integer experiment's schedule.
- FP16 model autocast with FP32 quantization geometry and chunked soft-target
  cross-entropy; TF32 disabled. No auxiliary geometry loss.
- Fixed-seed 100-image previews every 500 successful optimizer updates.
- Integer-stream sampling: top-k 1,400, top-p 1, temperature 1.
- FID on 50,000 generated images at epochs 1, 10, 20, ..., 90. Original RQVAE
  Inception and the same 126,227-image real reference used in the comparison.
- Each epoch saves full `last.pt`; the best measured FID retains a full
  recoverable checkpoint. Online uploads run in a bounded background worker
  and verify remote COMMITTED status, byte counts, and MD5 digests. The frozen
  tokenizer, fitted integer codebook, configuration, and calibration are also
  uploaded. Finalization drains pending uploads before marking completion.

## Reproducibility and recovery

The prior integer dependency snapshot is copied into `stage2-source`, including
the recovery and target-policy helpers. Its SHA-256 manifest is checked before
training. Drivers and launch inputs have a separate `launch-manifest.json`.
Stage-1 epoch provenance correctly records three epochs throughout.

The preflight uses the real full cache and batch size, takes two updates, saves
all eight RNG states with optimizer/scheduler/scaler state, samples and decodes
on all eight ranks, then resumes for one further update. Its weights are never
loaded into production. Production progress is in `train/status.json`, W&B
details in `train/wandb.json`, and verified upload receipts in
`train/checkpoint-upload.json` and `train/tokenizer-upload.json`.

The old integer driver's final status log ran after W&B was finished. This
isolated driver clears the finished run before writing final local status,
preventing a completed experiment from being falsely reported as failed.

Cache preparation completed in 136 seconds. The calibrated scalar target
temperature is 0.125. The matched 4,096-image integer reconstruction FID is
6.194472037608477, compared with 9.182462344814155 for the previous one-epoch
integer tokenizer. Their pixel PSNRs are 18.3981 and 19.1164 respectively; the
perceptual FID improvement does not imply improved pixel distortion.

Production started online after both preflights passed. The observed training
rate over optimizer steps 10–20 was 0.5949 seconds/update, or 3,442 images/sec
across eight H200s (excluding evaluation and checkpoint I/O). No AMP updates
were skipped in the first epoch. The remote W&B configuration was checked for
90 epochs, global batch 2,048, stage-1 epoch 3, fresh stage 2, LR 5e-4, 500-step
previews, and sampling top-k 1,400 / top-p 1 / temperature 1.

Throughput fixes were requested after launch. See
[the optimization audit](church-integer-throughput-2026-09-21.md). The same
online run continues from step 1,045 using the isolated
`optimizations/throughput-v1` runtime; the original runtime remains preserved.
