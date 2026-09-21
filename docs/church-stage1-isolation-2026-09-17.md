# Church stage-1 isolation, 2026-09-17

The matched reconstruction test does **not support native stage-1 reconstruction
quality as the main explanation for the Church generation gap**. The frozen
three-epoch LASER model has lower average perceptual and pixel error than the
released Church RQVAE checkpoint. Converting LASER to the active compact codebook
consistently worsens perceptual reconstruction error, but the resulting average
is close to the released RQVAE's.

This revises the earlier suspicion based on compact reconstructions alone.
Reconstruction fidelity does not measure how easily the transformer learns the
codes, and cannot rule out a tokenizer contribution to generated-image quality.

## Results

All 300 official Church validation images were reconstructed by every method
using identical original RQVAE image loading and preprocessing. Lower LPIPS/MSE
and higher PSNR are better. PSNR below is the mean of per-image PSNR values.

| Reconstruction path | Mean LPIPS | Mean PSNR, dB | Mean pixel MSE |
|---|---:|---:|---:|
| Native LASER: historical OMP, continuous coefficients | 0.24350 | 19.250 | 0.0140215 |
| Continuous greedy control | 0.24544 | 19.177 | 0.0142190 |
| Active compact LASER | 0.25414 | 18.953 | 0.0148281 |
| Released Church RQVAE | 0.25490 | 18.972 | 0.0146024 |

Compact conversion increases LPIPS by **0.01064, or 4.37%**, relative to native
LASER, with worse LPIPS on **300/300 images**. The paired bootstrap 95% interval
for this increase is [0.01016, 0.01111]. Mean pixel MSE increases by 5.75%, and
mean PSNR falls by 0.297 dB.

Native LASER has lower LPIPS than released RQVAE on 285/300 images; its mean
difference is -0.01140, with a paired interval of [-0.01217, -0.01064]. Compact
LASER's mean LPIPS differs from RQVAE by only -0.00076, while its mean pixel MSE
is slightly higher. These metrics support treating their average reconstruction
quality as similar, rather than attributing the much larger generation gap to
a clearly inferior LASER reconstruction ceiling.

The continuous greedy control has mean LPIPS only 0.00194 above native OMP;
compact LASER adds another 0.00870 relative to this control. This suggests that
the compact approximation contributes more of the observed reconstruction cost
than the OMP-to-greedy change. It is not a fixed-support coefficient ablation:
changing the coefficient values can also change subsequent atom selections.

## Interpretation

The representative and largest-degradation grids show loss of fine windows,
tracery, texture, and lettering across reconstruction methods. Compact LASER
adds visible local changes in some examples, but native LASER does not show an
obvious general collapse relative to the released reference.

The stronger remaining concern is stage-2 generalization and the learnability
of the compact codes. The existing run has increasing validation cross-entropy
while training loss falls: validation soft CE rises from 7.9545 at epoch 20 to
9.7813 at epoch 70, while logged training loss falls from 7.6229 to 5.1588.
Its best observed FID50k is 12.2719 at epoch 60; epoch 70 gives 13.9205.
These observations support investigating the prior, but this reconstruction
experiment does not establish the cause of the generation gap.

Training continued during this evaluation. No model, tokenizer, learning-rate,
sampling, or checkpoint-retention setting was changed.

## Protocol and validation

- Exact three-epoch LASER checkpoint and active compact codebook, restored through
  the running job's frozen loader and verified against its recorded hashes.
- Native quantization uses the checkpoint's historical stage-1 OMP implementation
  and configuration. All LASER paths share the same encoder and decoder; their
  codec source files match the stage-1 snapshot byte for byte.
- Released Church RQVAE stage-1 weights are loaded strictly and SHA-256 verified.
- Original LSUN LMDB decoding, bilinear resize to 256, center crop, and normalization
  to [-1,1]. All 36 recorded validation pixel probes match the audited loader.
- FP32 inference with TF32 disabled; models frozen and in evaluation mode.
  Metrics use reconstructions clamped to [0,1] before PNG quantization. LPIPS uses
  the released RQVAE VGG implementation with verified perceptual weights and
  inputs mapped back to [-1,1].
- Compact decoding matches the production decode path bit for bit on the first
  batch. The first four fresh encoder latents match the production validation
  cache exactly. LPIPS of each first-batch input against itself is zero.
- All three model/quantizer state hashes remain unchanged after inference.
- Paired intervals use 10,000 bootstrap samples with seed 20260917. They are
  descriptive and assume independent validation images. This is a 300-image
  reconstruction test, not a generated-image FID evaluation. Latent MSE is not
  comparable across separately trained LASER and RQVAE latent spaces.

The full evaluation completed in 30.5 seconds with peak GPU tensor allocation
of 2.123 GiB. A four-image preflight also passed.

## Artifacts and reproduction

- [Interactive viewer: all 300 images](../outputs/church-stage1-isolation-20260917/validation300/comparison.html)
- [Representative grid](../outputs/church-stage1-isolation-20260917/validation300/representative-grid.png): eight evenly spaced dataset indices, selected independently of error.
- [Largest compact degradation grid](../outputs/church-stage1-isolation-20260917/validation300/largest-compact-degradation.png): eight largest compact-minus-native LPIPS differences; intentionally selected, not representative.
- [Complete measurements, checksums, and verification](../outputs/church-stage1-isolation-20260917/validation300/result.json)
- [Per-image metrics and LMDB keys](../outputs/church-stage1-isolation-20260917/validation300/per-image.json)
- [Prior generation-quality diagnosis](../outputs/church-quality-diagnosis-20260917/diagnosis.md)

The diagnostic refuses to overwrite an existing output directory. Use a new
directory when repeating it in this workspace:

```bash
.venv-imagenet-stage2/bin/python scripts/tools/compare_church_tokenizer_reconstructions.py \
  --output outputs/church-stage1-isolation-repeat --images 300 --batch-size 4 --device cuda:0
.venv-imagenet-stage2/bin/python scripts/tools/render_church_reconstruction_comparison.py \
  outputs/church-stage1-isolation-repeat
```
