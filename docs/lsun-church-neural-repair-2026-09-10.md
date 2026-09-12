# Church neural latent repair — September 10, 2026

This experiment trains **only a new 4,887,040-parameter repair network**. The existing Church tokenizer, dictionary, coefficient bins, decoder, and epoch-50 autoregressive prior remain frozen. No stage-1 or stage-2 retraining is included.

Training completed all 3,000 steps (96,000 image presentations) in [W&B](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-neural-repair-20260910). The trained repair layer **did not improve generation quality** in this experiment: paired independent FID-50k changed from **13.70469 to 13.78753** (+0.08284, lower is better). All existing model weights remained frozen. The layer is saved for inspection and is not enabled in the generation pipeline.

## Independent 50,000-sample comparison

The frozen epoch-50 AR prior generated 50,000 new images using seed 17701, separate from the 2,048-sample screen (seed 12701). The baseline and repair used the exact same atom/coefficient tokens, original RQ-VAE Inception implementation, and LSUN Church reference statistics. Sampling used atom top-k 250 and temperature 1 for both atom and coefficient predictions. The final trained checkpoint was evaluated even though the holdout selection rule chose identity.

| Measurement | Unchanged decoder | Trained final repair layer |
|---|---:|---:|
| FID-50000 | 13.70469 | 13.78753 |
| Inception feature covariance trace | 87.68289 | 87.04239 |
| Perceptual distance between 256 fixed generated-image pairs | 0.688497 | 0.687301 |

The small adverse FID change is not a formal significance claim, but neither the screening comparison nor the larger independent comparison shows a quality gain. The diversity diagnostics decrease slightly and do not establish mode coverage. Synthetic denoising improved, while gains on long fixed-support coefficient rollouts were small and the held-out anchored AR-error recovery score did not improve. This does not rule out other error-correcting models; it does not justify deploying or scaling this version.

[Independent metrics](../outputs/lsun-church-neural-repair-20260910/independent-50000-comparison.json) and [paired images](../outputs/lsun-church-neural-repair-20260910/independent-50000-final-repair/paired-images.png) are saved. W&B contains the FID-50k summary, paired sample grid, evaluation artifact, and the actual trained `final.pt` model artifact.

## Completed pilot results

| Measurement | Unchanged decoder | Trained final repair layer |
|---|---:|---:|
| FID-2048 on identical generated tokens | 19.05509 | 19.08359 |
| Held-out anchored AR-error LPIPS, 256 images | 0.060994 | 0.061149 |
| Held-out clean reconstruction LPIPS drift | 0 | 0.000396 |
| Validation clean reconstruction LPIPS drift, 300 images | 0 | 0.000761 |
| Validation coefficient rollout from site 8: LPIPS | 0.389157 | 0.387802 |
| Validation coefficient rollout from site 40: LPIPS | 0.180090 | 0.179931 |
| Synthetic held-out corruption LPIPS, 128 images | 0.183373 | 0.175388 |
| Synthetic held-out corruption PSNR | 19.2070 dB | 19.5639 dB |
| Fixed generated-image-pair LPIPS, 256 pairs | 0.695051 | 0.693787 |

The fixed checkpoint-selection rule chose the initial identity (`best.pt`, step 0). Every trained candidate scored worse on held-out anchored AR-error recovery after accounting for clean-image drift. The table evaluates the **actual trained `final.pt` at step 3,000**, not the identity. Its synthetic-noise improvement does not establish recovery of AR mistakes or improved unconditional image generation. The coefficient-only long-rollout gains are small, and the FID screen is essentially unchanged.

The synthetic-noise diagnostic uses a fixed fresh corruption seed on 128 holdout images, excluded from training. It was added after the step-1,500 result to distinguish basic denoising from transfer to AR mistakes; it was not used to select checkpoints. At the final step its paired PSNR change is +0.357 dB with standard error 0.069 dB, while LPIPS changes by -0.00798 with standard error 0.00144.

[Training curves](../outputs/lsun-church-neural-repair-20260910/training-diagnostics.png), [paired generated images](../outputs/lsun-church-neural-repair-20260910/train/final-generation/paired-images.png), [full pilot results](../outputs/lsun-church-neural-repair-20260910/train/results.json), and [trained checkpoint](../outputs/lsun-church-neural-repair-20260910/train/final.pt) are saved locally. Runtime checks confirmed that all base-model parameters and buffers remained unchanged.

## Implementation

Sparse codes are converted to their physical 256-channel, 8x8 latent grid. A six-layer bidirectional transformer of width 256 predicts a residual in that same space. Its output layer starts at zero, giving exact identity behavior before training. Normalization statistics use training images only. The repaired continuous latent is passed through the unchanged post-quantization projection and decoder.

The result is a latent adapter, not a newly learned tokenizer or a guarantee of integer-token correction. Existing atom/coefficient IDs and the AR sampling process remain unchanged. The adapter does not reproject its output into four sparse pairs.

Training uses 8,192 image-disjoint training examples with corrupted spans generated by the frozen AR checkpoint. Half of the batches generate both atom and coefficient predictions; half generate only coefficients with fixed true supports. Spans cover 4, 8, or 16 sites, with clean tokens before and after the span. The exterior anchors pair each corrupted representation with a known clean target. Ground-truth values within generated spans are not read by the sampler, except explicitly forced atoms in the coefficient-only condition.

Each batch of 32 mixes eight clean images, eight images with synthetic atom/coefficient errors, and sixteen cached AR-corrupted examples. Clean and synthetic examples are sampled from the 125,203-image repair-training split. Synthetic errors include nearby dictionary atom swaps, occasional unrelated atom swaps, coefficient perturbations, and sign flips; atom supports remain distinct. Targets are the clean quantized latent and its original frozen-decoder reconstruction.

Only the repair weights receive optimizer updates. The loss is normalized latent MSE + 0.5 LPIPS + 0.1 pixel L1, with clean examples weighted four times. AdamW uses a learning rate of 3e-4, 100 warmup steps, and cosine decay to 3e-5 over 3,000 steps. A batch is accumulated in microbatches of eight. Decoder and perceptual-loss weights are frozen, while their input gradients reach the repair layer.

## Evaluation fixed before training

Every 500 steps, evaluate 256 held-out examples spanning both AR corruption modes and all three span lengths. Select the smallest `noisy LPIPS + 2 * clean LPIPS`, requiring clean-image LPIPS drift at most 0.01. The initial identity is eligible. Holdout images are excluded from repair learning; the existing pretrained tokenizer and AR prior previously saw the original training population.

After training, evaluate the selected checkpoint and, if different, the final checkpoint on all 300 official validation images, the earlier fixed-support coefficient rollouts, and the exact same 2,048 unconditional generated samples. Report paired FID, Inception feature dispersion, perceptual distance between 256 fixed image pairs, clean-image drift, and corruption-recovery metrics. These diversity diagnostics do not establish full mode coverage. FID-2048 is a screening estimate; a positive result would still need a larger independent comparison.

The initial [frozen encode/decode cycle](lsun-church-token-error-correction-2026-09-10.md) was not a useful error corrector. That result does not determine the outcome of this learned experiment.

## Verification and artifacts

- Eight focused tests passed, including exact initial identity, input-gradient flow through a frozen decoder, causal sampling without clean-target leakage, and valid synthetic sparse supports.
- A completed GPU smoke trained the actual repair layer through the real Church decoder, evaluated both selected and final weights, and verified frozen base parameters and buffers.
- Save/resume restored the data and corruption RNG states. An initial comparison across GPUs differed numerically because convolution gradients were nondeterministic; the final trainer enables deterministic cuDNN. Bitwise reproducibility across GPUs is not claimed.
- [Pair-generation configuration](../outputs/lsun-church-neural-repair-20260910/pairs/config.json), [training configuration](../outputs/lsun-church-neural-repair-20260910/train/config.json), and [source snapshot hashes](../outputs/lsun-church-neural-repair-20260910/source-snapshot/sha256.json) record provenance.
- [Repair model](../src/sparse_latent_repair.py), [pair builder](../scripts/build_church_repair_pairs.py), [trainer](../scripts/train_church_latent_repair.py), and [standalone evaluator](../scripts/evaluate_church_latent_repair.py).

To evaluate a saved repair checkpoint against another set of generated codes:

```bash
TORCH_HOME=/workspace/tmp/official-rqvae-eval-cache \
LASER_VGG16_WEIGHTS=/workspace/tmp/laser-vgg/vgg16-397923af.pth \
/tmp/laser-sign-venv/bin/python scripts/evaluate_church_latent_repair.py \
  --checkpoint outputs/lsun-church-neural-repair-20260910/train/final.pt \
  --codes outputs/lsun-church-bar-20260910/source-baseline/generated-codes.pt \
  --output outputs/church-repair-evaluation
```
