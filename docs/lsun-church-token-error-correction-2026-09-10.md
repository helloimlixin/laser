# Church token error correction: frozen baseline and proposed repair layer

The proposed direction is a context-aware denoiser for the sparse representation. The existing encoder, dictionary, coefficient bins, decoder, and AR prior remain frozen while a separate repair network is trained. The user authorized this new-layer-only experiment with “pls proceed.” The previously prepared long scratch priors have not been launched.

## Why this could help

A generated atom/coefficient pair can be individually valid but inconsistent with neighboring image content. Integer IDs alone do not identify that inconsistency. A denoiser could learn to use the surrounding sites to correct some such errors. It cannot uniquely recover an intended image when the generated tokens are equally consistent with another plausible image.

For LASER, the useful geometry is the physical latent

\[
z_{hw}=\sum_{d=1}^4 c_{hwd}D_{a_{hwd}},
\]

not numerical distance between atom IDs. Different sparse supports can also approximate the same latent, so exact token recovery is not the only meaningful objective.

The connection to perturbation methods is denoising and robustness training. Adding differential-privacy noise does not by itself supply a repair rule. Genuine error-correcting codes require redundancy and assumptions about which errors occur; computing parity from an already incorrect sampled token does not identify the intended token.

## Completed inference-only experiment

One frozen cycle was tested:

`AR codes -> frozen decoder -> frozen encoder -> OMP4 -> existing coefficient bins -> frozen decoder`

The encoder, FP32 OMP, float16 coefficient cache round-trip, and binning match the established Church cache recipe. Decoding and the original RQ-VAE Inception metric use FP32. No weights were updated, and no hyperparameters were selected.

The unconditional comparison uses the exact same 2,048 saved codes from the source epoch-50 prior. The conditional comparison uses 300 official validation images and the earlier sampled-coefficient rollouts with true atom supports. Those rollouts are a stress test, not unconditional generation.

| Measurement | Original decode | After one frozen cycle |
|---|---:|---:|
| Unconditional FID-2048, lower is better | 19.05509 | 19.02332 |
| Inception covariance trace, dispersion diagnostic | 87.66739 | 89.02200 |
| Rollout from site 8: PSNR to clean reconstruction | 14.25571 | 13.93394 |
| Rollout from site 8: LPIPS to clean reconstruction | 0.38916 | 0.39745 |
| Rollout from site 40: PSNR to clean reconstruction | 18.30452 | 17.27438 |
| Rollout from site 40: LPIPS to clean reconstruction | 0.18009 | 0.22232 |
| Clean codes: LPIPS drift from their original reconstruction | 0 | 0.08764 |

FID is essentially unchanged at this screening size; the 0.032 difference is not evidence of a reliable quality improvement. The known-error reconstructions get worse. Paired PSNR changes are -0.322 dB (standard error 0.008) and -1.030 dB (standard error 0.039), respectively. The clean-code cycle also changes already reconstructed images. Covariance trace alone cannot establish image diversity or mode coverage.

This result rejects this particular consistency pass as a demonstrated error corrector. It does not test or rule out a trained denoiser.

Artifacts: [full metrics](../outputs/lsun-church-frozen-repair-20260910/main/summary.json), [paired generated images](../outputs/lsun-church-frozen-repair-20260910/main/paired-unconditional.png), [probe script](../scripts/probe_church_frozen_token_repair.py).

## Learned experiment

Implementation and the 3,000-step pilot are described in the [experiment record](lsun-church-neural-repair-2026-09-10.md). The following describes its design; results are recorded separately.

1. Put a small bidirectional spatial network on the 8x8 physical latent grid. Use an identity-initialized residual output so its initial behavior reproduces the current decoder input.
2. Build pairs from clean training codes and realistic corruptions: nearby atom substitutions, coefficient/sign errors, and short AR-generated spans anchored to the same training image. Exclude holdout and validation images from learning. Never assume an unrelated unconditional sample has a particular real-image target.
3. Train only the repair layer to recover the clean latent and its frozen-decoder reconstruction, using latent and perceptual losses. Include clean inputs with an identity loss. A latent MSE-only model could average ambiguous alternatives and smooth details.
4. First measure clean preservation and held-out corruption recovery. Then decode exactly the same unconditional samples with and without repair and compare FID, paired images, and diversity measures. A conditional reconstruction gain alone is insufficient.
5. If integer corrected tokens are required, separately test projecting the repaired latent through the existing dictionary and coefficient bins. A continuous latent adapter does not automatically produce a valid four-pair sparse code; projection introduces another approximation.

This experiment can use the current AR checkpoint. Retraining the prior is not necessary to test whether a repair layer helps it.

## Closest research precedents

- [RobusTok, Image Tokenizer Needs Post-Training](https://arxiv.org/html/2509.12474v1): perturbed-token reconstruction and decoder adaptation to generated latents. Its post-training freezes the encoder/quantizer but updates the decoder. The proposed separate adapter is an adaptation of the idea, not a replication.
- [Residual Decoder Adapter](https://github.com/CSU-JPG/RDA): learns a residual decoding branch while preserving the base tokenizer and AR model. Its application is text rendering; it does not establish correction of Church sparse-code errors.
- [Infinity, section 3.4](https://arxiv.org/html/2412.04431v1): simulates bit mistakes and recomputes later residual targets. Its self-correction is learned during training, not conferred by representing IDs as bits.
- [reAR, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/7d4647db960780c8b5a4e7d9e4a58d68-Abstract-Conference.html): predicts/reconstructs visual embeddings under noisy context while preserving the tokenizer. It requires AR training, so it is relevant only if that scope is allowed again.

These are relevant mechanisms, not evidence that an untested LASER adaptation will reach the papers' reported image-generation quality.
