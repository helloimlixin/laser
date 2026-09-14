# LSUN Church FID without image resizing

Completed September 14, 2026. Both real and generated images enter Inception directly at **256 × 256**, with `resize_input=False`. All real and generated features were recomputed for this setting.

| Model | Native-256 FID, 50,000 generated images |
| --- | ---: |
| Current LASER, three-epoch tokenizer fine-tune, stage 2 epoch 50 | **14.866753259798315** |
| Released RQ-VAE / RQ-Transformer pair | **9.53443274373194** |

Both scores use the same freshly computed reference of **126,227 unique real training images**. The model quality gap remains under this evaluation. These are a separate metric from standard FID with Inception's 299 × 299 input preprocessing; do not compare their numerical values directly with earlier or published standard FID scores.

## What happened to the pixels

- Real images retain the exact stage-1 center crop and RGB/value preparation. An audit of every real image found that all 126,227 already have a 256-pixel shorter side. The stage-1 `Resize(256)` call therefore returns the image unchanged in the installed TorchVision implementation: **zero real images undergo interpolation**. The center crop still changes the framing of nonsquare images.
- Generated images are decoded at their full native 256 × 256 resolution, converted to RGB values in [0, 1], and passed directly to Inception. There is no image resize or crop after decoding.
- Inception's 256 → 299 spatial resize is disabled for both populations. Its normal value normalization remains enabled.
- A hook at Inception's first block checked every batch's actual input shape, asserting `B × 3 × 256 × 256`: 1,263 real batches and 1,564 batches for each generated population.

## Reproducibility and scope

The comparison reuses each model's previously saved 50,000 autoregressive code sequences. The corresponding frozen tokenizers were verified against checkpoint and state hashes before and after decoding. Decoder batches contain 32 images, using FP32 with TF32 disabled. All features and real/fake statistics were freshly extracted at native 256; no standard-299 features were reused. Statistics and the Fréchet distance use the released repository's implementations.

This is a completed standalone evaluation. It does not change the active training job or its existing standard-FID logging.

Artifacts:

- Evaluator: [evaluate_church_native256_fid.py](../scripts/tools/evaluate_church_native256_fid.py)
- Combined scores: [result.json](../outputs/church-native256-fid-20260914/result.json)
- Full real-image dimension audit: [raw-dimension-audit.json](../outputs/church-native256-fid-20260914/raw-dimension-audit.json)
- Real transform and input-shape verification: [real/result.json](../outputs/church-native256-fid-20260914/real/result.json)
- LASER provenance: [laser/provenance.json](../outputs/church-native256-fid-20260914/laser/provenance.json)
- Released-model provenance: [published/provenance.json](../outputs/church-native256-fid-20260914/published/provenance.json)

Each population's directory also retains the 2,048-dimensional features, statistics, metric specification, completion status, and actual Inception input-shape checks.
