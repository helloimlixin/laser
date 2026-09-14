# Published Church RQ reproduction audit

The matched released checkpoint pair scores **7.6710 FID on 50,000
generated images** through our evaluation pipeline. The published value is
7.45, leaving a 0.2210 difference. This recovers the released
model's quality closely, although it is not an exact reproduction of 7.45.
The residual difference was not isolated with repeated seeds or hardware
comparisons. It is much smaller than the local baseline's gap.

## Controlled sampler comparison

The original local baseline's exact epoch-50 transformer and its own fine-tuned
tokenizer were evaluated with the same seed, 100 samples per GPU, both GPUs,
precision settings, reference statistics, and 4,096 generated images each.
Only top-k changes; sampling temperature remains 1.0 and top-p remains 1.0.

| Top-k | FID4096 |
|---|---:|
| Local YAML: 250 | 16.7153 |
| Released checkpoint config: 1400 | 15.6679 |

Changing top-k improves this diagnostic by 1.0474 FID. These are
4,096-image screens, not new FID50k values. The original epoch-50 FID50k remains
14.5340 at top-k 250. Sampling comparisons do not alter any model weights.

## Reconstruction quality before stage 2

All values below use the same first 4,096 Church training images, original
pixels, FP32 Inception and matched original/reconstructed populations, with
TF32 disabled. The original-feature cache exactly matched fresh Inception
features on the first 16 images. All tokenizer state hashes stayed unchanged.

| Tokenizer | Matched rFID4096 | PSNR from mean MSE, dB |
|---|---:|---:|
| Released Church RQVAE | 8.1311 | 18.5748 |
| Locally fine-tuned original RQVAE | 11.7208 | 15.8716 |
| Compact 32k sparse tokenizer, previous matched screen | 8.3923 | 18.3775 |

The local original-RQVAE tokenizer has a quality deficit before any prior is
trained. Its pixel MSE is 1.8634 times the released tokenizer's. The compact
sparse tokenizer's reconstruction is much closer to the released RQVAE.
This comparison does not prove which fine-tuning change caused the deficit,
and reconstruction FID is not an additive decomposition of generation FID.

## Reconciled settings

| Setting | Local original baseline | Released evidence |
|---|---|---|
| Transformer architecture | 370,087,936 parameters; 24+4 blocks, width 1024 | Exact match after applying released defaults |
| Tokenizer fine-tuning LR, generator and discriminator | 4e-5 | 4e-6 in checkpoint config and paper |
| Tokenizer fine-tuning duration | One completed epoch | Config lists 3; checkpoint stores no epoch; paper says 1 |
| Stage-2 global batch | 256 | Paper says 2048; checkpoint omits training settings |
| Optimizer updates per epoch on this dataset | 494 | 62 with global batch 2048 |
| Updates in a 300-epoch cosine schedule | 148,200 | 18,600 with global batch 2048 |
| Sampling top-k / top-p / temperature | 250 / 1.0 / 1.0 | 1400 / 1.0 / 1.0 in checkpoint config |

The local baseline followed the checked-in Church YAML. That YAML differs from
the configurations bundled with the released checkpoints and from the paper.
The original stage-2 training loop was not released, so its optimizer parameter
grouping and initialization call site cannot be verified from the weight-only
checkpoint. No alternative training recipe was launched by this audit.

The clean next control is a fresh prior trained against the **frozen released
Church tokenizer**, with the paper's global batch 2048 and corresponding cosine
update count, and the released sampling settings. This isolates stage-2 training
from the tokenizer fine-tuning discrepancy. If repeating the user's requested
one-epoch ImageNet-to-Church fine-tuning, use 4e-6 for both optimizers and verify
reconstruction before launching stage 2. The one-epoch request should not be
silently changed to three based on configuration metadata alone.

Concrete settings are recorded in `outputs/church-published-audit-20260913/reconciled-settings.json`. They also call for
retaining every best fixed-protocol FID checkpoint and replacing the former
cumulative AMP-skip guard with a consecutive-failure policy.

## Verification and provenance

The Church model archive's MD5 matches `deeb3e0ac6e09923754e3e594ede7b01`.
Both models load strictly and contain finite weights. The FID reference matches
the official archive byte-for-byte. The sampler was checked at all 256 cached
positions against the full causal model; maximum FP32 logit discrepancy was
0.00006295. Batched versus serial decoding differed by at most 0.00001454 on
the checked published samples. Independent NumPy aggregation of all 50,000
Inception features differs from our streaming FID by only
0.000001515. Frozen model/tokenizer hashes were
verified before and after generation.

Outputs include every generated token code, Inception features, statistics,
source/config hashes, sample grids, and the exact evaluation commands. An
initial audit stopped before sampling because it over-scoped its source check
to an unrelated sparse adapter changed elsewhere in the shared workspace; the
corrected audit verifies its own 76 dependency files. A second preparation issue
was reading trusted local stage-1 metadata with a tensor-only loader; the
corrected loader permits that format only for the exact hash-verified local
checkpoint. Failed startup logs are retained separately from results.

Sources: [official repository and checkpoint](https://github.com/kakaobrain/rq-vae-transformer),
[paper training details](https://arxiv.org/html/2203.01941#A3).
