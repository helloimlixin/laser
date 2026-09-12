# Church joint-pattern prior: poor generation diagnosis

**Follow-up:** the user authorized the coefficient-first experiment. The previous prior has now been gracefully paused and preserved, and the [matched ordering pilot](lsun-church-pattern-order-2026-09-11.md) is running. Statements below about continued training describe the earlier diagnostic session.

The 67-bit representation preserves reconstructions, but the current support-first prior generates badly warped buildings. The epoch-10 checkpoint has FID-4096 **51.0480**. The running model subsequently improves to **45.0732 at epoch 12**, so the available curve is still improving; these results do not establish a plateau or that longer training cannot help.

The frozen diagnostics use epoch 10, SHA-256 `df12006ce663eb4d29e80fb2d57087189682f7db713386761b6a8cab8ce98b26`. The prior, tokenizer, dictionary, pattern codebook, and decoder receive no updates. Existing training sources and schedules are intact.

## Atom prediction adds little information at later depths

On the first 256 held-out images, compare the model with per-depth atom frequencies fitted on all 125,203 training images. Frequencies use a 0.5 pseudocount per atom. Both the model and the frequency baseline exclude already selected atoms. These are natural-log NLLs; lower is better.

| Atom depth | Frequency-only NLL | Model NLL | Model exact top-1 accuracy |
| --- | ---: | ---: | ---: |
| 1 | 8.86561 | 7.64411 | 5.249% |
| 2 | 9.68928 | 9.58668 | 0.0305% |
| 3 | 9.68252 | 9.65291 | 0.0244% |
| 4 | 9.66295 | 9.65489 | 0.0244% |

The fourth atom gains only **0.00806 nats** over frequencies alone. Low exact-token accuracy does not itself establish poor image quality: many sparse codes can represent plausible images. Together with the weak likelihood gain and inspected samples, it identifies a poorly learned part of this particular prior.

## Coefficient predictions drift even with correct support

Supply every real atom identity at every site. Compare the argmax joint coefficient pattern using real preceding spatial patterns with the same decision using its own preceding pattern predictions. No atom is sampled or replaced. Reference images decode the real support and its nearest calibrated pattern. All 256 images are evaluated; the grid shows the first eight in cache order.

| Fixed-support condition | Sign accuracy | Physical coefficient MAE | LPIPS to true-code reconstruction | PSNR |
| --- | ---: | ---: | ---: | ---: |
| Real preceding patterns | 77.90% | 1.6127 | 0.26239 | 17.264 dB |
| Generated preceding patterns | 68.12% | 2.5130 | 0.41992 | 14.184 dB |

This is a conditional reconstruction stress test, not unconditional FID. It demonstrates substantial coefficient error even with correct atoms, and further drift when predictions become context. It is not an additive decomposition of the causes of unconditional image defects. Joint-pattern posterior-mean coefficients also have only **78.39%** sign accuracy with real history, so switching from argmax to the mean does not remove this failure.

[Fixed-support comparison](../outputs/church-support-pattern-integer-20260911/generation-diagnosis/fixed-support.png): columns are true-code reconstruction, coefficient prediction with real history, coefficient rollout with generated history. [Measurements](../outputs/church-support-pattern-integer-20260911/generation-diagnosis/diagnostics.json) and [per-image values](../outputs/church-support-pattern-integer-20260911/generation-diagnosis/per-image.pt).

## Numerical and causal checks

On two real held-out images, all 64 sites and four atom depths are replayed with true history. FP32 cached generation and full teacher forcing agree to a maximum logit difference of **0.0000172**, for both atom and pattern heads. Their maximum probability total-variation differences are below 0.0000032. This check passes on the actual trained 201M model.

BF16 uses SDPA during full teacher forcing and explicit attention while caching. It introduces rounding differences: mean total variation is **0.00140** for atoms and **0.00829** for patterns, with maximum pattern TV **0.03876** in this two-image check. Argmax IDs need not match around close ties; an initial diagnostic assertion requiring exact BF16 argmax equality was replaced with distribution measurements and FP32 checks. No training or sampling implementation was changed in response. These numerical measurements alone do not establish the effect of FP32 sampling on FID.

The frozen checkpoint also reproduces the saved first **128 unconditional atom and pattern grids exactly**, with the original seed and batch size. [Full-grid cache check](../outputs/church-support-pattern-integer-20260911/generation-diagnosis/cache-check.json).

## Frozen-checkpoint sampling result

Narrowing the joint-pattern nucleus to **p=0.5**, with atom top-k 2048 unchanged, improves the same epoch-10 checkpoint's **FID-4096 from 51.04804 to 44.05895**. The inspected first 32 samples still contain warped towers, blank patches, and smeared facades. This is a sampling improvement, not a demonstrated solution to image quality.

Four settings were first screened on 1,024 images, with fixed seed 18701 and batch 128:

| Atom top-k | Joint-pattern nucleus | FID-1024 |
| --- | ---: | ---: |
| 2048 | Full distribution | 58.63330 |
| 250 | Full distribution | 61.49547 |
| **2048** | **0.5** | **54.11582** |
| 250 | 0.5 | 57.60597 |

The winner was extended to 4,096 images and compared with the archived 4,096-image baseline from the same checkpoint, seed, batch size, tokenizer, decoder precision, and Inception reference. The first 1,024 images are included in the larger result, so this is not independent confirmation. No FID-50,000 was run. Neither checkpoint selection across epochs nor sampling selection across these four settings is accounted for by a confidence interval.

[Improved-sampling grid](../outputs/church-support-pattern-integer-20260911/generation-diagnosis/pattern05-4096/samples.png) · [Full comparison](../outputs/church-support-pattern-integer-20260911/generation-diagnosis/results.json).

## Interpretation and next experiment

The codec gate tested distortion from quantizing a known sparse code. It did not test whether a prior could predict that code. Generation now requires four atom choices before the joint coefficient pattern, so within-site support prediction has no coefficient signs or magnitudes available. The measured weak atom modeling and coefficient drift justify investigating this ordering. They do not isolate ordering from the different parameter count, objective weighting, learning-rate schedule, or amount of training.

For context, the calibrated hard pair model had FID-4096 **29.4430 at epoch 10**, using the same sample count, seed, and Inception reference. Its architecture, objective, LR, and scalar-coefficient nucleus sampling differ. The support-first result is a practical regression against that checkpoint, not a controlled ordering ablation.

A bounded next architecture comparison could predict the joint coefficient pattern first, then condition all four atom choices on it and on weighted partial sparse reconstructions. This still defines the same complete-site integer and the same decoder. It changes the probability factorization rather than further compressing the representation. That is an untested hypothesis requiring a matched training comparison; it was not launched during this diagnostic. Existing training continues under its recorded stopping rules.

Reproduction uses `scripts/diagnose_church_support_pattern_generation.py` and `scripts/check_church_support_pattern_cache.py`, with the saved frozen checkpoint. Sampling sweeps use temperature 1, BF16 AR, FP32 decoding, the original RQ-VAE Inception, seed 18701, and batch 128. Nucleus probability scans move to CPU only inside the standalone diagnostic, preserving strict deterministic execution without changing any live trainer source.

The reusable `scripts/sample_church_support_pattern.py` accepts a frozen checkpoint, atom top-k, pattern nucleus cutoff, sample count, and seed. Its default is atom top-k 2048 / pattern p=0.5 / temperature 1. It saves complete-site integers and verifies both exact support and pattern recovery. A 128-image run reproduces the diagnostic sampler's atom and pattern fields exactly; all **8,192** integers independently recover the original support and all signed coefficient-bin IDs. [Sampler verification](../outputs/church-support-pattern-integer-20260911/generation-diagnosis/sampler-verification.json).

For example, with the existing evaluation environment:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
TORCH_HOME=/workspace/tmp/official-rqvae-eval-cache \
LASER_VGG16_WEIGHTS=/workspace/tmp/laser-vgg/vgg16-397923af.pth \
/tmp/laser-sign-venv/bin/python scripts/sample_church_support_pattern.py \
  --checkpoint outputs/church-support-pattern-integer-20260911/generation-diagnosis/frozen.pt \
  --output outputs/church-support-pattern-sampling-reproduction \
  --samples 4096 --pattern-top-p 0.5 --fid
```

Use a fresh output directory. The example reproduces the frozen epoch-10 comparison; using a later checkpoint constitutes a new measurement.
