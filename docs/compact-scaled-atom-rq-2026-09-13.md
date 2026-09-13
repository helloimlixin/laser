# Smaller joint sparse-code vocabularies

The completed frozen reconstruction screen supports the 32,769-token candidate:
its matched 4,096-image reconstruction FID is **8.3923**, versus **8.5846** for
the current 131,073-token book. It trades 0.1126 dB of PSNR for four times fewer
output classes. This is a screening result, not a 50k evaluation or generation
result.

The current 131,073-entry vocabulary expands 16,384 dictionary atoms by the same
eight signed coefficient levels, plus one zero vector. Each image still has
8×8×4 = 256 tokens; the large number is the number of choices for each token.
The flat output classifier alone has 134,349,825 trainable parameters.

This experiment fits coefficient levels separately for each atom. A token still
selects both an atom and its coefficient, and greedy residual selection still
leaves all earlier contributions fixed. The encoder, decoder, dictionary
directions, spatial resolution, and residual depth stay frozen.

With two levels per atom, the shared RQ book has 32,769 vectors, one quarter of
the current vocabulary. Those levels are a negative and positive value specific
to that atom. They are not two coefficient magnitudes shared by all atoms. With
four levels per atom, it has 65,537 vectors.

## Calibration and evaluation

- Source checkpoint SHA256:
  `93f19adc8ca2cfae0cf629d48d2d6f5f7039440b1a48ad1c7ba43cf17932a388`.
- Calibration: 4,096 frozen FP32 Church training latents, indices
  64,000–68,095, from the verified stage-2 cache.
- Initialization: the corresponding two/four shared coefficient levels from the
  original frozen reconstruction study, replicated for each atom.
- Eight constrained Lloyd passes. At each pass, current RQ residuals determine
  atom/level assignments. Each level moves toward the mean scalar projection of
  its assigned residuals, with four pseudo-observations at its original level to
  stabilize rare entries. Codebooks change between passes, not during encoding.
- Image comparison: first 4,096 training images, disjoint from coefficient
  calibration; additional imagewise metrics on all 300 official validation
  images. Training-image screens are not unseen examples for the source encoder.
- Every candidate uses identical cached screen latents and original pixels, the
  same decoder and Inception model, FP32 arithmetic, and disabled TF32.
- Fresh encoder output is checked against the cached latents. Full backbone
  fingerprints before and after evaluation establish that no backbone weights
  changed.
- Two tests verify exact nearest-neighbor residual selection against an explicit
  expanded codebook and equivalence to the original quantizer when every atom's
  coefficient levels are initialized identically.

Simple frequency pruning is not a free reduction: in the previous 4,096-image
screen, the current tokenizer used 107,373 distinct tokens. The most frequent
65,536 entries (including zero) covered only 87.6% of observed positions. The
compact experiment recalibrates codeword geometry instead of assuming that
discarding half the labels would preserve existing assignments.

## Capacity

| Coefficient construction | Vocabulary | Classifier parameters | RQTransformer parameters |
| --- | ---: | ---: | ---: |
| Eight shared levels, current | 131,073 | 134,349,825 | 487,644,161 |
| Four levels per atom | 65,537 | 67,175,425 | 420,469,761 |
| Two levels per atom | 32,769 | 33,588,225 | 386,882,561 |

Parameter counts retain the original Church 24+4 layer, width-1024 architecture
and fixed codebook-vector inputs; only classifier width changes. Smaller class
counts reduce logit and target tensor sizes, but an actual training benchmark is
needed to quantify throughput. Sequence length remains 256 tokens per image.

## Artifacts

`outputs/church-compact-scaled-rq-20260913/adaptive4-8passes/compact-codebook.pt`
and `adaptive2-8passes/compact-codebook.pt` contain the calibrated books and source
checkpoint identities. Each fitting directory contains its trace, latent screen,
source hashes, and a copy of the fitting script. The `reconstruction` directory
contains image metrics, Inception features, contact sheets, and the final frozen
state audit. Implementation: `src/adaptive_scaled_atom_rq.py`.

Following this screen, the user approved the 32k stage-2 experiment. A fresh prior
now trains on both H200s with target temperature recalibrated to 0.125. The 131k
run was gracefully paused and its complete training state preserved at update
58,833. See [the stage-2 record](compact-rq-stage2-2026-09-13.md). Reconstruction
quality alone does not establish generation quality or prove that vocabulary
size caused the larger run's FID plateau.

## Reconstruction results

All FID values below use 4,096 reconstructed training images. Matched FID uses
those same original images; the published-reference column uses the existing
Church reference statistics. Neither column is a 50,000-reconstruction estimate.

| Construction | Vocabulary | Matched rFID | rFID vs published reference | PSNR (dB) |
| --- | ---: | ---: | ---: | ---: |
| Eight shared levels, current | 131,073 | 8.5846 | 7.4550 | 18.4900 |
| Four shared levels, control | 65,537 | 8.7827 | 7.5630 | 18.3136 |
| Four levels per atom | 65,537 | 8.4801 | 7.3769 | 18.4510 |
| Two levels per atom | 32,769 | 8.3923 | 7.2497 | 18.3775 |

On the official 300-image validation set, PSNR was 18.2629, 18.0999, 18.2296,
and 18.1623 dB in the same row order. The compact two-level construction thus
retains imagewise reconstruction substantially better than the original two
shared levels, which had 17.3144 dB validation PSNR in the earlier study.

Lower reconstruction FID here does not imply uniformly better reconstructions:
both pixel MSE and latent MSE increase. The two-level candidate is selected for
the strong vocabulary/quality tradeoff. Larger-sample evaluation and a separate
prior experiment are needed to establish its final generation quality.

The 32k choice reduces classifier parameters from 134.35M to 33.59M and total
prior parameters from 487.64M to 386.88M, retaining four residual tokens per
latent pixel. The calibrated coefficient magnitudes vary across atoms from
approximately 1.11 to 14.85, instead of forcing every atom to use one universal
positive/negative magnitude.
