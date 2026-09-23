# Full-history compound conditioning diagnosis

The user subsequently authorized a [corrected fresh relaunch](church-compound-balanced-relaunch-2026-09-23.md).
The statements below about retaining the live original run describe the audit
decision before that authorization; the original run is now retired.

The current 448M model has a serious learned-conditioning weakness. Correct
causal dependencies and a large parameter count did not establish that the
trained coefficient decoder would use its current atom effectively.

This audit uses the actual epoch-13 / step-806 weights, the frozen production
runtime, and all 300 validation probe images. The running job's parameters,
optimizer and data were not changed. Evidence and executable probes are in
`outputs/church-compound-raw-fullar-scratch300-h200x1-20260922/diagnosis/`.

## Findings

The atom decoder's output has RMS 48.31, while the current-atom embedding has
RMS 0.02395: approximately a 2,017-fold difference. The implementation directly
adds these fields before the coefficient decoder. The local partial-latent
projection has RMS 1.16. Layer normalization inside subsequent residual blocks
does not independently balance these conditioning fields before their sum.

Shuffling only the current-atom conditioning across images barely affects
coefficient KL: depth 0 changes from 2.549785 to 2.549912, and the other three
depths change by less than 0.000003. This permutation changes the coefficient
decoder's current-atom inputs across the sequence, while keeping the atom
decoder's true paired history, classifier masks and evaluation targets fixed.
It does not alter atom NLL. This supports ineffective use of that conditioning
path, rather than merely a large activation ratio.

Shuffling completed-pair embeddings instead increases coefficient KL at every
depth and increases atom NLL. Both decoders use history, but the current atom is
effectively ignored by the coefficient predictor at this checkpoint.

The atom predictor also learns slowly. Validation NLL at depths 1–3 is
9.601, 9.656 and 9.642 nats, versus depth-only training-marginal baselines of
9.705, 9.702 and 9.671. A coefficient-conditioning change alone is not yet a
demonstrated solution to all of the model's learning problems.

## Comparison with the previous compound model

Both runs use the exact same cache SHA-256, stage-1 tokenizer, 126,227 training
images, coefficient vocabulary, physical soft-target temperature, global batch
2,048, and official FID50k protocol. They differ in architecture and random
training trajectories; this is an informative control, not a multi-seed causal
ablation.

| Completed epoch | Previous hierarchical compound FID50k | Current full-history FID50k |
|---:|---:|---:|
| 6 | 105.27 | 114.60 |
| 10 | 94.77 | 136.84 |
| 13 | 42.03 | 116.96 |

The earlier separate coefficient-history model also reached FID 42.97 at
epoch 13. These comparisons do not support dismissing the current result as
ordinary early training noise or blaming the shared token cache alone.

## Checks that passed

The trained dense and incremental decoders agree through all 256 events:
maximum FP32 logit difference is 1.05e-5 for both fields. BF16 maximum logit
differences are 0.03125 / 0.125, with maximum distribution KL 5.32e-5 / 8.10e-5
for atom / coefficient. No indexing or cache-order discrepancy was found.

Raw FP32 coefficients remain unclipped and unnormalized, and atom/coefficient
trajectories remain paired. Nearest-bin RMSE across depths is 0.0025–0.0052.
The actual stochastic coefficient targets add approximately 0.25 coefficient
RMSE; this is materially more noise than nearest rounding. On a 16-image
latent probe, nearest rounding adds MSE 2.70e-7 and one sampled soft-token draw
adds 9.43e-4. This is not a reconstruction-FID estimate. The decoded comparison
of the first eight images remains recognizably faithful. The earlier successful
compound model uses the same stochastic coefficient targets.

## Candidate correction and validation

The maintained model now supports opt-in normalization of the four coefficient
conditioning fields before summation: atom-decoder history, current atom,
previous pair and local physical prefix. These are neural hidden-feature
normalizations; they do not change raw coefficients, bin centers, clipping,
target distributions, sequence ordering or full-history attention.

Thirteen focused tests passed, including dense/cache agreement and causality
for both variants, gradient flow, preservation of prior events, and a regression
test that current-atom information survives a large history activation scale.

A matched 32-update continuation pilot used the same epoch-14 snapshot, restored
AdamW moments, sampled training rows, random draws, batch 256 and LR 1e-4. Its
128-image validation coefficient KL after training was 1.7033 for unchanged
continuation and 1.8389 for normalized fusion. Abruptly changing this learned
model's activation distribution worsened validation, so that retrofit was not
deployed. These were disposable trials and did not become production weights.

The separate fresh-initialization comparison is recorded in
`diagnosis/fresh-fusion-pilot.json`. Its limited budget and smaller batch make
it an optimization diagnostic, not proof of better FID50k.

Both fresh arms used the same shared-parameter initialization, 64 updates of
batch 256, LR 5e-4, sampled training rows and random draws, with no checkpoint
or optimizer state loaded. On the same 128 validation images:

| Fresh pilot after 64 updates | Unchanged fusion | Normalized fusion |
|---|---:|---:|
| Mean atom NLL | 9.5667 | 9.5446 |
| Mean coefficient KL | 2.0127 | 1.9400 |
| Depth-0 coefficient KL increase when current atom is shuffled | 0.0000 | 0.0611 |

This is a modest improvement in one short trial, not a demonstrated FID fix.
Later-depth current-atom conditioning is still weak. The live run remains on
its original architecture with monitoring and checkpoint uploads active. The
candidate is opt-in and has not been substituted into the production run.
The investigation completed on September 23 UTC.

## Relation to DCTransformer

The model adopted DCTransformer's stacked conditional factorization, rather
than reproducing the paper's complete architecture and training recipe.
The [official supplement](https://proceedings.mlr.press/v139/nash21a/nash21a-supp.pdf)
also specifies ReZero residual gates and 1,000 warmup steps; the running model
uses ungated residuals and inherited the previous RQ-style no-warmup recipe.
The field-scale failure above is measured directly in this implementation;
the paper does not establish that any single proposed change fixes its FID.
