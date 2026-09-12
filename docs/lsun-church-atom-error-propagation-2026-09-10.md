LSUN Church: atom mistakes and coefficient-history drift, September 10, 2026

**A wrong atom disrupts the next few coefficient predictions, but a larger failure appears even with every atom correct: coefficient signs drift when the model consumes its own predictions.** In the sampled fixed-support diagnostic, downstream sign accuracy falls from 86.52% with real coefficient history to 68.99% with generated coefficient history. Correcting signs during rollout substantially restores the reconstructed image. This identifies a useful training target; it does not establish a fix for unconditional generation.

This investigation used the original epoch-50 categorical pair model, with no training, on all 300 official Church validation images. The tokenizer is the verified one-epoch Church fine-tune from the ImageNet rFID 4.21 checkpoint. The BAR-inspired checkpoint was not used in this investigation. The [prior BAR pilot](lsun-church-bar-coefficients-2026-09-10.md) motivated testing the difference between real and generated histories.

![Controlled atom and coefficient-history diagnostics](../outputs/lsun-church-atom-errors-20260910/main/diagnostic.png)

The main fixed-support comparison is below. At two predetermined starting sites, a real prefix is supplied, all subsequent atom identities remain real, and coefficient predictions either receive real preceding coefficients or their own preceding predictions. Measurements exclude the initial intervention coefficient and average the two starting sites equally within each image.

| Coefficient history | Greedy sign accuracy | Sampled sign accuracy | Greedy coefficient MAE | Sampled coefficient MAE |
| --- | ---: | ---: | ---: | ---: |
| Real preceding coefficients | 87.64% | 86.52% | 1.3233 | 1.4550 |
| Model-generated preceding coefficients | 70.74% | 68.99% | 2.5369 | 2.6639 |

For sampled coefficients, the sign-accuracy difference is -17.53 percentage points, with a paired image bootstrap 95% interval of -18.19 to -16.89. Physical coefficient MAE increases by 1.2089, interval 1.1581 to 1.2592. Greedy selection also degrades strongly, so this effect is present without sampling randomness.

Correcting each generated coefficient's sign before feeding it into later predictions produces a much larger improvement than correcting its magnitude:

| Rollout condition; all atoms real | Greedy PSNR | Sampled PSNR |
| --- | ---: | ---: |
| Generate sign and magnitude | 16.52 dB | 16.28 dB |
| Supply the true sign; generate magnitude | 24.26 dB | 23.82 dB |
| Generate sign; supply the true magnitude | 18.13 dB | 17.75 dB |

PSNR is measured against each image's true-code reconstruction, not the original image. The unchanged real prefix contributes to this full-image score; comparisons use exactly the same prefixes. The sampled sign correction gains 7.54 dB (95% interval 7.33 to 7.76), while magnitude correction gains 1.47 dB (1.25 to 1.70). These corrections require ground truth and cannot be applied during unconditional generation. Because each correction also changes the history for subsequent predictions, their benefits include downstream effects; they are not an additive decomposition of sign and magnitude error.

The [sampled oracle grid](../outputs/lsun-church-atom-errors-20260910/main/oracle-sample-site8.png) shows the first four validation examples without selection. Columns are true-code reconstruction, true-sign rollout, true-magnitude rollout, and full truth replay. Correcting signs visibly preserves much more of the reference structure in these examples. Full truth replay recovered every atom and coefficient ID exactly, with zero latent and decoded-image error.

The single-atom test answers the original hypothesis more directly. Each intervention changes one atom at spatial sites 8, 24, 40, or 56, at depth 0 or 2. Two alternatives are tested: the closest different dictionary vector by signed cosine similarity, and the highest-logit alternative under the original model. All four original atoms at that site are excluded from replacement candidates to prevent duplicate support. This exclusion uses diagnostic knowledge of the real support and is not a generation algorithm.

Changing the atom changes the coefficient it should receive. For the adjusted-coefficient intervention, we choose the coefficient minimizing the change to the original atom's contribution while holding the other three contributions fixed:

```text
c_adjusted = c_original * dot(old_atom, new_atom) / dot(new_atom, new_atom)
```

That scalar is mapped back to the existing quantization grid with the original depth scale. It is a best single-coefficient adjustment, not a joint refit of all four coefficients. A second intervention lets the original model predict the changed atom's coefficient. We never score the changed atom against the old atom's coefficient target in downstream comparisons. Later coefficients retain their original targets, so this measures sensitivity around the reference representation; a different support could also support a different valid image.

With real tokens restored after the changed pair, the nearest replacement plus adjusted coefficient has the following effects:

| Distance after the changed pair | Change in sign accuracy | Change in physical coefficient MAE |
| --- | ---: | ---: |
| Next coefficient | -13.17 percentage points | +0.6237 |
| Next 2–3 coefficients | -5.40 percentage points | +0.1949 |
| Offsets 4–15 | -0.43 percentage points | +0.0320 |
| Offsets 16–63 | -0.14 percentage points | +0.0102 |
| Offsets 64 onward | +0.007 percentage points | +0.0005 |

The next-coefficient sign effect has a 95% interval of -14.92 to -11.46 percentage points. Thus, there is clear local sensitivity, but the isolated disturbance does not keep growing when later history is restored. The nearest alternatives have average cosine similarity 0.7607; even these are appreciably different vectors, so this is not an infinitesimal perturbation test. Model-preferred alternatives have mean signed cosine 0.0160 and are substantially different on this measure. Signed cosine alone does not quantify equivalence after coefficient-sign changes.

The offset averages hide a second spatially local effect: sensitivity increases again at offset 32, the same depth directly below the changed site in the next 8-wide spatial row. For the nearest replacement with its model-predicted coefficient, this adds 0.3559 MAE (95% interval 0.2223 to 0.4898) and reduces sign accuracy by 2.44 percentage points (-3.67 to -1.28). For the model-preferred alternative, the corresponding changes are +0.5296 MAE and -3.33 percentage points. This is structured propagation to a spatial neighbor, not monotonic decay at every raster step. Bottom-row interventions have no such neighbor and are excluded from this particular measurement.

When coefficients continue rolling forward after the replacement, a nearest wrong atom with a predicted coefficient changes sampled sign accuracy from 68.99% to 68.42%, an additional -0.56 percentage points (95% interval -1.09 to -0.005). Greedy selection shows no clear additional effect: +0.14 percentage points, interval -0.41 to +0.69. A model-preferred wrong atom has a larger additional effect, reaching 67.74% sampled sign accuracy, or -1.24 percentage points (interval -1.80 to -0.69). These effects are much smaller than the loss from generating coefficient history with no atom intervention.

The adjusted-coefficient rollout uses a separate matched control that clamps the intervention coefficient to its true value. This avoids crediting an intervention for receiving oracle information that its control lacked. Relative to that control, the nearest replacement reduces sampled downstream sign accuracy by 1.04 percentage points (interval -1.57 to -0.48). All detailed conditions and intervals are retained in [summary.json](../outputs/lsun-church-atom-errors-20260910/main/summary.json).

The [sampled rollout grid](../outputs/lsun-church-atom-errors-20260910/main/rollout-sample-site8.png) uses columns: true-code reconstruction, teacher-forced coefficients, ordinary coefficient rollout, rollout with the initial coefficient clamped, nearest atom with adjusted coefficient, nearest atom with predicted coefficient, model alternative with adjusted coefficient, model alternative with predicted coefficient. Rows are the same first four validation images. The [greedy counterpart](../outputs/lsun-church-atom-errors-20260910/main/rollout-greedy-site8.png) and both grids starting at site 40 are also saved.

The practical next experiment should target robustness to generated coefficient history, with sign errors as the primary diagnostic. A bounded hypothesis is a matched continuation of the existing categorical model using short generated coefficient prefixes, compared with ordinary teacher-forced continuation. Keep the tokenizer and integer bins fixed, monitor sign accuracy under rollout, and judge any candidate by matched unconditional FID. This requires a measured training experiment: the current probe does not show that replay training will work, nor that support generation is solved. No such training was launched here.

All results come from one fixed model and the 300-image official validation set, which has previously been used for checkpoint selection. The rollout starts are zero-indexed sites 8 and 40 in the 8×8 raster, at depth 0; their remaining coefficient horizons differ. Each image contributes the average of its two case metrics. Local effects average the eight intervention positions per image, using only positions with an available requested horizon. Uncertainty uses 5,000 bootstrap resamples of images with seed 3701, keeping each image's interventions together. It does not measure checkpoint, sampling-seed, or training variability. The many reported intervals are descriptive and are not adjusted for multiple comparisons.

Sampled rollouts use untruncated categorical distributions at temperature 1. They share one uniform random value per image and coefficient position across conditions, applied through inverse-CDF sampling on ordered coefficient bins. This reduces irrelevant randomness in paired comparisons while preserving each categorical marginal. Greedy rollouts use argmax. AR computations use BF16; decoding uses FP32 and clamps the resulting images to [0,1]. There is no atom sampling after the single injected alternative. Fixed-support rollouts are conditional stress tests, not ordinary unconditional samples, and no FID claim is made from their reconstruction error.

Checkpoint provenance:

```text
Source AR run: helloimlixin-rutgers/laser/churchftdc-20260909024359
Source AR: epoch 50, global step 3050
Source SHA256: cbb7d0a8c22c31e042ed04004201b360b9bacd1238f6fa677d28ba3e0b21ce4a
Tokenizer SHA256: 93f19adc8ca2cfae0cf629d48d2d6f5f7039440b1a48ad1c7ba43cf17932a388
```

The full-validation baseline exactly reproduces the preceding probe's 88.299477% greedy sign accuracy. The suffix-specific numbers above differ because they measure only positions after the prescribed prefixes. Every local intervention passed assertions that earlier predictions are unchanged and that changing the current coefficient cannot change its own predictor. Every rollout preserved the real prefix and all downstream atoms. Full oracle replay passed exact token equality checks.

The implementation is in [probe_church_atom_error_propagation.py](../scripts/probe_church_atom_error_propagation.py), with aggregation and figures in [summarize_church_atom_errors.py](../scripts/summarize_church_atom_errors.py). The [diagnostic tests](../tests/test_atom_error_probe.py) check least-squares replacement targets, sign/rescaling equivalence, exclusion of duplicate supports, coupled categorical sampling, and preservation of the unmodified physical component in each oracle. Together with the existing masked-head and pair-causality tests, **16 tests passed**. One four-image smoke test preceded the full runs.

To reproduce using the existing checkpoint/cache assets, choose a fresh output directory and run the local phase first:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 \
/tmp/laser-sign-venv/bin/python scripts/probe_church_atom_error_propagation.py \
  --phase local --output outputs/church-atom-errors-reproduction --batch-size 64

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 \
/tmp/laser-sign-venv/bin/python scripts/probe_church_atom_error_propagation.py \
  --phase rollout --sampling greedy \
  --output outputs/church-atom-errors-reproduction --batch-size 100

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 \
/tmp/laser-sign-venv/bin/python scripts/probe_church_atom_error_propagation.py \
  --phase rollout --sampling sample \
  --output outputs/church-atom-errors-reproduction --batch-size 100
```

Repeat the two rollout commands with `--phase oracle`, then summarize:

```bash
/tmp/laser-sign-venv/bin/python scripts/summarize_church_atom_errors.py \
  outputs/church-atom-errors-reproduction
```

All per-image measurements, coefficient trajectories, configurations, and sample grids are saved under `outputs/lsun-church-atom-errors-20260910/main/`. Run-level console logs are in its parent directory.
