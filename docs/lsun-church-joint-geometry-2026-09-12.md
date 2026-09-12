# Compound-token audit and joint geometry correction

The user suspected token construction or training configuration after Church samples remained poor. The trained epoch-10 bounded-relative-noise checkpoint was audited directly, including ground-truth and generated histories. The audit found an inconsistent auxiliary geometry objective and a widening train–validation gap. It did not find a packing, scale, decoder or causal-cache mismatch in the checked paths.

## Evidence from the trained checkpoint

The immutable audit snapshot is `outputs/church-token-audit-20260912/relative-epoch10.pt`, SHA-256 `1a14c15f24e834ee951f92f6fdd178556406f38e3467b3139320de0527494b99`, step 4,930. The frozen Church tokenizer hash, continuous-cache dictionary and production source hashes match. This audit loaded a trained stage-2 snapshot for read-only diagnosis; that snapshot is never an initialization source for the new production run.

`scripts/audit_church_compound_path.py` records results in `outputs/church-token-audit-20260912/audit.json`:

- Packed atom/coefficient IDs round-trip correctly. Direct dictionary contributions and the auxiliary embedding path match exactly; direct image decoding and compound decoding match exactly. Nearest-bin relative latent MSE on the two checked validation images is 7.66e-7.
- Cached predictions across every 8×8×4 position, on one validation and one generated history, match full teacher forcing in strict FP32: maximum atom/coeff logit errors 4.96e-5 and 5.91e-5, with no top-1 disagreement.
- The actual mixed-precision cached path differs from the FP32 training path by mean probability KL about 2.18e-6 for atoms and 2.17e-6 for coefficients. This small discrepancy does not support a major precision mismatch as the cause of bad images.
- Poisoning current coefficient and all future tokens changes no earlier/current predictions. Changing either the support or coefficient of a completed previous pair changes the next prediction. Both components are causal conditioning inputs.

These are bounded diagnostic checks, not a proof that every possible input or sampling behavior is correct.

## Confirmed geometry-loss inconsistency

The archived FFHQ loss first averages candidate atom vectors, then multiplies by the expected coefficient conditioned on the ground-truth atom:

`old_prediction = sum_a w(a) * D[a] * E[c | history, ground_truth_atom]`

The compound prior actually predicts `p(a | history) * p(c | history,a)`. Each candidate therefore needs its own coefficient expectation:

`corrected_prediction = sum_a w(a) * D[a] * E[c | history,a]`

The signed-code counterexample is decisive: atom `+D` with coefficient `+1`, or atom `-D` with coefficient `-1`, both reconstruct `D`. Their equal-probability mixture also reconstructs `D`. The old objective predicts zero and assigns a normalized geometry loss of one; the corrected objective predicts `D` and assigns zero. A regression test exercises the complete geometry function, not only the expectation helper, and separately verifies gradients reach alternative-atom coefficient predictions.

On the first 32 official validation images at epoch 10, candidate coefficient means disagree in sign with the ground-truth-conditioned coefficient mean **50.0%** of the time. Their mean absolute difference is **3.86 physical coefficient units**. Replacing the inconsistent mean with the candidate-conditioned mean changes the predicted contribution by squared error equal to **0.869 of target contribution energy**. The old geometry loss is 1.122; recomputing the corrected candidate expectation on the same trained weights gives 1.038. This is a diagnostic of the objective, not evidence that training the correction improves FID.

On two validation images, the old weighted geometry gradient with respect to coefficient logits has norm 20.4% of the coefficient classification gradient, with cosine -0.236. It can oppose the likelihood objective. This local gradient measurement does not establish that the loss explains all sample defects.

The fix retains the existing top-four-plus-ground-truth candidate pool and its renormalized weights. That remains an approximation: those top four atoms carry only about 11% of total atom probability in this validation probe. The correction fixes conditional coefficient consistency within that candidate approximation; it does not claim an exact expectation over all 16,384 atoms. Ground-truth candidates reuse their original coefficient logits/dropout branch. Other candidates use the same coefficient head conditioned on their own atom vector.

## Learning dynamics

The relative-noise run's training-probe atom NLL improves from 8.402 at epoch 5 to 7.259 at epoch 10, while official-validation NLL changes from 8.676 to 8.697. The fixed-noise control reaches training-probe NLL 6.075 and validation NLL 9.190 at epoch 15. These show a widening generalization gap; the 300 validation images have been reused historically. Generation FID can continue improving despite this gap, so validation likelihood alone is not treated as a generation-quality stopping criterion.

The old 300-epoch cosine still uses approximately 99.7% of peak LR at epoch 10, and 99.4% at epoch 15. This does not address the user's earlier request for earlier decay. The new run uses a continuous cosine from 5e-4 to 5e-5 over its first 10 epochs, then from 5e-5 to 1e-6 through epoch 300. At epoch 5 the LR is 2.75e-4. Zero warmup is retained. This schedule is a controlled response to the observed gap, not a proven optimal LR.

## Fresh production experiment

Run `church-joint-geometry-20260912` starts with random stage-2 weights and an empty AdamW optimizer. The common parameter fingerprint is identical to the earlier scratch runs. No stage-2 weights from the audit, fixed-noise control or relative-noise control are transferred. The model is the archived FFHQ compound architecture, with a subclass exposing its existing causal head output for the auxiliary objective; parameter count remains 404,738,048 and generation methods are inherited unchanged.

Two intended training changes are bundled: corrected candidate-conditioned geometry and earlier LR decay. Their individual effects cannot be separated by this experiment alone. Everything else stays with the verified bounded-relative-noise recipe: signed 2,048-bin coefficients, `sigma(c)=min(0.1875,0.05*abs(c))`, three-sigma truncation/nearest-bin fallback, 16,384 atoms, four complete pairs per 8×8 site, full 126,227-image center-crop cache, batch 256/microbatch 32, original optimizer betas/weight decay/dropout, classification atom weight 1.5, and geometry weight 0.05 delayed two epochs/ramped over three. The existing target calibration is copied byte-for-byte; no new noise calibration or tokenizer training occurred.

The relative-noise run `church-relative-noise-20260911` continues on GPU 1 as the comparison. Its epoch-10 FID-50k is **23.97294**, versus **25.75672** for the fixed-sigma control, using the same 50,000-sample evaluation seed. These modest gains do not resolve the remaining sample defects. The fixed-sigma control is saved and paused at step **7,460**, with optimizer/stream verified, to free GPU 0. It reached epoch-15 FID-4096 **18.96595** before pausing. Its checkpoints are preserved under `outputs/church-ffhq-noise-20260911/calibrated`; the pause receipt is `outputs/church-joint-geometry-20260912/paused-fixed-noise.json`.

Evaluation remains FID-4096 at epoch one/every five and FID-50k at epoch 10/every 50, followed by independent selected-checkpoint confirmation at completion. Generation batch remains 32. Throughput optimization is a separate outstanding task, not silently included in this fix.

Production launch waits for 23 tests, actual full-batch training, six-step continuous/resume equivalence through geometry activation, generation/checkpoint selection, unchanged comparison source hashes and matched random initialization. The authoritative launch evidence is `outputs/church-joint-geometry-20260912/verification.json`, written only after those checks pass.

Implementation: `src/church_joint_geometry.py`, `scripts/train_church_joint_geometry.py`, `scripts/finalize_church_joint_geometry.py`, `scripts/launch_church_joint_geometry.py`, `tests/test_church_joint_geometry.py`. The archived source and both previous production trainers are unchanged.
