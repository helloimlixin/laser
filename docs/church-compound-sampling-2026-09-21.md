# Compound-token sampling comparison, September 21, 2026

The user selected the sampling comparison from the stage-two recipe audit.
The sweep started after the 90-epoch training run and its final verified
checkpoint uploads completed. It uses all eight H200s.

The comparison is complete. **Atom top-k 250 and coefficient top-p 1.0** had
the lowest confirmed FID, **10.9382361**, versus **11.7321463** for the baseline
coefficient top-p 0.85. Both used 50000 images, identical disjoint confirmation
rank seeds, and the frozen epoch-50 checkpoint. Temperatures remain 1.0.
The confirmed improvement is 0.7939102 FID; no significance claim is made.

| Atom top-k | Coefficient top-p | Screening FID4096 | Confirmation FID50000 |
|---:|---:|---:|---:|
| 250 | 0.85 | 14.1699161 | 11.7321463 |
| 250 | 1.0 | 13.5347693 | **10.9382361** |
| 1400 | 0.85 | 14.6321859 | Not selected for confirmation |
| 1400 | 1.0 | 14.3128305 | Not selected for confirmation |

The authoritative result artifact is
`helloimlixin-rutgers/laser/church-laser-ft3best-compound-sampling-20260921-evaluation:v1`,
with verified remote MD5 digests and sizes. It includes the selected sampling
configuration and all six evaluation grids. The parent best/last artifact is
`church-laser-ft3best-compound-nogeom90-h200x8-20260921-selected-checkpoints:v8`.
The exact frozen checkpoint SHA256 is
`5e246dbc7cfef4d4356bbede0bab9654438f61bd1ca04f3226cf971616ca8dec`.

Parent: `church-laser-ft3best-compound-nogeom90-h200x8-20260921`.
Evaluation: `church-laser-ft3best-compound-sampling-20260921`.
Output: `outputs/church-compound-ft3best-sampling-20260921`.

The final best-FID training checkpoint is hard-linked into the evaluation
directory and verified against the online artifact's MD5 and size. Every
sampling setting uses that exact checkpoint and the same selected stage-one
tokenizer, coefficient scales, and full 126227-image real FID reference.
The evaluator imports the parent's frozen runtime and calls its native
`evaluate_generation_metrics` function. A decoder wrapper only captures the
first 64 samples and reports progress; it does not alter generation.

| Setting | Atom top-k | Coefficient top-p |
|---|---:|---:|
| Current baseline | 250 | 0.85 |
| Unfiltered coefficients | 250 | 1.0 |
| Wider atom support | 1400 | 0.85 |
| Wider atoms and unfiltered coefficients | 1400 | 1.0 |

Both temperatures stay at 1.0, atom top-p at 1.0, and coefficient top-k at
0 (unrestricted). The batch is 512 images per GPU, matching production FID.
Native inference uses FP16 autocast for cached transformer operations, FP32
decoding and Inception, and enabled TF32, matching production evaluation.

All four settings generate 4096 images with seed 20260921 plus rank. The
lowest-FID nonbaseline setting and baseline then generate 50000 images each
with seed 20261921 plus rank. These eight confirmation seeds are disjoint from
the eight screening seeds. This compares two different samplers even when
baseline wins screening. The lower confirmation FID determines the saved
recommended sampling configuration. Single-seed differences do not establish
statistical significance. FID4096 and FID50000 have different sample-count
bias and are reported separately.

The first confirmation attempt incorrectly used base seed 20260922, which
overlapped seven screening rank streams. Its two 50000-image evaluations and
artifact `church-laser-ft3best-compound-sampling-20260921-evaluation:v0` are
superseded. The original files are retained under
`superseded-overlapping-rank-seeds`. Both confirmation settings were rerun
with base seed 20261921, and the evaluator now rejects overlapping rank seeds.

Online W&B publication includes all six metric results and 64-image grids,
preview token IDs, selected settings, evaluator source, and checkpoint
provenance. Completion requires a committed artifact with matching remote
file sizes and MD5 digests. Parent best/last checkpoints are already handled
by the parent upload pipeline. The evaluation does not change training losses,
learning rate, checkpoint selection history, or the 90-epoch budget.

## Investigation of distorted samples

The user reported distorted generated images while the sweep was queued.
Visual inspection of the step-6000 baseline grid found warped towers, broken
rooflines, inconsistent facades, and smeared detail. The selected tokenizer's
held-out reconstructions preserve overall architecture much better, though
fine detail softens. This suggests stage-two generation contributes materially;
it does not establish whether filtering, target noise, or training duration
causes the errors. The cached reconstruction preview also shows source-image
watermarks, so generated watermark-like text is consistent with dataset content.

A CPU check loaded the actual 404738048-parameter epoch-50 checkpoint with a
strict state-dict match, then compared cached incremental atom/coefficient
logits with a parallel teacher-forced pass for all 256 positions of cache image
17. Both paths received identical prefixes and used FP32. All comparisons
passed with atol/rtol 1e-4; exact observed errors are in `diagnostics.json`.
This checks inference consistency, not free-running sample quality or AMP
equivalence. The sweep grids must be visually reviewed alongside FID.

The diagnosis additionally records physical coefficient distributions from
every 31st cached image (4072 images, 260608 sites per depth) and the training
target's approximate physical standard deviation of 0.25. These measurements
do not by themselves prove target smoothing causes visual distortion.

An additional eight-image CPU decode holds the real cached atom supports fixed
and compares nearest coefficients, a draw from the current tau=0.125 target,
and a draw from tau=0.03125. The current target perturbation produced latent
MSE 0.000958 and pixel MAE 0.00978 on [0,1]; the narrower target produced
0.000244 and 0.00501. The three rows of `target-noise-comparison.png` preserve
similar building geometry. This small probe does not reproduce the severe
structural errors in free generation. It tests a single perturbation of real
codes, not accumulated autoregressive errors or the effect of retraining.

Visual review of the corrected confirmation grids still found malformed
building geometry in both samplers. The winning sampler improves the measured
distributional score; the structural problem is not resolved. Preview-code
diagnostics found no repeated atoms inside a support and no broad coefficient
clipping failure. On the small 64-image screening preview, latent RMS was
0.5322 for baseline and 0.5521 for coefficient top-p 1.0, versus 0.5602 on 994
cached training images. The unfiltered coefficients better preserve marginal
latent magnitude in that probe; this does not demonstrate architectural coherence.
