# Depth-specific sampling calibration on the frozen Church checkpoint

This trial tests whether opening only later compound-code depths preserves the covariance benefit of broader atom sampling while limiting the feature-mean penalty. It uses the existing best epoch-60 checkpoint, fixed tokenizer, learned atom-conditioned coefficient heads, and full physical compound vectors. No training, sparse recoding, coefficient refitting, dictionary changes, or decoder changes occur.

The original sampler uses atom top-k 250 at all four depths, atom temperature 1, coefficient temperature 1, and coefficient nucleus 0.85. Four candidate policies change only the atom support:

| Policy | Atom top-k by depth | Atom top-p by depth |
|---|---|---|
| Open depth 4 | 250, 250, 250, 16384 | 1, 1, 1, 1 |
| Open depths 3–4 | 250, 250, 16384, 16384 | 1, 1, 1, 1 |
| Open depths 2–4 | 250, 16384, 16384, 16384 | 1, 1, 1, 1 |
| Cap later tails | 250, 16384, 16384, 16384 | 1, .90, .90, .90 |

Atom uniqueness, coefficient sampling, cached causal spatial/depth inference, and learned embeddings remain identical. The depth sampler reproduces native sampled atom IDs and coefficient IDs exactly under uniform top-250 and full-support settings. Uniform nucleus-0.90 sampling and the imported nucleus probability transform are separately checked. All features use continuous decoded pixels and official FP32 Inception with TF32 disabled.

## Screening and selection

Each policy generates 4,096 images from the same four RNG streams and batch size as the previous audit. Two GPUs handle each policy, preserving the original sample order. These are screening estimates; full-dimensional 4,096-image scores have substantial finite-sample bias and are not comparable to the published 50,000-image FID scale.

Selection rules were recorded before candidate receipts existed. A simultaneous mean/covariance improvement was preferred. Otherwise, a practical tradeoff required lower covariance, at least 0.10 lower total distance, and no more than 0.25 higher mean error in both full 2,048-dimensional and fixed 256-PC comparisons. At most two policies could qualify, and the primary had to be frozen before fresh confirmation.

| Policy | Screening mean change | Covariance change | Total change | Decision |
|---|---:|---:|---:|---|
| Open depth 4 | +0.1404 | −0.3741 | −0.2337 | Preselected primary |
| Open depths 3–4 | +0.3899 | −0.4341 | −0.0442 | Does not qualify |
| Open depths 2–4 | +0.8813 | −0.4545 | +0.4268 | Does not qualify |
| Cap later tails | +0.5357 | −0.7548 | −0.2191 | Does not qualify |

Opening only depth 4 is the sole qualifying policy, as a mean/covariance tradeoff. No policy improves all three terms. Its fixed-256-PC total change is −0.1520; the 64-PC paired bootstrap total interval crosses zero. The screen therefore motivates confirmation and does not establish a robust full-FID improvement.

## Combined vectors and spatial histories

The physical analysis uses z = Σ c_d D[a_d], not separate atom/coeff scalar errors. As later depths open, channel covariance distance to cached real codes improves from 0.628 to 0.527, 0.384, and 0.203. Within-image covariance trace increases from 70.77 toward the real-code value 74.38; neighboring latent vectors become less excessively similar. The all-atom sampler has even closer physical moments, yet its image-feature mean error is worse.

These observations make physical moment matching a useful diagnostic, not a replacement for decoded-image evaluation. Preserving the depth-1 sampling rule does not preserve its distribution at later image positions: changed complete compound vectors become the context for subsequent spatial predictions. The sweep intervenes on sampling and history propagation; it does not change the training objective or establish that exposure bias has been fixed.

The montage uses the same predetermined first eight RNG rows for every policy. Semantic layouts differ after histories diverge. Eight examples support no reliable population-quality ranking.

## Independent confirmation

The primary is fixed as `open_d4` before a new set of eight RNG streams, seed 2026092867 + rank. The unchanged native sampler and selected policy each generate 50,000 images: 6,250 per GPU, generation batch 128, decoder/Inception batch 32. The reference, checkpoint, and physical coefficient scales are identical. Success requires both lower total official FID and lower covariance than the fresh matched control, with the mean tradeoff disclosed.

The first fresh comparison passed:

| Seed set | Sampler | Mean term | Covariance term | FID50k |
|---|---|---:|---:|---:|
| 2026092867 + rank | Native top-250 | 2.809208 | 7.042044 | 9.851253 |
| 2026092867 + rank | Open only depth 4 | 2.881849 | 6.830140 | 9.711988 |

The candidate lowers total FID by 0.139264 and covariance by 0.211905, while mean error increases 0.072640. The unchanged baseline itself scores below 10 on these fresh seeds, so crossing 10 is not attributed to the policy change. The relevant effect is the matched difference.

Because the baseline moved from 10.015 on the original seed set to 9.851 on the fresh set, a second independent matched comparison was preregistered **before the primary candidate result existed**, conditional on a primary win. It keeps the exact same candidate and control, with seeds 2026093867 + rank. Both pairs must lower total FID and covariance before the recipe is described as a repeated improvement; no additional policy tuning occurs.

The second fresh comparison also passes:

| Seed set | Sampler | Mean term | Covariance term | FID50k |
|---|---|---:|---:|---:|
| 2026093867 + rank | Native top-250 | 2.825804 | 7.080062 | 9.905866 |
| 2026093867 + rank | Open only depth 4 | 2.906815 | 6.890036 | 9.796852 |

The second matched changes are total FID -0.109015, covariance -0.190026, and mean +0.081011. Across the two independent seed sets, the arithmetic mean FID50k is 9.878559 for the control and 9.754420 for open depth 4. This is an average of two evaluations, not a pooled 100,000-image FID or a formal confidence interval.

Recommended recipe: **open_d4**, recorded in `selected-sampler.json`. The checkpoint weights and training configuration remain untouched. This is a modest repeatable sampler improvement, with the mean-error tradeoff reported above. It does not establish that the remaining learned-distribution or exposure-bias problem has been solved.


Full features, source hashes, immutable policy manifests, receipts, statistics, plots, and selection verification accompany the W&B evaluation. The original checkpoint remains unchanged and linked through its verified W&B artifact. Any selected sampler recipe is a separate artifact.

Implementation: `sampler.py::sample_compound_depth` is the tested depth-aware sampler, and `confirmation_worker.py` is a working evaluation entry point. The historical model's native `sample_compound` accepts only a scalar atom cutoff; a four-value list must be routed through the new sampler, not passed directly to that original method. Training defaults were not changed. The recommendation is tied to the verified epoch-60 checkpoint, not asserted for arbitrary future checkpoints.

W&B evaluation: https://wandb.ai/helloimlixin-rutgers/laser/runs/church-depth-sampler-calibration-20260928

Evidence: `outputs/church-depth-sampler-calibration-20260928`.

Verified online artifact: `helloimlixin-rutgers/laser/church-depth-sampler-calibration-20260928-results:v0`. All 181 file sizes and MD5 digests matched the committed W&B manifest.
