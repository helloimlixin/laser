# Church stochastic OMP continuation and latent calibration

Update: the original run was checkpointed and stopped at epoch 65, step 4,030,
to start the user-requested fresh transformer experiments. Both new runs reset
weights, optimizer, and scheduler; details are in the
[fresh-training report](church-target-temperature-fresh-2026-09-25.md).
The continuation and diagnostics below describe the preceding experiment.

The original run passed a full epoch-61 checkpoint validation on three H200s.
Its native-sampler best improved from recovered FID50k 12.4743 to
11.4868 at epoch 55. On frozen checkpoints, two independent paired FID50k tests
improved stochastic epoch 42 from 12.4804 to 11.2299 and deterministic epoch 63
from 10.6699 to 10.2655. RQTransformer remains ahead at 8.0017 on the shared
reference. A separate teacher-decoding diagnostic supports testing lower
coefficient target temperature; that change is now being tested in fresh training.

The requested run was resumed under its original W&B ID:
[church-stochomp-softatoms-normt025-b2048-4h100-20260925](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-stochomp-softatoms-normt025-b2048-4h100-20260925).

## Recovery and continuation

Recovered full epoch-44 state, optimizer update 2,728, from selected-checkpoints
artifact `v43`. The retained best state is epoch 42, update 2,604, historical
FID50k 12.47429967. All downloaded checkpoint and cache files were verified
against their artifact MD5 digests. The tokenizer, token cache, and real FID
reference also match the original SHA256 identities. The recovered source
manifest verified 271 files; missing helper files were recovered from older
provenance archives and checked against this run's own source manifest.

The current machine has three H200s. Training initially used GPUs 0–1 with microbatch 256
and four accumulation steps, retaining global batch 2,048, 62 optimizer updates
per epoch, the existing AdamW state, and the original 18,600-step cosine schedule.
The remaining GPU evaluated immutable checkpoints. Both training ranks verified
all 517 optimizer parameter states at step 2,728 and scheduler step 2,728;
resume learning rate was 0.00047392794006. After the finite evaluations completed,
training resumed on all three H200s from epoch 60, update 3,720. It uses two
accumulation steps and a maximum microbatch of 342, with exactly 2,048 images per
full optimizer update. All three workers verified 517 optimizer parameter states
and scheduler step 3,720; handoff learning rate was 0.00045225424859.

The migration retains the saved random streams for ranks 0 and 1. The four-rank
trajectory cannot be reproduced bitwise on two ranks. Model, objective, dropout,
tokenizer, training examples, coefficient target temperature, global batch, and
learning-rate schedule are retained. Training FID keeps the original sampler.

The archived trainer assumed every stochastic cache contained `bank_identity`,
but this soft-atom cache and its checkpoint both predate that field. The restored
trainer accepts this legacy format only after checking the exact authenticated
cache SHA256; mismatched identities or hashes still fail. Layout migration is
restricted to validated 2-, 3-, and 4-rank epoch-boundary layouts at batch 2,048.

Recovery metadata and patched runtime are archived in
[church-stochomp-h200-recovery-20260925](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-stochomp-h200-recovery-20260925).
Local records and evaluator source copies are in
`outputs/church-stochomp-h200-continuation-20260925/`.
Live runtime and full checkpoints are under
`/mnt/laser-church/stochastic-softatoms-20260925/`.

## Earlier evidence

The [matched released-model comparison](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-released-rq-vs-laser-features-20260925)
measured LASER epoch 63 at 10.6757 and released RQTransformer at 8.0017 using the
same real reference and 50,000 samples. Their mean mismatch terms were 3.3644 and
3.4179; covariance mismatch was 7.3113 and 4.5839. RQ's 7.6253 score uses a
different, released reference and should not be mixed into the shared-reference
comparison.

LASER's total Inception variance was 90.44% of real variance. Its diagonal
variance mismatch contributed only 0.5907 versus 6.7207 from the additional
correlation mismatch. Thus covariance mismatch is not evidence that all feature
variance is too large. An analytic centered scalar adjustment in Inception space
would optimally expand by 1.0131 and reduce FID by only 0.0157. This is a diagnostic
calculation, not a valid image-generation improvement or a modified FID score.

The [earlier epoch-63 sampler sweep](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-epoch63-sampler-sweep-20260925)
improved the mean paired FID50k from 10.6727 to 10.4547 with atom top-k 700 and
coefficient top-p 0.85. Both independent generation seeds improved.

## New experiments

The [stochastic-checkpoint energy sweep](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-stochomp-energy-sweep-20260925)
uses the frozen epoch-42 best state. Fourteen settings test physical latent scales,
later-depth attenuation, atom top-k 700, and coefficient sampling temperatures.
Energy scaling is applied to dictionary contributions before the original
decoder; autoregressive histories remain unchanged. The identity intervention
was verified bitwise against the original decoder, and official Inception output
was verified finite FP32. The first 10,000 baseline samples have latent squared
norm 87.5079 versus 80.5670 for 2,048 clean training-bank images, an 8.6% excess.

The stronger deterministic epoch-63 checkpoint completed a 14-setting
follow-up of promising sampler changes, atom-temperature controls, and latent covariance calibration. Strong
shrinkage settings rejected by the first screen are omitted. Calibration fits a 256-channel Gaussian
transport map from independent generated latents to clean training-bank latents,
then interpolates correction strengths 0.25, 0.5, 0.75, and 1 before decoding.
The map is fitted from 8,192 training images and 10,000 generated images with a
separate seed. It does not transform Inception features or reference statistics.
The map must satisfy a covariance-matching numerical check before evaluation.

Each sweep selects on 10,000 images and confirms the selected setting against its
own unchanged baseline with two new paired 50,000-image seeds. A candidate is
accepted only if both paired FIDs improve. The real reference, official evaluator,
sample count, checkpoint, and batching are fixed within each comparison. The
generated seed checks do not establish statistical significance or eliminate
reference-set tuning. Results are sampler/decoder interventions on fixed weights;
they do not establish a training-objective improvement.

Live statuses, per-setting results, grids, feature moments, and final summaries:

- `/mnt/laser-church/energy-sweep-20260925/`
- `/mnt/laser-church/deterministic-energy-sweep-20260925/`
- `/mnt/laser-church/recovery-20260925/evaluation-pipeline-status.json`

The continuation and evaluations ran independently. Original best checkpoints
remain available. No result beating RQTransformer has been established yet.

The completed 14-setting stochastic screen selected atom top-k 700 and coefficient
sampling temperature 0.9, without latent rescaling. FID10k improved from 13.4083
to 12.0595. Both independent FID50k pairs improved:

| Generation seed | Original sampler | Selected sampler | Reduction |
|---|---:|---:|---:|
| 2026092511 | 12.528006 | 11.261156 | 1.266850 |
| 2026092521 | 12.432789 | 11.198574 | 1.234215 |
| Mean | 12.480398 | 11.229865 | 1.250533 |

This is a 10.02% relative reduction, with 200,000 confirmation images in total.
The selected policy is recorded in
`outputs/church-stochomp-h200-continuation-20260925/stochastic-selected-sampling.json`.
The original training FID sampler is retained so its history remains comparable.
A scale of 0.95 helped with top-k 250, but did not
improve the top-k 700 result. Stronger uniform shrinkage and later-depth
attenuation worsened the screening FID.

The [deterministic-checkpoint follow-up](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-detomp-energy-sweep-20260925)
selected top-k 700 with coefficient sampling temperature 0.8, atom temperature 1,
coefficient top-p 0.85, and no latent rescaling or covariance transport. Both
independent FID50k pairs improved:

| Generation seed | Original sampler | Selected sampler | Reduction |
|---|---:|---:|---:|
| 2026092511 | 10.639098 | 10.314146 | 0.324953 |
| 2026092521 | 10.700668 | 10.216809 | 0.483859 |
| Mean | 10.669883 | 10.265477 | 0.404406 |

This is a 3.79% relative reduction. It remains above the released RQTransformer's
matched-reference FID50k of 8.001728. The earlier top-k-700, temperature-1 sampler
had mean FID50k 10.454730 with different seeds, so that cross-sweep comparison is
descriptive, not a paired test. Full latent covariance transport did not improve
the 10k screen: 11.501451 versus 11.414965 with the same top-k-700 tokens. Partial
transport helped modestly, but the selected sampler was better. The transport
numerical check passed with relative covariance error 9.56e-8.

The requested training continuation also improved its native-sampler best to
11.486811 at epoch 55, update 3,410; epoch 56 scored 11.487850. An immutable full
epoch-55 snapshot is retained at `recovery-20260925/stochastic-best55.pt` under
`/mnt/laser-church`. The selected sampler has not been evaluated on that snapshot,
so its improvement should not be added to the separately measured training gain.

After both sampler sweeps and the teacher-decoding diagnostic completed, the
supervisor resumed on all three H200s from the complete epoch-60 checkpoint.
The existing exact-batch
sampler supports unequal rank sizes: two accumulation steps, maximum microbatch
342, exactly 2,048 images per update, and a final batch of 1,299. A full-epoch
coverage check found no missing or repeated images; simulated DDP gradients
matched direct full-batch gradients within 7e-16 for both full and final batches.
Rank 2 receives an independent stream derived from the saved RNG states. The
handoff preserves the schedule and objective. As with the training trajectory,
the generated FID sample set changes with the number of ranks; the sampler,
sample count, feature extractor, and real reference remain fixed.

The first full three-rank epoch completed at epoch 61, update 3,782, with native
FID50k 11.738735. Its full checkpoint contains three rank RNG states and the
unchanged batch-2,048 optimizer/scheduler progression. The continuation then
ran to epoch 65 before being checkpointed for the fresh experiments. The
best native-sampler checkpoint remains epoch 55 at FID50k 11.486811.

## Joint distribution diagnosis and next training ablations

The active checkpoint already enables `compound_pair_autoregressive=True`,
two coefficient micro-transformer layers, and depth-specific coefficient heads.
Its factorization is `p(a_d | earlier pairs) p(c_d | earlier pairs, a_d)`.
This chain rule can represent a joint distribution; separate atom and coefficient
heads do not, by themselves, impose independence between the pair's components.

An exact finite-bin moment calculation sampled 2,048 training-bank images,
eight random spatial locations per image, and one uniform bank trajectory per
location. Conditional coefficient moments were summed analytically over the
actual 2,048 bin centers, then combined using the law of total covariance.
The dictionary and coefficient scales match the frozen tokenizer.

| Coefficient target temperature | Expected latent energy | Added conditional noise energy | Relative covariance change from clean |
|---|---:|---:|---:|
| Clean continuous coefficients | 80.652381 | 0 | 0 |
| 0.25, current training | 91.350217 | 10.705854 | 0.119721 |
| 0.0625 | 83.329307 | 2.676931 | 0.029984 |
| 0.015625 | 81.321614 | 0.669233 | 0.007496 |
| Hard nearest bin | 80.653107 | 0 | 0.000043 |

At temperature 0.25, the squared shift of the global latent mean is only
3.71e-8. Thus nearly unchanged mean and altered covariance arise in the teacher
distribution itself, before prior prediction errors. These are tokenizer latent
moments over spatial locations, not Inception moments or image-level FID.
They neither explain the entire FID gap nor establish that lower temperature will
improve generation. The source and results are `target_moments.py` and
`target-moments.json` in the output directory.

The first proposed training ablation is target temperature 0.0625, then 0.015625,
against 0.25 with the same architecture, data, optimizer schedule, and support
bank. Changing target temperature requires recomputing the bank's coefficient
log-normalizers and updating the prefix-conditioned atom posterior consistently;
changing only the coefficient loss would make the teacher distribution
inconsistent. Keep these under separate run IDs and preserve the requested
continuation as the control. Matched continuation branches test short-term
adaptation; only matched fresh runs establish the full training-recipe effect.

A second hypothesis is to construct joint coefficient targets conditional on
support using latent reconstruction error. For support dictionary D_A, coefficient
perturbation energy is `delta_c.T @ (D_A.T @ D_A) @ delta_c`. Independent
per-depth kernels omit these cross terms. A joint teacher can account for them,
but must produce consistent autoregressive conditional targets. The sampled
support Gram matrices have median condition number 1.925 and 95th percentile
2.748; severe dictionary ill-conditioning is not demonstrated here.

A concrete candidate teacher, conditional on a fixed full-rank support, is
`q(c | A,z) proportional to exp(-||z - D_A c||^2 / tau)`.
With continuous, unbounded coefficients and no coefficient prior, its mean is
the least-squares solution and its covariance is
`(tau/2) * inverse(D_A.T @ D_A)`. This derives a correlated coefficient target
from reconstruction geometry. Finite bins, coefficient bounds, and support
mixtures require additional treatment; this formula is a proposal, not the
distribution currently implemented. Its physical temperature is not directly
comparable to the current normalized scalar temperature. Any comparison should
match added latent distortion first, to separate correlation changes from simply
reducing the total noise.

Before enlarging the prior, compare decoded clean tokens, decoded samples from
the training teacher, and free-running prior samples using identical image
metrics. This separates tokenizer distortion, teacher noise, and prior error.
Also measure correlations across neighboring spatial sites, which the global
256-channel latent calibration cannot recover. Matching covariance alone is
insufficient to identify a full joint distribution.

RQ's soft targets use distances to residual code vectors, so copying a numeric
temperature to normalized scalar coefficients does not preserve the amount or
geometry of perturbation; see the original paper's
[soft labeling and stochastic sampling formulation](https://arxiv.org/html/2203.01941#S3.SS2.SSS3).
No new training objective has been enabled in the continued run.

The subsequent [teacher-decoding diagnostic](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-teacher-distribution-20260925)
decoded the same 10,000 training-bank images, with the same uniformly sampled
support trajectory at each spatial site. Coefficient temperatures share inverse-CDF
uniform draws and use their full categorical distributions, without top-k/top-p.
Only the coefficient noise changes; the frozen decoder and official Inception
evaluator are unchanged.

| Teacher coefficients | Diagnostic FID10k | Mean mismatch | Covariance mismatch |
|---|---:|---:|---:|
| Clean continuous | 3.730344 | 1.192043 | 2.538302 |
| Temperature 0.25 | 5.850389 | 2.015498 | 3.834892 |
| Temperature 0.0625 | 3.809968 | 1.178315 | 2.631653 |
| Temperature 0.015625 | 3.711033 | 1.174306 | 2.536727 |

Lower temperature substantially reduces distortion of the decoded teacher
distribution. This strengthens the case for a matched training ablation at
0.0625 and 0.015625. It is not a generator result, and the 2.04-point difference
between temperatures 0.25 and 0.0625 must not be subtracted from generation FID.
The small advantage of 0.015625 over clean coefficients is a single finite-sample
diagnostic, not evidence that noise universally improves reconstruction quality.
After the nonlinear decoder, the current noise changes both mean and covariance;
its nearly zero latent-mean shift should not be generalized to Inception features.

The existing fixed probe also shows a growing generalization gap: from resumed
epoch 45 to epoch 55, training atom NLL fell from 5.822 to 5.372, while validation
atom NLL rose from 9.596 to 9.788. Each split has 300 images. These probe histories
use deterministic OMP and nearest coefficient bins, rather than stochastic-bank
histories, so they are a diagnostic and not a direct estimate of the active
training objective. This is a reason to investigate generalization and validation
selection before adding model capacity. The continued run retains its original
best checkpoint and schedule.
