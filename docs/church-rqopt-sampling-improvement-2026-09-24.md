The controlled sampling comparison completed: coefficient top-p 1.0 improved
FID50k on both generation seeds using the current Church run's preserved epoch-95
checkpoint. Mean FID fell from 11.63934513 to 11.20815820 (0.43118693, or 3.70%).
The weights, tokenizer, coefficient vocabulary, and training run were unchanged.

Evaluation run:
https://wandb.ai/helloimlixin-rutgers/laser/runs/church-rqopt-best95-sampling-20260924

The evaluator and request are saved under
`outputs/church-rqopt-best95-sampling-20260924`. `results.json` and `report.json` contain all four evaluations; `complete.json`
records the completed online result-artifact upload. The improved sampler is
saved as `selected-sampling.json`.

## Fixed comparison

- Full epoch-95 checkpoint, step 5795, original reported best FID 11.63741375.
- Checkpoint SHA256: `3505983d4c1023767372933cbcb21909a8728a8d8476cecd4a44ff1cecf27858`.
- Same verified frozen stage one and immutable training runtime.
- Full atom/coefficient autoregression and dictionary-vector conditioning.
- Atom temperature 1, top-k 250, top-p 1; coefficient temperature 1 and all 2048 bins.
- Only coefficient top-p changes: 0.85 versus 1.0.
- 50,000 generated images per evaluation, four H100 ranks, batch 128 per rank.
- Two independent rank-seed groups: 2026092401–2026092404 and 2026092411–2026092414.
- The baseline and candidate use identical seed groups and batching.
- Native FP16 transformer sampling, FP32 decoder/Inception, TF32 enabled, matching
  the parent runtime. Decoder chunks of 16 and Inception chunks of 32 bound memory.
- Same released RQ-VAE FID implementation and current real-reference statistics,
  SHA256 `809489d8316b9e6eb9dc3bc021b6d602f4b6d816cc80621c6b9c189a9253a7f6`.
- Select top-p 1.0 only if it improves both paired FID comparisons.

Generated feature means/covariances and fixed first-sample grids are retained for
every evaluation. The old 11.6374 score is context, not the paired baseline for
this experiment. Different sampling seeds and decoder/Inception batching can
change the measured value. Two seeds do not establish a general training gain or
resolution of structural distortions.

The preflight strictly loaded all 404,738,048 transformer parameters, generated
512 images across four ranks, and passed finite decoder/Inception checks. Peak
allocated memory was 12,484,121,088 bytes per GPU. The current training continues
in its original process and W&B run.

## Earlier interventions that already ran

Online history for
`church-laser-ft3best-stochastic-rawcoeff90-h200x5-20260921` confirms the following
continuation experiments. Each training pilot lasted six epochs and began from
the selected complete checkpoint. The retained best FID was 10.68846321.

| Intervention | Best pilot FID50k |
|---|---:|
| Lower coefficient target temperature, 0.125 to 0.03125 | 11.2361 |
| Cell-integrated coefficient targets | 10.9049 |
| Moderate depth-specific atom cutoffs | 10.7133 |
| Wider depth-specific atom cutoffs | 11.0095 |
| Fresh OMP support sampling every visit | 11.0714 |

None beat the retained checkpoint; the final selected policy kept the original
temperature, center-based targets, top-k 250, and finite support bank. The raw
online rows are archived in
`outputs/church-compound-rqopt-scratch-b2048-4h100-20260924/earlier-fid-comparison/sparse-fixes-results.json`.

These continuation results do not establish what fresh OMP or stronger dropout
would do when used from random initialization. The next controlled training test
should restore the complete earlier 9.9 recipe as a control and change only one
regularizer, with fixed held-out atom likelihood, matched FID evaluation, and
preserved best/last checkpoints. Increasing residual dropout from 0.1 to 0.2 is
an untested candidate motivated by the measured train/validation divergence.
No new training run or claimed training improvement is implied by this sampling
evaluation.

## Reference-statistics cross-check

The exact old reference was recovered from W&B recovery artifact
`church-dictattn-machine-migration-20260923:latest`; its SHA256 matches
`ad3b5a341831d877110afdff37d6fff4be855612053e3eb9e2e7119f5593d364`.
The old and current reference distributions have a mutual FID of 0.06244927.
For the first baseline's identical 50,000 generated features, current-reference
FID is 11.69252739 and old-reference FID is 11.89435304. The current reference
lowers this checkpoint's score by 0.20182565. Thus this measured reference effect
does not explain why the earlier checkpoint scored better. The effect can depend
on the generated distribution; it is not a universal correction constant.

## Completed results

| Generation seed | Coefficient top-p 0.85 | Coefficient top-p 1.0 | FID reduction |
|---|---:|---:|---:|
| 2026092401 | 11.69252739 | 11.19732570 | 0.49520169 |
| 2026092411 | 11.58616287 | 11.21899070 | 0.36717217 |
| Mean | 11.63934513 | 11.20815820 | 0.43118693 |

Every entry uses 50,000 images; the full comparison generated 200,000 images.
Both seeds improve, so the predeclared rule selects coefficient top-p 1.0.
The first 64-image grids for both samplers were visually reviewed. Both contain
warped facades/towers and other structural defects. The FID improvement does not
establish that these defects were resolved, and the result remains above the
earlier 9.9 benchmark.

The online evaluation artifact includes the exact request, evaluator, checkpoint
identity, selected sampler, all four sample grids, all four generated-feature
statistics, and the results. Source best/last training checkpoints remain in the
parent run. This is a sampling improvement on fixed weights, not a retrained
model or evidence that increased training-target stochasticity helps.
