# Where the Church covariance distance comes from

The evidence localizes the extra discrepancy to stage-2 generation of spatially arranged compound vectors. Fixed atom truncation is one demonstrated contributor. The pilot also changes the learned distribution unfavorably; improving its conditional training objective did not improve its free-running image distribution. This audit does not establish a single cause for every historical run.

This was inference and CPU analysis of saved checkpoints. No training, sparse recoding, coefficient fitting, dictionary updates, or decoder updates were performed. Baseline is the repaired run's best epoch 60/step 7380; pilot is the failed whole-combination continuation at epoch 70/step 8610. All new diagnostics use the same frozen tokenizer. The source code, identities, seeds, feature arrays, statistics, and numerical checks are included in the W&B artifact.

## What the official covariance term measures here

FID splits into squared distance between Inception feature means and the Bures distance between feature covariances. Its covariance term depends on both directional variance and relationships among feature directions; it is not the difference between two total-variance scalars. See the [original FID paper](https://arxiv.org/abs/1706.08500).

The saved official 50,000 statistics reproduce to numerical precision:

| Checkpoint | Mean term | Covariance term | FID 50,000 |
|---|---:|---:|---:|
| Baseline 60 | 2.846458 | 7.168579 | 10.015037 |
| Pilot 70 | 3.290274 | 7.955266 | 11.245540 |
| Parent continuation 70 | 3.251959 | 7.259545 | 10.511505 |

From baseline to pilot, total generated feature variance **increases 90.503 → 92.109** toward real 102.022, but covariance distance worsens 7.169 → 7.955. The discrepancy in marginal variances along the fixed real principal axes improves 0.658 → 0.584, and the eigenvalue-only lower bound improves 0.451 → 0.387. The deterioration is not captured by these marginal-variance or eigenvalue-only bounds. Actual feature relationships also change: for example, the correlation between real principal components 1 and 4 shifts from -0.0660 to -0.1079, versus zero in the real reference. These are descriptive mathematical comparisons, not isolated causal components.

The optimal hypothetical uniform rescaling of pilot features could remove only 0.0104 of its covariance distance. Uniformly rescaling these feature vectors would remove little of the discrepancy. This does not rule out sampling changes that reshape the joint distribution. Feature rescaling is a diagnostic calculation, not a realizable generator intervention.

The increase spans all real-reference PCA rank bands: +0.124 at 1–16, +0.189 at 17–64, +0.245 at 65–256, +0.183 at 257–1024, +0.045 at 1025–2048. It is not only a numerical tail effect. Exact additive contributions use the diagonal of a positive-semidefinite Gaussian transport residual covariance; formulas and checks are in `matrices/summary.json`.

## Frozen tokenizer versus generated codes

The audit reconstructed 10,000 fixed randomly selected training images three ways: original cached continuous coefficients with their original physical scales; the current clipped/FP16 continuous cache; and current 2,048-bin coefficients. Atom supports and decoder stay fixed. This never invokes encoding or refitting.

Against the same official reference, covariance distance is 2.500939 for the original continuous reconstructions and 2.500950 after clipping/binning. The direct distribution distance introduced by clipping has covariance 0.008917; subsequent binning adds only 0.000026. In fixed 64-dimensional real PCA space, the paired bootstrap interval for the original → binned covariance change is [-0.000803,+0.001559]. Thus coefficient clipping/binning does not explain the large covariance drift observed here.

The frozen stage-1 reconstruction path introduces a distribution change, but reconstruction remains much closer to real than generation in a comparison using equal 4,096-image counts and the same fixed 64 real principal components:

| Condition | Projected covariance distance |
|---|---:|
| Real-image subset | 0.1858 |
| Original continuous reconstruction | 0.7344 |
| Current binned reconstruction | 0.7346 |
| Baseline free generation | 3.7107 |
| Pilot free generation | 3.9721 |

These are diagnostic projected distances, not FID 50,000. Equal sample counts do not remove model-dependent finite-sample bias ([Chong and Forsyth, 2020](https://arxiv.org/abs/1911.07023)). Reconstruction FID is not a guaranteed generation floor, and none of these distances may be subtracted to claim a percentage caused by the tokenizer. Full 10,000 reconstruction scores and the real-subset sampling baseline are in `reconstruction/summary.json`.

## The physical compound vectors lose spatial variation

Measurements use the actual combined vector z=Σ_d c_dD[a_d], including all cross-depth terms. They do not treat atom IDs and coefficients as independent scalar errors. Population covariance across latent sites is decomposed exactly into average covariance within each image plus covariance between image means.

| Physical latent statistic | Cached real codes 10,000 | Baseline free 4,096 | Pilot free 4,096 |
|---|---:|---:|---:|
| Within-image spatial covariance trace | 74.382 | 70.772 | 67.054 |
| Between-image mean covariance trace | 4.103 | 5.512 | 5.744 |
| Horizontal centered neighbor similarity | 0.153 | 0.249 | 0.274 |
| Vertical centered neighbor similarity | 0.141 | 0.213 | 0.209 |

The physical shift is less variation within an image and more image-wide shared variation. Horizontal neighbor similarity rises further in the pilot; vertical similarity remains excessive but does not worsen. These are normalized centered latent dot products, not labels for visual semantics or Inception covariance terms.

The energy drop is not dominated by atoms cancelling each other: the sum of cross-depth terms changes +0.296 → +0.087 while the sum of depth self-energies changes 80.110 → 76.941. This rules out a large increase in **average net cancellation** as the explanation for lost latent energy; it does not rule out all directional or per-example cross-depth dependence problems.

With nearest-bin real validation spatial prefixes, the pilot's within-image trace is 74.303 versus matched real 74.308; horizontal similarity is 0.169 versus matched real 0.167. The earlier checkpoint is 76.050 and 0.165. These conditional physical moments improve while free generation moves away from real-code moments. This does not mean every physical covariance direction improves: the full 256-channel covariance distance under real histories is essentially flat/slightly worse 0.1301 → 0.1370, whereas free generation worsens 0.6278 → 0.7815. The forced nearest-bin histories differ from both runs' stochastic training histories; this is a context intervention, not an on-distribution evaluation of their training losses.

An independent 256-image feature check also finds improved pilot covariance under real spatial prefixes across 16/32/64 principal components. However, these images assemble independent site-level conditional draws under true histories. They are not ancestral model samples and their absolute feature distances are worse than free generation. This supports investigating sensitivity to generated history; it does **not** establish an unconditional FID rescue or prove exposure bias is the sole cause. Details, bootstrap intervals, and sample-block sensitivity are in `teacher-context-feature/report.md`.

## A direct atom-cutoff intervention

The same top 250 atom cutoff becomes more restrictive as the model distribution spreads out. In production free histories it retains probability mass by depth:

| Checkpoint | Depth 1 | Depth 2 | Depth 3 | Depth 4 |
|---|---:|---:|---:|---:|
| Baseline 60 | 97.9% | 71.0% | 52.1% | 34.9% |
| Pilot 70 | 97.0% | 65.0% | 41.0% | 22.4% |

This is measured mass under each checkpoint's atom head after excluding forbidden repeats. It is not the probability that the sampled atom is incorrect. After renormalization, the coefficient nucleus changes the conditional physical second moment by only about 1–3%. That statistic does not rule out higher-order effects of coefficient sampling; atom truncation is tested directly below.

At fixed weights, removing only the atom top-k cutoff—keeping atom/coeff temperatures 1, coefficient top-p 0.85 and uniqueness—reduces the pilot's horizontal latent similarity 0.274 → 0.190 and raises within-image covariance trace 67.054 → 69.071. The pilot's 64-PC Inception covariance distance drops 3.972 → 3.341, paired bootstrap 95% interval for the change [-0.855,-0.416]. Its mean error grows 2.839 → 3.589. This demonstrates a covariance/mean tradeoff, not a solved generator.

The first 256 real principal components show the same covariance reduction 6.742 → 5.700. Baseline also benefits in covariance, so the cutoff is a contributor to the existing mismatch and not uniquely a pilot defect. Full atoms greatly improve the physical combined-vector channel covariance too, but physical moment matching does not guarantee matching nonlinear decoded image features: the projected Inception mean error increases.

The full 50,000-image confirmation is complete, using the native sampler and official full 2,048-dimensional FID calculation. Only the atom cutoff changes; seed/rank partitions, coefficient nucleus, checkpoints and decoder remain fixed:

| Checkpoint and atom sampler | Mean term | Covariance term | FID 50,000 |
|---|---:|---:|---:|
| Baseline 60, top 250 | 2.846458 | 7.168579 | **10.015037** |
| Baseline 60, all atoms | 3.672293 | 6.478086 | 10.150379 |
| Pilot 70, top 250 | 3.290274 | 7.955266 | 11.245540 |
| Pilot 70, all atoms | 4.124364 | 6.808119 | 10.932483 |

Removing the cutoff reduces pilot covariance by **1.147147**, but increases mean error by **0.834090**. Total FID improves only 0.313057 and remains worse than the original baseline. Baseline covariance also improves 0.690493, but its total FID worsens 0.135342. No tested policy beats the original baseline in total FID.

The observed baseline-to-pilot covariance gap shrinks from 0.786687 with top 250 to 0.330033 with all atoms. Thus the fixed atom cutoff amplifies the measured checkpoint regression by 0.456654 in this matched evaluation. This is a specific sampler/checkpoint interaction, not a unique decomposition of all training causes. These are single matched seed partitions, not estimates averaged across repeated training or evaluation seeds.

Feature variance actually falls when removing the cutoff: pilot 92.109 → 86.381 while covariance distance improves 7.955 → 6.808. This independently shows why total feature variance is an inadequate target for this failure. Official statistics and verified source/checkpoint/feature hashes are in `sampling50k/results.json` and each checkpoint directory.

## What is established, and what remains open

Established: substantial additional feature-covariance error appears when stage 2 generates codes; the adapted coefficient grid contributes very little; free generation has excess adjacent-site similarity in the combined latent vectors; atom truncation directly contributes to that similarity and feature-covariance mismatch; a better conditional objective value is insufficient evidence of a better free-running distribution.

Not established: which individual training change caused the pilot regression, how much unconditional FID is caused specifically by exposure bias, or that a particular semantic class explains a principal axis. The pilot changed the combination teacher, atom loss weight, and training history handling together. The audit does not isolate these training interventions. PCA example montages illustrate architecture, viewpoint, context and photographic variation but are not semantic causal labels.

Protocol note: all Inception extraction is FP32 with TF32 disabled and continuous decoded pixels. Generated and forced-context images use decoder TF32 enabled, matching production; the reconstruction bridge uses decoder TF32 disabled. The within-checkpoint sampler comparison keeps decoder precision identical, and physical code moments are unaffected. On the same 128 generated codes, toggling decoder TF32 produces only 0.000041 of paired 64-PC covariance distance; the same check on 128 real codes gives 0.000028. This numerical effect is tiny on the checked inputs, not a universal population bound. The precision check is included in `decoder-precision`.

Keep the epoch 60 best checkpoint as the reference. Before another training launch, use a bounded fixed-checkpoint sampler comparison that evaluates **both** mean and covariance; a covariance-only win can worsen total FID. Any later training change should also pass a free-rollout combined-vector spatial-moment check, rather than being selected solely for teacher-conditioned loss or scalar coefficient moments. No training was launched by this audit.

W&B evaluation: https://wandb.ai/helloimlixin-rutgers/laser/runs/church-covariance-origin-audit-20260928

Local evidence: `outputs/church-covariance-origin-audit-20260928`.

Verified online artifact: `helloimlixin-rutgers/laser/church-covariance-origin-audit-20260928-results:v0`. All 283 file sizes and MD5 digests match the committed W&B manifest.
