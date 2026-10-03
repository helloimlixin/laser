# Church fresh LASER: feature rebound and sampler audit

Completed 2026-09-26. All values below come from frozen checkpoints and saved feature arrays. The live training objectives, optimizer schedules, and production samplers were preserved.

[W&B evaluation run](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-fresh-feature-rebound-sampler-20260926). Artifact: `helloimlixin-rutgers/laser/church-fresh-feature-rebound-sampler-20260926:v0`; all 186 remote file digests and sizes were verified.

## Matched FID50k comparison

| System | Seed pair 1 | Seed pair 2 | Mean |
|---|---:|---:|---:|
| LASER epoch 114, default sampler | 9.788941 | 9.834291 | 9.811616 |
| LASER epoch 172, default sampler | 11.204728 | 11.212717 | 11.208722 |
| Original RQTransformer epoch 80 | 10.091115 | 10.084616 | 10.087866 |

Each evaluation uses 50,000 generated images, the same 126,227-image reference, official RQ-VAE FP32 Inception features with TF32 disabled, two sequential RNG streams, generation batches of 4,096, and decoder/Inception chunks of 64. Seed bases are 2026092601 and 2026092701; each second stream uses base + 1. These pairs do not overlap with the screening seeds.

The locally trained original baseline is `church-original-rqvae-released-tokenizer-control-20260917`, selected epoch 80 / step 4960 from artifact version 249. Its historical best was 10.073105 on a rebuilt reference. It was re-evaluated here on the LASER reference; no reference conversion or offset was assumed. This is distinct from the released pretrained RQTransformer, whose earlier matched-reference result was 8.001728. The latter was not re-evaluated in this audit.

This compares selected checkpoints and end-to-end systems, not equal training steps or isolated stage-2 architecture: LASER has 404,738,048 parameters and its own tokenizer; the original RQ prior has 370,087,936 parameters and the released Church tokenizer. Both native samplers use FP16 autocast with FP32 operations outside autocast contexts. Runtime linear-output probes verified this; LASER training uses BF16. An initial sampler-metadata label confused the training and inference dtypes and was corrected without changing computation.

## Feature-space findings

| System | Mean term | Covariance term | Generated / real variance | Per-axis variance error | Additional correlation error |
|---|---:|---:|---:|---:|---:|
| LASER epoch 114, default sampler | 3.2419 | 6.5697 | 87.71% | 0.6730 | 5.8967 |
| LASER epoch 172, default sampler | 3.9504 | 7.2583 | 82.84% | 1.0399 | 6.2184 |
| Original RQTransformer epoch 80 | 3.6621 | 6.4257 | 87.10% | 0.6914 | 5.7344 |

The epoch-114 to epoch-172 change adds 0.7085 FID from the mean term and 0.6886 from the covariance term. Generated feature variance falls rather than rises. High covariance *distance* does not mean excessive total covariance.

The covariance decomposition uses the real-reference PCA basis. The per-axis term compares variance along those axes; the remaining nonnegative gap measures additional mismatch due to correlations in that basis. It is not a semantic decomposition, and PCA axes have no established scene labels. These feature moments neither identify the full joint distribution nor isolate tokenizer error from prior error.

As a diagnostic, the optimal mean-preserving scalar rescaling of epoch-114 Inception features on seed pair 1 is 1.0333; its theoretical FID is 9.6903. This is an operation on extracted features, not an implemented or necessarily realizable image-generation fix.

## Tokenizer latent distribution

The clean reference uses 8,192 fixed random training-bank images, one uniform stochastic OMP variant per spatial site, and continuous physical coefficients. All four depth pairs at a site share the selected variant. This is a sampled tokenizer-latent reference, separate from the Inception reference.

| Distribution, seed pair 1 | Energy | Channel variance | Squared mean norm | Horizontal correlation | Vertical correlation |
|---|---:|---:|---:|---:|---:|
| Clean bank | 80.2324 | 78.5338 | 1.6988 | 0.1924 | 0.1866 |
| LASER epoch 114, default sampler | 81.1762 | 78.1565 | 3.0198 | 0.2783 | 0.2390 |
| LASER epoch 172, default sampler | 78.0921 | 75.1620 | 2.9301 | 0.2771 | 0.2417 |

Neighbor correlations are centered, normalized aggregate channel inner products at horizontally or vertically adjacent latent sites. Full channel covariance, cross-depth energies, atom usage, and PCA results are saved in the artifact. Global latent moments and image-space FID do not rank these checkpoints identically; covariance/energy matching alone is therefore insufficient. At epoch 114, first-component energy is 57.123 versus clean 54.669, while depths 2-4 carry less energy than their clean counterparts. The sum of off-diagonal cross-depth energy terms is +0.0957 for generated latents versus -0.8742 for the clean bank. These are uncentered energy interactions, not centered correlations, and they do not establish the cause of the image-space FID gap.

## Sampling ablations

| Atom temperature | Coefficient temperature | Screening FID10k |
|---:|---:|---:|
| 0.95 | 0.90 | 10.923601 |
| 1.05 | 0.90 | 10.752150 |
| 1.10 | 0.90 | 10.623387 |
| 1.00 | 0.60 | 10.926806 |
| 1.00 | 0.70 | 10.856628 |
| 1.00 | 0.80 | 10.836966 |
| 1.00 | 0.90 | 10.566700 |
| 1.00 | 1.00 | 10.674853 |

All screens use seed base 2026092801, atom top-k 700 / top-p 1, and coefficient top-p 0.85. Coefficient temperatures were tested first. The atom-temperature ablation was added after the first feature audit showed underdispersion, while holding coefficient temperature at 0.9. Each family selects its minimum FID10k and confirms a nondefault winner on both independent FID50k seed pairs. FID10k must not be compared directly with FID50k.

Coefficient screen selected **0.9**. A nondefault improvement was confirmed on both seeds: **False**. Paired FID differences versus default: `[0.0, 0.0]`. When the default wins the screen, no extra candidate confirmation is run.

Atom screen selected **1.0**. A nondefault improvement was confirmed on both seeds: **False**. Paired FID differences versus default: `[0.0, 0.0]`.

Recommended tested case: **laser-best**. Exact sampler settings are in `outputs/church-feature-rebound-20260926/recommended-sampler.json`. Two evaluation seeds are not a statistical-significance claim. The selected coefficient and atom changes were tested separately; their combination was not tested.

## Lower training noise and next experiments

Priority update: the user selected matching the original RQ training recipe and stochastic mechanism before additional temperature, EMA, or LR ablations. The candidates below are deferred; see docs/church-rq-recipe-parity-2026-09-26.md for the subsequent audit.

Training target temperature and sampling temperature are different interventions. The existing teacher diagnostic estimated clean latent energy 80.6524, energy 83.3293 at target temperature 0.0625, and 81.3216 at 0.015625. On matched teacher decodes, FID10k was 3.8100, 3.7110, and 3.7303 for target 0.0625, target 0.015625, and clean coefficients respectively. These are reconstruction/teacher diagnostics, not generator FIDs.

A fresh 0.015625 target-temperature arm remains a reasonable bounded ablation, with new coefficient log-normalizers and consistently updated prefix-conditioned atom posteriors. The teacher diagnostic suggests less remaining benefit than the first noise reduction. It does not justify simply shrinking generated latents or changing only the coefficient loss.

For the FID rebounds, a separately tracked EMA or reduced-learning-rate continuation from a good checkpoint is another candidate. Neither was tested here. On the fixed 300-image training / 300-image validation probe, atom NLL changed from 3.9951 / 10.8904 at epoch 114 to 3.4037 / 11.4516 at epoch 172. This probe uses deterministic OMP supports and nearest-bin coefficient histories, rather than an expectation over the stochastic teacher. The growing gap supports investigating generalization and calibration, but does not prove a single cause of the FID oscillations. Exact records and probe provenance are in heldout-evidence.json.

## Execution and provenance

The main low-target-temperature run continued on GPUs 0 and 1. The 0.25 control paused at epoch 92 / step 5704 for the finite audit on GPU 2. It then resumed from the same full checkpoint; all 517 optimizer parameter states, scheduler step, batch 2048, microbatch 512, accumulation 4, and subsequent epoch completion were verified.

Frozen LASER checkpoints, original RQ checkpoint/tokenizer identities, exact evaluator revisions, frozen model runtime sources, shared reference identity, settings, images, statistics, and all result tables are retained. Raw per-image Inception features remain locally under `/mnt/laser-church/feature-rebound-20260926/results/`; their hashes are in the uploaded local feature manifest.

![Feature comparison](../outputs/church-feature-rebound-20260926/feature-comparison.png)
