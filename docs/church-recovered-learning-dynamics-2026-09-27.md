The recovered FFHQ Church run is numerically stable in the inspected window, but its held-out prediction deteriorates as training fit improves. Exposure bias remains relevant; it cannot explain this entire generalization gap because the gap already appears under real histories.

Subsequent user constraint: **no refitting**. For the proposed residual teacher, each sampled atom and physical coefficient stays fixed. Later depths use `r_next = r - c * D[a]`; neither intermediate nor final least-squares solves revise the prefix. The existing residual-pair prototype already has this property. A follow-up [validation](../outputs/church-recovered-learning-dynamics-20260927/no-refit-validation.json) passed with solve/refit operations disabled, exact prefix preservation when extending depths 1–3 to 4, and targets matching independently computed distributions on the actual sampled residual (maximum error 6.67e-16). This clarification does not launch a new experiment or replace the running checkpoint. The active stage-2 loop reads cached codes and performs no refitting; OMP describes how that existing cache was originally encoded.

[W&B diagnostics](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-recovered-learning-dynamics-20260927) contains tables, generated grids, plots, per-image arrays, frozen diagnostic source, and checkpoint lineage. The results artifact is `church-recovered-learning-dynamics-20260927-results:v0`; all 71 files passed remote size/MD5 verification. The frozen epoch-55 and epoch-79 checkpoints matched the production checkpoint artifact v116. See the [full report](../outputs/church-recovered-learning-dynamics-20260927/report.md) and [upload receipt](../outputs/church-recovered-learning-dynamics-20260927/wandb-upload.json).

| Frozen checkpoint | Train atom NLL | Held-out atom NLL | Train coefficient KL | Held-out coefficient KL |
| --- | ---: | ---: | ---: | ---: |
| Epoch 55, best FID | 3.8319 | 11.4886 | 0.2361 | 0.5053 |
| Epoch 79 | 3.0170 | 12.2738 | 0.2196 | 0.5328 |

The probes contain 300 training and all 300 held-out images, averaging four fixed stochastic coefficient histories at the production target temperature. The paired held-out atom NLL increase is +0.7852, with image-bootstrap 95% interval [+0.7665, +0.8035]. Coefficient KL subtracts target entropy from cross-entropy. All four depths worsen. Fresh encoding and cached/parallel inference parity checks passed; the report records TF32 arithmetic qualifications.

The cosine schedule follows its intended phase. Only 2.6% of sampled production updates clip between epochs 55 and 80, and inspected gradients and optimizer moments are finite. Increasing batch 128 to 1,024 changed updates per image and the Adam averaging windows measured in images by eightfold. Matching throughput does not preserve learning dynamics.

Four isolated branches restored the exact epoch-55 model and Adam state, then processed the same ordered 8,192 training images. LR factors apply to the checkpoint LR, approximately 0.000412362. Schedule phase advances by equal image exposure.

| Effective batch | LR factor | Updates | Train atom NLL change | Held-out atom NLL change | Held-out coefficient KL change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 1 | 8 | -0.0627 | +0.3000 | +0.0133 |
| 1,024 | 0.25 | 8 | -0.1105 | +0.2707 | +0.0037 |
| 128 | 1 | 64 | +0.6960 | -0.1027 | +0.0031 |
| 128 | 0.25 | 64 | -0.0950 | +0.2710 | +0.0090 |

Batch 128 at the checkpoint LR improves held-out atom prediction over this short interval, but loses training fit, clips on every update, and does not improve coefficient KL. These are single-seed likelihood pilots with accumulation microbatch four, not matched production DDP trajectories or FID experiments. They do not establish an optimal batch or justify a production recipe switch. Lowering LR alone did not resolve the observed behavior. Local raw-gradient noise estimates around 198–223 images are descriptive, not an optimal Adam batch prescription.

For exposure bias, the model factorization remains valid:

`p(atom, coefficient | history) = p(atom | history) p(coefficient | atom, history)`.

The recovered teacher samples noisy coefficients while retaining fixed OMP atom targets. Later atom targets do not adapt to the residual left by earlier sampled pairs. In contrast, [RQ stochastic sampling and soft labels](https://arxiv.org/html/2203.01941#S3.SS2.SSS3) use a residual-dependent distribution at every depth. A compound adaptation is:

`q(a, c | r) ∝ exp(-||r - c D[a]||² / τ)`.

Use physical coefficients (including the depth scales), marginalize over coefficients for the atom soft target, sample the atom, and use its conditional coefficient distribution as the coefficient soft target. Sample the coefficient from that same law, subtract the complete contribution from the residual, and recompute the next depth's targets. This keeps target distributions consistent with the sampled within-site history. It does not make training histories identical to model-generated spatial histories or guarantee that exposure bias disappears.

This is a representation change: full-refit OMP revises earlier coefficients, whereas a causal residual teacher freezes earlier pairs. OMP final pairs remain a valid autoregressive sequence; the difficulty is substituting a local residual conditional for their joint distribution. The existing [prototype](../src/training/residual_pair_targets.py) implements the compound Gibbs rule but remains outside production. Its prior eight-site probe had larger residual error than OMP, with different noise mechanisms and temperatures; see the [parity audit](church-rq-recipe-parity-2026-09-26.md). It needs reconstruction, physical noise, vocabulary-mask, and throughput validation before a long training comparison. Preserving OMP instead requires coherent complete stochastic trajectories; independently perturbing a prefix and retaining its old suffix does not implement the RQ residual teacher.

On 64 held-out images at epoch 55, removing the real prefix increases source-target atom NLL by 2.3416; forcing real atoms while sampling coefficients reduces that increase to 0.1714. These measure history sensitivity, not an isolated causal exposure-bias penalty: valid alternate completions can disagree with the source image. The exact training coefficient kernel contributes expected latent error equal to 9.40% of clean latent energy, whereas nearest-bin rounding contributes negligibly. These are latent reconstruction comparisons, not FID decompositions.

The next useful controlled comparison is a batch/LR study with fixed held-out metrics and matched FID evaluation, separate from a validated stochastic-teacher change. Throughput improvements should preserve the chosen statistical batch, or explicitly revalidate update count, schedule, and optimizer averaging. Teacher generation can be batched across images and spatial sites while retaining its depth dependency.

Production continued unchanged during these diagnostics. At the final check it had completed epoch 85 with FID50k 11.8713; the best remained epoch 55 at 10.9243. The last and best checkpoints were verified online in production artifact v124, and sample grids continued every 200 steps.

![Learning dynamics and matched-image pilots](../outputs/church-recovered-learning-dynamics-20260927/learning-dynamics.png)

![Sensitivity to sampled histories](../outputs/church-recovered-learning-dynamics-20260927/rollout-sensitivity.png)
