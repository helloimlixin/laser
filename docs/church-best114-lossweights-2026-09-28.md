# Church historical checkpoint loss-weight study

The user requested a return to the original approximately 9.8-FID model and a controlled adjustment of loss contributions. The isolated runtime is `/tmp/laser-church-best114-lossweights-20260928`; durable results are in `outputs/church-best114-lossweights-20260928`.

The controller preserved the outgoing vector run at its completed step-20000 checkpoint and launched all four parallel continuations on September 29 UTC. All eight rank startup proofs passed. At step 7100, every arm was actively training; all first paired-target draws and initial unweighted losses matched across arms. Only the normalized weights differed. See `outputs/church-best114-lossweights-20260928/post-launch-audit.json`.

The outgoing run's final completed 50k evaluation at step 19,560 was FID 9.9607711, mean 3.5743103, covariance 6.3864608. Its full LAST at step 20,000 and best-FID checkpoint at step 9,780 (FID 9.9185895) were uploaded and remotely verified in `church-vector-lrhalf-20260928-selected-checkpoints:v14`. Pilot updates initially took approximately 1.88 seconds per global batch of 2,048 on each two-GPU arm, with all eight GPUs at 95–100% utilization.

## Recovered parent and fixed recipe

- Original run: `church-stochomp-fresh-t00625-k700ct09-b2048-20260925`.
- Full checkpoint: epoch 114, step 7068, 517 Adam states, 404,738,048 parameters. SHA256: `65d9f1993cfea4b354e00fa69c1dd3636c2123082f0a8cff570a11a26d7e719d`.
- Historical reported FID was 9.8003147. Separate evaluations of these weights were 9.6012163 and 9.5389032, with different draws/protocol layouts. The latter is the stored eight-stream reference for this study; these scores are not interchangeable exact replays.
- Original paired 16-trajectory bank recovered from W&B, including coefficient log partitions at temperature 0.0625. SHA256: `8ec80c5b10ea4fb14d8d7bac0a41827cc0c58beaf0d2e6f9ef5652a3d46e6976`. The older recovery cache had a different log-partition temperature and is not used.
- Frozen tokenizer, dictionary, coefficient scales and 2,048 bins. No fitting, reward, geometry loss, coefficient regression, CRPS or causal-prefix objective is added.
- Preserve Adam moments, betas (0.9, 0.95), weight decay 0.0001, clipping 1, dropout 0.2 and original BF16 outer autocast. Preserve the original cosine schedule: base LR 0.0005, horizon 18,600, initial resumed LR 0.0003420311381711698.
- Global batch remains 2,048, including correct weighting of the 1,299-image final batch. Pilot layout is two GPUs, microbatch 128, accumulation 8; the original layout was microbatch 512, accumulation 2. Statistical batching is preserved, not the exact dropout/RNG trajectory of the original layout.

## Comparison and continuation

Each arm starts from the same full parent state and runs 200 updates, to step 7268. Only the normalized atom/coefficient CE weighting changes:

| Arm | Atom weight | Coefficient weight |
|---|---:|---:|
| baseline | 1.50 | 1.00 |
| equal | 1.25 | 1.25 |
| coeff | 1.00 | 1.50 |
| atom | 1.75 | 0.75 |

The objective is `(wa * atom_CE + wc * coefficient_CE) / 2.5`. The coefficient loss uses its original conditional paired target; atom and coefficient trajectories are never independently mixed. Raw CE, coefficient target entropy and CE-minus-entropy are logged separately.

Every pilot receives an official 50,000-image FID evaluation. Two physical ranks reproduce the eight logical generation streams, including seeds, per-stream batch sequence, decoder/feature precision, sampler and feature concatenation order. All arms use the historical atom top-k 700 and coefficient temperature 0.9/top-p 0.85 sampler. Grids are generated every 200 updates. Full LAST and best-FID checkpoints are uploaded and their remote manifest entries verified.

Select the lowest-FID arm only if it improves over the post-pilot baseline by at least 0.10; otherwise select baseline. This is a practical selection threshold, not a significance test. Held-out loss is not a veto. Continue the selected full LAST on all eight GPUs to step 18,600, keeping global batch 2,048. The two-to-eight-rank RNG migration is explicit, and original RNG states remain in the checkpoint lineage.

## Validation

Strict parent tensor loading, original teacher targets/RNG, coefficient kernels and scheduler/Adam restoration passed. A disposable two-rank GPU test performed two actual training updates with all 517 finite parameter gradients and exactly synchronized rank weights; peak allocation was 36.18 GiB per rank. CPU accumulation/cursor checks cover both two- and eight-rank layouts and the partial final batch. Controller tests cover selection, checkpoint schema, archival and recovery before the first pilot update.

The strict native-sampler bitwise check failed only on fixed-context FP32 coefficient logits, with maximum error 4.7684e-6 from the restored short-attention kernel. Actual native atoms, coefficients, physical vectors, pixels, Inception features and final RNG states were exact for four generated images across two seeds; cached hidden states and atom logits were also exact. The original failed receipt is retained. A separate numerical review accepts the small FP32 discrepancy; no 50,000-image bitwise equivalence is claimed and no production kernel was changed to satisfy the check.

## Is the bank too restrictive?

The audit covers 256 fixed random training images, or 16,384 latent locations. There are 16 bank entries per location, and locations draw independently; this is not a set of only 16 whole-image encodings. Generation is not constrained to a location's bank, although positive teacher atom labels come from compatible saved entries.

The 16 entries contain only 4.850 distinct complete paired trajectories on average. At 93.04% of locations they all use the same first atom. Coefficient-conditioned effective atom counts at the four depths average 1.044, 1.157, 1.441 and 2.224. Diversity increases with depth; the main finding is not progressive collapse to a single later-depth trajectory.

Across four nested subset draws, using 8 entries instead of 16 changes the final-depth atom target by mean total variation 0.120 on valid prefixes and omits 3.10% of sampled reference prefixes. However, pooled covariance of the stored continuous combined latent vectors changes only approximately 0.08%. This excludes coefficient-bin noise and is not Inception covariance. The saved bank cannot reveal unseen teacher mass or predict improvement from 16 to a larger bank.

These results establish a coverage limitation, not the cause of the FID floor. The loss-weight study keeps bank size fixed to avoid confounding its comparison. [The bank audit is published on W&B](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-best114-bank-audit-20260928), with three tables and eight artifact files verified by remote MD5 and size. Durable copies are in `outputs/church-best114-lossweights-20260928/diagnostics`.
