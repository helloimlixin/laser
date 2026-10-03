The user authorized a fix and relaunch on 2026-09-28, following the requirement that both stochastic labels and samples account for the whole sparse combined vector, with no refitting. This attempt changes the teacher and its corresponding likelihood objective. It does not establish the cause of the FID plateau or promise a lower FID.

The new run is `church-combination-norefit-b1024-8h100-20260928`. Its immutable launch sources, plans, validation results, and predecessor receipt are under `outputs/church-combination-norefit-b1024-8h100-20260928`. Local execution uses `/tmp/laser-church-combination-20260928` and the recovered Torch 2.4.1 environment. Production launch follows an eight-rank optimizer, sampling, FID smoke, and checkpoint validation.

For each spatial site, the teacher uses the actual frozen encoder output z. At each of four depths it proposes six token pairs: the cached atom or its nearest positive-cosine dictionary neighbor, each paired with the nearest cached coefficient bin or a symmetric random bin displacement. The displacement is drawn uniformly from 1 through 256. Coefficients use their actual per-depth physical scales. The same cached coefficient center is used for either atom; no least-squares solve, coefficient optimization, re-encoding by OMP, or revision of an emitted prefix occurs.

The Cartesian product contains up to 6^4 = 1,296 complete codes. Exact duplicate local pairs are removed, and complete codes with repeated atoms are excluded. Conditional on the proposed pool G, valid distinct pairs have uniform product base mass and

    q_G(S | z) ∝ exp(-||z - sum_d c_d D[a_d]||² / 2).

The complete-vector score includes cross terms. A complete code is sampled from this distribution. Atom soft labels marginalize all remaining valid completions given the actual preceding token pairs; coefficient soft labels additionally condition on the sampled current atom. Both labels and samples therefore use one consistent joint law. The two existing output heads implement its chain rule. Their cross-entropies have equal weight, with a common factor 1/(2K). Previously used atoms are masked during training, matching the generator.

This is exact within each proposed finite grid, not full-vocabulary Gibbs sampling. Randomized pools define a mixture of these normalized distributions. The sampled-prefix soft-target loss is an unbiased cross-entropy objective for that mixture. Reported CE minus conditional pool entropy is component-teacher KL, not KL to the marginalized mixture. Fixed probe seeds support comparisons between checkpoints within this recipe; absolute KL values should not be compared directly with earlier teachers.

Coverage is deliberately limited. On the selected held-out probe, the probability of changing the cached atom is approximately 1.34%, 6.37%, 18.65%, and 32.62% across depths. This is local augmentation. It trains on varied valid teacher prefixes; it does not train on model-generated image histories or eliminate all exposure bias. The spatial architecture remains the recovered FFHQ architecture.

Validation before production:

- 25 CPU tests passed: brute-force probabilities, full-vector cross terms, actual-prefix conditionals, duplicate aliases, distinct support, sampling frequencies, sparse-loss gradients, RNG replay, and forbidden solver calls.
- The existing 126,227-image FP32 encoder cache was authenticated by SHA256, converted locally, and compared with 300 independently encoded training probes. Maximum difference was 1.04e-5. The 300 validation signals were independently encoded. No training-image OMP operation ran.
- Temperature and proposal radius were selected from a declared 24-setting grid by maximizing alternative-atom probability subject to expected latent distortion no greater than the previous coefficient-noise teacher on both 64-image train and validation probes. Selected temperature 2, radius 256, nearest neighbor only. Expected validation distortion was 4.864 versus the previous teacher's sampled estimate 5.091.
- An independent stochastic reconstruction draw on 64 validation images gave mean latent squared error 4.858 versus 5.231 for the previous teacher; 95th-percentile site error 9.589 versus 10.946. Decoded-image PSNR was 18.390 versus 18.236 dB. These are reconstruction checks, not generative FID improvements.
- H100 teacher benchmark: 8,192 sites, chunks of 2,048, approximately 15 ms and 370 MB peak allocated memory. Physical scoring uses FP32 with TF32 disabled, restoring the training setting afterward.

The user then clarified that reducing FID covariance drift is the main goal. To test this directly, the pilot continues from the predecessor's best epoch-60 checkpoint, step 7,380, with its model, optimizer, scheduler, and all-rank RNG state. The frozen tokenizer, eight H100s, global batch 1,024, dropout 0.2, and original 24,600-update cosine horizon remain. The inherited LR is approximately 0.000396946. It stops at epoch 70, step 8,610, after 10 additional epochs (1,230 updates). Changing the pilot cap does not compress or restart the schedule. Samples are logged every 200 global updates; official-reference 50,000-sample FID runs before the first update and at epochs 65 and 70. Fixed matched-teacher training and held-out probes run before updates and every epoch. Full latest and best-FID checkpoints include model, optimizer, scheduler, and all-rank RNG state, with remote digest verification. Preflight weights are discarded; production loads the same verified parent best checkpoint again.

Matched reference components are:

| Parent epoch | Mean | Covariance | FID |
| --- | ---: | ---: | ---: |
| 60 | 2.846459 | 7.168579 | 10.015038 |
| 65 | 3.347664 | 7.920665 | 11.268329 |
| 70 | 3.251959 | 7.259545 | 10.511505 |

Covariance is not monotonically worsening: it spikes at 65 and recovers substantially by 70. Approximately 82% of the epoch-60-to-70 FID increase is in the mean term. The pilot must therefore report both terms and total FID, comparing the same sampling seed, reference, and optimizer steps. A lower covariance term accompanied by worse mean and total FID is not declared a fix. Apparent improvements require an independent matched 50k sampling seed on the candidate and parent baseline before a robust improvement claim. This continuation tests the teacher/objective package's ability to reverse deterioration at this checkpoint, not a unique historical root cause or its possible from-scratch benefit.

The predecessor stopped cleanly at epoch 73, step 8,979. Its final last and best checkpoints were reverified in W&B artifact `helloimlixin-rutgers/laser/church-ffhq-repair-norefit-b1024-8h100-20260927-selected-checkpoints:v104`. The preserved best remains FID 10.0150 at epoch 60; the earlier matched-reference best remains 9.6012. Existing durable checkpoint files were retained.

Distributed preflight passed on all eight ranks: the parent checkpoint was restored, three real optimizer updates completed through step 7,383, the saved model reloaded strictly, all model and Adam tensors were finite, and all 517 optimizer entries, the scheduler, and eight RNG states were checked. Peak allocated GPU memory was 55,310,040,064 bytes. Sampling and matched-teacher held-out monitoring passed. The 1,024-sample FID smoke score is only an infrastructure check and is not comparable to the 50k baseline.

Validation was uploaded and all 87 recorded files were verified in `helloimlixin-rutgers/laser/church-combination-norefit-b1024-8h100-20260928-validation:v0`. Production launched as PID 50336. Run: https://wandb.ai/helloimlixin-rutgers/laser/runs/church-combination-norefit-b1024-8h100-20260928 . FID mean/covariance components, baseline deltas, matched-parent deltas, and per-evaluation feature statistics are recorded online; full latest and best-FID checkpoint uploads retain remote digest verification.

The production 50k baseline reproduced FID 10.015037123, mean 2.846458435, covariance 7.168578688; total FID differs from the parent by only -4.30e-7. Statistics were committed to `church-combination-norefit-b1024-8h100-20260928-fid-statistics:v0`. Training was observed healthy through step 7,500 at approximately 2,200 images/second between ancillary work, with a sample grid at step 7,400. Full last and best checkpoint files were verified online in `church-combination-norefit-b1024-8h100-20260928-selected-checkpoints:v1`. This verifies the relaunch and upload path, not a covariance improvement; the next matched 50k measurement is epoch 65. The observed launch state is recorded in `validation/launch-health.json`.

The pilot subsequently completed its planned ten additional epochs and stopped at epoch 70, step 8,610. It failed the covariance target:

| Pilot epoch | Mean | Covariance | FID |
| --- | ---: | ---: | ---: |
| 60, baseline | 2.846458 | 7.168579 | 10.015037 |
| 65 | 3.362749 | 7.708245 | 11.070994 |
| 70, endpoint | 3.290274 | 7.955266 | 11.245540 |

Endpoint covariance increased by 0.786687 (about 11%) relative to the baseline. Relative to the unchanged parent's epoch-70 control, covariance was worse by 0.695721 and FID by 0.734036. Epoch 65's temporary covariance advantage over its matched parent did not persist. The new teacher's fixed held-out component KL improved from 16.278188 to 15.743973 while generation worsened; this reinforces that teacher-loss improvement alone is not the desired outcome. These results reject this particular warm-start teacher/objective bundle as a covariance fix, without establishing a unique root cause or disproving every alternative whole-combination teacher.

Full endpoint last and best checkpoints were committed and verified online in `helloimlixin-rutgers/laser/church-combination-norefit-b1024-8h100-20260928-selected-checkpoints:v16`. The best remains the reproduced epoch-60 baseline at FID 10.015037. The training process exited, and all eight GPUs were idle when the user requested the follow-up status. No extension or new training was launched.
