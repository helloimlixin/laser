# Expanded full-combination targets, 2026-09-27

The previous teacher offered nine complete four-atom supports: greedy OMP plus two replacements at each position. This constrained both the number of alternatives and simultaneous atom changes. The expanded teacher keeps sparsity at four atoms but searches a substantially wider deterministic pool on each fresh image.

For each slot, all 16,384 dictionary atoms are ranked by the error after replacing that slot and jointly refitting all four coefficients. The pool combines the best 128 replacements per slot, the original anchor, and all 9^4 combinations of each slot's anchor and eight leading replacements. Duplicate ordered supports and singular combinations are excluded. From approximately 7,000 searched combinations, the anchor plus the best 255 are retained. Final support probabilities use complete sparse reconstruction error, with support temperature 0.0625.

Coefficient targets retain the correlated inverse-Gram Gaussian construction and exact prefix conditioning over the specified finite mixture. The calibrated coefficient temperature is 1.7737924579079655e-05. Compact Gaussian evaluation uses an omitted-mass upper bound of 1e-30 and falls back to the full bin grid when needed. It changes neither the vocabulary nor the model's pair ordering. Earlier refitted coefficients can still identify future supports, so conditional atom targets remain much sharper than the complete-support prior.

## Calibration

The same 384 fresh ImageNet training-image crops used for the prior teacher were retained: 128 for temperature calibration and 256 disjoint images for checking. These are not a validation-set generalization experiment. The coefficient temperature matches approximately 80% of the official RQ quantizer's calibration entropy at its configured temperature 0.5. Three stochastic draws are used for the latent checks and 32 fixed images for decoded distortion.

| Check | Expanded teacher | Previous nine-support teacher | Original RQ quantizer |
|---|---:|---:|---:|
| Total conditional entropy, nats / four-pair site | 2.3563 | 2.4067 | 2.8376 |
| Sampled / deterministic latent error | 0.97000 | 0.97713 | 1.00936 |
| Perturbation energy / deterministic latent energy | 1.5620% | 1.4617% | 6.0608% |
| Decoded sampled MSE, pixels in [0,1] | 0.00818895 | 0.00820393 | 0.00978812 |

Absolute latent errors across the two encoders are not comparable. Their normalized ratios use each encoder's own deterministic baseline. Target entropy and reconstruction distortion do not establish prediction difficulty or generation FID improvement.

## Coverage and approximation

At the old support temperature 0.125, the original nine candidates contain only 66.0% of the expanded pool's probability mass on calibration images. The wider pool also includes simultaneous changes to multiple support positions.

At the selected temperature 0.0625, the retained 256 contain 99.9219% of the searched pool's mass on calibration images and 99.9706% on check images, averaged across sites. The check-set first percentile is 99.9721%.

These percentages are relative to the finite search pool. A separate audit against the union of **all 65,520 valid single replacements** and the searched Cartesian multi-position alternatives gives 98.4355% mean retained mass on check sites; the first percentile is 40.10%, and the minimum is 1.17%. Some sites therefore retain a small fraction of a much flatter full-vocabulary distribution. This remains a truncated finite-support teacher, not enumeration of all four-atom supports or exact global Gibbs sampling.

## Implementation and compute

Small full-support solves are compiled as batched elementwise operations. Duplicate detection and atom-label entropy use linear candidate memory. A float64 least-squares cross-check on 4,096 real candidate combinations found maximum coefficient error 1.01e-5, complete-energy error 1.17e-6, and covariance error 7.87e-6. Eight targeted tests cover full-combination scoring, simultaneous changes, deduplication, prefix-conditional probabilities, soft-loss gradients, reproducibility, and compact coefficient kernels.

The continuation is running under `/mnt/laser-imagenet-combination-expanded-20260927`; persistent evidence is under `outputs/imagenet-rfid421-combination-expanded-20260927`. It inherits the complete model, Adam state, data cursor, RNG streams, and original learning-rate schedule. The total remains 63,500 optimizer updates / 100 epochs, with global batch 2,016, FID every five epochs, and preview samples every 200 updates. The prior epoch-90 evaluation is allowed to finish before switching.

Contended sidecar measurements are not final training throughput: with other training on the same GPU, the nine-support teacher took 1.329 s per rank batch and the expanded teacher 1.804 s. The selected site chunk is 4,032. Actual eight-GPU training and durable recovery are checked after launch.

With the parent acknowledged stopped and the GPU otherwise idle, the same 252-image per-rank batch took 0.5537 s for the nine-support teacher and 0.8253 s for the expanded teacher (13.36 GB maximum allocated in the standalone probe). This adds about 0.272 s of teacher time per update; actual DDP throughput is measured separately. The preceding nine-support continuation reached FID 21.8770 at epoch 90, before switching; this is not a result from the expanded teacher.

The frozen runtime passed 34 targeted tests, with two unrelated VAR-backend checks excluded because FoundationVision/VAR is not installed.

## Live verification

The expanded run resumed at step 57,151 (epoch 90, batch 1), preserving all 870 model tensors, 870 Adam states, and eight RNG streams. All eight ranks passed finite-state checks after 20 retained updates. As of step 57,350, the median logged throughput after compilation is 1367.8 images/s, compared with 1,710.1 for the previous teacher (about 20% slower). Median online conditional entropy is 2.4068 nats per complete site.

The first full recovery checkpoint at step 57,200 passed strict model/optimizer/scheduler reload, finite-value checks, and matching SHA-256 checksums of the local and persistent payloads. Its digest is `88921c569cc2752e6a56a8ceb16137bccc50bcd320f92c184b84b476aa36e27f`. No generation FID has yet been measured for the expanded teacher; the next scheduled evaluation is epoch 95.
