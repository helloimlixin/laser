# Calibrated coefficient noise on the archived FFHQ/Church baseline

The user requested calibration after the copied FFHQ target distribution was shown to perturb Church coefficients by approximately half their RMS magnitude. This experiment changes only coefficient target/context noise on the verified archived model. It does not reuse the older 219M calibrated experiment, which also changed architecture, augmentation, regularization, sampling, and learning rate.

The running unmodified comparison is `church-ffhq-archived-control-20260911`. Its model, data order, optimizer, LR, sampler, and source files remain untouched. The looped experiment is paused at update 1,601, with model/optimizer/stream preserved, to free GPU 0. Receipt: `outputs/church-ffhq-noise-20260911/paused-looped.json`. The new calibrated non-looped model starts from random stage-2 weights and empty optimizer state, with the same initialization seed and common parameter fingerprint as the original control. No trained stage-2 weights are transferred.

## Distribution and calibration

Use the existing signed 2,048 coefficient bins and depth scales. For a clean physical coefficient c and a physical bin center b, the target is proportional to `exp(-(c-b)^2/(2*sigma^2))`. Both the soft labels and sampled autoregressive context coefficients use this same distribution. The frozen dictionary has unit-norm columns; physical coefficient differences therefore express the size of each latent contribution's perturbation.

The chosen **physical sigma is 0.1875 for all four depths**. Previously it was `[3.70044, 2.08012, 1.22768, 0.82423]`. The new noise is approximately 2.5%, 4.6%, 7.1%, and 11.0% of per-depth coefficient RMS, rather than 46–51%. Multiple neighboring bins retain probability mass; this remains stochastic soft-target training.

The physical temperature is 0.0703125. Equivalent normalized temperatures differ by depth: `[0.00128371, 0.00406252, 0.01166286, 0.02587473]`. Reusing one normalized temperature across different depth scales was the source of the unwanted physical noise variation.

Before measuring candidates, `calibration/protocol.json` recorded widths `[0.0625, 0.125, 0.1875, 0.25, 0.5]`, selection on 512 randomly chosen training images, and confirmation on 512 disjoint training images. Select the largest candidate passing both: LPIPS mean plus two standard errors <=0.01 and relative latent MSE <=0.005. This retains the earlier distortion criteria. Selection uses the full untruncated distribution, as training does. The 300 official validation images are evaluated only after selection and do not choose the width.

All measurements below supply correct supports and continuous coefficients to the frozen tokenizer/decoder, and compare against its clean reconstruction. They are **not generation FID, image reconstruction FID, or evidence that the learned prior has improved**.

| Noise | Calibration LPIPS | LPIPS upper 2SE | Relative latent MSE | Accepted |
|---|---:|---:|---:|---|
| Old normalized T=0.5 | 0.25685 | 0.25972 | 0.25382 | No |
| Physical sigma 0.125 | 0.002888 | 0.002953 | 0.000790 | Yes |
| **Physical sigma 0.1875** | **0.006074** | **0.006205** | **0.001778** | **Yes** |
| Physical sigma 0.25 | 0.010113 | 0.010325 | 0.003160 | No |
| Physical sigma 0.5 | 0.031924 | 0.032518 | 0.012655 | No |

The separate 512-image confirmation measured LPIPS 0.006023 (upper bound 0.006155) and relative latent MSE 0.001764. On official validation, full-distribution LPIPS is 0.006194 and PSNR 36.94 dB; with the unchanged p=0.85 generation cutoff, LPIPS is 0.003474 and PSNR 39.65 dB. The observed per-depth full-distribution coefficient RMS perturbations are approximately 0.187, confirming the intended units.

Calibration manifests include image indices, seeds, candidate results, selected width, tokenizer/cache hashes, and the exact target-code hash. The trainer checks these against its loaded data and refuses a different calibration on resume. Clean/sampled reconstruction grids and full records are under `outputs/church-ffhq-noise-20260911/calibration`.

## Controlled training

The new run uses the same 404,738,048-parameter archived FFHQ-v4 class with official Church dimensions: width 1,024, 24 spatial blocks, four depth blocks, two coefficient micro-transformer blocks, and four coefficient classifiers. Complete support/coefficient pairs condition both autoregressive streams. The only behavior replacement is the auxiliary method producing the coefficient target distribution and sampled context IDs.

All of these match the original control: full 126,227-image continuous center-crop cache; frozen Church tokenizer; batch 256/microbatch 32; seed 0; 493 updates per epoch; 300-epoch cosine from 5e-4 to zero without warmup; AdamW betas (0.9,0.95), weight decay 1e-4; dropout 0.1; atom loss weight 1.5; distribution-geometry weight 0.05 delayed two epochs and ramped over three; gradient clipping at 1; archived precision scopes; atom top-k250/p1 and coefficient p0.85 sampling.

FID-4096 screens use the same seed and sample count at epoch one and every five epochs. FID-50k runs at epoch 10 and every 50 epochs, with an independent-seed confirmation of the selected checkpoint at completion. Compare matched training epochs and sample counts with the retained control. Raw coefficient cross entropy and KL have changed target distributions and must not be treated as direct quality comparisons. Common support likelihood, physical coefficient error, sample inspection, and generation FID provide the relevant comparisons.

Verification covers physical standard deviation at every depth, consistent labels/context distributions, stochastic sampling, boundary behavior, unchanged training/evaluation code, inherited pair causality/cache behavior, actual full batches, and exact checkpoint/resume through geometry activation and generation-based checkpoint selection. The production launcher waits for verification and checks source hashes. The previous looped run remains recoverable.

Code: `src/church_coefficient_noise.py`, `scripts/calibrate_church_ffhq_noise.py`, `scripts/train_church_ffhq_noise.py`, `scripts/launch_church_ffhq_noise.py`. Production run ID: `church-ffhq-physical-noise-20260911`; output: `outputs/church-ffhq-noise-20260911/calibrated`.

## Dictionary, coefficient-range and sparsity audit

The subsequent audit explicitly checks the user's three transfer concerns. The archived FFHQ configuration has 2,048 atoms, two active coefficients, and physical bin ranges ±108.625 and ±25.75. Church has 16,384 atoms, four active coefficients, and ranges approximately ±22.203, ±12.481, ±7.366 and ±4.945. Both use 2,048 scalar coefficient bins. FFHQ's actual coefficient population and dictionary checkpoint are unavailable locally; these FFHQ numbers are configuration evidence, not a measured FFHQ noise-to-signal comparison.

Dictionary cardinality is not itself a multiplier on coefficient perturbation. For fixed support S, latent error is `delta_z = D_S delta_c`. For independent draws with error mean mu and diagonal variance V, its expected squared norm is `trace(D_S^T D_S V) + mu^T D_S^T D_S mu`. The second term accounts for finite-range and discrete-bin bias. For zero-mean independent errors and unit-norm atoms this reduces to `sum(sigma_d^2)`. Thus doubling active depth doubles expected noise energy at fixed sigma; holding absolute noise energy fixed would require dividing sigma by sqrt(2). Changing dictionary size requires checking the actual atom norms, active support geometry and signal energy, rather than scaling noise by 16,384/2,048. Wrong-support errors are a separate source of latent error.

`scripts/audit_church_noise_scale.py` reads the production target implementation, exact frozen dictionary, continuous cache and calibration manifest. It verifies hashes and scales, then evaluates exact moments of the full discrete target distribution on 1,024 additional training images, disjoint from both selection and confirmation. It also measures coefficient magnitudes and atom usage over all 126,227 training images. It uses CPU only and does not alter training or select a new width. Records and image indices: `outputs/church-ffhq-noise-20260911/scale-audit/{protocol,audit}.json`.

| Church depth | Coefficient RMS, full training | Physical bin spacing | Sigma in bins | Sigma / coefficient RMS |
|---|---:|---:|---:|---:|
| 1 | 7.4020 | 0.02169 | 8.64 | 2.53% |
| 2 | 4.0890 | 0.01219 | 15.38 | 4.59% |
| 3 | 2.6555 | 0.007197 | 26.05 | 7.06% |
| 4 | 1.7008 | 0.004832 | 38.81 | 11.02% |

All active atoms have unit norm to within 4e-7. Active support absolute inner product has median 0.1008 and p99 0.3544; the largest support-Gram eigenvalue has p99 1.6205. These statistics are from actual selected supports. Nearest-atom correlations are separately sampled from 256 dictionary anchors and must not be described as an exhaustive dictionary coherence measurement. No duplicate supports occurred in the audit images.

Exact expected per-image relative latent MSE averages **0.00177792 (0.178%)**, versus **0.254822 (25.5%)** for the original normalized temperature. The calibrated width passes the existing mean limit of 0.005; all 1,024 individual image expectations are also below it (maximum 0.0029393). The aggregate latent RMS error / signal RMS is **4.18%**. This computation sums all four active contributions using their dictionary vectors, so the Church sparsity and dictionary geometry are accounted for. The separate earlier decoder confirmation still provides the image-space check: LPIPS mean 0.006023.

The average criterion is not a uniform guarantee. Site-level expected relative latent MSE has p95 0.003996, p99 0.005530 and maximum 0.02722; 1.68% of sites exceed 0.005. Fourth-depth noise RMS relative to each nonzero coefficient's magnitude has median 11.56%, p95 20.61% and p99 29.78% in the audit. Its expected sign-flip probability averages 0.00611%. In the full training population, 0.520% of fourth-depth magnitudes are below three sigma. A shared sigma therefore remains proportionally stronger on weak coefficients, even though its aggregate distortion is small. No per-site or per-coefficient cap was added silently.

The current 0.1875 width remains unchanged to complete the controlled comparison. It is calibrated for this frozen Church dictionary, coefficient population and depth, not a portable constant for another tokenizer. Any new dictionary, coefficient scale/binning or sparsity setting must repeat the calibration and these diagnostics. If generation or weak-coefficient errors motivate a further change, depth-relative or magnitude-dependent widths require a separate calibration and experiment; coefficient smoothing alone does not prove robustness to autoregressive support mistakes.

Three additional tests pass: expected energy versus Monte Carlo with biased discrete draws and correlated atoms; invariance to adding inactive dictionary columns; and the predicted sparsity/noise scaling. Production source files and the running noise distribution are unchanged by this audit.
