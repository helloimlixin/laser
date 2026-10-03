Church's current training noise is internally consistent and conservative enough to serve as the control. The next candidate should increase atom noise mainly at earlier depths, while keeping coefficient temperature at 0.125. Uniformly increasing both temperatures is poorly supported by the reconstruction measurements.

This diagnostic used fresh pixels from 64 training and 64 validation images with the selected frozen Church stage 1. Atom measurements average three independent draws per temperature. Coefficient measurements compute the exact conditional expected latent squared error, including variance and mean shift, on one fixed stochastic support draw. A supplementary pixel check decodes 16 images per split with one coefficient draw. These are exploratory representation diagnostics, not generated-image FID measurements or an independent test set for the chosen candidate.

The successful FFHQ run `ffhqcmp0804205803` used the following, verified against its saved trainer and W&B configuration:

| Source of randomness | Successful FFHQ | Current Church trial |
|---|---|---|
| Training atoms | Deterministic cached two-step OMP; no support noise | Four-step stochastic OMP bank, 16 trajectories/site, temperature 0.0625 |
| Training coefficient input | Fresh categorical draw every visit | Fresh categorical draw every visit |
| Coefficient target | Full soft distribution at temperature 0.5 in normalized units | Full soft distribution at temperature 0.125 in physical units |
| Approximate coefficient standard deviation | 0.5 normalized; 18.10 and 4.29 physical units at the bin center | 0.250, 0.250, 0.248, 0.244 physical units |
| Residual dropout | 0.1 | 0.2 |
| Global batch | 128 | 2,048 |
| Generation atoms | Temperature 1, top-k 250, top-p 1 | Same |
| Generation coefficients | Temperature 1, **top-p 0.85** | Temperature 1, top-p 1 |

The FFHQ generation top-p entry corrects the previous audit table, which incorrectly said 1.0. Training coefficient sampling uses the entire target distribution; generation top-p is a separate operation.

FFHQ's coefficient kernel is `q_j ∝ exp(-(u - center_j)^2 / 0.5)`, using 2,048 uniform centers in `[-3,3]`. Away from the endpoints its variance is approximately `temperature / 2`, giving normalized standard deviation 0.5. The depth scales are 36.2083 and 8.5833. Near the endpoints, truncation narrows and shifts the distribution. These physical magnitudes cannot be transferred directly to a different tokenizer. At a centered coefficient the entropy is 6.558 nats, or approximately 705 effective bins; that count also depends on bin spacing and is not a geometry-independent noise measure.

Church uses physical coefficients and shared nonuniform Lloyd–Max centers. Its active kernel is `q_j ∝ exp(-(c - center_j)^2 / 0.125)`. Input coefficients are sampled from precisely the distribution used for soft cross-entropy targets. Sampled atom supports retain their matching least-squares coefficients, and already selected atoms are excluded. The frozen dictionary has unit columns, so a coefficient perturbation has a direct physical latent interpretation. The paper's general principle is to pair sampled inputs with matching soft targets; its code-vector temperature is not a universal scalar to copy into LASER's differently scaled atom and coefficient kernels. [RQ-Transformer stochastic sampling and soft labeling](https://arxiv.org/html/2203.01941#S3.SS2.SSS3).

On this Church validation probe, coefficient RMS magnitudes were `[7.508, 4.103, 2.684, 1.722]`. Current noise standard deviations are approximately 3.3%, 6.1%, 9.3%, and 14.2% of those magnitudes. Thus, the same physical coefficient temperature produces stronger relative noise at later depths. Nonuniform centers also cause small mean shifts; these are included in the distortion calculation.

With continuous least-squares coefficients, increasing one scalar atom temperature gives:

| Atom temperature, all depths | Changed ordered supports | Extra latent MSE over greedy OMP |
|---|---:|---:|
| 0.03125 | 21.4% | 0.37% |
| **0.0625, current** | **39.3%** | **1.99%** |
| 0.125 | 65.2% | 9.57% |
| 0.25 | 89.7% | 39.56% |
| 0.5 | 99.0% | 111.91% |

Current atom-selection entropy by depth is `[0.036, 0.126, 0.360, 1.158]` nats. Early selections are already very sharp, while the last depth is much more stochastic. Raising every depth together spends much of the added noise on the last depth.

Depth-specific temperatures improve that tradeoff on this probe:

| Temperatures, depths 1–4 | Changed ordered supports | Extra latent MSE | Entropy by depth, nats |
|---|---:|---:|---|
| `[0.0625, 0.0625, 0.0625, 0.0625]` | 39.3% | 1.99% | `[0.036, 0.126, 0.360, 1.158]` |
| `[0.125, 0.125, 0.0625, 0.0625]` | 42.9% | 2.10% | `[0.073, 0.286, 0.345, 1.151]` |
| `[0.25, 0.125, 0.0625, 0.0625]` | 44.6% | 2.20% | `[0.151, 0.286, 0.346, 1.151]` |
| **`[0.25, 0.25, 0.125, 0.0625]`** | **54.8%** | **3.16%** | **`[0.151, 0.673, 0.778, 1.011]`** |

The training probe agrees with this direction: the final profile changes 55.2% of supports for 2.77% extra latent MSE. The experimental sampler was checked against the original implementation with a uniform profile and identical RNG: atoms and coefficients were bitwise identical. No production sampling or training code was changed for this test.

With atom temperature fixed at 0.0625, increasing coefficient noise gives:

| Coefficient temperature | Approximate physical sigma | Total extra expected latent MSE over greedy OMP | Mean pixel PSNR, 16 validation images |
|---|---:|---:|---:|
| 0, nearest center | 0 | 2.00% | 19.043 dB |
| 0.03125 | 0.125 | 4.53% | 19.028 dB |
| **0.125, current** | **0.25** | **12.02%** | **19.001 dB** |
| 0.25 | 0.35 | 21.85% | 18.962 dB |
| 0.5 | 0.47–0.50 | 41.09% | 18.896 dB |
| 1.0 | 0.64–0.71 | 78.53% | 18.696 dB |

The latent percentages include atom-support distortion and coefficient noise together. The atom-only table excludes coefficient noise. The small pixel sample is descriptive; PSNR and latent MSE do not establish generated-image FID or perceptual quality. These measurements support holding coefficient temperature at 0.125 for the next atom-noise experiment, not declaring it universally optimal.

Increasing the finite bank's size supplies more rare alternatives, but does not change the underlying sampling kernel. On the same 256 validation sites, 16, 32, and 64 draws produced 5.43, 8.41, and 13.26 distinct ordered supports on average. The probability that two independent visits select the same support only fell from 50.08% to 48.31% to 47.48%. Expanding the bank alone therefore adds substantially less effective variability than the unique-support counts might suggest. The larger earlier measurement of 4.80 distinct supports used a different, 100,000-site training sample.

The prepared next candidate uses atom temperatures `[0.25, 0.25, 0.125, 0.0625]`, coefficient temperature 0.125, and the same 16-trajectory bank size. It requires rebuilding paired trajectories with coefficients refitted on each sampled support. A matched training experiment must keep dropout, initialization, optimizer, batch, coefficient codebook, and FID protocol fixed. The active dropout trial retains its current noise policy, preserving that control. The candidate is calibrated but has not been launched or shown to improve FID.

During this audit, the dropout trial was found stopped after epoch 4 due to retained optimizer storage after checkpoint serialization. The restored older runtime lacked a serialization garbage-collection fix needed in the current PyTorch environment. A storage-lifetime regression failed before the fix and passed afterward; the exact-batch tests also passed (8 total). Explicit collection after checkpoint save fixes the reproduced retention issue. The original microbatch, accumulation, optimizer, scheduler, and four RNG streams were retained when resuming step 248. Epochs 5 and 6 subsequently completed FID50k and training advanced past step 430. At epoch 6, allocated GPU memory after FID was 17.56 GiB on every rank, without the prior rank-zero accumulation. The full epoch-5 checkpoint was verified online at step 310. The recovery source and metadata are uploaded separately to the same W&B run. This repair changes checkpoint memory management, not the noise experiment.

Reproduction: [noise calibration](../scripts/tools/calibrate_church_pair_noise.py) and [depth/bank probe](../scripts/tools/probe_church_depth_noise.py). Machine-readable measurements, sampled image indices, and the encoded probe are in `/mnt/laser-church/dropout-experiment/noise-calibration/`. The active run is [church-rqrecipe300-dropout02-b2048-4h100-20260924](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-rqrecipe300-dropout02-b2048-4h100-20260924).
