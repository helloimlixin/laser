Deterministic OMP with stochastic coefficients in normalized units is a closer transfer of the successful FFHQ noise mechanism. It is a plausible comparison, but the evidence does not establish that Church needs deterministic supports: the stronger Church 9.955 run used stochastic supports.

The proposed four-depth distribution is:

`u_d = c_d / s_d`

`q_d(j) ∝ exp(-(u_d - b_j)^2 / tau_d)`

Use a fixed deterministic four-atom OMP support, refit its four physical coefficients, normalize each coefficient by its Church depth scale, sample a new coefficient token from `q_d` on every training visit, and use that same full distribution as the coefficient CE target. Keep both atom and coefficient predictors fully autoregressive, with dictionary-vector conditioning. Atom targets remain deterministic; previously sampled coefficients still condition every later event.

The verified deterministic Church cache already provides the required supports and normalized coefficients. It has 126,227 images, an 8×8×4 code shape, FP32 coefficients, 2,048 uniform coefficient bins over `[-3,3]`, and scales `[7.662354, 4.158035, 2.633324, 1.651212]`. These scales were fitted as each depth's maximum absolute training coefficient divided by three. They are **range scales, not RMS normalization**. The selected tokenizer hash remains `762c51a10267ed6fa55709ff0d6cf997940d21056b16c7a0a0399b3ebf93868d`.

The exact successful FFHQ coefficient kernel works unchanged with four coefficient positions. Its temperature is 0.5 in normalized units. Away from bin boundaries, normalized variance is approximately `tau/2`:

| Interpretation | Four-depth temperature | Sigma per normalized coefficient | Sum of normalized variances |
|---|---:|---:|---:|
| Same per-coefficient distribution as FFHQ | 0.5 | 0.5 | 1.0 |
| Same total normalized variance as two-depth FFHQ | 0.25 | 0.3536 | 0.5 |
| Intermediate lower-noise control | 0.125 | 0.25 | 0.25 |

Two-depth FFHQ at temperature 0.5 has total normalized variance approximately 0.5. The 0.25 row therefore follows only if the desired matching criterion is that total. Four depths alone do not force halving the temperature: signal energy changes too, and physical latent noise depends on the depth scales and support geometry. FFHQ's actual coefficient population is not available locally, so equal normalized temperature has not been shown to match its noise-to-signal ratio. Its recorded depth scales are `[36.208333, 8.583333]`.

The new probe uses deterministic OMP and the actual target implementation on the same 64 training and 64 validation photographs as the preceding exploratory noise diagnostic. It computes exact discrete target moments. Normalized targets are checked against the explicit formula; the saved FFHQ method is also checked at depth four with the same coefficients and RNG, including boundary/out-of-range inputs. Coefficient IDs and full soft targets match bitwise at temperatures 0.125, 0.25, and 0.5.

Validation measurements:

| Target | Physical sigma by depth | Expected added latent-noise energy / clean latent energy | Pixel PSNR against photograph, 16 images |
|---|---|---:|---:|
| Physical T=0.125 control | `[0.25, 0.25, 0.25, 0.25]` | 0.30% | 19.02 dB |
| Normalized T=0.125 | `[1.916, 1.040, 0.658, 0.413]` | 6.47% | 18.09 dB |
| Normalized T=0.25 | `[2.709, 1.470, 0.931, 0.584]` | 12.94% | 17.37 dB |
| Normalized T=0.5 | `[3.826, 2.077, 1.315, 0.824]` | 25.83% | 16.37 dB |

Clean deterministic reconstruction has mean PSNR 19.08 dB on those 16 validation images. No sampled clean coefficients fell outside the bin range in this probe. The training split gives similar expected noise-energy fractions: 0.32%, 6.76%, 13.51%, and 26.97%. The percentages here use **clean latent energy** as their denominator, unlike the earlier atom-noise table's excess reconstruction error. They must not be compared as if they used the same denominator.

These measurements show that normalized coefficient noise is much stronger than the active physical-temperature recipe. They do not prove it is too strong for training: regularization can improve generation while degrading individual noisy reconstructions. My earlier reconstruction-based preference for gentle physical noise was a conservative diagnostic criterion, not a demonstrated generative optimum.

There is also existing training evidence that must not be overlooked. [Church paper-batch run](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-compound-rqpaper-b2048-4h100-20260923) already used deterministic OMP, four depths, these same scales and bins, normalized temperature 0.5, fresh stage-2 weights, and global batch 2,048. Its configuration was re-read from W&B for this audit. Its historical best was FID50k 19.4865 under atom top-k 1,400 and coefficient top-p 1. The separately recorded fixed-checkpoint FFHQ-sampler evaluation gave 13.7017 with atom top-k 250 and coefficient top-p 0.85. That sampler comparison is documented in [the repair report](church-ffhq-adaptation-repair-2026-09-24.md).

The subsequent repair used physical temperature 0.125, batch 128, a new lower-LR optimizer/schedule, and additional training, reaching FID50k 11.6910. Because multiple settings changed, this is not an isolated temperature ablation. Likewise, FFHQ differed in batch, sparsity, vocabulary, dataset, tokenizer and geometry loss. Neither history establishes that deterministic OMP or one temperature is optimal.

The concrete next comparison is normalized T=0.25 versus T=0.5 on deterministic supports, holding the normalization, codebook, initialization, batch, dropout, optimizer, loss, and FID sampler fixed. T=0.5 is the literal per-coefficient FFHQ reference; T=0.25 tests the equal-total-normalized-variance interpretation. This requires fresh stage-2 training: the active stochastic/raw-coefficient run uses a different coefficient codebook, so its weights should not be silently reinterpreted under the normalized vocabulary. No active training policy was changed and no additional training run was launched by this diagnostic.

The prepared comparison is `/mnt/laser-church/dropout-experiment/normalized-coefficient-probe/proposed-comparison.json`; full measurements and provenance are in that directory. Reproduction: [probe script](../scripts/tools/probe_church_normalized_coefficient_noise.py).
