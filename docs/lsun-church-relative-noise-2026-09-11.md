# Bounded magnitude-relative coefficient noise

The fixed physical sigma 0.1875 has small aggregate Church reconstruction distortion, but its relative RMS error reaches about 30% for the weakest 1% of fourth-depth coefficients. The user requested a fix. This experiment changes the coefficient target/context distribution so its width shrinks with each coefficient's magnitude and its tails have finite support.

## Target rule and quantization exception

For clean physical coefficient c, use `sigma(c) = min(0.1875, 0.05 * abs(c))`. The existing physical bin centers receive Gaussian weights with this sigma, restricted to centers satisfying `abs(b-c) <= 3*sigma(c)`. The nearest bin is always eligible. Therefore each supported perturbation obeys `abs(b-c) <= max(min(0.5625, 0.15*abs(c)), nearest_bin_error)`, up to the existing FP32 normalization round trip.

The nearest-bin exception is explicit: the existing 2,048 uniformly spaced signed bins do not include exact zero and cannot represent every tiny coefficient within a fixed relative tolerance. If no center falls inside the relative interval, use a deterministic nearest-bin target. There is no positive noise floor that would broaden weak coefficients. Neither coefficient bins nor the frozen tokenizer change. Soft labels and stochastic history IDs use exactly the same distribution. The sampler at generation remains the archived learned-distribution sampler; the relative bound concerns training target/context perturbations around known coefficients, not unknown errors of generated predictions or wrong supports.

## Calibration and measured effect

Before measurement, `outputs/church-relative-noise-20260911/calibration/protocol.json` records candidate relative widths 0.025, 0.0375 and 0.05, fixed physical cap 0.1875, three-sigma truncation, and 512 fit/512 disjoint confirmation images from the 126,227-image training population. Choose the largest candidate passing both sets: LPIPS mean + two standard errors <=0.01; sampled and exact expected per-image relative latent MSE <=0.005; p99 coefficient RMS error/magnitude <=0.052 at every depth; zero support-bound violations. The 300 official validation images are reported only after selection. This selected 0.05.

The evaluation uses the actual 16,384-atom frozen Church dictionary and all four active coefficients together. Exact finite-bin moments include quantization/truncation bias and active-support cross terms; coupled inverse-CDF samples pass through the frozen decoder. Correct supports and clean continuous reconstructions are supplied. These metrics are reconstruction perturbation diagnostics, not rFID or unconditional generation FID.

| Matched fit images | Fixed sigma 0.1875 | Bounded relative sigma 0.05 |
|---|---:|---:|
| LPIPS | 0.006054 | 0.003986 |
| Sampled relative latent MSE | 0.001763 | 0.001087 |
| Fourth-depth p99 relative coefficient RMS error | 29.92% | 4.94% |
| Site p99 expected relative latent MSE | 0.005465 | 0.002308 |
| Maximum site expected relative latent MSE | 0.013065 | 0.003304 |

The independent 512-image confirmation has LPIPS 0.004022 (upper 2SE 0.004104), sampled relative latent MSE 0.001091, and fourth-depth p99 relative coefficient RMS error 4.939%. Mean physical RMS perturbations by depth are 0.18472, 0.17057, 0.12980 and 0.08412. The cap preserves approximately the previous absolute width for strong first-depth coefficients, while weaker later coefficients receive smaller perturbations. All depths have zero exact target sign-flip probability in the fit, confirmation and validation data; no fallback or support-bound violation occurred in these samples. Tests separately cover zero and tiny coefficients where fallback is needed.

Official validation reports LPIPS 0.004124 and sampled relative latent MSE 0.001093; fourth-depth p99 relative coefficient RMS error is 4.939%. All measured individual target RMS/magnitude ratios in fit, confirmation and validation are below 5%; this finite-data observation does not override the unavoidable nearest-bin exception for arbitrary inputs.

Calibration records the data/decoder/target-source hashes, exact indices, candidate results, expected error distributions, realized decoder distortions, sign-flip probabilities and bin fallbacks. This rule is calibrated for this dictionary, coefficient population and active depth. Recalibrate if those change.

## Controlled scratch training

The fixed-sigma run `church-ffhq-physical-noise-20260911` continues on GPU 0 as the direct comparison. Its first-epoch FID-4096 was 133.32; that remains poor and does not establish that the new bounded rule will improve generation. The older original large-noise control `church-ffhq-archived-control-20260911` is saved and paused at step 3,791, with optimizer and stream verified, to free GPU 1. Pause receipt: `outputs/church-relative-noise-20260911/paused-legacy-control.json`. The earlier looped experiment also remains paused and recoverable.

The new `church-relative-noise-20260911` run starts from random stage-2 weights and an empty optimizer, using the same seed and parameter fingerprint as the fixed-sigma control. It uses the actual archived FFHQ-v4 compound model: 404,738,048 parameters, width 1,024, 24 spatial blocks, four depth blocks and two coefficient micro blocks. Full support/coefficient pairs continue conditioning both autoregressive streams. The stage-1 tokenizer is frozen and already had its one-epoch Church finetune.

Training retains the full 126,227-image continuous cache, batch 256/microbatch 32, 493 updates per epoch, 300-epoch cosine from 5e-4 without warmup, AdamW betas (0.9,0.95), weight decay 1e-4, dropout 0.1, clipping 1, atom weight 1.5, and geometry weight 0.05 delayed two epochs and ramped over three. Sampler, precision scopes, evaluation and checkpoint selection are unchanged. FID-4096 is measured at epoch one and every five; FID-50k at epoch 10/every 50, with independent selected-checkpoint confirmation at completion. Compare matched epochs and sample counts, not raw coefficient cross entropy across different targets. This experiment does not address the separate LR-schedule concern yet.

Production launch is gated on the calibration, 15 unit tests, full-size batch execution, exact continuous-versus-resumed model/optimizer/stream state through six updates and geometry activation, and the existing evaluation/selection path. The running comparison's source hashes are verified unchanged. No trained stage-2 weights are transferred into the production run. The authoritative pass record and snapshot hashes are written to `outputs/church-relative-noise-20260911/verification.json` before launch.

Files: `src/church_relative_noise.py`, `scripts/calibrate_church_relative_noise.py`, `scripts/train_church_relative_noise.py`, `scripts/finalize_church_relative_noise.py`, `scripts/launch_church_relative_noise.py`, `tests/test_church_relative_noise.py`.
