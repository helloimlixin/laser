# Church complete-site integer: plateau diagnosis and larger-bit probe

The learned 16,384-entry complete-site codec stopped at step 4,500 with validation reconstruction LPIPS **0.32125**, compared with **0.35220** initially. Most improvement occurred in the first 250 steps; the remaining 4,250 steps gained only 0.00589. Relative latent MSE worsened from 0.36746 to 0.41779. This configuration has not met the compression quality criteria, and its completed checkpoint is preserved. [Detailed diagnosis](lsun-church-learned-site-codec-2026-09-11.md).

The 14-bit vocabulary was our implementation choice. The user's requirement was one integer determining the **complete** sparse code, including all four atom IDs and all four signed coefficients. Increasing integer width is compatible with that requirement, but packing several fields into one number does not make their prediction a single manageable categorical decision.

## Bounded follow-up

We fitted nested residual codebooks on the same 262,144 training-only sites used in the earlier direct-codebook experiment. The 128 calibration images remain excluded. Each stage has 512 entries: 511 Lloyd centers fitted for eight iterations, plus a zero-residual entry. Each additional stage consumes nine bits. Three, five, and seven stages pack into one nonnegative 27-, 45-, or 63-bit integer per latent site. Decoding unpacks that integer, sums its selected codewords, then applies the frozen four-atom OMP and signed-bin projection. Source atom IDs and coefficient values are not available to this decoder.

A second 63-bit construction packs four exact 14-bit atom IDs plus one seven-bit coefficient-pattern ID. Its 128 signed physical coefficient patterns are fitted under the selected support's Gram metric, then quantized to the existing signed bins. The packed integer alone recovers both the support and all coefficient-bin IDs. Preserving support uses 56 of the 63 bits and leaves relatively little capacity for coefficients. The implementation verifies this round trip on every evaluated site.

No neural weights were trained. Frozen stage-1 and LPIPS parameter/buffer versions were checked after the probe. All reconstructions use the existing one-epoch Church finetune from the ImageNet rFID 4.2109 checkpoint. The complete probe took 166 seconds on GPU 0, alongside the existing prior job.

## Results

All numbers below use the same 300 validation images and compare against continuous frozen-LASER reconstructions. These are compression measurements, not generation FID.

| Complete-site construction | Nominal bits/site | LPIPS ↓ | PSNR dB ↑ | Relative latent MSE ↓ |
| --- | ---: | ---: | ---: | ---: |
| Learned single codeword, selected step 4,500 | 14 | 0.32125 | 16.819 | 0.41779 |
| Three residual fields, projected to sparse code | 27 | 0.24617 | 18.659 | 0.19983 |
| Five residual fields, projected to sparse code | 45 | 0.14505 | 21.115 | 0.09429 |
| Seven residual fields, projected to sparse code | 63 | 0.08905 | 23.525 | 0.04865 |
| Exact support + coefficient pattern | 63 | **0.03568** | **28.327** | **0.01480** |
| Original support + nearest scalar coefficient bins | 100 | 0.00000315 | 70.490 | 0.00000073 |

Bit counts exclude shared codebook/dictionary storage. The 100-bit row is the original ordered four-pair fixed-width representation, not a minimum entropy bound. The 63-bit rows fit in a signed int64; larger Python integers remain possible and were already verified for exact 100-bit packing.

The nested residual construction improves with additional bits. At the same 63-bit budget, preserving the sparse support and compressing the coefficient tuple performs substantially better. Its holdout LPIPS is 0.03568 and calibration LPIPS is 0.03459, so the improvement is not confined to validation. The learned 14-bit comparison changes both architecture and bit budget; it is not a controlled estimate of the effect of bit count alone.

All four new constructions still fail the predeclared calibration criteria: LPIPS mean plus two standard errors ≤ 0.01 **and** relative latent MSE ≤ 0.005. For the best new construction those values are 0.03597 and 0.01474. We have not relaxed the criteria or launched a new autoregressive prior on these failed candidates.

The next justified direction is a structured complete code that retains explicit support and allocates more capacity to the joint coefficient pattern. It can still be stored as one integer. A predictor must model the internal fields, and better reconstruction alone would not establish better unconditional generation. Extending the plateaued 14-bit run or swapping in a looped transformer has no supporting result from these probes.

The authorized follow-up now passes the unchanged reconstruction gate with 2,048 coefficient patterns and a 67-bit complete-site integer. Validation LPIPS is 0.00912 and relative latent MSE is 0.00295. [Calibration, fresh-view checks, and stage-2 candidate](lsun-church-support-pattern-integer-2026-09-11.md).

## Verification and artifacts

Nine focused tests pass across the new integer codec and the existing complete-site and learned-site codecs. They cover the largest positive signed-int64 value, mixed field widths, invalid values, exact unpacking, residual decoding followed by complete sparse reconstruction, frozen decoder gradients, and checkpoint continuation. GPU evaluation additionally checks all packed field round trips, reconstruction from recovered sparse codes, disjoint splits, source checkpoint hashes, and frozen weight versions.

- [Bit/distortion chart](../outputs/church-residual-site-integer-20260911/bit-distortion.png), [PDF](../outputs/church-residual-site-integer-20260911/bit-distortion.pdf).
- [Results and configuration](../outputs/church-residual-site-integer-20260911/results.json).
- [Validation reconstructions](../outputs/church-residual-site-integer-20260911/validation-reconstructions.png): continuous reference, 27-bit residual, 45-bit residual, 63-bit residual, 63-bit exact-support/pattern.
- `outputs/church-residual-site-integer-20260911/codebooks.pt`: fitted residual tables, coefficient-pattern bin IDs, bin/scales metadata, fitting and calibration indices.
- Per-image metrics, packed site IDs, decoded atom/bin grids, source snapshots, and fitting history are saved in the same directory.
- Implementation: `src/residual_site_integer_codec.py` and `scripts/probe_church_residual_site_integer.py`.

Reproduce with `/tmp/laser-sign-venv/bin/python -u scripts/probe_church_residual_site_integer.py --output <new-directory>`, setting `CUDA_VISIBLE_DEVICES=0`, `OMP_NUM_THREADS=8`, `OPENBLAS_NUM_THREADS=8`, `MKL_NUM_THREADS=8`, `TORCH_HOME=/workspace/tmp/official-rqvae-eval-cache`, and `LASER_VGG16_WEIGHTS=/workspace/tmp/laser-vgg/vgg16-397923af.pth`.
