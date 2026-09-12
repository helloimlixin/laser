# One integer for a complete Church sparse latent site

The prototype produces an **8×8 integer grid**. Each ID selects an entire codeword containing **four dictionary atom IDs and four signed coefficient-bin IDs**. The decoder needs only that grid, the shared codebook, and the frozen LASER checkpoint. It does not receive the source image's atom support.

The lookup works, but the tested finite vocabularies discard substantial image detail. None passed the reconstruction criterion set before fitting. These are oracle compression results: every source latent is assigned its nearest available codeword, with no autoregressive prediction errors.

## Reconstruction results

All measurements below use the same 300 official validation images. PSNR is higher-is-better and LPIPS is lower-is-better, relative to each image's **continuous frozen-LASER reconstruction**, rather than the original photograph.

| Representation | Logical bits/site | PSNR (dB) | LPIPS | Relative latent MSE |
|---|---:|---:|---:|---:|
| Existing four atom/coefficient pairs, nearest coefficient bins | 100 | 69.109 | 0.0000052 | 0.00000073 |
| One ID, 4,096 complete sparse codewords | 12 | 16.558 | 0.38056 | 0.40398 |
| One ID, 16,384 complete sparse codewords | 14 | 16.891 | 0.35116 | 0.36594 |
| One ID, 65,536 complete sparse codewords | 16 | 17.272 | 0.32175 | 0.32233 |

Logical bit counts describe fixed-width token payloads and exclude the shared codebook. The prototype artifact stores site IDs as int32 for convenience; these are not measured file-compression ratios.

The independent 256-image holdout agrees: LPIPS is 0.37839, 0.34884, and 0.32006 for the three vocabularies. At 65,536 entries, the training-calibration LPIPS upper bound is 0.32607 and relative latent MSE is 0.32551, exceeding the predeclared limits of 0.01 and 0.005.

[Full comparison data](../outputs/church-complete-site-token-20260911/comparison.json)

The following grid uses the first eight validation images without sample selection. Columns are: continuous LASER reconstruction; existing scalar bins; complete sparse codeword; unconstrained dense cluster center used as a diagnostic.

![65,536-codeword reconstruction comparison](../outputs/church-complete-site-token-20260911-large/vocab-65536/validation-reconstructions.png)

## Construction and controls

1. Reuse the Church tokenizer already finetuned for one epoch from the ImageNet rFID 4.2109 checkpoint. Encoder, dictionary, and decoder remain frozen; no prior is trained in this experiment. Checkpoint SHA-256: `93f19adc8ca2cfae0cf629d48d2d6f5f7039440b1a48ad1c7ba43cf17932a388`.
2. Sample 262,144 sites from the training-only continuous cache. Reserve 128 whole training images for calibration, excluding all their sites from fitting. The separate 256-image holdout and all 300 validation images also remain excluded.
3. Quantize fitting coefficients to the existing signed bins. Reconstruct each complete sparse code's 256-dimensional latent vector. Fit cluster centers with eight Lloyd iterations in this physical latent space, with seed 9701.
4. Replace each cluster center with its nearest assigned observed training code. This retains a complete, valid four-pair sparse representation in every codeword. It is a cluster-center projection experiment, rather than globally optimized k-medoids.
5. Assign each evaluation latent to the nearest final sparse codeword, retrieve both its atoms and coefficients using the single ID, and decode the resulting latent grid. All vocabulary sizes use exactly the same fitting sites and evaluation images.

The dense-center diagnostic also degrades reconstruction: validation LPIPS is 0.37402, 0.34432, and 0.31748. Thus projecting centers back to sparse representatives accounts for only part of the observed loss in this experiment. This is not a bound on what an end-to-end learned tokenizer could achieve.

The 65,536-entry extension was added after the smaller vocabularies failed, to check the effect of increasing vocabulary size. Selection still uses the same reserved training-calibration images and unchanged acceptance thresholds.

## Exact packing versus a prediction vocabulary

An individual pair has 16,384 atom choices and 2,048 coefficient choices, or 25 fixed-layout bits. Four ordered pairs take 100 bits. The exact packing helper uses a Python integer to preserve all of these bits, including the coefficient signs encoded by the bin IDs. It has a nominal integer range of `0` through `2**100 - 1`; it preserves already-quantized values, not the original continuous coefficients.

That representation is reversible, but its approximately 1.27×10³⁰ nominal states cannot serve as an ordinary flat categorical prediction vocabulary. The finite codebooks tested above reduce the output vocabulary by deliberately approximating complete sparse codes. A codebook ID is a categorical label; numerical closeness between two IDs has no defined geometric meaning.

## Files and verification

- Codec: `src/complete_sparse_codec.py`.
- Reproducible fitting and evaluation: `scripts/probe_church_complete_site_tokens.py`.
- Six tests passed, covering 100-bit packing, support-and-coefficient lookup, signed reconstruction, geometric assignment, and valid observed sparse prototypes. Exact packing was additionally checked on 1,024 real sites.
- Every evaluation grid was checked to have 64 tokens, and reconstructing through the recovered sparse components matched the stored codeword latent vectors.
- [Verification and source hashes](../outputs/church-complete-site-token-20260911/verification.json).
- Codebooks and evaluated token grids are in `outputs/church-complete-site-token-20260911/vocab-{4096,16384}` and `outputs/church-complete-site-token-20260911-large/vocab-65536`.

The measured result supports keeping this as a compression diagnostic rather than replacing the current generator's representation. A learned compression model optimized through reconstruction would be a different experiment. These findings do not establish that one-token-per-site generation is impossible, and no unconditional FID claim is made.
