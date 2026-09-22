The current scalar discretization preserves Church reconstructions very closely. Actual stochastic tokenization causes a small perceptual degradation. These measurements do not make full encoder/dictionary/decoder retraining the first priority.

All conditions use the same frozen selected tokenizer, raw coefficient units, 2,048 physical coefficient centers, and no explicit coefficient clipping. The exact tokenization and decoding helpers are imported from the active run’s frozen runtime.

| Reconstruction path | Matched train rFID4096 ↓ | Validation LPIPS ↓ | Validation PSNR ↑ |
| --- | ---: | ---: | ---: |
| Native greedy OMP, continuous coefficients | 6.064011 | 0.24728790 | 19.109122 |
| Same greedy supports, nearest coefficient bins | 6.063920 | 0.24728936 | 19.109110 |
| Stage-2 stochastic supports, continuous coefficients | 6.066050 | 0.24765399 | 19.102555 |
| Same stochastic supports, nearest coefficient bins | 6.066231 | 0.24765512 | 19.102566 |
| Exact stage-2 sampled tokens, coefficient temperature 0.125 | 6.084885 | 0.24892064 | 19.053926 |

A second coefficient draw on the same validation supports gives LPIPS 0.24910144 and PSNR 19.062316. It confirms the direction of the small soft-token effect.

The per-image tail check also remains small on this sample: the largest validation LPIPS increase from the entire stochastic token stream relative to native greedy reconstruction is 0.008316, with 95th percentile 0.004871. No image in this 300-image draw increases by more than 0.01. These are measured perceptual-score differences, not a guarantee about every possible sample.

Nearest-bin rounding changes the decoded [0,1] image by mean squared error 6.8801837e-08; the sampled coefficient distribution changes it by 0.00038476083 relative to continuous decoding on exactly the same support. That large ratio reflects the extremely small rounding error; it does not mean comparably large degradation against the original image.

The coefficient softmax has physical coefficient variance about 0.06 per depth, consistent with standard deviation approximately 0.25. Its measured coefficient MSE is 0.061986, versus 1.1207476e-05 for nearest bins. The noise distribution, not the integer ID representation itself, is the meaningful codec mismatch.

The 4,096 reconstruction FID images are the first 4,096 unique training images in LMDB key order, compared against the existing statistics from exactly those originals. The coefficient centers were fitted on training indices 4096:126227, excluding this subset. Perceptual and pixel metrics use all 300 official validation images. RGB conversion, short-side bilinear resize to 256, center crop to 256, and normalization match the stage-2 cache input. This diagnostic subset does not change the active training population or its FID50k reference: those still use all 126,227 training images.

FID uses the released RQ-VAE Inception implementation, FP32 features/means, NumPy covariance, and released Frechet-distance code with only the previously audited SciPy API bridge. The decoder and encoder are FP32 with TF32 disabled. No PNG/uint8 round trip is used for metrics. These are reconstruction FID4096 scores, not generated FID50k and not full-data reconstruction FID; small score differences should be interpreted only within this matched probe.

The native baseline reproduces the earlier rFID4096 6.06401077 with difference -1.5488581e-07. Both train and validation encoder checks matched cached latents to the recorded tolerance, pixel hashes matched, and direct compound-token decoding agreed exactly with the common reconstruction path. All stochastic conditions share the same complete four-pair support trajectory at every spatial site. Seeds, selected variants, per-image results, source hashes and FP32 Inception features are retained locally.

The probe uses one support-bank draw per image and two coefficient draws only on validation. Paired standard errors in the JSON quantify across-image variation conditional on these draws; they do not establish uncertainty over all possible support or generation seeds. Reconstruction quality cannot establish how easy the tokens are for an autoregressive prior to model.

Independent stage-2 monitoring is a stronger warning: from epoch 19 to epoch 61, training atom NLL changes from 8.0785 to 4.5078, while validation atom NLL changes from 8.7591 to 10.6854. This is consistent with substantial prior overfitting. It is not proof that one particular regularizer will improve generated FID.

Recommended order: prioritize controlled stage-2 generalization experiments; if investigating codec mismatch, test a decoder-only adaptation on the exact sampled tokens against the frozen-decoder control. That preserves the existing encoder, dictionary, bins and stage-2 prior. A lower coefficient-target temperature is a separate training ablation, not just a generation-temperature adjustment. Full quantization-aware stage-1 fine-tuning remains a possible representation experiment, but ordinary scalar rounding is not currently a measured bottleneck. Changing encoder, dictionary or coefficient centers requires cache regeneration and a fresh prior for a clean comparison.

The earlier FFHQ and ImageNet K=2 successful runs also froze the codec and introduced scalar bins in stage 2. Their historical source/configuration audit is in [the companion report](church-tokenized-bottleneck-audit-2026-09-22.md). No tokenizer or training configuration was changed by this diagnostic. The scratch Church run continues.

Artifacts: `outputs/church-tokenized-bottleneck-audit-20260922/reconstruction`. [Unselected first-eight validation comparison](../outputs/church-tokenized-bottleneck-audit-20260922/reconstruction/val-paired-reconstructions.png).
