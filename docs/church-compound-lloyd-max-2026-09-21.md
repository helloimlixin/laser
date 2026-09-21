The new compound cache uses unclipped FP32 coefficients and 2,048 Lloyd–Max bins. It substantially reduces the small amount of error added by coefficient storage and quantization; it has little effect on the much larger four-atom sparse approximation error. The existing integer stage-2 training run was not changed or restarted.

Cache: [compound-cache-fp32-lloyd-max.pt](../outputs/church-compound-cache-accuracy-20260921/compound-cache-fp32-lloyd-max.pt), 194,159,593 bytes for all 126,227 training images. The previous cache was 129,511,995 bytes. Each image still has 8×8×4 atom/coefficient pairs. Atoms remain int16; normalized continuous coefficients are now FP32. The nonuniform centers are stored in `meta.coeff_bin_centers`; coefficient IDs are derived from those centers.

Stage 1 is the selected three-epoch full-model finetune, checkpoint SHA-256 `762c51a10267ed6fa55709ff0d6cf997940d21056b16c7a0a0399b3ebf93868d`. The original compound supports and the prebuilt FP32 encoder latents have matching checkpoint and data-protocol provenance. The existing four-atom supports were preserved; their coefficients were refitted using FP64 least squares and saved in FP32 without the previous ±3 normalized clipping. The largest recovered normalized coefficient magnitude is 4.764464. This changes neither the encoder nor the dictionary.

Lloyd–Max alternates nearest-center assignment and empirical centroid updates. A shared normalized codebook is compatible with the current compound codec, which multiplies its centers by four existing depth scales. Two 2,048-center versions were fitted: normalized coefficient MSE and physical coefficient MSE with depth-scale-squared weights. Both used 122,131 training images, excluding the first 4,096 images and all validation images. The normalized version was selected using added latent MSE on those 4,096 held-out training images. The independent validation split was not used for selection. The fit ran for 250 iterations; this is an empirical local solution, not a claim of global optimality.

All 300 validation images, with identical encoder latents and supports across the comparison:

| Representation | Total latent MSE against encoder output | Added latent MSE against unclipped FP32 sparse reconstruction |
| --- | ---: | ---: |
| Unclipped FP32, before coefficient binning | 0.0095384651 | 0 |
| Previous clipped FP16 cache + uniform 2,048 bins | 0.0095463058 | 7.84070e-6 |
| Unclipped FP32 + uniform 2,048 bins over the fitted full range | 0.0095387471 | 2.81980e-7 |
| Unclipped FP32 + selected Lloyd–Max 2,048 bins | 0.0095385459 | 8.07727e-8 |
| Unclipped FP32 + physical-weighted Lloyd–Max alternative | 0.0095385364 | 7.12377e-8 |

The selected cache reduces added latent error by 98.97% against the previous representation. Removing clipping accounts for most of that gain; Lloyd–Max supplies a further 71.35% reduction against unclipped uniform bins. Total latent MSE improves by only 0.0813%. The physical-weighted variant scored slightly better on validation but slightly worse on the held-out training subset; the selection was retained to keep validation independent.

Paired decoder evaluation on the first 64 validation images, CPU FP32, pixels clamped to [−1,1], pixel MSE measured in [0,1]:

| Representation | Pixel MSE | Mean PSNR (dB) | Mean LPIPS |
| --- | ---: | ---: | ---: |
| Unclipped FP32 sparse reconstruction | 0.0134922711 | 19.491343 | 0.24119928 |
| Previous clipped FP16 + uniform bins | 0.0134997092 | 19.486811 | 0.24126024 |
| New unclipped FP32 + Lloyd–Max | 0.0134922916 | 19.491341 | 0.24119952 |

This is a small reconstruction improvement, not evidence of a generated FID improvement. No new stage-2 model or generated FID evaluation was run. The [reconstruction grid](../outputs/church-compound-cache-accuracy-20260921/paired-reconstructions.png) has rows: original images, unclipped FP32 sparse reconstructions, previous uniform-bin reconstructions, and Lloyd–Max reconstructions.

The existing `SparseTokenCacheDataset` loaded the artifact with FP32 coefficients. The existing `LaserAux` accepted the centers and passed hard nearest-bin agreement, token packing/unpacking, and exact physical embedding checks on eight images. Soft targets from the existing discrete RQ rule `exp(-physical_distance² / temperature)` are finite and normalized. A new stage-2 run should retain that rule for a controlled recipe comparison and recalibrate temperature/target entropy for the changed codebook. Integrating Gaussian mass over nonuniform cells would be a separate change of objective; it has not been applied.

The cache must be paired with its saved centers and depth scales. Use metadata `coeff_max` (4.764464378356934) and disable coefficient clipping when freshly encoding. Existing compound stage-2 checkpoints were trained with different bin meanings; replacing their centers at sampling time would be invalid. A fresh stage-2 run is required to evaluate generation with this cache. The currently running integer model uses a different cache and remains unchanged.

The CPU cache build and latent evaluations took 43.0 seconds; the 64-image decoder check took 63.7 seconds. No training GPU was used. Validation preprocessing/order was independently checked by re-encoding the first four images on CPU: MSE against the cached GPU FP32 encoder latents was 5.22e-13.

Reproducible artifacts:

- [Builder](../outputs/church-compound-cache-accuracy-20260921/build_lloyd_cache.py), [decoder evaluation](../outputs/church-compound-cache-accuracy-20260921/evaluate_decoder.py), and [codec verification](../outputs/church-compound-cache-accuracy-20260921/verify_codec.py).
- [Completion receipt and latent metrics](../outputs/church-compound-cache-accuracy-20260921/complete.json), [per-image decoder metrics](../outputs/church-compound-cache-accuracy-20260921/decoder-evaluation.json), and [codec verification results](../outputs/church-compound-cache-accuracy-20260921/codec-verification.json).
- Cache SHA-256: `38e3ee5e17fc6b6b8115ae97e5fdd1ed874935421f6a23b6b155bfb28a21291e`.
