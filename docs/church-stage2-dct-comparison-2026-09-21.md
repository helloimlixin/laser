The new matched probe gives strong evidence that stage 2 is overfitting the
training images. This is more informative than the unsuccessful six-epoch
coefficient and support interventions. It does not establish that stage 1 is
optimal or that a particular regularization change will lower generated FID.

I evaluated the preserved epoch-60 best model and epoch-71 LR-only continuation
on 300 randomly selected training images and all 300 official validation images.
Both splits use fresh stochastic OMP at temperature 0.0625 and physical
coefficient targets at temperature 0.125. The same sampled supports and
coefficient histories are reused across checkpoints. Both checkpoints run in
evaluation mode, with BF16 forward passes and FP32 losses. Losses are in nats
per predicted atom, averaged over sites and depths.

| Checkpoint | Train atom NLL | Held-out atom NLL | Train coefficient KL | Held-out coefficient KL |
|---|---:|---:|---:|---:|
| Epoch 60, original best | 3.9371 | 11.1714 | 0.8460 | 1.2844 |
| Epoch 71, LR-only continuation | 3.7594 | 11.3860 | 0.8295 | 1.3056 |

The paired held-out atom-loss change is **+0.21466 ± 0.00343 standard error**;
training loss changes by -0.17775. Coefficient KL also improves on training
images while worsening on held-out images. Standard errors treat images as
the independent units; this is one fixed stochastic support draw per image,
not a multi-seed training comparison. Differences between the official splits
can contribute to the absolute gap, but cannot explain away the deterioration
on exactly the same held-out examples after continuation.

At epoch 60, fresh-support training atom losses by depth are
0.5285 / 2.6322 / 5.0097 / 7.5781; held-out losses are
12.6177 / 11.7731 / 10.5280 / 9.7666. The failure includes the earliest,
highest-energy pairs. Merely improving later residual predictions is unlikely
to be a sufficient response.

The cached training bank gives atom NLL 3.8785 versus 3.9371 with fresh
supports on the same training images. The much larger unseen-image gap
persists after removing reuse of the 16 cached support variants. Fresh OMP
adds code variation but does not add new image content or image augmentations.

I checked the validation cache independently against five original LMDB
images, the recorded pixel hashes, and the actual frozen encoder. All pixel
hashes match; maximum latent discrepancy is 1.073e-6, with MSE 2.688e-14.
The tokenizer SHA256 matches the training-cache provenance. This rules out
those preprocessing, ordering, and encoder-identity errors on the probes.

The original training history points in the same direction: logged training
atom NLL falls from 4.9108 at epoch 60 to 4.5020 at epoch 80 while FID rises
from 10.6885 to 11.1958. Those training-mode losses include dropout and should
not be compared numerically with the evaluation-mode losses above. The active
loop has no held-out token-loss pass; its `val_loader`, when constructed,
supplies real images for FID rather than validation of the autoregressive model.

DCTransformer reports **LSUN Churches FID 7.56**, using 50,000 generated
images against the entire training set. It models sparse DCT triples with
separate channel, position, and value decoders, and orders content from low
to high frequency. This is a directly compressed image representation rather
than a learned LASER tokenizer. See the [paper, sections 2–4 and Table 1](https://proceedings.mlr.press/v139/nash21a/nash21a.pdf).

The published LSUN settings and the actual frozen LASER runtime compare as
follows. The DCT column comes from [Appendices B–C and Table 3](https://proceedings.mlr.press/v139/nash21a/nash21a-supp.pdf).

| Setting | DCTransformer LSUN | This LASER run |
|---|---|---|
| Model parameters | 448M | 404.738M, measured |
| Image geometry | Variable aspect ratio, long side 384 | Center crop 256×256 |
| Representation | 8×8 DCT blocks, quality 75 | 8×8 latent sites, four atom/coefficient pairs |
| Batch | 512 chunks | 960 images |
| Peak LR | 5e-4 | 5e-4 originally |
| Warmup | 1,000 steps | None originally |
| Schedule | Cosine, 300B target elements | Cosine, 11,790 updates over 90 epochs originally |
| Dropout | 0.1 | Residual 0.1; attention and embedding 0 |
| Training allocation | Favors early, low-frequency chunks | Equal weighting across sparse depths |

Their chunk selection uses a cubic-decay preference with a floor of 0.1.
Our sequence completes each site's four depths before moving spatially;
OMP depth is not a DCT frequency band. Transferring their bias requires an
experiment, not copying the schedule mechanically.

Our epoch-60 run has 7,860 updates and 1.932B compound-pair events. DCT's
token budget implies approximately 654K updates if all 512 target chunks
contain 896 elements. These event units, sequence lengths, and compute costs
differ substantially. In light of the measured generalization deterioration,
the large budget difference is not evidence that simply training this model
longer will help. The preprocessing and evaluator differences also prevent
treating 7.56 as a directly matched score against our 10.6885.

The next priorities should be:

1. **Track held-out atom NLL and coefficient KL per depth alongside FID.**
   Keep FID50k as the requested generation-selection metric. Use validation
   to distinguish improved fitting from worsening generalization and to locate
   the onset of overfitting in an earlier checkpoint or a new run.
2. **Test image augmentation and stronger stage-2 regularization separately.**
   The current cache fixes each image's crop and orientation. A horizontal-flip
   intervention should flip original pixels and re-encode them with the frozen
   tokenizer; flipping the latent grid alone does not reproduce that operation.
   Residual dropout is already 0.1, so adding that nominal rate is not a fix.
3. **Test a normalized preference for earlier sparse depths.**
   A modest starting candidate is weights [2, 1, 0.5, 0.5], applied consistently
   to the atom and coefficient objectives. This is a proposed LASER experiment,
   not DCTransformer's exact objective. Depth-1 held-out loss must be monitored:
   increasing its weight alone could intensify memorization.
4. **For a fresh stage-2 control, include warmup and compare update budgets.**
   An epoch-60 continuation cannot test the effect of initialization-time
   warmup. Compare batch size and model capacity in separate experiments,
   keeping representation, preprocessing, and evaluation fixed.

The investigation did not change the running training policy. All sparse-code
pilots have now completed without improving the original best; the authorized
selected-policy continuation is running. No lower FID is claimed here.

Reproducible evidence is in
[outputs/church-stage2-dct-audit-20260921](../outputs/church-stage2-dct-audit-20260921/):
`measure_generalization.py`, `fixed-probe.pt`, `per-image-losses.pt`,
`generalization.json`, `verify_validation_cache.py`,
`validation-cache-verification.json`, and `training-history.json`.
