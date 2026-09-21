# Lloyd–Max compound distortion diagnosis and original RQ control

The Lloyd–Max compound samples still contain warped towers, bent rooflines,
and inconsistent facade details. The original RQ control supplied by the user
has more coherent building outlines in the reviewed best and latest grids,
although it also has structural errors and learned stock-photo watermarks.
This is a qualitative comparison, not a measured geometric-quality score.

Runs:

- [Lloyd–Max compound](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-laser-ft3best-compound-lloydmax90-h200x8-20260921)
- [Original RQ control](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-original-rqvae-released-tokenizer-control-20260917)

All new model probes ran on CPU in FP32, leaving the eight training GPUs
available. Training weights, losses, sampling settings, and the authorized
90-epoch budget were not changed. Artifacts and executable probes are under
`outputs/church-compound-lloydmax-distortion-20260921/`.

## What the current checkpoint checks establish

The frozen compound epoch-50 checkpoint has SHA-256
`d482e6780b51f3479f53222b6ac9799c8549e05080d914da95ad4efd47894477`.
Its nonuniform coefficient centers and depth scales match the training cache
exactly, and coefficient clipping is disabled.

Incremental cached inference agrees with parallel teacher-forced inference
on all 256 positions of one cached image. Maximum absolute logit errors were
4.53e-5 for atoms and 2.00e-5 for coefficients, within the 1e-4 tolerances.
This checks FP32 inference consistency, not every possible bug or GPU
mixed-precision discrepancy.

Recomputing OMP from FP32 latents gave identical atom IDs at every depth on
32 sampled training images and 32 validation images. The difference between
recomputed and cached sparse reconstructions was below 4.1e-14 MSE. Thus the
train/validation prediction comparison below is not explained by different
OMP ordering, atom indexing, or CPU-versus-GPU support selection on this probe.

## Coefficient isolation

Sixteen fixed validation images were decoded with true atom supports while
changing only coefficients. Pixel errors below use images scaled to [0,1].
The model coefficient predictions use the true current atom and true earlier
tokens, with training coefficient noise. They are an optimistic conditional
probe, not a complete free-running generation test.

| Coefficients | Pixel MSE versus original | Added latent MSE versus native sparse codes |
|---|---:|---:|
| Native continuous coefficients | 0.015432 | 0 |
| Nearest Lloyd–Max bins | 0.015432 | 0.000000079 |
| Training target noise, tau=0.125 | 0.015519 | 0.000954 |
| Model conditional mean | 0.023609 | 0.069542 |
| Model conditional sample | 0.028072 | 0.108674 |

`coefficient-isolation.png` shows, from top to bottom: originals, native
coefficients, Lloyd–Max rounding, training noise, model conditional means,
and model conditional samples. The latter two rows show substantially greater
distortion even though the atoms remain correct. This identifies coefficient
prediction as a contributor, without establishing that it is the only cause
or that a specific auxiliary loss would solve it. A conditional mean can also
average distinct valid modes, so it is not automatically a better sampler.

The 16-image sampling previews compare the baseline with coefficient
temperature 0.7 and with both temperatures 0.9, using the same random seed.
All three retain structural errors. No new FID was computed for these previews,
and no winning sampling setting was declared.

## Held-out prediction

The frozen epoch-50 model was evaluated on all 300 official validation images
and 300 uniformly sampled training images. Both use the same evaluation mode,
physical coefficient target temperature 0.125, and stochastic true contexts.

| Metric | Training, 300 images | Validation, 300 images |
|---|---:|---:|
| Atom NLL | 3.5086 | 11.1264 |
| Atom top-1 accuracy | 45.49% | 1.41% |
| Coefficient soft CE | 6.2783 | 6.6579 |
| Coefficient KL above target entropy | 0.8819 | 1.2635 |
| Physical coefficient MAE | 0.5969 | 1.2766 |

There is a large generalization gap. It does not by itself prove memorization
or explain why the compound samples look worse than the RQ control. In
particular, the control also logs a large gap: near epoch 245, validation soft
CE is 10.4387 and the latest training epoch soft CE is 3.2799. The two models'
absolute losses should not be compared as if their targets were identical.

## Comparison with the supplied original RQ control

The online run is more recent than its local copy, which stops near epoch 32.
The online state records a checkpointed pause at epoch 247.016, step 15313,
despite the W&B run being marked finished. Its best retained checkpoint is
epoch 80, step 4960, FID 10.073105; epoch 40 reached 10.099374. Its latest
FID measurement is 11.378375 at epoch 240. The original 300-epoch plan should
not be described as a completed 300-epoch run.

| Property | Original RQ control | Lloyd–Max compound |
|---|---|---|
| Best recorded FID50k | 10.0731, epoch 80 | 11.0899, epoch 50 |
| Recorded FID50k at epoch 80 | 10.0731 | 11.2809 |
| Transformer parameters | 370,087,936 | 404,738,048 |
| Representation | Residual vector-code indices | OMP atom/coefficient pairs |
| Training targets | Distance-based soft targets over residual code vectors | Hard atom targets, soft scalar coefficient targets |
| Stochastic training contexts | Residual code choices regenerated per visit | Fixed atom supports, randomized coefficients |
| Target temperature | 0.5 in vector-distance units | 0.125 in physical coefficient-distance units |
| Objective | Mean soft code CE | (1.5 × atom CE + coefficient soft CE) / 2.5 |
| Peak LR / optimizer | 5e-4, AdamW (0.9,0.95), weight decay 1e-4 | Same |
| Cosine budget | 300 planned epochs | 90 planned epochs |
| Global batch | 2048 | 1536 |
| Sampling | T=1, k=1400, p=1 | Atom T=1/k=250/p=1; coefficient T=1/p=1 |

Both record 50,000 generated images against all 126,227 training images.
However, the original control resumed elsewhere with rebuilt real statistics:
its current reference hash is `31d7e694a5298cf994030c307a2dea6e30cc8b12b19573a854c5082ba7f13b6c`,
whereas the compound run uses
`ad3b5a341831d877110afdff37d6fff4be855612053e3eb9e2e7119f5593d364`.
These are recorded-run comparisons, not a newly matched FID reevaluation.

The original control already reached its best by epoch 80. The evidence does
not support explaining the current visual gap simply by saying it needs 300
epochs. Likewise, improved reconstruction FID or scalar bin precision does not
establish that a representation is easier for the autoregressive prior to model.

The next controlled training hypothesis should address how the compound model
learns atom/support and coefficient distributions, especially the absence of
RQ-style stochastic alternatives for atom targets. Any stochastic-support
adaptation must refit matching coefficients and verify reconstruction quality;
randomly changing atom IDs would be invalid. This hypothesis has not been
tested here. No new training run or geometry-loss change was launched.
