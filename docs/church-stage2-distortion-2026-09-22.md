# Church stage-2 distortion audit

The evidence supports prioritizing stage-2 learning and generalization. Sampling
changes produce a modest FID gain, while overfitting is already visible in the
new 32769-token integer run. No training parameters were changed by this audit.

## What was measured

The fixed compound epoch-57 sampling sweep completed 18 FID50k evaluations.
Coefficient temperature 1.20, with atom temperature 1, atom top-k 250 and both
top-p values 1, improved the two-seed mean from 9.933752 to 9.827983 (1.065%).
Both seeds improved. All 900,000 images used the released RQ-VAE evaluator and
the same 126,227-image training reference. This does not show that structural
distortions have been resolved.

The current integer run's validation KL reached its minimum of 7.630978 at
epoch 17, then rose to 8.457585 by epoch 32. Over the same interval, training KL
fell from 7.009464 to 5.183411. The same fixed 300-image training and 300-image
validation probes, target temperature and sampled histories are used each
epoch. Validation images are only for monitoring; all 126,227 training images
remain in training and the FID reference.

A separate 128-image paired probe used the frozen integer run's best checkpoint
(epoch 22, FID50k 12.352177), tokenizer and raw coefficient codebook. It compared
native continuous-coefficient OMP, hard integer codes, and stochastic integer
codes through the same decoder. Pixel MSE is measured in [0,1].

| Representation | Encoder-latent MSE | Pixel MSE | Mean target entropy, nats |
|---|---:|---:|---:|
| Native OMP | 0.009569 | 0.013135 | — |
| Hard integer codes | 0.017798 | 0.013984 | — |
| Integer targets, τ=0.125 | 0.018027 | 0.014017 | 0.320089 |
| Integer targets, τ=0.25 | 0.018911 | 0.014035 | 0.744251 |
| Integer targets, τ=0.50 | 0.022904 | 0.014412 | 1.726144 |

Moving τ from 0.125 to 0.25 increases mean target entropy about 2.33-fold while
raising pixel MSE about 0.13% on this probe. At the first residual depth,
τ=0.125 assigns an average 97.2% of the probability to one token. The calibration
previously chose τ=0.125 using a bound on the *relative latent reconstruction
error* versus a different tokenizer's reference; it did not optimize stage-2
generalization or match the original RQ model's target entropy. The original
calibration control had mean entropy about 1.015 nats across depths.

This is evidence for a regularization experiment, not evidence that τ=0.25
already improves generated FID. The paired image probe is small and pixel MSE
does not measure every perceptual distortion. Neither its size nor its metric
is interchangeable with FID50k.

## Visual and implementation checks

The eight matched examples in `representation-grid.png` retain their overall
building layout through native and integer reconstruction, with detail loss
and local deformation visible in both. Its rows are: original, native OMP,
hard integer, τ=0.125 integer, τ=0.25 integer, τ=0.50 integer.

`rollout-grid.png` shows original, hard-integer reconstruction, independently
sampled predictions given true past tokens, generation of the final 32 spatial
sites given the true first 32, and unconditional generation. Larger structural
changes appear in the learned-prior rows. The independently sampled row is a
diagnostic of predictions under true histories, not a coherent autoregressive
sample. Unconditional images are unpaired and cannot be scored for paired
reconstruction accuracy. Eight examples do not establish a distortion rate.

On the actual trained model, all 512 tested cached versus full-forward events
agreed in FP32 to maximum absolute logit error 2.575e-5, with no argmax
disagreements. BF16 introduced mean distribution KL 1.642e-4 nats and 3.71%
argmax disagreements, consistent with sensitivity near tied logits; the small
KL is far below the observed model prediction loss. This check found no gross
cache/conditioning mismatch. It is not a universal equivalence proof.

## Prioritized next experiments

1. Compare training-target τ=0.25 with τ=0.125 from the same preserved integer
   checkpoint, changing only target stochasticity. Keep the 32769 vocabulary,
   raw fitted coefficient values, decoder, full training set, transforms,
   optimizer state and sampling protocol fixed. Recompute targets each visit.
   Evaluate both with the same fixed monitoring target distribution, FID50k
   every epoch, and fixed visual previews. Comparing KL across different target
   distributions without a common evaluation distribution would be misleading.
2. Separately test an earlier learning-rate decay or stronger dropout. The
   current 300-epoch cosine keeps the learning rate close to its initial value
   through the point where validation already worsens. Do not combine these
   changes in the first ablation, because their effects would be confounded.
3. Keep representation quality under review. Two fitted raw levels per atom
   are lossy even without clipping; the paired probe shows about 6.5% higher
   pixel MSE than native OMP. A representation or decoder change would need its
   own matched reconstruction and generation comparisons.

The best and latest checkpoints remain preserved, and the existing training
continues. Audit files and figures are under
`outputs/church-stage2-distortion-audit-20260922`. Sampling results are under
`outputs/church-epoch57-sampling-sweep-20260922` and online at
https://wandb.ai/helloimlixin-rutgers/laser/runs/church-laser-epoch57-sampling-sweep-20260922.
