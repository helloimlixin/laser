# Sequence-decoder regression audit

The joint sequence experiment currently generalizes worse than the preceding
T0.0625 LASER experiment. This comparison freezes both curves at epoch 142;
the live sequence run continues while the audit reads saved logs.

| Metric | Previous LASER | Joint sequence model |
| --- | ---: | ---: |
| Mean FID50k, epochs 121–142 | 10.7793 | 11.6354 |
| Best FID50k through epoch 142 | 9.8003, epoch 114 | 10.0861, epoch 76 |
| Training-probe support NLL, epoch 142 | 3.6746 | 2.5546 |
| Validation-probe support NLL, epoch 142 | 11.1792 | 12.4307 |
| Training-probe coefficient KL, epoch 142 | 0.3262 | 0.2380 |
| Validation-probe coefficient KL, epoch 142 | 0.6646 | 0.8309 |

All listed losses and FIDs are lower-is-better measures. The new model fits
training code histories more closely while assigning worse likelihood to
validation histories. Its sustained FID is also worse. This supports increased
overfitting, rather than an explanation based solely on early training or one
unlucky FID evaluation.

The new model initially learned faster: average FID over epochs 21–40 was 13.06
versus 14.59. After epoch 80, the earlier model continued improving while the
sequence run stayed around 11.5–11.8. The best sequence checkpoint at epoch 76
has been preserved independently of live checkpoint rotation.

## What the comparison controls

The 300-image training and 300-image validation probes are byte-identical, as is
the evaluation code. Their SHA256 values are recorded in `comparison.json`.
Both use the same coefficient target temperature and code histories during this
probe. Therefore the validation regression is not explained by different probe
data or loss normalization. The raw logged total training losses are not directly
comparable, because the new training objective changed weighting and scale.

The FID metric implementation, real reference, sample count, and sampling
temperatures/top-k/top-p agree. GPU count and generation batch size differ, so
the curves are not a common-seed paired checkpoint evaluation. There is also
only one training realization per setting. No causal conclusion about one
individual change follows from these runs alone.

## Changed factors and next test

The launch combined additional sequence decoders, residual dropout reduced from
0.2 to 0.1, unweighted joint CE, and exact conditional coefficient-mixture targets.
The dropout reduction followed the requested RQ recipe, but its interaction with
the LASER architecture and finite OMP teacher was not isolated.

The first controlled test should restore residual dropout to **0.2**, keeping the
sequence architecture, joint objective, coefficient target mixture, stochastic
teacher, sampler, global batch 2048, and learning-rate schedule fixed. Initialize
from scratch using the same weights as the current sequence experiment. The
draft configuration records the expected initialization SHA256. This is a
prepared experiment, not a launched replacement.

This test can measure whether restoring regularization helps; it cannot guarantee
FID improvement. A subsequent matched decoder ablation would be needed to
attribute any remaining regression to the added sequence stacks.

At epoch 142, generated Inception covariance trace is 83.678 versus 101.622 for
real images (82.34%). The FID mean term is 4.256 and covariance term is 7.313.
A large covariance-distance term does not imply excessive total feature variance.
These observations do not support further blanket feature-energy suppression as
the first intervention.

Artifacts are under `outputs/church-sequence-regression-20260926`, with a PNG/PDF
curve comparison, machine-readable measurements, and the draft dropout control.
The independent best-checkpoint hard link is under
`/mnt/laser-church/sequence-regression-audit-20260926`.
