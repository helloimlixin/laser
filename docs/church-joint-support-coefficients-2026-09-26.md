# Church: joint support/coefficient modeling and DCTransformer

The active LASER model already predicts a joint distribution over completed
OMP pairs. The next useful comparison is how well that distribution generalizes
and whether a dedicated coefficient sequence decoder helps under a matched
recipe. Standard OMP remains the representation; coefficient refitting does not
invalidate autoregressive training on its final pairs.

## What DCTransformer contributes

DCTransformer factorizes each sparse entry as
`p(channel | history) p(position | channel, history) p(value | channel, position, history)`.
It uses stacked causal channel, position, and value decoders. Earlier complete
triples enter the history. The value decoder also receives spatially gathered
features from an encoded partial DCT image. Predictions are categorical over
quantized values. This is conditional factorization, not independent support
and value modeling. [Paper, equations 2–9](https://proceedings.mlr.press/v139/nash21a/nash21a.pdf)

The supplement also reports nonuniform training allocation favoring early
chunks, and notes that values contribute most of its measured coding loss.
Those choices are representation-specific; they are not instructions to assign
LASER atom depth a DCT frequency interpretation.
[Supplement, appendices C–D](https://proceedings.mlr.press/v139/nash21a/nash21a-supp.pdf)

## What the current LASER model does

The frozen production source implements

```
p(A, C) = product_d p(a_d | earlier completed pairs)
                   p(c_d | earlier completed pairs, a_d)
```

Spatial position follows the fixed raster/depth order. Each coefficient
classifier is conditioned on its selected dictionary atom through a two-token,
two-layer micro-transformer. Earlier atom IDs and coefficient IDs enter the
full-pair depth history. Generation samples an atom, conditions the coefficient
head on that sampled atom, samples its coefficient, and only then advances.
Support atoms are masked consistently to remain distinct within an OMP site.

This factorization can represent support/value dependence; separate output
heads do not imply independence. The architecture difference worth testing is
direct sequence attention in the coefficient branch. The current local
conditioner receives a compressed backbone state and current atom. A separate
sequence decoder can attend to explicit earlier pair states and the partial
reconstruction. That is an architectural hypothesis, not evidence of lower FID.

An implementation already exists in `src/training/compound_history.py` and
`src/models/compound_coefficient_decoder.py`: two width-512 layers, about 7.88M
additional parameters, shifted pair history, and strictly prior physical
contributions. It is **not enabled in the active Church run**. Its earlier
fresh run, `church-laser-compound-history-scratch300-b2048-h200x5-20260922`,
reported best FID50k 10.4682 and was marked crashed near epoch 117. It used
different coefficient bins/units, target temperature, and sampling. This is not
a matched verdict against today’s 9.8003 best, and not a new untried architecture.
The earlier planned continuation control did not complete.

## Trained-checkpoint coupling audit

I evaluated the fixed epoch 114 checkpoint in CPU FP32 on 64 training and 64
held-out images, using the existing authenticated deterministic-OMP/nearest-bin
probe. Image rows are fixed by seed 2026092608. Losses below are means over
sites and depths; the same examples and targets are used in all conditions.

| Condition | Train atom NLL | Train coefficient KL | Held-out atom NLL | Held-out coefficient KL |
|---|---:|---:|---:|---:|
| Matched history and current atom | 4.1190 | 0.3442 | 10.9046 | 0.6591 |
| Wrong atom supplied only to coefficient conditioner | 4.1190 | 2.4675 | 10.9046 | 2.3347 |
| Earlier within-site coefficient history permuted | 6.1646 | 0.8492 | 11.1029 | 1.2070 |

Changing the current atom input increases held-out coefficient KL by
1.6757±0.0392 standard error. Permuting within-site coefficient histories
increases held-out atom NLL by 0.1983±0.0199 and coefficient KL by 0.5480±0.0141.
The current-atom intervention leaves atom logits unchanged. These results
establish that the trained model uses both kinds of coupling. They do not
measure the effect of training a different architecture.

The train/held-out gap is substantially larger for support prediction than for
coefficient KL. At depth one, atom NLL is 0.3801 on these training images versus
12.7070 held out. Split differences and the fixed-history probe affect absolute
values, so this is a diagnostic of poor held-out likelihood, not proof of one
cause of the FID gap. Further coefficient-only capacity is not automatically
the right remedy. Track the unweighted joint loss and both contributions.

Changing the final target coefficient leaves all predictions exactly unchanged
in this actual checkpoint. The existing causality and cache-equivalence tests
also pass for the full-pair model and optional history decoder.

## Exact conditional coefficient targets implemented

The existing bank teacher samples one full-refit trajectory per site. It then
draws coefficient histories from that trajectory's soft kernels. Atom targets
already integrate the posterior over compatible bank trajectories, but the
coefficient target uses the selected trajectory's kernel. That estimator is
unbiased for its specified conditional-mixture objective.

`src/training/omp_joint_targets.py` now computes both exact finite-bank
conditionals. At depth d it forms the posterior over trajectories given earlier
sampled pairs, produces the atom target, conditions that posterior on the
current selected atom, and averages the compatible coefficient kernels. It
never conditions the current target on its own sampled coefficient or future
pairs. Duplicate bank entries retain their empirical weights. It also provides
an unweighted joint soft cross-entropy, summing atom and coefficient terms in
nats per pair, without the current 1.5 atom multiplier or division by two heads.

This is a target-estimator improvement, not a richer model family or a change
to the expected coefficient objective. It preserves full-refit OMP and the
bank's joint law. At a fixed history and current atom it removes randomness due
to choosing one compatible trajectory's coefficient kernel. It does not remove
the finite-bank coverage limitation or establish better generation.

On 128 fixed training images / 8192 sites, the atom targets agree with production
within 6.56e-7. Mean total-variation distances between the sampled coefficient
kernel and exact mixture are 0.0446, 0.0629, 0.0583, and approximately0 by depth.
At the final depth, compatible complete supports give the same final refitted
coefficient. The modest changes in earlier targets suggest a bounded ablation,
not an explanation of the entire FID gap. All target probabilities are finite
and normalized. No real-data signal, dictionary, or coefficient was refitted by
this test.

Twenty-three tests pass across the new target/loss implementation, full-pair
autoregression, cached generation, and the existing coefficient history
decoder. New tests independently enumerate the finite joint distribution and
verify its conditional probabilities, prefix causality, duplicate weighting,
and loss gradients.

## Controlled next comparison

The original-RQ recipe remains the baseline priority: batch 2048, the existing
AdamW/cosine budget, residual dropout 0.1, and explicitly recorded stochastic
teacher differences. Standard OMP and the tested full-pair causal paths stay.
More temperature tuning and the earlier causal residual-coder prototype are
deferred.

First isolate equal atom/coefficient likelihood weighting and exact conditional
targets, keeping the architecture fixed. Report joint validation likelihood,
support NLL, coefficient KL, generated latent cross-depth moments, and paired
FID50k. An exact-mixture comparison alone tests the estimator; changing the
relative atom weight additionally tests the objective. Keep those conclusions
separate.

Then compare the existing coefficient sequence decoder against the same
baseline with matched teacher, optimizer, data, and sampler. The primary
architectural question is whether explicit pair memory improves held-out joint
prediction and generated cross-depth dependence. Train from scratch for the
main comparison, use all three GPUs, and benchmark physical batches while
preserving exact global batch 2048. These changes are prepared and checked;
they have not been injected into the live runs or claimed to improve FID.

Reproducible probes are in
`outputs/church-joint-support-coefficients-20260926/`. The trained-weight audit
uses the previously authenticated frozen epoch 114 checkpoint; no optimizer
updates occurred in these probes.
