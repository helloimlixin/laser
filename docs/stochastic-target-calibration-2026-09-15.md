# Stochastic target calibration — 2026-09-15

The compact tokenizer now supports a fixed temperature per residual depth.
The new policy calibrates mean target entropy against the released RQ tokenizer
on the same original training pixels, then measures reconstruction distortion
independently. The original scalar-temperature API remains supported. This is
an experimental correction to the calibration criterion; improved prior
generalization and FID have not yet been established.

The previous calibration chose one scalar temperature from a bound on the
sampled/hard final latent-MSE ratio. That criterion selected 0.125, producing
targets much sharper than the reference. A scalar also coupled the amount of
stochasticity at all four depths, despite their different residual distributions.

## Mechanism

At depth k, form scores `-squared_distance(residual, codeword)` and normalize
them with temperature `temperature[k]`. Sample the input code from that same
distribution and use the whole distribution as the soft label. Subtract the
sampled codeword before constructing the next depth's distribution. The zero
token, atom/coefficient token order, vocabulary, deterministic encoding and
decoder remain unchanged.

Temperatures are fitted sequentially. Each depth's entropy search holds its
residual population fixed; after finding its temperature, the calibration
samples that depth's codes before fitting the next depth. An unreachable entropy
target raises an error rather than silently clipping to a temperature bound.
The fitted temperatures are fixed throughout training. Geometry and probability
construction run in FP32 even inside ambient autocast.

This preserves the stochastic-prefix/soft-label consistency of original RQ
training. It does not apply independent label smoothing to a hard input sequence,
nor adjust the prior's generation temperature.

## Calibration and independent checks

The reference is the **released Church RQ tokenizer**, evaluated afresh at
training-target temperature 0.5. This replaces the previously reused control
profile, so its entropy values differ from earlier audit tables. Both tokenizers
see identical pixels from the original RQ-VAE loader. Four fresh encodings were
also checked against the existing continuous latent cache.

Calibration uses 128 randomly selected training images. A disjoint set of 256
training images checks the resulting policy with three stochastic-code seeds.
No validation images were used to choose temperatures.

| Residual depth | Previous temperature | New temperature | Previous entropy | New entropy | Released-reference entropy |
|---|---:|---:|---:|---:|---:|
| 1 | 0.125 | 0.229965 | 0.0689 | 0.1299 | 0.1380 |
| 2 | 0.125 | 0.375955 | 0.1894 | 0.6941 | 0.6926 |
| 3 | 0.125 | 0.279208 | 0.3869 | 1.0211 | 1.0249 |
| 4 | 0.125 | 0.229370 | 0.6529 | 1.3688 | 1.3800 |

Entropies are nats per token, averaged over the three independent check seeds.
The mean rises from approximately 0.325 to 0.803, close to the reference's 0.809.
Matching entropy does not make the token distributions identical: geometry,
vocabulary and the probability of taking a non-greedy code still differ.

The stochastic latent-MSE ratio relative to hard reconstruction rises from
**1.0128 to 1.0602**. The deterministic tokenizer is identical in both checks.
In a separate 32-image, single-seed reconstruction diagnostic, PSNR changes
from **19.3708 to 19.3580 dB**. That small pixel difference supports testing the
policy, but the sample is small and it is not a FID measurement. We explicitly
relax the old distortion-based criterion; entropy matching does not satisfy
the previous MSE-ratio bound automatically.

## Training trial

A fresh 30-epoch trial is running in
`outputs/church-target-entropy-20260915/trial`, alongside the original run:
[W&B trial](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-laser-depth-entropy-rq32k-trial30-20260915).

It reuses the original tokenizer, codebook, continuous cache, full-training FID
reference, exact initial model weights, architecture, optimizer, global batch
2048 and 300-epoch cosine schedule. It pauses after epoch 30 without shortening
the schedule. Per-GPU microbatches are 32 with 32 accumulation steps to fit
alongside the existing run. This changes dropout and target-sampling RNG streams
relative to the original run's microbatches, so the comparison is not a bitwise
paired ablation. No additional pixel resizing or cropping is introduced.

Validation runs each epoch and logs soft CE, target entropy, CE minus entropy
(target-to-model KL), and hard-code NLL. Raw soft CE across different target
policies is not directly comparable. The three reportable FID evaluations are
scheduled at epochs 10, 20 and 30, each using **50,000 generated images against
all 126,227 training images**. Generation uses temperature 1, top-k 1400 and
top-p 1 to keep the sampling rule fixed. Decoding/Inception use batches of eight
to bound memory. Best validation and best FID50k model weights are retained,
along with the latest complete optimizer/RNG checkpoint.

The trial executes copied runtime files and verifies the original dependency
snapshot. Changing working-tree files does not alter its running policy. Its
calibration includes an implementation hash; resuming with a changed target
calibration or implementation is rejected. The existing run was not stopped,
restarted or modified.

## Verification

- 35 tests pass, covering dense-codebook equivalence, sampled-prefix conditioning,
  scalar compatibility, depth-specific codebooks, autocast, entropy calibration,
  invalid/unreachable targets, causal sampling and checkpoint compatibility.
- The full 386,882,561-parameter model made two successful DDP updates at global
  batch 2048, then resumed its full checkpoint and made update three. No AMP
  updates were skipped; gradients and losses were finite.
- On both GPUs, the new method with repeated scalar temperatures exactly matched
  the frozen implementation's probabilities, codes and final RNG state.
- The full distributed validation path processed all 300 validation images,
  measured target entropy 0.8063, and returned identical reduced metrics on both
  ranks. Tokenizer hashes remained unchanged.
- The preflight initial-model hash exactly matched the original run.

The training checks establish execution and consistency, not an improvement in
overfitting or FID. Those claims require the completed trial and matched
checkpoint evaluations.

## Files

- [Target policy](../src/training/stochastic_targets.py)
- [Compact quantizer integration](../src/compact_rq_training.py)
- [Calibration tool](../scripts/tools/calibrate_compact_target_entropy.py)
- [Training driver](../scripts/tools/train_consistent_rq_stage2.py)
- [Calibration measurements](../outputs/church-target-entropy-20260915/calibration/calibration.json)
- [GPU/runtime verification](../outputs/church-target-entropy-20260915/runtime-verification.json)
- [Trial launch and runtime hashes](../outputs/church-target-entropy-20260915/launch.json)
- [Trial status](../outputs/church-target-entropy-20260915/trial/status.json)
