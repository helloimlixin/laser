# Conditional sparse-code sampling on the existing RQ model

This adapts DCTransformer's published conditional sampling mechanism to the existing compound RQ checkpoint. It adds an explicit sampling interface and preserves the model architecture, learned weights, tokenizer, quantization, physical coefficient scales, and trained spatial/depth order. It creates no optimizer and performs no training.

For each fixed spatial site and sparse-code depth, the sampler obtains the original cached RQ hidden state, samples an atom, predicts the coefficient distribution conditioned on that selected dictionary atom, samples the coefficient bin, and commits the completed pair to history before advancing. The joint distribution is `p(atom | history) * p(coefficient | atom, history)`. Same-site duplicate atoms remain masked. The original model's pair embedding incorporates the signed, depth-scaled dictionary contribution along with atom and coefficient identities.

The RQ backbone keeps its spatial and depth KV caches. Its trained raster order must be preserved: this sampling-only adapter does not introduce the DCT paper's encoder, frequency order, variable-position predictor, or chunked training. DCTransformer's tuple factorization motivates the conditional draws; its exact numerical generation filters are not documented in the paper. [Paper, equation 2 and section 3.3](https://proceedings.mlr.press/v139/nash21a/nash21a.pdf).

The new standalone interface is:

```python
from src.compound_ancestral_sampling import sample_dc_ancestral

model.eval()
atoms, coefficient_ids = sample_dc_ancestral(
    model, batch_size=128, model_aux=aux, cond=labels,
    atom_temperature=1.0, coeff_temperature=1.0,
    atom_top_k=None, coeff_top_k=None,
    atom_top_p=None, coeff_top_p=None,
    amp=True,
)
```

The explicit default uses full eligible vocabularies at temperature 1. Optional per-field filters remain available; they are choices for this RQ model and are not labeled as the authors' exact settings. A full-vocabulary top-k operation is skipped because it removes no candidates. Native FP32 categorical arithmetic and nucleus-filter boundary behavior are retained for reproducibility. Joint-pair top-k enumeration is not introduced.

The selected frozen checkpoint is original RQ BEST10540, SHA256 `8e9a209aec912909b33ec8fd92c18c42d41f09a5d57e431b05a39d3285d9609b`. Its existing untruncated 50k FID is 11.169160294; historical filtering gives 9.493208036. The adapter is expected to reproduce the untruncated distribution. Its evaluation verifies implementation and measures runtime; it is not a new training or quality-improvement experiment.

The completed implementation checks passed 30 CPU tests, 108 independent categorical-probability/RNG cases, and 32 actual-checkpoint GPU cases covering both historical and untruncated settings at batch sizes 128 and 2 across eight ranks. Every sampled atom/bin and final RNG state matched the original sampler. All model and auxiliary parameter/buffer guards passed.

The new adapter then generated 50,000 images. All eight saved Inception feature files are byte-identical to the prior native untruncated evaluation, with identical moments and FID **11.169160294448574** (mean **4.190478801727295**, covariance **6.978681492721279**). This establishes implementation equivalence, not a quality improvement. The historical filtered setting remains the measured FID 9.493208 baseline; its native token/RNG behavior also passed the adapter parity check.

The full 50k pipeline took **153.19 seconds on eight H100 GPUs**, including generation, decoding, Inception and final FID computation. The prior untruncated run used four physical GPUs for eight logical streams, so total wall times should not be interpreted as an isolated sampler speedup. The adapter retains RQ caches and skips only redundant full-vocabulary top-k sorting.

Run: [RQ sparse-code sampling adaptation](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-rq-dc-sampling-best10540-20260929). Source, tests, checkpoint reference, preview images, metrics, feature arrays and validation receipts are recorded with this evaluation.
