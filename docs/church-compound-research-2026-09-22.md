The compound representation remains the better measured Church baseline. Its
epoch-57 checkpoint achieved FID50k 9.9552; the completed sampling comparison
reduced the two-seed mean from 9.9338 to 9.8280 by changing coefficient
temperature to 1.2. This investigation does not claim a new generation result.
The integer direction was stopped at an epoch boundary after the user switched
focus back to compound tokens; checkpoint publication is recorded separately.

The best checkpoint is
`/tmp/laser-church-integer-raw-rqrecipe300-20260922/compound-handoff/best_fid_9.9552_epoch_057.pt`.
Its frozen implementation is
`outputs/church-stage2-released-recipe-20260921/runtime/src/training/rqtransformer.py`.

**What our compound model actually does**

Each event has atom `a` and coefficient `c`, with the valid factorization
`p(a,c | completed pairs) = p(a | completed pairs) p(c | a, completed pairs)`.
There are 256 events: four depths at each of 8×8 sites. Sampling chooses the
atom first, then its signed coefficient, before proceeding to the next event.
The packed integer is transport encoding, not a single joint vocabulary.

The 16,384-way atom classifier and the four depth-specific, 2,048-way
coefficient classifiers are different heads. Coefficients use nonuniform bins
in raw units, scale one, and no explicit coefficient clipping. This remains
finite quantization with bounded output support; it is not continuous sampling.
The coefficient loss uses soft physical-distance targets at temperature 0.125.

Earlier pair embeddings contain the dictionary vector, a learned coefficient
embedding, and the physical contribution `c * dictionary[a]`. A learned adapter
is added to that physical contribution. The spatial body consumes sums over
completed sites; the depth head consumes cumulative earlier pair embeddings.
The coefficient head's two-layer micro-transformer sees just two tokens per
event: the causal history hidden state and the selected atom projection. It
therefore receives history through the backbone, without a separate attention
path over all earlier events or an encoded partial reconstruction.

A read-only FP32 probe using the actual epoch-57 weights confirmed:

| Perturbation | Observed effect |
|---|---|
| Change current coefficient and every future pair, at events 0, 1, 3, 4, 127, 255 | Exactly zero change to atom/coefficient logits through the current event |
| Change current atom at event 1 | Atom logits unchanged; coefficient logits change, maximum 1.74925 |
| Change coefficient of event 0 | Both heads change at subsequent events 1, 4, 127 |
| Incremental cached sampler versus dense teacher forcing, all 256 events | Maximum atom-logit difference 3.48e-5; coefficient-logit difference 1.91e-5 |

These are one-image FP32 dependency and cache checks, not a complete numerical
or quality test; mixed-precision behavior was not tested. The earlier OMP coefficients are solved jointly with later supports,
but including them as already-generated variables is legitimate chain-rule
modeling. A misleading comment in the maintained source was corrected; the
frozen training snapshot and model behavior were preserved.

Evidence and executable probe:
[audit.json](../outputs/church-compound-research-20260922/audit.json),
[audit.py](../outputs/church-compound-research-20260922/audit.py).

**DCTransformer and recent alternatives**

| Method | Mechanism | Transfer to LASER |
|---|---|---|
| [DCTransformer, ICML 2021](https://proceedings.mlr.press/v139/nash21a/nash21a.pdf) | Factorizes channel → position → value using three stacked causal decoders. The value decoder also receives spatially gathered features of an encoded partial DCT image. | Our atom → coefficient ordering already follows the relevant principle. The missing architectural comparison is a coefficient sequence decoder with direct history/partial-reconstruction context. |
| [GIVT, ECCV 2024](https://arxiv.org/html/2312.02116v4) | Projects continuous inputs and predicts a Gaussian mixture distribution, including a causal AR variant. | Keep atom classification; predict a continuous distribution for its raw signed coefficient. This is the smallest distributional change worth testing. |
| [JetFormer, ICLR 2025](https://arxiv.org/html/2411.19722) | Combines discrete text prediction with continuous image soft tokens and a Gaussian mixture loss, using a jointly learned normalizing-flow encoder. | Supports using different likelihood families for different fields. Its flow tokenizer is a much larger change than we need. |
| [MAR, NeurIPS 2024](https://arxiv.org/html/2406.11838), [NextStep-1, 2025](https://arxiv.org/html/2508.10711v1) | Use a context-conditioned diffusion or flow head for continuous token distributions. MAR also studies causal/raster variants; NextStep uses a causal backbone. | A conditional density head can preserve raw coefficients without bins. Iterative coefficient sampling adds cost at every pair; a scalar mixture is the cheaper first experiment. |
| [HART, ICLR 2025](https://arxiv.org/html/2410.10812) | Discrete structure plus continuous residuals, modeled with a small diffusion module. Its hybrid tokenizer is trained for both components. | A later option for residual refinement, with additional tokenizer/decoder compatibility work. It does not imply that refining coefficients can repair incorrect support. |
| [FAR, revised March 2026](https://arxiv.org/html/2503.05305v2) | Generates progressively higher-frequency continuous latent maps; uses diffusion loss and frequency-dependent masking/sampling. | Suggests generating global structure before detail. OMP depth is not a frequency band: changing our order alone would not reproduce FAR. |

DCTransformer represents nonzero DCT entries with bounded categorical values
and processes low frequencies before high ones. It reports Church FID 7.56,
but uses variable aspect ratio with long side 384, so this is not our matched
256×256 benchmark. Its [supplement](https://proceedings.mlr.press/v139/nash21a/nash21a-supp.pdf)
describes partial-image encoding, chunked training, and a preference for early
chunks. We should transfer conditional structure rather than coefficient
clipping or its very different token budget.

These papers establish useful mechanisms on their own representations and
datasets; they do not establish a Church FID improvement for LASER.

**The strongest measured problem is atom generalization**

These are existing evaluation-mode measurements on the same fixed 300-image
training probe and 300 official validation images, with fresh stochastic OMP
targets. Every optimizer epoch still uses all 126,227 training images.

| Compound epoch | Train atom NLL | Validation atom NLL | Train coefficient KL | Validation coefficient KL |
|---:|---:|---:|---:|---:|
| 17 | 8.4061 | 8.7782 | 1.1247 | 1.1708 |
| 57, best FID | 4.7110 | 10.4355 | 0.9254 | 1.2379 |
| 187 | 2.7566 | 12.9365 | 0.7569 | 1.4217 |

At epoch 57, the first-depth atom NLL is 0.9333 on training images and 11.3593
on validation images. The issue includes the first structural choices. Absolute
atom NLL and coefficient KL are different prediction tasks and should not be
compared as a common error scale. Their within-task trajectories nevertheless
show increasing overfitting. Better token likelihood is also not guaranteed to
give better FID; epoch 57 remains the generation baseline.

An exact all-atom dictionary screen found median nearest-other absolute cosine
0.7353; 0.97% of atoms have a neighbor above 0.90, 0.26% above 0.95, and none
above 0.99. This does not support assuming that almost all labels are duplicate
atoms. Equivalence between multi-atom supports remains a separate, unmeasured
possibility; broad atom-label smoothing is not justified by this screen alone.

**Recommended experiment order**

1. Establish a fresh compound control and a single regularization change:
   residual dropout 0.1 → 0.2. Retain the current full-data center-crop protocol,
   raw coefficients, optimizer/schedule, and tokenization. This prioritizes the
   measured atom overfitting; no improvement is assumed. Earlier prepared
   regularization pilots were superseded and never ran. Merely continuing the
   heavily memorized epoch-187 model is a poor test of prevention. Use epoch 57
   as the preserved generation benchmark and initialization only for explicitly
   labeled continuation comparisons.
2. Test an atom-conditioned continuous coefficient mixture as a separate
   representation ablation. For 16 scalar Gaussian components, output 16
   mixture logits, 16 means, and 16 positive scales: 48 outputs instead of
   2,048 bin logits. Train raw-coefficient log likelihood and retain atom CE.
   Feed true continuous previous coefficients through a numeric embedding plus
   their physical dictionary contributions; rounding them back to bins would
   undo the representation change. Use stable FP32 log-sum-exp and a positive
   scale floor; do not clip, normalize away, or truncate coefficient values.
   The floor constrains distribution width, not sampled coefficient magnitude.
   A mixture captures multimodality; scalar MSE regression does not. This is
   our adaptation of GIVT, not an evaluated LASER result. Head dimensionality
   falls, but total throughput still depends mostly on the backbone and atom
   classifier. New input/head parameters prevent an exact checkpoint resume.
3. Independently replace the two-token coefficient micro-transformer with a
   short causal coefficient decoder, supplied with current atom, backbone
   state, and explicit prefix reconstruction. Compare with the same coefficient
   likelihood to isolate history access. A causal memory of completed spatial
   sites could extend this. The current embedding already contains physical
   contributions; the proposed change gives a dedicated access path rather
   than adding physical information for the first time. Validate full-sequence
   and cached generation agreement before training.
4. Consider global coarse-to-fine generation or masked compound prediction
   only after the smaller tests. This may address large-scale consistency, but
   requires an explicit ordering/masking design and changes checkpoint and
   cache compatibility. Copying VAR/FAR names onto four OMP depths is not a
   valid implementation. Diffusion/flow coefficient heads are a subsequent
   density-model option if a mixture is demonstrably inadequate.

Keep per-depth training/validation diagnostics and fixed, unselected sample
grids. Each actual training epoch must retain official RQ FID50k against all
126,227 training images, using the same resize/center-crop inputs as the stage-2
cache. Select by generated FID, preserve full last/best checkpoints online,
and confirm a candidate with another generation seed. No new long training
run or new FID result was produced by this research audit.

The cache uses the frozen tokenizer with SHA256
`762c51a10267ed6fa55709ff0d6cf997940d21056b16c7a0a0399b3ebf93868d`.
The current real-statistics SHA256 is
`ad3b5a341831d877110afdff37d6fff4be855612053e3eb9e2e7119f5593d364`.
See also the [sampling report](church-epoch57-sampling-2026-09-22.md) and
[earlier generalization investigation](church-stage2-dct-comparison-2026-09-21.md).
