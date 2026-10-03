# One physical coefficient vocabulary for all sparse depths

The prepared LSUN Church representation uses the same 2,048 ordered, uniform
physical coefficient centers at all four depths. Every runtime depth scale is
one. A coefficient ID therefore denotes the same signed coefficient everywhere;
the paired sparse contribution is `dictionary[:, atom_id] * centers[coefficient_id]`.

The grid covers **[-22.9870662689209, 22.9870662689209]**, with spacing about
0.02245927. Its range includes every coefficient in the original canonical cache,
every coefficient in the original 16-variant bank, and all legacy grid endpoints.
The centers are defined once in CPU FP32 and transferred unchanged to the device.
No coefficient or dictionary fitting is performed.

Conversion multiplies each stored FP32 normalized coefficient by its original
depth scale exactly once. The converted cache then stores physical coefficients
with `coeff_scales=[1,1,1,1]`, `coeff_scale=1`, no normalization and no clipping.
Atom IDs, support order, labels, and all bank alternatives are retained. This
preserves the continuous physical coefficients available in the source cache;
it cannot undo rounding that occurred when that source cache was created.

The changed bins invalidate cached coefficient IDs, probabilities and old
coefficient log normalizers. Those tables are removed and recorded in provenance.
The cache and coefficient vocabulary receive new identities. The canonical
single-trajectory cache is not substituted for the bank that initialized the
current model's fresh paired teacher.

## Measured representation error

| Check | Legacy depth-specific grid | Shared physical grid |
|---|---:|---:|
| Added latent MSE per channel, 4,096 canonical images | 2.3961e-7 | 6.5575e-7 |
| Added latent MSE per channel, 1,024 bank images and all 16 variants | 2.3931e-7 | 6.5588e-7 |
| Pixel MSE to continuous decode, 64 canonical images | 3.7550e-7 | 9.9988e-7 |
| Pixel MSE to continuous decode, 64 bank images with one fixed-seed variant each | 3.7763e-7 | 9.9964e-7 |

Pixel comparisons use the frozen Church decoder in FP32, TF32 disabled, and
pixels in [-1,1]. These are reconstruction probes, not generated images or FID.
Continuous sparse contributions before and after conversion are bitwise equal
in the decoder probe. The shared-grid runtime's paired embeddings also match
explicit physical vector construction exactly.

Later depths receive coarser quantization: their old resolutions improve on this
shared grid by factors about 1.84, 2.91 and 4.64. The resulting absolute errors
remain small. Consistent token meaning does not establish easier prediction or
lower generated FID. A previous shared physical **Lloyd–Max** trial reached
FID 10.688 before regression; that trial had different targets, sampling and
training settings, and used a fitted nonuniform grid. It does not isolate the
effect of physical units. See [that trial](church-stochastic-rawcoeff90-2026-09-21.md).

## Checkpoint and teacher compatibility

The existing RQ BEST10540 checkpoint remains the control with its original codec.
Its shared coefficient embedding and four coefficient heads learned the old
depth-specific meanings. A shape-compatible state-dict load does not make those
weights compatible with the new vocabulary. The common 2,048-bin grid merges
legacy physical values, so a row permutation cannot preserve the model exactly.
The new representation requires explicit vocabulary adaptation or fresh training.
This preparation does not launch either and does not change the control weights.

Twenty-one focused helper and trainer checks pass, including CPU/CUDA identity
of the common grid and rejection of incompatible resume and initialization
checkpoints before loading weights. Both converted caches also pass exhaustive
support, label, physical coefficient, range and serialized-file checks.

The current fresh paired teacher already scores the whole sparse combination in
physical latent units. It can consume the converted physical bank and unit scales,
but its grid must be replaced explicitly. The existing physical noise parameter
is 0.4204482076268573; the historical normalized temperature 0.0625 and its cached
partitions are not equivalent replacements. Before a future adaptation, measure
the whole-vector perturbation on the new grid while keeping support and its
coefficient coupled. No refitting, reward objective, or architecture change is
introduced by this conversion.

A bounded integration probe runs the actual fresh paired teacher on 32 sites
from the converted bank at that unchanged physical temperature. All four depth
lookups are identical, probabilities normalize, and independently computed FP64
whole-vector energies agree with the atom and selected-atom coefficient targets.
Complete sampled pairs are preserved in the next depth's history. This checks
units and conditioning; the small probe is not a noise calibration or FID test.

Assets and checks are stored in
[the prepared output directory](../outputs/church-shared-physical-grid-20260929/).
The converter is
[prepare_shared_physical_coefficients.py](../scripts/tools/prepare_shared_physical_coefficients.py),
using [shared_physical_coefficients.py](../src/shared_physical_coefficients.py).
The audit and reconstruction comparison are recorded on
[W&B](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-shared-physical-grid-20260929).
