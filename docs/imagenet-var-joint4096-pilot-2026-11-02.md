# ImageNet VAR: 4,096 joint atom/coefficient symbols

This pilot transfers the finite joint-integer idea from
[the native RQ run](https://wandb.ai/helloimlixin-rutgers/laser/runs/imnet-j16384-a4096-best42-spawn-lr010-clip01-cont100-2b300-20261101)
to the successful multiscale VAR tokenizer. The requested budget is **4,096
joint symbols total**, matching the original VAR categorical vocabulary.
The reference RQ run instead uses 16,384 symbols, four residual depths, an 8x8
layout, another tokenizer, and another prior; its FID is not a VAR comparison.

The frozen source is the epoch-10, step-100090 LASER checkpoint with SHA256
`3f0c1b79272e8d6177e284f45e9acd923b5bc678890b2b9766a14e07ce92b6aa`.
Its full-validation reconstruction FID is 1.5207, versus 2.4988 for the paired
VQ control. The pilot does not retrain either image encoder or decoder.

## Token definition and fitting

For `A` selected dictionary directions and `L` coefficient levels, use
`id = atom * L + level`, with `A * L = 4096`. Compare `(A,L)` equal to
`(2048,2)`, `(1024,4)`, and `(512,8)`. Selection retains original frozen
directions ranked by squared coefficient usage on fitting images. Signed
coefficient centers are fitted from training-only OMP projections, with priors
for poorly observed directions. They depend on scale but are shared across
sparse depth. This matches vocabulary cardinality, not the original VAR's
exactly shared physical embedding table across all scales.

At each scale, greedy residual encoding chooses two fixed joint codewords,
excluding reuse of a direction within a site. Earlier contributions are never
refitted during encoding. Decoding uses only the two integer IDs, the fixed
tables, and the known scale. There are no per-image continuous coefficients.
The original trained residual convolutions combine the scale contributions.

The prior retains VAR's 680 spatial positions and causal next-scale attention.
A shared 4,096-way head predicts each contribution, conditioned on the summed
physical vectors of previous contributions at that site. Sparse depth is
autoregressive, and training and sampling impose identical support masks.

This still uses **1,360 joint integers/image**, versus 680 VQ integers/image:
16,320 versus 8,160 nominal fixed-width bits. Matching vocabulary size does
not match bitrate or the number of categorical predictions.

## Bounded evaluation

The local validation tar has an empty initial header and an unfinished aria2
receipt. It is not used for this pilot. Each of 128 evenly spaced ImageNet
training classes supplies one fitting image and a separate screening image.
The same fixed 288px Lanczos resize/256px center crop is used in every arm.
The heldout training screen reports paired PSNR and latent distortion; it
must not be reported as validation rFID or compared numerically with FID50k.

The learning check is a 40-update, eight-image fixed-batch fit of VAR-d16 with
AdamW LR 1e-4 and gradient clipping 0.1. It checks whether the joint-token
implementation can optimize, not whether it generalizes or beats VQ generation.
Full training needs a separate matched training budget and evaluation.

Implementation:

- `src/models/joint_multiscale_var.py`: integer encoding/decoding and joint prior.
- `scripts/tools/pilot_var_joint4096.py`: source verification, train-only fitting,
  paired reconstruction screening, and bounded learning check.
- `tests/test_joint_multiscale_var.py`: explicit expanded-codebook assignment,
  integer-only roundtrip, causal boundaries, finite gradients, and cached
  sampling/teacher-forcing agreement with and without CFG.

Pilot receipts and reconstruction previews are written under
`outputs/imagenet-var-joint4096-pilot-20261102/`; the fitting image and latent
cache is local to `/tmp/laser-var-joint-pilot-20261102/`.
