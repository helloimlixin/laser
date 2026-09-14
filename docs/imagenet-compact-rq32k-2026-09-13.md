# ImageNet compact atom-specific RQ vocabulary

This applies the Church compact vocabulary construction to ImageNet's frozen
dictionary. The user requested the change for the ImageNet process on this
machine and explicitly chose a fresh prior. The existing prior is
`imagenet-rfid421-rq8-refit-480m-20260913`; its weights cannot resume unchanged
after changing the categorical vocabulary and output classifier.

Preparation: `outputs/imagenet-compact-rq32k-stage2-20260913`.
Reconstruction study: `outputs/imagenet-compact-rq-study-20260913`.
The first run candidate was prepared and verified, but the user rejected its
4.865889 reconstruction FID. It has not been launched. The subsequent
[64k fallback](imagenet-depth-compact-rq64k-2026-09-13.md) achieved 4.383626
and was selected for the fresh restart under the unchanged fidelity limit.

## Representation and fitting

Each of 16,384 frozen ImageNet dictionary atoms receives its own negative and
positive coefficient. Code zero is the unique zero vector; other codes are
`1 + atom_id * 2 + coefficient_bin`. The vocabulary has 32,769 entries and each
image retains 8x8x4 = 256 tokens. Earlier residual contributions are never refit
during encoding. Encoder, decoder, and dictionary directions remain frozen.

Coefficient fitting uses 4,096 randomly selected training images from view zero
of the existing encoder cache, excluding the 128 temperature-calibration images.
Initialization fits two shared signed levels to continuous matching-pursuit
coefficients. Eight atom-specific Lloyd passes use the Church fitting procedure,
with four prior pseudo-observations per entry. A four-level alternative was also
screened. The exact indices, levels, source hashes, and fitting traces are saved.

## Reconstruction evidence

The screening comparison uses identical 4,096 validation images, decoder,
Inception model, FP32 arithmetic with TF32 disabled, and original-image reference
statistics. It reproduces the current shared-eight control's prior result.

| Construction | Vocabulary | Matched rFID-4096 | Latent MSE |
| --- | ---: | ---: | ---: |
| Current shared eight levels | 131,073 | 11.155910 | 0.012833 |
| Two levels per atom | 32,769 | 11.936511 | 0.018599 |
| Four levels per atom | 65,537 | 11.445910 | 0.015143 |

Unlike the Church screen, this ImageNet screen trades reconstruction fidelity
for the smaller vocabulary. These are reconstruction scores, not generation
scores. Full 50,000-image evaluation uses the previously verified full original
validation reference. The compact tokenizer measures **4.865889 rFID**, versus
**4.407900** for the current shared-eight tokenizer and **4.215117** for original
OMP4. This is +0.457990 versus the current tokenizer and +0.650772 versus the
original tokenizer. The source, decoder, dictionary, and full matched reference
are identical. Full evaluation confirms the backbone is unchanged.

## Training and cache changes

The prior retains width 1536, 12 spatial layers, four depth layers, 24 heads,
1,000 class labels, and fixed physical codeword conditioning. Reducing the
classifier changes total parameters from 657,198,081 to **506,104,833**.
The released 100-epoch ImageNet optimization recipe, global batch 2,048,
microbatch 128 per GPU, four accumulation steps, and FP16 model computation
remain in use. Tokenizer geometry and soft cross entropy use FP32.

ImageNet's new training-target temperature is **0.125**. On the same disjoint
128 training images used by the original calibration, sampled/hard latent MSE
is 1.010133, compared with the original RQ control's 1.008906. Temperature 0.25
gives 1.052867 and exceeds the existing control-plus-0.02 selection rule. The
original control is reused with its exact report and checkpoint hashes recorded;
fresh encoder output is checked against the selected cached image view.

Training only consumes unquantized latents and labels before regenerating
stochastic targets. Its cache reader now optionally skips hard IDs, so the new
run can link both existing training views without copying 162 GiB or rebuilding
unused training-token arrays. Validation hard IDs are regenerated with the new
codebook during the full reconstruction audit. Old token arrays remain attached
to the old run. The sampler obtains the coefficient count from the last tensor
dimension, supporting both shared and atom-specific levels.

The ten-class 10x8 preview and selected temperature-0.9 sampler are retained.
Original-sampler generation FIDs and selected-sampler FIDs remain distinct.

## Verification and operation

Twenty-two focused tests passed across ImageNet adapters, compact quantization,
chunked loss, cache reading, sampling, grids, and fidelity validation. The new
cache test exercises training without hard-token files; the new sampler test
checks atom-specific coefficient dimensions.

A four-GPU preflight completed three updates with finite model/optimizer states
and no AMP skips. Strict checkpoint reload, class conditioning, and decoded
sampling passed. Dense and chunked soft CE both measured 10.5933704. Peak
training allocation was 36.6376 GiB per GPU versus the old run's 61.2745 GiB.
The preflight shared GPUs with the existing job and reconstruction audit, so its
wall-clock throughput is not an isolated performance measurement.

The switch script validates the prepared run before requesting SIGTERM on the
identified old rank-zero process. It waits for the checkpointed pause, verifies
the full state, and preserves a second hard link to the old checkpoint before
launching the new run. Production uses fresh seed-zero weights and never loads
the preflight checkpoint. The current launcher's reconstruction-drift limit is
+0.20 relative to the original tokenizer's matched 4.215117 rFID.
