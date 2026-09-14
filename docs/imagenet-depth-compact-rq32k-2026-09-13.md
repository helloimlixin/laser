# ImageNet stage-specific compact RQ

The selected compact candidate has **32,769 output classes**, **506,104,833
prior parameters**, and **256 tokens per image**. Its coefficient tables depend
on both dictionary atom and residual stage. The source encoder, dictionary
directions, and decoder remain frozen.

Preparation is under
`outputs/imagenet-depth-compact-rq32k-gain110-stage2-20260913`.
Full reconstruction evaluation is under
`outputs/imagenet-compact-rq-depth-gain110-full-20260913`.
The full matched rFID is **4.501609**. This improves substantially over the
rejected compact candidate's 4.865889, leaving **+0.093709** relative to the
current tokenizer. It is **+0.286492** relative to original OMP4, so the existing
+0.20 limit still prevents launching this candidate. The subsequent
[64k fallback](imagenet-depth-compact-rq64k-2026-09-13.md) achieved 4.383626,
passed the unchanged quality gate, and was selected for the fresh restart.

## Representation

Code zero always denotes the zero vector. At residual stage `d`, a nonzero ID
`1 + atom * 2 + bin` reconstructs
`dictionary[:, atom] * levels[d, atom, bin]`. There is one negative and one
positive level per atom at each of the four stages. The already-known stage
selects the table, requiring no extra output classes or tokens. Tables are
fixed during encoding and prior training. Previous residual contributions are
never refit during encoding.

The frozen dictionary tensor has shape `[256, 16384]`; the level tensor has
shape `[4, 16384, 2]`. This is an extension of the Church compact construction:
Church shares its coefficient table across residual stages. The ImageNet
extension uses separate tables to represent the different residual magnitudes.
The transformer's body and depth head both receive the correct physical vector
for each token's stage. Its architecture and class conditioning are unchanged.

## Fitting and selection

The tables are fitted on 8,192 training images from the existing FP32 view-zero
cache, excluding the 128 temperature-calibration images. Initialization fits
two signed levels to each stage's continuous matching-pursuit coefficients.
Four constrained Lloyd passes fit atom-specific levels at each stage, with four
prior pseudo-observations per entry. A global gain of **1.10** was selected in a
matched 4,096-validation-image reconstruction screen. Candidate selection uses
validation; coefficient Lloyd fitting itself uses training images only.

The screening study compared additional fitting passes, joint final-residual
least squares, training-image L1 plus VGG LPIPS fitting, shared 64k alternatives,
and gains 0.95, 1.05, and 1.10. The joint least-squares and perceptual refinements
did not improve the shared-table reconstruction FID. Stage-specific tables
provided the main improvement.

| Candidate | Vocabulary | Matched rFID-4096 |
| --- | ---: | ---: |
| Current shared-eight tokenizer | 131,073 | 11.155910 |
| First compact shared-table candidate, rejected | 32,769 | 11.936511 |
| Four levels per atom, shared across stages | 65,537 | 11.445910 |
| Stage-specific two-level tables, eight fitting passes | 32,769 | 11.372179 |
| **Selected: four passes and gain 1.10** | **32,769** | **11.265401** |

All screens use the same images, decoder, Inception model, FP32 arithmetic with
TF32 disabled, and original-image reference statistics. Full matched rFID uses
all 50,000 validation images and the previously verified original-validation
reference. The existing +0.20 limit relative to original OMP4's **4.215117** rFID
remains in force. The rejected first candidate measured **4.865889**; the current
shared-eight tokenizer measures **4.407900**. Gains 1.15 and 1.20 and a later
16-pass fit did not improve on the selected candidate in the matched screen.

Selected codebook SHA256:
`5579abb8519a84ae2bbd65cf218c5705745079e438eacfedec025e292e5b11b1`.
The frozen stage-1 checkpoint SHA256 remains
`dd28db9306d526bdc9fbf8016403e106c4f317fd8f5c04792f0953e53310fdab`.

## Training and verification

Training retains the ImageNet width-1536, 12-spatial-layer, four-depth-layer
architecture, 24 heads, 1,000 classes, 100-epoch cosine schedule, AdamW recipe,
and effective batch 2,048 on four H200 GPUs. It starts from fresh seed-zero
weights. Microbatch 128 and four accumulation steps are retained. The old full
checkpoint will be preserved before any production switch.

Target temperature is **0.125**. Its sampled/hard latent-MSE ratio is **1.021231**,
within the original RQ control's **1.008906 + 0.02** limit. Temperature 0.25
measures 1.116467 and fails the same rule. The control report is reused with its
checkpoint and report hashes recorded. Cached latents are checked against fresh
encoder output for the corresponding image view.

Both training augmentation views are reused as links to the existing encoder
cache. Training regenerates stochastic full-vocabulary targets and does not
load old hard IDs. All 50,000 validation hard-token tensors are regenerated with
the selected codebook. The original and selected generation-sampler FID series
and the ten-class 10x8 sample grids are retained.

Twenty-seven focused tests passed, including both two-level and four-level
stage-specific books. Stage-specific quantization, cumulative
commitment, and stochastic targets match explicit separate RQ codebooks.
Cached and full-forward conditional sampling match. Other tests cover loss
gradients, causal conditioning, sparse coefficient fitting, cache reuse, and
the fidelity limit. The new four-GPU preflight completed three optimizer updates
with finite gradients and no AMP skips; peak allocation was **36.6841 GiB/GPU**
versus **61.2745 GiB/GPU** in the current run. Strict checkpoint reload, finite
model/optimizer states, class conditioning, and sampler decoding all passed.
Dense and chunked CE differed by less than 0.000001. Its throughput is not an isolated
benchmark because it shared GPUs with the old run and reconstruction audit.
