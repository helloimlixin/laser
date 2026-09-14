# ImageNet compact 64k reconstruction fallback

The later [original-recipe audit](imagenet-rq-original-recipe-audit-2026-09-13.md)
corrects training to fresh ImageNet augmentations and the published 480M
generation sampler (temperature 1.0, top-k 256, top-p 0.95). Details below about
two cached views and `original_t09` describe the initial launch configuration.

This candidate uses 65,537 output classes and 556,469,249 prior parameters,
with four coefficients per dictionary atom at each of four residual stages.
It retains 256 tokens per image. The frozen encoder, dictionary directions,
and decoder are the same as the current ImageNet run. The representation and
training adapter are described in
[the 32k experiment](imagenet-depth-compact-rq32k-2026-09-13.md).

The user rejected the first 32k candidate's reconstruction loss and requested
an improvement. The best subsequent 32k candidate achieved full matched rFID
4.501609 versus the current tokenizer's 4.407900, still exceeding the existing
limit of 4.215117 + 0.20. The 64k fallback prioritizes this reconstruction limit
while reducing output classes by half. Production switching requires a full
matched evaluation and the existing quality gate; the tolerance is unchanged.

The selected 64k candidate achieved full matched rFID **4.383626**, improving
on the current tokenizer by **0.024273**. Its drift from original OMP4 is
**+0.168509**, which passes the existing +0.20 limit. All 50,000 validation
images and the original reference statistics are used. The frozen backbone
hash is unchanged. This is reconstruction fidelity, not a claim about the
new prior's generated-image FID before training.

## Fitting and selection

Coefficient fitting uses 8,192 training images, excluding the 128 temperature
calibration images. Each residual stage has four signed levels per atom,
initialized from continuous matching-pursuit coefficients and refined with
constrained Lloyd updates using four prior pseudo-observations per entry.

| Candidate | Matched rFID on 4,096 validation images |
| --- | ---: |
| Current shared-eight tokenizer | 11.155910 |
| Stage-specific four-level tables, four passes | 11.241665 |
| Stage-specific four-level tables, eight passes | 11.259201 |
| Four passes, global gain 0.95 | 11.222781 |
| Four passes, global gain 1.05 | 11.254301 |
| **Selected: four passes, global gain 1.10** | **11.196998** |

A follow-up screen changed either the first two or last two stage gains to
1.00 or 1.20. Their rFIDs were 11.300136, 11.234823, 11.327767, and 11.218983,
respectively; none improved the selected candidate.

The global gain is selected using validation reconstruction FID. Coefficient
Lloyd fitting itself uses training images only. All screens use the same
images and reference statistics, FP32 arithmetic, and disabled TF32.

Selected codebook SHA256:
`8387618fb34020345198928e7cf3c8222976012edc91bc32915265002b11b1ab`.
The frozen tokenizer checkpoint SHA256 is
`dd28db9306d526bdc9fbf8016403e106c4f317fd8f5c04792f0953e53310fdab`.

Full evaluation artifacts are under
`outputs/imagenet-compact-rq-depth64-full-20260913` and run preparation is under
`outputs/imagenet-depth-compact-rq64k-stage2-20260913`.

## Training checks

Target temperature is 0.0625, with sampled/hard latent-MSE ratio 1.007795.
This meets the original RQ control's 1.008906 + 0.02 limit. Temperature 0.125
gives 1.039045 and fails that same calibration rule.

Twenty-seven focused tests passed. They cover explicit separate-codebook
equivalence at both vocabulary sizes, stochastic targets, cumulative
commitment, class conditioning, cached sampling, sparse coefficient fitting,
cache reuse, and the fidelity gate.

The four-GPU preflight passed three optimizer updates with finite gradients
and zero AMP skips. Strict checkpoint reload, finite saved model/optimizer
states, class conditioning, and sampler decoding passed. Peak GPU allocation
was 43.6236 GiB versus the existing run's 61.2745 GiB. This comparison uses the
same microbatch and accumulation settings. Throughput during preflight is not
an isolated benchmark because the GPUs were shared with the existing prior
and reconstruction evaluations.

The proposed production prior starts from fresh seed-zero weights with the
existing width-1536 architecture, 12 spatial layers, four depth layers, 24 heads,
1,000 classes, AdamW and 100-epoch cosine schedule. Effective batch size remains
2,048 on four H200 GPUs. The selected `original_t09` sampler, original-temperature
FID series, and ten-class 10x8 sample grids are retained. Existing encoded
training views are reused; changed validation hard IDs are regenerated.

## Production switch

Fresh run:
[imagenet-depth-compact-rq64k-scratch-20260913](https://wandb.ai/helloimlixin-rutgers/laser/runs/imagenet-depth-compact-rq64k-scratch-20260913).
The old `imagenet-rfid421-rq8-refit-480m-20260913` run paused at optimizer step
9,202, epoch 14, consumed batch 1,752. Its complete checkpoint, optimizer,
scheduler, scaler, four rank RNG states, cache metadata, and configuration are
preserved under the new base's `previous-run/last.pt`. All saved model and
optimizer tensors are finite. Checkpoint SHA256:
`4d6a54b68647da567d8efac301f4000eea01a992bbff76a5ff8200e0a648375e`.

The fresh production weights match the verified seed-zero initialization:
`8b4223cea46dd26c5e135ea95491a3df23015ce8961795068727a457581e0064`.
No prior weights or optimizer state were loaded. Production initialization
confirmed four ranks, 65,537 classes, 556,469,249 parameters, target temperature
0.0625, and effective batch 2,048. The launcher preserved and checked 86 source
files in the new run's source snapshot.

Startup verification completed after 25 successful production updates, with
zero AMP skips and a complete four-rank checkpoint at step 25. Peak production
GPU allocation was 43.7153 GiB, about 29% below the old run's 61.2745 GiB.
Updates after startup took about 2.28 seconds for 2,048 images. W&B reported
the new run running and the previous run finished. The durable verification
receipt is `outputs/imagenet-depth-compact-rq64k-stage2-20260913/launch-verification.json`.
