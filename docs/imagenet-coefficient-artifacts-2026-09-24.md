# ImageNet stage-2 coefficient artifact repair

The user reported black and saturated-color blotches in the ongoing ImageNet
1.458B compound run. A frozen step-13,500 checkpoint reproduces them. Its
source weights, optimizer and RNG state remain anchored locally; production
subsequently stopped with a full step-13,771 recovery checkpoint.

On 32 randomly selected cached training images, depth-zero predicted endpoint
probability averaged 0.0105035, versus 0.0000026883 in the soft targets. Among
158 endpoint probabilities above 0.001, the archived distribution geometry
gradient increases 89.2%; in 74 cases it overwhelms the opposing classification
gradient. Correctly conditioning each candidate atom's coefficient reduces
these counts to 1.9% and zero. The original loss incorrectly uses the teacher
atom's coefficient conditional for every candidate dictionary vector.

A separate frozen autoregressive sample generated 32 endpoint events among
2,048 depth-zero sites. Replacing only those coefficients with their interior
conditional means removes the conspicuous neon holes in matched images.
Fully black pixels fall from 1.3029% to 1.0744%; natural dark image content is
also included in this statistic. This intervention is a causal diagnostic,
not the production sampler or evidence of an improved generation FID.

Full and cached FP32 predictions agree across all 256 events of one generated
history: maximum atom and coefficient logit differences are 1.812e-5 and
1.764e-5. This check does not indicate a cached-autoregression mismatch.

The frozen decoder also exposes excessive coefficient target noise. Sixteen
seeded cache examples use the same true atoms for every construction:

| Coefficients | PSNR to continuous reconstruction | Relative latent MSE |
|---|---:|---:|
| Nearest bin | 71.353 dB | 0.000000887 |
| Normalized soft targets, T=0.5 | 17.681 dB | 0.307816 |
| Physical soft targets, T=0.125 | 35.819 dB | 0.003305 |

These are perturbation measurements against clean tokenizer reconstructions,
not dataset reconstruction FID or generated-image FID. Physical T=0.125 has
untruncated standard deviation 0.25 in dictionary coefficient units at every
depth. Bins, depth scales, and nearest-bin quantization are unchanged.

The continuation disables the faulty auxiliary geometry loss and calibrates
both coefficient soft labels and stochastic context tokens in physical units
at temperature 0.125. It retains full atom/coefficient autoregression,
compound tokens, selected-atom dictionary-vector conditioning, two coefficient
transformer layers, all model weights, optimizer state, LR schedule, global
batch 2,016, tokenizer, cache, sampling settings and FID protocol. It introduces
no endpoint mask or output-image correction. This intentionally departs from
the archived FFHQ loss and target-noise settings to address the measured bugs.

The new runtime logs predicted and target endpoint probability per depth.
It also synchronizes stop requests before recovery/preview collectives:
the earlier unsynchronized stop handler could let ranks enter different
collectives when a signal arrived around a preview boundary.

Twenty focused tests pass, covering target calibration, endpoint-gradient
direction, the preserved legacy objective, pair causality, attention and
resume behavior. An unrelated VAR schedule test is excluded because its
optional FoundationVision dependency is absent from this frozen runtime.

The original RQ-Transformer ImageNet 1.4B configuration uses global batch
2,048, LR 0.0005 and 100 epochs:
https://github.com/kakaobrain/rq-vae-transformer/blob/main/configs/imagenet256/stage2/in256-rqtransformer-8x8x4-1400M.yaml
This seven-GPU run keeps the near-equivalent global batch 2,016 and scaled
base LR 0.0004921875. Isolated full-AdamW/DDP probes of the pre-repair model
measured 235.5 images/s/GPU at batch 96 and 231.0 at batch 112; peak allocated
memory was 69.22 and 75.71 GiB. Batches 128 and 144 ran out of memory. Those
probes used one GPU per candidate and exclude seven-GPU communication; their
updated weights were discarded. Physical batch remains 96 with accumulation 3.

Evidence is retained in the run directory under `artifact-repair-20260924/`
and `batch-audit-20260924/`. Deployment and subsequent measured behavior are
recorded in their receipts. Long-run FID improvement remains to be established.

The seven-H100 test passed 20 updates and 64-image generation, peaking at
68.44 GiB allocated per GPU and reaching 1,665.6 images/s. Test weights were
discarded. Production resumed the exact step-13,771 checkpoint on the same
online run and continued at approximately 1,660 images/s. Reloading the
step-13,800 checkpoint verified all 1,457,980,928 model values and 2,511
optimizer tensors were finite, with seven RNG states and the correct saved
scheduler position. W&B confirms the physical target settings, geometry
weight zero, BF16 execution, and full pair autoregression.

A direct comparison uses 32 unmodified generations from step 13,771 and
step 13,800, after 29 repair updates. Labels, seed 240924, sampling settings,
CPU FP32 arithmetic and batch layout match; sampled contents can change
because the model weights changed. Depth-zero endpoint events fall from
36 to zero, fully black pixels from 0.8216% to 0.4445%, and out-of-range
decoder channels from 3.2367% to 2.1667%. These are early diagnostic results
on a small sample, not a full FID evaluation or proof all generation defects
are resolved. No post-hoc coefficient replacements are used in this comparison.
