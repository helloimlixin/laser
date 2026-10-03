The slowdown after enabling corrected geometry came primarily from FlashAttention
kernels applied to two-token coefficient sequences. Profiling the actual
step-7,000 Church checkpoint found 55.9% of coefficient-head CUDA time in
attention backward and 15.8% in attention forward. Matrix multiplications
accounted for a much smaller share. The four candidate conditionals multiplied
the cost of the poorly suited short-sequence kernel.

The new training backend evaluates the same causal attention directly. For
query/key/value rows 0 and 1, the result is `y0 = v0` and
`y1 = sigmoid((q1·k0 - q1·k1)/sqrt(d)) v0 + (1 - sigmoid(...)) v1`.
Half-precision inputs use FP32 score and weighted-value accumulation. PyTorch
compiles this parameter-free tensor operation into fused kernels. It adds no
model parameters or checkpoint keys. It runs only for two-token, causal,
gradient-enabled training with zero attention dropout. Residual and MLP
dropout retain their original calls and independent randomness. Evaluation,
sampling, KV caching, and nonzero attention dropout retain the original SDPA
or cached path. Candidate count, conditional geometry, coefficient targets,
loss weights, global batch, optimizer, schedule and all sampling settings are
unchanged.

The option is `--compound-pair-attention compiled`; `sdpa` retains the previous
implementation and `eager` runs the equivalent formula without compilation.
It is enabled only on the coefficient micro-transformer's attention blocks.
The spatial and depth transformers retain their existing attention kernels.
Compilation is lazy and cached under `/mnt/laser-church/torchinductor`.

Validation before deployment:

- Double-precision output and gradient comparisons against causal SDPA pass.
- The entire corrected compound loss and all model gradients match within
  numerical tolerance with residual dropout both disabled and at 0.1; the
  tests verify identical RNG state and unchanged checkpoint keys.
- Evaluation, no-grad sampling, and attention-dropout cases bypass the new
  kernel. The archived FFHQ and full-pair causality tests still pass.
- The focused regression suite passes 26 tests; the final inference guard
  passes all four pair-attention tests.
- On the real checkpoint in BF16, eager versus SDPA had identical scalar loss,
  coefficient-logit MAE 0.000754, and relative gradient L2 difference 0.00135.
  Compiled versus eager differed by 0.000000477 in loss and 0.000167 relative
  gradient L2. These are reduced-precision numerical differences, not bitwise
  trajectory equivalence.
- Coefficient-head benchmarks at batch 8 improved from approximately 122 ms
  to 52 ms with the eager formula and 45 ms with the fused formula. Full-model
  batch-16 forward/backward/AdamW steps improved from 322–326 ms to 161 ms.
  Peak allocated memory fell from 13.73 to 12.49 GiB in that test. These were
  paired tests on a GPU shared with training, so production throughput must be
  verified separately.
- A four-H100 checkpoint-resume canary completed two optimizer steps at batch
  16 per GPU, global batch 256. Every model and optimizer tensor was finite;
  peak allocated memory was 15.50 GiB per rank. Test weights are discarded.

The production handoff checkpoint is step 8,037, retaining all four RNG
streams and the original scheduler and optimizer. Best FID at handoff is
18.4209, epoch 15. The same scratch run ID and existing online checkpoint slots
are retained. The faster runtime is `/mnt/laser-church/runtime-fast-pair`;
the prior runtime remains intact. Frozen source, checksum manifest, patch,
preflight, handoff receipts and benchmark evidence are in
`outputs/church-ffhq-compound-350m-4h100-jointgeom-scratch-20260923/`, with detailed
measurements under `performance/`. The restart helper follows the selected
runtime in `resume-runtime.json`.

The source recipe is
`configs/stage2/lsun-church-ffhq-compound-350m-4h100-jointgeom-fast.yaml`.
The unchanged 50,000-image FID evaluation still pauses updates every five
epochs; this optimization targets training computation.

Production verification confirmed a median 1,096.54 images/s across 33 steady
ten-update windows at steps 8,080–8,400, versus 488.54 images/s before the
kernel change: a 2.2445× speedup. Both measurements use four H100s, batch 64 per
GPU, global batch 256, and the fully active corrected geometry objective.
Median update time is 0.23346 seconds; checkpoint, preview, and FID overhead
are additional. The final four-GPU preflight also used batch 64 per GPU and
peaked at 34.25 GiB allocated per rank, with finite model and optimizer state.
Production resumed step 8,037 rather than adopting either canary's weights.
The live receipt is `fast-pair-live-verified.json` in the run directory.

The subsequent [checkpoint I/O repair](church-checkpoint-io-2026-09-23.md)
resumed the same model at step 20,000 in
`/mnt/laser-church/runtime-local-checkpoints`. It preserves this attention
optimization and changes only local checkpoint-cache selection after an
immutable file's ctime is finalized by shared storage.
