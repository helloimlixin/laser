# Church execution tuning at fixed global batch 2,048

The selected execution layout measured **2,092.53 images/sec**, approximately
27% above the prior production rate of 1,650. It uses all five H100 NVL GPUs,
409–410 images per GPU, and one microbatch per optimizer update. The effective
batch stays 2,048 and the epoch still has 62 optimizer updates.

Three changes make the larger physical batch fit: compiled transformer blocks,
compact integer coefficient targets, and fused hard-label cross entropy with
FP32 reductions. BF16 AMP, FP32 gradient communication, AdamW states, learning
rate, clipping, residual dropout probability, fixed codes and sampling stay the
same. Compilation changes dropout masks and floating-point operation order;
this does not promise a bitwise identical training trajectory.

| Execution | Microbatch/GPU | Accumulation | Images/sec |
| --- | ---: | ---: | ---: |
| Prior production | 205 | 2 | ~1,650 |
| Compact targets | 205 | 2 | 1,665.62 |
| Compiled blocks and compact targets | 205 | 2 | 1,896.73 |
| Plus fused hard cross entropy | 205 | 2 | 1,946.49 |
| **Selected: larger physical batch** | **410** | **1** | **2,092.53** |

Each successful benchmark used isolated fresh weights, full forward/backward and
optimizer updates on all five GPUs, and a complete checkpoint write. The rate
excludes startup/compilation, checkpoint writes and FID evaluation. Timings use
the slowest rank after warmup. Peak allocated memory for the selected benchmark
was 84.37 GiB per GPU. A communication-protocol experiment had no meaningful
effect. Tiled Flash attention and vectorized tiny attention were slower and were
rejected; an untiled Flash trial failed finite-gradient checks and was rejected.

The numerical and resume checks passed:

- All 32,314,112 fixed coefficient labels matched original quantization exactly.
- Compact-target loss, gradients and held-out diagnostics matched exactly in tests.
- Fused loss maximum absolute error was 7.63e-6 and maximum relative gradient
  error was 1.66e-6 across FP32/BF16, both vocabulary sizes, masked logits and
  ordinary/extreme logit scales.
- Compiled block forward/gradient relative errors were 0.00115/0.00379 in a
  deterministic BF16 check with dropout disabled only for the comparison.
- All 62 optimizer batches retained exactly the same examples when accumulation
  changed from two to one; weighted gradients matched the global mean. Every
  training image is visited once, including the final 1,299-image batch.
- Five-GPU resume restored epoch 15/update 3,090, all 582 Adam states and five RNG
  records, then completed three finite updates with the expected cosine LR.
  These preflight updates are isolated and are not used for production training.

Production resumes the preserved epoch-15 state. Its scheduler position is 930,
its true optimizer count is 3,090, and LR is 0.0004969220851487838. Serialized
model, optimizer and RNG tensors are checked for exact preservation when only
execution metadata changes. The planned final true optimizer count remains
20,760 at epoch 300. The epoch-15 checkpoint is retained under
`before-kernel-tuning/`; the earlier batch-256 state is also retained.

The optimized production run sustained **2,086.93 images/sec** over five logged
intervals after startup and completed epoch 16/update 3,152. Its full checkpoint
passed finiteness checks with all 582 Adam states and five RNG records, scheduler
position 992, and the expected LR 0.0004964990092676259. The updated source
artifact is verified online as training-provenance version 1. Local and remote
checkpoint verification receipts are written to `kernel-production-local-accepted.json`
and `kernel-production-accepted.json`, respectively. Epoch 16/update 3,152 is
verified online as selected-checkpoints version 4, including file sizes and MD5
digests for the full last and best states.

All five GPUs report an active software power cap at **310 W**. NVIDIA reports
a supported maximum of 400 W, but changing the cap is denied by the host. This
is the fastest tested configuration under the available power limit, not a
claim of theoretical maximum throughput.

Runtime: `/mnt/laser-church/batch-throughput-20260926`.
Evidence and sources: `outputs/church-larger-batch-throughput-20260926`.
[W&B run](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-fixed-codes-joint-sequence-b2048-h100x5-resize-20260926).

To resume after stopping the owned training process:

```bash
python /mnt/laser-church/batch-throughput-20260926/launch_continuation.py --resume
```
