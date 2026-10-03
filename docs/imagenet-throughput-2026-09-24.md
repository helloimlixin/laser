# ImageNet Stage 2 throughput audit

Run: https://wandb.ai/helloimlixin-rutgers/laser/runs/imagenet-rfid421-ffhq-compound-1400m-7h100-20260923

The reported BF16 training setting was incorrect for the original runtime.
The outer training loop opened a BF16 autocast context, but passed
`amp=False` to the RQ backbone. The backbone's nested autocast context
disabled it. A hook on the real checkpoint's first attention projection
confirmed FP32 output. This behavior is also in the archived FFHQ recipe;
the optimization explicitly changes training arithmetic while retaining its
architecture, autoregressive factorization, targets, loss and optimizer.

For the resumed session from steps 3,180 through 12,700, W&B timestamps
show 19,192,320 training images in 26,185.999 seconds: 732.923 images/second
including pauses, versus a median reported compute window of 928.358.
Intervals spanning the epoch-10 and epoch-15 FID evaluations took 1,263
and 1,318 seconds. Full checkpoints and previews account for additional
pauses. The current model has 1,457,980,928 parameters, larger than the
350M FFHQ model, so its baseline throughput cannot be compared directly.

Candidate changes:

- Explicit BF16 model execution, with FP32 model parameters and optimizer
  state. Loss terms that explicitly use FP32 retain that behavior.
- The existing compiled two-token causal attention implementation, enabled
  only for coefficient attention during gradient-enabled training.
- Exclude cuDNN SDPA within the BF16 training context after an isolated
  large-batch backward probe failed. Evaluation retains its prior backend
  selection. PyTorch documents backend-dependent numerical differences:
  https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.scaled_dot_product_attention.html
- Training microbatch 96 per GPU and three accumulation steps, retaining
  global batch 2,016 and the same learning-rate schedule.
- FID generation batch 128 and preview batch 64. Retain 50,000 class-balanced
  FID samples every five epochs, the original reference statistics, and all
  sampling temperatures/top-k/top-p settings.
- Periodic full recovery checkpoints remain every 100 optimizer updates;
  full epoch saves coincide with the five-epoch FID cadence. Best FID and
  latest checkpoints continue to upload online.

The real step-12,700 checkpoint was benchmarked on GPU 6 while production
continued its epoch-20 evaluation. These interleaved measurements are not
isolated production throughput:

| Training microbatch | Arithmetic / coefficient attention | Forward + backward, seconds | Peak allocated GiB |
|---|---|---:|---:|
| 48 | Legacy FP32 / SDPA | 0.669 | 39.21 |
| 48 | BF16 / compiled pair attention | 0.413 | 28.05 |
| 96 | BF16 / compiled pair attention | 0.748 | 47.50 |

These tests exclude AdamW and distributed communication. At batch 48 the
fixed-seed loss was 7.438916 for legacy FP32 and 7.439104 for BF16; gradient
norms were 0.640667 and 0.641039. All gradients remained finite. The frozen
checkpoint and all benchmark updates remain separate from production.

Sampling plus decoding at batch 128 processed 19.98 images/second on the
shared GPU, versus 3.85 at batch 16. The batch-16 result includes cold
startup, and Inception extraction is excluded. Batch 256 was slower than
128, so 128 was selected for the full-allocation check.

The candidate runtime passed 26 focused checks covering the attention
formula and gradients, dropout RNG behavior, archived FFHQ objective,
full atom/coefficient causality, cached generation and checkpoint storage.
The numerical update is not bitwise equivalent to FP32 training.

Detailed evidence is retained under
`outputs/imagenet-rfid421-ffhq-compound-1400m-7h100-20260923/throughput-20260924/`.
Deployment and measured production throughput are recorded there after the
seven-GPU checkpoint-resume check completes.

The previous runtime completed epoch-20 FID at 53.692258 (IS 26.526068),
then stopped at update 12,701. The handoff checkpoint has optimizer state,
scheduler step 12,701, all seven RNG streams, and epoch-20 batch cursor 6.
Its persistent copy completed in 174.77 seconds. The completed trainer was
released from its W&B finish/upload wait after checkpoint validation;
resuming the same run requeues the fixed online checkpoint slots.

The seven-H100 canary passed 20 optimizer updates from the exact handoff
checkpoint. The data cursor remapped from six batch-48 microbatches to
three batch-96 microbatches. It restored the checkpoint's optimizer,
scheduler and all rank RNG streams, verified BF16 logits, and peaked at
69.22 GiB allocated per rank. The steady ten-update window reached
1,615.479 images/second. A 128-image generation/decode preview also passed.
Canary weights were discarded; production resumes the original step-12,701
checkpoint with detached launcher PID 19047 and the same online W&B ID.

Production verification: the first seven steady ten-update windows reached
median 1,615.394 images/second, 1.7401× the prior 928.358. W&B confirms
`training_precision=bf16`, `compound_pair_attention=compiled`, batch 96,
three accumulation steps, global batch 2,016 and BF16 logits. The first
production checkpoint serialized locally at step 12,800 in 11.32 seconds;
training continued at approximately 1,615 images/second during its
background persistent copy. The 100-update window from 12,720 to 12,820,
including serialization, averaged 1468.146 images/second. This window does
not include an FID pass; a full production evaluation at batch 128 is due
at epoch 25. The epoch-20 best-FID file's online MD5 matches its local
checkpoint; the fixed latest file continues to transfer asynchronously.

The step-12,800 persistent checkpoint completed in 83.70 seconds while
training continued. Reload verified all 1,457,980,928 model values and
2,511 optimizer tensors were finite; model parameters remain FP32. All
seven RNG states, epoch-20 batch cursor 300 and scheduler step 12,800
were present. Production subsequently continued beyond step 12,920 at
approximately 1,617 images/second. Detailed receipts accompany this report.
