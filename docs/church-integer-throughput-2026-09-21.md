# Integer-stage throughput fixes, 2026-09-21

Applies to `church-laser-ft3best-integer-scratch90-h200x8-20260921` in
`outputs/church-integer-ft3best-scratch90-20260921`. Optimized source, benchmarks,
preflights, and continuation supervisor are isolated under
`optimizations/throughput-v1`. The original source snapshot remains available.

## Changes

- Batch stochastic target computation in 2,048-position chunks, up from 128.
  Internally generated token IDs already satisfy vocabulary bounds, so target
  construction embeds these IDs directly instead of repeatedly synchronizing
  with the CPU to check bounds. Public token embedding retains its validation.
- Use fused Triton soft-target cross-entropy forward/backward with FP32
  probabilities, reductions, and gradient arithmetic. Preserve nonunit target
  mass in the gradient, as the previous implementation does. Address arithmetic
  uses 64-bit indices: a real full batch has more than 2^31 elements.
- Increase FID generation batches from 100 to 512 images per GPU. Decode and
  Inception batches remain 64. Sample count, real reference, original Inception,
  top-k 1,400, top-p 1, and sampling temperature 1 are unchanged.
- Save the full checkpoint once after each epoch's evaluation. When that model
  sets the best FID, hardlink the just-written immutable checkpoint as the best
  checkpoint. Subsequent last-checkpoint writes use atomic replacement, so the
  retained best state cannot change. This also gives W&B identical contents
  for last and best on selection epochs, avoiding duplicate large transfers.
- Remove the redundant immediate save/upload on resume. Record validation
  metrics without writing a separate validation-only model. Requested retention
  remains full last and full best-FID checkpoints, uploaded and verified online.
- Report individual training-update time, images/sec, checkpoint write time,
  and complete FID evaluation time.

The prior epoch remains recoverable until a new epoch finishes evaluation and
its full checkpoint commits. An interrupted evaluation can therefore require
repeating the most recent epoch. Cooperative interruption still saves the
current optimizer boundary. No asynchronous checkpoint writer was introduced.

## Numerical and performance checks

Benchmarks use the actual fitted integer book, cached Church latents, current
stage-2 weights, and H200 hardware. FP16 model autocast, FP32 book geometry, and
disabled TF32 are preserved.

| Operation | Previous | Optimized |
|---|---:|---:|
| Targets, 256-image GPU batch | approximately 183 ms | approximately 76 ms |
| Soft CE forward + backward, full GPU batch | approximately 73 ms | approximately 9.5 ms |
| Sampling + decode + Inception, images/sec/GPU | 46.4 at batch 100 | 89.4 at batch 512 |

Loss and gradient checks cover FP32 and FP16 logits, several logit scales,
one-hot targets at the final vocabulary index, and nonunit target mass. The
fused kernel is also benchmarked at the actual 256 x 8 x 8 x 4 x 32,769 shape.
Deterministic code targets match the previous implementation. With the original
128-position chunking, stochastic targets and codes match bit for bit. Larger
chunks preserve the conditional probability calculation but regroup RNG draws;
future training trajectories and evaluation draws are not bitwise identical.
The scalar temperature remains 0.125 and the stochastic training objective is
unchanged.

Full eight-GPU preflight resumes the paused checkpoint at optimizer step 1,045,
trains through the epoch-17 boundary to step 1,065, and saves a full checkpoint.
A second preflight resumes that optimized checkpoint through step 1,075. Both
exercise cached generation and decoding on every rank. Production resumes the
original step-1,045 anchor, not either preflight's weights.

The original paused checkpoint at step 1,045 and its selected best FID were
verified online before continuation. Learning-rate, optimizer, scaler, all
eight RNG states, sampler position, selected tokenizer, and resident latent
cache are restored. The run still ends at epoch 90, samples every 500 successful
updates, and computes 50,000-image FID at the original evaluation cadence.

Files: `kernel-benchmark.json`, `sampling-benchmark.json`, `preflight/`,
`resume-preflight/`, `runtime-manifest.json`, and `train.log` under the optimized
runtime; shared production metrics/checkpoints remain under the original
`train/` directory. The optimized supervisor publishes overall status in the
original experiment's `status.json` after the previous process has exited.

Production verification: the same online run resumed from step 1,045. Median
observed full-batch training throughput is 4,896 images/sec versus 3,425
before the change (approximately 43% higher throughput). Epoch-20
FID on 50,000 samples is 15.041147930491661; that evaluation completed in
102.71 seconds. Full best/last retention and one-write-per-epoch behavior were
verified after a later epoch replaced last.pt; the epoch-20 best state remained
unchanged. These records are in `verification.json` and
`checkpoint-retention-verification.json` under the optimized runtime.
