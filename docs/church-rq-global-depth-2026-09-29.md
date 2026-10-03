# Global depth-major compound-token RQ, LSUN Church

Run: https://wandb.ai/helloimlixin-rutgers/laser/runs/church-rq-globaldepth-20260929

The new stage 2 is initialized from scratch. Every generated atom/coefficient pair can attend to every individual pair at every spatial location in earlier sparse levels, plus earlier raster locations in the current level:

`p(Q) = product_d product_i p(q[i,d] | q[:, :d], q[:i, d])`.

The sequence has 256 events: all 64 locations at depth 0, then depths 1, 2, and 3. A flat 28-block, width-1024, 16-head causal transformer replaces the old spatial/depth hierarchy. Its inputs are shifted individual physical pair vectors with learned spatial/depth positions and BOS. It has 420,205,569 parameters and one 65,537-way joint classifier. Generation maintains a continuous preallocated KV cache across all events. External code tensors retain the `[B,8,8,4]` layout expected by the frozen decoder.

The previous RQ was autoregressive in a different order. This change expands access to individual earlier pairs; it does not establish that exposure bias or the FID floor is solved.

## Preserved recipe

- Frozen LSUN Church tokenizer, encoder-latent cache, and joint codebook. No refitting, reward, contrastive loss, or additional geometry loss.
- 16,384 atoms, four existing physical coefficient levels per atom shared across depths, and a zero token. Nonzero joint ID is `1 + 4 * atom_id + coefficient_level_id`.
- Full-vocabulary stochastic residual teacher at temperature 0.125 in squared physical distance. Each draw is a complete atom/coefficient vector; that exact vector is subtracted before the next depth. The objective remains full joint soft cross entropy.
- Original fresh joint-RQ optimizer recipe: AdamW, learning rate 0.0005, betas `(0.9,0.95)`, epsilon `1e-8`, weight decay `0.0001`, clipping at 1, no warmup, cosine decay through 18,600 updates.
- Eight H100 GPUs, global batch 2,048, microbatch 64, accumulation 4, 300 epochs of 62 updates. Teacher and model RNG streams are independent and saved for all eight ranks.
- Sampling uses complete joint tokens at temperature 1: the union of the original top 1,400 support and a nucleus retaining at least 95% of full-vocabulary probability mass.
- 64 preview images every 200 updates; official 50,000-image FID every 620 updates and at the end. Generation batch is 1,024 per GPU; FP32 decoder and feature batches are 32. The reference and eight logical evaluation streams are fixed.
- Full local checkpoints every 200 updates. First remote LAST at step 200, then verified remote and durable LAST/BEST at each FID evaluation and at completion. A new BEST requires a measured FID.

## Validation and measurements

CPU and eight-GPU checks cover causal ordering, absence of current/future leakage, influence from later spatial locations in earlier levels, full-forward/cache parity at all 256 events, joint sampling order, teacher math, and all parameter gradients/optimizer states. The codebook is identical to the earlier joint RQ codebook. The teacher, loss, and probability-filter functions are preserved.

A disposable fixed-64-image learning probe reduced mean soft-target KL from 10.693 to 2.649 over 128 updates. Shuffling lower-level history worsened later-depth cross entropy. This is a training-set learnability check, not evidence of improved generalization or FID; the shuffle includes same-site lower-level context.

The production-size training benchmark measured 1.367 seconds per update and 38.30 GiB peak allocation with microbatch 64. Microbatch 128 gained only 3.48% throughput and used 67.69 GiB, so 64 was retained.

| Generation batch per GPU | Aggregate images/second, eight GPUs | Peak allocated GiB per GPU |
| --- | ---: | ---: |
| 256 | 531.38 | 9.58 |
| 512 | 740.64 | 17.33 |
| 1,024 | 889.02 | 32.84 |

These are generation-only capacity measurements, excluding decoder, Inception, covariance computation, and publication. All batch sizes completed the actual 256-event sampler and an FP32 decode probe. They are not measurements of full FID turnaround or a speed comparison against the previous architectures.

The earlier native RQ generation preflight measured 306.68 images/second on one H100 at batch 1,024 using top-k 1,400. The new eight-GPU benchmark corresponds to 111.13 images/second per GPU at the same batch size, approximately 2.76 times lower throughput. This is only a rough comparison: GPU count and filtering policy differ, and no matched coverage-95 generation-only baseline was recorded. Native RQ performs approximately 2,560 cached block-token updates per image (24 spatial blocks for 64 positions plus four depth blocks for 256 draws); the new model performs 7,168 (28 blocks for all 256 draws) and attends to longer histories. Operation counts are not a measured latency ratio.

Earlier complete 50,000-image evaluations on eight GPUs took approximately 103–104 seconds for coverage-95 native RQ and 177–178 seconds for the shared-physical DC model. Those include generation, decoding, features, and FID computation, and cannot be compared directly to the new generation-only rate. A new full-pipeline measurement requires its first scheduled FID evaluation.

Resume exactly restored model/Adam tensors, the data cursor and schedule, and all 16 model/teacher RNG streams. A 3+1 versus uninterrupted 4-update comparison was not bitwise identical: maximum subsequent model difference was `1.1920928955078125e-7`, and maximum Adam difference was `8.731149137020111e-11`; logged losses and gradient norms matched. The raw bitwise failure is preserved alongside the separate numerical-tolerance audit. Bitwise continuation is not guaranteed.

## Recovery and provenance

Runtime: `/tmp/laser-church-rq-globaldepth-20260929`.

Durable source, receipts, and published checkpoints: `/workspace/Projects/laser/outputs/church-rq-globaldepth-20260929`.

Production plan and source are frozen and hash-checked by the worker/controller. The final configuration gate records the change from evaluation batch 256 in the initial pilots to independently tested batch 1,024; training code and training configuration stayed unchanged. Recovery assets include the full encoder-latent cache, physical joint codebook, tokenizer, official FID reference, and plan. Training source and recovery artifacts are verified against their remote manifests.

The preceding DC experiment was retired after preserving full LAST and BEST at step 1,860, FID 73.74568942042103. Both preserved checkpoints have SHA-256 `6a1602b894a9fb95043707b9ee3a14894946ec5405368d2a8e92d3ffd741d719`. Its training target was not marked completed. Its source, recovery assets, checkpoints, and retirement lineage were verified remotely and durably before handoff.

Production was launched with no resume checkpoint and an empty optimizer. All eight ranks reported identical initial parameters and the intended fresh recipe. Ongoing status belongs to the run and runtime receipts rather than this static launch note.
