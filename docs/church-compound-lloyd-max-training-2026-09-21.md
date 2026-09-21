This experiment trains a fresh compound-token stage-2 model using the [unclipped FP32 Lloyd–Max cache](church-compound-lloyd-max-2026-09-21.md) and the selected three-epoch Church stage-1 checkpoint. It does not reuse a transformer trained with uniform coefficient-bin meanings.

Online run: [church-laser-ft3best-compound-lloydmax90-h200x8-20260921](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-laser-ft3best-compound-lloydmax90-h200x8-20260921).

| Setting | Value |
| --- | --- |
| Training | 90 epochs, 82 updates/epoch, 7,380 optimizer updates |
| Hardware/batch | Eight H200s, 192 images/GPU, global batch 1,536 |
| Architecture | Church 350M RQTransformer preset; compound full-pair autoregression; two micro-transformer layers; depth-specific coefficient heads |
| Vocabulary | 16,384 atoms, 2,048 shared Lloyd–Max coefficient centers, four depths per 8×8 site |
| Cache | All 126,227 images; resident int16 atom IDs and unclipped FP32 coefficients; no training-time image encoder or OMP |
| Loss | `(1.5 × hard atom CE + soft coefficient CE) / 2.5`; no geometry loss |
| Coefficient targets | Original discrete physical-distance softmax, temperature 0.125; stochastic teacher tokens |
| Optimizer | Fused AdamW, LR 5e-4, betas (0.9, 0.95), weight decay 1e-4, gradient clipping 1 |
| LR schedule | Cosine to zero over 90 epochs, no warmup |
| Precision | BF16 model training, FP32 short depth attention, spatial FlashAttention; FP32 target construction |
| Sampling | Every 500 optimizer updates, 64-image preview; atom top-k 250, coefficient top-p 1.0, both temperatures 1.0 |
| FID | 50,000 samples at epochs 1, 10, 20, …, 90; original RQ-VAE Church reference |
| Evaluation memory | Generate 512/GPU, decode and compute Inception features in batches of 64 |
| Evaluation RNG | Independent seed `20261921 + rank`; training RNG restored after FID |
| Recovery/upload | Full last checkpoint every epoch and every 500 updates; retain best FID; bounded background uploads to online W&B |

The selected stage-1 checkpoint is frozen, SHA-256 `762c51a10267ed6fa55709ff0d6cf997940d21056b16c7a0a0399b3ebf93868d`. Cache SHA-256 is `38e3ee5e17fc6b6b8115ae97e5fdd1ed874935421f6a23b6b155bfb28a21291e`. The run loads its nonuniform centers directly from cache metadata, disables coefficient clipping, and stores the centers in each stage-2 checkpoint's configuration. Every rank checks FP32 cache input, exact center agreement, physical targets, and a frozen stage 1 during its first training batch.

The target-temperature check used 65,536 coefficients from the first 256 training images. At the unchanged temperature 0.125, mean physical target MSE is 0.060731 versus the uniform-cache baseline 0.062426, a 2.72% difference. This satisfies the declared 5% tolerance for retaining the baseline temperature. Mean discrete target entropy changes from 5.0456 to 5.3977 nats because the bin density changes. No Gaussian cell-mass replacement or other new loss was introduced.

The sampler follows the earlier successful sweep: coefficient top-p 1.0, replacing 0.85 from the original compound training evaluation. Thus historical training FIDs with coefficient top-p 0.85 are not an exactly matched sampler comparison. The separately reevaluated old best checkpoint's 50k FID 10.9382 used this top-p 1.0 sampler and is the relevant generated-sample reference. The previous cache's reconstruction improvement is small; this training experiment is needed to determine its effect on generation.

The integer comparison was cooperatively paused at optimizer step 3,722, epoch 60 plus two batches. Full optimizer, scheduler, AMP scaler, eight rank RNG states, and the best-FID checkpoint are preserved under `paused-integer/`. CPU resume validation passed for batch 256/GPU and world size eight. Its best FID at the pause is 11.28205, selected at epoch 50. A detached supervisor queues the original integer run, with the same W&B run ID, to resume after this new compound run finishes. Its final uploads continue in the previous process while the trial starts; that process performs no further training.

Experiment directory: `outputs/church-compound-lloydmax-scratch90-20260921`.

- `supervise.py` and `status.json`: durable trial supervision and queued integer continuation.
- `request.json` and `train/request.json`: requested and resolved production settings.
- `source-manifest.json`: hashes of 348 frozen execution/provenance files.
- `temperature-calibration.json`: target-noise and entropy measurements.
- `preflight/complete.json`: written only after an eight-GPU, 12-update scratch preflight, preview sampling, full checkpoint saving, and finite-weight checks pass. Production starts from scratch separately.
- `train/cache-codec-rank*.json`: per-rank cache and learned-bin checks.
- `train/token_cache_artifact.json`: verified selected stage-1 input artifact, newly uploaded Lloyd–Max cache, and training provenance. The old uniform cache is not recorded as the training input.
- `train/checkpoint-upload.json`: most recently verified online full last/best artifact. Transfers retain immutable hardlinks and the final transfer is drained before completion is recorded.
- `paused-integer/receipt.json` and `resume-validation.log`: preserved integer state and validated resume command.

Training is complete only when `train/complete.json` records epoch 90 and step 7,380 and the matching online checkpoint receipt verifies both last and best-FID files. The supervisor then resumes the integer comparison to its original 90-epoch endpoint.

Launch verification passed: the 12-update preflight completed, all eight production ranks verified the cache, W&B reported the run as online/running with 2,048 saved centers, and the new cache artifact was committed with matching file sizes and digests. Production reached approximately 5,040 images/second. The first 50,000-sample FID completed at epoch one (172.8617, an early training measurement); its full best checkpoint remains intact after later epochs replace `last.pt`. Both contain optimizer/scheduler state, all eight rank RNG states, and the learned centers. Production checkpoint transfers are queued asynchronously. See `launch-verification.json` and `online-launch-verification.json` for the checks and observed steps.
