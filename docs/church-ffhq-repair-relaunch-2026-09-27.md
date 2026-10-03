This repair attempts to improve the recovered-FFHQ Church pipeline using calibrated coefficient noise, stronger dropout, and removal of the geometry surrogate. It keeps the frozen Church tokenizer, cached atom supports, coefficient scales, and compound transformer architecture. Stage 2 performs no re-encoding or coefficient refitting. The sequential residual-pair teacher remains experimental because its reconstruction validation has not justified production use.

The [new training run](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-ffhq-repair-norefit-b1024-8h100-20260927) is active on all eight GPUs. The [validation run](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-ffhq-repair-norefit-b1024-8h100-20260927-validation) contains the sampler table, noise grids, and complete verified preflight artifact v1 (107 files). This is a fresh stage-2 initialization and optimizer, not a continuation from the overfit recent checkpoint. It is a bundled repair attempt, so any eventual improvement will not isolate an individual recipe change.

| Setting | Repair |
| --- | --- |
| Hardware and statistical batch | Eight H100 80GB GPUs; 128 images per GPU; global batch 1,024 |
| Model | Recovered FFHQ compound model, 404,738,048 parameters; Church K=4, 16,384 atoms, 2,048 coefficient bins |
| Coefficient target temperatures, depths 1–4 | 0.1603424730, 0.1947608508, 0.1820431413, 0.1770278027 |
| Residual dropout | 0.2, including the coefficient micro-transformer |
| Geometry loss | Zero |
| Sampling | Atom temperature 1 / top-k 250 / top-p 1; coefficient temperature 1 / top-p 0.85 |
| Atom loss weight | 1.5, retaining the recovered classification normalization |
| Optimizer | AdamW 5e-4, betas (0.9, 0.95), weight decay 1e-4; gradient clipping 1 |
| Schedule | 200 epochs, 123 updates per epoch, 24,600-update cosine to zero; no warmup |
| FID | Official Church reference, 50,000 samples after epoch 1 and every 5 epochs |
| Generation execution | 128 samples per rank, decoding in chunks of 32; FP32 Inception batches of 500 with TF32 disabled |
| Monitoring | Fixed 300-image train and 300-image held-out probes after epoch 1 and every 5 epochs |
| Samples and checkpoints | 64-image grid every 200 updates; full last and best FID states uploaded with remote digest verification |

The temperature vector follows `tau_new[d] = 0.0625 * (old_scale[d] / current_scale[d])²`, matching the earlier successful Church teacher's interior physical coefficient variance. Finite bins and clipping mean this is an initial calibration rather than an exact match of the full teacher distribution. On the same 64 held-out images, the actual finite-bin expected latent noise energy falls from 9.4035% to 3.2775% of clean latent energy. Under shared inverse-CDF uniforms, decoded pixel MSE relative to nearest-bin reconstructions falls from 0.008770 to 0.003735. These are teacher-distortion measurements, not generator FID gains.

The target tests verify exact compatibility with the old scalar-temperature probabilities and RNG draws, FP64 agreement for the new depth temperatures, reproducible sampling, unchanged inputs, and rejection of invalid temperatures. The frozen cache, tokenizer, and reference hashes match the preceding experiment. Training targets keep the original cached supports; this does not introduce the RQ residual-conditioned atom teacher or establish that exposure bias is solved.

Sampling is selected on the frozen epoch-55 best checkpoint before the fresh launch. A four-way FID10k screen tests atom top-k 250/700 and coefficient temperature 1.0/0.9. The winning nondefault setting must improve over the original sampler on both independent FID50k confirmation seeds before adoption. Evaluation uses the same checkpoint, image count, generation batching, feature extractor, and official reference within each pair. The sampler comparison does not establish repaired-training quality.

The held-out monitor uses the same fixed stochastic coefficient history from the preceding diagnostic at every check. It records atom NLL and per-depth metrics, coefficient KL under the common temperature-0.5 target, and coefficient KL under the repair target. The latter two are explicitly named to prevent comparing different target entropies as if they were the same objective. Training additionally records coefficient target entropy and coefficient KL. Sampling and monitoring preserve training RNG states.

The predecessor stopped at epoch 95 / update 12,548 after its last and best checkpoint upload completed. Both full files were copied into the repair's `predecessor/` directory and matched the online digests in predecessor artifact v139. The best remains epoch 55 with historical FID50k 10.9243. Its original executable source and assets were verified unchanged.

The launch sequence requires an eight-rank forward/backward/AdamW preflight, a sample grid, the held-out monitor, a 1,024-image FID pipeline check, strict full-checkpoint reload, finite model/optimizer tensors, all eight RNG states, and the expected scheduler phase. Preflight weights are never used by production. Final startup verification checks fresh initialization, first-loss agreement with preflight, production checkpoint integrity, online last/best artifacts, and the step-200 image upload.

Persistent runtime, source, plans, and receipts: [experiment directory](../outputs/church-ffhq-repair-norefit-b1024-8h100-20260927/). Execution uses `/tmp/laser-church-ffhq-repair-20260927/code` to avoid shared-mount import stalls. `launch.py --resume` resumes this run's full latest state. Creating `STOP_AFTER_EPOCH` in the persistent experiment directory requests a checkpointed stop that drains pending uploads.

The launch and startup receipts provide execution status; numerical preflight success alone does not establish improved generation quality.

The completed sampler confirmation retained the original settings. On seeds 2026092792 and 2026092793, temperature 1.0 scored 10.990421 and 10.998903; temperature 0.9 scored 11.011515 and 10.988199. The candidate failed the requirement to improve on both seeds. The original sampler mean was 10.994662 and the candidate mean was 10.999857. The four 10k screens and complete protocol are in `validation/sampling-screen.json`.

The distributed preflight passed all eight ranks and strictly reloaded its full checkpoint: 517 finite Adam states at step 3, eight RNG states, and scheduler step 3/24,600. Maximum allocated GPU memory was 55,588,156,416 bytes (51.77 GiB). Production PID 36080 started from step zero with empty optimizer state. The first launch attempt failed before any update because it inherited a W&B service socket from the exited validation supervisor. The corrected launcher removes that inherited connection so detached training owns its service; failed startup records are preserved in `failed-startup-wandb-service/`. The training objective and numerical preflight were unchanged by this launcher correction.

Startup verification passed. The local epoch-2 / step-246 checkpoint has finite model tensors, all 517 Adam states at the matching update, eight RNG states, and an aligned scheduler. Initial model hashes and first training losses match preflight; production never loaded preflight-trained weights. Artifact `helloimlixin-rutgers/laser/church-ffhq-repair-norefit-b1024-8h100-20260927-selected-checkpoints:v2` independently verifies the uploaded full last and best states, and the step-200 image downloaded from W&B matches the saved local PNG byte for byte. Local and uploaded checkpoint progress are recorded separately because uploads run asynchronously.

The first FID50k was 177.805898 after epoch 1 / 123 updates. This is an initialization-stage measurement, not evidence of improved final FID. The original recovered run took 986 updates in its first, smaller-batch epoch, so their epoch-one scores are not matched-update comparisons. Training continues, with the preserved 9.6012 official-reference Church checkpoint remaining the stronger established benchmark.
