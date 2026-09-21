# One-epoch Church tokenizer with compound tokens, September 20, 2026

The user requested a compound-token trial after the integer-token run completed with best FID 16.4650. They subsequently requested disabling geometry loss and restarting from scratch. The current experiment reuses the fully finetuned one-epoch tokenizer and complete compound cache, with a fresh transformer for 90 epochs and **geometry loss weight 0**.

Current online run: <https://wandb.ai/helloimlixin-rutgers/laser/runs/church-laser-rfid421-ft1-compound-nogeom90-h200x8-20260920>

The earlier geometry-enabled trial, `church-laser-rfid421-ft1-compound-scratch90-h200x8-20260920`, was stopped at the user's request. Its saved states and sample at step 500 are preserved separately. Its last/best full states are verified in the committed W&B artifact `church-laser-rfid421-ft1-compound-scratch90-h200x8-20260920-selected-checkpoints:v0`. Its brief training history does not establish whether geometry improves final FID. Historical geometry-free runs also changed targets, loss weights, and sampling settings, so they are not matched ablations.

## Tokenizer and cache

The tokenizer started from the ImageNet rFID 4.210914 checkpoint and completed one full Church epoch under the supplied stage-one YAML settings. The production audit confirms updates to the encoder, decoder, encoder/decoder projections, sparse bottleneck, and discriminator. Stage-one results are retained in `outputs/church-integer-ft1-scratch90-20260920/stage1-h200x8`.

The new cache is **compound OMP atom/coefficient pairs**, not the integer run's 32,769-entry vocabulary. It contains all 126,227 training images, each with an 8×8×4 support tensor and normalized continuous coefficients. Training samples coefficient-bin inputs and forms soft coefficient targets from this resident cache without running the image encoder or OMP again.

Preparation reused the verified FP32 encoder latents and generated all OMP pairs on eight GPUs. It matched a fresh encoding of 64 source images exactly, reproduced every checked OMP support against the training auxiliary implementation, and had maximum normalized coefficient storage error 0.000977. Preparation took 36.1 seconds. The per-depth coefficient scales were fitted to this tokenizer's 99.9th absolute percentile, giving approximately 0.1% clipping per depth, with 2,048 coefficient bins over [-3, 3].

- Tokenizer SHA256: `e7b189ff943fe94a80db4c1e6bd299127b5562d68f06d3c196a87ebddd61a937`
- Compound cache SHA256: `d309b7ee1b50898ac433a8b38736fffa3f3fbbe9048b1f8301f49eab0a62cf70`
- Cache receipt: `outputs/church-compound-ft1-scratch90-20260920/cache/complete.json`

## Transformer

The model uses the Church 350M RQTransformer preset with full compound-pair autoregression, two micro-transformer layers, and depth-specific coefficient heads. Atom loss weight is 1.5. Geometry loss and its candidate-specific forward computation are disabled. Tokenizer, cache, seed, architecture, other objective settings, batch size, LR, duration, and sampling settings are retained from the stopped compound trial.

That trial measured approximately 3,500 training images/second before geometry activated and a median 2,023 afterward. This is evidence of computational cost, not a quality ablation. Disabling both the loss and candidate forward path gives approximately **5,030 training images/second** in the new production run, with geometry weight and geometry loss both verified zero. First-step allocated memory is 47.86 GiB per GPU. These are training-only rates; checkpoint and FID time is additional. A controlled final-FID comparison remains outstanding.

All eight H200 GPUs use batch 192 each, global batch 1,536, without accumulation. Each epoch has 82 optimizer updates; 90 epochs have 7,380. AdamW uses learning rate 5e-4, betas (0.9, 0.95), weight decay 1e-4, gradient clipping at 1, and a fresh 90-epoch cosine schedule. The peak LR matches the integer comparison. Training uses BF16, dense FP32 attention on the short depth sequences, and FlashAttention on spatial sequences. The existing batch-size benchmark selected 192 because batch 256 added less than 1% throughput while using substantially more memory.

Sampling produces a 64-image grid every **500 optimizer updates**. FID evaluates 50,000 generated images at epoch 1 and every ten epochs, against the same original RQ-VAE train reference as the integer run. Generation batch is 512 per GPU; preview batch is 64. Sampling uses atom top-k 250, coefficient top-p 0.85, and temperature 1.

## Checkpoints and operation

Full optimizer/scheduler/RNG checkpoints are saved every epoch and every 500 steps. The best full FID checkpoint is retained. Immutable hardlinks allow uploads while training proceeds; the transfer queue keeps only an active transfer and the latest pending snapshot. Online model artifacts contain `last.pt` and `best-fid-01.pt` with `latest`, `last`, and `best-fid` aliases. The uploader waits for commitment and verifies the remote manifest's MD5 digests and sizes. Final uploads drain before W&B is closed. The tokenizer, cache, configuration, and stage-one audits are uploaded together as a recovery artifact.

Current code and runtime dependencies are isolated under `outputs/church-compound-ft1-nogeom90-20260920/runtime`. The supervisor checks source hashes and requires successful cache and eight-GPU training/sampling preflights. The geometry-free preflight completed 12 updates, saved a finite full state, and sampled successfully. The artifact uploader also passed a manifest-verification check. Preflight weights are discarded; production starts from scratch.

- Supervisor: `outputs/church-compound-ft1-nogeom90-20260920/supervise.py`
- Live status: `outputs/church-compound-ft1-nogeom90-20260920/status.json`
- Training log: `outputs/church-compound-ft1-nogeom90-20260920/train.log`
- Per-ten-update metrics: `outputs/church-compound-ft1-nogeom90-20260920/train/metrics.jsonl`
- Upload receipt: `outputs/church-compound-ft1-nogeom90-20260920/train/checkpoint-upload.json`

The supervisor survives terminal disconnection and can resume from the saved last checkpoint. It requires this machine to remain available. A filesystem lock prevents duplicate launches; credentials remain in the private key file outside the repository.

## Stage-one recipe and loss audit

The [September 20 stage-one audit](church-stage1-recipe-audit-2026-09-20.md)
verified 987 generator and 987 discriminator updates, the supplied Church loss
weights, and gradients in every trained component. A matched 4096-image screen
found native rFID 8.9197 for the current tokenizer versus 6.5315 for the earlier
three-epoch tokenizer; compound conversion changes either by less than 0.01.
The earlier tokenizer also changed LR and dictionary optimization, so this is
not an epoch-only ablation. The current transformer resumed from step 656 after
the diagnostic and retains its requested tokenizer and 90-epoch configuration.
