# Deterministic OMP and normalized coefficient noise comparison

Two fresh Church stage-2 runs compare normalized coefficient target temperatures 0.25 and 0.5. Each receives two H100s. They share initialization, data order, architecture, optimizer, schedules, dropout, sampling settings, and evaluation protocol. No trained stage-2 weights or optimizer state are transferred from an earlier run or a preflight.

| Setting | Both runs |
| --- | --- |
| Stage 1 | Frozen selected epoch-3 lr1e-5 tokenizer, full reconstruction FID 2.6393 |
| Stage-1 SHA256 | `762c51a10267ed6fa55709ff0d6cf997940d21056b16c7a0a0399b3ebf93868d` |
| OMP | Deterministic greedy support selection, four depths, no stochastic support bank |
| Cache | All 126,227 training images, FP32 coefficients, fixed center crop |
| Cache SHA256 | `4c03555ebedca1bf7b74f1db30e91752586889337a58547e856416c3aeeae586` |
| Coefficient scales | 7.662354, 4.158035, 2.633324, 1.651212 |
| Coefficient bins | 2,048 uniform normalized centers in [-3, 3] |
| Normalization | Per-depth training maximum absolute coefficient divided by 3; not RMS normalization |
| Objective | Full soft coefficient targets, fresh categorical coefficient histories each visit; atom CE weight 1.5, coefficient CE weight 1, geometry loss 0 |
| Architecture | Full atom/coefficient pair autoregression, dictionary atom vector conditioning, two-layer coefficient micro-transformer, depth-specific coefficient heads |
| Residual dropout | 0.2; attention and embedding dropout 0 |
| Optimizer | AdamW, lr 5e-4, betas (0.9, 0.95), weight decay 1e-4, gradient clip 1 |
| Batch | Exact global 2,048 = 2 GPUs × 256 images × 4 accumulation; final 1,299-image batch correctly weighted |
| Schedule | 300 epochs, 62 updates per epoch, 18,600-update cosine to zero, no warmup |
| Generation | Atom top-k 250 / top-p 1; coefficient top-k 0 / top-p 0.85; both sampling temperatures 1 |
| Evaluation | Official RQ-VAE FID, 50,000 generated images every epoch; same 126,227-image real reference, seed and two-rank layout |
| Held-out diagnostics | Fixed 300 train / 300 validation images and identical nearest-bin coefficient histories |
| Checkpoints | Full latest and best FID states, including optimizer, scheduler and both rank RNG states; immutable online W&B artifacts with size and digest verification |

For physical coefficient `c` and depth scale `s_d`, the target kernel is

`q(j | c, d) ∝ exp(-((c / s_d) - bin_j)^2 / T)`.

Away from bin boundaries, normalized standard deviation is approximately `sqrt(T / 2)`: 0.3536 at T=0.25 and 0.5 at T=0.5. T=0.25 approximately matches the total normalized variance of FFHQ's two coefficients at T=0.5; T=0.5 matches its per-coefficient variance. These are experimental hypotheses, not established optimal Church temperatures. The finite bin range truncates the kernel near boundaries.

The frozen historical runtime gains an explicit `--coeff-target-space` option. `normalized` prevents its previous automatic switch to physical-distance targets when depth scales are supplied. Its existing checkpoint serialization cycle collection remains enabled. Unrelated workspace training changes are not incorporated into these runs.

Validation includes 27 focused tests, exact two-rank batch coverage and gradient equivalence, upstream RQ-Transformer numerical checks, causal perturbation and cached-generation agreement, dictionary-vector conditioning gradients, and bitwise coefficient-kernel agreement with the saved successful FFHQ trainer at both temperatures. Fresh FP32 OMP matches every atom in 32 sampled training images; maximum normalized coefficient difference is below 6e-6. All 300 held-out validation images were freshly encoded with deterministic OMP. The successful FFHQ run's imported backbone revision was not recorded, so source equivalence to that entire historical environment is not claimed.

The previous dropout trial is paused at epoch 19 / update 1,178, with best FID 19.0465. Its full last/best states are preserved locally and verified in W&B artifact `church-rqrecipe300-dropout02-b2048-4h100-20260924-selected-checkpoints:v18`. It can resume with its original four-rank, accumulation-2 layout. Resume details are in `outputs/church-detomp-normalized-comparison-20260924/paused-dropout.json`.

Run links:

- [T=0.25](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-detomp-normt025-b2048-2h100-20260924)
- [T=0.5](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-detomp-normt050-b2048-2h100-20260924)

Live supervisor status, preflight results, initialization hashes and pause record are under `outputs/church-detomp-normalized-comparison-20260924/`. Per-run directories contain full configuration and source provenance, online upload receipts, per-epoch FID and held-out metrics, and preview grids. Raw coefficient CE/KL across different target temperatures is not a common objective; use the matched FID protocol and atom NLL alongside each run's held-out gap.

Both production runs started from the identical complete model state SHA256 `d35285243e65a37c448f990634eb1f0989995e84b7583c6486ff7d60cb005be5`. Training preflights peaked at 66.98 GiB allocated per GPU. Both completed epoch 1 and resumed training through epoch 2; first 50k-image FIDs were 149.3615 and 168.2032, respectively. These initial values do not establish a preferred temperature.

Both epoch-1 full last/best checkpoints are committed online in their respective `-selected-checkpoints:v0` artifacts. Each file is 4,857,539,129 bytes; manifests verify both size and MD5. The joint local and online verification record is `outputs/church-detomp-normalized-comparison-20260924/launch-verification.json`.
