# Church: matched official FID evaluation, 2026-09-27

The released RQ-Transformer scores **7.6793** and LASER's frozen stochastic-OMP epoch-114 best checkpoint scores **9.6012** under the same 50,000-image evaluation. The gap is **1.9219 FID**. Lower is better.

| Frozen model | Official church reference | Historical LASER reference | Images |
|---|---:|---:|---:|
| Released RQ-Transformer | 7.679277 | 8.063059 | 50,000 |
| LASER stochastic OMP, epoch 114 / update 7068 | 9.601216 | 9.859998 | 50,000 |

Each row uses exactly the same generated features in both reference columns. Thus the reference effect is measured directly, without sampling variation: -0.383782 for RQ and -0.258782 for LASER. Reference effects depend on the generated distribution; a single correction cannot be subtracted from every historical FID.

## Evaluation contract

- 50,000 unconditional RGB images at 256×256 per model.
- FP32 decoder outputs mapped to continuous [0,1] and clamped, with no PNG/uint8 conversion.
- Both image sets processed by the same released `compute_statistics_from_files`, `get_inception_model`, `mean_covar_numpy`, and `frechet_distance` functions at commit `341395e562ac347f5eb62db9f5f08b9f2cc42a60`.
- Inception batch 500, FP32, CUDA matmul TF32 and cuDNN TF32 disabled for both feature passes. The upstream resize and normalization are unchanged.
- FP32 feature means and unbiased `np.cov` covariance, as in the official code. Exactly 50,000 finite 2048-dimensional features verified per model.
- Official reference SHA256: `809489d8316b9e6eb9dc3bc021b6d602f4b6d816cc80621c6b9c189a9253a7f6`.
- Historical reference SHA256: `ad3b5a341831d877110afdff37d6fff4be855612053e3eb9e2e7119f5593d364`.

The RQ images are reused from the completed released-checkpoint sampler run; their features were freshly extracted under this contract. LASER images were freshly sampled from the immutable checkpoint. The new RQ feature pass changes its earlier reported FID by -0.00935995.

## Frozen models and inference settings

RQ uses the matched released church stage-1 and stage-2 checkpoints, temperature 1, top-k 1400, top-p 1, and the official sampler's AMP. LASER uses the best checkpoint from `helloimlixin-rutgers/laser/church-stochomp-fresh-t00625-k700ct09-b2048-20260925`, epoch 114 / update 7068. Its atom sampler uses temperature 1, top-k 700, top-p 1; its coefficient sampler uses temperature 0.9, all 2048 categories before top-p 0.85, and the existing sampler's AMP. These policies were fixed before this comparison; no sampler sweep was performed.

LASER model weights loaded strictly; its stage-2 model has 404,738,048 parameters versus 370,087,936 stored stage-2 parameter elements in the released RQ checkpoint. LASER checkpoint SHA256: `65d9f1993cfea4b354e00fa69c1dd3636c2123082f0a8cff570a11a26d7e719d`. Tokenizer SHA256: `762c51a10267ed6fa55709ff0d6cf997940d21056b16c7a0a0399b3ebf93868d`.

RQ generation used four ranks at 100 images per rank/batch. LASER used five ranks at 1000 images per rank/batch with decoding in chunks of 64. Both seed bases were 20260927, plus rank; different models, rank layouts, and RNG consumption mean these are independent draws, not paired latent samples. The sample count, image processing, features, and reference are matched. Architectures, tokenizers, training budgets, and established sampling policies differ, so this is an output-quality comparison and does not isolate a training-method effect. One fresh 50k draw per model does not estimate sampling uncertainty.

## Evaluator validation and remaining gap

On the same 64 input images from each model, our Inception wrapper and the released network agree with maximum absolute feature differences of 0 (RQ images) and 0 (LASER images). Our Frechet implementation and the released implementation agree to better than 1e-8 on both references and both generated distributions.

| Official-reference FID component | RQ | LASER |
|---|---:|---:|
| Squared mean mismatch | 3.196855 | 3.037736 |
| Covariance mismatch | 4.482421 | 6.563480 |

These checks validate the feature extractor and FID arithmetic on identical inputs. They do not establish that every training or token-sampling component is correct. The output-quality gap remains after matching the evaluator and reference.

## Supplemental fixed-code checkpoint rescore

The existing epoch-35 fixed-code feature dumps also contain 50,000 samples each and were extracted using the identical upstream Inception source, FP32 and TF32 disabled, with batch 64. These rows reuse those features rather than the primary batch-500 feature pass.

| Epoch-35 fixed-code draw | Official reference | Historical reference |
|---|---:|---:|
| Production seed | 10.237103 | 10.435344 |
| Independent seed | 10.259104 | 10.447178 |

## Artifacts and rerun

Persistent evidence is in `outputs/church-matched-official-fid-20260927/`: `protocol.json`, `comparison.json`, `comparison.csv`, per-model `result.json`, full `acts.npz` feature arrays, `statistics.npz`, generation provenance, source scripts, and logs. Raw float image files remain at the paths in each `result.json` under `/mnt/laser-church`.

Run `python -m torch.distributed.run --standalone --nproc-per-node=5 outputs/church-matched-official-fid-20260927/compare.py generate` to regenerate LASER images using the recorded local dependencies. Run `compare.py score --model rq-release` and `compare.py score --model laser-best114` on a selected CUDA device to re-extract features and score both references. `rescore_saved.py` re-evaluates historical feature dumps without generation. Use `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4` for these commands.

The isolated official checkout retains its earlier Python-3.12 dataclass, optional unused CLIP import, and SciPy `sqrtm` API compatibility changes. No FID math, feature-network source, or trained weights were changed. An initial reporting-key error in this comparison occurred after saving RQ features; it was fixed and scoring resumed from those exact features using `--reuse-acts`.
