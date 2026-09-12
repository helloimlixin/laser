LSUN Church joint-sign pilot — September 10, 2026

**Joint prediction improves complete sign-pattern likelihood and accuracy, but this pilot does not demonstrate improved reconstruction quality.** Both matched priors completed 2,000 optimizer steps, and both selected step 1,500 using minimum development-set pattern NLL.

| Official validation, 300 images | Independent signs | Joint 16-pattern head |
| --- | ---: | ---: |
| Pattern NLL, true previous signs (lower is better) | 2.2675 | **2.1962** |
| All four signs correct, true previous signs | 21.74% | **23.78%** |
| Individual signs correct, joint MAP decision | 68.38% | 68.16% |
| Individual signs correct, per-sign marginal decision | 68.38% | 68.57% |
| Reconstruction PSNR vs true codes, true previous signs | 14.3723 dB | 14.3729 dB |
| All four signs correct, predicted previous signs | 17.01% | **19.63%** |
| Individual signs correct, predicted previous signs | 63.20% | 63.42% |
| Reconstruction PSNR vs true codes, predicted previous signs | **12.6519 dB** | 12.5637 dB |

Every row uses real supports and real quantized coefficient magnitudes. Only the signs are predicted. The reconstruction reference is the frozen decoder's output from the true quantized codes, not the original image. These are oracle conditional results, not unconditional FID measurements.

The teacher-forced pattern-accuracy improvement is **2.04 percentage points**, with a paired image-bootstrap 95% interval of **1.30 to 2.78 points**. Pattern NLL improves by 0.0713 nats per site. The intervals for differences in individual-sign accuracy and latent MSE include zero. There is evidence that the joint head captures dependencies among signs; there is no demonstrated benefit to decoded image quality. These intervals describe variation across validation images for one training seed, not variation across retraining seeds.

The joint head's teacher-forced sign accuracy by OMP depth is 83.97%, 71.06%, 62.00%, and 55.60%. Using predicted signs in earlier spatial sites reduces these to 76.24%, 64.95%, 58.56%, and 53.93%. Both heads suffer substantially when their own sign mistakes enter subsequent context. The small pilot is not sufficient to determine whether a full pretrained stage-2 context model would overcome this limitation.

![Development curves and held-out sign metrics](../outputs/lsun-church-sign-pattern-20260910/comparison.png)

The tokenizer was recovered from W&B artifact `churchft1ep-20260909024359-stage1-checkpoints:v0`. Its SHA-256 is `93f19adc8ca2cfae0cf629d48d2d6f5f7039440b1a48ad1c7ba43cf17932a388`, matching the September Church experiments. Cache extraction reproduces their resize/center crop, BF16 encoder, FP32 OMP, float16 normalized coefficient storage, nearest 2,048-bin quantization, range ±20, and per-depth scales. The diagnostic stores dequantized coefficients in physical units.

The requested tokenizer adaptation was already completed on September 9: [the Church fine-tune](https://wandb.ai/helloimlixin-rutgers/laser/runs/churchft1ep-20260909024359) loaded `best_rfid_slot1_model.pt` from [the ImageNet run with rFID 4.210914](https://wandb.ai/helloimlixin-rutgers/laser/runs/imga16384k4altbn64-b128-b300-20260830000755). Its configuration specifies one epoch, learning rate 4e-6, and total batch size 128. The recovered checkpoint independently records `epoch=1`, `global_step=987`, and `steps_per_epoch=987`; the completed Church run reports validation rFID 4.902147. These rFID values belong to different datasets and are not a before/after comparison. Both sign priors used this adapted tokenizer, so no additional fine-tune was needed.

September 11 protocol correction: the archived run's loader labeled `validation` contained all 126,227 training images, rather than the 300 official validation images. Thus 4.902147 is the archived training-population reconstruction FID. [Audit and matched official-validation measurements](lsun-church-support-pattern-integer-2026-09-11.md).

The cache contains 16,384 randomly selected training images, 1,024 separate training-population development images, and all 300 official-validation images. LMDB keys do not overlap across these splits. The frozen tokenizer was previously trained on the original training population, and official validation was used in earlier experiments. Coefficients are finite, supports contain four distinct atoms at every site, and positive/negative signs are approximately balanced at every depth.

Each newly initialized prior has a four-layer, width-256 causal transformer with eight attention heads and roughly 4.82 million trainable parameters. Both receive identical minibatches and start with identical backbone weights. Inputs include the current site's complete atom support and magnitudes, plus signed coefficients from strictly earlier raster sites. The joint model outputs one distribution over 16 sign patterns; the independent model outputs four Bernoulli probabilities from the same context. Joint NLL is normalized by four for a comparable training objective. AdamW, batch 128, seed 2701, and the learning-rate schedule are matched. This is a small experiment trained from scratch, not a fine-tune of the existing 350M prior.

Seven focused tests pass: pattern encoding and normalization, current/future sign exclusion, future support/magnitude exclusion, matched backbone initialization, representability of correlated signs, physical cancellation in reconstruction metrics, and causal sign rollout. Both heads also overfit a tiny synthetic batch under BF16 CUDA training. A complete synthetic CLI run exercised checkpoint selection and final evaluation. The main runs finished successfully, saved checkpoints and per-image metrics locally, and synced results and reconstruction grids to W&B.

[Joint run](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-sign-joint16-20260910) · [Independent run](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-sign-independent-20260910) · [Full comparison data](../outputs/lsun-church-sign-pattern-20260910/comparison.json)

The implementation is in `src/sign_pattern_prior.py`, with cache extraction in `scripts/tools/build_sign_probe_cache.py` and training in `scripts/probe_sparse_sign_patterns.py`. To reproduce the matched training using the prepared cache, run the following once with `MODE=joint` and once with `MODE=independent`, choosing an unused output directory for each:

```bash
MODE=joint
python scripts/probe_sparse_sign_patterns.py \
  --cache outputs/lsun-church-sign-pattern-20260910/sign-cache.pt \
  --output outputs/sign-repeat/$MODE --mode "$MODE" \
  --steps 2000 --batch-size 128 --eval-every 250 \
  --decode-checkpoint outputs/lsun-church-sign-pattern-20260910/assets/best_rfid_slot1_model.pt

python scripts/compare_sign_pattern_probes.py outputs/sign-repeat
python -m pytest -q tests/test_sign_pattern_prior.py
```

The environment used here is `/tmp/laser-sign-venv/bin/python`. Optional W&B logging uses `--wandb-id` and an existing `WANDB_API_KEY`; `--wandb-key-stdin` accepts the key without echoing or storing it in source files. Existing stage-2 jobs and checkpoints were not modified.
