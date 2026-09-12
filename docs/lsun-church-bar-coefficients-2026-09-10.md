LSUN Church: completed BAR-inspired coefficient-head pilot, September 10, 2026

**This pilot did not improve unconditional generation.** The BAR-inspired head slightly improved coefficient errors given real preceding tokens, but its 2,048-image FID was 21.9275 versus 19.2919 for a matched categorical continuation and 19.0550 for the original checkpoint. Keep the original epoch-50 checkpoint as the default. Neither continuation demonstrated a generation improvement in this screen.

Both training runs completed 1,000 optimizer steps and their evaluations; W&B synchronization completed successfully. This is a short adaptation experiment with one training seed, not a conclusion that BAR cannot work on sparse coefficients. We did not proceed to FID-50k after the negative screen.

| Measurement | Original epoch-50 AR | Categorical continuation | BAR-inspired continuation |
| --- | ---: | ---: | ---: |
| Unconditional FID, 2,048 generated images; lower is better | 19.0550 | 19.2919 | 21.9275 |
| Validation coefficient sign accuracy | 88.2995% | 88.5885% | 88.6862% |
| Validation physical coefficient MAE; lower is better | 1.26931 | 1.23472 | 1.21070 |
| Validation latent MSE from coefficient prediction; lower is better | 0.104357 | 0.099496 | 0.097951 |
| Validation exact coefficient-ID accuracy | 0.6628% | 0.7279% | 0.6263% |
| Validation atom NLL; lower is better | 10.47922 | 10.49400 | 10.49549 |
| Generation/evaluation time for 2,048 images, seconds | 37.52 | 37.51 | 50.85 |

Validation contains all 300 official Church validation images. These coefficient diagnostics use the true preceding tokens and true current atom; they do not measure performance with generated history. The latent metric substitutes predicted coefficients on the true atoms and measures squared latent error. It is not decoded-image reconstruction FID. Categorical predictions use the most probable ID; BAR predictions use greedy progressive bit completion, which is not necessarily the globally most probable ID.

The BAR-minus-categorical paired image bootstrap gives a sign-accuracy difference of +0.0977 percentage points (95% interval +0.0234 to +0.1693), physical MAE difference -0.024018 (interval -0.029277 to -0.018745), and latent MSE difference -0.001545 (interval -0.002612 to -0.000428). These intervals resample validation images 5,000 times; they do not measure training-seed uncertainty or FID uncertainty. The apparent coefficient improvement therefore should not be promoted as an image-generation improvement.

![Matched continuation results](../outputs/lsun-church-bar-20260910/comparison.png)

The first 32 generated samples were saved without selection: [original](../outputs/lsun-church-bar-20260910/source-baseline/samples.png), [categorical](../outputs/lsun-church-bar-20260910/categorical/samples.png), [BAR-inspired](../outputs/lsun-church-bar-20260910/bar/samples.png). Both continuations produce recognizable churches with structural distortions and occasional watermark-like patterns. These small grids do not establish a reliable visual ranking. Although the initial random seed is shared, the samplers consume randomness differently; corresponding grid cells are not matched scenes.

The experiment changes how an existing coefficient ID is predicted. Each coefficient was already quantized to an integer ID from 0 through 2047. The new head represents that same ID with 11 ordinary binary bits and generates them in four rounds, revealing 2, 3, 3, and 3 bits. Its decoded coefficient can still be negative: the ID is a label for a signed bin. The quantization grid, coefficient scales, dictionary, and decoded value of every ID are unchanged. This tests a prediction head; it does not make physical coefficients positive or move signs into atom IDs.

The head uses three conditioned residual MLP blocks of width 512, with a depth embedding and the existing causal context conditioned on the current atom. Training hides a uniformly selected number of bits from 1 through 11 and predicts only hidden bits. Sampling starts with all 11 bits unknown and progressively commits its own predicted values at confident positions. The architecture is inspired by the [BAR paper](https://arxiv.org/html/2602.09024v1) and its [official implementation](https://github.com/amazon-far/BAR); this is an adaptation to scalar coefficient IDs, not a reproduction of BAR's learned tokenizer or reported ImageNet results.

The original pair order remains `atom1, coefficient1, atom2, coefficient2, ...`. The support predictor, spatial/depth transformer, and two-layer atom-conditioned coefficient refinement initialize from the same source weights in both arms. The categorical arm retains its pretrained depth-specific 2,048-way classifiers. BAR starts with a newly initialized masked-bit head and freezes its unused legacy classifiers. This is a practical replacement-head comparison; head initialization, capacity, objective, and sampling procedure differ by design.

Both arms use the same data order, dropout seeds, batch size 128, 250 steps fitting only the output head, and 750 steps updating the head and backbone. The head learning rate is 3e-4 and the backbone learning rate 1e-5, with cosine decay to 20%, AdamW betas (0.9, 0.95), weight decay 0.03, and gradient clipping at 1. The total is 128,000 image presentations per arm. Checkpoints are selected at the fixed final step, without validation-based selection. The BAR objective is masked-bit BCE multiplied by 11, combined equally with atom NLL. It is a surrogate objective, not exact categorical token NLL; raw training losses must not be compared as equivalent likelihoods.

The cache contains 125,203 continuation-training images, 1,024 monitoring images excluded from this continuation, and 300 official validation images, with disjoint keys. The original pretrained AR already saw the entire training population, including the monitoring images. That split is useful for monitoring this continuation, but it is not unseen validation. The official validation set also has a history of use in earlier checkpoint selection. The cache uses the established BF16 encoder, FP32 OMP, float16 normalized coefficient round-trip, and nearest 2,048-bin quantization. Exact recovery of physical coefficient values from cached IDs was checked within 2e-6.

All three generation evaluations use seed 12701, batch size 128, atom top-k 250, atom temperature 1, coefficient temperature 1, no guidance, BF16 AR inference, and FP32 decoding in batches of 32. They use the original RQ-VAE Inception implementation and Church reference statistics. FID at 2,048 samples has sampling error and finite-sample bias. It must not be directly compared with the historical FID-50k of approximately 13.72. BAR's screen is 2.6356 FID worse than the matched continuation and takes approximately 1.36 times as long in this measurement.

The requested tokenizer fine-tune had already completed on September 9. Run `churchft1ep-20260909024359` trained for one Church epoch from the ImageNet tokenizer whose best ImageNet rFID was 4.210914. The recovered Church checkpoint has epoch 1, global step 987, and Church validation rFID 4.902147. The ImageNet and Church scores concern different datasets and are not a before/after quality comparison. Both new arms used this exact frozen Church tokenizer:

```text
Tokenizer SHA256: 93f19adc8ca2cfae0cf629d48d2d6f5f7039440b1a48ad1c7ba43cf17932a388
Source AR SHA256: cbb7d0a8c22c31e042ed04004201b360b9bacd1238f6fa677d28ba3e0b21ce4a
Source AR run: helloimlixin-rutgers/laser/churchftdc-20260909024359
Source AR checkpoint: epoch 50, global step 3050, logged FID-50k 13.620649
```

The source run's saved `CompoundLaserRQTransformer` and `build_model` definitions were verified identical to the local definitions. Tokenizer provenance is saved in [the lineage verification](../outputs/lsun-church-sign-pattern-20260910/assets/tokenizer-lineage-verification.json); experiment verification is in [verification.json](../outputs/lsun-church-bar-20260910/assets/verification.json).

Implementation and validation:

- [Masked coefficient head](../src/masked_coefficient_head.py).
- [Matched training and evaluation script](../scripts/train_church_bar_coefficients.py).
- [Comparison and paired-bootstrap script](../scripts/compare_church_bar_coefficients.py).
- [Head and integration tests](../tests/test_masked_coefficient_head.py): all 2,048 IDs round-trip; hidden targets cannot influence predictions; sampling begins fully masked and preserves revealed bits; current/future targets cannot enter the causal coefficient context; cached inference matches parallel context; a tiny conditional problem can be learned.
- Those tests and the five existing compound pair-autoregression tests passed: **11 passed**. Both one-step GPU smoke runs and both full runs completed without nonfinite-gradient failures. The comparison script verified matched experiment settings and generated [comparison.json](../outputs/lsun-church-bar-20260910/comparison.json).

The run records include configurations, monitoring curves, final metrics, and sample grids: [BAR-inspired](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-bar-coefficients-20260910), [categorical control](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-bar-categorical-control-20260910), and [original checkpoint evaluation](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-bar-source-baseline-20260910). Final model checkpoints, generated token IDs, and per-image validation measurements remain under `outputs/lsun-church-bar-20260910/{bar,categorical,source-baseline}/`. Model checkpoints were saved locally; W&B received metrics and samples.

To reproduce the BAR arm from the prepared assets, use a fresh output directory:

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 \
TORCH_HOME=/workspace/tmp/official-rqvae-eval-cache \
/tmp/laser-sign-venv/bin/python scripts/train_church_bar_coefficients.py \
  --mode bar \
  --source-checkpoint outputs/lsun-church-bar-20260910/assets/epoch50-source.pt \
  --stage1 outputs/lsun-church-sign-pattern-20260910/assets/best_rfid_slot1_model.pt \
  --cache outputs/lsun-church-bar-20260910/church-cache.pt \
  --output outputs/lsun-church-bar-reproduction/bar \
  --steps 1000 --head-only-steps 250 --batch-size 128 --eval-every 250 \
  --fid-stats outputs/lsun-church-bar-20260910/assets/lsun_256_church.npz \
  --fid-samples 2048 --generation-batch 128
```

Use `--mode categorical` and a separate output directory for the matched control. W&B logging is optional; supply a fresh `--wandb-id` and `--wandb-key-stdin` to enter credentials through a hidden prompt. The original checkpoint evaluation uses categorical mode with `--steps 0`. Run the comparison script on a directory containing `source-baseline`, `categorical`, and `bar` results.

The practical finding is that improved prediction under real history did not transfer to better generated images in this adaptation. It does not identify whether the limiting factor is head training duration, the induced bit-sampling distribution, coefficient/atom compatibility under generated history, or broader model generalization. The next experiment should use the original categorical checkpoint to study generated-history errors and decoded-image quality, following the [method shortlist](lsun-church-stage2-method-shortlist-2026-09-10.md), while retaining this BAR run as a measured baseline. No further training or post-training method was launched as part of this pilot.
