# Original RQ-VAE → Church RQTransformer baseline

The previous LASER compound run was cancelled and saved at step 42,954. This experiment uses the actual KakaoBrain residual-quantization tokenizer and RQTransformer. Stage 2 starts with random weights and an empty optimizer after one epoch of tokenizer fine-tuning.

Source of truth: [KakaoBrain repository](https://github.com/kakaobrain/rq-vae-transformer), pinned to `341395e562ac347f5eb62db9f5f08b9f2cc42a60`. Runtime source is recorded under `outputs/church-rq-baseline-scratch-20260912/upstream-source`; `upstream-adaptations.json` identifies every changed upstream file. The parent project's existing submodule has a later compatibility commit; this experiment was prepared from the verified upstream commit.

## Checkpoint provenance

The [published ImageNet 480M archive](https://twg.kakaocdn.net/brainrepo/models/RQVAE/7518a004fe39120fcffbba76005dc6c3/imagenet_480M.tar.gz) contains the original ImageNet RQ-VAE plus an ImageNet prior. Its archive MD5 matches the publisher's `7518a004fe39120fcffbba76005dc6c3`. Only `stage1/model.pt` and `stage1/config.yaml` were extracted. No pretrained stage-2 weights were extracted or loaded.

The bundled tokenizer configuration specifies 10 ImageNet epochs, an 8×8×4 code map, 16,384 shared entries, and 256-dimensional embeddings. This is the tokenizer with **published ImageNet validation rFID 4.73** in Table 4 of the [paper](https://arxiv.org/abs/2203.01941), not the later 50-epoch, 3.20-rFID tokenizer. The 4.73 number is published provenance; it is not a new local measurement. The released checkpoint contains only `state_dict`, without an epoch field or discriminator. The original fine-tuning driver consequently initializes a fresh discriminator and fresh optimizers.

Downloaded assets and hashes are in `/workspace/tmp/original-rqvae-473/published-stage1/provenance.json`.

## Recipe

| Setting | Stage 1 | Stage 2 |
|---|---|---|
| Initialization | Published ImageNet tokenizer | Released normal(0, 0.02) initializer, fresh optimizer |
| Epochs | 1 | 300 |
| Global batch | 128 | 256 |
| Two-H200 allocation | 64 images/GPU | 32 images/GPU, 4 accumulation steps |
| Optimizer | Adam, β=(0.5, 0.9) | AdamW, β=(0.9, 0.95) |
| Learning rate | Constant 4e-5 | 5e-4, cosine to zero across 300 epochs |
| Weight decay | 0 | 1e-4 |
| Precision | Released FP32 stage-1 loop | FP16 autocast/GradScaler; FP32 tokenizer and soft CE |
| Objective | Original MSE, cumulative commitment, LPIPS and adaptive GAN loss | Original stochastic residual codes and soft-target CE, temperature 0.5 |
| Architecture | Original RQ-VAE, 8×8×4, shared 16,384-entry codebook | 24 spatial + 4 depth blocks; width 1024; 16 attention heads |

Stage-1 configuration is the unchanged `configs/lsun-church/stage1/church256-rqvae-8x8x4.yaml`, with only the local dataset root and per-device batch overridden. Two devices replace the README's four devices while keeping global batch 128. This changes discriminator BatchNorm's local population from 32 to 64. The generator's GroupNorm and the global codebook EMA aggregation retain their original definitions.

Stage-2 configuration is the unchanged [Church YAML supplied by the user](https://github.com/kakaobrain/rq-vae-transformer/blob/main/configs/lsun-church/stage2/lsun-church256-sqgan-8x8x4-350M-simp.yaml). Spatial inputs sum all four codebook embeddings from each preceding location. Depth inputs use cumulative residual embeddings. Both embeddings come from the frozen fine-tuned RQ-VAE.

**The paper and repository disagree:** the supplement reports LSUN fine-tuning LR 4e-6 and stage-2 batch 2048, while the released Church files specify 4e-5 and 256. This experiment follows the repository files, as requested. It also follows the YAML's sampling top-k 250, top-p 1.0, rather than the supplement's separate sampling settings.

## Released versus reconstructed code

The repository explicitly states that it **does not provide stage-2 training code**. Therefore a verbatim reproduction of that unpublished loop cannot be claimed. `scripts/tools/train_church_rq_baseline.py` supplies it using the released model, `Stage2Model._init_weights`, AdamW optimizer helper, scheduler, residual soft-target construction, loss, and sampler. The original unpublished optimizer parameter grouping and initialization call site cannot be verified. This driver applies the released initializer and uses the released optimizer helper's single parameter group.

Stage 1 calls the released `main_stage1.py` and its training loop directly. `scripts/tools/finetune_original_church_rqvae.py` adds scalar/image/status recording and completion checks. It reuses the same LMDB handle for the two identical datasets returned by upstream's LSUN factory, because modern LMDB forbids opening that database twice in one process. The official factory uses the **training population for both train and validation**; stage 1 preserves that behavior. Those validation values are not held-out results.

The four source compatibility changes are Python 3.11 dataclass factories, the current autocast API, converting the LSUN path to a string, and deferring an unrelated CLIP import when importing FID utilities. The neural architecture, quantizer, attention math, losses, stage-1 trainer and scheduler are unchanged.

## Caching, evaluation, and verification

All 126,227 training images receive the original Resize(256) → CenterCrop(256) → normalization transform. Stage 1 makes 987 updates for one complete distributed pass, including upstream DistributedSampler's single padded duplicate. Stage 2 makes 494 update attempts per epoch, with the short final batch normalized by its actual sample count.

After fine-tuning, the frozen encoder's continuous **FP32** outputs are cached once, without image augmentation. The original quantizer recomputes its distance distributions and resamples all four codes on every visit. Subsequent depth targets use the sampled residual prefix. Hard codes and soft distributions are not cached. Frozen codebook hashes are checked before and after cache creation.

The cache pass also computes 50,000-image reconstruction FID, reporting both matched original/reconstructed populations and the published Church reference statistics. Held-out soft CE and hard-code NLL use the separate 300-image official validation LMDB. Generation uses the released cached sampler and decoder. FID-4096 is a diagnostic screen at epoch 1 and every five epochs; FID-50k replaces it every 50 epochs. Sample counts are explicit in metrics and filenames. These evaluation additions do not alter the published learning-rate schedule.

`scripts/tools/verify_original_church_rq.py` exercises the full-size model on both GPUs: empty optimizer, equal random initialization, cached/uncached stochastic targets, sampled-prefix residual distances, causal masking, all 256 cached prediction positions, actual global-batch-256 DDP updates, tokenizer immutability, and the real validation/FID path. Initial diagnostic failures and their logs are retained alongside the final verification result. The released straight-through expression and embedding sum differ at floating-point roundoff; decoder equivalence is checked with TF32 convolution disabled to isolate that arithmetic from Tensor Core rounding.

The production directory records the exact initial weight hash, parameter count, tokenizer hash, source hashes, complete config, progress and W&B URL. Stage 2 refuses to proceed unless stage 1 reports one completed epoch, exactly 987 updates, a non-smoke checkpoint, and a matching SHA-256. Periodic and cancellation checkpoints retain the transformer, optimizer, scheduler, scaler, and per-rank RNG state. No custom coefficient noise, looped blocks, or adaptive LR cuts are added to this baseline.

## Files and launch

- Fine-tuning: `outputs/church-rq-baseline-scratch-20260912/stage1/`
- Stage-2 training: `outputs/church-rq-baseline-scratch-20260912/baseline/`
- Source and verification: `outputs/church-rq-baseline-scratch-20260912/`
- Original downloaded tokenizer: `/workspace/tmp/original-rqvae-473/published-stage1/model.pt`

The launcher keeps the existing authorized W&B credential in memory and injects it into the verified stage-2 process. Credentials are not written into source, configuration, receipts, or documentation. `launch-spec.json` contains the exact new command arguments. `verification.json` records the passed checks and hashes reviewed before the launch.
