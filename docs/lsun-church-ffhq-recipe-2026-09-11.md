# Church stage 2: recovered FFHQ compound recipe

**Paused September 11, 03:22 UTC:** both long training jobs were stopped gracefully after poor samples and rising held-out losses. Reference stopped at epoch 28.81 / step 28,204; balanced at epoch 33.74 / step 33,033. Full optimizer/data-order checkpoints and the best generation-screen checkpoints are retained. The planned 200-epoch training and automatic epoch-50 evaluations have not completed. See the target-noise audit below.

The successful FFHQ run is [`ffhqcmp0804205803`](https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhqcmp0804205803): 200 epochs, 50,000-image FID **8.1744** against the full 70,000-image FFHQ training population. Its actual archived source was downloaded and checked. The later constant-LR v5b run is a different baseline (FID 14.0472 at epoch 100); comparing those runs alone does not isolate an LR effect.

## What the history shows

FFHQ FID fell from 46.83 at epoch 5 to 20.49 at epoch 25, fluctuated around 18–21 through epoch 70, then improved to 11.50 at epoch 100 and 8.17 at epoch 200. The median LR was still 4.72e-4 in epoch 30 and 4.25e-4 in epoch 50. These observations motivate earlier decay; they do not prove that LR caused the plateau. Training atom NLL eventually reached about 0.11, so the Church comparison also measures held-out stage-2 likelihood rather than relying only on training loss and FID against a training population.

[Recorded history and LR plot](../outputs/church-ffhq-recipe-20260911/recipe-analysis.png) · [PDF](../outputs/church-ffhq-recipe-20260911/recipe-analysis.pdf) · [Compact source analysis](../outputs/church-ffhq-recipe-20260911/reference-analysis.json)

## Recovered settings

- One compound event per atom/coefficient pair. Both spatial and depth streams consume **shifted complete pairs**. The maintained implementation needs `pair_autoregressive=True` to reproduce the old v4 behavior.
- Learned atom/coefficient/physical-contribution embedding; a two-layer atom-conditioned micro-transformer; one coefficient classifier per sparse depth.
- **2,048 coefficient bins on [-3, 3] in depth-normalized coordinates.** Soft targets are proportional to `exp(-(normalized_coefficient - bin)^2 / 0.5)`. Context coefficients are sampled from those distributions during training. This is materially different from later physical-distance targets and deterministic nearest-bin training.
- Atom loss weight 1.5; distribution-geometry weight 0.05, delayed two epochs and ramped over three; geometry top-k 4.
- AdamW, betas (0.9, 0.95), weight decay 1e-4; residual dropout 0.1; global batch 128; gradient norm clipped at 1.

The frozen Church tokenizer remains the previously verified **one-epoch Church finetune of the ImageNet rFID 4.2109 checkpoint**. No tokenizer or decoder training is included. Its checkpoint SHA-256 is `93f19adc8ca2cfae0cf629d48d2d6f5f7039440b1a48ad1c7ba43cf17932a388`.

The cache was re-encoded from images using the frozen encoder and FP32 OMP to recover continuous coefficients. It contains 125,203 training images, 1,024 disjoint holdout images, and the 300 official validation images. The scales were fitted **only on training coefficients**, using per-depth maximum absolute value / 3: `[7.4008713, 4.1602407, 2.4553516, 1.6484600]`. No holdout or validation coefficients exceed these ranges. The encoder uses BF16; continuous coefficients are stored as FP32. Images use Resize(256), CenterCrop(256), with no latent-space flips. Stage-1 saw the original training population, including the stage-2 holdout.

Church retains its 16,384-atom dictionary and four sparse depths; FFHQ used 2,048 atoms and two depths. These dataset/tokenizer differences prevent an exact numerical reproduction of FFHQ quality on Church. The maintained distinct-support mask is retained in training and sampling.

## Capacity and learning-rate comparison

Both arms start from random stage-2 weights and share data order, token recipe, LR schedule, effective batch, and evaluation settings. The difference is model width and allocation of transformer layers.

| Setting | Reference capacity | Balanced capacity |
|---|---:|---:|
| Parameters | 404,738,048 | 218,802,176 |
| Width | 1,024 | 768 |
| Spatial layers | 24 | 20 |
| Within-site depth layers | 4 | 6 |
| Attention heads | 16 | 12 |
| Atom-conditioned micro layers | 2 | 2 |
| Spatial context | all 64 sites | all 64 sites |
| Planned epochs | 200 | 200 |
| Effective batch | 128 | 128 |

The narrower arm allocates a greater share of its capacity to modeling the four within-site events and reduces total capacity to test generalization. This is an experiment, not an assumption that smaller is necessarily better. Both arms use the new LR, so this comparison isolates the combined capacity/architecture change, **not** the independent effect of LR.

LR is determined by absolute image-epoch progress: one-epoch linear warmup from 1e-6 to 2e-4, immediate exponential decay to 5e-5 by epoch 30, then cosine decay to 2e-6 at epoch 200. Approximate rates are 1.30e-4 at epoch 10, 8.06e-5 at 20, 4.84e-5 at 50, 3.26e-5 at 100, and 1.15e-5 at 150. Resume cannot restart the schedule. Each arm has 195,800 optimizer steps and 25,040,600 image presentations, including the final partial batch of every epoch.

A microbatch of 128 triggered a BF16 classifier-backward CUBLAS execution failure on the installed runtime. Training therefore accumulates two microbatches of 64; effective batch stays 128. The runtime guard prevents unsupported larger training microbatches.

DCTransformer uses 896-event target chunks with 128-event overlap, conditioning on an encoding of the preceding partial DCT image. Our sequence has only **256 compound events**, with the existing RQ hierarchy reducing spatial attention to 64 sites. Full context fits comfortably, so the first comparison does not truncate context or add a new chunk encoder. [DCTransformer paper, §3.1](https://proceedings.mlr.press/v139/nash21a/nash21a.pdf)

## Evaluation and recovery

- Every epoch: atomically replace the full model/optimizer/data-order checkpoint.
- Epoch 1 and every five epochs: deterministic-context train-probe, holdout, and validation diagnostics, including coefficient KL, sign accuracy, physical coefficient MAE, and depth-specific losses. Fixed-seed FID-4096 and sample grids screen generation.
- Epochs 50, 100, 150, 200: FID-50,000. After training: independent-seed FID-50,000 for the best screening checkpoint, plus held-out diagnostics. Small-sample FID is explicitly labeled and is not compared directly with historical FID-50,000.
- Sampling: temperature 1; atom top-k 2,048 (approximately the same vocabulary fraction as FFHQ's 250/2,048); coefficient nucleus p=0.85, matching FFHQ. Thus historical Church FID under atom top-k 250 / untruncated coefficients is not an isolated architecture comparison.
- Original RQ-VAE Inception and Church reference statistics; BF16 AR, FP32 decoder. Evaluation RNG is isolated from training. W&B logs live metrics/grids and uploads the selected model after completion.
- SIGTERM/SIGINT save a recoverable checkpoint. Resume restores optimizer and exact shuffled-data position, checks configuration/source hashes, and uses deterministic per-step stochastic-context/dropout seeds.

The recipe/causality/objective test suite passed **29 tests**. Both capacities passed actual GPU training and sampling smoke tests. A balanced-model stop at step 4 followed by resume to step 6 produced **bitwise identical weights, optimizer state, data stream, and step** to uninterrupted six-step training. Smoke FIDs use random weights and 64 images solely to exercise the evaluator; they are not quality results.

[Resume verification](../outputs/church-ffhq-recipe-20260911/resume-verification.json) · [GPU checks](../outputs/church-ffhq-recipe-20260911/gpu-verification.json)

## Run locations

- [405M reference-capacity run](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-ffhq-v4-reference-earlydecay-20260911)
- [219M balanced-capacity run](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-ffhq-v4-balanced-earlydecay-20260911)
- Local output: `outputs/church-ffhq-recipe-20260911/{reference,balanced}`; each contains `status.json`, `history.jsonl`, `config.json`, checkpoints, and evaluations.
- Source snapshots and hashes: `outputs/church-ffhq-recipe-20260911/source-snapshot`.
- Launcher: `scripts/launch_church_ffhq_recipe.py`; use `--resume --arms reference balanced` with an existing private W&B credential to recover stopped runs. The launcher rejects already-active processes.

Training is a long-running experiment. Successful startup and validation do not establish improved image quality; compare held-out curves and the completed generation evaluations before selecting a replacement prior.


## Target-noise audit after pausing

The launch tested deterministic nearest-bin reconstruction drift (69.1 dB PSNR) but did not test samples from the actual soft-target distributions. That was a material omission: faithful reuse of the FFHQ normalized-space temperature does not establish that its perturbations are appropriate for the Church tokenizer.

For `exp(-(c-bin)^2/T)`, the untruncated Gaussian standard deviation is `sqrt(T/2)`. At T=0.5 this is 0.5 in normalized coordinates, or `[3.7004, 2.0801, 1.2277, 0.8242]` in physical coefficient units. Those standard deviations are 46–51% of the corresponding Church coefficient RMS magnitudes. The training sampler uses the full distribution; generation uses nucleus p=0.85.

A frozen-decoder test supplies the true supports and true underlying continuous coefficients, then draws bins directly from the target distributions. All methods use the same images and coupled uniforms. Values below compare with the continuous clean tokenizer reconstruction, not with the original image.

| Oracle coefficient construction | Validation PSNR ↑ | Validation LPIPS ↓ | Relative latent MSE ↓ |
|---|---:|---:|---:|
| Nearest bin | 69.11 | 0.0000052 | 0.00000073 |
| Copied normalized target, full sampling | 17.80 | 0.2603 | 0.2549 |
| Copied normalized target, nucleus 0.85 | 19.83 | 0.1787 | 0.1330 |
| Physical-distance target, T=0.5, nucleus 0.85 | 31.28 | 0.0197 | 0.0066 |

This used all 300 official validation images and independently checked 256 stage-2 holdout images. Holdout LPIPS was 0.1778 for normalized/nucleus targets and 0.0194 for physical/nucleus targets. On validation, their paired LPIPS difference was 0.1590 ± 0.0013 standard error. No model parameters were trained.

The target perturbations visibly distort geometry even with oracle supports. This identifies a problematic perturbation scale; it does not establish that changing the training target alone will improve unconditional FID or solve the separate atom-overfitting problem. Physical-distance targets were used in earlier Church experiments, so this is a correction to the transfer experiment, not a novel solution already known to beat the established prior.

[Oracle comparison grid](../outputs/church-ffhq-recipe-20260911/target-noise-probe/validation-oracle-targets.png): columns are continuous reconstruction, nearest bin, normalized/full, normalized/nucleus-0.85, physical/nucleus-0.85. [Full measurements](../outputs/church-ffhq-recipe-20260911/target-noise-probe/results.json)

[Training/holdout/FID curves at pause](../outputs/church-ffhq-recipe-20260911/paused-training-audit.png). Both models' held-out atom and coefficient losses worsened after their early minima while their training fits improved. Smaller capacity reduced this gap but did not yield better generation in the available checkpoints. Neither run completed the planned 200 epochs or a FID-50,000 evaluation, so later improvement is untested; pausing preserves the option to resume.


### Frozen-checkpoint sampling check

The 405M epoch-25 checkpoint was evaluated with the same 4,096 samples, seed 15701, batch 128, BF16 AR, and FP32 decoder. No weights changed. Only atom top-k and/or coefficient nucleus cutoff changed.

| Atom top-k | Coefficient nucleus p | FID-4096 ↓ |
|---:|---:|---:|
| 2048 | 0.85 | 19.4552 |
| 250 | 0.85 | 19.1054 |
| 2048 | 0.5 | 16.7586 |
| 250 | 0.5 | 17.8565 |

Narrower coefficient sampling improves this screening score, but severe structural/black-patch artifacts remain in the inspected grids. These are sampling comparisons for one frozen checkpoint and one screening seed, not FID-50,000 confirmation or proof that retraining with narrower targets will work. The long training runs remain paused.

[Full sampling measurements](../outputs/church-ffhq-recipe-20260911/sampling-probe/comparison.json) · [Narrower-coefficient sample grid](../outputs/church-ffhq-recipe-20260911/sampling-probe/atom2048-coeff05/samples.png)
