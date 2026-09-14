# ImageNet RQ-Transformer recipe audit

The active compact ImageNet prior follows the published 480M RQ-Transformer
architecture and training hyperparameters, with the sparse tokenizer and its
calibrated soft-target temperature retained. The user requested verification
of batch size, learning rate, sampling, and the rest of the ImageNet recipe.

## Authoritative sources

- [Official training configuration](https://github.com/kakaobrain/rq-vae-transformer/blob/main/configs/imagenet256/stage2/in256-rqtransformer-8x8x4-480M.yaml).
- [Paper, Appendix A.3](https://arxiv.org/html/2203.01941v2#A3), which specifies
  AdamW, learning rate, weight decay, global batch, cosine scheduling, and dropout.
- [Official pretrained checkpoints](https://github.com/kakaobrain/rq-vae-transformer#pretrained-checkpoints).
  The local `/workspace/tmp/imagenet_480M.tar.gz` contains the published stage-2
  configuration. Its configuration was extracted without loading the pretrained
  transformer weights into our run and saved as
  `outputs/imagenet-depth-compact-rq64k-stage2-20260913/original-published-stage2-config.yaml`.
- [Official ImageNet transforms](https://github.com/kakaobrain/rq-vae-transformer/blob/main/rqvae/img_datasets/transforms.py).

The released repository includes the transformer implementation, training
configuration, and sampling pipeline, but not its stage-2 training driver.
This audit verifies the published configuration and available implementation;
it does not claim a bitwise reproduction of the unpublished driver.

## Verified configuration and corrections

| Setting | Published recipe | This run after correction |
| --- | --- | --- |
| Global batch | 2,048 | 2,048 |
| Hardware batching | YAML microbatch 32; global batch 2,048 | 128 per GPU x 4 GPUs x 4 accumulation steps |
| Optimizer | AdamW | AdamW, same all-parameter grouping from the released optimizer helper |
| Initial learning rate | 0.0005 | 0.0005 |
| Adam betas | 0.9, 0.95 | 0.9, 0.95 |
| Weight decay / epsilon | 0.0001 / 1e-8 | 0.0001 / 1e-8 |
| Gradient clipping | 1.0 | 1.0 |
| Schedule | Cosine, 100 epochs, zero warmup, minimum LR zero | Same; existing scheduler state preserved |
| Architecture | Width 1536, 12 spatial / 4 depth blocks, 24 heads | Same body and head blocks |
| Residual dropout | 0.1 | 0.1 |
| AMP | Enabled | FP16 transformer, FP32 frozen tokenizer and soft CE |
| Image augmentation | Resize shorter edge to 256, random 256 crop, horizontal flip p=0.5 | Fresh transforms each image and epoch; frozen encoder runs online |
| Generation temperature / top-k / top-p | **1.0 / 256 / 0.95** in the published 480M checkpoint | Same for previews and generation FID |

The top-k/top-p distinction matters: the training YAML's `experiment.sample`
contains **16,384 / 0.92**, whereas the published 480M checkpoint's `sampling`
contains **256 / 0.95**, at temperature **1.0**. Earlier code called the training
YAML settings `original` and also used temperature 0.9 for selected previews.
The corrected production sampler is explicitly named `published_imagenet_480m`.
Its FID is logged under `generation/published_fid_4096` and
`generation/published_fid_50000`; earlier FID histories retain their original
meaning. No rejection sampling is applied. The reference statistics are the
released ImageNet-256 training statistics.

Fresh augmentation replaces the earlier two-view latent cache for training.
The crop seed depends on image index and epoch, giving new augmentations each
epoch and reproducible mid-epoch resumes. Validation retains its deterministic
center-crop latent cache and regenerated compact hard IDs. The checkpoint's
image order, epoch, optimizer, scheduler, scaler, and per-rank RNG states are
restored when applying the correction.

## Sparse-tokenizer adaptations

The vocabulary is 65,537 rather than 16,384 because this experiment uses the
user's sparse atom/coefficient representation. The frozen coefficient tables
depend on atom and residual stage. This expands the classifier while retaining
the original transformer body/head architecture and 256-token image shape.

Soft training targets use temperature **0.0625**, compared with **0.5** in the
original RQ-VAE configuration. This is separate from generation temperature.
The sparse tokenizer has a different distance scale: at 0.0625 its
sampled/hard residual-MSE ratio is **1.007795**, close to the original RQ-VAE's
**1.008906** at 0.5. Literal 0.5 on the sparse tokenizer gives **1.661568**.
The calibration is retained and explicitly recorded in configuration and the
audit receipt. Full reconstruction rFID remains 4.383626; no frozen tokenizer
weights or coefficients change in this recipe correction.

## Validation and run history

Twenty-eight focused tests passed, including changing augmentation across
epochs and reproducible resumed crops. A four-GPU preflight checks real-image
encoding, stochastic soft targets, three optimizer updates, finite saved state,
strict checkpoint reload, class conditioning, and the selected published sampler.
The audit script also checks saved AdamW parameter groups and the actual cosine
scheduler state rather than relying only on configuration labels.

The source update is applied to the existing run after a checkpointed pause.
The prior source snapshot, configuration, launch receipt, and complete training
checkpoint are preserved under `source-revisions/`. The run's optimizer progress
is retained; this correction does not restart its training schedule.

The correction was applied at optimizer step **3,757**, during epoch seven
(six completed epochs, four consumed microbatches per rank). The learning rate
at the pause was **0.0004955694614313902**, cosine position 3,757 of 62,600,
and AMP scale 131,072. Training resumed with those states and then advanced
with zero AMP skips. Earlier epochs retain their historical cached-view and
sampling settings; the correction does not relabel their metrics.

The old snapshot and complete checkpoint are preserved at
`outputs/imagenet-depth-compact-rq64k-stage2-20260913/source-revisions/before-1789332712/`.
W&B confirms `training_data.mode=online-images`, the published generation
sampler, and the unchanged optimizer configuration. A separate 80-image,
ten-class preview of the preserved step-3,757 checkpoint verifies the published
sampler and is saved under the run's `recipe-preview/` directory.

Fresh image encoding takes about **4.4 seconds per 2,048-image update**, compared
with approximately 2.3 seconds using cached latents. Peak allocated GPU memory
remains approximately **43.7 GiB**. The online-encoding cost restores the original
augmentation distribution rather than cycling through two fixed views.

The first corrected production checkpoint at **step 3,800** passed the saved-state
audit: model architecture, optimizer groups, actual cosine LR, restored step,
all four RNG states, online-image mode, and published sampling settings match
the declared recipe. Training continued beyond that checkpoint with zero
skipped updates. Receipts are `production-recipe-audit.json` and
`recipe-correction-verification.json` in the run's preparation directory.
