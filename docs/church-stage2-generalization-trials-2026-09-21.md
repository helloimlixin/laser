These queued trials were superseded before training by the user's request for
a fresh transformer with the original Church recipe. Preparation and monitoring
verification completed; the augmentation/dropout trials did not launch. See
[the scratch-run record](church-released-recipe-scratch-2026-09-21.md).

The prepared stage-2 trials use **all 126,227 Church training images**.
The 300 held-out images are the official validation split and are never used
for optimizer updates. No validation subset is removed from the training set.
A fixed random 300-image training probe provides a cheap comparison with the
validation losses; that probe does not determine the training population.

Output: `outputs/church-stage2-generalization-20260921/`.
Online run: [church-laser-ft3best-stochastic-rawcoeff90-h200x5-20260921](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-laser-ft3best-stochastic-rawcoeff90-h200x5-20260921).

Held-out monitoring measures how well the current prior predicts sparse codes
on unseen Church images. It runs without gradients and reports atom NLL and
coefficient KL, both overall and separately at each sparse depth. The fixed
support draws and coefficient histories make epochs comparable. Both training
and validation probes use evaluation mode; monitoring restores model mode and
the CPU/GPU RNG streams afterward. Five ranks partition each split without
duplication and aggregate image-level means and standard errors.

The initial audit found train/validation atom NLL 3.9371/11.1714 at epoch 60
and 3.7594/11.3860 at epoch 71. This is strong evidence of worsening
generalization during the LR-only continuation. It does not show that an
augmentation or a particular dropout rate will improve generated FID.

All trials start independently from the preserved full epoch-60 model,
AdamW state, and five rank RNG states. Their common controls are global batch
960, LR 3e-5 with the already tested 30-epoch cosine/plateau controller,
coefficient target temperature 0.125, and the original generation sampler.
Each pilot trains six epochs, after an isolated 12-update preflight. FID uses
50,000 generated images every epoch, alongside the held-out measurements.

| Order | Intervention | Comparison |
|---|---|---|
| 1 | Pixel-space horizontal flips, p=0.5; fresh OMP | Completed fresh-OMP control with identical LR and no flips |
| 2 | Residual dropout 0.1 → 0.2 | Original bank and unchanged target/loss settings |
| 3 | Normalized depth weights [2,1,0.5,0.5] on atom and coefficient losses | Original bank and residual dropout 0.1 |

The flip cache covers every training image, using FP32 frozen-encoder
inference with TF32 disabled. It flips the preprocessed image pixels and
re-encodes them; it does not flip latent grids. Orientation is drawn once per
image visit using the checkpointed rank RNG, followed by fresh stochastic OMP.
Original pixels and native-cache row alignment are checked against recorded
probes before preparation. All cache rows must be finite before training.

Full latest and best-FID checkpoints are uploaded online with remote MD5/size
verification. Only a strict improvement over the original FID 10.6884632 is
promoted. A winning policy may continue to epoch 90. If no pilot wins, the
original best remains selected and the unsuccessful baseline continuation is
not repeated. Validation loss informs diagnosis but does not replace FID as
the requested selection criterion.

The prior continuation was preserved at epoch 70, step 9,170, in
`/tmp/laser-stage2-generalization-20260921/handoff/last.pt` before stopping its
supervisor and workers. The bootstrap publishes that full state and the
original best before starting the new online writer.

The original RQ release is an important comparison, with discrepancies between
its example configurations and other published evidence:

- The released LSUN transform uses resize to 256, center crop, and normalization;
  it adds no random crop or horizontal flip. Our flip trial is an additional
  experiment. [Transforms](https://github.com/kakaobrain/rq-vae-transformer/blob/main/rqvae/img_datasets/transforms.py).
- The example stage-2 YAML specifies AdamW, LR 5e-4, betas (0.9,0.95), decay
  1e-4, clipping 1, no warmup, global batch 256, and 300 epochs. Its scheduler
  is cosine to zero. [YAML](https://github.com/kakaobrain/rq-vae-transformer/blob/main/configs/lsun-church/stage2/lsun-church256-sqgan-8x8x4-350M-simp.yaml),
  [scheduler](https://github.com/kakaobrain/rq-vae-transformer/blob/main/rqvae/optimizer/scheduler.py).
- The paper's stage-2 batch is **2,048**. The earlier released-tokenizer control
  uses that value, not the example YAML's 256. [Paper, Appendix A.3](https://arxiv.org/html/2203.01941#A3).
- Residual dropout is 0.1; attention and embedding dropout are zero. These are
  already our baseline settings. [Defaults](https://github.com/kakaobrain/rq-vae-transformer/blob/main/rqvae/models/rqtransformer/configs.py).
- The example stage-1 YAML uses Adam, betas (0.5,0.9), zero weight decay,
  constant LR 4e-5, one epoch, and zero dropout. The locally preserved config
  bundled with the released tokenizer instead lists constant LR 4e-6, three
  epochs, and global batch 128 for generator and discriminator training.
  The checkpoint's actual completed epoch count is not available from that
  config alone. [Example](https://github.com/kakaobrain/rq-vae-transformer/blob/main/configs/lsun-church/stage1/church256-rqvae-8x8x4.yaml),
  [bundled config](/workspace/tmp/original-rqvae-church-published/extracted/church/stage1/config.yaml).

The released pair was previously evaluated at FID50k 7.6710 versus the reported
7.45, leaving an unexplained 0.2210 residual. The fresh original-prior control
using the released tokenizer reached best FID50k 10.073105. Its current online
record is finished at step 15,313, approximately epoch 247.016, with two skipped
AMP updates and latest recorded FID50k 11.378375 at epoch 240. The 300-epoch
setting must not be described as 300 verified completed epochs. The current
W&B snapshot is saved as `original-control-status.json`. The original stage-2
training pipeline was not released, so unpublished optimizer grouping and
initialization details remain unresolved reproduction variables.

The new monitoring and normalized depth-weighting helpers have 12 passing
focused tests with the existing sparse-policy and scheduler tests. Full-model
five-rank validation and all-row flip-cache checks are required by the bootstrap
before training is permitted to start. Their results are written to
`monitor-baseline.json`, `flip-cache.json`, and `validation.json`.
