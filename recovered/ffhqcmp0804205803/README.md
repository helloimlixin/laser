# Recovered FFHQ compound-v4 run

Source run: [helloimlixin-rutgers/laser/ffhqcmp0804205803](https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhqcmp0804205803).

The run's uploaded training script was downloaded directly from W&B and
preserved without edits at
[`scripts/train_official_rqtransformer_laser_stage2.py`](scripts/train_official_rqtransformer_laser_stage2.py).
It matches the existing `src/ffhq_v4_archived.py` in the parent repository byte
for byte. Its SHA-256 is:

```text
9ba1b49b4e5e339f0076bebee6fbac5629f6c391601de467019a3723c9d3e33f
```

## Contents and provenance

- `scripts/`: the exact W&B-uploaded training entry point.
- `src/`: the imported RQ-Transformer, RQ-VAE, interfaces, and label helper,
  reconstructed from the parent repository's July 31 commit `449ed55`.
  These dependencies were not included in the run's W&B upload, and its
  metadata records no Git commit. Their origin is therefore documented
  separately from the verified training script. The original directory
  layout makes the trainer import this bundle's copies of those modules.
- `records/config.yaml`, `requirements.txt`, `wandb-metadata.json`,
  `wandb-summary.json`, and `output.log`: unmodified W&B files. Download sizes
  and MD5 hashes were checked against W&B.
- `records/run.json`: the current W&B run record, with configuration values
  unpacked for reading.
- `records/launch-args.json` and `original-command.txt`: launch arguments from
  W&B's stored process metadata. The command retains the historical run ID,
  output paths, and checkpoint locations; it is a provenance record.
- `records/local-launch-config.json`, `stage1-audit.json`,
  `token-cache-validation.json`, and `final-slurm-output.log`: saved local
  records from the completed run.
- `reference/`: copies of the workspace's submission script and cache
  builder at recovery time. These were not uploaded with the run and may
  include later changes. The submission script includes evaluation flags
  that the uploaded trainer does not accept.
- `MANIFEST.json`: per-file origins, sizes, and SHA-256 hashes.
- `SHA256SUMS`: checksums for the completed recovery bundle.

## Recorded training recipe

| Setting | Value |
| --- | --- |
| Architecture | `compound-v4-micro2-rqtransformer-ffhq-350m` |
| Data | 70,000 FFHQ images, 256 x 256, unconditional |
| Frozen tokenizer | LASER A=2,048, K=2; epoch 146, rFID 6.227079379843531 |
| Compound layout | 8 x 8 x 2 = 128 atom/coefficient pairs |
| Transformer | Width 1,024; 24 spatial / 4 depth layers; 16 heads |
| Coefficient head | Two micro-transformer layers; depth-specific classifiers |
| Coefficient vocabulary | 2,048 bins on normalized [-3, 3] |
| Per-depth physical scales | [36.208333333333336, 8.583333333333334] |
| Targets and context | Normalized soft targets at T=0.5; stochastic coefficient tokens |
| Training | 200 epochs; 109,200 optimizer steps |
| Batch | 16 per GPU; 8 GPUs; global 128; accumulation 1 |
| Optimizer | AdamW, betas (0.9, 0.95), weight decay 1e-4 |
| Schedule | Cosine 5e-4 to zero over 200 epochs; no warmup |
| Precision / regularization | BF16 autocast; residual dropout 0.1; gradient clipping 1.0 |
| Objective | Atom weight 1.5; distribution geometry 0.05, delay 2 epochs, ramp 3, top-k 4 |
| Atom sampling | Temperature 1.0; top-k 250; top-p 1.0 |
| Coefficient sampling | Temperature 1.0; top-p 0.85 |

The full settings and absolute paths to the stage-1 checkpoint, token cache,
and trained stage-2 checkpoints are in `records/run.json` and
`records/local-launch-config.json`. Those large assets remain at their
existing locations; this bundle contains code and run records.

## Evaluation version boundary

The final run logged FID **8.174392700195312** at epoch 200, using 50,000 fake
images and all 70,000 FFHQ real images. The final local log records a resume
at epoch 132 / step 72,500 and explicitly resets the old best-FID history
when changing the real-image reference population from 50,000 to 70,000.

The exact uploaded trainer recovered here predates that evaluator change:
its evaluator uses the same sample count for real and generated images, and
it computes Inception Score unconditionally. The final saved configuration
contains `fid_reference_full_dataset=true`, `fid_reference_num_samples=70000`,
and `inception_score=false`, which are not switches in this uploaded source.
Consequently, this is an exact recovery of the **uploaded compound-v4
training source**, with reconstructed imports, rather than a claim to have
recovered every late runtime edit or the final evaluator. Keep this snapshot
unchanged when building a separately identified evaluation adaptation.

## Validation

The download is checked against W&B MD5/size metadata and against the known
archived SHA-256. Import/CLI and syntax checks use the existing PyTorch 2.4.1
container with Python 3.11 and the configured stage-2 dependency directory.
See `VALIDATION.json` for the completed checks. This recovery does not claim
a new training or FID reproduction.

To verify the packaged bytes from this directory:

```bash
sha256sum -c SHA256SUMS
```
