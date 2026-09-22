The FFHQ evaluation now uses a verified reference built from the exact
60,000 cached training images. The tokenizer and trained weights are preserved.
Generated samples and real images use the same released FID Inception weights,
bilinear 299-by-299 resize (align_corners=False, antialias=False), and input
normalization. Real images have no random crop or flip; generated FP32 pixels
are scored directly without a PNG/uint8 roundtrip. Grids remain 8-by-8.

| Checkpoint | FID50k: exact training reference | FID50k: published RQ reference |
| --- | ---: | ---: |
| Epoch 60 | 24.61224834 | 24.61270009 |
| Epoch 100 | 23.59408144 | 23.33948111 |

Each row scores the same 50,000 generated-image moments against two explicitly
separate references, each containing 60,000 real images. Epoch 60 reuses its
saved 50k moments, whose published-reference score was reproduced exactly.
Epoch 100 was generated again; its published-reference score reproduces the
completed training evaluation to within 0 FID.
The best matched score among the evaluated epochs 60 and 100 is epoch 100.
Epochs 70, 80, and 90 retain their original published-reference scores and have
not been relabeled as matched-reference measurements.

The resize/split audit extracted both Lanczos and bilinear views of all 70,000
FFHQ images and formed four 60,000-image references. Rebuilding the RQ split
with its published bilinear preprocessing reproduces its released statistics
to about 1.74e-8 FID. Resize alone gives about 1.71 real-to-real FID; split alone
gives about 0.09. These distances are not additive corrections to model FID.
The actual matched-reference evaluations above measure the impact directly.

The matched-reference checksum is `e2eb3396c6a9b7c80d61422580524260cddc802017c195d00f3ac93c29c053ef`.
The reference manifest records the training IDs, dataset fingerprints and
cache checksums, source resize method, Inception implementation and weight
checksums, normalization, pixel convention, and software versions. Evaluation
rejects reference/checksum/split/data/preprocessing mismatches, conditional
teacher prefixes, or a sample count other than 50,000 for this benchmark.
The reference manifest checksum is also part of the future training resume
contract, preventing best-FID comparisons across silently changed protocols.
Twenty-seven focused tests passed for these checks and existing evaluation,
sampling, and resume behavior.

Use `configs/experiments/ffhq256-var341-matched-evaluation.yaml` and
`scripts/tools/investigate_ffhq_var_sampling.py`, with the reference manifest,
saved `sampling-plan.json`, and 50,000 samples. The working inference alias
`configs/experiments/ffhq256-var341-stage2-fixed.yaml` points to this profile.
The historical epoch-60 profile remains in its immutable run directory and
online artifact. The stored epoch-100 training checkpoint retains its original
training contract and published-reference score; this evaluation's manifest
and result are authoritative for the matched-reference score. Any subsequent
training must make an explicit metric-contract transition before resuming.

The matched-reference column must not be compared directly with RQ's published
10.38, which uses different real statistics. The published-reference column is
retained for that benchmark comparison, with the existing training-data caveats.

Run: https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhq256-var341-matched-fid50k-20260922
Local evaluation: `outputs/ffhq256-fid-consistency-20260922/`.
