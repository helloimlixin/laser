The improved FFHQ stage-2 model is continuing from epoch 60 through epoch 100.
The starting model scored FID50k 24.6127001 against the released RQ-VAE
`ffhq_256_train.npz` statistics for its full 60,000-image training set. This is
a bounded test of whether additional training helps; it does not assume the
remaining gap to RQ-Transformer's reported 10.38 will close.

The epoch-47 tokenizer remains frozen. Training retains normalized per-position
scale weights `[1, 8, 4, 2, 1]`, batch 384 across three H200s, the existing
stochastic token cache, AdamW state, and per-rank random-number states. The
original 50-epoch learning-rate schedule remains authoritative; additional
epochs hold its final learning rate, 0.00003, without restarting warmup or decay.

At epochs 70, 80, 90, and 100, checkpoint selection generates 50,000 images
against the same released RQ training reference. Evaluation uses atom
temperature 0.6, coefficient temperature 1.0, atom top-k 250, both top-p values
1.0, CFG 0.0, seed 73000, and batch 64 per rank. Decoding is FP32 and Inception
receives continuous [0,1] pixels without uint8 rounding. Previews are 8-by-8.
The original training split differs from RQ's shuffled split even though these
evaluation statistics match RQ's released reference.
The subsequent transform audit also found that our training data uses LANCZOS
downsampling from 1024 to 256, whereas RQ's published real-image evaluation
transform uses BILINEAR. The FID Inception preprocessing matches; full data
preparation does not. See `ffhq-fid-transform-audit-2026-09-22.md`.

The best-FID checkpoint is initialized from the confirmed epoch-60 model with
score 24.6127001. Subsequent checkpoints replace it only when their full
50,000-sample FID improves. The prior 2,000-sample held-out diagnostic score is
not carried across as the selection threshold. The explicit transition receipt
records the old/new contracts, parent checkpoint hash, unchanged training
objective, optimizer/RNG transfer, and baseline evaluation artifact.

The training contract now records the selected FID protocol, reference checksum,
sample count, evaluation batch size, and pixel convention. Changing these on
resume is rejected. Existing recipes retain their previous held-out diagnostic
selection behavior. Twenty-nine focused tests passed, including protocol
routing, rejection of shortened evaluations, reference-change resume guards,
continuous evaluation pixels, compound sampling, optimizer schedule extension,
checkpoint transfers, and supervisor behavior.

The supervisor retries failed workers up to three attempts and restarts workers
that stop reporting progress. Full evaluations emit sampling progress. Local
checkpoints are saved every 200 optimizer updates and each epoch. Last,
best-FID, and best-validation checkpoints upload asynchronously at the initial
resume, after each full FID evaluation, and at completion; the supervisor
verifies the final committed online last/best-FID files. Uploaded bundles
include the resolved sampling/selection configuration and FID result records.

Run: [W&B continuation](https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhq256-var341-stage2-continue100-20260922).
Recipe: `configs/experiments/ffhq256-var341-stage2-continue100.yaml`.
Run directory: `outputs/ffhq256-var341-stage2-continue100-20260922/`.
Checkpoint directory: `/tmp/laser-var-checkpoints/ffhq256-var341-stage2-continue100-20260922/train/`.
Progress is recorded in `pipeline-status.json` and `train/status.json`; completion
and online verification are recorded in `train/complete.json` and
`train/online-checkpoints-verified.json`. The completed epoch-60 experiment and
its online checkpoint bundle remain preserved.
