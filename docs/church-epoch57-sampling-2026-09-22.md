# Church epoch-57 sampling comparison

This experiment keeps the best compound LASER checkpoint fixed and changes only
the sampling distribution. The selected checkpoint is epoch 57, optimizer step
3534, from `church-laser-scratch-rqrecipe300-b2048-h200x5-20260921`. Its recorded
FID50k is 9.95524456181542. The newer 32769-token integer-model training continues
in its own processes throughout this evaluation.

Completed: 16 selection settings and two fresh-seed confirmations, 900,000
generated images in total. The selected change is coefficient temperature 1.20;
atom temperature remains 1, atom top-k 250, and both top-p values 1. Coefficient
sampling uses all 2048 bins. Selection-seed FID improved 9.955245 → 9.820518;
fresh-seed FID improved 9.912260 → 9.835447. The two-seed means are 9.933752 and
9.827983, a 1.065% reduction. This is a modest metric improvement and does not
establish that structural distortions have been resolved.

The complete report and verified online artifact are in
`outputs/church-epoch57-sampling-sweep-20260922/report.md` and
`online-report-verification.json`. W&B report artifact:
`helloimlixin-rutgers/laser/church-laser-epoch57-sampling-sweep-20260922-report:v0`.

Evaluation run:
https://wandb.ai/helloimlixin-rutgers/laser/runs/church-laser-epoch57-sampling-sweep-20260922

The unchanged full checkpoint is already online in
`helloimlixin-rutgers/laser/church-laser-scratch-rqrecipe300-b2048-h200x5-20260921-selected-checkpoints:v49`,
file `best-fid-01.pt`. Its manifest MD5 is `CEhVoXlTJu958WE0AKGn6g==` and its size
is 4,857,570,762 bytes. Local weights are preserved at
`/tmp/laser-church-integer-raw-rqrecipe300-20260922/compound-handoff/best_fid_9.9552_epoch_057.pt`.

## Comparison protocol

- Each scored candidate generates exactly 50,000 images across five GPUs,
  batch size 2048 per GPU, with decoder and Inception chunks of 64.
- Baseline atom sampling: temperature 1, top-k 250, top-p 1.
- Baseline coefficient sampling: temperature 1, full 2048-bin vocabulary,
  top-p 1. Coefficients remain in physical units with no coefficient clipping.
- Exploration uses the original seed base 20261921 plus rank, resetting before
  every candidate. The selected configuration and baseline are then compared
  on a second seed that was not used for selection.
- Atom and coefficient temperatures and filtering are varied separately before
  testing combinations. Results from all attempted settings are retained.
- Weights, tokenizer, frozen model runtime, precision, decoding, image count,
  GPU partitioning and official RQ-VAE FID implementation stay fixed. Inception
  is constructed after each seed reset as in the original evaluator.
- Real-image statistics contain all 126,227 LSUN Church training images with
  the same RGB, bilinear short-side resize to 256, center crop to 256, and
  normalization as the stage-2 token-cache input. No validation images are in
  the reference. SHA256 of the statistics file:
  `ad3b5a341831d877110afdff37d6fff4be855612053e3eb9e2e7119f5593d364`.
- Sample grids show the first 64 rank-zero images already used for FID;
  they do not add sampling draws.
- Each inference process has a 28% GPU-memory allocator limit to coexist
  with the active trainer. A distinct distributed process group isolates the
  evaluations from training collectives.

The experiment directory is
`outputs/church-epoch57-sampling-sweep-20260922`. `provenance.json` records exact
asset/source hashes; `queue.json` records all candidates; `results.json` and
`results.csv` contain measurements. The final `selection.json` and `report.md`
record the selected settings and independent-seed comparison when completed.

FID is being tuned against the requested training-image reference. A second
seed checks sampling variability; it does not provide a held-out dataset
evaluation or guarantee a globally optimal sampler.
