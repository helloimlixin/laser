The Church paper-batch run plateaued: best FID50k was 19.486483 at epoch 35,
followed by scores mostly between 22 and 27 through epoch 110. It was stopped
with a complete recovery checkpoint at step 6,711. The parent run and its best
checkpoint are preserved.

The active tokenizer is the selected three-epoch Church LASER fine-tune, SHA256
`762c51a10267ed6fa55709ff0d6cf997940d21056b16c7a0a0399b3ebf93868d`, with previously
measured full-population reconstruction FID 2.639338. The active FP32 cache
hash is `4c03555ebedca1bf7b74f1db30e91752586889337a58547e856416c3aeeae586`.
Both hashes were recomputed. A new independent check of 16 original LMDB
photographs matched every atom ID; maximum normalized coefficient error was
4.59e-6. Training does not re-encode cached reconstructions. The original RQ-VAE
Church FID reference remains unchanged, including bilinear Inception resizing.

The actual successful FFHQ run is
[ffhqcmp0804205803](https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhqcmp0804205803).
It used two sparse depths and 2,048 atoms, effective batch 128, LR 5e-4, a
200-epoch cosine, atom CE weight 1.5, normalized coefficient soft targets at
temperature 0.5, atom top-k 250 and coefficient top-p 0.85. Its FID 8.174393 is
for FFHQ, not Church. The Church run retains four sparse depths and 16,384 atoms;
matching numerical target temperature alone does not match latent corruption.

A new frozen-decoder probe used 256 randomly selected cached original images,
fixed true atom IDs and coefficients, coupled sampling uniforms, and the
actual training target implementation. Scores below measure perturbation from
the continuous clean tokenizer reconstruction, not reconstruction against the
photograph or unconditional generation FID.

| Coefficient construction | PSNR to clean reconstruction | Relative latent MSE |
| --- | ---: | ---: |
| Nearest bin | 70.539 dB | 0.000000783 |
| Current normalized T=0.5, full sampling | 17.930 dB | 0.271823 |
| Normalized T=0.5, nucleus p=0.85 | 19.957 dB | 0.142550 |
| Physical T=0.5, nucleus p=0.85 | 31.617 dB | 0.006699 |
| Physical T=0.125, full sampling (training distribution) | 34.747 dB | 0.003176 |
| Physical T=0.125, nucleus p=0.85 | 37.398 dB | 0.001691 |

Current untruncated target standard deviations are 48–52% of the physical
coefficient RMS at each depth. Physical T=0.125 gives standard deviation 0.25
in the dictionary's coefficient units, or 3.4%, 6.2%, 9.5%, and 14.7% of the
measured depth RMS. This temperature also has precedent in the earlier Church
compound run that achieved FID50k 11.85. These diagnostics support a target-noise
correction; they do not establish the cause of the entire generation FID gap.

The repair is a separate fine-tuning experiment initialized from the parent's
best epoch-35 model. It keeps the full atom/coefficient autoregressive chain,
dictionary-vector conditioning, two-layer coefficient micro-transformer,
depth-specific classifiers, tokenizer, cache, and unclipped coefficient bins.
It changes the target metric to physical units at T=0.125 and restores FFHQ's
atom weight 1.5, effective batch 128, atom top-k 250 and coefficient top-p 0.85.
Four H100s each process 32 examples per update. Geometry remains disabled,
as in the better earlier Church physical-target baseline; the archived FFHQ
geometry objective has a documented conditional-expectation error.

Because this is a repair fine-tune, the optimizer and schedule start afresh at
LR 1e-4 with a 50-epoch cosine to zero, rather than treating it as another
fresh 200-epoch FFHQ run. AdamW betas (0.9, 0.95), weight decay 1e-4 and clipping
at one remain the same. FID50k is scheduled every five epochs; complete last
and best-FID checkpoints are uploaded online. A separate immutable initializer
preserves the source weights independently of parent-run checkpoint retention.

Thirty-one focused tests passed for coefficient targets, full-pair causality,
cached autoregression and pair attention. Numeric data, source, image grids,
configuration and subsequent live receipts are retained in
`outputs/church-ffhq-repair-20260924/`. The four-GPU preflight passed two actual updates, full-state reload and
128-image generation per GPU. Model/optimizer tensors and generated pixels
were finite. Peak allocated memory was 11.08 GiB during the first update and
40.70 GiB in the generation preflight. The cosine horizon is 49,300 updates.

A matched frozen-checkpoint comparison used 50,000 generated images for each
sampler, seed 240925, four ranks and the same decoder/evaluator/reference.
Paper sampling (atom k1400, coefficient p1) scored 19.476667; FFHQ sampling
(atom k250, coefficient p0.85) scored **13.701683**. These are identical model
weights; the improvement is a sampler effect, not a result of repaired training.
This is one evaluation seed. The source checkpoint and both sample grids are
preserved in the audit directory.

The repair run is online at
https://wandb.ai/helloimlixin-rutgers/laser/runs/church-compound-ffhq-adapt-4h100-20260924.
Its first scheduled training FID is after five additional epochs. The initial
comparison for that fine-tune is 13.701683 under the same sampler. Improved
target fidelity must still be confirmed by generated FID and samples after
training. Both parent last/best files were verified online by complete MD5
agreement before the repair launch.

The first repaired generation evaluation completed after five additional epochs,
step 4,930: **FID50k 11.691016**. The unchanged-weight FFHQ-sampler baseline
was 13.701683, and the matched paper-sampler baseline was 19.476667. The
fine-tune evaluation uses the same 50k-sample protocol and sampler, with new
generation draws; this is one checkpoint/evaluation, not a statistical claim
or a guarantee that remaining structural artifacts are resolved. Peak live FID
allocation was 59.88 GiB on rank zero and 56.86 GiB on the other three ranks.
The best and latest complete recovery states are selected for online upload.
