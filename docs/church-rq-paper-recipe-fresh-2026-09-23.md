The user requested a fresh Church stage-two run matching the original
RQ-VAE/RQ-Transformer recipe. The selected basis is the published paper's
stage-two recipe. The checked-in Church YAML differs from that paper: its
batch is 256 and sampler top-k is 250. The paper specifies batch 2,048 and
Church sampling top-k 1,400. These sources are recorded separately rather
than treating them as one configuration.

Primary sources: [paper appendix A.3 and B.1](https://arxiv.org/html/2203.01941v2#A3),
[released Church YAML](https://github.com/kakaobrain/rq-vae-transformer/blob/main/configs/lsun-church/stage2/lsun-church256-sqgan-8x8x4-350M-simp.yaml).

The new run is
`helloimlixin-rutgers/laser/church-compound-rqpaper-b2048-4h100-20260923`.
Stage-two model, optimizer, scheduler and RNG initialize fresh with seed 0.
The previous run's output and selected checkpoints are preserved.

| Setting | New recipe |
| --- | --- |
| Optimizer | AdamW, betas 0.9/0.95, weight decay 0.0001 |
| Peak LR / minimum LR | 0.0005 / 0 |
| Schedule | Cosine for 300 epochs; no warmup |
| Effective batch | 2,048 |
| Hardware layout | Four H100s; batch 128 per GPU, four accumulation steps (after memory recovery) |
| Gradient clipping | Global norm 1 |
| Transformer backbone | Width 1,024; 24 spatial and four depth layers; 16 attention heads |
| Objective adaptation | Equal atom/coefficient CE weights, no geometry or regression auxiliaries |
| Atom sampler | Temperature 1, top-k 1,400, top-p 1 |
| Coefficient sampler | Temperature 1, unrestricted top-k, top-p 1 |
| FID | 50,000 samples every five epochs, original RQ evaluator and official Church reference |

Matching these transferable settings does not turn LASER into the original
RQ model. The frozen LASER tokenizer, full autoregression over atom IDs and
coefficients, dictionary atom-vector conditioning, two-layer coefficient
conditioner, and depth-specific coefficient heads remain. Atom supports are
deterministic cached OMP supports. Coefficients use normalized soft targets
at temperature 0.5 and stochastic coefficient-token contexts, preserving the
FFHQ-derived representation. The original RQ paper instead constructs soft
targets and stochastic contexts from residual-code embedding distances.
Their numerical target temperatures are not equivalent across representations.
Stage one is not retrained: the selected three-epoch fine-tuned LASER tokenizer
is retained, rather than the paper's original one-epoch RQ-VAE fine-tune.

Training reuses the validated original-image cache, containing 126,227
8x8x4 examples. It does not train on decoder reconstruction images. Cache
SHA256 is `4c03555ebedca1bf7b74f1db30e91752586889337a58547e856416c3aeeae586`;
tokenizer SHA256 is
`762c51a10267ed6fa55709ff0d6cf997940d21056b16c7a0a0399b3ebf93868d`.

FID retains bilinear 256-to-299 interpolation with `align_corners=False` and
`antialias=False`, and continuous float32 RGB. The official real-reference
SHA256 is `809489d8316b9e6eb9dc3bc021b6d602f4b6d816cc80621c6b9c189a9253a7f6`.
See the [resize and learning-dynamics audit](church-learning-dynamics-resize-audit-2026-09-23.md).

With the existing loader's complete-batch policy, each shuffled epoch has
61 optimizer updates and processes 124,928 images; its last 1,299 images are
omitted for that epoch. This omission changes with each epoch's shuffle.
The resulting cosine horizon is 18,300 updates. The paper does not publish
the exact stage-two loader rounding rule, so this small implementation
difference is explicit.

The recipe uses BF16, fused AdamW, cached coefficients in RAM, the validated
compiled coefficient attention, and the corrected asynchronous checkpoint
storage/upload path. Full `last.pt` recovery checkpoints are saved every
250 updates and every five epochs. Full best-FID checkpoints upload when
selected. The first best checkpoint can exist only after an evaluation.

Reproducibility files are under
`outputs/church-compound-rqpaper-b2048-4h100-20260923/`.
The frozen runtime is `/mnt/laser-church/runtime-rqpaper`;
working validation files are under
`/mnt/laser-church/original-recipe-20260923/`.
The supervisor now records its configured recipe source instead of always
labeling every run as the FFHQ reference. The recovery helper now reports
the actual configured batch and accumulation in its launch receipt.

Validation: 24 existing full-pair/objective tests pass.
An independent accumulated-gradient comparison through the actual compound
objective matched a single full batch within 7.45e-9 absolute error.
The four-GPU batch-128 training preflight completed two updates, peaking at
33.31 GiB allocated per GPU. The selected batch-256 preflight also completed
two updates, at 60.89 GiB allocated and 71.92 GiB reserved per GPU. Its full
checkpoint contains finite model and optimizer tensors, all four RNG states,
and the expected cosine scheduler and accumulation cursor. A separate reload
generated 128 finite images per rank using the production sampler, peaking
at 56.32 GiB allocated per GPU. These disposable preflight weights are not
used for production initialization.

The old run stopped at step 36,976 after epoch 75, with best FID 13.271944
from epoch 55. Both full online checkpoints match their local MD5 values:
`last.pt` is `5829fefd4f9836fea730df778ee924b3` and `best-fid-01.pt` is
`57b119e648c59fac178f81bcd8ffb4c4`. Both have optimizer and four-rank RNG state.
The final upload took several minutes to drain; the new recovery cadence is
spaced to avoid requesting a multi-gigabyte upload every few training minutes.

After a production recovery checkpoint exists, restart this same run with:

```bash
python outputs/church-compound-rqpaper-b2048-4h100-20260923/resume.py
```

Live production verification reached optimizer step 30 with finite loss
8.375698 and gradient norm 0.051595. The logged LR 0.0004999966844992657
matches the new cosine schedule. W&B confirms `resume=false`, no stage-two
initialization checkpoint, batch 256 per GPU, accumulation 2, effective batch
2,048 and all four ranks. An initial compute window measured 2,404.57 images/s,
or about 52 seconds per epoch, excluding FID, previews and checkpoint overhead.
GPU utilization medians were 100.0%, 100.0%, 100.0%, 100.0%.

The W&B console resumes the run ID created by the supervisor; this is not a
model-weight resume. Training starts with a fresh optimizer and scheduler at
step zero. The first generation FID and best-checkpoint selection are scheduled
at epoch five. New-run checkpoint uploads are enabled; the prior run's already
verified checkpoint receipts are recorded separately.

The initial batch-256 launch later ran out of GPU memory during backward just
after the step-600 preview. Rank 0 needed a 4.00 GiB allocation with only
3.61 GiB free. PyTorch reported 67.45 GiB allocated and 5.54 GiB reserved but
unallocated. The short preflight did not cover this later memory peak; batch
256 did not leave adequate headroom. Epoch-five FID and the step-500 recovery
save had both completed successfully before the failure.

Recovery uses batch 128 per GPU and accumulation four, retaining effective
batch 2,048, all four GPUs, model, objective, optimizer and 300-epoch schedule.
The complete step-500 checkpoint has finite model and optimizer state and four
RNG streams. The epoch-eight batch cursor remaps from 24 to 48, preserving the
same count of already-consumed examples. Approximately 100 updates are replayed.
The saved LR is 0.0004990795908619189 at schedule step 500/18,300.
The selected external recovery config is `recovery-b128.yaml`; the frozen
source archive and original batch-256 launch remain available as provenance.
Recovery checks and subsequent live verification are stored in
`memory-recovery/`. The resume helper follows `resume-runtime.json` to retain
the smaller per-GPU batch on future recoveries.

Recovery verification passed through step 780, including epoch-ten FID-50k
(54.389655), resumed training, and the persisted step-750 recovery checkpoint.
Finite loss/gradient norm and the expected cosine LR were verified. Stable
training windows were approximately 2,270 images/s, or about 55 seconds per
epoch excluding evaluation, previews and checkpoint overhead. Peak allocated
memory was 36.32 GiB/GPU at the first resumed training step and 57.92 GiB/GPU
during FID. The full verification receipt is `memory-recovery/live-verified.json`.
