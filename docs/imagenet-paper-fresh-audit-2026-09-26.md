# ImageNet fresh augmentation and optimizer-budget audit

The corrected experiment uses the published 1.4B RQ-Transformer optimization
budget with the existing LASER orthogonal tokenizer. It starts a fresh prior,
optimizer, and 100-epoch cosine schedule. It does not inherit the stopped
double-batch experiment's weights or reinterpret its epochs.

Run: `helloimlixin-rutgers/laser/imagenet-rfid421-orthogonal-paper-fresh-20260926`.
The production launch and checkpoint verification receipts are stored under
`outputs/imagenet-rfid421-orthogonal-paper-fresh-20260926/`.

## Optimizer and data budget

| Setting | Reference recipe | Corrected experiment |
| --- | --- | --- |
| ImageNet training population | 1,281,167 | Same, official archive MD5 verified |
| Effective batch | 2,048 | 2,048 |
| Physical batching on eight GPUs | YAML microbatch 8 implies accumulation 32 | 256 per GPU, accumulation 1 |
| Updates per epoch | Approximately 625–626; stage-2 driver unreleased | Exactly 625 |
| Updates over 100 epochs | Approximately 62,500–62,600 | Exactly 62,500 |
| Images processed per epoch | Depends on unreleased tail handling | 1,280,000; shuffled tail of 1,167 omitted |
| Optimizer | AdamW | Fused AdamW, FP32 weights and state |
| Initial LR | 0.0005 | 0.0005 |
| Betas / weight decay | (0.9, 0.95) / 0.0001 | Same |
| Epsilon | Not specified in YAML | 1e-8 (PyTorch AdamW default) |
| Gradient clipping | Norm 1.0 | Once after accumulation and DDP synchronization |
| LR schedule | Cosine to zero, no warmup | 62,500-step cosine to zero, no warmup |
| Image transform | Resize short edge 256, random 256 crop, flip p=0.5 | Same transform, freshly drawn per image/epoch |
| Evaluation / preview cadence | YAML test every 2 epochs | User override: FID every 5 epochs; preview every 200 updates |
| Generation sampler | YAML top-k 16,384, top-p 0.92 | Atom top-k 16,384, top-p 0.92; coefficient top-p 0.92, full 2,048-bin vocabulary |

The loader uses a distributed shuffled sampler and drops incomplete physical
batches. It also drops incomplete accumulation groups. For the selected shape:
`ceil(1,281,167 / 8) = 160,146` samples per rank, then
`floor(160,146 / 256) = 625` complete batches. The same 625 updates result from
microbatch 8 and accumulation 32. Training processes 128 million image visits
over 100 epochs. No extra factor of eight belongs in the optimizer-update count.

The compound objective divides by accumulation exactly once. DDP averages
gradients across ranks. Unsynchronized microbatches use `no_sync`; clipping,
AdamW, cosine stepping, gradient clearing, and the global-step increment occur
once per completed update. BF16 training has no FP16 GradScaler skip behavior;
a nonfinite gradient stops training instead of silently advancing the schedule.

The main `train/loss` metric now averages the entire optimizer batch across all
accumulated microbatches and ranks. Previously it reported rank zero's last
microbatch, so batch-layout changes could distort apparent loss comparisons.
Detailed atom/coefficient diagnostics retain their explicitly recorded rank-zero
microbatch scope. Update throughput, cumulative image visits, and fractional
epoch progress are logged separately.

An eight-GPU numerical audit compared microbatch/accumulation pairs 8/32,
128/2, and 256/1 against a full 2,048-example forward of the orthogonal model
and coefficient-history decoder, with dropout disabled to isolate batching.
Maximum absolute gradient differences were below 3e-8. All configurations
advanced Adam and the scheduler once. FP32 reduction noise on nearly zero
attention-bias gradients can produce first-step Adam differences up to 1.2e-5;
this is mathematical batch equivalence, not bitwise trajectory equivalence.

## Fresh-image correctness

The complete ImageNet training archive was fetched from the pinned mirror
revision recorded in the original cache. Its 147,897,477,120 bytes match the
official MD5 `1d675b47d978889d74fa0da5fadfb00e`. Extraction verifies 1,000 class
directories and 1,281,167 images. Sorted synset directories define class IDs.

`EpochImageFolder` draws the original Resize/RandomCrop/RandomHorizontalFlip
transform from an image-and-epoch seed. Persistent workers see epoch changes,
and resuming a sampler cursor reproduces the same crop independently of worker
prefetch and rank. Training does not consume the two-view token cache.

Orthogonal mode previously required a token cache. The general online encoder
returns dictionary coefficients; feeding those directly to the orthogonal
decoder would change the represented latent. The newly enabled fresh-image
path converts physical dictionary coefficients to orthogonal coefficients with
the same ordered-support basis used to create the original cache, then applies
the recorded per-depth coefficient scales. A reconstruction-equivalence test
checks this conversion, and a chunking test checks that changing frozen-encoder
batch size preserves the result. The frozen encoder uses FP32 with TF32
convolutions; OMP and coordinate conversion use FP32 matrix arithmetic.

## Scope of comparison

The spatial/depth backbone matches the supplied geometry: 8×8×4, width 1,536,
42 spatial blocks, 6 depth blocks, 24 heads, input width 256, residual/MLP
dropout 0.1, attention dropout 0, shared embeddings and cumulative depth context.
LASER adds its atom/coefficient factorization and two coefficient-history
layers at width 512, giving 1,466,384,384 parameters. Its operations are consequently not
identical to the paper's model.

The tokenizer remains the user's frozen rFID-4.21 sparse tokenizer, with 16,384
atoms and 2,048 scalar coefficient bins. It uses deterministic OMP supports and
stochastic coefficient inputs with physical soft-target temperature 0.03125.
The paper instead stochastically samples residual codebook codes at temperature
0.5. Those temperatures measure different quantities; literal substitution
would not reproduce the paper's noise distribution. The loss averages atom
NLL and coefficient soft CE with equal weight and has no geometry auxiliary
loss. These are explicit experimental differences, not paper-reproduction
claims. BF16 transformer arithmetic also differs from the original FP16-era AMP
implementation.

FID uses 50,000 generated images, the original RQ TensorFlow-FID Inception
implementation, and released ImageNet training statistics. Sampling is balanced
at 50 images per class, with no rejection sampling. The YAML sampler is the
comparison setting; a released checkpoint may use a different selected sampler.
Published scores obtained with rejection sampling are not direct targets for
this evaluation. The earlier W&B FID-16 run used a different tokenizer and
evaluation reference, so its raw score is not directly comparable either.

The released repository explicitly omits the transformer training driver.
Its exact final partial-batch policy, optimizer grouping, loss-scaling details,
and historical runtime cannot be independently verified. This experiment
matches the published optimization settings and augmentation with documented
LASER adaptations; it is not an exact RQ-VAE reproduction.

## Validation and preserved history

Twenty-one focused tests passed: fresh crop changes and resumed crops with
persistent workers, online coordinate conversion and chunking, actual loader
update counts, orthogonal codec, coefficient-history behavior, checkpoint
resume, and learning-rate scheduling. One unrelated VAR-specific test was
excluded because the frozen runtime does not contain FoundationVisionVAR.
The separate accumulation audit passed on all eight H200s. The full 1.466B
model also completed 60 fresh-image updates on all eight ranks: all 870 model
tensors and Adam states were finite, optimizer steps agreed, and the live LR
matched the exact 62,500-step cosine schedule. Steady throughput was about
2,020 images/s (0.99 updates/s), including fresh frozen-encoder work.

A matched four-GPU fresh-image control also passed 60 updates, using physical
batch 256 and accumulation 2 to retain global batch 2,048. Excluding startup
and the final full-state check, median throughput was 1,040.805 images/s on four
GPUs and 2,021.720 on eight: **1.9425× optimizer-update throughput**. The
eight-GPU result is **1.2345×** the original four-GPU cached-view throughput;
fresh tokenization changes the workload. It does not meet the original 2×
target relative to that cached workload. The comparable fresh-workload scaling
is close to twofold. Training compute alone is about 17.6 hours for 100 epochs;
previews, evaluation, and recovery snapshots add overhead.

Production passed update 360 after generating the first 64-image preview at
update 200. The complete 17,597,810,669-byte update-200 checkpoint passed a
strict model/Adam/scheduler reload, checks of all 870 model and optimizer
tensors, all eight RNG streams, and a full SHA-256 comparison between local
serialization and persistent storage. Both copies hash to
`71855f3998faa30cb1a078c8da9d65af40ef889e822247a44bfc81f54b596b90`.
Production steady throughput through update 200 was 2,013.7 images/s. The first
checkpoint-plus-preview pause added 26.2 seconds. Extrapolating that cadence
and the previous eight-GPU evaluator's approximately 386 seconds per FID gives
about 22.1 hours total; allow **22–24 hours** for the 100-epoch run. This is an
estimate, not a completed-run measurement.

The earlier doubled-batch run
`imagenet-rfid421-orthogonal-8h200-20260926` was stopped at update 1,060 and its
full recovery checkpoint preserved. Its global batch 4,032 would have yielded
31,700 updates over 100 epochs. It is superseded by this fresh experiment.
The source run and its checkpoints are preserved.

A disposable 200-update continuation of the source at global batch 2,016
measured approximately 3,206 images/s on eight GPUs, or 1.96× the four-GPU
cached-view baseline. That benchmark excludes fresh image encoding and must
not be used as the corrected run's throughput estimate. More optimizer updates
per second is measurable; twice the quality improvement is an empirical
hypothesis to assess with loss and FID at matched update/image budgets.

## Primary sources

- [Official 1.4B configuration](https://github.com/kakaobrain/rq-vae-transformer/blob/main/configs/imagenet256/stage2/in256-rqtransformer-8x8x4-1400M.yaml)
- [Paper and appendix](https://arxiv.org/html/2203.01941v2)
- [Official ImageNet transforms](https://github.com/kakaobrain/rq-vae-transformer/blob/main/rqvae/img_datasets/transforms.py)
- [Official scheduler helper](https://github.com/kakaobrain/rq-vae-transformer/blob/main/rqvae/optimizer/scheduler.py)
- [Official architecture defaults](https://github.com/kakaobrain/rq-vae-transformer/blob/main/rqvae/models/rqtransformer/configs.py)
- [Released-code coverage and evaluation protocol](https://github.com/kakaobrain/rq-vae-transformer#evaluation-of-rq-transformer)
