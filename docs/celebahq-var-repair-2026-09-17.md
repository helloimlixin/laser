# CelebA-HQ VAR/LASER repair, September 17, 2026

The requested source run is
[`celebahq256-var-laser-fix-20260917`](https://wandb.ai/helloimlixin-rutgers/laser/runs/celebahq256-var-laser-fix-20260917).
It reached prior epoch 39 and W&B reports `crashed`. W&B did not retain its
checkpoint or the actual training snapshot, so the exact remote crash cause
cannot be reconstructed. Its logged configuration already claims a bilinear
coefficient-head fix; the older checkout on Amarel did not contain that fix.
The repaired local head implements and tests the interaction explicitly. This
does not establish that the missing local change caused the source run's plateau.

The source's generation curve mixes sample counts:

| Epoch | Generated images | Reported FID | Held-out joint NLL |
| --- | ---: | ---: | ---: |
| 5 | 128 | 111.27 | 11.406 |
| 14 | 128 | 94.58 | — |
| 25 | 2,000 | 52.35 | 9.862 |
| 39 | 128 | 109.10 | 9.946 |

The 128-image curve does not demonstrate that full-count FID stopped improving
at epoch 5. Only one 2,000-image score survived in the source history. Held-out
NLL does stop improving around epoch 25, despite declining training NLL; this
is evidence of overfitting, separate from sample-count bias. A lower future FID
is not guaranteed by these repairs.

The completed sample-count audit compares disjoint **real training faces**
against the same 2,000 real validation images. FID is **66.42 with 128 images,
31.77 with 512, and 15.55 with 2,000**. Thus a large apparent score difference
can come entirely from sample count, even without a generative model. These
are one fixed subset per count, not a universal FID floor or a correction to
subtract from model scores.

The local input audit decoded all 30,000 source JPEGs successfully: all are
1024×1024 RGB. It found 26 duplicate pixel pairs after the deterministic 256px
resize: 24 within training and two across train/validation. The repaired
manifest excludes 26 training copies, leaving 27,974 training images and the
original 2,000 validation images. The source files remain intact. The old
remote run's exact data files are unavailable, so this finding applies to the
Amarel dataset. Audits and exact excluded filenames are retained in the run
directory.

Both stages restart from random initialization under the new run ID
`celebahq256-var-laser-repair-20260917`. The recipe is
[`configs/experiments/celebahq-var-laser-repair.yaml`](../configs/experiments/celebahq-var-laser-repair.yaml).
It retains 50 tokenizer epochs and 50 prior epochs, the original learning rates
and schedules, 4096 atoms, sparsity 2, 257 coefficient levels, and the depth-16
VAR body. Four GPUs use tokenizer microbatch 8 × accumulation 4, and prior
microbatch 16 × accumulation 4, preserving effective batches 128 and 256.
The sampling recipe remains CFG 1.5, top-k 900, top-p 0.96.

The loader preserves the aligned full face frame at 256px, uses a single
unconditional face label, checks split identities, and raises on corrupt
images. Training uses reproducible horizontal flips as a conservative
regularization change for the observed overfitting; validation remains
unflipped. CPU augmentation and initialization no longer reset CUDA randomness.
Evaluation preserves training RNG across checkpoints and resumes. Training and
sampling share the same context/atom coefficient computation; tests compare
all per-scale, per-depth sampling logits with teacher forcing, with and
without guidance.

The main checkpoint-selection metric is always `generation/fid_2000`, computed
against the same 2,000 real validation images with matching RGB preprocessing
and Inception features. Epochs 25 and 50 also export 10,000 samples and log the
separate `generation/fid_10000` metric. These are finite-sample validation FID
measurements, not a published full-dataset benchmark. The best prior checkpoint
is retained independently of the last checkpoint. Checkpoint artifacts include
the tokenizer, applicable prior/best prior, configuration, source archive and
provenance every five epochs and at completion.

The launcher is [`scripts/submit_celebahq_var_repair.sh`](../scripts/submit_celebahq_var_repair.sh).
It verifies the allocation contains four A100 or L40S GPUs, stages data and
runtime on node-local storage, and holds an exclusive allocation lock. Before
production it runs three tokenizer updates with GAN active, three prior updates,
and a prior resume through update four. It checks parameter changes, finite
checkpoint tensors, four saved RNG states, labels, sparse encode/decode round
trips and generated images. A smoke failure prevents production from starting.
The tokenizer's original rFID quality threshold of 50 also remains enforced
before stage 2. Safe walltime checkpoints can requeue the allocation.

Run directory: `/scratch/xl598/runs/laser/celebahq256-var-laser-repair-20260917`.
The regression results, distributed CPU smoke receipt, GPU smoke receipt,
launch receipt, immutable source manifest, original W&B histories, input audit
and sample-count FID audit are stored there as they complete. The CPU smoke
uses a smaller model and FP32; it cannot replace the production BF16 GPU smoke.

Final launch: **SLURM job 61682639**, partition `gpu`, one node × four A100
GPUs (`ampere`), 24 CPUs, 128 GiB RAM, 72-hour limit. The earlier held
preparation jobs were cancelled before allocation. This job has no manual
hold or dependencies; the input audit and CPU verification have completed.
It is pending cluster scheduling. The `gpuk[001-018]` nodes are excluded for
this launch, using the established `gpu` node pool. An earlier Camden attempt,
job 61682500 on `gpuc004`, exited in two seconds before creating stdout/stderr
files and never reached training. Its startup cause is unconfirmed.

Validation completed: 26 regression tests; full input decode and split audit;
four-process training with gradient accumulation and active GAN updates;
tokenizer sparse-code/decode round trips; prior validation and generation;
prior checkpoint resume from update 3 to update 4; finite model/optimizer
tensors (with the intentional negative-infinity causal mask checked separately);
and an import/config check inside the packaged runtime. The frozen source
archive digest and exact launch settings are recorded in `launch.json`.
