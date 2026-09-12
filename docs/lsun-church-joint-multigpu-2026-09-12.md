# Continue the corrected Church run on both H200 GPUs

The user requested both GPUs for the corrected geometry and early-decay run. This is a continuation of `church-joint-geometry-20260912`, from its own saved epoch-10 checkpoint (step 4,930). Its random initialization, optimizer moments, data permutation, learning-rate progress, tokenizer, relative coefficient targets, sampling thresholds, and geometry objective are preserved. No stage-2 initialization or tokenizer training is performed.

The relative-noise comparison run is saved and paused at step 16,672, epoch 33.8173. Its weights, optimizer and stream remain in `outputs/church-relative-noise-20260911/relative/last.pt`. Immutable hard links preserve the corrected run's pre-migration last and best-screen checkpoints under `outputs/church-joint-geometry-20260912/multi-gpu/`.

## Execution changes

`scripts/tools/train_church_joint_distributed.py` uses two-process PyTorch DDP. The entire corrected objective, including alternative-atom coefficient branches, is inside the DDP forward graph. Each rank handles four of the original eight microbatches of 32, retaining their original global microbatch indices and random seeds. DDP averages the two local gradients, producing the same effective batch of 256. The original per-microbatch geometry normalization is preserved. Numerical reduction order differs between one and two GPUs.

The frozen tokenizer's parameters are synchronized once by DDP at initialization. Its parameters and buffers are checked for mutation after initialization. Only the prior's parameters belong to AdamW. Checkpoint saves retain the original recipe config; `execution.json` and checkpoint `execution` fields separately record the runtime implementation and GPU counts. Existing completed evaluations are reused on resume.

Generation uses batch 512 per GPU, with decoding and original RQ-VAE Inception in batches of 32. Each GPU generates a disjoint half of the requested population; FID combines sufficient statistics across both ranks, rather than averaging two FID scores. Exactly 50,000 images are used for the full evaluation. Sample counts, code shapes/ranges, distinct support, and cached evaluation resume are verified.

Generation RNG streams are now `base_seed + rank*1000003`. The base seeds and sampling distribution are preserved, but changing batches and rank streams changes the actual sampled population relative to the previous single-GPU protocol. New 50k results are comparable estimators, not identical random draws. Each shard saves its codes, local feature statistics, and CPU/CUDA RNG state every four generation batches and at completion. The old evaluator had no partial-generation checkpoint, so its unfinished image set is discarded; the epoch-10 training state is preserved exactly.

The workspace was reorganized concurrently: historical scripts/configs moved into `archive/`, and the tokenizer implementation moved to `src/training/rqtransformer.py`. The relevant implementation AST matches the original production snapshot after accounting for ROOT relocation and excluding CLI-only entrypoints. The Church config path now points to its archived location. The archived FFHQ model source, corrected objective, and relative-noise target implementation remain byte-identical.

## Verification and measured throughput

- 23 existing model, token, noise, and geometry tests pass. Three additional tests exercise real DDP gradient/update equivalence with the full joint objective, exact global training-index/seed partitioning, and evaluation population partitioning.
- The real 404,738,048-parameter checkpoint was continued for three updates in both serial and distributed execution. Maximum parameter difference is 5.84e-7; relative parameter L2 difference is 2.49e-7. Maximum optimizer-state difference is 3.31e-7. Data streams and optimizer settings match exactly. These are floating-point-equivalent updates, not bitwise-equivalent single-vs-dual-GPU training.
- The distributed three-update continuation is separately compared with a one-update save followed by a two-update resume, including model tensors, AdamW state, and data-stream state. The authoritative receipt is `multi-gpu/resume-verification.json`.
- Full-model warm training steps measure approximately 2.03 seconds on one GPU and 1.03 seconds on two GPUs. The effective batch and precision remain unchanged.
- Single-GPU exploratory generation probes include decoding and Inception. Batch 512 reached about 143 images/sec in a warm probe; larger batches gave inconsistent gains and were not selected. These short isolated probes are not a guarantee of production speed. Live distributed throughput is recorded separately.

The complete gate, sources, benchmark data and migration receipts are under `outputs/church-joint-geometry-20260912/multi-gpu/`. The launcher checks the verified sources, starts the replacement while retaining the suspended original evaluator as a fallback, and retires the original only after both new ranks generate images. The same W&B run ID is resumed.

Production checks do not establish improved image quality. This change allocates compute more efficiently to the existing corrected experiment.

## Production outcome

The handoff completed successfully. Both ranks generated 25,000 images, and the epoch-10 full FID is **22.13660**, compared with the paused control's epoch-10 **23.97294**. The completed distributed evaluation took **192.88 seconds**. A 30.5-second live window measured **285.5 images/sec** combined, with mean GPU utilization **90.2% / 92.5%**. Training then resumed from step 4,930 and runs at approximately **1.03 seconds/update** on both GPUs; a training snapshot showed **97% / 96%** utilization. `multi-gpu/completion.json` records the live training status and evaluation result. The sampling population differs from the old single-GPU seed stream as described above.
