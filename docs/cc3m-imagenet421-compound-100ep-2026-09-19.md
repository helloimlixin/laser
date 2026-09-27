# CC3M ImageNet-4.21 compound reproduction

Submitted Slurm job **61708015** for four nodes × two L40S GPUs on `gpu`.
Run: https://wandb.ai/helloimlixin-rutgers/laser/runs/cc3m-imagenet421-compound-650m-100ep-20260919

Run directory:
`/scratch/xl598/runs/laser/cc3m-imagenet421-compound-650m-100ep-20260919`.

This is a new 100-epoch training run. The requested source,
`helloimlixin-rutgers/laser/cc3m-imagenet421-rq32k-650m-20260915`, used compact
32,769-word residual codes. The user explicitly requested replacing that method
with the successful FFHQ compound method from `ffhqcmp0804205803` (FID 8.1743927).
The new run does not need the missing source stage-two checkpoint or codebook.

## Model and data

- Exact frozen ImageNet rFID 4.210914 tokenizer, verified SHA-256
  `dd28db9306d526bdc9fbf8016403e106c4f317fd8f5c04792f0953e53310fdab`.
- Compound shape **8×8×4**, 16,384 OMP atoms, 2,048 coefficient bins, two-layer
  atom-conditioned micro-transformer, four coefficient classifiers, atom loss
  weight 1.5, distribution geometry weight 0.05, top four candidate atoms,
  geometry starting at epoch two and warming up over three epochs.
- The successful normalized FFHQ coefficient target policy: temperature 0.5,
  normalized range [-3,3], per-depth scales calibrated at absolute quantile
  0.995. Continuous physical FP32 coefficients remain in the cache.
- Official CC3M backbone: width 1,280, 26 spatial plus four depth layers, 20
  attention heads, BPE vocabulary 16,384 and context length 32. With the compound
  heads the actual model has **705,039,360** parameters.
- AdamW, LR 0.0005, betas (0.9,0.95), weight decay 0.0001, gradient norm limit
  one, no warmup, cosine decay to zero over 100 epochs, image/text weights
  0.9/0.1. BF16 training and FP32 targets, losses, tokenizer extraction and OMP.
- Global batch 2,048. Local batch 16, accumulation 16 on eight GPUs; the driver
  also supports four or sixteen GPUs without changing the global batch.
- All 2,905,954 local pixparse CC3M training pairs. Build two deterministic
  official DALL-E crops per image and 100 BPE dropout-0.1 text views. Alternate
  crops by image index/epoch; select one cached text view per epoch.
- Each epoch contains 1,418 complete global batches; shuffle then drop the
  1,890-image tail. This differs from the source's logged 1,419 updates/epoch.

The immutable model implementation is the August CC3M extension of the
successful FFHQ implementation recovered from:
`cc3m-official-rqt650m-compound-a16384-k2-20260808_124922/stage2/evaluations/fid_clip_current_20260815_024840/source_snapshot`.
The launcher uses its own frozen copy, not the changing repository checkout.
The upstream recipe was fetched from
https://github.com/kakaobrain/rq-vae-transformer/blob/main/configs/cc3m/cc3m-rqtransformer-8x8x4-650M.yaml
and included in the source archive.

## Evaluation and checkpoint contract

Every epoch generates one image for each of all **13,443 official validation
captions** and computes FID against those paired validation images, together
with mean CLIP ViT-B/32 image/text cosine similarity (unscaled). Sampling uses
temperature one, full atom/coefficient vocabularies, and top-p 0.7. A fixed
eight-prompt × eight-image preview is generated in batches of eight.

`RUN_ID-checkpoints` artifacts contain `last.pt`, `best-fid.pt`, and
`best-clip.pt` once metrics exist. **All three include complete resumable model,
optimizer, scheduler, data cursor and per-rank RNG state.** Lower FID and higher
CLIP select their respective best snapshots independently. Best snapshots are
retained in every subsequent artifact. Alias `latest`/`last` tracks the latest
state; `best-fid`/`best-clip` move only on improvement.

The last checkpoint is saved/uploaded every 500 updates, before and after epoch
evaluation, at orderly time-limit handoff, and at completion. Uploads use
immutable files, skip the redundant W&B cache copy, and wait for W&B's commit
acknowledgment before updating `last-upload.json`. All training checkpoint and
W&B staging directories are on node-local `/mnt/scratch`.

At each allocation start the launcher resolves a single exact latest artifact
version/digest and stages that same version on every node. It resumes at the
same global-batch cursor. An evaluation interrupted at an epoch boundary is
repeated before advancing training. At the time limit it saves/uploads and
requeues the same Slurm job; completed cache shards are retained across requeues.

## Verification and scheduling

CPU checks passed for the actual tokenizer hash and loading, architecture,
normalized targets, text-conditioned compound gradients, causal prefixes,
cached/full logits, deterministic crops and BPE dropout, cache shard indexing,
batch/cursor consistency on 4/8/16 GPUs, best-checkpoint retention, and refusing
to advance the durable receipt after a failed upload. Evidence: `cpu-preflight.json`.

The job first runs three real GPU optimizer updates plus a small FID/CLIP
evaluation. Production caching/training is gated on that check. GPU checks and
full-cache validation have **not** passed merely because the job is queued.

Scheduler probes compared A100 and L40S shapes for 4, 8 and 16 GPUs, including
shorter 12/24-hour requests and a flexible GPU-family constraint. Four L40S GPUs
saved only about ten minutes in the probe, so eight were retained. Immediately
after submission Slurm estimated September 20 at 12:13:41 Eastern, earlier than
the pre-submit probes. By the final check the estimate had moved to September
22 around 00:31 Eastern; the latest four-GPU probes still saved only minutes
relative to eight. Estimates can change. No other running jobs were changed.

Entry points:
`scripts/submit_cc3m_imagenet421_compound.sh`,
`scripts/train_cc3m_compound.py`, and
`scripts/tools/verify_cc3m_compound_reproduction.py`.

Inspect `submission.json`, `gpu-preflight.json`, `cache/ready.json`,
`status.json`, `last-upload.json`, `evaluation-epoch-*.json`, and `complete.json`
in the run directory. `complete.json` is written only after epoch 100 and its
checkpoint artifact have completed.
