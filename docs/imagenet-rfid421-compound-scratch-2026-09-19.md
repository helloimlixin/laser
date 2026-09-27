# ImageNet rFID 4.21 compound stage-two scratch launch

Submitted Slurm job **61707883** on September 19, 2026. It is pending, with
the last observed estimate **09:18 EDT September 19**. Scheduler estimates can
change. No training progress or GPU validation is claimed while pending.

Run directory:
`/scratch/xl598/runs/laser/imagenet-rfid421-compound-1400m-scratch-20260919`.
W&B online logging starts with training under
`helloimlixin-rutgers/laser/imagenet-rfid421-compound-1400m-scratch-20260919`.

The tokenizer is the unchanged epoch-10 ImageNet checkpoint with reported
reconstruction FID **4.210914134979248**, artifact
`imga16384k4altbn64-b128-b300-20260830000755-stage1-checkpoints:v9`, member
`best_rfid_slot1_model.pt`. Its SHA256 is
`dd28db9306d526bdc9fbf8016403e106c4f317fd8f5c04792f0953e53310fdab`.
The tokenizer remains frozen; the stage-two model has no initialization or
resume checkpoint, and starts with a fresh optimizer and schedule.

The compound implementation is copied into an isolated source directory from
the recovered source of
`church-laser-rfid421-ft3-compound-scratch90-20260918`. It includes the
`atom_conditional_v2` geometry correction, full pair autoregression, two
micro-transformer layers, and depth-specific coefficient heads. This correction
was absent from the working-tree trainer; existing training sources were not
overwritten. Original and launch-source hashes are retained in the run directory.

The [original ImageNet 1.4B recipe](https://github.com/kakaobrain/rq-vae-transformer/blob/main/configs/imagenet256/stage2/in256-rqtransformer-8x8x4-1400M.yaml)
sets the 8×8×4 shape, width 1536, 42 spatial/6 depth layers, 24 heads,
1,000 class conditions, local batch 8, global batch 2048, AdamW LR 0.0005,
weight decay 0.0001, betas (0.9, 0.95), gradient clipping 1, and 100 epochs.
The compound model has **1,457,980,928 parameters**. Training uses the existing
cosine implementation, no warmup, and minimum LR zero.

LASER-specific adaptations retain the Church objective: 16,384 atoms, 2,048
coefficient bins, coefficient range ±3, physical soft-target temperature 0.125,
atom-loss weight 1.5, and geometry weight 0.05 starting at epoch 2 and warming
over 3 epochs. ImageNet sampling uses temperature 1 and top-p 0.92 for both
atom and coefficient predictions, retaining all 16,384 atom candidates before
nucleus filtering. Unlike the original image augmentation recipe, this launch
uses a deterministic center-crop sparse cache with stochastic coefficient targets
regenerated on every visit, matching the reference compound workflow.

The pipeline builds the full 1,281,167-image compound cache first and calibrates
per-depth coefficient scales from ImageNet maxima, rather than reusing Church
normalization. It requires exact support agreement, finite coefficients,
shape/count/provenance checks, and a passed cache validation report. Next it
runs a separate eight-or-four-rank model/optimizer memory smoke with the geometry
objective active. Production training starts from a new random initialization
after that smoke completes.

The initial eight-L40S submission was adjusted in place after the user allowed
relaxed GPU constraints for an earlier start. It now requests **4 nodes × 1 GPU**,
accepting A100/L40S or a mix. This reduced the observed queue estimate from
13:51 EDT to 09:29 EDT. Local batch remains 8; **64 accumulation steps** preserve
global batch 2048. Each node requests 6 CPUs and 96 GiB host RAM. Allocation
duration is 72 hours, with a graceful save/upload 30 minutes before its end.
The 100-epoch target may require later continuation from the saved checkpoint.

FID-50k and Inception Score run every two epochs using upstream Inception and
published ImageNet training reference statistics (SHA256
`3f9c92d15755e76ec312964a819e3e19b9cf3aadc618a7ae99f5e8aa96501260`).
The launch uploads full last and best-FID checkpoints as versioned W&B artifacts,
including `latest`, `last`, `epoch-N`, and qualifying `best-fid` aliases. Recovery
checkpoints also upload every 500 optimizer steps and on a graceful stop.
Checkpoints, W&B staging/cache, and framework caches live in node-local scratch;
artifact uploads use immutable files and skip the extra W&B cache copy.

Validation completed before submission: tokenizer SHA256 and model construction
preflight, **24 focused tests passed**, one CUDA-only test skipped on CPU, and
shell/Python syntax checks. A real CPU load also verified all tokenizer weights
and its 256×16,384 dictionary, with every tokenizer parameter frozen. This caught
and fixed a NumPy-2 checkpoint metadata compatibility issue; the isolated loader
checks the pinned SHA256 before loading the trusted legacy metadata. GPU type,
numerical memory smoke, cache extraction,
and actual online training are checked by the queued pipeline.

Inspect `launch.json`, `recipe.json`, `preflight.json`, `scheduling.json`,
`cache/ready.json`, `smoke/rank-*.json`, `train/rank-*.json`, and `train/status.json`
under the run directory. Slurm logs are `slurm-61707883.out` and `.err`.
