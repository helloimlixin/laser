# K4 continuation: epochs 101–150

Authorized continuation of `helloimlixin-rutgers/laser/ffhq-a2048-k4-stage1-100ep-20260910`, using the epoch-100 `v99/last_model.pt` checkpoint (SHA256 `6abdb445b573d013bc6047d809891942194f0d0b1a0ca239d13c2fb19cbe1c70`). Both model and discriminator weights, Adam moments/step counters (46,900), and scheduler states are restored. LR remains 4e-5. This is an explicitly recorded hardware/precision rebase, not a bitwise continuation.

- Hardware: four NVLink-connected H100 80GB GPUs.
- DDP: 32 images/GPU, one accumulation step, effective batch 128. Discriminator BatchNorm remains local with reference batch 32.
- BF16 training operations; FP32 model parameters, dictionary/OMP arithmetic, Adam states. No activation checkpointing. Autocast weight caching disabled because the legacy trainer is enclosed in an epoch-level context; cached casts must never span parameter updates.
- Original training split: all 60,000 official FFHQ IDs, original source PNG and decoded RGB MD5 verification, BICUBIC 1024→256 preprocessing matching the original K4 run.
- Validation: all 10,000 original FFHQ validation IDs, BILINEAR preprocessing, original RQ-VAE Inception/FID, FP32 single-image reconstruction. Epoch-100 matched baseline: 6.703591551335251. See the earlier matched evaluation report for protocol evidence.
- Checkpoints: recovery every 200 optimizer steps; epoch checkpoint and top-three checkpoints, uploaded as versioned W&B artifacts. Bounded automatic retry from the latest full checkpoint.
- Training workers restart each epoch so saved RNG state can reproduce worker seeding during recovery; validation workers persist. Pinned memory and local `/tmp` data avoid shared-filesystem bottlenecks.

Runtime: `/tmp/laser-ffhq-k4-continue150`. Logs: `training-attempt*.log`; current state: `training-status.json`, `supervisor-status.json`. Supervisor PID is recorded in the status file. W&B credentials are inherited through process environment, never stored in the launch scripts.

Run: https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhq-a2048-k4-stage1-continue150-20260923

Frozen source snapshot, scripts, configuration, benchmark/preflight results, data manifest, and mirrored status/evaluation JSONs are in `outputs/ffhq-k4-continue150-20260923/`. The archived original model/trainer sources match the checkpoint's recorded hashes, with explicit compatibility adaptations for Python 3.12 imports, local discriminator BatchNorm, and the native Inception call signature. The custom launch driver handles audited checkpoint restoration and distributed evaluation.

The initial disposable mixed-precision smoke test exposed stale autocast weight caching; those weights were discarded. Corrected smoke tests start afresh from the original checkpoint and check reconstruction quality, finite parameters, constant LR, optimizer step advancement, and exact checkpoint weight roundtrip. Initial corrected four-GPU throughput: approximately 168 images/s including startup, peak allocated memory approximately 54.3GB/GPU. Sustained run throughput is logged separately in W&B.

Production launch verified: supervisor PID 11452, torchrun PID 13055, all four GPUs active, W&B online. Before updating weights, the driver independently evaluates the complete epoch-100 validation set and requires rFID within 0.1 of the previously verified 6.70359 baseline.

Production baseline passed: rFID **6.7035750143** on all 10,000 images (73.0 seconds), matching the independently measured 6.7035915513. Epoch 101 is now training with finite losses on all four GPUs.

## Requested validation additions

The user requested a learning-rate review, sparse-coefficient heatmaps, reconstruction FLIP heatmaps, and NVIDIA FLIP. The source epoch-100 and continuation epoch-103 checkpoints both contain generator/dictionary and discriminator LR 4e-5. Their post-warmup cosine schedulers have base LR equal to minimum LR (4e-5), hence are constant. A simulated replay of both saved schedules through epoch 150 verifies that rate remains 4e-5 without a warmup restart. Effective batch remains 128. LR values for model, dictionary, and discriminator are explicitly logged each epoch.

Training is restarted from the full epoch-103 checkpoint using a narrowly validated signature migration: the prior checkpoint identity and source signature must match, all frozen training runtime files and all training configuration values must be unchanged. The changed driver adds validation and logging only; it preserves model/discriminator/optimizer/scheduler/RNG restoration.

Added `valid/nvidia_flip`: official NVIDIA `flip-evaluator==1.7`, LDR, sRGB inputs, default pixels per degree (~67.02), mean pixel error per image averaged over all 10,000 validation images and all four ranks. Identical-image and perturbed-image checks verify zero and positive scores respectively. The existing matched rFID calculation is retained.

Fixed first 64 validation images are logged alongside reconstructions, `valid/reconstruction_flip_heatmap_8x8` (magma, absolute [0,1] scale), `valid/sparse_coefficients_heatmap_8x8` (four signed OMP coefficient slots per image, coolwarm, shared symmetric 99th-percentile absolute scale), and `valid/sparse_coefficient_energy_heatmap_8x8` (coefficient L2 norm per latent spatial position, magma). Heatmap scales are saved and coefficient scales are also logged. Images are saved under the runtime training directory's `visuals/epochNNN/` and uploaded to W&B.

Validation additions verified on the full epoch-103 checkpoint: rFID 7.132202990149267 (exactly unchanged by the metric addition), NVIDIA FLIP 0.19967049323022365 over all 10,000 validation images. Evaluation took 224 seconds including FLIP. The fixed 64-image previews rendered successfully; training resumed at epoch 104 with LR 4e-5.

## Online dictionary update and larger-batch continuation (from epoch 105)

The user requested online dictionary learning matching ImageNet rFID 4.21 and larger batches for GPU throughput. The exact reference is `helloimlixin-rutgers/laser/imga16384k4altbn64-b128-b300-20260830000755`, rFID 4.210914134979248. Its source commit plus uploaded diff reconstruct an `alternating_dictionary_update_after_step_` method that is AST-identical to this run's frozen runtime method; see `online-update-code-comparison.json`.

The FFHQ epoch-105 checkpoint is pinned locally as `epoch105-before-online.pt`. Continue its weights and backbone/discriminator optimizer state to epoch 150. Enable `dictionary_update_mode=alternating_residual`, relaxation 0.25, minimum usage 2, maximum backtracks 6, and maximum updated atoms 2048 (all FFHQ atoms, corresponding to the reference's all-16384-atoms setting). The FP32 updater aggregates fixed-code residual sufficient statistics across all four GPUs. Dictionary parameters no longer receive Adam updates; their dormant Adam moments are removed. Backbone/discriminator LR remains the restored conservative 4e-5 without warmup restart. Other FFHQ architecture and loss settings are retained; this is not a wholesale copy of the ImageNet model or its ten-epoch LR schedule.

Full four-GPU tests with online updates, after warmup:

| Batch/GPU | Effective batch | Training images/s | Peak allocated GB/GPU |
|---|---:|---:|---:|
| 32 | 128 | 311.87 | 54.26 |
| 40 | 160 | 317.26 | 67.34 |
| **44 selected** | **176** | **318.94** | **73.89** |

The measured improvement over batch 32 is modest (~2.3%); filling memory is not itself a speed measure. A separate single-GPU batch-64 test with activation checkpointing was slower (66.59 images/s/GPU versus ~79–81 without checkpointing). Batch 44 leaves operating headroom on the 80GB H100s. Its local discriminator BatchNorm scope is now 44, explicitly matching the new local batch (previously 32). This batch/normalization change is recorded as a recipe rebase, not bitwise continuation.

Each distributed test ran 100 online updates and checked identical dictionaries across ranks, unit atom norms, actual dictionary changes, no dictionary Adam state/gradient, finite losses, unchanged LR, and bounded validation-quality change on 128 images. Batch 40/44 tests additionally saved and checked checkpoint roundtrips, update counters, rank RNGs, and signatures. Probe weights were discarded; production starts from the unmodified epoch-105 checkpoint. Actual optimizer steps now supply checkpoint clocks, and step logs compensate for the changed batches-per-epoch so their axes remain monotonic.

Online update activity is logged under `dictionary/update_step`, `dictionary/updated_atoms`, `dictionary/relaxation`, and `dictionary/fixed_code_relative_improvement`. Existing NVIDIA FLIP metrics, coefficient/FLIP heatmaps, and full matched rFID continue each epoch. The epoch-105 full baseline before this recipe change was rFID 7.5072865 and NVIDIA FLIP 0.2059674; the small benchmark validation scores are not substitutes for full-set rFID.

Production restart confirmed at epoch 106 from epoch-105 Adam step 49,245, batch 44/GPU (176 total). The first online update changed all 2,048 atoms with relaxation 0.25 and reduced fixed-code squared residual by 9.327%. A ten-sample production observation measured mean GPU utilizations 97.6%, 96.9%, 98.3%, 96.7%, with approximately 77,975 MiB device memory used per GPU. W&B config and online-update metrics were independently queried to verify the deployed recipe.

## Reconstruction artifact zoom gallery

A CPU-only watcher generates paper-style figures from the saved validation grids, without changing or restarting training. The linked W&B visualization run is `helloimlixin-rutgers/laser/ffhq-k4-continue150-zooms-20260923`; PNG/PDF exports and crop metadata are also uploaded under `zoom/epochNNN/` in the original training run's Files tab.

Each epoch has two pages covering the first eight official validation examples, unfiltered. Full reference/reconstruction pairs have matching colored boxes. Each example has an eye-region crop and a second high-detail region selected solely by gradient magnitude in the source image, then frozen across epochs. Each 64×64 crop is enlarged 3× using nearest-neighbor sampling, with matching reference, reconstruction, and NVIDIA FLIP panels. FLIP retains its fixed 0–1 color scale. No sharpening, smoothing, or generative enhancement is applied. PDF images use lossless embedding to avoid introducing JPEG artifacts. Selection details and exact coordinates are in `zoom/fixed-regions.json`.

The watcher backfills available epochs from 103 and adds subsequent epochs through 150. Runtime script/log/status: `/tmp/laser-ffhq-k4-continue150/zoom/{watch.py,watcher.log,watcher-status.json}`. Workspace exports: `outputs/ffhq-k4-continue150-20260923/zoom/epochNNN/`.
