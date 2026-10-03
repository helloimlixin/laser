This fresh Church Stage 2 experiment adapts the recovered code and recipe
from [FFHQ run ffhqcmp0804205803](https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhqcmp0804205803).
The new run is
[church-ffhq-recovered-b128-8h100-20260927](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-ffhq-recovered-b128-8h100-20260927).

At the user's request, the run was scaled after epoch 1 from global
batch 128 to **1,024 (128 per GPU)**. Its W&B ID and history remain the same;
the display name reflects batch 1,024. The first full FID50k was
**110.31462355** at step 986. The saved model, AdamW states, and all eight
RNG states are retained for the continuation.

The model originally started from random Stage 2 weights. The frozen Church tokenizer
is the selected three-epoch LASER fine-tune, previously measured at rFID
2.639338, with SHA256
`762c51a10267ed6fa55709ff0d6cf997940d21056b16c7a0a0399b3ebf93868d`.
Its 126,227-image deterministic OMP cache was verified against SHA256
`4c03555ebedca1bf7b74f1db30e91752586889337a58547e856416c3aeeae586`.
No trained Church or FFHQ Stage 2 weights are used for initialization.

The recovered trainer and July dependency bundle remain unchanged. A small
[adapter](../scripts/tools/church_recovered_ffhq_adapter.py) extends the
frozen decoder interface and model configuration from two to four sparse
depths and from 2,048 to 16,384 atoms. It reuses the original compound
transformer, two-layer coefficient micro-transformer, depth-specific heads,
autoregressive sampler, soft-target function, geometry surrogate, and loss.
Returning the adapter to K=2/A=2,048 reproduced the original model config
and all 509 initialized state tensors bit for bit under a shared seed.

| Setting | New Church run |
| --- | --- |
| Hardware / runtime | 8 H100s; Python 3.11.9, PyTorch 2.4.1+cu121 |
| Parameters | 404,738,048 |
| Transformer | Width 1,024, body 24 layers, depth head 4 layers, 16 heads |
| Sparse sequence | 8×8×4 atom/coefficient pairs |
| Global batch | 1,024: 128 per GPU, no accumulation; epoch 1 used 128 |
| Schedule | 200 epochs; epoch 1 had 986 updates, remaining epochs have 123 each; 25,463 total updates |
| Optimizer | Fused AdamW, LR 5e-4→0 cosine, no warmup, betas (0.9, 0.95), weight decay 1e-4 |
| Gradient clipping / residual dropout | 1.0 / 0.1 |
| Atom loss / coefficient targets | Atom weight 1.5; normalized soft targets at temperature 0.5 |
| Geometry | Original distribution surrogate, weight 0.05, delayed 2 epochs and ramped over 3 |
| Atom sampling | Temperature 1, top-k 250, top-p 1 |
| Coefficient sampling | Temperature 1, top-p 0.85 |
| FID | 50,000 samples after epoch 1, then every 5 epochs |
| Sample grids | 64 fixed-seed images every 200 optimizer steps, uploaded to W&B |

The source cache retains its validated FP32 Church encoder/OMP extraction.
Physical coefficients are reconstructed, rounded to FP16, calibrated at the
99.5th absolute percentile per depth, normalized, clipped to ±3, and stored
in FP16, following the recovered FFHQ cache recipe. New scales are
`[4.78385417, 2.35546875, 1.54296875, 0.98111979]`; approximately 0.5% of
coefficients per depth are clipped. Atom IDs and labels are unchanged.
The adapted cache SHA256 is
`995d22ed46957c0210b9f70c6c866f430684264f93b337c5dbece9fe3224147d`.

This intentionally follows the recovered FFHQ coefficient-noise and
geometry choices. Their success on FFHQ does not establish that they are
optimal for Church. Earlier Church experiments used several other target
spaces, scales, geometry formulations, initializations, and larger batches;
this run should be assessed as a fresh transfer experiment.

Training preserves the original outer BF16 context and `amp=False` forward
call, including the recovered backbone's nested autocast behavior. It does
not silently apply the later autocast fix or later attention optimizations.
The source initialization is preserved without an added global initializer.
The distributed sampler and dropped incomplete batches follow the original
training behavior, yielding 126,208 processed examples in epoch 1 and 125,952 per epoch at the larger batch.

FID uses the official Church reference SHA256
`809489d8316b9e6eb9dc3bc021b6d602f4b6d816cc80621c6b9c189a9253a7f6`.
The upstream Inception receives continuous FP32 pixels with TF32 disabled,
using feature batches of 500. Generation uses the FFHQ policy and batch 32
per rank, with a fixed evaluation seed base 20260927. The metric uses the
official Church reference rather than FFHQ statistics or cached reconstructions.

The eight-GPU preflight completed two actual training updates with geometry
enabled and a 256-image generation/FID smoke check. That smoke score is
solely a pipeline check, not a reported FID50k result. The 4.86 GB checkpoint
reloaded strictly, all model/optimizer tensors were finite, all 517 optimizer
states were at step 2, the scheduler horizon was 197,200, and all eight RNG
states were present. Maximum allocated GPU memory was 14.32 GB in this
preflight. Production consumes none of its weights or optimizer state.

Full checkpoints include model, optimizer, scheduler, epoch/batch cursor,
configuration, and all rank RNG states. `last.pt` was first saved at step 20. It is saved on the first update after
resume, every 500 updates, every epoch, and after FID. A new best
FID retains a separate full checkpoint. The bounded uploader creates
immutable local snapshots, uploads both `last.pt` and `best-fid-01.pt` when
a best exists, commits a W&B model artifact, and verifies remote sizes and
MD5 digests. Aliases are `latest`, `last`, and `best-fid`. Upload failures
propagate to training, and normal shutdown drains pending uploads.

Persistent assets, checkpoints, source manifests, launch configuration,
validation reports, and logs are under
`outputs/church-ffhq-recovered-b128-8h100-20260927/`. Execution uses a verified
local source copy to avoid concurrent import stalls on the shared mount.
W&B media/cache directories also use local disk. Credentials remain only in
process memory/environment and are excluded from the source artifact.

`launch.py --resume` starts detached training from this run's full latest
checkpoint. The current configuration requires resume and a successful
batch-scaling preflight. `process.json` records the launcher PID,
`train.log` records progress, and `train/checkpoint-upload.json` records the
last verified online checkpoint publication.

Startup verification confirmed fresh initialization on all eight ranks,
empty optimizer state, scheduler step zero, and exact agreement between
the production and preflight *initial* model hashes. Production passed
500 updates with finite loss and gradients. The first full checkpoint,
at step 20, was committed as
`church-ffhq-recovered-b128-8h100-20260927-selected-checkpoints:v0`; its
4,857,612,346-byte `last.pt` matched the remote size and MD5 digest. Further
latest uploads continue at the configured cadence. At that initial startup verification, no production FID50k had completed.
Epoch 1 subsequently produced FID50k 110.31462355 and a full best checkpoint.

The larger batch was selected with actual forward/backward/AdamW steps,
including the full geometry loss and original FP32 logit behavior:

| Images per GPU | Images/sec per GPU | Peak allocated / reserved GiB | Result |
| --- | ---: | ---: | --- |
| 64 | 296.03 | 27.67 / 33.42 | Passed |
| 128 | 310.74 | 50.35 / 56.84 | Selected |
| 160 | 310.57 | 61.69 / 70.56 | Passed, no throughput improvement |
| 192 | — | — | CUDA out of memory |
| 256 | — | — | Recovered attention kernel launch limit |

These are isolated single-GPU measurements; distributed throughput also
includes communication, sampling, and checkpoint I/O. The eight-GPU
preflight exercises the chosen batch under DDP.

The base LR stays 5e-4. At the epoch boundary, the scheduler changes from
step 986/197,200 to 123/24,600, preserving its phase and LR exactly
(0.0004999691581204155). AdamW and the global step counter remain at 986.
There are 199 × 123 remaining optimizer updates. CPU tests checked the
next three cosine updates against the analytic curve and rejected a
batch-size change at a partial epoch. The fixed-seed sample-grid path
checks that training CPU/CUDA RNG states are restored after sampling.

Batch benchmarks, schedule tests, the original launch files, the resume
checkpoint hash, and distributed preflight results live in the run's
`batch-scaling/` directory. The production checkpoint is the epoch-1
checkpoint, never the checkpoint updated by the preflight.

The scaling preflight passed on all eight GPUs: three resumed training
updates with geometry enabled, a 64-image sample grid, and a 256-image FID
pipeline check. Peak allocated memory was 51.86 GiB per GPU. Strict
checkpoint reload verified all model and optimizer tensors finite, all
517 AdamW states at step 989, eight saved RNG states, the preserved prior
best FID, and cosine step 126/24,600. Production resumed independently
from step 986 using the original epoch-1 checkpoint.

Production verification after scaling confirmed all eight ranks resumed
step 986 with 517 optimizer states, global batch 1,024, and cosine step 123.
Training passed step 1200 with finite loss and gradients. The
64-image grid at step 1,000 was verified on W&B by remote size and SHA256.
Artifact `helloimlixin-rutgers/laser/church-ffhq-recovered-b128-8h100-20260927-selected-checkpoints:v4` committed both the full latest checkpoint
(step 1000) and epoch-1 best-FID checkpoint; both remote
sizes and MD5 digests matched. Further latest/best uploads continue in the
background. Sustained training intervals reached about 2,495 images/sec
before geometry ramping, excluding sample generation and checkpoint I/O.
The run remains active; `batch-scaling/production-verification.json`
contains the verification snapshot.
