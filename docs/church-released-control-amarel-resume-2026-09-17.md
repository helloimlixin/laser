# Church released-tokenizer control on Amarel

SLURM job `61681659` requests 8 L40S GPUs as 4 nodes × 2 GPUs on `gpu`,
with a 72-hour limit. It continues W&B run
`helloimlixin-rutgers/laser/church-original-rqvae-released-tokenizer-control-20260917`.
The initial submission is pending; GPU execution has not yet been verified.

The recovered `selected-checkpoints:v33` artifact contains epoch 32, optimizer
step 1,984, all 460 optimizer states, the cosine scheduler, AMP scaler, and two
original RNG states. The 370,087,936-parameter model loads strictly. CPU preflight
verified the saved LR and its next update, both supported GPU counts, and rejection
of a GPU-count change in the middle of an epoch.

The released tokenizer was recovered from KakaoBrain's Church archive. Its file,
config, and model-state SHA-256 hashes match the checkpoint. The original Church
LMDB files were recovered from Hugging Face dataset `RichardErkhov/LSUN`, revision
`1ee52c98c617f4222220fb124d761bef00ae5fbc`; the train/validation counts and ordered
key hashes match the checkpoint (126,227 / 300 images).

On allocation, every node stages the same immutable checkpoint version resolved
once by the batch process. Before training, the GPUs rebuild continuous FP32
latents and Inception statistics over all training images, using the released
Resize/CenterCrop transform. Original cache/reference bytes are unavailable; the
new hashes and original provenance are explicitly recorded. This is a stateful
continuation, not a bitwise replay across different hardware and GPU counts.

Training preserves global batch 2,048, 62 updates per epoch, the 300-epoch cosine
schedule, stochastic soft targets, and frozen tokenizer. Microbatch 32 uses
accumulation 8 on 8 GPUs or 4 on 16 GPUs. Validation and FID retain the original
two evaluation streams, including 50,000 generated samples, top-k 1,400, and
seed 71,000. Extra ranks contribute no evaluation examples. Newly added training
ranks receive distinct deterministic RNG states; the original two retain theirs.

Checkpoint files and W&B staging use node-local scratch. The latest full state
and best three FID checkpoints are uploaded each epoch. Shared scratch holds the
recovered assets, verified source snapshot, rebuilt cache, receipts, and logs.

Run directory:
`/scratch/xl598/runs/laser/church-original-rqvae-released-tokenizer-control-20260917`.
Use `scripts/submit_church_released_control_resume.sh` for subsequent submissions;
it rejects a second live allocation. Supporting scripts are under `scripts/tools/`.
`preflight.json`, `ready.json`, `launch.json`, and `runtime-manifest.json` record
verification and launch provenance. Repairs to the snapshot must also update its
manifest.

The requested detached Codex monitor has session
`01a0b0ed-6c70-70c3-a1df-0c3e7a64775e`, supervised by PID `1990769` on `amarel4`.
It monitors queue/startup, fixes launch failures, and relaunches this run as needed.
It stops after confirming the actual allocated GPU identities and increasing
finite-loss training steps across at least 60 seconds. Receipts, status, actions,
and transcripts are retained under the run directory's `monitor/` folder.
