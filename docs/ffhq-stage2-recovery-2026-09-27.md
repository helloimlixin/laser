Recovered and successfully sampled the Stage 2 model from
[ffhqcmp0804205803](https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhqcmp0804205803).
The recovered prior is epoch **200**, step **109,200**, with historical
FID50k **8.1743927** against 70,000 FFHQ training images. The initial recovery
sweep did not recompute FID. The subsequent
[two-seed FID reproduction](ffhq-stage2-fid-reproduction-2026-09-27.md)
scored **8.4384 and 8.5303** with the original sampling policy; the exact
historical score remains unconfirmed.

The completed sweep is published in the
[W&B sampling run](https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhqcmp0804205803-recovered-sweep-20260927).
Artifact `ffhqcmp0804205803-recovered-sweep-20260927-results:v0` is committed;
all 3,061 manifest entries were verified for size, digest, and reference
identity. The run includes both comparison sheets, the original-settings
grid, and a table containing all 44 settings. The artifact contains all
2,816 individual PNGs, sampled tokens, and the recovered runtime.

The complete recovery is in
[`outputs/ffhqcmp0804205803-recovery-20260927`](../outputs/ffhqcmp0804205803-recovery-20260927/).

- [Interactive comparison](../outputs/ffhqcmp0804205803-recovery-20260927/index.html)
- [Original-settings 64-image grid](../outputs/ffhqcmp0804205803-recovery-20260927/sweep/atom_t1.00_k0250/grid.png)
- [Atom temperature / top-k sweep](../outputs/ffhqcmp0804205803-recovery-20260927/sweep/atom_contact_sheet.jpg)
- [Coefficient temperature / top-p sweep](../outputs/ffhqcmp0804205803-recovery-20260927/sweep/coefficient_contact_sheet.jpg)
- [Sweep manifest](../outputs/ffhqcmp0804205803-recovery-20260927/sweep/manifest.json)
- [Verification results](../outputs/ffhqcmp0804205803-recovery-20260927/verification.json)

The Stage 2 checkpoint came from immutable W&B artifact
`helloimlixin-rutgers/laser/ffhqcmp0804205803-checkpoint:v39`, entry
`best_fid_8.1744_epoch_200.pt`. It retains the optimizer and scheduler and
contains 385,839,104 model parameters. Its 4,630,705,374 bytes match the
artifact's MD5; SHA256 is
`0e1542a0af199059b79eb23006d6a619c3e680b72bf5fe2e1eb9e99a3472a4ff`.

The Stage 1 dependency is the exact filename recorded in that checkpoint:
`best_rfid_slot3_model.pt`, recovered from
`ffhq-a2048-k2-rqvae-strict-20260720-145706`. Its checkpoint epoch is 146.
Its 1,211,989,078 bytes match the W&B file MD5; SHA256 is
`c5843b63fe1b420b491e45754738c2298d4c094750346501a9dac25d1f2b607d`.
The run metadata does not record a historical Stage 1 hash; provenance is
the recorded run/path/filename and the current W&B file's verified digest.

The trainer downloaded from W&B is byte-identical to
`src/ffhq_v4_archived.py`, SHA256
`9ba1b49b4e5e339f0076bebee6fbac5629f6c391601de467019a3723c9d3e33f`.
It is the compound-v4 model with two atom-conditioned micro-transformer
blocks, depth-specific coefficient heads, 2,048 atoms, two sparse depths,
and 2,048 coefficient bins. The physical coefficient scales are
`[36.208333333333336, 8.583333333333334]`; Stage 1 attention is at resolution 16.

The original run did not record the revisions of every imported module.
The isolated `runtime/` preserves the exact saved trainer with a hashed
snapshot of compatible repository backbone and decoder modules. This is
not a claim of bitwise recovery of the entire historical environment.
Current versions are recorded in `environment.json`; the run uses PyTorch
2.8.0/CUDA 12.8, FP16 sampler autocast, FP32 decoding, and TF32 enabled.
cuDNN benchmarking is disabled for repeatability.

All eight workers loaded Stage 2 with `strict=True`, with no missing or
unexpected keys. Every Stage 1 encoder, decoder, and projection key matched;
the archived auxiliary loader separately restores and normalizes the LASER
dictionary. All generated token IDs were in range, each two-atom support
contained distinct atoms, and all decoded pixels were finite. The original
policy reproduced identical tokens and byte-identical PNG pixels between
the GPU-0 preflight and GPU-4 sweep run.

The completed sweep has **44 unique settings × 64 images = 2,816 images**:

| Sweep | Temperature | Filtering | Other settings |
| --- | --- | --- | --- |
| Atom | 0.70, 0.85, 1.00, 1.15, 1.30 | top-k 64, 128, 250, 512, 2048 | atom top-p 1; coefficient T 1, top-p 0.85 |
| Coefficient | 0.70, 0.85, 1.00, 1.15, 1.30 | top-p 0.70, 0.85, 0.95, 1.00 | atom T 1, top-k 250, top-p 1 |

The shared baseline is generated once. Each setting resets seed 20260927,
samples a batch of 64, and decodes in batches of eight. Every setting retains
all 64 individual PNGs, an 8×8 grid, a 16-image preview, sampled atom and
coefficient IDs, and its policy/result JSON. Contact sheets show the first
16 samples per setting. These are qualitative comparisons; no setting is
promoted as a measured FID improvement.

Reproduce from the repository root, using the preserved sampler snapshot:

```bash
recovery=outputs/ffhqcmp0804205803-recovery-20260927
python "$recovery/sample_recovered_ffhq_compound.py" \
  --runtime "$recovery/runtime" \
  --checkpoint "$recovery/checkpoints/best_fid_8.1744_epoch_200.pt" \
  --stage1-checkpoint "$recovery/stage1/best_rfid_slot3_model.pt" \
  --output "$recovery/rerun" \
  --devices 0,1,2,3,4,5,6,7
```

While the verified local scratch copies remain available, add
`--weights-directory /tmp/laser-ffhqcmp0804205803-checkpoints` to avoid
concurrent checkpoint reads from the shared filesystem. The authoritative
recovered files remain under the persistent output directory. Authentication
was held in the recovery process and was not included in scripts or reports.
