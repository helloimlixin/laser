# ImageNet Stage 1 reconstruction zooms

[W&B gallery](https://wandb.ai/helloimlixin-rutgers/laser/runs/imagenet-rfid421-stage1-zooms-20260924)
logs two LASER-only zoom pages, eight matched comparison pages, an overview,
the nine-row model coverage table, and per-example FLIP measurements.
Lossless PNG/PDF exports and all native crops are saved in the online
`imagenet-rfid421-stage1-zoom-comparison:v0` evaluation artifact.

The LASER 8×8×4 source checkpoint SHA-256 is
`dd28db9306d526bdc9fbf8016403e106c4f317fd8f5c04792f0953e53310fdab`,
matching the verified rFID 4.21 tokenizer used for Stage 2. Its archived
reconstructions were recovered from
`helloimlixin-rutgers/laser/imagenet-stage1-table-reconstruction-figures:v2`.
Rendering ran entirely on CPU while the seven-GPU training run continued.
The training run links this gallery in its notes and has the exports in
`stage1-zooms/` under Files.

The four rendered models are VQGAN 16×16×1 / 16K, RQ-VAE 8×8×4 / 16K,
LASER 8×8×2 / 16K, and LASER 8×8×4 / 16K. The released VQGAN is marked
with the supplied table's 4.90 and a visible discrepancy note: its official
release README reports 4.98. Exact checkpoints remain unavailable for
VQGAN† 4.32, VQGAN 8×8×1 at 16K/64K/128K, and RQ-VAE 8×8×2. These rows
have no substituted or fabricated images.

The eight inputs are the first lexicographic validation image from each
of the first eight synsets, matching the earlier comparison. Both 64×64
crops are selected using reference gradients only and held identical
across models. Zooms are exact 3× nearest-neighbor repeats. NVIDIA LDR-FLIP
is computed at native 256×256 resolution before cropping, at 67.020645
pixels/degree, with a fixed [0,1] magma scale. This qualitative set does
not measure rFID or establish representative dataset performance.

The renderer is [render_imagenet_stage1_zooms.py](../scripts/tools/render_imagenet_stage1_zooms.py).
It asserts source PNG hashes, checkpoint provenance, and direct pixel
repetition for every enlarged crop. All PDF exports have the expected
page counts and no JPEG streams. Full provenance, crop coordinates,
reproduction commands, publication receipt, and online verification are
in [the output directory](../outputs/imagenet-stage1-zoom-comparison-20260924/README.md).
