# Enforced 6 kbps LASER/RVQ comparison

The user requires a hard limit, K4, and 4096 total learned dictionary vectors.
Both new arms satisfy an exact packet inequality for every supported clip:

`8 * packet_bytes * 48000 <= 6000 * original_samples`.

The limit includes all coefficients, the 16-byte format/length/frame-count/CRC
header, and final byte padding. Codec weights and learned coefficient centers
are shared model metadata, as codebooks are for RVQ. This format supports mono
48kHz clips of at least 1536 samples (32ms); shorter standalone clips are rejected
explicitly. Validation checks every packet, rather than only the dataset mean.

## Representation and rate control

LASER has one shared 4096-by-32 dictionary and four distinct OMP selections per
coded frame. Sorting atom/coefficient pairs leaves the reconstructed sparse sum
unchanged. A combinatorial rank represents the four-atom support set; a base-nine
integer represents eight signed nonzero coefficient levels plus zero at each
position. Their combined alphabet fits in **57 bits per coded frame**. This is
lossless packing of sparse codes, not coefficient omission or vocabulary pruning.
RVQ has four independent 1024-by-32 dictionaries and four 10-bit codes, **40 bits
per coded frame**. Each arm has 4096 learned dictionary vectors, 131072 scalars.

For `N` original samples, the packet receives `floor(N/64)` bytes. After reserving
16 bytes for the header, the frame budget is
`floor((floor(N/64)-16)*8/bits_per_frame)`. Frames are packed continuously, so only
the last byte can have padding. Neither codec relies on entropy statistics to
meet the bound; arbitrary valid code IDs satisfy it.

Both codecs apply adaptive average pooling to the encoder's 150Hz latent grid
before quantization, using that frame budget. Quantized latents are linearly
interpolated back to the native decoder grid. These parameter-free operations
are active throughout training and inference. LASER approaches 105.26 coded
frames/s and RVQ 150 before header overhead; actual rates are lower. Thus the
backbone topology and initial weights match, while coded temporal resolution
differs to accommodate the transmitted coefficient information.

## Fresh paired training

Both arms restart from the exact common encoder/decoder/discriminator
initializations. Neither reuses the unconstrained model's learned backbone.
Both use the original 40936 VCTK training utterances, 48-example batches,
7960-sample deterministic crops, seed, losses, optimizers, and 200000 generator
and discriminator updates. Each complete epoch verifies the original data/crop
audit, and both learn through the hard rate controller. Dictionary learning,
RVQ projection modules, and coded frame rates remain method differences. This
is a comparison of rate-constrained codecs, not a pure quantizer-only ablation
or a comprehensive SOTA claim.

LASER's initial scalar centers are fitted on 256 training crops after the rate
controller. Only training batches update the centers. Validation never fits
scales, entropy tables, dictionaries or packet settings.

Every validation waveform is decoded from its actual packet. The fixed 128-file
validation set logs mean and maximum packet bitrate, violation count, coded
frame rate, ViSQOL and STOI. Any over-budget packet fails immediately. Latest
and the top three validation-ViSQOL checkpoints upload every five completed
epochs; the same cadence includes eight audio previews, waveform/MDCT/log-mel
and spectral-error figures. Both runs have 12 assigned-GPU-hour ceilings.

The preceding entropy-only LASER run and older RVQ continuation are stopped and
preserved. Their results are not relabeled as hard-6kbps results. The old K2 TTS
queue stays held until compatible K4 codec checkpoints and caches are available.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/tools/train_mdctcodec_hard6k.py --prepare
CUDA_VISIBLE_DEVICES=0 python scripts/tools/train_mdctcodec_hard6k.py --arm laser --smoke
CUDA_VISIBLE_DEVICES=1 python scripts/tools/train_mdctcodec_hard6k.py --arm rvq --smoke
CUDA_VISIBLE_DEVICES=0 python scripts/tools/train_mdctcodec_hard6k.py --arm laser
CUDA_VISIBLE_DEVICES=1 python scripts/tools/train_mdctcodec_hard6k.py --arm rvq
```

The production launcher is `python scripts/tools/run_mdctcodec_hard6k_pair.py`.
It requires recorded GPU-preflight and online-checkpoint-restore checks, runs one
arm on each GPU, enforces the two 12-hour ceilings, and gracefully stops the peer
if either arm fails or stops before its update budget. `status.json` reports both
processes. A root `STOP` file or SIGTERM requests checkpointed shutdown.

Output: `outputs/mdctcodec_k4_a4096_hard6k_20260913`.
