The current FID50k run uses the released RQ FFHQ training statistics and the
released RQ Inception preprocessing, but our training-data preparation is not
identical to RQ's real-image evaluation preparation.

The generated-image path decodes directly to 256-by-256 RGB in FP32, maps
[-1,1] to [0,1], and clamps. It applies no random crop, flip, or PNG/uint8
roundtrip before FID. The released Inception wrapper bilinearly resizes those
pixels to 299-by-299 with `align_corners=False`, then maps [0,1] to [-1,1].
Its pretrained FID weights and preprocessing code are shared with the released
RQ implementation. RQ's sampling script likewise saves continuous float
pixels after decoding and clamping. The saved PNG grids are previews only.

The real-side statistics are the publisher's `ffhq_256_train.npz`, SHA256
`7f8f54ad5eee50c5b1fc1583e9d65c5a46652c3d60ba031965c9f46c9f2fa12b`.
They are not recomputed from our training loader or its augmentations.

There is an upstream preparation difference. Our data builder converts the
original aligned 1024-by-1024 FFHQ images to RGB, downsamples to 256-by-256
using Pillow LANCZOS, and stores lossless PNGs. RQ's published instructions
start from 1024-by-1024 FFHQ images; its real-image evaluation transform uses
`Resize(256)` with the torchvision BILINEAR default, followed by a center crop,
ToTensor, and normalization. The center crop has no geometric effect for these
square resized images. RQ's training augmentation also includes random resized
crops; our current training views use horizontal flips of the pre-resized data.
Our training image IDs additionally differ from RQ's shuffled split.

Consequently, the current scores share RQ's published reference and feature
preprocessing, and our own checkpoint comparisons use a fixed evaluation
protocol. They do not establish identical training-data preprocessing. The
initial audit did not measure the impact of LANCZOS versus BILINEAR on model FID;
the follow-up below measures it directly. The NPZ statistics do not
encode the complete original software/environment provenance, so this audit
checks the released implementation rather than claiming bitwise reproduction
of the historical statistic-building job. The active training/evaluation
configuration was preserved during this audit.

The running snapshot was checked against the inspected evaluator, Inception
wrapper, and FID source files; their contents match.

The completed follow-up rebuilt statistics for both resize methods and both
training splits. RQ's split with bilinear resizing reproduces the published
reference to 1.74e-8 FID. Using our exact cached training images gives epoch-100
FID50k 23.59408144; the same generated moments give 23.33948111 against the
published reference. Epoch 60 gives 24.61224834 and 24.61270009 respectively.
The matched evaluation and immutable reference manifest are documented in
`ffhq-fid-consistency-2026-09-22.md`. The two reference protocols have separate
metric names, and incompatible references are rejected before generation.

Sources:

- [RQ FFHQ transforms](https://github.com/kakaobrain/rq-vae-transformer/blob/main/rqvae/img_datasets/transforms.py)
- [RQ data preparation](https://github.com/kakaobrain/rq-vae-transformer/blob/main/data/README.md)
- [RQ Inception preprocessing](https://github.com/kakaobrain/rq-vae-transformer/blob/main/rqvae/metrics/inception.py)
- [RQ generated-image decoding](https://github.com/kakaobrain/rq-vae-transformer/blob/main/main_sampling_fid.py)
- [torchvision 0.10 Resize default](https://github.com/pytorch/vision/blob/v0.10.0/torchvision/transforms/transforms.py)
- Local data builder: `scripts/tools/prepare_ffhq_var_data.py:56`.
- Active evaluator: `src/training/rq_reference_evaluation.py:36`.
