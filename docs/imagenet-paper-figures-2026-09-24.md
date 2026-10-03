# ImageNet reconstruction figures for the paper

[W&B gallery](https://wandb.ai/helloimlixin-rutgers/laser/runs/imagenet-rfid421-paper40-20260924)

Expanded the qualitative set from eight to 40 examples and replaced the
annotated diagnostic layout with model headings and matched detail crops.
The outputs include a two-example main figure, ten four-example comparison
pages, 40 individual panels, ten LASER-only pages, and a separate FLIP
supplement. PDFs use vector headings and lossless raster image embedding.

The original eight samples are retained. Thirty-two new samples come from
a seeded permutation of ImageNet validation IDs, keeping one image per
previously unseen class. The selection is fixed before inference; no
reconstruction-quality ranking is applied. The official validation archive
was downloaded and checked against MD5 `29b22e2961454d5413ddabcf34fc5622`.

All four checkpoint hashes match the prior comparison: public VQGAN
16×16×1, RQ-VAE 8×8×4, and LASER 8×8×2 and 8×8×4. The latter is the
rFID 4.21 checkpoint used by the active Stage 2 run. All 160 new
reconstructions ran in CPU FP32. Reference preprocessing agrees exactly
with the eight archived inputs; CPU versus prior GPU reconstruction
differences are recorded because discrete code choices can vary with
numerical backend. The figures use fresh inference consistently.

Only column headings remain within the paper figures. Method provenance,
the VQGAN 4.90/4.98 reporting discrepancy, the five unavailable exact
baselines, selection details, hashes, and error-map settings live in W&B
metadata and the artifact README. No rFID is computed from this small set.

The renderer verifies exact pixel repetition for every 6× crop. Export
checks confirm expected PDF page counts, 6.5-inch width, vector headings,
and absence of JPEG image streams. All uploaded artifact checksums are
verified against the local exports.

Shared exports and the short LaTeX caption are in
[outputs/imagenet-stage1-paper40-20260924](../outputs/imagenet-stage1-paper40-20260924/README.md).
The W&B artifact includes native images, original JPEGs, FLIP arrays,
selection/crop manifests, checkpoint configs, source hashes, and scripts.
Stage 2 continued on seven H100 GPUs throughout this work.

The follow-up revision explicitly labels VQGAN as 16×16 in every comparison
heading. RQ-VAE D=2 and all three VQGAN 8×8 table variants still require
checkpoint/config locations: a search of 277 relevant W&B runs, local
assets, and official public releases found no matching weights. The
artifact README records the search and primary-source release discussions.

The single-region revision retains only the original first 64×64 crop for
each example. Every full image has one blue box, and one centered 6×
enlargement appears below. This applies to the main figure, all comparison
pages, individual panels, LASER-only pages, and the FLIP supplement. The
region is identical across models; checkpoint pixels and sample selection
are unchanged. W&B config, summaries, caption, and artifact metadata record
one crop per image.

The equal-size revision displays both the full image and its single detail
crop at 384×384 pixels. The 64×64 native crop uses exact 6× nearest-neighbor
pixel repetition. All 40 examples, page layouts, individual exports and the
FLIP supplement were regenerated. The online artifact is
`helloimlixin-rutgers/laser/imagenet-rfid421-paper-figures:v3`; all 727 file
digests match the local exports.

The Samoyed/cheeseburger supplement adds four official validation examples
per requested class (indices 41–48). A fixed seed selects the images before
inference; the main figure pairs the first example of each class. All four
verified checkpoints run on CPU in FP32. The dogs' faces are cropped using
the reference photos; cheeseburgers retain the reference-only central-detail
rule. Each single crop is shared across models and enlarged to 384×384,
equal to the full-image display size. All 40 zoom panels were compared
pixel-for-pixel against exact 6× repetitions of their native crops.

[Class gallery](https://wandb.ai/helloimlixin-rutgers/laser/runs/imagenet-rfid421-samoyed-cheeseburger-20260924)
and [PDF exports](../outputs/imagenet-stage1-samoyed-cheeseburger-20260924/README.md)
include a paired main figure, two full comparison pages, individual panels,
and separate FLIP maps. The artifact is
`helloimlixin-rutgers/laser/imagenet-rfid421-samoyed-cheeseburger-figures:v1`; all 168 uploaded file digests were
verified against the local files. The previous 40 examples are preserved.
Stage 2 remained running throughout.

The original-run revision reuses all eight examples from
`helloimlixin-rutgers/laser/imagenet-rfid421-stage1-zooms-20260924`. All 40
native reference/reconstruction PNGs were copied byte-for-byte from its
original `imagenet-rfid421-stage1-zoom-comparison:v0` artifact; 43 source
files (pixels and provenance) matched the server-side digests. No model
inference was rerun. Each example retains only original crop A, displayed
at 384×384, equal to the full image; exact crop pixels were verified in
all 40 rendered zoom panels.

The same W&B run now includes minimal-label PDFs and updated versions of
its eight comparison panels and two LASER zoom pages.
[Original-eight exports](../outputs/imagenet-stage1-original8-20260924/README.md)
are separate from the unchanged Samoyed/cheeseburger set. Artifact
`helloimlixin-rutgers/laser/imagenet-rfid421-original8-paper-figures:v0` has 172 files, all verified against
the local exports. Stage 2 remained running at publication.
