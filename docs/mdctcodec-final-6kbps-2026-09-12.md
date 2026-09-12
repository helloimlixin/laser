# Completed MDCTCodec–LASER stage 1, VCTK, 6 kbps

**Superseded comparison:** the DAC row below uses raw encode/decode and omits the
official compressor's loudness normalization/restoration. A corrected eight-system
comparison is now complete, including a new 200-file confirmation set, common-bandwidth
speech metrics, FlowDec at 6 kbps, and longer-segment checks:
[online report](https://wandb.ai/helloimlixin-rutgers/laser/runs/r2zvuoh9).
The table below preserves the historical measurements. Neither table is a
comprehensive SOTA result. See the
[September 12 audit](mdctcodec-fairness-audit-2026-09-12.md) for the inference,
bandwidth, training-budget, metric and test-exposure limitations.

Both continuation runs completed 300 total epochs (last zero-based epoch 299).
The original schedule's final report finished September 11 at approximately
23:48 UTC; the lower-LR report finished September 12 at approximately 00:02 UTC.
Neither training process nor final evaluation watcher remains running.

Both selected checkpoints use two sparse atoms per 150 Hz latent frame, with
13-bit dictionary indices and 7-bit signed coefficients: 6,000 raw bits/second.
Actual packed payload rate on the 200 full test recordings is 6.007956 kbps due
to padding. Headers and the shared coefficient bound are excluded. All rows
below use the same locked VCTK test recordings and Google ViSQOL audio mode at
48 kHz. Baselines have different training histories; these are comparisons of
checkpoints rather than experiments with equal training budgets.

| System | Payload kbps | ViSQOL | STOI |
| --- | ---: | ---: | ---: |
| Released MDCTCodec | 6.007956 | 4.282415 | 0.921449 |
| LASER lower LR, selected epoch 267 | 6.007956 | 4.219922 | 0.904440 |
| LASER original schedule, selected epoch 287 | 6.007956 | 4.211150 | 0.902612 |
| Recovered trained RVQ control | 6.007956 | 4.109210 | 0.889778 |
| DAC 44.1 kHz, 8 kbps pretrained model, seven codebooks | 6.040587 | 3.595808 | 0.946379 |
| EnCodec 48 kHz, duplicated mono input | 6.057610 | 2.871605 | 0.864689 |
| EnCodec 24 kHz | 6.012709 | 2.819118 | 0.870417 |

Each run's checkpoint was selected by its original 32-recording validation
ViSQOL. The original schedule's selected checkpoint scored 4.150097 on that
validation set; the lower-LR selection scored 4.144818. Both test outcomes are
reported; their test ranking is not used to redefine checkpoint selection.

Paired 95% intervals use 5,000 speaker-bootstrap draws over the eight test
speakers. Lower LR minus original schedule is +0.008771, interval
[−0.003308, 0.021088], so this test does not establish a schedule improvement.
The lower-LR checkpoint exceeds the recovered trained RVQ control by 0.110712,
interval [0.081652, 0.147196], but trails released MDCTCodec by 0.062494,
interval [−0.093625, −0.034888]. The original schedule trails released MDCTCodec
by 0.071265, interval [−0.106896, −0.039254]. No overall state-of-the-art claim
is established.

## Reports and checkpoints

- [Original schedule final report](https://wandb.ai/helloimlixin-rutgers/laser/runs/eqhkq5i8)
- [Lower-LR final report](https://wandb.ai/helloimlixin-rutgers/laser/runs/qohfvaez)
- [Original training and checkpoint artifacts](https://wandb.ai/helloimlixin-rutgers/laser/runs/2ng5thjx)
- [Lower-LR training and checkpoint artifacts](https://wandb.ai/helloimlixin-rutgers/laser/runs/puutcpau)

Both final selected-checkpoint artifacts are verified COMMITTED online, with
aliases `latest`, `best-plus-last`, and `epoch-300`. Each contains `last.ckpt`
and all three checkpoints ranked by validation ViSQOL. Original retained epochs
are 254, 286 and 287; lower-LR retained epochs are 255, 267 and 281.

Original selected checkpoint:
`outputs/vctk_mdctcodec_stage1_6kbps/checkpoints/run_20260911_194041/laser/mdct-laser-6kbps-epoch=287-step=0490418.ckpt`

Lower-LR selected checkpoint:
`outputs/vctk_mdctcodec_stage1_6kbps_low_lr/checkpoints/run_20260911_210212/laser/mdct-laser-6kbps-low-lr-epoch=267-step=0456298.ckpt`

Per-utterance scores, manifests, calibration metadata and source snapshots are
in each run's `final_comparison` and `final_report` directories and their linked
online W&B benchmark/code artifacts. Earlier interim evaluations and recovered
historical benchmarks remain separately identified.
