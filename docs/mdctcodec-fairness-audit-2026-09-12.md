# Stage 1 benchmark fairness audit — 2026-09-12

The corrective evaluation is now complete:
[W&B report r2zvuoh9](https://wandb.ai/helloimlixin-rutgers/laser/runs/r2zvuoh9).
Its eight systems, 432 items each, include corrected DAC/ECDC inference,
FlowDec at 6 kbps, a new 200-recording confirmation set, common 16 kHz metrics,
32 nine-second segments, speaker-bootstrap intervals and a blinded listening
worksheet. Local report: `outputs/mdctcodec_fair_comparison/comparison.md`.
This audit remains a record of the problems in the earlier table.

The existing results support a comparison of specific codec checkpoints under
the recorded VCTK protocol. They do not establish current state of the art or a
controlled causal advantage of LASER over RVQ.

## What was verified

All six source evaluations underlying the final table have identical ordered
200-recording test manifests. The test has 25 recordings from each of eight
speakers excluded from codec optimization. No file appears in both the current
32-file checkpoint-selection validation set and the current test set.
All systems use original-level mono references, full utterances, reconstructed
signals trimmed to original length, and the same Google ViSQOL v3.3.3 audio-mode
implementation at 48 kHz. STOI uses the same reference/reconstruction pairs.

LASER's 6 kbps path really packs and parses two 13-bit atom IDs and two 7-bit
coefficients into five bytes per 150 Hz frame. LASER and both MDCTCodec rows have
the same measured 6.007956 kbps payload. The final DAC row is 6.040587 kbps and
EnCodec rows are 6.012709/6.057610 kbps. These small rate differences do not explain
the large quality gaps, but the table reports token payload, not complete files.
Headers and side information, including EnCodec scales, are excluded.

## Limitations that affect interpretation

1. **DAC inference differs from the official compressor.** The existing benchmark
   invokes raw `encode`/`decode` at original loudness. The official compressor
   defaults to -16 LUFS normalization, peak limiting and one-second chunks;
   decompression restores recorded input loudness and resamples to original rate.
   See the [official implementation](https://github.com/descriptinc/descript-audio-codec/blob/main/dac/model/base.py).
   The existing result is a raw inference result; it has not demonstrated the
   official compressor's performance. A separate 16-file, eight-speaker validation
   audit compares raw inference, official full-utterance compression, and official
   default chunked compression, always using seven quantizers from the same
   standard 44.1 kHz 8 kbps checkpoint. Its output is
   `outputs/mdctcodec_fairness_audit/dac_inference_paths.json`.

   The completed validation diagnostic found:

   | DAC inference path | Payload kbps | ViSQOL | STOI |
   | --- | ---: | ---: | ---: |
   | Existing raw encode/decode | 6.038306 | 3.573555 | 0.941708 |
   | Official full-utterance compressor | 6.038306 | 3.686552 | 0.947551 |
   | Official default one-second chunks | 8.522944 | 3.670687 | 0.949078 |

   Full-utterance official inference improves ViSQOL by 0.112996 on these
   validation clips, with gains on 15 of 16 clips. It changes loudness handling
   and the resampling path together, so the difference is not a pure estimate
   of normalization alone. This validates the need to rerun the full test; it
   is not a replacement 200-file test score. Default chunked compression has
   substantial padding overhead on these short clips and is not a 6 kbps row.

2. **Sample rate and channels differ.** DAC is a 44.1 kHz model used at seven
   quantizers near 6 kbps; this differs from its advertised full 8 kbps mode.
   EnCodec 24 kHz cannot reconstruct the original 48 kHz signal's full bandwidth.
   EnCodec 48 kHz is an official stereo model, fed duplicated mono and averaged
   on output. These are useful practical configurations but do not isolate
   architecture quality for native 48 kHz mono codecs. See
   [EnCodec's model documentation](https://github.com/facebookresearch/encodec).
   Native-bandwidth and common-bandwidth evaluations answer different questions;
   neither should silently replace the other.

3. **Training is not controlled.** LASER, recovered RVQ and released MDCTCodec have
   different optimization histories and checkpoint selection procedures. The
   universal DAC/EnCodec checkpoints also have different training corpora.
   Equal bitrate does not make an RVQ-versus-LASER architectural ablation.
   Such an ablation needs matched data, backbone, loss/discriminator, examples
   seen, tuning budget and repeated seeds. Equal training budgets are not required
   for a clearly labeled comparison of available pretrained systems.

4. **The test is small and has been inspected during development.** It contains
   eight speakers and has been evaluated at intermediate stages. Sixteen files
   also occur in the archived oyg7smih test. Checkpoint selection for the final
   runs used validation, but this is not a newly untouched confirmation set.
   Validation and test speakers are the same eight people, with disjoint files.
   Speaker-bootstrap intervals account for within-speaker dependence, but not
   training-seed variance, metric bias or adaptive experimental decisions.

5. **Metrics and duration need broader checks.** All 200 test files are shorter
   than eight seconds (minimum 2.006, median 2.906, maximum 7.075 seconds).
   [ViSQOL's input guidance](https://github.com/google/visqol#guidelines) recommends
   approximately 8–10 seconds and warns that scoring mode/sample rate matter.
   This does not automatically invalidate paired short-utterance scores, but
   warrants a fixed, prespecified longer-segment protocol as a sensitivity check.
   LASER lower LR scores 4.219922 ViSQOL versus raw DAC's 3.595808, while DAC has
   higher STOI (0.946379 versus 0.904440). Objective ranking is metric-dependent.
   No blinded listening evaluation has established an overall preference.

6. **Coverage is insufficient for a current SOTA claim.** The table includes a
   small set of established baselines. It is not a comprehensive contemporary
   rate-distortion comparison, and the experiments do not reproduce the MDCTCodec
   paper's full protocol. Its published 4.18 cannot be directly compared to our
   4.219922: released MDCTCodec scores 4.282415 when rerun on our own manifest.
   See the [MDCTCodec paper](https://arxiv.org/html/2411.00464v1).

## Defensible result and next evaluation

At roughly 6 kbps on this locked 200-file test, the lower-LR LASER checkpoint has
higher ViSQOL than the recovered RVQ checkpoint (4.219922 versus 4.109210), and
lower ViSQOL than released MDCTCodec (4.282415). The paired speaker intervals
support those specific comparisons. They do not establish that swapping RVQ for
LASER alone causes the improvement or that LASER beats SOTA.

Before stronger claims: rerun the official DAC path and verify EnCodec against
its reference compression path; report payload and side information separately;
add common-bandwidth speech metrics and a blinded listening test; evaluate a
fresh confirmation manifest and an additional corpus. For the architectural
claim, train a matched RVQ control with multiple seeds. Keep old measurements
as historical results with their exact inference labels.
