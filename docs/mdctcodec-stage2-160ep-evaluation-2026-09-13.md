# Stage-2 160-epoch evaluation — 2026-09-13

Stage 2 completed normally at epoch 160, step 114194. Validation selected epoch 140; the terminal epoch-160 checkpoint was evaluated as a separately prespecified endpoint.

Both checkpoints were evaluated on the same frozen 100-recording VCTK manifest as the existing released-model baselines. See [W&B report and listening samples](https://wandb.ai/helloimlixin-rutgers/laser/runs/syh3dvnb).

# Fixed VCTK TTS system comparison

100 fixed recordings across 100 known speakers. Long-run checkpoints use validation WER selection and the terminal endpoint.

| System | WER % ↓ | CER % ↓ | ECAPA similarity ↑ | UTMOS ↑ | RTF ↓ | Cap % ↓ |
|---|---:|---:|---:|---:|---:|---:|
| reference | 1.05 | 0.21 | 0.6911 | 4.071 | — | 0.0 |
| codec | 1.63 | 0.59 | 0.5641 | 3.705 | — | 0.0 |
| laser_extension_last | 20.49 | 13.74 | 0.5391 | 3.535 | 0.673 | 0.0 |
| f5 | 1.75 | 0.65 | 0.7411 | 4.115 | 0.244 | 0.0 |
| chatterbox | 1.40 | 0.54 | 0.6861 | 4.268 | 0.216 | 0.0 |
| laser_long_best_wer | 17.69 | 12.31 | 0.5421 | 3.575 | 0.657 | 0.0 |
| laser_long_last | 16.30 | 11.67 | 0.5408 | 3.639 | 0.660 | 0.0 |

System-level comparison, not matched training-data/compute or a zero-shot LASER evaluation.
Prior targets have transcript-disjoint train/validation/test splits. Frozen codec saw these speakers and may have seen target audio.
Released baseline pretraining overlap with VCTK is unknown. No claim of uncontaminated external training.
UTMOS is a learned proxy, not human MOS. ECAPA similarity is not the WavLM SIM used in published F5 tables.
These VCTK scores are not directly comparable to published Seed-TTS/LibriSpeech scores.
This fixed set has informed earlier experiments; the extension is a regression comparison, not a fresh confirmation test.

Full per-file results, paired speaker-bootstrap confidence intervals and model hashes are in the artifact.


## Additional training effect

Epoch 160 reduced the test WER point estimate from 20.49% to 16.30% (20.45% relative). The paired speaker-bootstrap difference is −4.19 percentage points, with 95% interval [−9.07, +0.82]; this sample does not establish a nonzero WER improvement. UTMOS increased by 0.1047, with paired 95% interval [0.0564, 0.1522].

Epoch 140 remains the validation-selected checkpoint despite epoch 160 having the lower test point estimate. No checkpoint was selected by these test results.

The validation-selected model makes 152 word errors over 859 reference words. Eight test utterances with at least 15 words account for 73 errors; these are repeated versions of a difficult long sentence, so this descriptive grouping does not isolate length as the cause. All failure cases remain in the raw results and W&B listening table.

## Execution and fairness

The RVQ trainer process group was paused for 673 seconds to give synthesis exclusive active computation on GPU 1. Its model stayed resident in memory; the timing environment is recorded in long_selection.json. RVQ resumed after successful report upload, but an official ViSQOL subprocess that was in flight during the pause exceeded its 120-second timeout. This stopped RVQ and triggered the paired supervisor to stop LASER. Recovery restores LASER from the committed epoch-425 artifact at step 361880 and RVQ from its last fully validated checkpoint at step 233228; weights and optimizer moments are preserved, and stale within-epoch counters are normalized. Both two-update GPU recovery preflights passed. At most one epoch of discarded updates is replayed per arm (LASER: 767 updates; RVQ: 852 updates). The recovered supervisor retains the original measured GPU-hour usage and the 24-hour ceiling. The scheduled evaluator now detects this completed report and skips duplicate execution.

These are complete-system comparisons with unequal training data and compute. The [F5-TTS paper](https://arxiv.org/html/2410.06885v3) describes 100k hours of training data; our prior uses the VCTK training split. F5-TTS v1 Base and Chatterbox Turbo are pinned released references, not an exhaustive survey of current systems.
