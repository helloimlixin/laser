# VCTK TTS continuation and released-model benchmark

Stage 2 completed 40 epochs / 28,545 optimizer steps. Its lowest validation
token NLL was 3.105630 at epoch 29; epoch 40 scored 3.109035. A 20-epoch
continuation resumes the full optimizer state in the same
[W&B run](https://wandb.ai/helloimlixin-rutgers/laser/runs/13z2ta24).
The learning rate continues from 0.0000300002166 and follows a new cosine to
0.00001 at the 60-epoch horizon, approximately 42,840 total steps. The
continuation has a three-hour process cap. It retains the frozen 6 kbps codec,
training split, architecture and losses. This is a modest continuation of a
flattening validation curve, not evidence of improved quality.

Configuration is `configs/research/mdctcodec_tts_rangefix_continue60.yaml`.
The original epoch-40 optimizer, top-three models, completion record and resolved
configuration are preserved under `outputs/mdctcodec_tts_rangefix/epoch40_snapshot/`.
A two-update GPU preflight verified optimizer continuation, finite parameters
and continuity of the learning rate. Two schedule tests cover the continuation
and original warmup/cosine endpoints.

Preview count increases from four to sixteen fixed known-speaker validation
prompts. Each includes original-level generated/reference audio, waveform and
log-mel panels, blue text/audio cross-attention, ASR and EOS diagnostics.
Attention probabilities use the sequential `Blues` palette over a fixed 0–1
range with nearest-neighbor cells. `media.attention_cmap` is reread from the
training YAML for each preview, allowing later palette changes without a
training restart. The spectrogram colors are unchanged. A graceful
checkpoint/save at step 30,163 loaded this configuration and media update without
discarding optimizer progress. The 60-epoch target is unchanged. Latest plus
top-three validation-NLL checkpoints still upload every five complete epochs.
The initial grayscale style was subsequently replaced with blue at the user's
request. The sixteen epoch-50 alignments were recomputed from their exact
step-35,684 checkpoint and original generated token sequences, then published
with the new palette. Training resumes from the saved step-35,685 optimizer.

## Fixed evaluation protocol

`outputs/mdctcodec_tts_benchmark/manifest.json` is frozen before production
synthesis. SHA-256:
`ed40073115737eb6aaa8c02d3af6d80703216abdab7ad7fe412254f453cf715e`.
There is one deterministically hash-selected utterance for each of 100 known
speakers, from the transcript-disjoint TTS test split. Eligibility is 2–8 seconds
and 5–35 normalized words. Every source waveform hash and sample count is
verified. Eight codec-held-out speakers are excluded because this prior uses
learned speaker IDs and does not support unseen-speaker voice cloning.
The 100 recordings contain 70 distinct normalized texts; some common VCTK
sentences recur across speakers. The primary intervals resample speakers, not
independent text clusters.

Each released baseline receives the same *different* training utterance from the
target speaker, chosen by duration nearest eight seconds within 3–10 seconds.
It never receives the target waveform. LASER receives its learned speaker ID.
This compares complete systems for the same desired text and voice. It does not
match training data, compute, model size or conditioning architecture. The frozen
codec saw these speakers before the prior split and may have seen test audio.
Released-model training overlap with VCTK cannot be excluded.

Systems:

- LASER checkpoint selected by lowest validation NLL available after 40 epochs:
  epoch 29 / step 20,697. This is selected before examining test scores. The
  immutable copy is used while continuation may replace live top-three files.
- [F5-TTS v1 Base](https://huggingface.co/SWivid/F5-TTS), released checkpoint,
  32 flow steps, CFG 2, sway −1 and speed 1; released Vocos 24 kHz decoder.
- [Chatterbox Turbo](https://huggingface.co/ResembleAI/chatterbox-turbo), released
  checkpoint and package 0.1.6 defaults, including its watermark.
- Ground-truth target audio and frozen-codec reconstructions as diagnostic
  controls, not competing TTS systems.

One fixed seed is used per prompt, with no rerolling or output selection. LASER
retains temperature 0.8 and top-k 50. Generation has a 15-second safety cap with
cap/EOS failures retained and reported. No system is forced to match the target
duration. Raw audio is preserved at native rates and original output levels;
metrics resample consistently to 16 kHz. Model repository revisions and weight
SHA-256 hashes are recorded under `models/`.

## Measurements

- Corpus word and character error use
  [Whisper-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3),
  float16, beam 5, English, temperature zero, VAD disabled and previous-text
  conditioning disabled. All arms use Whisper's English text normalizer.
  Ground-truth ASR provides the recognizer's reference error floor.
- Speaker similarity is cosine similarity from
  [SpeechBrain ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
  against the separate enrollment utterance. This is not WavLM SIM, so values
  must not be compared directly with published F5-TTS SIM tables.
- [UTMOS22 strong](https://github.com/tarepan/SpeechMOS) estimates naturalness.
  It is a learned proxy, not a human listening MOS experiment.
- Batch-one synthesis wall time includes raw-text processing, reference
  preparation where applicable, sampling and waveform decoding. CUDA is
  synchronized before and after. Model loading, one enrollment-only warmup and
  file writing are excluded. RTF divides total synthesis time by total generated
  duration. Median/p95 latency and peak PyTorch-allocated GPU memory are retained.
  Systems run sequentially on GPU 1 (H200); training and previews use GPU 0.
  Backend precision follows the respective inference paths and is recorded.
- Paired speaker-bootstrap intervals use 5,000 resamples, one row per speaker.
  These quantify uncertainty on this fixed VCTK subset, not universal model rank.

The metric families follow common TTS evaluation, including the
[official F5 evaluation guide](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/eval/README.md).
This is a local system comparison against competitive released open models, not
a claim to reproduce published Seed-TTS or LibriSpeech leaderboards.

## Execution and outputs

`scripts/tools/benchmark_mdctcodec_tts.py` provides prepare, generate, score and
report phases. `scripts/tools/run_mdctcodec_tts_benchmark.py` runs the remaining
phases sequentially after LASER synthesis completes. Generation and scoring
write per-utterance JSON records and verify hashes on resume; a missing or failed
item prevents a complete report instead of silently shrinking the denominator.
Two metric tests check edit counts and corpus/duration weighting.

Baseline dependencies are isolated in `/workspace/tts-benchmark-env`; downloaded
weights are in `/workspace/tts-benchmark-models`. `setup/runtime_freeze.txt`
records the environment. The baseline runtime shares the host's PyTorch and
torchaudio 2.4.1; this differs from Chatterbox's tested 2.6.0 dependency pin.
Only inference dependencies are installed, so unused UI/training dependency
warnings may appear. Actual synthesis and metric execution are verified.

W&B receives the complete per-file score table, playable samples for every TTS
system and target, waveform/log-mel plots for the first eight preselected speakers,
confidence intervals, model hashes, source and complete raw audio artifacts.
The fixed enrollment/reference clips accompany the listening table. No test
metric is used to alter sampling settings or select a checkpoint.

`scripts/tools/evaluate_tts_after_extension.py` waits for normal completion of
the continuation and the initial benchmark. It freezes the lowest-validation-NLL
checkpoint, evaluates it on this identical manifest and updates the same report.
If the selected checkpoint bytes are unchanged, it reuses the existing samples
and scores and explicitly records that the extension did not select a better
model. Initial results are preserved as `results_before_extension.json`.
Status is recorded in `after_extension_status.json`; a failure is explicit.
The follow-up also evaluates the immutable epoch-40 last checkpoint and the
terminal continuation checkpoint as fixed-budget endpoints. Both are reported,
without choosing between them by test performance. This checks the effect on
free-running speech even if teacher-forced validation keeps selecting epoch 29.

## Verified initial results

[Online benchmark](https://wandb.ai/helloimlixin-rutgers/laser/runs/ci7y4jty),
artifact `mdctcodec-tts-benchmark-ci7y4jty:v0`, is finished and committed. It
contains 500 WAVs across the five arms, 32 waveform/log-mel plots, all per-file
results, 300 TTS listening-table rows and reproducibility metadata. Every arm
has exactly 100 scored utterances. All three TTS systems terminated within the
15-second cap. LASER's actual serialized payload is 6.0013 kbps.

| System | WER % ↓ | ECAPA similarity ↑ | UTMOS ↑ | RTF ↓ |
|---|---:|---:|---:|---:|
| Ground truth | 1.05 | 0.6911 | 4.071 | — |
| Frozen codec reconstruction | 1.63 | 0.5641 | 3.705 | — |
| LASER, validation-selected epoch 29 | 22.35 | 0.5289 | 3.446 | 0.713 |
| F5-TTS v1 Base | 1.75 | 0.7411 | 4.115 | 0.244 |
| Chatterbox Turbo | 1.40 | 0.6861 | 4.268 | 0.216 |

The selected prior's WER has speaker-bootstrap 95% interval 16.50–28.14%.
The remaining intelligibility gap is substantial. The low reconstruction WER
shows the frozen codec can preserve words much more reliably than this prior
currently generates them. This does not isolate a particular training fix.
These results are from the model selected before the extension, not the future
60-epoch model. UTMOS exceeding the reference is not evidence of human preference.

At verification, training had completed 46 epochs and was continuing. The
epoch-45 latest-plus-best checkpoint artifact and sixteen-prompt preview artifact
are committed: 32 audio files, 32 figures, 16 serialized payloads, and EOS in all
16 previews. Records are `media16_verified.json` in the training root and
`online_verified.json` in the benchmark root. The follow-up watcher is running.
