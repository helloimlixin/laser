# Stage 2 TTS with the matched range-fix LASER codec

Update: the initial 40 epochs completed and the same run is continuing to 60
epochs. Previews now include 16 prompts and blue attention alignments.
The first 100-speaker comparison against F5-TTS and Chatterbox Turbo is complete;
see the [continuation and benchmark record](mdctcodec-tts-benchmark-2026-09-12.md)
for current configuration, measured results and the queued post-extension test.

Both matched stage-1 arms finished 200,000 generator updates. The final report
upload failed because the executing script had moved into `archive/`. The saved
200-file evaluation outputs, waveform hashes, payload sizes, manifest hashes and
identical 200,000-batch audit streams were verified and published by
`scripts/tools/recover_mdctcodec_report.py`. The recovered
[comparison run](https://wandb.ai/helloimlixin-rutgers/laser/runs/c5ymko2t) is complete.
Locked-test ViSQOL audio48k is 4.1022 for LASER, 4.1588 for matched RVQ and 4.2716
for the separately pretrained released MDCTCodec. The matched LASER–RVQ difference
is −0.0566, with speaker-bootstrap 95% CI [−0.0798, −0.0367].

## Frozen-checkpoint diagnostic

On the 128-file validation split, the selected LASER checkpoint scores 4.1189.
Removing coefficient quantization raises this only to 4.1200, a difference of
0.0011 with 95% CI [−0.0001, 0.0026]. Dense bottleneck bypass scores 4.1463,
an increase of 0.0274 with 95% CI [0.0094, 0.0429]. Matched RVQ's selected
validation score is 4.1677. These results suggest little remaining loss from
coefficient precision at inference, and motivate investigating sparse approximation
and joint dictionary/encoder/decoder training. Bypass changes the decoder input
distribution; it neither isolates a unique training cause nor establishes an
achievable bound. Float and bypass probes have no 6 kbps rate claim.
The [diagnostic run](https://wandb.ai/helloimlixin-rutgers/laser/runs/54j9qg1a)
contains per-file results and the paired intervals.

## New TTS run

Launched at 14:18 UTC on September 12:
[stage-2 run 13z2ta24](https://wandb.ai/helloimlixin-rutgers/laser/runs/13z2ta24).
Launch and verification records are in `outputs/mdctcodec_tts_rangefix/`.
The first production epoch completed at update 714 with validation token NLL
3.8396. Its four generated/reference audio pairs, four waveform/log-mel figures
and four text-attention figures are committed online. Initial preview mean ASR
WER is 1.0 and EOS was reached in two of four samples; this is early training,
and generated speech quality remains unestablished. Training continued into epoch 2
with finite losses. Source, fresh token cache and the frozen codec are committed
as W&B input artifacts.

Stage 2 freezes the highest-validation-ViSQOL checkpoint from the new LASER run:
`laser-epoch=215-step=0368064.ckpt`, after 216 completed epochs and 184,032
generator updates. Selection uses validation only, not the completed test result.
Its SHA-256 is `3f4d31216ff54c45e0294d17eb51fa83709561c9a3528551c5fb5fed2fbeaf29`.
Its global coefficient bound is 34.14048767089844; two 13-bit atom IDs plus two
7-bit signed127 coefficients occupy five bytes at 150 frames/second.

The earlier 20-epoch TTS pilot completed but used a different frozen codec. Its
learned audio embeddings cannot be resumed against this new atom dictionary.
This run starts a fresh 61.9M-parameter phoneme/speaker-conditioned prior, using
the tested 512-wide model with three text-encoder layers, eight causal audio
decoder layers and a recurrent four-field depth decoder. The old phoneme
inventory is reused after checking every transcript, text split, sample count,
sampling rate and speaker restriction. Every audio token is re-encoded with the
new frozen checkpoint in FP32. Serialized token round trips are verified.

The cache contains 40,892 full VCTK mic2 utterances, 38.2877 hours and 100 speakers.
All eight codec held-out speakers are excluded. Normalized transcript hashes keep
repeated texts in a single split: 38,288 training, 1,371 validation and 1,233 test
utterances. The prior uses a fixed speaker-balanced 256-recording validation subset.
This evaluates text generalization for known speakers; it is not zero-shot voice
cloning. The codec itself was trained on these speakers before the prior split.

Configuration: `configs/research/mdctcodec_tts_rangefix.yaml`, launched through
`scripts/tools/train_mdctcodec_tts.py`. AdamW LR 0.0003, 500-step warmup and cosine
decay, bf16, frame budget 8,192, at most 16 utterances per microbatch and four-step
gradient accumulation. Guided attention starts at 0.2 and decays over 8,000 updates.
The run is bounded by 40 epochs, 40,000 updates or eight training hours, whichever
comes first. The expected epoch-limited budget is approximately 28,560 optimizer
updates; batch counts can vary slightly with length bucketing. Extending the pilot
budget from 20 to 40 epochs is not evidence that intelligibility will improve.

GPU 0 trains the prior. GPU 1 decodes previews and runs the existing ASR diagnostic.
Latest resumable checkpoints are saved every 250 updates and every validation;
latest plus top-three validation-NLL checkpoints upload online every five complete
epochs and at exit. TTS checkpoints use token NLL for selection: ViSQOL is not an
appropriate ranker for freely generated speech with different timing.

Four fixed known-speaker validation prompts produce free-running previews after
epoch 1 and every five completed epochs. Sampling uses a fixed seed per prompt,
preserving training RNG. W&B receives generated and reference WAVs at original
levels, waveform/log-mel panels with independent durations, phoneme cross-attention
maps, generated transcripts, ASR word error, reference ASR error and EOS success.
WAVs, bitstreams, figures and diagnostics are also uploaded as preview artifacts.
The ten-second generation cap is logged; lower teacher-forced loss alone does
not establish intelligible TTS.

## Operation

The cache, launch records, checkpoints and logs live in
`outputs/mdctcodec_tts_rangefix/`. The active trainer lives in
`src/training/mdctcodec_tts.py`; historical launchers remain preserved in `archive/`.

```bash
python scripts/tools/train_mdctcodec_tts.py --config configs/research/mdctcodec_tts_rangefix.yaml
python scripts/tools/train_mdctcodec_tts.py --resume outputs/mdctcodec_tts_rangefix/stage2/checkpoints/last.pt
```

GPU preflight includes three optimizer updates, validation, a short generated
bitstream decoded by the frozen codec, ASR, waveform/log-mel plotting and text
alignment plotting. A separate one-step restore checks optimizer continuation.
Seventeen focused codec, prior and audio-media tests pass. The short preflight
audio verifies execution only, not intelligibility.
