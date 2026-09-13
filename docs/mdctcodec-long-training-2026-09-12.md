# Larger MDCTCodec-LASER training budgets

Launched 2026-09-12 at 19:51 UTC, following the request to investigate undertraining in both stages.

The epoch-60 TTS endpoint reduced fixed-test Whisper WER from 26.89% at epoch 40 to 20.49%, although the best teacher-forced validation checkpoint remained epoch 29. Stage-1 LASER's mean validation ViSQOL increased from 4.0813 during updates 100k–150k to 4.0970 during 150k–200k. These trends motivate a larger-budget experiment; they do not establish that more compute will close the entire gap. The existing 200k codec budget already matches the iteration count stated in the [MDCTCodec paper](https://arxiv.org/html/2411.00464v1).

## Running experiments

| Job | Parent | Target | W&B |
|---|---|---|---|
| Stage-1 LASER | 200,000 generator updates | 400,000 updates | [ouayw08z](https://wandb.ai/helloimlixin-rutgers/laser/runs/ouayw08z) |
| Stage-2 TTS | epoch 60 / step 42,821 | epoch 160, subject to time budget | [s03xag7g](https://wandb.ai/helloimlixin-rutgers/laser/runs/s03xag7g) |
| Stage-1 RVQ | 200,000 generator updates | 400,000 updates | Queued automatically on the first available GPU |

The supervisor targets at most 24 assigned GPU-hours across training, validation, and dependent benchmarks. It sums each active job's wall time, including pauses for evaluation or uploads. Graceful checkpoint shutdown can add several minutes. Stage 2 also has a six-hour job limit. These are ceilings, so epoch/update targets are not completion claims. The supervisor does not terminate the cluster; idle provisioned hardware may still be billed by the provider.

Monitor `outputs/mdctcodec_long_campaign/status.json`; per-job logs live beside it. Supervisor PID is recorded in `launch.json`. A failed training job stops the other active training job; checkpoint and traceback records remain available. Dependent codec benchmarking requires both full 400k budgets and exact matching data-order audits.

## Stage 1

Both arms retain their complete 200k training prefix, model weights, optimizer moments, discriminator, and LASER coefficient-range observer. The common learning rate starts continuously at 0.00015825383404281294 and decays by cosine to 0.00002 over 200k additional generator updates. Both generator and discriminator follow that schedule. Architecture, losses, data, crop sampler, precision, quantization, and the nominal 6 kbps serialized format are unchanged.

The old checkpoint had closed a partial terminal epoch but retained stale Lightning batch counters. Preparation resets only the within-epoch counters and starts saved epoch 235 at batch zero in both arms, preserving global step counters. The paired crop/data-order audit includes the original prefix and the new join. This is an explicit continuation, not a claim to reproduce an uninterrupted run's entire stochastic trajectory.

Latest and top-three validation-ViSQOL checkpoints upload every five completed epochs and at exit. The prior best three are copied into the new output directory and stay eligible if the continuation regresses. Eight fixed validation examples provide original-level WAVs, serialized payloads, log-mel/STFT plots, waveform/MDCT views, error plots, and frequency profiles every five epochs.

Preparation and preflight evidence:

- `outputs/mdctcodec_matched_6kbps_long/prepared.json`: parent hashes, resumed state hashes, optimizer preservation, lineage.
- `outputs/mdctcodec_matched_6kbps_long/preflight_verified.json`: both arms completed two additional generator/discriminator updates, advanced optimizer counters, matched batches/crops, continuous LR, and full-audio serialized validation near nominal 6 kbps (frame padding included).
- `scripts/tools/continue_mdctcodec_matched.py`: prepare/train entry point.

## Stage 2

The prior keeps the same 61.9M-parameter architecture, whole-utterance training data, conditioning, optimizer moments, and frozen codec/cache. Learning rate rises linearly from the saved 0.000010000096597397508 to 0.0001 over 1,000 optimizer steps, then decays by cosine to 0.00001 at the configured endpoint. The low terminal LR from the previous run is therefore not simply extended for another 100 epochs.

The codec hash remains `3f4d31216ff54c45e0294d17eb51fa83709561c9a3528551c5fb5fed2fbeaf29`. The concurrently improving stage-1 dictionary cannot be silently substituted into this prior: adopting it requires re-encoding a compatible token cache and training a corresponding prior.

Checkpoint selection now adds corpus Whisper-large-v3 WER on 64 fixed, speaker-balanced validation prompts. The manifest is chosen deterministically before any continuation updates; its texts are checked against both train and test. A checkpoint-60 baseline anchors the same validation set, and generation evaluation repeats every ten complete epochs. The scorer pins the same Whisper model and transcription settings used by the benchmark and records reference-speech WER, CER, EOS fraction, audio hashes, payloads, and per-file transcripts. The two-clip/30-frame smoke exercise is stored separately and is not a model-quality estimate.

Latest full optimizer checkpoint, top three validation-WER checkpoints, and top three token-NLL checkpoints upload every five completed epochs and at exit. The 16 existing validation previews continue every five epochs, with original-level audio, waveforms, log-mel plots, and blue attention maps.

Config: `configs/research/mdctcodec_tts_long.yaml`. Validation manifest: `outputs/mdctcodec_tts_long/stage2/generation_validation_manifest.json`.

## Automatic evaluation

`scripts/tools/run_mdctcodec_long.py` schedules both training stages, the matched RVQ continuation, and `scripts/tools/evaluate_mdctcodec_long.py`.

Stage 1 compares validation-selected 400k LASER/RVQ and released MDCTCodec on the existing fixed 200-recording set after verifying matched total updates and crop audits. Stage 2 evaluates the validation-WER-selected prior and the terminal endpoint against epoch 60, F5-TTS, Chatterbox, reference speech, and codec reconstruction on the previous fixed 100-recording benchmark. It uses original-level waveforms, Whisper WER/CER, ECAPA similarity, UTMOS, timing, and paired bootstrap intervals.

These existing test sets have informed experimentation, so the new results are regression comparisons, not fresh confirmation tests. The codec comparison has one paired training seed; released TTS models have different training data/compute and unknown VCTK pretraining overlap. Neither benchmark supports a comprehensive SOTA claim by itself.

Verification: 26 relevant unit tests passed across model/data/serialization, schedules, validation-only prompt selection, media rendering, benchmark aggregation, and prior-extension evaluation logic; both codec arms and the TTS prior passed GPU continuation preflights.
