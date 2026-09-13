# Shared utilities

Normal training uses [`train.py`](../../train.py) and the stage/dataset YAML
recipes documented in the [README](../../README.md). Stage 2 builds its cache
automatically; most runs do not need to call these utilities directly.

- `build_token_cache.py`: extract sparse tokens from Lightning stage-1 checkpoints.
- `build_official_imagenet_token_cache.py`: extract compound caches from historical
  upstream checkpoints, including supported face and LSUN datasets.
- `compute_rfid.py`: reconstruction FID (also exposed by root `compute_rfid.py`).
- `watch_mdctcodec_audio_media.py`: publish matched codec listening samples,
  spectrograms, waveforms and token diagnostics to a live W&B dashboard from
  committed checkpoints; see the [audio dashboard notes](../../docs/mdctcodec-audio-media-2026-09-12.md).
- `cache_mdctcodec_tts.py` and `train_mdctcodec_tts.py`: prepare full-utterance
  tokens and train the frozen-codec speech prior; see the
  [MDCTCodec TTS run](../../docs/mdctcodec-stage2-rangefix-2026-09-12.md).
- `benchmark_mdctcodec_tts.py` and `run_mdctcodec_tts_benchmark.py`: run a fixed
  100-speaker TTS comparison with F5-TTS and Chatterbox Turbo, including ASR,
  speaker similarity, predicted quality, timing and listening samples; see the
  [TTS benchmark protocol](../../docs/mdctcodec-tts-benchmark-2026-09-12.md).
- Other files support cache conversions, diagnostics, sampling reports, and local
  run maintenance. Their individual `--help` describes required inputs.

Experiment launchers have moved to [`archive/scripts/`](../../archive/scripts/).
