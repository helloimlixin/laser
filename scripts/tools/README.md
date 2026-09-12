# Shared utilities

Normal training uses [`train.py`](../../train.py) and the stage/dataset YAML
recipes documented in the [README](../../README.md). Stage 2 builds its cache
automatically; most runs do not need to call these utilities directly.

- `build_token_cache.py`: extract sparse tokens from Lightning stage-1 checkpoints.
- `build_official_imagenet_token_cache.py`: extract compound caches from historical
  upstream checkpoints, including supported face and LSUN datasets.
- `compute_rfid.py`: reconstruction FID (also exposed by root `compute_rfid.py`).
- Other files support cache conversions, diagnostics, sampling reports, and local
  run maintenance. Their individual `--help` describes required inputs.

Experiment launchers have moved to [`archive/scripts/`](../../archive/scripts/).
