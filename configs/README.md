# Training configuration

Run `python train.py --config configs/stage1/<dataset>.yaml` or the corresponding
`stage2` recipe. Dataset names are `celebahq`, `ffhq`, `lsun-church`, `lsun-bedroom`,
`lsun-cat`, `imagenet`, and `cc3m`.

Hydra composes the `defaults` list in order. A dataset recipe includes its stage's
`base.yaml`, selects a `data/` configuration, then applies its own settings. Model
settings are shared in `model/image.yaml`; stage defaults are in
`stage1/base.yaml` and `stage2/base.yaml`. Command-line overrides apply last.

Copy a dataset recipe within its stage directory to start an experiment. Keep
`# @package _global_` so its sections merge into the root configuration. Give
independent runs distinct output directories and cache paths. Use `--dry-run` to
inspect all inherited settings before training.

`config.yaml`, `config_ar.yaml`, and the remaining model/data groups are shared
Hydra foundations and compatibility presets. Historical standalone/pipeline
recipes are in [`archive/configs/`](../archive/configs/).

The optional `stage2/ffhq-compound.yaml` uses the preserved upstream checkpoint
backend; its required `options.checkpoint` and `options.token_cache` must be set
before launch or dry run.
