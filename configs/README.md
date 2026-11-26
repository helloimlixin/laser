# Training configuration

For a new run, choose the file named after your dataset in [`stage1/`](stage1/)
or [`stage2/`](stage2/). The standard recipe names are `celebahq`, `ffhq`,
`lsun-church`, `lsun-bedroom`, `lsun-cat`, `imagenet`, `cc3m`, and `coco2014`.

```bash
python train.py --config configs/stage1/ffhq.yaml --dry-run
```

Use [`experiments/`](experiments/) for documented comparisons and the CPU smoke
test, and [`research/`](research/) for standalone historical trials. Some
historical recipes still live alongside the standard recipes because older
launchers or regression tests refer to those paths.

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
