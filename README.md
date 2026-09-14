# LASER

LASER trains an image autoencoder with a sparse dictionary bottleneck (stage 1),
then a transformer over its atom/coefficient tokens (stage 2).
**Launch training through `train.py`; put experiment settings in YAML.**

## Install

Use Python 3.10+ and a PyTorch installation compatible with your GPU.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Training logs to W&B. Run `wandb login`, or set `WANDB_MODE=offline` for local logs
(`WANDB_MODE=disabled` disables W&B). Dataset downloads are separate from training.

## Dataset paths

Run commands from the repository root. Set the common dataset directory:

```bash
export DATA_DIR=/path/to/datasets
```

| Dataset | Default location under `DATA_DIR` | Individual override |
| --- | --- | --- |
| CelebA-HQ | `celebahq/` | `CELEBAHQ_DIR` |
| FFHQ | `ffhq/` | `FFHQ_DIR` |
| LSUN Church | `lsun/church/` | `LSUN_CHURCH_DIR` |
| LSUN Bedroom | `lsun/bedroom/` | `LSUN_BEDROOM_DIR` |
| LSUN Cat | `lsun/cat/` | `LSUN_CAT_DIR` |
| ImageNet | `imagenet/` | `IMAGENET_DIR` |
| CC3M | `cc3m/` | `CC3M_DIR` |

You can also set `data.data_dir` in a recipe or append
`data.data_dir=/path/to/dataset` to a command. Without `DATA_DIR`, the default is
`../data` relative to the repository root.

- **CelebA-HQ / FFHQ:** extracted images; FFHQ accepts recursive image folders or
  explicit `train/` and `val/` folders. A flat FFHQ root gets a seeded 90/5/5 split.
- **LSUN:** extracted images in the category directory, or standard
  `<category>_train_lmdb/` and `<category>_val_lmdb/` databases inside `lsun/`.
  Church accepts the standard `church_outdoor_*_lmdb` names as well as
  `church_*_lmdb`. All three category recipes accept either layout.
- **ImageNet:** `train/<class>/*.JPEG` and `val/<class>/*.JPEG`, with matching class
  directory names. Organize validation images by class before training.
- **CC3M:** local WebDataset `.tar` shards containing image/caption pairs with
  matching stems (`.jpg` plus `.txt`, or a caption in `.json`). Shards can be
  directly in the dataset root or its `wds/` or `webdataset/` directory.

## Stage 1: train the autoencoder

```bash
python train.py --config configs/stage1/celebahq.yaml
python train.py --config configs/stage1/ffhq.yaml
python train.py --config configs/stage1/lsun-church.yaml
python train.py --config configs/stage1/lsun-bedroom.yaml
python train.py --config configs/stage1/lsun-cat.yaml
python train.py --config configs/stage1/imagenet.yaml
python train.py --config configs/stage1/cc3m.yaml
```

Choose the command for your dataset. These recipes use 256×256 images and an
8×8 sparse latent grid. Shared model settings live in
[`configs/model/image.yaml`](configs/model/image.yaml); shared training settings
live in [`configs/stage1/base.yaml`](configs/stage1/base.yaml). Dataset recipes
contain only their differences. Face recipes use 2,048 atoms with sparsity 2;
the other recipes use 16,384 atoms with sparsity 4.

## Stage 2: train the prior

Run after stage 1 finishes for the same dataset:

```bash
python train.py --config configs/stage2/celebahq.yaml
python train.py --config configs/stage2/ffhq.yaml
python train.py --config configs/stage2/lsun-church.yaml
python train.py --config configs/stage2/lsun-bedroom.yaml
python train.py --config configs/stage2/lsun-cat.yaml
python train.py --config configs/stage2/imagenet.yaml
python train.py --config configs/stage2/cc3m.yaml
```

Stage 2 finds a checkpoint under `outputs/<dataset>/stage1`, builds a token cache,
and trains the prior. LSUN output directory names use underscores, such as
`outputs/lsun_church/`. Later launches reuse the cache. ImageNet uses class
conditioning; CC3M caches captions and uses text conditioning; face and LSUN
priors are unconditional. Shared settings live in
[`configs/stage2/base.yaml`](configs/stage2/base.yaml).

To choose the stage-1 checkpoint explicitly:

```bash
python train.py --config configs/stage2/ffhq.yaml \
  token_cache.stage1_checkpoint=/path/to/stage1.ckpt
```

When changing the source checkpoint, dataset split, image size, seed, or tokenization,
use a new `token_cache_path` or set `token_cache.force=true` to rebuild. If you
changed the stage-1 output directory, set `token_cache.stage1_output_root` to it.
An existing compatible cache can be used with
`token_cache.build=false token_cache_path=/path/to/train.pt`.

## Customize a run

Edit the YAML for settings you want to share with collaborators. For a separate
experiment, copy a dataset recipe beside the original, give it a new name, and
set `output_dir` and the corresponding stage-1 checkpoint/cache paths.
Command-line `key=value` overrides take precedence over YAML:

```bash
# Inspect the fully resolved configuration without loading models or data.
python train.py --config configs/stage1/ffhq.yaml --dry-run

# Four GPUs; batch sizes are per process. Reduce them to fit GPU memory.
python train.py --config configs/stage1/ffhq.yaml train.devices=4 data.batch_size=8
python train.py --config configs/stage2/ffhq.yaml train_ar.devices=4 train_ar.batch_size=16

# Resume model, optimizer, and training progress from a trusted checkpoint.
python train.py --config configs/stage1/ffhq.yaml ckpt_path=/path/to/stage1/last.ckpt
python train.py --config configs/stage2/ffhq.yaml ckpt_path=/path/to/stage2/last.ckpt
```

The standard recipes use the Lightning checkpoint/cache path. Historical upstream
`.pt` checkpoints use a different format. The preserved compound RQ-Transformer
is also reachable through `train.py`, with its settings in
[`configs/stage2/ffhq-compound.yaml`](configs/stage2/ffhq-compound.yaml):

```bash
python train.py --config configs/stage2/ffhq-compound.yaml \
  options.checkpoint=/path/to/upstream-stage1.pt \
  options.token_cache=/path/to/compound-cache.pt
```

The matched ImageNet VAR experiment trains both VQ and LASER OMP from random
initialization, using two GPUs per arm. Use
[`imagenet-var-vq-scratch.yaml`](configs/experiments/imagenet-var-vq-scratch.yaml)
and [`imagenet-var-scratch.yaml`](configs/experiments/imagenet-var-scratch.yaml).
The [scratch comparison notes](docs/imagenet-var-scratch.md) describe the shared
initialization, factor-16 layout, training budgets, W&B runs, and evaluation.
The user authorized this matched ImageNet-only recipe after reviewing the
differences from published VAR. See the [source audit](docs/var-stage1-protocol-audit.md).

For the compound RQ backend, multiple GPUs use `torchrun --standalone --nproc_per_node=4
train.py --config ...`; supply an existing compound cache and retain its calibrated
coefficient metadata. Historical results and their original recipes are in
[`docs/results.md`](docs/results.md). The standard recipes are starting points for
new runs, not reproductions of every archived experiment.

## Code map

| Path | Responsibility |
| --- | --- |
| [`train.py`](train.py) | Single public training entry point |
| [`src/training/cli.py`](src/training/cli.py) | Compose YAML, validate settings, select a stage |
| [`src/training/stage1.py`](src/training/stage1.py) | Autoencoder training and reconstruction evaluation |
| [`src/training/stage2.py`](src/training/stage2.py) | Token-cache creation and prior training |
| [`src/training/common.py`](src/training/common.py) | Shared checkpoint and logging helpers |
| [`configs/`](configs/) | Shared settings and dataset recipes |
| [`src/models/`](src/models/) | Autoencoders, dictionary bottleneck, and priors |
| [`src/data/`](src/data/) | Dataset and token-cache loaders |
| [`scripts/tools/`](scripts/tools/) | Cache, evaluation, and maintenance utilities |
| [`archive/`](archive/) | Previous launchers, sweeps, one-off configs, and research scripts |

Add reusable behavior to `src/`, experiment settings to `configs/`, and relevant
coverage to `tests/`. New experiments should not need another launcher script.
Old flag-based and pipeline commands are handled separately in
`src/training/legacy_cli.py` for compatibility.

## Tests

```bash
pytest -q tests/test_training_recipes.py tests/test_lsun_lmdb.py
pytest -q
```

## License

See [LICENSE](LICENSE).
