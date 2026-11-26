# Manual debugging and Git sync

Run the commands from the repository root in the environment from the
[installation instructions](../README.md#install).

## Inspect a recipe

```bash
python train.py --config configs/stage1/ffhq.yaml --dry-run data.batch_size=2
```

`--dry-run` prints the composed YAML after applying the command-line overrides.
Check paths, batch size, and the selected `stage` and `backend` in that output.
For the default Lightning backend, execution goes through
[`src/training/stage1.py`](../src/training/stage1.py) or
[`src/training/stage2.py`](../src/training/stage2.py). A named backend has its
own module under [`src/training/`](../src/training/).

## Run a CPU smoke test

```bash
debug_dir=$(mktemp -d)
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 WANDB_MODE=disabled \
  python train.py --config configs/experiments/continuous-compound-smoke.yaml \
  pilot.device=cpu "output_dir=$debug_dir"
ls "$debug_dir"
```

This exercises the continuous-coefficient stage-2 pilot with synthetic inputs.
Use a fresh output directory for each run. The [pilot implementation](../src/training/continuous_compound.py)
handles model construction, optimization, and evaluation.

To step through dispatch, insert `-m pdb` after `python` in the smoke
command. Enter `n` until you reach the call to `main()`, then `s` to enter
it. Enter `b run` and `c` to stop at backend dispatch, where `p cfg`
shows the composed configuration. For a failing test, install pytest and
use `python -m pytest -x --pdb path/to/test.py`.

## Review changes before syncing

```bash
git status --short
git diff --check
git diff --cached --stat
git diff --cached
```

Stage explicit paths for source, recipes, and supporting tests or notes. Keep
datasets, environment files, checkpoints, W&B logs, and live run state local.
Record the recipe overrides and artifact identifiers alongside results so a
fresh checkout can find the inputs for a run.
