# Historical experiments

Current image training starts at [`train.py`](../train.py), with recipes under
[`configs/stage1/`](../configs/stage1/) and [`configs/stage2/`](../configs/stage2/).
See the [launch guide](../README.md).

- `scripts/` contains the former top-level training launchers, sweeps, scheduler
  submissions, experiment drivers, and specialized evaluation scripts.
- `configs/` contains former one-off/pipeline YAML files. The old root-level
  `config.yaml` is preserved as `root-config.yaml`.
- `docs/` contains the former root-level experiment notes and TODO list.

These files preserve experiment history. Shell scripts retain original paths,
run IDs, and machine-specific assumptions, and may need adaptation to run from
this archive. Python import paths and repository-root calculations were updated
where modules are still used by research code and regression tests.

The reusable compound RQ-Transformer implementation moved from
`scripts/train_official_rqtransformer_laser_stage2.py` to
[`src/training/rqtransformer.py`](../src/training/rqtransformer.py). Launch its
configurable FFHQ recipe through `train.py`, as described in the launch guide.
Shared cache and maintenance utilities remain under [`scripts/tools/`](../scripts/tools/).

Previous launcher-style YAML can still be inspected through the compatibility path:

```bash
python train.py --config archive/configs/exp1.yaml --dry-run
```

Develop new experiments in the active modules and configs. Historical results
and reconstruction figures are preserved in [`docs/results.md`](../docs/results.md).
