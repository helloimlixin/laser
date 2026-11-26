# Utilities

Start training with [`train.py`](../train.py) and use the
[manual debugging guide](../docs/manual-debugging.md) to inspect or step through
a recipe.

| Task | Command or tool |
| --- | --- |
| Inspect a composed recipe | `python train.py --config configs/stage1/ffhq.yaml --dry-run` |
| Build a Lightning token cache manually | [`build_token_cache.py`](tools/build_token_cache.py) |
| Build an upstream compound token cache | [`build_official_imagenet_token_cache.py`](tools/build_official_imagenet_token_cache.py) |
| Measure reconstruction FID | [`compute_rfid.py`](../compute_rfid.py) |

Previous launchers, sweeps, and experiment scripts are preserved in
[`archive/scripts/`](../archive/scripts/), including one-off audits and
diagnostics. See the [archive notes](../archive/README.md).
