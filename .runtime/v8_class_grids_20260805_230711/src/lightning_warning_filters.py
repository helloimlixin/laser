"""Filter benign Lightning / PyTorch training warnings (upstream deprecations, known-intentional setups)."""

import warnings


def register() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r".*LeafSpec.*is deprecated.*TreeSpec.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*does not have many workers which may be a bottleneck\..*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Found \d+ module\(s\) in eval mode at the start of training\..*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*You are using `torch\.load` with `weights_only=False`.*",
    )
