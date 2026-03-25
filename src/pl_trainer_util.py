"""Helpers for constructing Lightning Trainers from Hydra configs."""


def resolve_val_check_interval(datamodule, raw):
    """Ensure int val_check_interval is <= len(train_dataloader); Lightning rejects larger values."""
    if raw is None:
        return 1.0
    if isinstance(raw, bool):
        return 1.0
    if isinstance(raw, float) and 0.0 < raw <= 1.0:
        return raw
    try:
        steps = int(raw)
    except (TypeError, ValueError):
        return raw
    if steps < 1:
        return raw
    datamodule.setup("fit")
    n = len(datamodule.train_dataloader())
    if n < 1:
        return 1.0
    return min(steps, n)
