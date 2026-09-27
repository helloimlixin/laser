"""Helpers for constructing Lightning Trainers from Hydra configs."""

from math import ceil

from lightning.pytorch.callbacks import Callback


class TrainerGlobalStepLogger(Callback):
    """Log an explicit W&B-compatible global-step metric every train batch."""

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(getattr(trainer, "global_step", 0) or 0)
        if step < 0:
            return
        pl_module.log(
            "trainer/global_step",
            float(step),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=False,
        )
        pl_module.log(
            "global_step",
            float(step),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=False,
        )


def define_wandb_step_metrics(logger):
    """Tell W&B to use trainer/global_step as the default scalar x-axis."""
    experiment = getattr(logger, "experiment", None)
    if experiment is None:
        return
    try:
        experiment.define_metric("trainer/global_step")
        experiment.define_metric("*", step_metric="trainer/global_step")
    except Exception:
        return


def resolve_val_check_interval(datamodule, raw, devices=1):
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
    try:
        per_rank_n = ceil(n / max(1, int(devices)))
    except (TypeError, ValueError):
        per_rank_n = n
    return min(steps, per_rank_n)
