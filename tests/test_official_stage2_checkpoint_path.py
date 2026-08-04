import math
from pathlib import Path

import pytest
import torch

from scripts.train_official_rqtransformer_laser_stage2 import (
    create_cosine_lr_scheduler,
    persistent_checkpoint_dir,
)


def test_checkpoint_dir_defaults_below_output():
    output = Path("/workspace/Projects/laser/outputs/example/stage2")

    assert persistent_checkpoint_dir(output, None) == output / "checkpoints"


def test_checkpoint_dir_rejects_ephemeral_tmp():
    with pytest.raises(ValueError, match="must be under /workspace"):
        persistent_checkpoint_dir(Path("/workspace/outputs/example"), Path("/tmp/checkpoints"))


def make_optimizer(lr=5e-4):
    parameter = torch.nn.Parameter(torch.ones(()))
    return torch.optim.AdamW([parameter], lr=lr)


def test_cosine_lr_scheduler_backfills_legacy_checkpoint_progress():
    optimizer = make_optimizer()
    scheduler = create_cosine_lr_scheduler(
        optimizer,
        initial_lr=5e-4,
        min_lr=0.0,
        total_steps=62_500,
        completed_steps=12_500,
    )

    expected = 5e-4 * 0.5 * (1.0 + math.cos(math.pi * 0.2))
    assert scheduler.last_epoch == 12_500
    assert optimizer.param_groups[0]["lr"] == pytest.approx(expected)


def test_cosine_lr_scheduler_round_trips_checkpoint_state():
    optimizer = make_optimizer()
    scheduler = create_cosine_lr_scheduler(
        optimizer, initial_lr=5e-4, min_lr=0.0, total_steps=100
    )
    for _ in range(17):
        optimizer.step()
        scheduler.step()

    resumed_optimizer = make_optimizer(lr=optimizer.param_groups[0]["lr"])
    resumed = create_cosine_lr_scheduler(
        resumed_optimizer,
        initial_lr=5e-4,
        min_lr=0.0,
        total_steps=100,
        completed_steps=17,
        state_dict=scheduler.state_dict(),
    )

    assert resumed.last_epoch == 17
    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(
        optimizer.param_groups[0]["lr"]
    )
