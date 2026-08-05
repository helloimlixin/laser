import math
from pathlib import Path

import pytest
import torch

from scripts.train_official_rqtransformer_laser_stage2 import (
    create_cosine_lr_scheduler,
    optimizer_state_to_ids,
    optimizer_state_to_names,
    optimizer_state_uses_names,
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


def test_optimizer_state_rekeys_between_ddp_ids_and_fsdp_names():
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.GELU(),
        torch.nn.Linear(4, 2),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(torch.ones(2, 3)).square().mean()
    loss.backward()
    optimizer.step()
    id_state = optimizer.state_dict()

    named_state = optimizer_state_to_names(id_state, model)
    assert optimizer_state_uses_names(named_state)
    assert set(named_state["state"]) == set(dict(model.named_parameters()))

    parameter_names = [name for name, _ in model.named_parameters()]
    restored = optimizer_state_to_ids(named_state, parameter_names)
    assert not optimizer_state_uses_names(restored)
    assert restored["param_groups"] == id_state["param_groups"]
    for parameter_id, state in id_state["state"].items():
        assert restored["state"][parameter_id].keys() == state.keys()
        for key, value in state.items():
            assert torch.equal(restored["state"][parameter_id][key], value)
