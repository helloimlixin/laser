import copy
import math
from pathlib import Path

import pytest
import torch

import scripts.train_official_rqtransformer_laser_stage2 as stage2
from scripts.train_official_rqtransformer_laser_stage2 import (
    atomic_torch_save,
    create_cosine_lr_scheduler,
    create_warmup_linear_lr_scheduler,
    gather_rank_rng_states,
    model_only_best_snapshot,
    optimizer_state_to_ids,
    optimizer_state_to_names,
    optimizer_state_uses_names,
    persistent_checkpoint_dir,
    remap_resume_batch_index,
    restore_rank_rng_state,
    snapshot_checkpoint,
    uses_inception_score,
)


def test_checkpoint_dir_defaults_below_output():
    output = Path("/workspace/Projects/laser/outputs/example/stage2")

    assert persistent_checkpoint_dir(output, None) == output / "checkpoints"


def test_checkpoint_dir_rejects_ephemeral_tmp():
    with pytest.raises(ValueError, match="must be under /workspace"):
        persistent_checkpoint_dir(Path("/workspace/outputs/example"), Path("/tmp/checkpoints"))


def test_checkpoint_snapshot_reuses_storage(tmp_path):
    source = tmp_path / "last.pt"
    target = tmp_path / "best.pt"
    source.write_bytes(b"checkpoint payload")

    snapshot_checkpoint(source, target)

    assert target.read_bytes() == source.read_bytes()
    assert target.stat().st_ino == source.stat().st_ino


def test_model_only_best_snapshot_keeps_evaluation_state_not_resume_state():
    snapshot = {
        "epoch": 15,
        "global_step": 7395,
        "fid": 25.1,
        "state_dict": {"weight": torch.ones(2)},
        "optimizer": {"state": {1: {"step": 3}}},
        "scheduler": {"last_epoch": 7395},
        "rng_state_by_rank": [{"torch_cpu": torch.get_rng_state()}],
        "config": {"dataset": "lsun_church"},
        "best_fid": [(25.1, "best.pt")],
        "best_inception": [],
    }

    compact = model_only_best_snapshot(snapshot)

    assert compact["checkpoint_kind"] == "model_only_best"
    assert compact["resume_capable"] is False
    assert compact["state_dict"] is snapshot["state_dict"]
    assert "optimizer" not in compact
    assert "scheduler" not in compact
    assert "rng_state_by_rank" not in compact


def test_atomic_save_stages_locally_and_retries_failed_persistent_copy(
    tmp_path, monkeypatch
):
    target = tmp_path / "persistent" / "last.pt"
    target.parent.mkdir()
    target.write_bytes(b"previous valid checkpoint")
    staging = tmp_path / "local-staging"
    real_copyfile = stage2.shutil.copyfile
    attempts = 0

    def flaky_copyfile(source, destination):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            Path(destination).write_bytes(b"incomplete")
            raise OSError("simulated network filesystem interruption")
        return real_copyfile(source, destination)

    monkeypatch.setenv("LASER_CHECKPOINT_STAGING_DIR", str(staging))
    monkeypatch.setattr(stage2.shutil, "copyfile", flaky_copyfile)
    monkeypatch.setattr(stage2.time, "sleep", lambda _seconds: None)

    atomic_torch_save({"value": torch.arange(8)}, target)

    assert attempts == 2
    assert torch.equal(
        torch.load(target, map_location="cpu", weights_only=True)["value"],
        torch.arange(8),
    )
    assert not target.with_suffix(".pt.tmp").exists()
    assert list(staging.iterdir()) == []


def test_rank_rng_state_round_trips_exactly_on_cpu():
    device = torch.device("cpu")
    torch.manual_seed(1234)
    states = gather_rank_rng_states(device)
    expected = torch.rand(16)
    torch.manual_seed(9999)

    assert restore_rank_rng_state({"rng_state_by_rank": states}, device, seed=7)
    assert torch.equal(torch.rand(16), expected)


def test_legacy_rng_resume_uses_deterministic_rank_seed():
    device = torch.device("cpu")

    assert not restore_rank_rng_state({}, device, seed=4321)
    first = torch.rand(16)
    assert not restore_rank_rng_state({}, device, seed=4321)
    assert torch.equal(torch.rand(16), first)


def test_rng_checkpoint_reproduces_dropout_optimizer_trajectory():
    device = torch.device("cpu")
    inputs = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 31
    targets = torch.arange(16, dtype=torch.float32).reshape(8, 2) / 15

    def make_model():
        return torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(8, 2),
        )

    def step(model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
        return loss.detach().clone()

    torch.manual_seed(17)
    uninterrupted = make_model()
    uninterrupted_optimizer = torch.optim.AdamW(uninterrupted.parameters(), lr=1e-3)
    step(uninterrupted, uninterrupted_optimizer)
    checkpoint = {
        "model": copy.deepcopy(uninterrupted.state_dict()),
        "optimizer": copy.deepcopy(uninterrupted_optimizer.state_dict()),
        "rng_state_by_rank": gather_rank_rng_states(device),
    }
    expected_loss = step(uninterrupted, uninterrupted_optimizer)

    resumed = make_model()
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    resumed.load_state_dict(checkpoint["model"])
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    restore_rank_rng_state(checkpoint, device, seed=0)
    resumed_loss = step(resumed, resumed_optimizer)

    assert torch.equal(resumed_loss, expected_loss)
    for expected, actual in zip(uninterrupted.parameters(), resumed.parameters()):
        assert torch.equal(actual, expected)


def test_resume_cursor_remaps_legacy_four_rank_checkpoint_to_two_ranks():
    remapped, old_world_size = remap_resume_batch_index(
        8_000,
        saved_config={"batch_size": 32, "total_batch_size": 2_048},
        batch_size=32,
        world_size=2,
        total_batch_size=2_048,
        global_step=13_000,
        start_epoch=20,
        optimizer_steps_per_epoch=625,
    )

    assert old_world_size == 4
    assert remapped == 16_000


def test_resume_cursor_uses_recorded_world_size_and_new_microbatch():
    remapped, old_world_size = remap_resume_batch_index(
        4_000,
        saved_config={
            "batch_size": 64,
            "total_batch_size": 2_048,
            "world_size": 2,
        },
        batch_size=16,
        world_size=4,
        total_batch_size=2_048,
        global_step=13_000,
        start_epoch=20,
        optimizer_steps_per_epoch=625,
    )

    assert old_world_size == 2
    assert remapped == 8_000


def test_inception_score_is_only_used_for_imagenet():
    assert uses_inception_score("imagenet")
    assert not uses_inception_score("ffhq")
    assert not uses_inception_score("celebahq")


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


def test_var_warmup_linear_schedule_warms_then_decays_to_floor():
    optimizer = make_optimizer(lr=3.2e-4)
    scheduler = create_warmup_linear_lr_scheduler(
        optimizer,
        initial_lr=3.2e-4,
        min_lr=3.2e-6,
        total_steps=100,
        warmup_steps=10,
        warmup_start_ratio=0.005,
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.6e-6)
    for _ in range(10):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3.2e-4)
    for _ in range(90):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3.2e-6)


def test_var_warmup_linear_schedule_round_trips_state():
    optimizer = make_optimizer(lr=3.2e-4)
    scheduler = create_warmup_linear_lr_scheduler(
        optimizer,
        initial_lr=3.2e-4,
        min_lr=3.2e-6,
        total_steps=100,
        warmup_steps=10,
    )
    for _ in range(17):
        optimizer.step()
        scheduler.step()

    resumed_optimizer = make_optimizer(lr=optimizer.param_groups[0]["lr"])
    resumed = create_warmup_linear_lr_scheduler(
        resumed_optimizer,
        initial_lr=3.2e-4,
        min_lr=3.2e-6,
        total_steps=100,
        warmup_steps=10,
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
