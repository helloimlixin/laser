import errno
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp


ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ROOT / "third_party" / "rq-vae-transformer"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trainer_module = _load_module(
    "rqvae_trainer_checkpoint_under_test",
    THIRD_PARTY / "rqvae" / "trainers" / "trainer.py",
)
checkpoint_module = _load_module(
    "rqvae_checkpoint_under_test",
    THIRD_PARTY / "rqvae" / "utils" / "checkpoint.py",
)
writer_module = _load_module(
    "rqvae_writer_checkpoint_under_test",
    THIRD_PARTY / "rqvae" / "utils" / "writer.py",
)
TrainerTemplate = trainer_module.TrainerTemplate
validate_resume_checkpoint = checkpoint_module.validate_resume_checkpoint


def _trainer_stub(result_path):
    trainer = object.__new__(TrainerTemplate)
    trainer.config = SimpleNamespace(
        result_path=str(result_path),
        experiment={"rfid_backend": "original-rqvae"},
    )
    trainer.distenv = SimpleNamespace(master=True, world_size=1, world_rank=0)
    trainer.device = torch.device("cpu")
    trainer.resume_signature = {"version": 1, "sha256": "test-signature"}
    trainer.lineage_exact = True
    trainer.lineage_origin = "test"
    trainer._last_checkpoint_id = None
    return trainer


def _distributed_save_worker(rank, world_size, init_file, result_path):
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        trainer = _trainer_stub(result_path)
        trainer.distenv = SimpleNamespace(
            master=rank == 0,
            world_size=world_size,
            world_rank=rank,
        )
        trainer.loader_trn = [None] * 4
        trainer.model = torch.nn.Linear(2, 2)
        trainer.model_ema = None
        trainer.writer = None
        optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1.0e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        trainer.model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        scheduler.step()
        trainer._train_accumulator = SimpleNamespace(
            state_dict=lambda: {"counter": 1, "rank": rank}
        )

        random.seed(700 + rank)
        np.random.seed(800 + rank)
        torch.manual_seed(900 + rank)
        trainer._epoch_start_rng_state = trainer._capture_rng_state()
        random.random()
        np.random.rand()
        torch.rand(2)
        trainer.save_ckpt(
            optimizer,
            scheduler,
            epoch=0,
            batch_idx=1,
            global_step=1,
            upload=False,
        )
    finally:
        torch.distributed.destroy_process_group()


def test_full_topk_rotation_does_not_exceed_existing_checkpoint_footprint(
    tmp_path, monkeypatch
):
    payload_size = 256
    contents = {
        "last_model.pt": b"candidate",
        "best_rfid_slot1_model.pt": b"epoch4",
        "best_rfid_slot2_model.pt": b"epoch5",
        "best_rfid_slot3_model.pt": b"epoch3",
    }
    for name, marker in contents.items():
        (tmp_path / name).write_bytes(marker.ljust(payload_size, b"."))
    (tmp_path / "checkpoint_policy.json").write_text(
        json.dumps(
            {
                "metric_backend": "original-rqvae",
                "best": [
                    {"epoch": 4, "rfid": 9.7, "slot": 1, "path": "best_rfid_slot1_model.pt"},
                    {"epoch": 5, "rfid": 9.9, "slot": 2, "path": "best_rfid_slot2_model.pt"},
                    {"epoch": 3, "rfid": 14.9, "slot": 3, "path": "best_rfid_slot3_model.pt"},
                ],
            }
        )
    )

    real_link = trainer_module.os.link

    def quota_counted_link(source, target):
        checkpoint_bytes = sum(
            path.stat().st_size
            for path in tmp_path.iterdir()
            if path.is_file() and path.stat().st_size == payload_size
        )
        if checkpoint_bytes + Path(source).stat().st_size > 4 * payload_size:
            raise OSError(errno.EDQUOT, "Disk quota exceeded")
        return real_link(source, target)

    monkeypatch.setattr(trainer_module.os, "link", quota_counted_link)
    trainer = _trainer_stub(tmp_path)
    best = trainer._update_best_checkpoints(
        tmp_path / "last_model.pt", epoch=6, rfid=7.7
    )

    assert [item["epoch"] for item in best] == [6, 4, 5]
    assert (tmp_path / "last_model.pt").read_bytes().startswith(b"candidate")
    assert (tmp_path / "best_rfid_slot1_model.pt").read_bytes().startswith(b"candidate")
    assert (tmp_path / "best_rfid_slot2_model.pt").read_bytes().startswith(b"epoch4")
    assert (tmp_path / "best_rfid_slot3_model.pt").read_bytes().startswith(b"epoch5")
    assert not list(tmp_path.glob(".best-rfid-*.tmp"))
    assert len(
        [path for path in tmp_path.iterdir() if path.stat().st_size == payload_size]
    ) == 4


def test_saved_checkpoint_contains_all_training_and_rng_state(tmp_path):
    trainer = _trainer_stub(tmp_path)
    trainer.loader_trn = [None] * 5
    trainer.model = torch.nn.Linear(3, 2)
    trainer.model_ema = None
    trainer.discriminator = torch.nn.Linear(2, 1)
    trainer.writer = None

    optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
    disc_optimizer = torch.optim.Adam(trainer.discriminator.parameters(), lr=1.0e-3)
    disc_scheduler = torch.optim.lr_scheduler.StepLR(disc_optimizer, step_size=10)
    trainer.disc_optimizer = disc_optimizer
    trainer.disc_scheduler = disc_scheduler

    trainer.model(torch.randn(2, 3)).sum().backward()
    optimizer.step()
    scheduler.step()
    trainer.discriminator(torch.randn(2, 2)).sum().backward()
    disc_optimizer.step()
    disc_scheduler.step()
    trainer._epoch_start_rng_state = trainer._capture_rng_state()
    trainer._train_accumulator = SimpleNamespace(
        state_dict=lambda: {"counter": 1, "metrics_sum": {}, "codebooks": []}
    )

    trainer.save_ckpt(
        optimizer,
        scheduler,
        epoch=0,
        batch_idx=1,
        global_step=1,
        upload=False,
    )
    checkpoint = torch.load(
        tmp_path / "last_model.pt", map_location="cpu", weights_only=False
    )

    assert checkpoint["checkpoint_format_version"] == 5
    assert checkpoint["checkpoint_world_size"] == 1
    assert len(checkpoint["rng_state_by_rank"]) == 1
    assert len(checkpoint["epoch_start_rng_state_by_rank"]) == 1
    assert checkpoint["train_accumulator_state_by_rank"][0]["counter"] == 1
    assert {
        "state_dict",
        "optimizer",
        "scheduler",
        "discriminator",
        "discriminator_optimizer",
        "discriminator_scheduler",
    }.issubset(checkpoint)
    metadata = validate_resume_checkpoint(
        checkpoint, steps_per_epoch=5, world_size=1
    )
    assert metadata == {
        "epoch": 0,
        "batch_idx": 1,
        "global_step": 1,
        "format_version": 5,
        "lineage_exact": True,
        "lineage_origin": "test",
        "checkpoint_id": checkpoint["checkpoint_id"],
        "warnings": [],
    }

    state = checkpoint["rng_state_by_rank"][0]
    trainer._restore_rng_state(state)
    expected = (random.random(), np.random.rand(), torch.rand(3))
    random.random()
    np.random.rand()
    torch.rand(7)
    trainer._restore_rng_state(state)
    actual = (random.random(), np.random.rand(), torch.rand(3))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_resume_validation_rejects_cursor_state_mismatch():
    checkpoint = {
        "checkpoint_format_version": 3,
        "epoch": 2,
        "batch_idx": 3,
        "global_step": 24,
        "steps_per_epoch": 10,
        "state_dict": {},
        "optimizer": {"state": {}},
        "scheduler": {"last_epoch": 24},
    }
    with pytest.raises(RuntimeError, match="Resume cursor is inconsistent"):
        validate_resume_checkpoint(checkpoint, steps_per_epoch=10, world_size=1)


def test_resume_validation_rejects_legacy_within_epoch_checkpoint():
    checkpoint = {
        "checkpoint_format_version": 4,
        "checkpoint_world_size": 1,
        "epoch": 2,
        "batch_idx": 3,
        "global_step": 23,
        "steps_per_epoch": 10,
        "state_dict": {},
        "optimizer": {"state": {}},
        "scheduler": {"last_epoch": 23},
        "rng_state_by_rank": [{}],
        "epoch_start_rng_state_by_rank": [{}],
    }
    with pytest.raises(RuntimeError, match="legacy within-epoch resume"):
        validate_resume_checkpoint(checkpoint, steps_per_epoch=10, world_size=1)


def test_resume_validation_rejects_accumulator_cursor_mismatch():
    checkpoint = {
        "checkpoint_format_version": 5,
        "checkpoint_id": "checkpoint-test",
        "lineage_exact": True,
        "lineage_origin": "test",
        "checkpoint_world_size": 1,
        "epoch": 2,
        "batch_idx": 3,
        "global_step": 23,
        "steps_per_epoch": 10,
        "state_dict": {},
        "optimizer": {"state": {}},
        "scheduler": {"last_epoch": 23},
        "rng_state_by_rank": [{}],
        "epoch_start_rng_state_by_rank": [{}],
        "train_accumulator_state_by_rank": [{"counter": 2}],
        "resume_signature": {"sha256": "same"},
    }
    with pytest.raises(RuntimeError, match="accumulator cursor"):
        validate_resume_checkpoint(
            checkpoint,
            steps_per_epoch=10,
            world_size=1,
            expected_resume_signature={"sha256": "same"},
        )


def _signature_rebase_checkpoint():
    return {
        "checkpoint_format_version": 5,
        "checkpoint_id": "precision-parent",
        "lineage_exact": True,
        "lineage_origin": "fp32",
        "checkpoint_world_size": 1,
        "epoch": 2,
        "batch_idx": 0,
        "global_step": 20,
        "steps_per_epoch": 10,
        "state_dict": {},
        "optimizer": {"state": {}},
        "scheduler": {"last_epoch": 20},
        "rng_state_by_rank": [{}],
        "epoch_start_rng_state_by_rank": [{}],
        "train_accumulator_state_by_rank": [None],
        "resume_signature": {
            "sha256": "fp32-signature",
            "config_sha256": "fp32-config",
            "source_sha256": {
                "trainer.py": "old-trainer",
                "model.py": "same-model",
            },
        },
    }


def test_explicit_precision_rebase_accepts_only_audited_signature_changes():
    checkpoint = _signature_rebase_checkpoint()
    metadata = validate_resume_checkpoint(
        checkpoint,
        steps_per_epoch=10,
        world_size=1,
        expected_resume_signature={
            "sha256": "bf16-signature",
            "config_sha256": "bf16-config",
            "source_sha256": {
                "trainer.py": "new-trainer",
                "model.py": "same-model",
            },
        },
        signature_rebase={
            "baseline_config_sha256": "fp32-config",
            "allowed_source_changes": {"trainer.py"},
        },
    )
    assert metadata["lineage_exact"] is True
    assert metadata["warnings"] == [
        "accepted an explicit precision rebase with audited source changes: trainer.py"
    ]


def test_precision_rebase_rejects_unaudited_source_changes():
    checkpoint = _signature_rebase_checkpoint()
    with pytest.raises(RuntimeError, match="unaudited source changes: model.py"):
        validate_resume_checkpoint(
            checkpoint,
            steps_per_epoch=10,
            world_size=1,
            expected_resume_signature={
                "sha256": "bf16-signature",
                "config_sha256": "bf16-config",
                "source_sha256": {
                    "trainer.py": "new-trainer",
                    "model.py": "changed-model",
                },
            },
            signature_rebase={
                "baseline_config_sha256": "fp32-config",
                "allowed_source_changes": {"trainer.py"},
            },
        )


def test_distributed_checkpoint_gathers_rank_local_rng_and_accumulators(tmp_path):
    result_path = tmp_path / "distributed"
    result_path.mkdir()
    init_file = tmp_path / "gloo-init"
    mp.spawn(
        _distributed_save_worker,
        args=(2, str(init_file), str(result_path)),
        nprocs=2,
        join=True,
    )

    checkpoint = torch.load(
        result_path / "last_model.pt", map_location="cpu", weights_only=False
    )
    validate_resume_checkpoint(
        checkpoint,
        steps_per_epoch=4,
        world_size=2,
        expected_resume_signature={"sha256": "test-signature"},
    )
    assert [
        state["rank"] for state in checkpoint["train_accumulator_state_by_rank"]
    ] == [0, 1]
    assert len(checkpoint["rng_state_by_rank"]) == 2
    assert not torch.equal(
        checkpoint["rng_state_by_rank"][0]["torch"],
        checkpoint["rng_state_by_rank"][1]["torch"],
    )


def test_checkpoint_upload_cannot_perturb_training_rng(tmp_path):
    trainer = _trainer_stub(tmp_path)
    trainer.loader_trn = [None] * 2
    trainer.model = torch.nn.Linear(2, 2)
    trainer.model_ema = None
    optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    class RandomConsumingWriter:
        def upload_checkpoint_files(self, last_path, best):
            random.random()
            np.random.rand()
            torch.rand(5)

    trainer.writer = RandomConsumingWriter()
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    before = trainer._capture_rng_state()
    trainer.save_ckpt(optimizer, scheduler, epoch=0, upload=True)
    after = trainer._capture_rng_state()

    assert before["python"] == after["python"]
    assert np.array_equal(before["numpy"][1], after["numpy"][1])
    assert torch.equal(before["torch"], after["torch"])


def test_wandb_checkpoint_upload_uses_waited_versioned_artifact(
    tmp_path, monkeypatch
):
    for name in (
        "last_model.pt",
        "best_rfid_slot1_model.pt",
        "checkpoint_policy.json",
    ):
        (tmp_path / name).write_text(name)
    (tmp_path / "checkpoint_lineage.jsonl").write_text(
        json.dumps({"checkpoint_id": "abc", "epoch": 4}) + "\n"
    )

    created = []

    class FakeArtifact:
        def __init__(self, name, type, metadata):
            self.name = name
            self.type = type
            self.metadata = metadata
            self.files = []
            self.waited = False
            created.append(self)

        def add_file(self, path, **kwargs):
            self.files.append((Path(path).name, kwargs))

        def wait(self):
            self.waited = True
            return self

    class FakeRun:
        def __init__(self):
            self.aliases = None

        def log_artifact(self, artifact, aliases):
            self.aliases = aliases
            return artifact

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Artifact=FakeArtifact))
    monkeypatch.setenv("WANDB_RUN_ID", "trajectory/run")
    writer = object.__new__(writer_module.Writer)
    writer.result_path = str(tmp_path)
    writer.wandb_run = FakeRun()
    writer.upload_checkpoint_files(
        tmp_path / "last_model.pt",
        [{"epoch": 4, "path": "best_rfid_slot1_model.pt"}],
    )

    artifact = created[0]
    assert artifact.name == "trajectory-run-stage1-checkpoints"
    assert artifact.waited
    assert writer.wandb_run.aliases == ["latest", "epoch-0004"]
    assert {name for name, _ in artifact.files} == {
        "last_model.pt",
        "best_rfid_slot1_model.pt",
        "checkpoint_policy.json",
        "checkpoint_lineage.jsonl",
    }
    assert all(options["policy"] == "immutable" for _, options in artifact.files)
