import os
from pathlib import Path

import torch

from src.training.rqtransformer import (
    upload_selected_checkpoint_files,
)


class _Run:
    def __init__(self):
        self.saved = []

    def save(self, path, *, base_path, policy):
        self.saved.append((path, base_path, policy))


def test_stage2_fixed_file_upload_replaces_last_and_ranked_best_fid_slots(tmp_path):
    last = tmp_path / "checkpoints" / "last.pt"
    best_a = tmp_path / "checkpoints" / "best_fid_12.0_epoch_005.pt"
    best_b = tmp_path / "checkpoints" / "best_fid_10.0_epoch_010.pt"
    best_c = tmp_path / "checkpoints" / "best_fid_11.0_epoch_015.pt"
    best_d = tmp_path / "checkpoints" / "best_fid_13.0_epoch_020.pt"
    for path, payload in (
        (last, b"last"),
        (best_a, b"a"),
        (best_b, b"b"),
        (best_c, b"c"),
        (best_d, b"d"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    run = _Run()
    upload_dir = tmp_path / "uploads"
    uploaded = upload_selected_checkpoint_files(
        run,
        last_checkpoint=last,
        best_fid=[(12.0, str(best_a)), (10.0, str(best_b)),
                  (11.0, str(best_c)), (13.0, str(best_d))],
        upload_dir=upload_dir,
    )

    slots = [
        upload_dir / "last.pt",
        upload_dir / "best-fid-01.pt",
        upload_dir / "best-fid-02.pt",
        upload_dir / "best-fid-03.pt",
    ]
    assert uploaded == slots
    assert [path.read_bytes() for path in slots] == [b"last", b"b", b"c", b"a"]
    assert all(
        os.stat(slot).st_ino == os.stat(source).st_ino
        for slot, source in zip(slots, [last, best_b, best_c, best_a])
    )
    assert [(os.path.basename(path), policy) for path, _, policy in run.saved] == [
        ("last.pt", "now"),
        ("best-fid-01.pt", "now"),
        ("best-fid-02.pt", "now"),
        ("best-fid-03.pt", "now"),
    ]

    replacement = tmp_path / "checkpoints" / "replacement.pt"
    replacement.write_bytes(b"replacement")
    upload_selected_checkpoint_files(
        run,
        last_checkpoint=replacement,
        best_fid=[(10.0, str(best_b))],
        upload_dir=upload_dir,
    )
    assert (upload_dir / "last.pt").read_bytes() == b"replacement"
    assert (upload_dir / "best-fid-01.pt").read_bytes() == b"b"
    assert not (upload_dir / "best-fid-02.pt").exists()
    assert not (upload_dir / "best-fid-03.pt").exists()


def test_fixed_upload_includes_highest_inception_and_replaces_slots(tmp_path):
    last, fid, is_low, is_high = [tmp_path / name for name in (
        "last.pt", "fid.pt", "is_low.pt", "is_high.pt"
    )]
    for path in (last, fid, is_low, is_high):
        path.write_bytes(path.name.encode())
    run = _Run()
    upload_dir = tmp_path / "uploads"
    uploaded = upload_selected_checkpoint_files(
        run, last_checkpoint=last, best_fid=[(10.0, str(fid))],
        best_inception=[(40.0, str(is_low)), (80.0, str(is_high))],
        upload_dir=upload_dir,
    )
    assert [p.name for p in uploaded] == [
        "last.pt", "best-fid-01.pt", "best-is-01.pt", "best-is-02.pt"
    ]
    assert (upload_dir / "best-is-01.pt").read_bytes() == b"is_high.pt"
    assert (upload_dir / "best-is-02.pt").read_bytes() == b"is_low.pt"
    assert all(policy == "now" for _, _, policy in run.saved)
    upload_selected_checkpoint_files(
        run, last_checkpoint=last, best_fid=[(10.0, str(fid))],
        best_inception=[(90.0, str(is_low))], upload_dir=upload_dir,
    )
    assert (upload_dir / "best-is-01.pt").read_bytes() == b"is_low.pt"
    assert not (upload_dir / "best-is-02.pt").exists()


def test_local_upload_cache_keeps_immutable_snapshots_and_rejects_stale_files(tmp_path, monkeypatch):
    from src.training import rqtransformer as training

    monkeypatch.setenv("LASER_CHECKPOINT_STAGING_DIR", str(tmp_path / "staging"))
    monkeypatch.setenv("LASER_CHECKPOINT_UPLOAD_CACHE_DIR", str(tmp_path / "local"))
    target = tmp_path / "persistent" / "last.pt"
    training.atomic_torch_save({"step": 1, "weights": torch.arange(7)}, target)
    local = training._checkpoint_upload_source(target)
    assert local != target
    assert local.read_bytes() == target.read_bytes()

    run = _Run()
    slot, = training.upload_selected_checkpoint_files(
        run, last_checkpoint=target, best_fid=[], upload_dir=tmp_path / "uploads"
    )
    assert slot.is_relative_to(tmp_path / "local")
    assert slot.stat().st_ino == local.stat().st_ino
    # Saving a later checkpoint must not mutate an inode already being uploaded.
    training.atomic_torch_save({"step": 2, "weights": torch.arange(7) + 1}, target)
    assert torch.load(slot, weights_only=True)["step"] == 1
    assert torch.load(local, weights_only=True)["step"] == 2
    assert training._checkpoint_upload_source(target) == local

    # An external replacement invalidates the local copy even at equal size.
    replacement = target.with_name("replacement.pt")
    replacement.write_bytes(b"x" * target.stat().st_size)
    replacement.replace(target)
    assert training._checkpoint_upload_source(target) == target
    assert Path(run.saved[0][1]) == slot.parent
