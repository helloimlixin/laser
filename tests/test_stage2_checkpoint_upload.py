import os

from scripts.train_official_rqtransformer_laser_stage2 import (
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
