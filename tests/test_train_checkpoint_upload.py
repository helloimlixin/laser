import os
from types import SimpleNamespace

import train


class _Experiment:
    def __init__(self):
        self.saved = []

    def save(self, path, *, base_path, policy):
        self.saved.append((path, base_path, policy))


def test_fixed_file_checkpoint_upload_uses_ranked_hard_link_slots(tmp_path):
    best_a = tmp_path / "laser-epoch=000.ckpt"
    best_b = tmp_path / "laser-epoch=001.ckpt"
    best_c = tmp_path / "laser-epoch=002.ckpt"
    last = tmp_path / "last-source.ckpt"
    for path, payload in (
        (best_a, b"a"),
        (best_b, b"b"),
        (best_c, b"c"),
        (last, b"last"),
    ):
        path.write_bytes(payload)

    checkpoint = SimpleNamespace(
        best_k_models={str(best_a): 0.3, str(best_b): 0.1, str(best_c): 0.2},
        best_model_path=str(best_b),
        last_model_path=str(last),
        mode="min",
        save_top_k=3,
    )
    experiment = _Experiment()
    trainer = SimpleNamespace(
        is_global_zero=True,
        sanity_checking=False,
        current_epoch=0,
        logger=SimpleNamespace(experiment=experiment),
    )
    upload_dir = tmp_path / "upload"
    Callback = train._make_selected_checkpoint_file_callback(object)
    callback = Callback(checkpoint, upload_dir=upload_dir, every_n_epochs=1)

    callback.on_validation_end(trainer, None)

    expected_sources = [best_b, best_c, best_a, last]
    expected_slots = [
        upload_dir / "best-01.ckpt",
        upload_dir / "best-02.ckpt",
        upload_dir / "best-03.ckpt",
        upload_dir / "last.ckpt",
    ]
    assert [path.read_bytes() for path in expected_slots] == [
        path.read_bytes() for path in expected_sources
    ]
    assert all(
        os.stat(slot).st_ino == os.stat(source).st_ino
        for slot, source in zip(expected_slots, expected_sources)
    )
    assert [(os.path.basename(path), policy) for path, _, policy in experiment.saved] == [
        ("best-01.ckpt", "now"),
        ("best-02.ckpt", "now"),
        ("best-03.ckpt", "now"),
        ("last.ckpt", "now"),
    ]

    callback.on_train_end(trainer, None)
    assert len(experiment.saved) == 4
