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


def test_selected_checkpoint_paths_orders_maximized_scores(tmp_path):
    paths = [tmp_path / f"laser-{index}.ckpt" for index in range(3)]
    for index, path in enumerate(paths):
        path.write_bytes(str(index).encode())
    last = tmp_path / "last.ckpt"
    last.write_bytes(b"last")
    checkpoint = SimpleNamespace(
        best_k_models={str(paths[0]): 3.9, str(paths[1]): 4.4, str(paths[2]): 4.1},
        best_model_path=str(paths[1]),
        last_model_path=str(last),
        mode="max",
    )

    selected = train._selected_checkpoint_paths(checkpoint)

    assert selected == [
        paths[1].resolve(),
        paths[2].resolve(),
        paths[0].resolve(),
        last.resolve(),
    ]


def test_refresh_last_checkpoint_uses_current_trainer_state(tmp_path):
    saved = []

    def save_checkpoint(path):
        saved.append(path)
        with open(path, "wb") as handle:
            handle.write(b"current")

    trainer = SimpleNamespace(save_checkpoint=save_checkpoint)
    checkpoint = SimpleNamespace(
        save_last=True,
        dirpath=str(tmp_path),
        last_model_path="",
    )

    train._refresh_last_checkpoint(trainer, checkpoint)

    expected = tmp_path / "last.ckpt"
    assert saved == [str(expected)]
    assert expected.read_bytes() == b"current"
    assert checkpoint.last_model_path == str(expected)


def test_artifact_upload_waits_until_checkpoint_callback_has_selected_top_k():
    checkpoint = SimpleNamespace()
    Callback = train._make_selected_checkpoint_artifact_callback(object)
    callback = Callback(checkpoint, every_n_epochs=1)
    uploads = []
    callback._upload = lambda trainer, reason: uploads.append(
        (trainer.current_epoch, reason)
    )

    callback.on_train_epoch_start(SimpleNamespace(current_epoch=0), None)
    assert uploads == []

    callback.on_train_epoch_start(SimpleNamespace(current_epoch=1), None)
    assert uploads == [(1, "post_checkpoint_validation")]


def test_artifact_fork_at_upload_boundary_requires_restartable_last(tmp_path):
    parent_best = tmp_path / 'parent.ckpt'
    parent_best.write_bytes(b'parent')
    checkpoint = SimpleNamespace(save_last=True, last_model_path='',
                                 best_model_path=str(parent_best), best_k_models={})
    experiment = SimpleNamespace(log_artifact=lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError('Must not publish a parent checkpoint as the new latest')))
    trainer = SimpleNamespace(current_epoch=255, is_global_zero=True,
                              logger=SimpleNamespace(experiment=experiment))
    callback = train._make_selected_checkpoint_artifact_callback(object)(checkpoint, every_n_epochs=5)
    callback.on_train_epoch_start(trainer, None)
