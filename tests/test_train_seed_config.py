from pathlib import Path

from hydra import compose, initialize_config_dir


def _compose(*overrides: str):
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
        return compose(config_name="config", overrides=list(overrides))


def test_data_configs_inherit_root_seed():
    for dataset in ("celeba", "cifar10", "imagenette2"):
        cfg = _compose(f"data={dataset}", "seed=123")
        assert cfg.seed == 123
        assert cfg.data.seed == 123


def test_training_is_deterministic_by_default():
    cfg = _compose("data=cifar10")
    assert cfg.train.deterministic is True
