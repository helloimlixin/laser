from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from src.training import cli
from src.training.paths import ROOT


DATASETS = ["celebahq", "ffhq", "lsun-church", "lsun-bedroom", "lsun-cat", "imagenet", "cc3m"]


@pytest.mark.parametrize("dataset", DATASETS)
def test_recipes_connect_both_stages_and_conditioning(dataset, monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    first = cli.load_config(ROOT / f"configs/stage1/{dataset}.yaml")
    second = cli.load_config(ROOT / f"configs/stage2/{dataset}.yaml")
    for cfg in (first, second):
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        assert cfg.data.dataset == dataset.replace("-", "_")
        assert cfg.data.image_size == 256
        assert cfg.data.seed == cfg.seed
    assert first.data.data_dir == second.data.data_dir
    assert second.token_cache.stage1_output_root == first.output_dir
    assert second.token_cache.build is True
    assert second.token_cache.output == second.token_cache_path
    assert second.ar.class_conditional == (dataset == "imagenet")
    assert second.ar.text_conditional == (dataset == "cc3m")
    if dataset == "imagenet":
        assert second.ar.num_classes == 1000
    if dataset == "cc3m":
        assert second.ar.text_conditioning_mode == "rq_prefix"
        assert second.token_cache.text_max_length > 0


def test_cli_overrides_win_and_interpolations_follow(tmp_path):
    cfg = cli.load_config(ROOT / "configs/stage2/ffhq.yaml", [
        "seed=123", "train_ar.devices=2", "train_ar.batch_size=4",
        f"data.data_dir={tmp_path}", f"output_dir={tmp_path}/output",
    ])
    assert cfg.data.seed == 123
    assert cfg.train_ar.devices == 2
    assert cfg.train_ar.batch_size == 4
    assert cfg.data.data_dir == str(tmp_path)
    assert cfg.token_cache_path == f"{tmp_path}/output/token_cache/train.pt"
    assert cfg.wandb.save_dir == f"{tmp_path}/output/wandb"


def test_dry_run_never_imports_training_or_creates_outputs(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli, "import_module", lambda _: pytest.fail("dry run imported training"))
    assert cli.main(["--config", str(ROOT / "configs/stage1/lsun-cat.yaml"),
                     "--dry-run", f"output_dir={tmp_path}/unused"]) == 0
    assert "stage: stage1" in capsys.readouterr().out
    assert not (tmp_path / "unused").exists()


def test_cli_rejects_misspelled_settings_before_training(monkeypatch):
    monkeypatch.setattr(cli, "run", lambda _: pytest.fail("invalid config dispatched"))
    with pytest.raises(SystemExit) as error:
        cli.main(["--config", str(ROOT / "configs/stage1/ffhq.yaml"), "train.devicess=2"])
    assert error.value.code == 2


@pytest.mark.parametrize("stage", ["stage1", "stage2"])
def test_cli_dispatches_only_the_selected_stage(monkeypatch, stage):
    calls = []
    def import_stage(name):
        calls.append(name)
        return SimpleNamespace(run=lambda cfg: calls.append(cfg.stage))
    monkeypatch.setattr(cli, "import_module", import_stage)
    assert cli.main(["--config", str(ROOT / f"configs/{stage}/ffhq.yaml")]) == 0
    assert calls == [f"src.training.{stage}", stage]


def test_cache_bootstrap_passes_lsun_dataset_checkpoint_and_seed(monkeypatch, tmp_path):
    from src.training import stage2
    from scripts.tools.build_token_cache import IMAGE_TOKEN_CACHE_DATASETS

    checkpoint = tmp_path / "stage1.ckpt"
    checkpoint.touch()
    cfg = cli.load_config(ROOT / "configs/stage2/lsun-cat.yaml", [
        f"output_dir={tmp_path}/stage2", f"token_cache.stage1_checkpoint={checkpoint}", "seed=123",
    ])
    calls = []
    monkeypatch.setattr(stage2.subprocess, "run", lambda cmd, **kwargs: calls.append((cmd, kwargs)))
    stage2._maybe_build_token_cache(cfg)
    cmd, kwargs = calls[0]
    assert Path(cmd[1]).is_file()
    assert cmd[cmd.index("--dataset") + 1] == "lsun_cat"
    assert "lsun_cat" in IMAGE_TOKEN_CACHE_DATASETS
    assert cmd[cmd.index("--stage1_checkpoint") + 1] == str(checkpoint)
    assert cmd[cmd.index("--seed") + 1] == "123"
    assert kwargs["check"] is True

    cache = Path(cfg.token_cache_path)
    cache.parent.mkdir(parents=True)
    cache.touch()
    stage2._maybe_build_token_cache(cfg)
    assert len(calls) == 1


def test_compound_recipe_dispatch_and_option_conversion(monkeypatch, tmp_path):
    from src.training.options import options_to_argv
    from src.training.rqtransformer import build_parser

    cfg = cli.load_config(ROOT / "configs/stage2/ffhq-compound.yaml", [
        f"options.checkpoint={tmp_path}/upstream.pt", f"options.token_cache={tmp_path}/cache.pt",
    ])
    calls = []
    monkeypatch.setattr(cli, "import_module", lambda name: SimpleNamespace(run=lambda _: calls.append(name)))
    cli.run(cfg)
    assert calls == ["src.training.rqtransformer"]
    argv = options_to_argv(OmegaConf.to_container(cfg.options, resolve=True))
    assert "--compound-tokens" in argv
    assert "--no-resume" in argv
    assert argv[argv.index("--checkpoint") + 1] == str(tmp_path / "upstream.pt")
    options = build_parser().parse_args(argv)
    assert options.dataset == "ffhq"
    assert options.compound_tokens is True
    assert options.resume is False
