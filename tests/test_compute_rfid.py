import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute_rfid import build_data_args, infer_config_path, infer_model_type


def test_infer_config_path_prefers_matching_run_timestamp(tmp_path: Path):
    ckpt = tmp_path / "outputs" / "checkpoints" / "run_20260408_010203" / "laser" / "last.ckpt"
    good_cfg = ckpt.parents[1] / ".hydra" / "config.yaml"
    stale_cfg = tmp_path / "outputs" / "checkpoints" / "run_20260407_000000" / ".hydra" / "config.yaml"

    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_text("ckpt", encoding="utf-8")
    good_cfg.parent.mkdir(parents=True, exist_ok=True)
    good_cfg.write_text("good", encoding="utf-8")
    stale_cfg.parent.mkdir(parents=True, exist_ok=True)
    stale_cfg.write_text("stale", encoding="utf-8")

    assert infer_config_path(ckpt) == good_cfg.resolve()


def test_infer_model_type_detects_laser_and_vqvae_hparams():
    assert infer_model_type({"sparsity_level": 8}, None, "auto") == "laser"
    assert infer_model_type({"decay": 0.99}, None, "auto") == "vqvae"
    assert infer_model_type({}, "laser", "auto") == "laser"
    assert infer_model_type({}, None, "vqvae") == "vqvae"


def test_build_data_args_uses_defaults_and_cli_overrides():
    args = argparse.Namespace(
        dataset=None,
        data_dir=None,
        image_size=0,
        batch_size=100,
        num_workers=-1,
        mean=None,
        std=None,
    )
    cfg = {
        "seed": 7,
        "data": {
            "dataset": "cifar10",
            "data_dir": "${env:CIFAR10_DIR}",
            "image_size": 32,
            "num_workers": 3,
        },
    }

    data_args = build_data_args(args, cfg)

    assert data_args.dataset == "cifar10"
    assert data_args.data_dir == ""
    assert data_args.image_size == 32
    assert data_args.batch_size == 100
    assert data_args.num_workers == 3
    assert data_args.seed == 7
    assert data_args.mean == (0.4914, 0.4822, 0.4465)
    assert data_args.std == (0.2470, 0.2435, 0.2616)

    args.mean = [0.1, 0.2, 0.3]
    args.std = [0.9, 0.8, 0.7]
    args.data_dir = "/tmp/data"
    data_args = build_data_args(args, cfg)

    assert data_args.data_dir == "/tmp/data"
    assert data_args.mean == (0.1, 0.2, 0.3)
    assert data_args.std == (0.9, 0.8, 0.7)
