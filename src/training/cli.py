"""Config composition and stage dispatch, without importing training dependencies."""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
import sys

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from src.training.paths import CONFIG_DIR


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Compose a recipe and apply CLI overrides last; reject unknown config keys."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Config file not found: {path}")
    config_dir = CONFIG_DIR if path.is_relative_to(CONFIG_DIR) else path.parent
    config_name = path.relative_to(config_dir).with_suffix("").as_posix()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.2"):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    if cfg.get("stage") not in {"stage1", "stage2"}:
        raise ValueError("Config stage must be stage1 or stage2.")
    if cfg.get("backend", "lightning") not in {"lightning", "rqtransformer"}:
        raise ValueError("Config backend must be lightning or rqtransformer.")
    if cfg.get("backend") == "rqtransformer" and cfg.stage != "stage2":
        raise ValueError("The rqtransformer backend requires stage2.")
    return cfg


def run(cfg: DictConfig) -> None:
    """Import only the requested stage, after configuration has been validated."""
    backend = cfg.get("backend", "lightning")
    module = "rqtransformer" if backend == "rqtransformer" else cfg.stage
    import_module(f"src.training.{module}").run(cfg)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Train LASER using a stage/dataset YAML recipe.",
        epilog="Example: python train.py --config configs/stage1/ffhq.yaml train.devices=2",
    )
    parser.add_argument("--config", help="YAML recipe, e.g. configs/stage2/ffhq.yaml")
    parser.add_argument("--dry-run", "--dry_run", action="store_true", help="Print the resolved config and exit")
    args, overrides = parser.parse_known_args(argv)

    # Old flag-based, raw Hydra, and pipeline commands remain isolated here.
    # New recipes use Hydra defaults; archived launcher YAMLs use stage/overrides.
    if not args.config and argv:
        from src.training.legacy_cli import main as legacy_main
        return legacy_main(argv)
    if not args.config:
        parser.error("--config is required")
    try:
        raw = OmegaConf.load(Path(args.config).expanduser())
        if isinstance(raw, DictConfig) and "defaults" not in raw:
            from src.training.legacy_cli import main as legacy_main
            return legacy_main(argv)
        cfg = load_config(args.config, overrides)
        # Resolve before importing torch or creating output directories.
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    except Exception as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return 0
    run(cfg)
    return 0
