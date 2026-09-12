#!/usr/bin/env python3
"""Launch LASER training from YAML: python train.py --config configs/stage1/ffhq.yaml."""

from src.training.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
