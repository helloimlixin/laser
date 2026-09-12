#!/usr/bin/env python3
"""Compatibility bridge for the pending launcher after the user canceled integers.

The original, unlaunched trainer is preserved under
outputs/church-interleaved-pattern-20260911/abandoned-source/scripts/.
No interleaved-pattern or complete-site-integer training starts here.
"""
import argparse
from pathlib import Path
from launch_church_looped_pair import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wandb-id', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    assert args.output.resolve() == root / 'outputs/church-interleaved-pattern-20260911/train'
    assert args.wandb_id == 'church-interleaved-pattern-20260911'
    main()
