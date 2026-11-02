#!/usr/bin/env python3
"""Entry point for the shared upstream RQ ImageNet driver."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from runtime.rq_imagenet_entry import launch

if __name__ == '__main__':
    launch('laser')
