#!/usr/bin/env python3
"""Train or resume the frozen-codec MDCTCodec-LASER text-to-speech prior."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.training.mdctcodec_tts import main

if __name__=='__main__':main()
