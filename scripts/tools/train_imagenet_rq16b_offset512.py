"""Entry point shared by the canary and production jobs."""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BASE), str(BASE/'runtime'), str(BASE/'scripts/tools')]

if __name__ == '__main__':
    from imagenet_rq16b_offset512_runtime import install, training, HeldoutValidation
    install()
    training.main(epoch_validation_callback=HeldoutValidation())
