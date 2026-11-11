"""Reuse the verified checkpoint uploader with this experiment's output directory."""
from pathlib import Path
import sys

ROOT = Path('/tmp/laser-compound128-20261107')
BASE = ROOT / 'frozen'
sys.path.insert(0, str(BASE / 'runtime'))
import cloud_worker

cloud_worker.OUT = ROOT / 'production-nvme'

if __name__ == '__main__':
    cloud_worker.main()
