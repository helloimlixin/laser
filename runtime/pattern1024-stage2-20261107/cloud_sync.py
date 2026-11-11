"""Use the existing verified binary-part checkpoint uploader for this run."""
from pathlib import Path
import sys

ROOT=Path('/tmp/laser-pattern1024-stage2-20261107')
sys.path.insert(0,str(ROOT/'frozen/runtime'))
import cloud_worker
cloud_worker.OUT=ROOT/'production-nvme'

if __name__=='__main__':cloud_worker.main()
