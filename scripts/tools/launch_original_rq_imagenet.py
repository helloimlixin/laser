#!/usr/bin/env python3
"""Supervise the original 1.4B ImageNet baseline with a 20-epoch stop."""
import fcntl
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import shutil

ROOT=Path('/workspace/Projects/laser')
sys.path.insert(0,str(ROOT))
from runtime.rq_imagenet_entry import freeze_shared_runtime

BASE=Path('/tmp/laser-original-rq-imagenet-20261014')
OUT=Path('/workspace/Projects/laser/outputs/imagenet-original-rq1400m-rfid473-scratch20-4h100-20261014')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    lock=(BASE/'launch.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((BASE/'data/ready.json').read_text())['ready']
    assert json.loads((OUT/'smoke/verification.json').read_text())['passed']
    assert json.loads((OUT/'smoke-reload-verification.json').read_text())['passed']
    assert json.loads((OUT/'stage1-rfid-verification.json').read_text())['passed']
    checkpoint=OUT/'checkpoints/last.pt'
    if not checkpoint.is_file():checkpoint=OUT/'train/checkpoints/last.pt'
    if args.resume:
        assert checkpoint.exists()
        assert json.loads((OUT/'recovery-checkpoint-audit.json').read_text())['passed']
    else:assert not checkpoint.exists()
    # Runs already in progress keep their imported frozen snapshot. Every
    # subsequent launch uses the same shared source as the LASER entry point.
    freeze_shared_runtime(BASE,ROOT)
    shutil.copyfile(ROOT/'scripts/tools/train_original_rq_imagenet.py',BASE/'train.py')
    env=os.environ.copy();env.update(json.loads((BASE/'environment.json').read_text()))
    env.update(OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',OMP_NUM_THREADS='4')
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=4',str(BASE/'train.py'),'--epochs','20']
    if args.resume:command+=['--resume',str(checkpoint)]
    child=None
    def stop(signum,_frame):
        if child is not None and child.poll() is None:child.send_signal(signal.SIGINT)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with (OUT/'train.log').open('a') as log:
        child=subprocess.Popen(command,cwd=BASE,env=env,stdout=log,stderr=subprocess.STDOUT)
        status=dict(phase='running',supervisor_pid=os.getpid(),torchrun_pid=child.pid,
                    command=command,started_at=time.time())
        (OUT/'launch-status.json').write_text(json.dumps(status,indent=2)+'\n')
        print(json.dumps(status),flush=True)
        result=child.wait()
    status.update(phase='completed' if result==0 else 'stopped_or_failed',returncode=result,finished_at=time.time())
    (OUT/'launch-status.json').write_text(json.dumps(status,indent=2)+'\n')
    return result


if __name__=='__main__':sys.exit(main())
