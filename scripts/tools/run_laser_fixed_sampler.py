"""Complete the fixed-weight sampler diagnostic before resuming training."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path('/workspace/Projects/laser')
BASE=Path('/tmp/laser-fixed-sampler-20261015')
OUT=ROOT/'outputs/imagenet-laser-fixed-sampler-20261015'
RECOVERY=ROOT/'outputs/imagenet-laser-historical-recovery-20261015/controller.json'
child=None
cancelled=False


def record(**data):
    p=OUT/'controller.json';t=p.with_suffix('.tmp')
    t.write_text(json.dumps(dict(data,pid=os.getpid(),time=time.time()),indent=2)+'\n');t.replace(p)


def stop(*_):
    global cancelled
    cancelled=True
    if child is not None and child.poll() is None:child.send_signal(signal.SIGINT)


def main():
    global child
    lock=(BASE/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    for name,expected in json.loads((OUT/'source-manifest.json').read_text()).items():
        assert hashlib.sha256((BASE/name).read_bytes()).hexdigest()==expected,name
    record(phase='waiting_for_historical_recovery')
    while json.loads(RECOVERY.read_text())['phase']!='complete':
        if cancelled:return
        time.sleep(4)
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        if cancelled:return
        time.sleep(4)
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=4',str(BASE/'evaluate.py'),'--sampler','historical']
    with (BASE/'historical.log').open('a') as log:
        child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
        record(phase='fid50k',child_pid=child.pid)
        code=child.wait()
    if code:
        record(phase='failed',returncode=code);raise RuntimeError('Sampler evaluation failed')
    result=json.loads((OUT/'historical/fid50k.json').read_text())
    record(phase='complete',fid=result['fid'],training_updates_added=0,
           matched_fid=json.loads((OUT/'plan.json').read_text())['matched_fid'])


if __name__=='__main__':main()
