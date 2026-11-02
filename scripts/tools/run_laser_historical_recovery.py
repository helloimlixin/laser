"""Queue the two historical-checkpoint evaluations after the matched ablation."""
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
BASE=Path('/tmp/laser-historical-recovery-20261015')
OUT=ROOT/'outputs/imagenet-laser-historical-recovery-20261015'
TRIALS=ROOT/'outputs/imagenet-laser-pair-ablation-20261015'
child=None
cancelled=False


def record(**data):
    target=OUT/'controller.json'
    temporary=target.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(data,pid=os.getpid(),time=time.time()),indent=2)+'\n')
    temporary.replace(target)


def stop(*_):
    global cancelled
    cancelled=True
    if child is not None and child.poll() is None:child.send_signal(signal.SIGINT)


def main():
    global child
    lock=(BASE/'controller.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGINT,stop)
    signal.signal(signal.SIGTERM,stop)
    assert json.loads((OUT/'metadata-verification.json').read_text())['passed']
    for name,expected in json.loads((OUT/'source-manifest.json').read_text()).items():
        assert hashlib.sha256((BASE/name).read_bytes()).hexdigest()==expected,name
    record(phase='waiting_for_matched_ablation')
    while not (TRIALS/'comparison.json').exists():
        if cancelled:return
        status=json.loads((TRIALS/'controller.json').read_text())
        if status['phase']=='failed':raise RuntimeError('Matched ablation failed')
        time.sleep(4)
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        if cancelled:return
        time.sleep(4)
    for sampler in ['matched','historical']:
        if cancelled:return
        if (OUT/sampler/'fid50k.json').exists():continue
        command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=4',
                 str(BASE/'evaluate.py'),'--sampler',sampler]
        with (BASE/(sampler+'.log')).open('a') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
            record(phase='fid50k',sampler=sampler,child_pid=child.pid)
            code=child.wait()
        if code:
            record(phase='failed',sampler=sampler,returncode=code)
            raise RuntimeError(sampler)
    record(phase='complete',fid={s:json.loads((OUT/s/'fid50k.json').read_text())['fid']
                               for s in ['matched','historical']},training_updates_added=0)


if __name__=='__main__':main()
