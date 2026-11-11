"""Keep cloud failures outside the optimizer process and retry pending saves."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from retention import cleanup

OUT=Path('/tmp/laser-aligned-beam451-20261105/production-nvme')
CHILD=None


def stop(sig,_):
    if CHILD is not None and CHILD.poll() is None:os.killpg(CHILD.pid,signal.SIGTERM)
    raise SystemExit(128+sig)


def main():
    global CHILD
    lock=(OUT/'cloud-worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,stop)
    while True:
        cleanup(OUT)
        pending=[p for p in sorted((OUT/'cloud-queue').glob('*.json')) if not (OUT/'cloud-receipts'/p.name).exists()]
        if not pending:time.sleep(10);continue
        job=pending[0]
        with (OUT/'cloud-upload.log').open('ab',buffering=0) as log:
            CHILD=subprocess.Popen([sys.executable,'-u',str(Path(__file__).with_name('upload_once.py')),str(job)],
                stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            try:code=CHILD.wait(timeout=2100)
            except subprocess.TimeoutExpired:
                os.killpg(CHILD.pid,signal.SIGTERM)
                try:CHILD.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(CHILD.pid,signal.SIGKILL);CHILD.wait()
                code=-9
        if code:
            status=dict(state='RETRYING',job=job.name,returncode=code,time=time.time(),
                        training_interrupted=False,local_checkpoint_preserved=True)
            temp=OUT/'cloud-sync-status.pending';temp.write_text(json.dumps(status,indent=2)+'\n');temp.replace(OUT/'cloud-sync-status.json')
            time.sleep(30)


if __name__=='__main__':main()
