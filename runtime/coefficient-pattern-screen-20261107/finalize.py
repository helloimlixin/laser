"""Publish the completed screen without changing the ongoing AR experiment."""
from common import ROOT, write
import json
import subprocess
import sys
import time


def main():
    started = time.time()
    while not (ROOT/'results.json').exists():
        progress = json.loads((ROOT/'progress.json').read_text()) if (ROOT/'progress.json').exists() else None
        write(ROOT/'status.json', dict(phase='full_validation', progress=progress, time=time.time()))
        if time.time()-started>1800:
            raise RuntimeError('Full validation did not finish within thirty minutes')
        time.sleep(10)
    with (ROOT/'logs/publish.log').open('ab', buffering=0) as log:
        subprocess.run([sys.executable, '-u', str(ROOT/'frozen-code/publish.py')],
            check=True, stdout=log, stderr=subprocess.STDOUT, timeout=300)
    receipt = json.loads((ROOT/'cloud-verification.json').read_text())
    write(ROOT/'status.json', dict(phase='completed', cloud=receipt, time=time.time(),
        baseline_training_unchanged=True, new_ar_training_launched=False))


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        write(ROOT/'failure.json', dict(error=str(error), time=time.time()))
        raise
