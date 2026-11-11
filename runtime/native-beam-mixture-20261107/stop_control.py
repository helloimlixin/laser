"""Preserve the existing control at its next complete checkpoint."""
import json
import os
from pathlib import Path
import signal
import time

CONTROL=Path('/tmp/laser-aligned-beam451-20261105/production-nvme')
ROOT=Path('/tmp/laser-native-beam-mixture-20261107')


def main():
    ROOT.mkdir(exist_ok=True)
    while True:
        saved=json.loads((CONTROL/'checkpoint.json').read_text())
        if saved['global_step']>=17250 and saved['local_state']=='COMPLETE':
            status=json.loads((CONTROL/'status.json').read_text())
            pid=status['supervisor_pid']
            assert b'aligned-beam451-20261105/supervise.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
            progress=json.loads((CONTROL/'train/progress-rank0.json').read_text())
            os.kill(pid,signal.SIGTERM)
            receipt=dict(checkpoint=saved,progress=progress,original_run=status['run_id'],
                reason='User requested native-soft/beam mixture fix',time=time.time())
            (ROOT/'preserved-control.json').write_text(json.dumps(receipt,indent=2)+'\n')
            print(json.dumps(receipt),flush=True);return
        time.sleep(.25)


if __name__=='__main__':main()
