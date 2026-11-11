"""Yield the GPUs only after epoch4 metrics and a complete resumable checkpoint."""
import json
import os
from pathlib import Path
import signal
import time

RUN='imnet-j16384-a4096-beam16-aligned-soft-scratch2-4a100-20261105'
OUT=Path('/tmp/laser-aligned-beam451-20261105/production-nvme')
SPEED=Path('/tmp/laser-speed-aligned-20261105')


def main():
    SPEED.mkdir(exist_ok=True)
    while True:
        checkpoint=OUT/'checkpoint.json';evaluation=OUT/'evaluations/epoch004.json'
        if checkpoint.exists() and evaluation.exists():
            state=json.loads(checkpoint.read_text())
            if state['global_step']>=2504 and state['local_state']=='COMPLETE':
                status=json.loads((OUT/'status.json').read_text())
                assert status['run_id']==RUN
                pid=status['supervisor_pid']
                cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
                assert b'aligned-beam451-20261105/supervise.py' in cmd
                os.kill(pid,signal.SIGTERM)
                proof=dict(stopped_at_checkpoint=state,epoch4_evaluation=json.loads(evaluation.read_text()),time=time.time())
                (SPEED/'stopped.json').write_text(json.dumps(proof,indent=2)+'\n')
                print(json.dumps(proof),flush=True);return
        time.sleep(.25)


if __name__=='__main__':main()
