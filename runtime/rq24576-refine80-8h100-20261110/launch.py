"""Install and validate the earlier-source replacement before production."""
import json
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
PLAN=json.loads((HERE/'plan.json').read_text());OUT=Path(PLAN['output'])


def status(phase,**fields):
    path=OUT/'launch-pipeline.json';pending=path.with_suffix('.pending')
    pending.write_text(json.dumps(dict(phase=phase,source_epoch=80,target_epoch=110,time=time.time(),**fields),indent=2)+'\n')
    pending.replace(path)


def main():
    for name in ['install.py','check_refinement.py','validate.py']:
        status(name)
        subprocess.run([sys.executable,'-u',str(HERE/name)],check=True)
    with (OUT/'supervisor.log').open('ab',buffering=0) as log:
        child=subprocess.Popen([sys.executable,'-u',str(HERE/'supervise.py')],stdout=log,
            stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    status('production_supervisor_launched',supervisor_pid=child.pid)
    print(json.dumps(dict(supervisor_pid=child.pid,run_id=PLAN['run_id'],source_epoch=80)),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        status('failed',error=str(error),error_type=type(error).__name__);raise
