"""Validate the matched error requirement, restore state, then replay58to60."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PLAN=json.loads((HERE/'plan.json').read_text());BASE=Path(PLAN['base']);OUT=Path(PLAN['output'])
COMPARISON=ROOT/'outputs/imagenet-rq24576-vs-original-latenterror-val50k-20261110/report.json'


def status(phase,**fields):
    target=OUT/'launch-pipeline.json';pending=target.with_suffix('.pending')
    pending.write_text(json.dumps(dict(phase=phase,restore_epoch=58,noise_change_epoch=60,
        teacher_temperature_after=.1,target_epoch=100,time=time.time(),**fields),indent=2)+'\n');pending.replace(target)


def resume_search_on_error():
    p=OUT/'sampler-pause.json'
    if not p.exists():return
    state=json.loads(p.read_text())
    if state['resumed']:return
    for pid in state['paused_workers']:
        cmd=Path(f'/proc/{pid}/cmdline')
        if cmd.exists() and 'rq24576-ema-sampler-search-20261110/worker.py' in cmd.read_bytes().replace(b'\0',b' ').decode():
            os.kill(pid,signal.SIGCONT)
    state.update(resumed=True,resume_reason='launch_failed',resumed_at=time.time());p.write_text(json.dumps(state,indent=2)+'\n')


def main():
    for script in ['configure.py','check_refinement.py']:
        status(script);subprocess.run([sys.executable,'-u',str(HERE/script)],check=True)
    status('waiting_for_full50k_matched_comparison')
    deadline=time.monotonic()+1800
    while not COMPARISON.exists():
        assert time.monotonic()<deadline,'Full50k matched comparison did not finish'
        time.sleep(5)
    report=json.loads(COMPARISON.read_text())
    assert report['passed'] and report['images']==50000
    assert report['original_greedy_rfid_reproduced'] and report['laser_greedy_rfid_reproduced']
    assert all(report['teacher010_comparison'].values()),'Requested temperature failed matched reconstruction checks'
    shutil.copyfile(COMPARISON,BASE/'inputs/teacher010-matched-50k-validation.json')
    status('validating_resume_and_teacher010')
    subprocess.run([sys.executable,'-u',str(HERE/'validate.py')],check=True)
    with (OUT/'supervisor.log').open('ab',buffering=0) as log:
        child=subprocess.Popen([sys.executable,'-u',str(HERE/'supervise.py')],stdout=log,
            stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    status('production_supervisor_launched',supervisor_pid=child.pid)
    print(json.dumps(dict(supervisor_pid=child.pid,run_id=PLAN['run_id'])),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        resume_search_on_error();status('failed',error_type=type(error).__name__,error=str(error));raise
