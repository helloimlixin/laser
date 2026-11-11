"""Freeze the verified epoch78 LR-trial state and evaluate its strong samplers."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
LOCAL=Path('/tmp/laser-rq24576-lrhold-tuned-epoch78-20261110')


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
 path=Path(path);pending=path.with_name(path.name+f'.{os.getpid()}.pending')
 pending.write_text(json.dumps(value,indent=2)+'\n');pending.replace(path)


def main():
 lock=(LOCAL/'launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (LOCAL/'launch-pids.json').exists()
 plan=read(LOCAL/'plan.json');out=Path(plan['output']);base=Path(plan['base']);parent=Path(plan['source_training_output'])
 def status(phase,**values):
  evidence=dict(phase=phase,source_epoch=78,source_step=48828,time=time.time(),queue_pid=os.getpid(),**values)
  for root in [LOCAL,out]:write(root/'queue-status.json',evidence)
 status('waiting_for_verified_epoch78_checkpoint')
 while True:
  index=read(parent/'epoch-checkpoints.json') if (parent/'epoch-checkpoints.json').exists() else {}
  receipt=index.get('48828')
  if receipt:
   assert receipt['epoch']==78 and receipt['verified_bytes']==receipt['bytes']
   break
  progress=read(parent/'status.json')
  if progress['phase'] in ['failed','completed']:
   raise RuntimeError('Training stopped without the required verified epoch78 checkpoint')
  time.sleep(5)
 frozen=Path(plan['checkpoint']);source=Path(receipt['path'])
 if not frozen.exists():
  if source.exists():os.link(source,frozen)
  else:
   sys.path.insert(0,str(base));from runtime.persistent_copy import copy_file
   temporary=frozen.with_suffix('.partial');copy_file(Path(receipt['persistent_path']),temporary);temporary.replace(frozen)
 with frozen.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
 assert actual==receipt['sha256'] and frozen.stat().st_size==receipt['bytes']
 baseline=read(parent/'evaluations/epoch078.json')
 plan.update(checkpoint_sha256=actual,checkpoint_bytes=receipt['bytes'],input_checkpoint_ready=True,
  source_baseline_fid=baseline['eval/fid'],logged_baseline=baseline,
  retained_training_checkpoint=receipt['persistent_path'])
 for root in [LOCAL,out]:write(root/'plan.json',plan)
 write(out/'checkpoint-provenance.json',dict(passed=True,**receipt,frozen_evaluation_path=str(frozen),
  evaluation_checkpoint_sha256=actual,source_training_run=plan['source_run']))
 for script in HERE.glob('*.py'):shutil.copyfile(script,out/script.name)
 with (LOCAL/'coordinator.log').open('ab',buffering=0) as stream:
  child=subprocess.Popen([sys.executable,'-u',str(HERE/'sweep.py')],stdout=stream,stderr=subprocess.STDOUT,
   stdin=subprocess.DEVNULL,start_new_session=True)
 write(LOCAL/'launch-pids.json',dict(coordinator=child.pid))
 status('launched',coordinator_pid=child.pid,checkpoint_sha256=actual)
 print(json.dumps(dict(coordinator_pid=child.pid,source_epoch=78,checkpoint_sha256=actual)),flush=True)


if __name__=='__main__':
 try:main()
 except Exception as error:
  plan=read(LOCAL/'plan.json');evidence=dict(phase='failed',error_type=type(error).__name__,error=str(error))
  for root in [LOCAL,Path(plan['output'])]:write(root/'queue-status.json',evidence)
  raise
