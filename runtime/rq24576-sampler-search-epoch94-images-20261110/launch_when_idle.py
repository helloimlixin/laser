"""Queue epoch94 inference after the existing epoch88 study releases GPU memory."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
LOCAL=Path('/tmp/laser-rq24576-sampler-search-epoch94-images-20261110')
PREVIOUS=Path('/tmp/laser-rq24576-sampler-search-epoch88-images-20261110')


def read(p):return json.loads(Path(p).read_text())


def persist(value):
 plan=read(LOCAL/'plan.json')
 value=dict(source_epoch=94,source_step=58844,queue_pid=os.getpid(),time=time.time(),**value)
 for root in [LOCAL,Path(plan['output'])]:
  pending=root/'queue-status.pending';pending.write_text(json.dumps(value,indent=2)+'\n');pending.replace(root/'queue-status.json')
 parent=Path('/workspace/Projects/laser/outputs')/plan['source_run'].split('/')[-1]
 p=parent/'epoch94-sampler-search.pending';p.write_text(json.dumps(dict(**value,report=str(Path(plan['output'])/'report.json')),indent=2)+'\n');p.replace(parent/'epoch94-sampler-search.json')


def active(pid):
 try:
  return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
 except FileNotFoundError:return False


def main():
 persist(dict(phase='queued',waiting_for=PREVIOUS.name))
 while True:
  prior=read(PREVIOUS/'status.json')
  if prior['phase']=='failed':
   persist(dict(phase='blocked_by_previous_evaluation_error',error=prior.get('error')))
   raise RuntimeError('Previous evaluation failed; no second GPU job launched')
  if prior['phase']=='completed':
   workers=read(PREVIOUS/'worker-pids.json')['workers']
   if not any(active(pid) for pid in workers):
    result=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    values=[int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    if len(values)==8 and max(values)<62*1024:break
  time.sleep(10)
 children={}
 for key,script in [('coordinator','sweep.py'),('finalizer','finalize.py')]:
  with (LOCAL/(key+'.log')).open('ab',buffering=0) as stream:
   child=subprocess.Popen([sys.executable,'-u',str(HERE/script)],stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
  children[key]=child.pid
 (LOCAL/'launch-pids.json').write_text(json.dumps(children,indent=2)+'\n')
 persist(dict(phase='launched',children=children))
 print(json.dumps(children),flush=True)


if __name__=='__main__':main()
