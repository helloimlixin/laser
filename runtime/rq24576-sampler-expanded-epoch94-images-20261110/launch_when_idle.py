"""Reuse completed epoch94 evaluations, then run the added sampler batch."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
LOCAL=Path('/tmp/laser-rq24576-sampler-expanded-epoch94-images-20261110')


def read(p):return json.loads(Path(p).read_text())


def write(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 pending=p.with_name(p.name+f'.{os.getpid()}.pending');pending.write_text(json.dumps(value,indent=2)+'\n');pending.replace(p)


def active(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
 except FileNotFoundError:return False


def main():
 lock=(LOCAL/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (LOCAL/'launch-pids.json').exists(),'This batch has already been launched'
 plan=read(LOCAL/'plan.json');previous=Path(plan['predecessor']);output=Path(plan['output'])
 def status(phase,**values):
  value=dict(phase=phase,time=time.time(),queue_pid=os.getpid(),source_epoch=94,new_configs=12,total_configs=18,**values)
  for root in [LOCAL,output]:write(root/'queue-status.json',value)
  parent=Path('/workspace/Projects/laser/outputs')/plan['source_run'].split('/')[-1]
  write(parent/'epoch94-sampler-expanded.json',dict(**value,report=str(output/'report.json')))
 status('queued',waiting_for=previous.name)
 while True:
  prior=read(previous/'status.json')
  if prior['phase']=='failed':
   status('previous_evaluation_needs_attention',error=prior.get('error'))
   raise RuntimeError('Previous evaluation failed; added batch not launched')
  if prior['phase']=='completed' and (Path(plan['predecessor_output'])/'finalization.json').exists():
   workers=read(previous/'worker-pids.json')['workers']
   if not any(active(pid) for pid in workers):
    result=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    memory=[int(x.strip()) for x in result.stdout.splitlines() if x.strip()]
    if len(memory)==8 and max(memory)<62*1024:break
  time.sleep(10)
 prior_plan=read(previous/'plan.json')
 for key in ['checkpoint_sha256','reference_sha256','source_step','source_epoch','batch','world_size','is_shuffle_seed','active_recipe_sha256']:
  assert prior_plan[key]==plan[key],key
 assert prior_plan['previews']==plan['previews']
 cached=[];hardlinks=0
 def link(source,target):
  nonlocal hardlinks
  target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists():assert source.stat().st_ino==target.stat().st_ino
  else:os.link(source,target)
  hardlinks+=1
 for path in sorted((previous/'confirm').glob('*.json')):
  row=read(path);name=row['policy']
  assert row['checkpoint_sha256']==plan['checkpoint_sha256'] and row['reference_sha256']==plan['reference_sha256']
  assert row['sampling']==plan['policies'][name] and row['seed']==plan['policy_seeds'][name] and row['generated_images']==50000
  row['reused_from']=str(path)
  write(LOCAL/'confirm'/path.name,row);write(output/'confirm'/path.name,row)
  for rank in range(8):
   for suffix in ['.json','.npz']:
    source=previous/'confirm'/name/(f'rank-{rank}'+suffix)
    assert source.exists();link(source,LOCAL/'confirm'/name/source.name)
  cached.append(name)
 for folder in sorted((previous/'previews').iterdir()):
  if not folder.is_dir():continue
  for rank in range(8):
   receipt=read(folder/f'rank-{rank}.json')
   assert receipt['checkpoint_sha256']==plan['checkpoint_sha256'] and receipt['sampling']==plan['policies'][folder.name]
   for suffix in ['.json','.png']:
    source=folder/(f'rank-{rank}'+suffix);link(source,LOCAL/'previews'/folder.name/source.name)
 for rank in range(8):
  proof=read(previous/f'verification-rank-{rank}.json');assert proof['passed']
  write(output/f'cache-source-verification-rank-{rank}.json',proof)
 provenance=dict(passed=True,source_run=read(previous/'wandb.json')['url'],source_checkpoint_sha256=plan['checkpoint_sha256'],
  reference_sha256=plan['reference_sha256'],cached_evaluations=cached,cached_full50k_evaluations=len(cached),
  source_files_hardlinked=hardlinks,metric_stats_and_rng_receipts_retained=True,matched_images_reused=True)
 for root in [LOCAL,output]:write(root/'cache-provenance.json',provenance)
 children={}
 for key,script in [('coordinator','sweep.py'),('finalizer','finalize.py'),('best_tracker','track_best.py')]:
  with (LOCAL/(key+'.log')).open('ab',buffering=0) as stream:
   child=subprocess.Popen([sys.executable,'-u',str(HERE/script)],stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
  children[key]=child.pid
 write(LOCAL/'launch-pids.json',children);status('launched',children=children,cached_evaluations=len(cached))
 print(json.dumps(dict(children=children,cache=provenance)),flush=True)


if __name__=='__main__':main()
