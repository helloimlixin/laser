"""Upload immutable checkpoint links without stalling four training GPUs."""
from concurrent.futures import ThreadPoolExecutor
import json,os,time
from pathlib import Path
POOL=ThreadPoolExecutor(max_workers=1)
JOBS=[]

def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.pending');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def publish(run,snapshot,codebook,config,out,receipt,aliases):
 import wandb
 try:
  a=wandb.Artifact(run.id+'-prior',type='model',metadata=receipt)
  a.add_file(str(snapshot),name='checkpoint.pt',policy='immutable',skip_cache=True)
  a.add_file(str(config),name='config.json');a.add_file(str(codebook),name='vector-codebook.pt')
  a=run.log_artifact(a,aliases=aliases);a.wait();assert str(a.state)=='COMMITTED'
  result=dict(receipt,artifact=a.qualified_name,state='COMMITTED',time=time.time())
  write(out/'cloud-checkpoints'/f"step-{receipt['progress']['step']:09d}.json",result);write(out/'cloud-checkpoint.json',result)
  run.summary['checkpoint/latest_committed_step']=receipt['progress']['step'];snapshot.unlink()
  print(json.dumps(dict(phase='cloud_checkpoint_committed',**result)),flush=True)
 except Exception as error:
  write(out/'cloud-checkpoint-error.json',dict(receipt,error=str(error),snapshot=str(snapshot)))
  raise

def submit(run,path,codebook,config,out,receipt,aliases):
 queue=out/'upload-queue';queue.mkdir(exist_ok=True)
 snapshot=queue/f"step-{receipt['progress']['step']:09d}-{time.time_ns()}.pt";os.link(path,snapshot)
 # Bound disk use and make an earlier failed upload visible before proceeding.
 for job in JOBS:
  if job.done():job.result()
 while len([job for job in JOBS if not job.done()])>=2:
  next(job for job in JOBS if not job.done()).result()
 copied=json.loads(json.dumps(receipt))
 JOBS.append(POOL.submit(publish,run,snapshot,codebook,config,out,copied,aliases))

def finish():
 for job in JOBS:job.result()
