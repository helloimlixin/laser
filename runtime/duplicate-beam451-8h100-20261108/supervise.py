"""Keep the single cutoff5% experiment training on all eight H100 GPUs."""
import fcntl,json,os,signal,subprocess,time,threading,shutil
from pathlib import Path
from environment import *
RUN='imagenet-beam451-cutoff05-scratch100-8h100-20261108'
TRAIN_BASE=BASE/'arms/cutoff05'
TRAIN_OUT=OUT/'arms/cutoff05'
CHILD=None
STOP=False

def read(p):return json.loads(Path(p).read_text())
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.pending');q.write_text(json.dumps(d,indent=2,default=str)+'\n');q.replace(p)
def status(phase,**extra):
 p=TRAIN_BASE/'live-training/train/progress-rank0.json'
 write(OUT/'status.json',dict(phase=phase,supervisor_pid=os.getpid(),child_pid=CHILD.pid if CHILD else None,
 run_id=RUN,world_size=8,microbatch=128,accumulation_steps=2,global_batch=2048,epochs=100,
 baseline_run='helloimlixin-rutgers/laser/imagenet-beam451-aligned-joint16k-scratch100-4h100-20261107',
 progress=read(p) if p.exists() else None,time=time.time(),**extra))
def stop(sig,frame):
 global STOP
 STOP=True
 if CHILD and CHILD.poll() is None:
  try:os.killpg(CHILD.pid,signal.SIGTERM)
  except ProcessLookupError:pass

def latest_checkpoint():
 files=list((TRAIN_BASE/'checkpoints').glob('step-*.pt'))
 if files:return max(files,key=lambda p:int(p.stem.split('-')[-1]))
 receipt=read(TRAIN_OUT/'checkpoint.json');p=Path(receipt['persistent_path']);assert p.stat().st_size==receipt['bytes'];return p

def main():
 global CHILD
 lock=(BASE/'optimization-supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
 assert read(OUT/'preflight-verification.json')['full_state_4_to_8_restore']
 assert read(BASE/'data/ready.json')['training_images']==1281167
 config=read(TRAIN_BASE/'inputs/execution.json');config.update(world_size=8,hardware='8 NVIDIA H100 80GB HBM3',
 microbatch=128,accumulation_steps=2,global_batch=2048,epochs=100,experiment='single higher-noise experiment; original running run supplies baseline',pilot_epochs=None)
 config.pop('ablation',None);write(TRAIN_BASE/'inputs/execution.json',config)
 write(OUT/'wandb.json',read(TRAIN_OUT/'wandb.json'))
 write(OUT/'selection.json',dict(selected='cutoff05',reason='User requested all eight GPUs on one experiment; original run is the baseline',selection_metric=None,quality_superiority_unmeasured=True,epochs=100,world_size=8,time=time.time()))
 attempts={}
 while not STOP:
  checkpoint=latest_checkpoint();step=int(checkpoint.stem.split('-')[-1]);attempts[step]=attempts.get(step,0)+1
  if attempts[step]>3:raise RuntimeError(f'Repeated worker failure from checkpoint step {step}')
  command=[PYTHON,'-u','-m','torch.distributed.run','--standalone','--nproc-per-node=8',str(TRAIN_BASE/'train.py'),
  '--epochs','100','--microbatch','128','--resume',str(checkpoint)]
  env=environment(TRAIN_BASE,TRAIN_OUT,'0,1,2,3,4,5,6,7')
  with (TRAIN_OUT/'training.log').open('ab',buffering=0) as log:
   CHILD=subprocess.Popen(command,env=env,cwd=TRAIN_BASE,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  write(TRAIN_OUT/'launch.json',dict(command=command,pid=CHILD.pid,devices='0,1,2,3,4,5,6,7',world_size=8,epochs=100,resume_step=step,time=time.time()))
  while CHILD.poll() is None:
   status('training_single_experiment',resume_step=step)
   try:CHILD.wait(timeout=15)
   except subprocess.TimeoutExpired:pass
  if STOP:status('stopped_by_signal');return
  if CHILD.returncode==0:
   result=read(TRAIN_OUT/'completed.json');assert result['epochs']==100;status('completed',result=result);return
  status('recovering',exit_code=CHILD.returncode,resume_step=step)
if __name__=='__main__':
 try:main()
 except Exception as error:status('failed',error=str(error));raise
