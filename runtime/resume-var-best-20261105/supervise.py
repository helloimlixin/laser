"""Supervise the authorized best-checkpoint VAR continuation."""
import json,os,sys,fcntl,signal,subprocess,time,shutil
from pathlib import Path
BASE=Path('/tmp/laser-var-resume-20261105')
OUT=Path('/workspace/Projects/laser/outputs/imnet-var-j4096-bestep1-continue200-4rtx6k-20261105')
HERE=Path(__file__).resolve().parent
child=None;stopping=False

def read(p):return json.loads(p.read_text())
def write(p,v):
 tmp=p.with_suffix('.pending');tmp.write_text(json.dumps(v,indent=2,default=str)+'\n');tmp.replace(p)
def mirror():
 for name in ['status.json','checkpoint.json','resume-verification.json','wandb.json','last-preview.json','resume-runtime-artifact.json','initialization.json']:
  p=BASE/'run'/name
  if p.exists():write(OUT/name,read(p))
 for p in (BASE/'run').glob('fid-epoch*.json'):write(OUT/p.name,read(p))
 for p in (BASE/'run').glob('failure-rank*.json'):write(OUT/p.name,read(p))
def status(phase,**extra):
 mirror();write(OUT/'supervisor-status.json',dict(phase=phase,pid=os.getpid(),time=time.time(),**extra))
def stop(sig,_):
 global stopping
 stopping=True;(BASE/'run/STOP').touch()
 if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)
 status('stopping',signal=sig)

def main():
 global child
 OUT.mkdir(parents=True,exist_ok=True)
 lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for s in [signal.SIGTERM,signal.SIGINT]:signal.signal(s,stop)
 smoke=read(BASE/'preflight/smoke-complete.json')
 assert smoke['completed'] and smoke['all_rank_rng_bit_exact'] and smoke['model_head_bit_exact'] and smoke['optimizer_age']==[1670]
 write(OUT/'preflight-verification.json',smoke)
 assert not (BASE/'run/STOP').exists()
 command=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=4',str(HERE/'entry.py'),'--config',str(BASE/'config.json')]
 env=dict(os.environ,WANDB_API_KEY=Path('/root/.config/laser/wandb.key').read_text().strip(),WANDB_BASE_URL='https://api.forge.coreweave.com',WANDB_DISABLE_GIT='true',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TORCH_HOME='/tmp/laser-beam16-duplicate-20261105/frozen/torch-cache',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0,1,2,3')
 for k in ['WANDB_SERVICE','_WANDB_SERVICE']:env.pop(k,None)
 for attempt in range(1,4):
  if stopping or (BASE/'run/STOP').exists():status('stopped');return
  with (OUT/'production.log').open('ab',buffering=0) as log:
   child=subprocess.Popen(command,cwd=BASE/'runtime',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  write(OUT/'launch.json',dict(pid=child.pid,supervisor_pid=os.getpid(),attempt=attempt,command=command,cwd=str(BASE/'runtime'),gpus=[0,1,2,3],source_step=1668,source_artifact=read(OUT/'source-verification.json')['source_artifact'],time=time.time()))
  while child.poll() is None:
   status('running',training_pid=child.pid,attempt=attempt)
   try:child.wait(timeout=10)
   except subprocess.TimeoutExpired:pass
  mirror()
  if child.returncode==0:
   checkpoint=read(BASE/'run/checkpoint.json') if (BASE/'run/checkpoint.json').exists() else None
   phase='completed' if checkpoint and checkpoint['progress']['epoch']>=200 else 'stopped'
   status(phase,returncode=0,checkpoint=checkpoint);return
  status('retry_wait' if attempt<3 else 'failed',returncode=child.returncode,attempt=attempt)
  if attempt<3:time.sleep(30)

if __name__=='__main__':main()
