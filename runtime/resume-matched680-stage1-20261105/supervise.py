"""Keep the requested matched-budget LASER VAR stage-1 continuation supervised."""
import json,os,sys,fcntl,signal,subprocess,time
from pathlib import Path
BASE=Path('/tmp/laser-var-stage1-20261105')
OUT=Path('/workspace/Projects/laser/outputs/imnet-s1-laser-matched680-best24608-resume-20261105')
HERE=Path(__file__).resolve().parent
child=None;stopping=False

def read(p):return json.loads(p.read_text())
def write(p,v):
 tmp=p.with_suffix('.pending');tmp.write_text(json.dumps(v,indent=2,default=str)+'\n');tmp.replace(p)
def mirror():
 for name in ['status.json','checkpoint.json','wandb.json','learning-dynamics.json','guard-decision.json','best-reconstruction.json','resume-runtime-artifact.json','recovery.json','last-preview.json']:
  p=BASE/'run'/name
  if p.exists():write(OUT/name,read(p))
 for p in (BASE/'run').glob('failure-rank*.json'):write(OUT/p.name,read(p))
def status(phase,**extra):
 mirror();write(OUT/'supervisor-status.json',dict(phase=phase,stage='stage1',pid=os.getpid(),time=time.time(),**extra))
def stop(sig,_):
 global stopping
 stopping=True;(BASE/'run/STOP').touch();status('stopping',signal=sig)

def main():
 global child
 lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for s in [signal.SIGINT,signal.SIGTERM]:signal.signal(s,stop)
 proof=read(OUT/'preflight-verification.json');assert proof['passed'] and proof['tokens_per_image']==680 and proof['stage2_enabled'] is False
 assert not (BASE/'run/STOP').exists()
 command=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=1',str(HERE/'entry.py'),'--config',str(BASE/'run/config.json')]
 env=dict(os.environ,WANDB_API_KEY=Path('/root/.config/laser/wandb.key').read_text().strip(),WANDB_BASE_URL='https://api.forge.coreweave.com',WANDB_DISABLE_GIT='true',CUDA_VISIBLE_DEVICES='0',LASER_VGG16_WEIGHTS=str(BASE/'weights/vgg16-397923af.pth'),TORCH_HOME='/tmp/laser-beam16-duplicate-20261105/frozen/torch-cache',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
 for k in ['WANDB_SERVICE','_WANDB_SERVICE']:env.pop(k,None)
 for attempt in range(1,4):
  if stopping or (BASE/'run/STOP').exists():status('stopped');return
  with (OUT/'production.log').open('ab',buffering=0) as log:
   child=subprocess.Popen(command,cwd=BASE/'frozen/runtime',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  write(OUT/'launch.json',dict(stage='stage1',pid=child.pid,supervisor_pid=os.getpid(),attempt=attempt,command=command,cwd=str(BASE/'frozen/runtime'),gpu=0,source_step=43166,source_artifact=read(OUT/'resume-verification.json')['source_artifact'],tokens_per_image=680,vocabulary=4096,nominal_bits_per_image=8160,stage2_enabled=False,time=time.time()))
  while child.poll() is None:
   status('training',training_pid=child.pid,attempt=attempt)
   try:child.wait(timeout=10)
   except subprocess.TimeoutExpired:pass
  mirror()
  if child.returncode==0:status('stopped',returncode=0);return
  status('retry_wait' if attempt<3 else 'failed',returncode=child.returncode,attempt=attempt)
  if attempt<3:time.sleep(10)
if __name__=='__main__':main()
