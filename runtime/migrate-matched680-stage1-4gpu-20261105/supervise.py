"""Supervise the four-GPU continuation without changing the optimizer/data clock."""
import fcntl,json,os,signal,subprocess,sys,time,shutil
from pathlib import Path
BASE=Path('/tmp/laser-var-stage1-4gpu-20261105')
OUT=Path('/workspace/Projects/laser/outputs/imnet-s1-laser-matched680-best24608-resume-20261105')
child=None;stopping=False

def read(p):return json.loads(p.read_text())
def write(p,v):
 tmp=p.with_suffix('.4gpu-pending');tmp.write_text(json.dumps(v,indent=2,default=str)+'\n');tmp.replace(p)
def mirror():
 for name in ['status.json','checkpoint.json','wandb.json','best-reconstruction.json','learning-dynamics.json','guard-decision.json','recovery.json','four-gpu-runtime-artifact.json']:
  p=BASE/'run'/name
  if p.exists():write(OUT/name,read(p))
 for p in (BASE/'run').glob('failure-rank*.json'):write(OUT/p.name,read(p))
def status(phase,**extra):
 mirror();write(OUT/'supervisor-status.json',dict(phase=phase,stage='stage1',world_size=4,pid=os.getpid(),time=time.time(),**extra))
def stop(sig,_):
 global stopping
 stopping=True;(BASE/'run/STOP').touch();status('stopping',signal=sig)
def main():
 global child
 lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for s in [signal.SIGINT,signal.SIGTERM]:signal.signal(s,stop)
 proof=read(BASE/'preflight-verification.json')
 assert proof['passed'] and proof['world_size']==4 and proof['all_rank_parameters_bit_exact'] and proof['distributed_best_recovery_passed']
 shutil.copyfile(BASE/'preflight-verification.json',OUT/'four-gpu-preflight.json')
 assert not (BASE/'run/STOP').exists()
 command=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=4',str(BASE/'runtime/four_gpu_driver.py'),'--config',str(BASE/'run/config.json')]
 env=dict(os.environ,WANDB_API_KEY=Path('/root/.config/laser/wandb.key').read_text().strip(),WANDB_BASE_URL='https://api.forge.coreweave.com',WANDB_DISABLE_GIT='true',CUDA_VISIBLE_DEVICES='0,1,2,3',LASER_VGG16_WEIGHTS='/tmp/laser-var-stage1-20261105/weights/vgg16-397923af.pth',TORCH_HOME='/tmp/laser-beam16-duplicate-20261105/frozen/torch-cache',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
 for k in ['WANDB_SERVICE','_WANDB_SERVICE']:env.pop(k,None)
 for attempt in range(1,4):
  if stopping or (BASE/'run/STOP').exists():status('stopped');return
  with (OUT/'production-4gpu.log').open('ab',buffering=0) as log:
   child=subprocess.Popen(command,cwd=BASE/'runtime',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  write(OUT/'launch-4gpu.json',dict(stage='stage1',world_size=4,pid=child.pid,supervisor_pid=os.getpid(),command=command,attempt=attempt,gpus=[0,1,2,3],microbatch=32,accumulation=1,global_batch=128,source_step=read(BASE/'migration-receipt.json')['source_step'],source_artifact=read(BASE/'migration-receipt.json')['source_artifact'],tokens_per_image=680,vocabulary=4096,time=time.time()))
  shutil.copyfile(OUT/'launch-4gpu.json',OUT/'launch.json')
  while child.poll() is None:
   status('training',training_pid=child.pid,attempt=attempt)
   try:child.wait(timeout=10)
   except subprocess.TimeoutExpired:pass
  mirror()
  if child.returncode==0:status('stopped',returncode=0);return
  status('retry_wait' if attempt<3 else 'failed',returncode=child.returncode,attempt=attempt)
  if attempt<3:time.sleep(10)
if __name__=='__main__':main()
