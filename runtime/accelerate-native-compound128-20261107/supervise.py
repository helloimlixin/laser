"""Continue the same native compound128 run with verified execution improvements."""
import fcntl,json,os,signal,subprocess,sys,time,shutil
from pathlib import Path
BASE=Path('/tmp/laser-native-speed-20261107')
OUT=Path('/workspace/Projects/laser/outputs/imnet-s2-laser-compound128-omp-var-d16-4gpu-20261107')
child=None;stopping=False

def read(p):return json.loads(p.read_text())
def write(p,v):
 temp=p.with_suffix('.pending');temp.write_text(json.dumps(v,indent=2,default=str)+'\n');temp.replace(p)
def mirror():
 for name in ['status.json','checkpoint.json','wandb.json','initialization.json','generation-progress.json','last-preview.json']:
  p=BASE/'run'/name
  if p.exists():write(OUT/name,read(p))
 for pattern in ['failure-rank*.json','fid-epoch*.json']:
  for p in (BASE/'run').glob(pattern):write(OUT/p.name,read(p))
def status(phase,**extra):
 mirror();write(OUT/'supervisor-status.json',dict(phase=phase,stage='stage2',native_omp=True,beam_search=False,compiled_forward=True,
  tokenizer_frozen=True,world_size=4,pid=os.getpid(),time=time.time(),**extra))
def stop(sig,_):
 global stopping
 stopping=True;(BASE/'run/STOP').touch();status('stopping',signal=sig)
def main():
 global child
 lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,stop)
 proof=read(BASE/'preflight-verification.json');resume=read(BASE/'resume-verification.json');numeric=read(BASE/'numerical-verification.json')
 assert proof['passed'] and proof['all_rank_parameters_bit_exact'] and proof['all_rank_rng_bit_exact'] and proof['tokenizer_digest_unchanged']
 assert resume['passed'] and resume['scientific_configuration_identical'] and resume['optimizer_bit_exact'] and numeric['passed']
 assert not (BASE/'run/STOP').exists()
 command=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=4',str(BASE/'runtime/fast_driver.py'),'--config',str(BASE/'config.json')]
 env=dict(os.environ,WANDB_API_KEY=Path('/root/.config/laser/wandb.key').read_text().strip(),WANDB_BASE_URL='https://api.forge.coreweave.com',
  WANDB_DISABLE_GIT='true',CUDA_VISIBLE_DEVICES='0,1,2,3',TORCH_HOME='/tmp/laser-beam16-duplicate-20261105/frozen/torch-cache',
  OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TORCHINDUCTOR_COMPILE_THREADS='8',PYTHONUNBUFFERED='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
 for k in ['WANDB_SERVICE','_WANDB_SERVICE']:env.pop(k,None)
 for attempt in range(1,4):
  if stopping or (BASE/'run/STOP').exists():status('stopped');return
  with (OUT/'production.log').open('ab',buffering=0) as log:
   child=subprocess.Popen(command,cwd=BASE/'runtime',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  write(OUT/'launch.json',dict(stage='stage2',native_omp=True,beam_search=False,compound_formulation=True,coefficient_bins=128,
   compiled_forward=True,tokenizer_frozen=True,world_size=4,global_batch=768,pid=child.pid,supervisor_pid=os.getpid(),command=command,
   attempt=attempt,resumed_from_step=595,model_optimizer_rng_and_cursor_preserved=True,gpus=[0,1,2,3],time=time.time()))
  while child.poll() is None:
   status('training',training_pid=child.pid,attempt=attempt)
   try:child.wait(timeout=10)
   except subprocess.TimeoutExpired:pass
  mirror()
  if child.returncode==0:
   completed=read(BASE/'run/checkpoint.json')['progress']['epoch']>=read(BASE/'config.json')['epochs']
   status('completed' if completed else 'stopped',returncode=0);return
  status('retry_wait' if attempt<3 else 'failed',returncode=child.returncode,attempt=attempt)
  if attempt<3:time.sleep(10)
if __name__=='__main__':main()
