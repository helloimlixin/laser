"""Run native OMP compound128 with hard categorical labels on all four GPUs."""
import fcntl,json,os,signal,subprocess,sys,time,shutil
from pathlib import Path
BASE=Path('/tmp/laser-var-stage2-compound128-omp-20261107')
OUT=Path('/workspace/Projects/laser/outputs/imnet-s2-laser-compound128-omp-var-d16-4gpu-20261107')
child=None;stopping=False

def read(p):return json.loads(p.read_text())
def write(p,v):
 tmp=p.with_suffix('.pending');tmp.write_text(json.dumps(v,indent=2,default=str)+'\n');tmp.replace(p)
def mirror():
 for name in ['status.json','checkpoint.json','wandb.json','initialization.json','generation-progress.json','last-preview.json']:
  p=BASE/'run'/name
  if p.exists():write(OUT/name,read(p))
 for pattern in ['failure-rank*.json','fid-epoch*.json']:
  for p in (BASE/'run').glob(pattern):write(OUT/p.name,read(p))
def status(phase,**extra):
 mirror();write(OUT/'supervisor-status.json',dict(phase=phase,stage='stage2',class_conditional=True,tokenizer_frozen=True,world_size=4,pid=os.getpid(),time=time.time(),**extra))
def stop(sig,_):
 global stopping
 stopping=True;(BASE/'run/STOP').touch();status('stopping',signal=sig)
def main():
 global child
 OUT.mkdir(parents=True,exist_ok=True)
 lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for s in [signal.SIGINT,signal.SIGTERM]:signal.signal(s,stop)
 proof=read(BASE/'preflight-verification.json')
 assert proof['passed'] and proof['four_gpu_training'] and proof['full_state_roundtrip'] and proof['tokenizer_digest_unchanged'] and proof['sampling_finite']
 cfg=read(BASE/'config.json');assert cfg['world_size']==4 and cfg['global_batch']==768 and cfg['evaluation_cadence']=='epoch_end_only'
 assert cfg['source_checkpoint_sha256']==proof['tokenizer_sha256']
 shutil.copyfile(BASE/'preflight-verification.json',OUT/'preflight-verification.json')
 assert not (BASE/'run/STOP').exists()
 command=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=4',str(BASE/'runtime/train_native_compound128_stage2.py'),'--config',str(BASE/'config.json')]
 env=dict(os.environ,WANDB_API_KEY=Path('/root/.config/laser/wandb.key').read_text().strip(),WANDB_BASE_URL='https://api.forge.coreweave.com',WANDB_DISABLE_GIT='true',CUDA_VISIBLE_DEVICES='0,1,2,3',TORCH_HOME='/tmp/laser-beam16-duplicate-20261105/frozen/torch-cache',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
 for k in ['WANDB_SERVICE','_WANDB_SERVICE']:env.pop(k,None)
 for attempt in range(1,4):
  if stopping or (BASE/'run/STOP').exists():status('stopped');return
  with (OUT/'production.log').open('ab',buffering=0) as log:
   child=subprocess.Popen(command,cwd=BASE/'runtime',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  write(OUT/'launch.json',dict(stage='stage2',class_conditional=True,tokenizer_frozen=True,world_size=4,pid=child.pid,supervisor_pid=os.getpid(),command=command,attempt=attempt,gpus=[0,1,2,3],global_batch=768,microbatch=cfg['microbatch'],accumulation=cfg['accumulation'],source_stage1_epoch=10,source_greedy_stage1_rfid=cfg['source_greedy_stage1_rfid'],encoding='native_greedy_omp',beam_search=False,pair_token_format='atom_id*128+coefficient_bin_id',atom_vocabulary=1024,coefficient_bins=128,shared_event_vocabulary=1152,compound_pairs_per_image=680,scalar_events_per_image=1360,peak_lr=1e-4,teacher='hard_omp_atom_and_coefficient_bin_labels',tokenizer_sha256=cfg['source_checkpoint_sha256'],evaluation='epoch_end_only_full50k',time=time.time()))
  while child.poll() is None:
   status('training',training_pid=child.pid,attempt=attempt)
   try:child.wait(timeout=10)
   except subprocess.TimeoutExpired:pass
  mirror()
  if child.returncode==0:
   checkpoint=read(BASE/'run/checkpoint.json');completed=checkpoint['progress']['epoch']>=cfg['epochs']
   status('completed' if completed else 'stopped',returncode=0);return
  status('retry_wait' if attempt<3 else 'failed',returncode=child.returncode,attempt=attempt)
  if attempt<3:time.sleep(10)
if __name__=='__main__':main()
