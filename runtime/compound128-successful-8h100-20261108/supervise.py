"""Verify the new 128-bin codec, screen throughput, and train fresh for 100 epochs."""
import os,json,time,signal,subprocess,fcntl,statistics,shutil,hashlib,zipfile
from pathlib import Path
from environment import *
from omegaconf import OmegaConf
CHILD=None;STOP=False

def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.pending');q.write_text(json.dumps(v,indent=2,default=str)+'\n');q.replace(p)
def status(phase,**kw):
 p=BASE/'live-training/train/progress-rank0.json';progress=read(p) if p.exists() else None
 write(OUT/'status.json',dict(phase=phase,supervisor_pid=os.getpid(),child_pid=CHILD.pid if CHILD else None,run_id=OUT.name,world_size=8,epochs=100,global_batch=2048,progress=progress,time=time.time(),**kw))
def stop(sig,frame):
 global STOP
 STOP=True
 if CHILD and CHILD.poll() is None:
  try:os.killpg(CHILD.pid,signal.SIGTERM)
  except ProcessLookupError:pass
 status('stopped_by_signal');raise SystemExit(128+sig)
def launch(command,phase,extra=None):
 global CHILD
 e=environment();e.update(extra or {})
 with (OUT/(phase+'.log')).open('ab',buffering=0) as log:
  CHILD=subprocess.Popen(command,env=e,cwd=BASE,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 while CHILD.poll() is None:
  status(phase)
  try:CHILD.wait(timeout=10)
  except subprocess.TimeoutExpired:pass
 return CHILD.returncode
def distributed(script,args):return [PYTHON,'-u','-m','torch.distributed.run','--standalone','--nproc-per-node=8',str(script),*args]
def benchmark(mb):
 folder=BASE/f'benchmarks/mb{mb}';folder.mkdir(parents=True,exist_ok=True)
 if not (folder/'verification.json').exists():
  code=launch(distributed(BASE/'train.py',['--epochs','100','--microbatch',str(mb),'--smoke-updates','5']),f'benchmark-mb{mb}',dict(LASER_PRIVATE_SMOKE_DIR=str(folder)))
  if code:
   tail=(OUT/f'benchmark-mb{mb}.log').read_text()[-12000:]
   if 'out of memory' in tail.lower():return dict(microbatch=mb,passed=False,reason='GPU memory limit')
   raise RuntimeError(f'Benchmark mb{mb} failed: '+tail[-2500:])
 proof=read(folder/'verification.json');assert proof['passed'] and proof['frozen_tokenizer_unchanged']
 throughput=[];mem=[]
 for rank in range(8):
  s=read(folder/f'startup-rank{rank}.json');assert s['fresh_stage2'] and s['fresh_optimizer'] and s['world_size']==8 and s['coefficient_bins']==128 and s['training_autocast_dtype']=='bfloat16'
  t=read(folder/f'targets-rank{rank}.json');assert t['coefficient_bins']==128 and t['stochastic_prefixes'] and t['soft_coefficient_labels']
  values=[json.loads(l)['images_per_second'] for l in (folder/f'throughput-rank{rank}.jsonl').read_text().splitlines() if json.loads(l)['step']>=3]
  throughput.append(statistics.median(values));mem.append(read(folder/f'progress-rank{rank}.json')['peak_allocated_gib'])
 return dict(passed=True,microbatch=mb,images_per_second=min(throughput),peak_allocated_gib=max(mem),parameter_count=s['parameter_count'],optimizer_state_count=proof['optimizer_state_count'])
def publish_repro():
 r=OUT/'reproducibility';r.mkdir(exist_ok=True)
 for p in (BASE/'inputs').iterdir():
  if p.is_file() and p.stat().st_size<50*1024**2:shutil.copyfile(p,r/p.name)
 for p in HERE.glob('*.py'):shutil.copyfile(p,r/p.name)
 for p in [BASE/'tokenizer-verification.json',OUT/'coefficient-calibration.json',OUT/'throughput-benchmarks.json']:
  if p.exists():shutil.copyfile(p,r/p.name)
 with zipfile.ZipFile(r/'prepared-source.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
  for folder in ['source','upstream','runtime','inputs']:
   for p in (BASE/folder).rglob('*'):
    if p.is_file() and '__pycache__' not in p.parts and p.stat().st_size<50*1024**2:z.write(p,p.relative_to(BASE))
  z.write(BASE/'train.py','train.py')
def main():
 lock=(BASE/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,stop)
 while not (OUT/'previous-experiment-preserved.json').exists():status('waiting_for_previous_checkpoint');time.sleep(5)
 while not (BASE/'inputs/compound-codec.pt').exists():status('calibrating_coefficient_bins');time.sleep(5)
 if not (BASE/'tokenizer-verification.json').exists():
  code=launch(distributed(BASE/'probe.py',['quality']),'codec-quality')
  if code:raise RuntimeError('Full reconstruction check failed')
 q=read(BASE/'tokenizer-verification.json');assert q['passed'] and q['atoms']==4096 and q['coefficient_bins']==128
 assert q['rfid_soft']<=q['rfid_hard']+.15,'Coefficient noise worsens reconstruction beyond allowed calibration margin'
 rows=[benchmark(128),benchmark(64)]
 if rows[0]['passed'] and rows[0]['peak_allocated_gib']<47:rows.append(benchmark(256))
 valid=[r for r in rows if r['passed']];assert valid
 best=max(valid,key=lambda r:r['images_per_second']);mb=best['microbatch']
 write(OUT/'throughput-benchmarks.json',dict(candidates=rows,selected=best,global_batch=2048,world_size=8))
 recipe=OmegaConf.load(BASE/'inputs/recipe.yaml');recipe.microbatch=mb;OmegaConf.save(recipe,BASE/'inputs/recipe.yaml')
 execution=read(BASE/'inputs/execution.json');execution.update(microbatch=mb,accumulation_steps=2048//(8*mb),measured_images_per_second=best['images_per_second']);write(BASE/'inputs/execution.json',execution)
 # Benchmark and restore tests use disposable weights, never production initialization.
 resume=read(BASE/'smoke/checkpoint.json');smoke_step=resume['global_step'];f=BASE/'benchmarks/resume';f.mkdir(exist_ok=True)
 code=launch(distributed(BASE/'train.py',['--epochs','100','--microbatch',str(mb),'--resume',resume['path'],'--smoke-updates',str(smoke_step+1)]),'resume-preflight',dict(LASER_PRIVATE_SMOKE_DIR=str(f)))
 if code:raise RuntimeError('Full optimizer/scheduler/RNG restoration check failed')
 assert read(f/'verification.json')['passed']
 write(OUT/'preflight-verification.json',dict(passed=True,fresh_model_and_optimizer=True,full_state_restore=True,coefficient_bins=128,atoms=4096,codec=q,hardware=best))
 publish_repro()
 attempts={}
 while not STOP:
  command=distributed(BASE/'train.py',['--epochs','100','--microbatch',str(mb)]);step=0
  if (OUT/'checkpoint.json').exists():
   r=read(OUT/'checkpoint.json');p=Path(r['path']) if Path(r['path']).exists() else Path(r['persistent_path']);assert p.stat().st_size==r['bytes'];step=r['global_step'];command+=['--resume',str(p)]
  attempts[step]=attempts.get(step,0)+1
  if attempts[step]>3:raise RuntimeError('Repeated training failure at same complete checkpoint')
  write(OUT/'launch.json',dict(command=command,world_size=8,epochs=100,microbatch=mb,global_batch=2048,resume_step=step,time=time.time()))
  code=launch(command,'training')
  if code==0:
   assert read(OUT/'completed.json')['epochs']==100;status('completed');return
  if not (OUT/'checkpoint.json').exists():raise RuntimeError('Training failed before first checkpoint')
  status('recovering',exit_code=code)
if __name__=='__main__':
 try:main()
 except Exception as e:status('failed',error=str(e));raise
