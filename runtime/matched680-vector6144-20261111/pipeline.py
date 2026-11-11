"""Durable local controller: quality gates, measured hardware tuning, then training."""
import fcntl,hashlib,json,os,signal,statistics,subprocess,sys,tarfile,time,shutil
from pathlib import Path
BASE=Path('/tmp/laser-matched680-vector6144-20261111')
ROOT=Path('/workspace/Projects/laser')
OUT=ROOT/'outputs/imnet-s2-matched680-vector6144-anchor-var-d16-4h100-20261111'
RUNTIME=BASE/'runtime/runtime'
child=None

def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.pending');tmp.write_text(json.dumps(v,indent=2,default=str)+'\n');tmp.replace(p)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def status(phase,**v):
 row=dict(phase=phase,time=time.time(),pid=os.getpid(),child_pid=child.pid if child else None,**v)
 write(OUT/'supervisor-status.json',row);print(json.dumps(row),flush=True)
def stop(sig,_):
 (BASE/'train-anchor/STOP').parent.mkdir(exist_ok=True);(BASE/'train-anchor/STOP').touch()
 if child and child.poll() is None:child.send_signal(signal.SIGTERM)
 status('stop_requested',signal=sig)
def run(command,log,phase,check=True):
 global child
 status(phase,command=command)
 with Path(log).open('ab',buffering=0) as f:
  child=subprocess.Popen(command,cwd=RUNTIME,env=os.environ,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
  while child.poll() is None:
   for name in ['status.json','checkpoint.json','wandb.json','last-preview.json','initialization.json','cloud-checkpoint.json']:
    p=BASE/'train-anchor'/name
    if p.exists():write(OUT/name,read(p))
   status(phase)
   try:child.wait(timeout=10)
   except subprocess.TimeoutExpired:pass
  code=child.returncode
 if check and code:raise RuntimeError(f'{phase} exited {code}; inspect {log}')
 return code

def distributed(script,*args):return [sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(RUNTIME/script),*map(str,args)]

def main():
 OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
 if not (OUT/'cache-verification.json').exists():
  # A cache extraction may already have been started interactively; do not duplicate it.
  for _ in range(360):
   active=subprocess.run(['pgrep','-f','[v]ector_quality.py cache'],capture_output=True,text=True).stdout.strip()
   if not active:break
   status('caching_training_latents');time.sleep(10)
   if (OUT/'cache-verification.json').exists():break
  if not (OUT/'cache-verification.json').exists():run(distributed('vector_quality.py','cache'),BASE/'cache.log','caching_training_latents')
 if not (BASE/'quality/selected-pass.json').exists():run(distributed('vector_quality.py','fit'),BASE/'fit.log','fitting_vector_tables')
 if not (OUT/'selection.json').exists():run(distributed('vector_quality.py','select'),BASE/'select.log','selecting_on_training_holdout')
 # Never let a preallocated but unfinished archive into training or evaluation.
 for _ in range(720):
  log=Path('/tmp/fast_copy.log').read_text() if Path('/tmp/fast_copy.log').exists() else ''
  if 'DONE ILSVRC2012_img_val.tar' in log:break
  status('waiting_for_nvme_staging');time.sleep(10)
 else:raise RuntimeError('NVMe staging did not finish')
 if not (OUT/'codec-verification.json').exists():run(distributed('vector_quality.py','verify'),BASE/'verify.log','validating_codec_50k')
 assert read(OUT/'codec-verification.json')['passed'],'Codec failed full reconstruction quality gate'
 cfg=read(BASE/'config.json')
 if not (OUT/'hardware-benchmark.json').exists():
  results=[]
  # Same global batch and precision for all trials. Discard startup/preview I/O.
  for micro,accum,compiled in [(64,3,False),(96,2,False),(192,1,False),(None,None,True)]:
   if compiled:
    feasible=[r for r in results if r['passed']]
    if not feasible:continue
    fastest=max(feasible,key=lambda r:r['images_per_second']);micro=fastest['microbatch'];accum=fastest['accumulation']
   tag=f'mb{micro}'+('-compiled' if compiled else '')
   path=BASE/f'benchmark-{tag}.json';folder=BASE/f'benchmark-{tag}';folder.mkdir(exist_ok=True)
   c=dict(cfg,output=str(folder),microbatch=micro,accumulation=accum,workers=4,compile=compiled)
   write(path,c)
   log=BASE/f'benchmark-{tag}.log'
   code=run(distributed('train_vector_stage2.py','--config',path,'--smoke','--max-steps','12'),log,f'benchmark_microbatch_{micro}',check=False)
   points=[];memory=0
   for line in log.read_text().splitlines():
    try:r=json.loads(line)
    except (ValueError,TypeError):continue
    if r.get('phase')=='train':
     memory=max(memory,r.get('peak_memory_gib',0))
     if r['step']>=4:points.append(r['images_per_second'])
   proof=folder/'smoke-complete.json'
   result=dict(microbatch=micro,accumulation=accum,compile=compiled,returncode=code,passed=code==0 and proof.exists(),images_per_second=statistics.median(points) if points else 0,peak_memory_gib=memory)
   results.append(result);write(OUT/'hardware-benchmark-progress.json',results)
  feasible=[r for r in results if r['passed']];assert feasible,'No four-GPU training preflight passed'
  selected=max(feasible,key=lambda r:r['images_per_second'])
  cfg.update(microbatch=selected['microbatch'],accumulation=selected['accumulation'],compile=selected['compile'])
  write(BASE/'config.json',cfg);write(OUT/'config.json',cfg)
  write(OUT/'hardware-benchmark.json',dict(passed=True,gpus=4,hardware='NVIDIA H100 80GB HBM3',global_batch=768,precision='bf16',optimizer='fused AdamW',attention='PyTorch SDPA',data='local NVMe; 16 loader workers total',trials=results,selected=selected))
 else:cfg=read(BASE/'config.json')
 # Freeze everything needed to reconstruct the codec and prior; preserve locally and on W&B.
 for name in ['source-verification.json','codec-verification.json','cache-verification.json','hardware-benchmark.json','selection.json','fit-trace.json']:
  shutil.copyfile(OUT/name,BASE/name)
 with tarfile.open(BASE/'source-vector6144.tar.gz','w:gz') as archive:
  for p in RUNTIME.rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts:archive.add(p,arcname='runtime/'+str(p.relative_to(RUNTIME)))
  for name in ['source-verification.json','codec-verification.json','cache-verification.json','hardware-benchmark.json','selection.json','fit-trace.json']:archive.add(BASE/name,arcname=name)
 (OUT/'inputs').mkdir(exist_ok=True)
 for source,name in [(BASE/'quality/selected-codebook.pt','vector-codebook.pt'),(BASE/'source-vector6144.tar.gz','source.tar.gz')]:shutil.copyfile(source,OUT/'inputs'/name)
 if not (OUT/'inputs/stage1-checkpoint.pt').exists():shutil.copyfile(cfg['checkpoint'],OUT/'inputs/stage1-checkpoint.pt')
 write(OUT/'launch.json',dict(command=['python','train.py','--config','configs/stage2/imagenet-matched680-vector6144-anchor-classcond-4h100-20261111.yaml'],training_command=distributed('train_vector_stage2.py','--config',BASE/'config.json'),config_sha256=sha(BASE/'config.json'),source_sha256=sha(BASE/'source-vector6144.tar.gz'),source_stage1_sha256=cfg['source_checkpoint_sha256'],codebook_sha256=cfg['codebook_sha256'],world_size=4,global_batch=768,microbatch=cfg['microbatch'],accumulation=cfg['accumulation'],epochs=200,time=time.time()))
 for attempt in range(1,4):
  code=run(distributed('train_vector_stage2.py','--config',BASE/'config.json'),OUT/'production.log','stage2_training',check=False)
  if code==0:
   finished=read(BASE/'train-anchor/checkpoint.json')['progress']['epoch']>=cfg['epochs'];status('completed' if finished else 'stopped');return
  status('training_failed',attempt=attempt,returncode=code)
  if attempt<3:time.sleep(10)
 raise RuntimeError('Stage2 failed three attempts')
if __name__=='__main__':
 try:main()
 except Exception as e:status('failed',error=str(e));raise
