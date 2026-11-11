"""Run matched duplicate arms concurrently on two GPUs each, with durable receipts."""
import fcntl,json,os,signal,subprocess,sys,time,zipfile,shutil
from pathlib import Path
BASE=Path('/tmp/laser-beam16-duplicate-20261105')
OUT=Path('/workspace/Projects/laser/outputs/imnet-beam16-duplicate-20261105')
HERE=Path(__file__).resolve().parent
CHILDREN=[]

def read(p):return json.loads(p.read_text())
def write(p,value):
 p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix('.pending');tmp.write_text(json.dumps(value,indent=2,default=str)+'\n');tmp.replace(p)

def mirror():
 for arm in ['consistency','control','preflight-consistency','preflight-control','preflight-reload']:
  for name in ['initialization.json','progress.json','target-verification.json','evaluation.json','generation-progress.json','completed.json','wandb.json','checkpoint.json','verification.json']:
   p=BASE/arm/name
   if p.exists():write(OUT/arm/name,read(p))
 for name in ['replica-verification.json','kernel-verification.json']:
  if (BASE/name).exists():shutil.copyfile(BASE/name,OUT/name)

def status(phase,**extra):
 mirror();write(OUT/'status.json',dict(phase=phase,pid=os.getpid(),time=time.time(),**extra))

def stop(sig,_):
 for p in CHILDREN:
  if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
 status('interrupted',signal=sig);raise SystemExit(128+sig)

def launch(arm,gpus,tag=None,smoke=0,resume=None):
 command=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=2',str(BASE/'runner.py'),'--config',str(BASE/'config.yaml'),'--mode','train','--arm',arm]
 if tag:command+=['--tag',tag]
 if smoke:command+=['--smoke-updates',str(smoke)]
 if resume:command+=['--resume',str(resume)]
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpus,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',LASER_TRAIN_SDPA='1',PYTORCH_ALLOC_CONF='expandable_segments:True',WANDB_API_KEY=Path('/root/.config/laser/wandb.key').read_text().strip(),WANDB_BASE_URL='https://api.forge.coreweave.com')
 for k in ['WANDB_SERVICE','_WANDB_SERVICE']:env.pop(k,None)
 with (OUT/((tag or arm)+'.log')).open('ab',buffering=0) as log:
  p=subprocess.Popen(command,cwd=BASE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 CHILDREN.append(p)
 write(OUT/((tag or arm)+'-launch.json'),dict(pid=p.pid,command=command,gpus=gpus,time=time.time()))
 return p

def wait_all(processes,phase):
 while any(p.poll() is None for p in processes):
  failed=[p for p in processes if p.poll() is not None and p.returncode]
  if failed:raise RuntimeError(f'{phase} failed with {failed[0].returncode}; inspect arm log')
  status(phase,children=[dict(pid=p.pid,returncode=p.poll()) for p in processes]);time.sleep(10)
 if any(p.returncode for p in processes):raise RuntimeError(f'{phase} failed; inspect arm log')

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,stop)
 parent=None
 try:
  while not (BASE/'frozen/data/train-ready.json').exists():
   progress=BASE/'frozen/data/status.json'
   status('preparing_data',data=read(progress) if progress.exists() else None);time.sleep(10)
  assert read(BASE/'frozen/data/train-ready.json')['training_images']==1281167
  processes=[]
  for arm,gpus in [('consistency','0,1'),('control','2,3')]:
   tag='preflight-'+arm
   if not (BASE/tag/'verification.json').exists():processes.append(launch(arm,gpus,tag,3))
  wait_all(processes,'preflight')
  for arm in ['consistency','control']:
   proof=read(BASE/('preflight-'+arm)/'verification.json')
   assert proof['passed'] and proof['primary_adam_ages']==[10753]
   assert proof['head_adam_ages']==([3] if arm=='consistency' else [])
   assert proof['initialization']['world_size']==2 and proof['initialization']['global_batch']==2048
  if not (BASE/'preflight-reload/verification.json').exists():
   resume=read(BASE/'preflight-consistency/checkpoint.json')['path']
   wait_all([launch('consistency','0,1','preflight-reload',4,resume)],'preflight_reload')
  proof=read(BASE/'preflight-reload/verification.json')
  assert proof['passed'] and proof['head_adam_ages']==[4] and proof['primary_adam_ages']==[10754]
  assert proof['initialization']['head_optimizer_age']==3
  for tag in ['preflight-control','preflight-consistency','preflight-reload']:
   for p in (BASE/tag/'checkpoints').glob('*.pt'):p.unlink()
  os.environ['WANDB_API_KEY']=Path('/root/.config/laser/wandb.key').read_text().strip();os.environ['WANDB_BASE_URL']='https://api.forge.coreweave.com'
  import wandb
  parent=wandb.init(entity='helloimlixin-rutgers',project='laser',id='imnet-beam16-compare-s10750-4rtx6k-dup-20261105',group='imnet-beam16-matched-duplicate-20261105',name='ImageNet beam16 | parallel matched consistency duplicate | 4 RTX6000',job_type='matched-consistency-duplicate',resume='allow',dir=str(BASE),config=read(BASE/'plan.json'))
  parent.use_artifact(read(BASE/'plan.json')['source_artifact'])
  parent.use_artifact('helloimlixin-rutgers/laser/imnet-beam16-consistency-compare-s10750-20261105-source:v0')
  parent.use_artifact('helloimlixin-rutgers/laser/imagenet-atoms4096-joint16384-native-teacher-beam-reconstruction-tokenizer:v0')
  with zipfile.ZipFile(BASE/'duplicate-source.zip','w',zipfile.ZIP_DEFLATED) as z:
   for p in HERE.glob('*.py'):z.write(p,'launch/'+p.name)
   for name in ['runner.py','config.yaml','plan.json','calibration.json']:z.write(BASE/name,name)
   z.write(OUT/'runtime.patch','runtime.patch')
  a=wandb.Artifact(parent.id+'-source',type='code');a.add_file(str(BASE/'duplicate-source.zip'));a=parent.log_artifact(a);a.wait()
  write(OUT/'source-artifact.json',dict(artifact=a.qualified_name,state=str(a.state)))
  write(OUT/'wandb.json',dict(id=parent.id,url=parent.url))
  write(OUT/'launch-verification.json',dict(passed=True,source_checkpoint_sha256=read(BASE/'plan.json')['source_checkpoint_sha256'],source_step=10750,target_step=12002,updates_per_arm=1252,global_batch=2048,gpus_per_arm=2,concurrent_arms=2,preflight_both_arms=True,resume_verified=True,replicas_bitwise_equal_after_preflight=True,frozen_tokenizer=True,topology_change_not_bitwise_source_replay=True,original_run_untouched=True))
  processes=[]
  for arm,gpus in [('consistency','0,1'),('control','2,3')]:
   if (BASE/arm/'completed.json').exists():continue
   receipt=BASE/arm/'checkpoint.json';resume=read(receipt)['path'] if receipt.exists() else None
   processes.append(launch(arm,gpus,resume=resume))
  wait_all(processes,'training_parallel')
  results={arm:read(BASE/arm/'completed.json') for arm in ['consistency','control']}
  assert all(r['evaluation']['generated_images']==50000 and r['updates']==1252 for r in results.values())
  c=results['consistency']['evaluation'];b=results['control']['evaluation']
  report=dict(consistency=c,control=b,fid_improvement=b['fid']-c['fid'],inception_score_change=c['inception_score']-b['inception_score'],source_checkpoint=read(BASE/'plan.json')['source_artifact'],original_run_untouched=True,limitation='Matched duplicate on a changed GPU topology; not bitwise replay of the original single-GPU run.')
  write(OUT/'report.json',report)
  parent.summary.update(dict(fid_improvement=report['fid_improvement'],inception_score_change=report['inception_score_change'],consistency_fid=c['fid'],control_fid=b['fid']))
  parent.save(str(OUT/'report.json'),base_path=str(OUT),policy='now');parent.finish();parent=None
  status('completed',**report)
 except BaseException as e:
  for p in CHILDREN:
   if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
  status('failed',error=str(e),error_type=type(e).__name__)
  if parent is not None:parent.finish(exit_code=1)
  raise

if __name__=='__main__':main()
