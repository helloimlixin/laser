"""Run the verified eight-GPU LR branch and preserve replay/comparison evidence."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
PLAN=json.loads((HERE/'plan.json').read_text())
BASE=Path(PLAN['base']);OUT=Path(PLAN['output'])
PARENT=Path('/workspace/Projects/laser/outputs')/PLAN['parent_run_id']
CHILD=None


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 pending=path.with_name(path.name+f'.{os.getpid()}.pending');pending.write_text(json.dumps(value,indent=2)+'\n');pending.replace(path)


def alive(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
 except FileNotFoundError:return False


def environment():
 spec=importlib.util.spec_from_file_location('parent_env',HERE.parent/'rq24576-native-8h100-20261108/environment.py')
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 env=mod.environment()
 env.update(LASER_STAGE2_OUTPUT=str(OUT),LASER_BEAM_BASE=str(BASE),WANDB_CACHE_DIR=str(BASE/'wandb-cache'))
 return env


def status(phase,**values):
 progress_path=BASE/'live-training/train/progress-rank0.json'
 progress=read(progress_path) if progress_path.exists() else None
 write(OUT/'status.json',dict(phase=phase,run_id=PLAN['run_id'],supervisor_pid=os.getpid(),
  child_pid=CHILD.pid if CHILD else None,world_size=8,source_epoch=PLAN['source_epoch'],
  requested_rewind_epoch=68,learning_rate_branch=PLAN['policy'],target_epoch=100,
  progress=progress,time=time.time(),**values))


def stop(sig,_):
 if CHILD is not None and CHILD.poll() is None:
  try:os.killpg(CHILD.pid,signal.SIGTERM)
  except ProcessLookupError:pass
 status('stopped_by_signal');raise SystemExit(128+sig)


def comparisons():
 rows=[]
 for f in sorted((OUT/'evaluations').glob('epoch*.json')):
  old=PARENT/'evaluations'/f.name
  if not old.exists():continue
  new,previous=read(f),read(old)
  assert new['sampler']==previous['sampler']
  epoch=new['train/epoch']
  rows.append(dict(epoch=epoch,phase='replay' if epoch<=68 else 'changed_learning_rate',
   original_fid=previous['eval/fid'],new_fid=new['eval/fid'],fid_improvement=previous['eval/fid']-new['eval/fid'],
   original_is=previous['eval/inception_score'],new_is=new['eval/inception_score'],
   is_improvement=new['eval/inception_score']-previous['eval/inception_score'],sampler=new['sampler'],
   generated_images=50000,reference_images=1281167,reference_split='train'))
 if rows:
  write(OUT/'matched-evaluations.json',dict(results=rows))
  write(OUT/'matched-evaluation-latest.json',rows[-1])
 refs=read(OUT/'tuned-reference.json') if (OUT/'tuned-reference.json').exists() else {}
 tuned=[]
 for f in sorted((OUT/'tuned-evaluations').glob('epoch*.json')):
  new=read(f);old=refs.get(str(new['epoch']))
  row=dict(new)
  if old:
   assert new['sampler']==old['sampling'] and new['seed']==old['seed']
   row.update(original_fid=old['fid'],fid_improvement=old['fid']-new['fid'],
    original_is=old['inception_score'],is_improvement=new['inception_score']-old['inception_score'])
  tuned.append(row)
 if tuned:write(OUT/'matched-tuned-evaluations.json',dict(results=tuned))


def audit_replay():
 old_path=PARENT/'train/metrics.jsonl';new_path=OUT/'train/metrics.jsonl'
 if not old_path.exists() or not new_path.exists():return
 old={}
 for line in old_path.read_text().splitlines():
  row=json.loads(line);step=row.get('train/global_step')
  if step is not None and PLAN['source_step']<step<=42568:old[step]=row
 comparisons=[]
 for line in new_path.read_text().splitlines():
  row=json.loads(line);step=row.get('train/global_step')
  if step in old:
   comparisons.append(dict(step=step,original_loss=old[step]['train/loss'],new_loss=row['train/loss'],
    loss_absolute_difference=abs(old[step]['train/loss']-row['train/loss']),
    lr_absolute_difference=abs(old[step]['train/lr']-row['train/lr'])))
 if comparisons:
  write(OUT/'replay-audit.json',dict(compared_updates=len(comparisons),source_epoch=62,target_epoch=68,
   max_loss_absolute_difference=max(r['loss_absolute_difference'] for r in comparisons),
   max_lr_absolute_difference=max(r['lr_absolute_difference'] for r in comparisons),
   bitwise_reproduction_claimed=False,comparisons=comparisons))


def main():
 global CHILD
 lock=(BASE/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,stop)
 assert read(OUT/'resume-preflight-verification.json')['passed']
 assert read(OUT/'lr-policy-verification.json')['passed']
 assert read(OUT/'source-checkpoint-verification.json')['passed']
 smoke=read(OUT/'smoke-launch.json')
 status('waiting_for_validation_workers_to_exit')
 while alive(smoke['pid']):time.sleep(2)
 archive=OUT/'reproducibility';archive.mkdir(exist_ok=True)
 for path in HERE.glob('*.py'):shutil.copyfile(path,archive/path.name)
 for name in ['recipe.yaml','parent-recipe.yaml','lr-branch.json','execution.json','stage2-training-config.yaml']:
  shutil.copyfile(BASE/'inputs'/name,archive/name)
 for name in ['resume-preflight-verification.json','lr-policy-verification.json','source-checkpoint-verification.json']:
  shutil.copyfile(OUT/name,archive/name)
 attempts={}
 while True:
  resume=Path(PLAN['source_checkpoint']);step=PLAN['source_step'];new_run=True
  if (OUT/'checkpoint.json').exists():
   receipt=read(OUT/'checkpoint.json');resume=Path(receipt['path'])
   if not resume.exists():resume=Path(receipt['persistent_path'])
   assert resume.stat().st_size==receipt['bytes'];step=receipt['global_step'];new_run=False
  attempts[step]=attempts.get(step,0)+1
  assert attempts[step]<=3,f'Repeated training failure at step {step}'
  args=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc-per-node=8',str(BASE/'train.py'),
   '--epochs','100','--microbatch','64','--resume',str(resume)]
  if new_run:args.append('--new-wandb-run')
  with (OUT/'training.log').open('ab',buffering=0) as stream:
   CHILD=subprocess.Popen(args,env=environment(),cwd=BASE,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  write(OUT/'launch.json',dict(command=args,pid=CHILD.pid,world_size=8,resume_step=step,source_epoch=PLAN['source_epoch'],
   lr_change_epoch=68,run_id=PLAN['run_id'],time=time.time()))
  last_audit=0
  while CHILD.poll() is None:
   progress=BASE/'live-training/train/progress-rank0.json'
   epoch=read(progress).get('train/epoch',PLAN['source_epoch']) if progress.exists() else PLAN['source_epoch']
   status('replaying_to_epoch68' if epoch<68 else 'training_with_adjusted_lr')
   if time.monotonic()-last_audit>60:
    comparisons();audit_replay();last_audit=time.monotonic()
   try:CHILD.wait(timeout=10)
   except subprocess.TimeoutExpired:pass
  comparisons();audit_replay()
  if CHILD.returncode==0:
   assert read(OUT/'completed.json')['epochs']==100
   status('completed');return
  status('recovering',exit_code=CHILD.returncode,resume_step=step)


if __name__=='__main__':
 try:main()
 except Exception as error:
  status('failed',error_type=type(error).__name__,error=str(error));raise
