"""Matched noise ablation, followed by full-GPU continuation of the FID winner."""
import fcntl,json,os,subprocess,sys,time,signal,math,shutil,statistics,hashlib
from pathlib import Path
from environment import *
from omegaconf import OmegaConf
CHILDREN=[]
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.pending');q.write_text(json.dumps(d,indent=2,default=str)+'\n');q.replace(p)
def status(phase,**kw):write(OUT/'status.json',dict(phase=phase,supervisor_pid=os.getpid(),time=time.time(),**kw))
def spawn(command,env,log,cwd=BASE):
 log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
 with log.open('ab',buffering=0) as f:p=subprocess.Popen(command,env=env,cwd=cwd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 CHILDREN.append(p);return p
def torchrun(script,world,extra):return [PYTHON,'-u','-m','torch.distributed.run','--standalone',f'--nproc-per-node={world}',str(script),*extra]
def stopped(signum,frame):
 status('stopping',signal=signum)
 for p in CHILDREN:
  if p.poll() is None:
   try:os.killpg(p.pid,signal.SIGTERM)
   except ProcessLookupError:pass
 raise SystemExit(128+signum)
def wait(p,phase,**kw):
 while p.poll() is None:
  status(phase,child_pid=p.pid,**kw);time.sleep(15)
 if p.returncode:raise RuntimeError(f'{phase} failed with exit code {p.returncode}')
def setup_arm(name,parameters):
 base=BASE/'arms'/name;out=OUT/'arms'/name;out.mkdir(parents=True,exist_ok=True)
 run='imagenet-beam451-'+name+'-scratch100-8h100-20261108'
 base.mkdir(parents=True,exist_ok=True)
 for folder in ['source','upstream','runtime','data','torch-cache']:
  if not (base/folder).exists():(base/folder).symlink_to(BASE/folder,target_is_directory=True)
 for folder in ['inputs','wandb','checkpoints']:(base/folder).mkdir(exist_ok=True)
 # This symlink intentionally resolves the script's root through LASER_BEAM_BASE.
 shutil.copyfile(BASE/'train.py',base/'train.py')
 for p in (BASE/'inputs').iterdir():
  if p.is_file() and p.name not in ['recipe.yaml','execution.json','stage2-training-config.yaml'] and not (base/'inputs'/p.name).exists():(base/'inputs'/p.name).hardlink_to(p)
 recipe=OmegaConf.load(BASE/'inputs/recipe.yaml');recipe.run_id=run
 recipe.name='ImageNet | beam451 | '+name+' | matched noise trial -> 8H100'
 recipe.targets.sequence_temperature=parameters['sequence_temperature']
 recipe.targets.max_relative_candidate_error=parameters['max_relative_candidate_error']
 recipe.targets.calibration='2048 fixed training views; measured sequence entropy and expected excess distortion; matched FID50k trial'
 recipe.targets.max_mean_relative_latent_error=.01 if name=='cutoff05' else .001
 recipe.epochs=recipe.lr_schedule_epochs=100
 recipe.microbatch=128
 recipe.lineage.source_run='helloimlixin-rutgers/laser/imagenet-beam451-aligned-joint16k-scratch100-4h100-20261107'
 recipe.lineage.noise_trial=name
 if name!='control':recipe.tokenizer.teacher_sampled_rfid=None
 OmegaConf.save(recipe,base/'inputs/recipe.yaml')
 config=OmegaConf.load(BASE/'inputs/stage2-training-config.yaml');config.experiment.batch_size=128;config.experiment.epochs=100;OmegaConf.save(config,base/'inputs/stage2-training-config.yaml')
 execution=read(BASE/'inputs/execution.json');execution.update(world_size=4,microbatch=128,accumulation_steps=4,hardware='4 NVIDIA H100 80GB HBM3; winner scales to 8',teacher_sequence_temperature=parameters['sequence_temperature'],teacher_max_relative_candidate_error=parameters['max_relative_candidate_error'],ablation='noise only; same model initialization, data order, global batch, optimizer, schedule and sampler',pilot_epochs=2)
 write(base/'inputs/execution.json',execution)
 repro=out/'reproducibility';repro.mkdir(exist_ok=True)
 for p in (base/'inputs').iterdir():
  if p.is_file() and p.stat().st_size<100000:shutil.copyfile(p,repro/p.name)
 write(out/'plan.json',dict(run_id=run,parameters=parameters,global_batch=2048,pilot_gpus=4,pilot_epochs=2,continuation_gpus=8,continuation_epochs=100,full_state_checkpoint=True))
 return dict(name=name,run_id=run,base=base,out=out,parameters=parameters)
def launch_arm(arm,devices,epochs,microbatch=128,receipt=None):
 base,out=arm['base'],arm['out'];command=torchrun(base/'train.py',len(devices.split(',')),['--epochs',str(epochs),'--microbatch',str(microbatch)])
 if receipt:command+=['--resume',str(receipt['path'] if Path(receipt['path']).exists() else receipt['persistent_path'])]
 p=spawn(command,environment(base,out,devices),out/'training.log',base)
 write(out/'launch.json',dict(command=command,pid=p.pid,devices=devices,epochs=epochs,resume_step=receipt['global_step'] if receipt else 0,time=time.time()))
 return p
def preflight_layout(arm,world,devices,target,checkpoint=None):
 output=BASE/'benchmarks'/target;output.mkdir(parents=True,exist_ok=True);env=environment(arm['base'],arm['out'],devices);env['LASER_PRIVATE_SMOKE_DIR']=str(output)
 args=['--epochs','100','--microbatch','128','--smoke-updates',str(2 if not checkpoint else checkpoint['global_step']+2)]
 if checkpoint:args+=['--resume',checkpoint['path']]
 p=spawn(torchrun(arm['base']/'train.py',world,args),env,BASE/'benchmarks'/(target+'.log'),arm['base']);wait(p,'preflight_'+target)
 proof=read(output/'verification.json');assert proof['passed'] and proof['frozen_tokenizer_unchanged']
 return read(arm['base']/'smoke/checkpoint.json')
def main():
 lock=(BASE/'optimization-supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,stopped)
 while not (BASE/'data/ready.json').exists():status('staging_verified_imagenet');time.sleep(15)
 assert read(BASE/'data/ready.json')['training_images']==1281167
 # The first eight-GPU benchmark is started interactively and must finish cleanly.
 while not (BASE/'benchmarks/mb128/verification.json').exists():
  pid=int((BASE/'benchmarks/mb128.pid').read_text())
  if not Path(f'/proc/{pid}').exists():raise RuntimeError('Initial eight-GPU benchmark exited without verification')
  status('benchmark_mb128',child_pid=pid);time.sleep(15)
 assert read(BASE/'benchmarks/mb128/verification.json')['passed']
 env=environment();env['LASER_CALIBRATION_REPORT']=str(OUT/'noise-calibration.json')
 if not (OUT/'noise-calibration.json').exists():
  p=spawn(torchrun(HERE/'calibrate.py',8,[]),env,OUT/'noise-calibration.log');wait(p,'calibrating_noise')
 calibration=read(OUT/'noise-calibration.json')
 assert calibration['training_images']==2048 and not calibration['validation_used']
 # Compare the attainable exact 64-image microbatch with 128; no altered global batch.
 bench=BASE/'benchmarks/mb64';bench.mkdir(exist_ok=True);env=environment();env['LASER_PRIVATE_SMOKE_DIR']=str(bench)
 if not (bench/'verification.json').exists():
  p=spawn(torchrun(BASE/'train.py',8,['--microbatch','64','--epochs','100','--smoke-updates','6']),env,BASE/'benchmarks/mb64.log');wait(p,'benchmark_mb64')
 rows=[]
 for mb in [64,128]:
  folder=BASE/f'benchmarks/mb{mb}';proof=read(folder/'verification.json');assert proof['passed']
  values=[]
  for rank in range(8):
   values.append(statistics.median([json.loads(l)['images_per_second'] for l in (folder/f'throughput-rank{rank}.jsonl').read_text().splitlines() if json.loads(l)['step']>=3]))
  rows.append(dict(microbatch=mb,images_per_second=min(values),peak_allocated_gib=max(read(folder/f'progress-rank{r}.json')['peak_allocated_gib'] for r in range(8))))
 selected_mb=max(rows,key=lambda r:r['images_per_second'])['microbatch'];write(OUT/'throughput-benchmarks.json',dict(rows=rows,selected_microbatch=selected_mb,world_size=8,global_batch=2048))
 arms=[setup_arm('control',calibration['baseline']),setup_arm('cutoff05',calibration['selected'])]
 # Validate changed targets and a full optimizer/scheduler/RNG 4->8 GPU restore.
 receipt=preflight_layout(arms[1],4,'0,1,2,3','noise4')
 preflight_layout(arms[1],8,'0,1,2,3,4,5,6,7','noise8-resume',receipt)
 for arm in arms:
  (arm['out']/'reproducibility/noise-calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
  # Smoke is isolated; no smoke weights are loaded into the real experiment.
  arm['process']=launch_arm(arm,'0,1,2,3' if arm['name']=='control' else '4,5,6,7',2)
 retries={a['name']:0 for a in arms}
 while True:
  progress={}
  for arm in arms:
   f=arm['base']/'live-training/train/progress-rank0.json'
   if f.exists():progress[arm['name']]=read(f)
   p=arm['process']
   if p.poll() is not None and p.returncode!=0:
    receipt=arm['out']/'checkpoint.json'
    if not receipt.exists() or retries[arm['name']]>=2:raise RuntimeError(f"Pilot {arm['name']} failed; inspect its training.log")
    retries[arm['name']]+=1;arm['process']=launch_arm(arm,'0,1,2,3' if arm['name']=='control' else '4,5,6,7',2,receipt=read(receipt))
  status('matched_noise_trials',world_size=8,allocation='4 GPUs per arm',progress=progress,runs=[dict(name=a['name'],run_id=a['run_id'],url='https://wandb.ai/helloimlixin-rutgers/laser/runs/'+a['run_id']) for a in arms])
  if all(a['process'].poll()==0 for a in arms):break
  time.sleep(15)
 results=[]
 for arm in arms:
  row=read(arm['out']/'evaluations/epoch002.json');assert row['generated_images']==50000 and row['real_images']==1281167
  results.append(dict(name=arm['name'],fid=row['eval/fid'],inception_score=row['eval/inception_score'],run_id=arm['run_id']))
 best=min(results,key=lambda r:r['fid']);winner=next(a for a in arms if a['name']==best['name'])
 # completed.json means the two-epoch pilot ended; archive it before continuation.
 p=winner['out']/'completed.json';p.rename(winner['out']/'pilot-completed.json')
 selection=dict(results=results,selected=winner['name'],selection_metric='epoch2 FID50k, matched sampler and full training reference',early_fid_does_not_establish_final_superiority=True,continuation_epochs=100,continuation_world_size=8,selected_microbatch=selected_mb,time=time.time())
 write(OUT/'selection.json',selection);write(OUT/'wandb.json',dict(id=winner['run_id'],url='https://wandb.ai/helloimlixin-rutgers/laser/runs/'+winner['run_id']))
 execution=read(winner['base']/'inputs/execution.json');execution.update(world_size=8,microbatch=selected_mb,accumulation_steps=math.ceil(2048/(8*selected_mb)),hardware='8 NVIDIA H100 80GB HBM3',selected_from=results);write(winner['base']/'inputs/execution.json',execution)
 failed_steps={}
 while True:
  receipt=read(winner['out']/'checkpoint.json');step=receipt['global_step'];failed_steps[step]=failed_steps.get(step,0)+1
  if failed_steps[step]>3:raise RuntimeError('Repeated failure from the same complete checkpoint')
  p=launch_arm(winner,'0,1,2,3,4,5,6,7',100,selected_mb,receipt)
  while p.poll() is None:
   f=winner['base']/'live-training/train/progress-rank0.json'
   status('training_selected_recipe',world_size=8,run_id=winner['run_id'],selection=selection,progress=read(f) if f.exists() else None);time.sleep(15)
  if p.returncode==0:
   assert (winner['out']/'completed.json').exists();status('completed',selection=selection,result=read(winner['out']/'completed.json'));return
  status('recovering_selected_recipe',exit_code=p.returncode,checkpoint_step=step)
if __name__=='__main__':
 try:main()
 except Exception as e:
  # Stop remaining trial ranks if one arm cannot recover; avoid silent resource use.
  for p in CHILDREN:
   if p.poll() is None:
    try:os.killpg(p.pid,signal.SIGTERM)
    except ProcessLookupError:pass
  status('failed',error=str(e));raise
