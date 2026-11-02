"""Validate and supervise fresh, paired RQ/LASER ImageNet controls."""
import hashlib,importlib.util,json,math,os,shutil,signal,subprocess,sys,tarfile,time,traceback
from pathlib import Path
from omegaconf import OmegaConf
import psutil
BASE=Path('/tmp/laser-imagenet-compound512-rqmatched-lowbit-20261027')
OLD=Path('/tmp/laser-imagenet-compound512-audit-20261026')
SOURCE=BASE/'source';ORIGINAL=BASE/'original'
ORIGINAL_RUN='imagenet-original-rq1400m-matched16-4a100-20261027-r2'
GROUP='imagenet-rqmatched-lowbit-20261027'

def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2,default=str)+'\n');temporary.replace(path)

def status(phase,**values):
 r=dict(phase=phase,controller_pid=os.getpid(),updated_unix=time.time(),**values);write(BASE/'status.json',r);print(json.dumps(r),flush=True);return r

def environment(arm):
 source=ORIGINAL if arm=='rq' else SOURCE
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3' if arm=='rq' else '4,5,6,7',
  PYTHONPATH=':'.join(map(str,[source/'upstream',source/'runtime'])) if arm=='rq' else ':'.join(map(str,[source,source/'runtime'])),
  ORIGINAL_RQ_BASE=str(ORIGINAL),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
  LASER_ACCUMULATION='8',LASER_ENCODER_BATCH='16',LASER_DATA_WORKERS='4',LASER_COMPILE_BLOCKS='1',
  TORCHINDUCTOR_COMPILE_THREADS='2',TORCHINDUCTOR_CACHE_DIR=str(BASE/'inductor-cache'),TORCHINDUCTOR_FX_GRAPH_CACHE='1',
  TORCH_HOME=str(OLD/'torch-cache'),WANDB_BASE_URL='https://api.forge.coreweave.com',WANDB_RUN_GROUP=GROUP,
  WANDB_DIR=str(BASE/'wandb'),WANDB_CACHE_DIR=str(BASE/'wandb-cache'),WANDB_DATA_DIR=str(BASE/'wandb-data'),
  PYTORCH_ALLOC_CONF='expandable_segments:True',NCCL_NVLS_ENABLE='0',MPLBACKEND='Agg',LASER_CHECKPOINT_IMMUTABLE_FILES='1')
 env['WANDB_API_KEY']=Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
 return env

def command(arm,canary=False,resume=False):
 result=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=4']
 if arm=='rq':
  result += [str(ORIGINAL/'train.py'),'--microbatch','64','--epochs','16']
  if canary:result += ['--smoke-updates','2']
  if resume:result += ['--resume',str(ORIGINAL/'run/train/checkpoints/last.pt')]
 else:
  name='canary.yaml' if canary else ('resume.yaml' if resume else 'production.yaml')
  result += [str(SOURCE/'train.py'),'--config',str(BASE/name)]
 return result

def launch(arm,canary=False,resume=False):
 label=arm+('-canary' if canary else '-train');path=BASE/(label+'.log')
 with path.open('a') as log:
  child=subprocess.Popen(command(arm,canary,resume),cwd=ORIGINAL if arm=='rq' else SOURCE,env=environment(arm),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 return child

def checkpoint(arm,canary=False):
 if arm=='rq':return ORIGINAL/('run/smoke/checkpoints/last.pt' if canary else 'run/train/checkpoints/last.pt')
 return BASE/('canary/checkpoints/last.pt' if canary else 'laser/checkpoints/last.pt')

def validate_state(arm,canary=False):
 import torch
 torch.set_num_threads(4);path=checkpoint(arm,canary);p=torch.load(path.resolve(),map_location='cpu',weights_only=False,mmap=True)
 ages={int(v['step']) for v in p['optimizer']['state'].values()};step=p['global_step'];assert ages=={step},(arm,step,ages)
 assert len(p['rng_state_by_rank'])==4
 if canary:
  assert step==2
  state=p.get('model',p.get('state_dict'));assert state is not None
  assert all(torch.isfinite(t).all() for t in state.values() if t.is_floating_point())
  assert all(torch.isfinite(v).all() for s in p['optimizer']['state'].values() for k,v in s.items() if k!='step' and isinstance(v,torch.Tensor))
 scheduler=p['scheduler'];clock=scheduler['after']['last_epoch'] if arm=='rq' else scheduler['last_epoch'];assert clock==step
 report=dict(passed=True,step=step,adam_ages=sorted(ages),rng_ranks=4,scheduler_step=clock,checkpoint=str(path.resolve()))
 write(BASE/(arm+('-canary-state.json' if canary else '-live-state.json')),report);return report

def archive_sources(arm):
 source=ORIGINAL if arm=='rq' else SOURCE;record=BASE/('original-records' if arm=='rq' else 'records');record.mkdir(exist_ok=True)
 names=[]
 for p in source.rglob('*'):
  if p.is_file() and p.suffix in {'.py','.yaml','.json'} and not any(n in p.parts for n in ['run','wandb','uploads','torch-cache','__pycache__']):names.append(p)
 manifest={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in names};write(record/'source-manifest.json',manifest)
 with tarfile.open(record/'source.tar.gz','w:gz') as archive:
  for p in names:archive.add(p,arcname=str(p.relative_to(source)))
 return record

def prepare(report):
 candidates=[r for r in report['results'] if r['bins']<512 and r['full_gate_passed']]
 if not candidates:raise RuntimeError('No lower-bit codec passed full50k reconstruction gate')
 selected=min(candidates,key=lambda r:r['bins']);bins=selected['bins'];assert abs(next(r['rfid'] for r in report['results'] if r['bins']==512)-4.215633109816565)<.03
 run=f'imagenet-compound{bins}-rqmatched16-4a100-20261027-r2'
 cfg=OmegaConf.load(OLD/'quality/production.yaml');cfg.audit_directory=str(BASE/'records');o=cfg.options
 updates=dict(output=str(BASE/'laser/train'),checkpoint_dir=str(BASE/'laser/checkpoints'),coeff_vocab_size=bins,lr=.0005,lr_schedule='cosine',warmup_epochs=0.,epochs=16,lr_schedule_epochs=100,seed=261001,resume=False,fid_every=2,fid_early_epochs=1,fid_seed=261027,sample_grid_seed=261027,atom_temperature=1.,atom_top_k=0,atom_top_p=1.,coeff_temperature=1.,coeff_top_k=0,coeff_top_p=1.,wandb_id=run,wandb_name=f'LASER compound{bins} | matched LR5e-4 no warmup | BF16 | fresh16')
 for k,v in updates.items():o[k]=v
 for k in ['coefficient_head','joint_pair_top_k','joint_pair_top_p','joint_pair_temperature','resume_checkpoint','init_stage2_checkpoint']:
  if k in o:del o[k]
 OmegaConf.save(cfg,BASE/'production.yaml');OmegaConf.save(cfg,SOURCE/'configs/matched-production.yaml')
 recovery=OmegaConf.create(OmegaConf.to_container(cfg));recovery.options.resume=True;OmegaConf.save(recovery,BASE/'resume.yaml')
 canary=OmegaConf.create(OmegaConf.to_container(cfg));canary.options.update(output=str(BASE/'canary/train'),checkpoint_dir=str(BASE/'canary/checkpoints'),wandb_mode='disabled',upload_checkpoints=False,fid_every=0,fid_early_epochs=0,sample_grid_every=0,save_step_freq=0,max_optimizer_steps=2);OmegaConf.save(canary,BASE/'canary.yaml')
 protocol=dict(group=GROUP,original_run=ORIGINAL_RUN,laser_run=run,selected=selected,original_stage1_sha256='e8eb312e6c0dbd8e490ca95f69e3c3b417162625ff9320bbfff791cbf832b694',laser_stage1_sha256='dd28db9306d526bdc9fbf8016403e106c4f317fd8f5c04792f0953e53310fdab',training=dict(lr=.0005,warmup_epochs=0,adam_betas=[.9,.95],weight_decay=.0001,clip=1.,global_batch=2048,microbatch=64,accumulation=8,world_size=4,hardware='4 A100 SXM4 80GB per arm',precision='BF16',teacher_precision='FP32 TF32 disabled',schedule_epochs=100,optimizer_updates_per_epoch=626,execute_epochs=16,execute_steps=10016,training_seed=261001,augmentation='Resize256 RandomCrop256 HorizontalFlip.5 Normalize.5; fresh online views'),generation=dict(temperature=1.,top_k=0,top_p=1.,samples=50000,samples_per_class=50,seed=261027,batch_per_rank=64,reference_images=1281167,reference_split='train',reference_sha256='3f9c92d15755e76ec312964a819e3e19b9cf3aadc618a7ae99f5e8aa96501260',cfg=False,ema=False),targets=dict(rq='native stochastic RQ codes and upstream soft targets T.5',laser='deterministic frozen OMP supports and hard nearest lower-bit coefficients'),sampling_note='At T1 without truncation, atom draw followed by its conditional coefficient samples the exact joint pair distribution. No fieldwise top-k/top-p filters.',nominal_bits_per_image=dict(rq=3584,laser=selected['nominal_bits_per_image'],laser_previous512=5888),fresh_weights_and_empty_adam_for_both=True,canary_weights_reused=False,compilation=dict(rq='native eager',laser='verified compiled attention blocks'),limitations=['One training seed.','Different total parameters and nominal code rates are reported method differences.','Native eager and compiled kernel implementations differ; no equal-compute/throughput claim.','No claim of lower generation FID before matched evaluation.'],preserved_previous_run_step=22800)
 write(BASE/'protocol.json',protocol);write(BASE/'selection.json',dict(selected=selected,selection='smallest bin count passing full50k paired gate: rFID<4.4 and delta vs512<.1'))
 records=BASE/'records';records.mkdir(exist_ok=True);shutil.copyfile(BASE/'production.yaml',records/'recipe.yaml');shutil.copyfile(BASE/'protocol.json',records/'audit.json');shutil.copyfile(BASE/'tests.xml',records/'tests.xml');write(records/'quantization.json',report)
 # Native driver publishes only this local reproducibility directory to W&B.
 reproducibility=ORIGINAL/'run/reproducibility';reproducibility.mkdir(parents=True,exist_ok=True)
 for name in ['protocol.json','original-runtime-changes.json','laser-runtime-changes.json']:shutil.copyfile(BASE/name,reproducibility/name)
 return protocol

def data_parity():
 sys.path[:0]=[str(SOURCE),str(SOURCE/'runtime'),str(ORIGINAL/'upstream')]
 from src.training.rq_imagenet_runtime import TarImageNet
 from src.training.rqtransformer import image_transform
 from rqvae.img_datasets.transforms import create_transforms
 spec=importlib.util.spec_from_file_location('matched_original_data',ORIGINAL/'runtime/original_rq_imagenet_runtime.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 cfg=OmegaConf.load(ORIGINAL/'inputs/stage2-training-config.yaml');a=TarImageNet(OLD/'data','train',image_transform(),seed=261001);b=module.TarImageNet(OLD/'data','train',create_transforms(cfg.dataset,split='train'),seed=261001)
 import torch,numpy as np
 assert np.array_equal(a.labels,b.labels) and np.array_equal(a.offsets,b.offsets)
 probes=[]
 for epoch in [0,1,15]:
  a.set_epoch(epoch);b.set_epoch(epoch)
  for index in [0,1701,1281166]:
   x,y=a[index];u,v=b[index];assert y==v and torch.equal(x,u);probes.append(dict(epoch=epoch,index=index,label=y,sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest()))
 write(BASE/'data-parity.json',dict(passed=True,probes=probes,identical_indexes=True,identical_per_image_views=True))

def main():
 os.environ.update({k:v for k,v in environment('laser').items() if k.startswith('WANDB_')});(BASE/'wandb').mkdir(exist_ok=True)
 import wandb
 wb=wandb.init(entity='helloimlixin-rutgers',project='laser',id=GROUP,name='Matched RQ vs LASER | lower-bit preflight and paired training',group=GROUP,dir=str(BASE/'wandb'),resume='allow')
 write(BASE/'wandb.json',dict(url=wb.url,id=wb.id));children={}
 try:
  status('preserving_previous_checkpoint_in_cloud')
  if not (BASE/'previous-checkpoint-cloud.json').exists():
   artifact=wandb.Artifact('imagenet-compound512-before-matched-comparison',type='training-checkpoint',metadata=dict(global_step=22800,optimizer_and_rng=True,origin_run='imagenet-compound512-audited-8a100-20261026'))
   artifact.add_file(str(BASE/'previous-production-checkpoint.pt'),name='last.pt');uploaded=wb.log_artifact(artifact,aliases=['step22800']);uploaded.wait();write(BASE/'previous-checkpoint-cloud.json',dict(artifact=uploaded.qualified_name,version=uploaded.version,step=22800))
  status('previous_checkpoint_cloud_verified',receipt=json.loads((BASE/'previous-checkpoint-cloud.json').read_text()))
  while not (BASE/'full50k/result.json').exists():status('waiting_for_full50k_quantization');time.sleep(10)
  report=json.loads((BASE/'full50k/result.json').read_text());assert report['samples']==50000 and report['token_codec_parity_passed'];protocol=prepare(report);data_parity()
  for arm in ['rq','laser']:archive_sources(arm)
  wb.config.update(protocol,allow_val_change=True);wb.summary.update(dict(selected_coefficient_bins=protocol['selected']['bins'],selected_reconstruction_rfid=protocol['selected']['rfid'],quantization_delta_rfid=protocol['selected']['delta_rfid_vs_512'],original_run=ORIGINAL_RUN,laser_run=protocol['laser_run']))
  for name in ['protocol.json','selection.json','data-parity.json','tests.xml','original-runtime-changes.json','laser-runtime-changes.json','previous-run-preservation-local.json']:wb.save(str(BASE/name),base_path=str(BASE),policy='now')
  wb.save(str(BASE/'full50k/result.json'),base_path=str(BASE),policy='now')
  for arm in ['rq','laser']:children[arm]=launch(arm,canary=True)
  while any(c.poll() is None for c in children.values()):
   failures={a:c.returncode for a,c in children.items() if c.poll() not in [None,0]}
   if failures:raise RuntimeError(f'GPU canary failed: {failures}')
   status('gpu_canaries',children={a:c.pid for a,c in children.items()});time.sleep(10)
  canaries={a:validate_state(a,canary=True) for a in children};write(BASE/'records/canary.json',dict(passed=True,arms=canaries,production_starts_fresh=True))
  shutil.copyfile(BASE/'records/canary.json',ORIGINAL/'run/reproducibility/canary.json')
  wb.save(str(BASE/'records/canary.json'),base_path=str(BASE),policy='now')
  shutil.rmtree(BASE/'canary/checkpoints');shutil.rmtree(ORIGINAL/'run/smoke/checkpoints')
  for arm in ['rq','laser']:children[arm]=launch(arm)
  verified=set();pinned=set()
  while any(c.poll() is None for c in children.values()):
   rows={}
   for arm,c in children.items():
    code=c.poll()
    if code not in [None,0]:raise RuntimeError(f'{arm} training exited {code}')
    metrics=ORIGINAL/'run/train/metrics.jsonl' if arm=='rq' else BASE/'laser/train/metrics.jsonl'
    if metrics.exists():
     lines=metrics.read_text().splitlines()
     if lines:rows[arm]=json.loads(lines[-1])
    receipt=ORIGINAL/'run/train/checkpoint.json' if arm=='rq' else BASE/'laser/checkpoints/latest.json'
    if receipt.exists():
     saved=json.loads(receipt.read_text());step=saved['global_step']
     if arm not in verified and step>=100:validate_state(arm);verified.add(arm)
     if step in [6260,10016] and (arm,step) not in pinned:
      target=BASE/'retained'/f'{arm}-step{step}.pt';target.parent.mkdir(exist_ok=True);os.link(checkpoint(arm).resolve(),target);pinned.add((arm,step))
   status('training',children={a:c.pid for a,c in children.items()},runs={'rq':ORIGINAL_RUN,'laser':protocol['laser_run']},latest_metrics=rows,live_checkpoint_verified=sorted(verified))
   time.sleep(10)
  for arm,c in children.items():assert c.returncode==0
  status('completed');wb.summary['status']='completed';wb.finish()
 except BaseException as error:
  status('failed',error=repr(error),traceback=traceback.format_exc())
  for c in children.values():
   if c.poll() is None:
    try:os.killpg(c.pid,signal.SIGTERM)
    except ProcessLookupError:pass
  wb.summary['status']='failed';wb.summary['error']=repr(error);wb.finish(exit_code=1);raise
if __name__=='__main__':main()
