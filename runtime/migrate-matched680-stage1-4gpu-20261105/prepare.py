"""Checkpoint-preserving one-to-four GPU repartition for the live stage-1 run."""
import json,shutil,hashlib,sys,time,tarfile
from pathlib import Path
import torch
OLD=Path('/tmp/laser-var-stage1-20261105')
BASE=Path('/tmp/laser-var-stage1-4gpu-20261105')
HERE=Path(__file__).resolve().parent
OUT=Path('/workspace/Projects/laser/outputs/imnet-s1-laser-matched680-best24608-resume-20261105')
EXCLUDED={'output','wandb','workers','checkpoint_every','preview_every','source_bundle'}
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,default=str)+'\n')
def contract_sha(c):return hashlib.sha256(json.dumps({k:v for k,v in c.items() if k not in EXCLUDED},sort_keys=True).encode()).hexdigest()
def main():
 torch.set_num_threads(4)
 old_status=json.loads((OUT/'supervisor-status.json').read_text())
 assert old_status['phase']=='stopped','Wait for the single-GPU checkpointed stop'
 BASE.mkdir(exist_ok=True)
 runtime=BASE/'runtime';assert not runtime.exists(),'Do not overwrite a prepared runtime'
 shutil.copytree(OLD/'frozen/runtime',runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 shutil.copyfile(HERE/'driver.py',runtime/'four_gpu_driver.py')
 sys.path.insert(0,str(runtime))
 from migrate_stage1_world_size import global_indices,assert_equal,fresh_rank_rng
 c=json.loads((OLD/'run/config.json').read_text());source=OLD/'run/tokenizer-last.pt'
 p=torch.load(source,map_location='cpu',mmap=True,weights_only=False)
 assert p['world_size']==1 and len(p['rng'])==1 and p['contract_sha256']==contract_sha(c)
 source_receipt=json.loads((OLD/'run/checkpoint.json').read_text());assert sha(source)==source_receipt['checkpoint_sha256']
 current_progress=dict(p['progress']);assert current_progress['batch']%4==0
 new_progress=dict(current_progress,batch=current_progress['batch']//4)
 assert new_progress['epoch']*10009+new_progress['batch']==new_progress['step']
 before=global_indices(1281167,1,32,4,current_progress['batch'],current_progress['epoch'],c['seed'])
 after=global_indices(1281167,4,32,1,new_progress['batch'],new_progress['epoch'],c['seed'])
 assert before==after and all(len(x)==128 for x in after)
 c.update(accumulation=1,workers=4,source_bundle=str(BASE/'source.tar.gz'))
 c['continuation'].update(hardware_topology='4 RTX PRO 6000 Blackwell 96GB',baseline_inherited_same_hardware_and_protocol=False)
 best=json.loads((OLD/'run/best-reconstruction.json').read_text());best_source=OLD/'run/tokenizer-best-rfid.pt'
 best_payload=torch.load(best_source,map_location='cpu',mmap=True,weights_only=False)
 assert best_payload['world_size']==1 and len(best_payload['rng'])==1
 old_ages={k:sorted({int(s['step']) for s in p[k]['state'].values() if 'step' in s}) for k in ['optimizer','discriminator_optimizer']}
 seeds={rank:970001+c['seed']+current_progress['step']*4+rank for rank in [1,2,3]}
 c['migration']=dict(old_world_size=1,new_world_size=4,source_step=current_progress['step'],source_artifact=source_receipt['artifact'],source_checkpoint_sha256=sha(source),best_source_artifact=best['checkpoint']['artifact'],global_batch=128,microbatch=32,old_accumulation=4,new_accumulation=1,original_rank_rng_preserved=True,additional_rank_seeds=seeds,bitwise_trajectory_replay=False)
 c['runtime_sha256']['four_gpu_driver.py']=sha(runtime/'four_gpu_driver.py')
 for n,h in c['runtime_sha256'].items():assert sha(runtime/n)==h,n
 for folder in ['run','preflight']:
  cfg=dict(c,output=str(BASE/folder));local=BASE/folder;local.mkdir(exist_ok=True)
  rng=[p['rng'][0]]+[fresh_rank_rng(seeds[r],r) for r in [1,2,3]]
  migrated=dict(p,progress=new_progress,world_size=4,contract_sha256=contract_sha(cfg),rng=rng,topology_migration=c['migration'])
  path=local/'tokenizer-last.pt';torch.save(migrated,path)
  check=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
  for key in ['model','optimizer','discriminator','discriminator_optimizer','learning_dynamics']:assert_equal(p[key],check[key])
  assert_equal(p['rng'][0],check['rng'][0])
  best_progress=dict(best_payload['progress']);assert best_progress['batch']%4==0;best_progress['batch']//=4
  best_seeds={rank:970001+c['seed']+best_payload['progress']['step']*4+rank for rank in [1,2,3]}
  best_rng=[best_payload['rng'][0]]+[fresh_rank_rng(best_seeds[r],r) for r in [1,2,3]]
  migrated_best=dict(best_payload,progress=best_progress,world_size=4,contract_sha256=contract_sha(cfg),rng=best_rng,topology_migration=c['migration'])
  torch.save(migrated_best,local/'tokenizer-best-rfid.pt')
  check_best=torch.load(local/'tokenizer-best-rfid.pt',map_location='cpu',mmap=True,weights_only=False)
  for key in ['model','optimizer','discriminator','discriminator_optimizer']:assert_equal(best_payload[key],check_best[key])
  best_record=dict(best)
  best_record['checkpoint']=dict(best['checkpoint'],checkpoint_sha256=sha(local/'tokenizer-best-rfid.pt'),world_size=4,original_checkpoint_sha256=best['checkpoint']['checkpoint_sha256'],serialization_migrated=True)
  for name,value in [('config.json',cfg),('contract.json',{k:v for k,v in cfg.items() if k not in EXCLUDED}),('best-reconstruction.json',best_record),('initialization.json',dict(initialization='full_state_1_to_4_gpu_continuation',source_step=current_progress['step'],source_checkpoint_sha256=sha(source),world_size=4,tokens_per_image=680,vocabulary=4096)),('checkpoint.json',dict(progress=new_progress,world_size=4,checkpoint_sha256=sha(path),contract_sha256=contract_sha(cfg),full_state=True))]:write(local/name,value)
  for name in ['guard-decision.json','learning-dynamics.json']:
   if (OLD/'run'/name).exists():shutil.copyfile(OLD/'run'/name,local/name)
 report=dict(passed=True,source_step=current_progress['step'],source_artifact=source_receipt['artifact'],old_world_size=1,new_world_size=4,source_progress=current_progress,progress=new_progress,global_batch=128,microbatch_per_gpu=32,accumulation=1,updates_per_epoch=10009,model_and_both_optimizers_bit_exact=True,learning_dynamics_preserved=True,original_rank_rng_preserved=True,additional_rank_seeds=seeds,next_three_global_batches_identical=True,optimizer_ages=old_ages,stage='stage1',tokens_per_image=680,vocabulary=4096,stage2_enabled=False,bitwise_trajectory_replay=False,time=time.time())
 write(BASE/'migration-receipt.json',report);write(OUT/'four-gpu-migration.json',report)
 shutil.copyfile(BASE/'run/config.json',OUT/'config-4gpu.json')
 with tarfile.open(BASE/'source.tar.gz','w:gz') as archive:
  for path in runtime.rglob('*'):
   if path.is_file() and '__pycache__' not in path.parts:archive.add(path,arcname='runtime/'+str(path.relative_to(runtime)))
 print(json.dumps(report),flush=True)
if __name__=='__main__':main()
