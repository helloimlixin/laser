"""Four-GPU full-state continuation of the existing matched-budget stage-1 run."""
import argparse,json,os,shutil,sys,time
from pathlib import Path
BASE=Path('/tmp/laser-var-stage1-4gpu-20261105')
RUNTIME=BASE/'runtime'
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(RUNTIME)]
import torch
import torch.distributed as dist
import train_matched_stage1 as trainer
from dynamics import initial_state,observe


def broadcast(value):
 objects=[value if dist.get_rank()==0 else None]
 dist.broadcast_object_list(objects,src=0)
 return objects[0]


def freeze_factory():
 original=trainer.FourScaleJointVAE
 class FrozenTokenizerVAE(original):
  def __init__(self,*args,**kwargs):
   super().__init__(*args,**kwargs)
   self.encoder.requires_grad_(False);self.quant_conv.requires_grad_(False);self.quantize.requires_grad_(False)
 trainer.FourScaleJointVAE=FrozenTokenizerVAE


class DistributedContinuation(trainer.Experiment):
 def __init__(self,cfg,smoke=False):
  self.learning_dynamics=initial_state();self.quality_stop=None
  super().__init__(cfg,smoke)
  assert self.world==4 and self.micro==32 and self.accum==1 and self.global_batch==128 and self.updates==10009
  self.recoveries=self.learning_dynamics.get('recoveries',0)
  assert not any(p.requires_grad for p in self.model.encoder.parameters())
  assert not any(p.requires_grad for p in self.model.quantize.parameters())
  dist.barrier()
  if not smoke:
   if self.rank==0:
    self.run.summary.update({'training/stage':'stage1','stage2/enabled':False,
     'training/world_size':4,'training/microbatch_per_gpu':32,'training/accumulation':1,
     'training/global_batch':128,'training/target_epochs':'until explicitly stopped',
     'initialization/trainable_weights':'full-state continuation across GPU topology change',
     'initialization/fresh_optimizer':False,'tokenizer/encoder_frozen':True,'tokenizer/quantizer_frozen':True,
     'training/hardware':'4 NVIDIA RTX PRO 6000 Blackwell 96GB',
     'guard/action_on_regression':'restore_best_reduce_lr_continue',
     'training/optimizer_age_equals_stream_step_after_rewind':False,
     'migration/source_step':cfg['migration']['source_step'],
     'migration/original_rank_rng_preserved':True,'migration/bitwise_trajectory_replay':False})
    self.run.use_artifact(cfg['migration']['source_artifact'])
    import wandb
    code=wandb.Artifact(cfg['wandb']['id']+'-4gpu-runtime',type='code')
    for p in HERE.glob('*.py'):code.add_file(str(p),name=p.name)
    code.add_file(str(BASE/'migration-receipt.json'));code.add_file(str(BASE/'preflight-verification.json'))
    published=self.run.log_artifact(code);published.wait()
    trainer.atomic(self.out/'four-gpu-runtime-artifact.json',dict(artifact=published.qualified_name,state=str(published.state)))
    self.publish_migrated_best()
   dist.barrier()
   self.save(publish=True)
   self.log('migration_complete',step=self.progress['step'],world_size=4,microbatch=32,accumulation=1,
            global_batch=128,tokens_per_image=680,optimizer_preserved=True)

 def log(self,phase,**values):
  super().log(phase,**values)
  if self.rank==0:
   row=json.dumps(dict(phase=phase,time=time.time(),**values))+'\n'
   with (self.out/'metrics.jsonl').open('a') as f:f.write(row)
   if not self.smoke:
    self.mirror('status.json')
    with (Path(self.cfg['continuation']['durable'])/'metrics.jsonl').open('a') as f:f.write(row)

 def mirror(self,name):
  if self.smoke or self.rank!=0:return
  source=self.out/name
  if source.exists():
   target=Path(self.cfg['continuation']['durable'])/name
   temporary=target.with_suffix(target.suffix+'.4gpu-pending')
   shutil.copyfile(source,temporary);temporary.replace(target)

 def restore(self):
  super().restore()
  saved=torch.load(self.out/'tokenizer-last.pt',map_location='cpu',mmap=True,weights_only=False)
  self.learning_dynamics=dict(saved.get('learning_dynamics',initial_state()))

 def save(self,publish=False):
  super().save(publish)
  if publish and self.rank==0:
   for name in ['tokenizer-last.pt','checkpoint.json','joint-codebook.pt']:self.mirror(name)

 def publish_migrated_best(self):
  marker=self.out/'migrated-best-published.json'
  if marker.exists():return
  best=json.loads((self.out/'best-reconstruction.json').read_text())
  path=self.out/'tokenizer-best-rfid.pt'
  saved=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
  import wandb
  receipt=dict(progress=saved['progress'],checkpoint_sha256=trainer.sha(path),bytes=path.stat().st_size,
    full_state=True,world_size=4,contract_sha256=self.contract_sha,tokens_per_image=680,vocabulary=4096,
    inherited_rfid=best['rfid'],metric_is_new_measurement=False,source_artifact=self.cfg['migration']['best_source_artifact'])
  artifact=wandb.Artifact(self.cfg['wandb']['id']+'-tokenizer',type='model',metadata=receipt)
  artifact.add_file(str(path),name='checkpoint.pt',policy='immutable');artifact.add_file(str(self.out/'config.json'))
  published=self.run.log_artifact(artifact,aliases=['best-rfid','topology-4gpu-best']);published.wait()
  receipt['artifact']=published.qualified_name;best['checkpoint']=receipt
  trainer.atomic(self.out/'best-reconstruction.json',best);trainer.atomic(marker,receipt)
  self.run.summary.update({'reconstruction/best_rfid':best['rfid'],'reconstruction/best_artifact':published.qualified_name})
  self.mirror('best-reconstruction.json');self.mirror('tokenizer-best-rfid.pt')

 def preserve_best(self):
  assert self.rank==0
  temporary=self.out/'tokenizer-best-rfid.pending'
  shutil.copyfile(self.out/'tokenizer-last.pt',temporary);temporary.replace(self.out/'tokenizer-best-rfid.pt')
  best=json.loads((self.out/'best-reconstruction.json').read_text())
  if self.run:
   import wandb
   a=wandb.Api().artifact(best['checkpoint']['artifact'])
   if 'best-rfid' not in a.aliases:a.aliases.append('best-rfid');a.save()
   self.run.summary['reconstruction/best_artifact']=best['checkpoint']['artifact']
  self.mirror('best-reconstruction.json');self.mirror('tokenizer-best-rfid.pt')

 def evaluate(self):
  previous=broadcast(json.loads((self.out/'best-reconstruction.json').read_text())['rfid'] if self.rank==0 else None)
  self.log('evaluation_started',step=self.progress['step'],count=50000,world_size=4)
  started=time.monotonic()
  super().evaluate()
  result=broadcast(json.loads((self.out/f"reconstruction-epoch{self.progress['epoch']:03d}.json").read_text()) if self.rank==0 else None)
  assert result['count']==result['reference_count']==50000
  self.learning_dynamics,decision=observe(self.cfg['continuation'],self.learning_dynamics,previous,result['rfid'])
  if decision['action'].startswith('stop_'):self.stop=True;self.quality_stop=decision['action']
  self.save(publish=True)
  if self.rank==0:
   trainer.atomic(self.out/'learning-dynamics.json',self.learning_dynamics)
   trainer.atomic(self.out/'guard-decision.json',dict(step=self.progress['step'],**decision))
   trainer.atomic(self.out/f"reconstruction-step{self.progress['step']:09d}.json",result)
   if result['rfid']<previous:
    trainer.atomic(self.out/'best-reconstruction.json',dict(result,checkpoint=json.loads((self.out/'checkpoint.json').read_text())))
    self.preserve_best()
   for name in ['learning-dynamics.json','guard-decision.json',f"reconstruction-step{self.progress['step']:09d}.json"]:self.mirror(name)
   if self.run:self.run.summary.update({'guard/action':decision['action'],'guard/lr_factor':self.learning_dynamics['lr_factor'],'guard/nonimproving_checks':self.learning_dynamics['nonimproving_checks']})
  self.log('evaluation_completed',step=self.progress['step'],rfid=result['rfid'],seconds=time.monotonic()-started,
           **{k:v for k,v in decision.items() if k!='score'})
  dist.barrier()

 def recover(self,reason):
  best=broadcast(json.loads((self.out/'best-reconstruction.json').read_text()) if self.rank==0 else None)
  saved=torch.load(self.out/'tokenizer-best-rfid.pt',map_location='cpu',mmap=True,weights_only=False)
  assert len(saved['rng'])==self.world
  previous=dict(self.progress);factor=self.learning_dynamics['lr_factor']
  self.model.load_state_dict(saved['model']);self.discriminator.load_state_dict(saved['discriminator'])
  self.optimizer.load_state_dict(saved['optimizer']);self.d_optimizer.load_state_dict(saved['discriminator_optimizer'])
  if saved.get('record_hit') is not None:self.model.quantize.record_hit=saved['record_hit']
  trainer.restore_rng(saved['rng'][self.rank])
  self.learning_dynamics=dict(saved.get('learning_dynamics',initial_state()))
  self.learning_dynamics.update(lr_factor=max(.01,factor*.5),nonimproving_checks=0,recoveries=self.recoveries+1)
  self.recoveries+=1;self.stop=False;self.quality_stop=None
  self.optimizer.zero_grad(set_to_none=True);self.d_optimizer.zero_grad(set_to_none=True)
  assert self.progress==previous
  receipt=dict(reason=reason,stream_progress=previous,restored_weight_step=saved['progress']['step'],best_rfid=best['rfid'],
    lr_factor=self.learning_dynamics['lr_factor'],recoveries=self.recoveries,
    optimizer_age=sorted({int(v['step']) for v in self.optimizer.state.values() if 'step' in v}),
    data_cursor_preserved=True,all_rank_rng_restored=True,time=time.time())
  if self.rank==0:
   trainer.atomic(self.out/'recovery.json',receipt);self.mirror('recovery.json')
   if self.run:self.run.summary.update({'training/status':'running','recovery/count':self.recoveries,'recovery/best_rfid':best['rfid'],'recovery/lr_factor':self.learning_dynamics['lr_factor']})
  self.log('automatic_recovery',**{k:v for k,v in receipt.items() if k!='time'})
  dist.barrier()

 def train_loop(self,max_steps=None):
  pending=broadcast(json.loads((self.out/'guard-decision.json').read_text()) if self.rank==0 and (self.out/'guard-decision.json').exists() else None)
  if pending and pending['action'].startswith('stop_') and pending['step']==self.progress['step']:self.recover('restart_pending_quality_recovery')
  while True:
   super().train_loop(max_steps)
   if self.quality_stop:self.recover(self.quality_stop)
   else:return


def verify_replicas(experiment):
 maximum=torch.zeros((),device=experiment.device)
 with torch.no_grad():
  for model in [experiment.model,experiment.discriminator]:
   for p in model.parameters():
    reference=p.detach().clone();dist.broadcast(reference,src=0)
    maximum=torch.maximum(maximum,(reference-p).abs().max())
 dist.all_reduce(maximum,op=dist.ReduceOp.MAX)
 assert float(maximum)==0.,float(maximum)
 return float(maximum)


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);p.add_argument('--smoke',action='store_true');args=p.parse_args()
 cfg=json.loads(args.config.read_text());torch.set_num_threads(4)
 trainer.var_dist.initialize(backend='nccl',timeout=120);freeze_factory()
 if args.smoke:trainer.SmokeImages=lambda _:trainer.ArchiveImages(cfg['train_archive'],cfg['train_index'],True,cfg['seed'])
 experiment=None
 try:
  experiment=DistributedContinuation(cfg,args.smoke)
  if args.smoke:
   source_step=experiment.progress['step']
   old_g={k:int(v['step']) for k,v in experiment.optimizer.state_dict()['state'].items() if 'step' in v}
   old_d={k:int(v['step']) for k,v in experiment.d_optimizer.state_dict()['state'].items() if 'step' in v}
   trainable=[parameter.requires_grad for parameter in experiment.model.parameters()]
   frozen={n:p.detach().cpu().clone() for n,p in experiment.model.named_parameters() if not p.requires_grad}
   experiment.train_loop(source_step+2)
   assert experiment.progress['step']==source_step+2
   for n,parameter in experiment.model.named_parameters():
    if n in frozen:assert torch.equal(parameter.detach().cpu(),frozen[n]),n
   new_g={k:int(v['step']) for k,v in experiment.optimizer.state_dict()['state'].items() if 'step' in v}
   new_d={k:int(v['step']) for k,v in experiment.d_optimizer.state_dict()['state'].items() if 'step' in v}
   expected_g={k:v+(2 if trainable[k] else 0) for k,v in old_g.items()}
   assert new_g==expected_g and new_d=={k:v+2 for k,v in old_d.items()}
   maximum=verify_replicas(experiment);progress=dict(experiment.progress)
   experiment.recover('private_distributed_recovery_check')
   assert experiment.progress==progress
   verify_replicas(experiment)
   experiment.restore()
   saved=torch.load(experiment.out/'tokenizer-last.pt',map_location='cpu',mmap=True,weights_only=False)
   current=trainer.capture_rng();expected=saved['rng'][experiment.rank]
   assert torch.equal(current['torch'],expected['torch']) and torch.equal(current['cuda'],expected['cuda'])
   assert torch.equal(experiment.model.decoder.conv_out.weight.detach().cpu(),saved['model']['decoder.conv_out.weight'])
   report=dict(passed=True,world_size=4,global_batch=128,microbatch=32,accumulation=1,source_step=source_step,final_step=source_step+2,
    full_state_roundtrip=True,all_rank_rng_bit_exact=True,optimizer_ages_advanced_correctly=True,all_rank_parameters_bit_exact=True,
    maximum_replica_parameter_error=maximum,encoder_quantizer_unchanged=True,distributed_best_recovery_passed=True,
    tokens_per_image=680,vocabulary=4096,peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)
   if experiment.rank==0:trainer.atomic(BASE/'preflight-verification.json',report);print(json.dumps(report),flush=True)
  else:
   experiment.train_loop()
   if experiment.run:experiment.run.summary['training/status']='user_stop_checkpointed';experiment.run.finish()
 except Exception as error:
  if experiment is not None:
   trainer.atomic(experiment.out/f'failure-rank{experiment.rank}.json',dict(type=type(error).__name__,error=str(error),progress=experiment.progress))
   if experiment.run:experiment.run.finish(exit_code=1)
  raise
 finally:
  if dist.is_initialized():dist.destroy_process_group()
if __name__=='__main__':main()
