"""Full tokenizer stage1 with the established VQ recipe and epoch-end-only FID50k."""
import argparse,json,os,shutil,sys,time
from pathlib import Path
BASE=Path('/tmp/laser-var-full-stage1-20261105')
RUNTIME=BASE/'runtime'
HERE=Path(__file__).resolve().parent
OUT=Path('/workspace/Projects/laser/outputs/imnet-s1-laser-matched680-full-continue-best-4gpu-20261105')
sys.path[:0]=[str(RUNTIME)]
import torch
import torch.distributed as dist
import train_matched_stage1 as trainer


def broadcast(v):
 values=[v if dist.get_rank()==0 else None];dist.broadcast_object_list(values,src=0);return values[0]


class FullStage1(trainer.Experiment):
 def __init__(self,cfg,smoke=False):
  self.learning_dynamics=dict(enabled=False,lr_factor=1.);self.quality_stop=None;self.evaluation_calls=0
  super().__init__(cfg,smoke)
  assert self.world==4 and self.global_batch==128 and self.updates==10009
  assert self.arm=='laser' and all(p.requires_grad for p in self.model.parameters())
  assert cfg['evaluation']['cadence']=='epoch_end_only' and cfg['epochs']==10
  dist.barrier()
  if not smoke:
   if self.rank==0:
    self.run.summary.update({'training/stage':'stage1','training/status':'running','training/full_tokenizer_training':True,
     'tokenizer/encoder_frozen':False,'tokenizer/quantizer_frozen':False,'tokenizer/decoder_frozen':False,
     'initialization/trainable_weights':'full-state resume from selected best tokenizer','initialization/fresh_optimizer':False,
     'training/world_size':4,'training/global_batch':128,'training/hardware':'4 RTX PRO 6000 Blackwell 96GB',
     'training/generator_lr':cfg['lr'],'training/discriminator_lr':cfg['discriminator_lr'],
     'evaluation/cadence':'epoch_end_only','evaluation/images':50000,'evaluation/mid_epoch_enabled':False,
     'comparison/baseline_run':cfg['fair_comparison']['baseline_run'],'comparison/baseline_reported_rfid':cfg['fair_comparison']['baseline_reported_rfid'],
     'comparison/warm_start_history_disclosed':True,'stage2/enabled':False,
     'reconstruction/best_rfid':cfg['lineage']['source_rfid'],'reconstruction/best_epoch':4,
     'reconstruction/best_artifact':cfg['lineage']['source_artifact']})
    self.run.use_artifact(cfg['lineage']['source_artifact'])
    import wandb
    a=wandb.Artifact(cfg['wandb']['id']+'-full-stage1-runtime',type='code')
    for p in HERE.glob('*.py'):a.add_file(str(p),name=p.name)
    for n in ['resume-verification.json','preflight-verification.json','fair-comparison.json']:
     path=BASE/n
     if path.exists():a.add_file(str(path),name=n)
    a.add_file(str(BASE/'full-stage1-runtime.patch'))
    logged=self.run.log_artifact(a);logged.wait()
    trainer.atomic(self.out/'runtime-artifact.json',dict(artifact=logged.qualified_name,state=str(logged.state)))
   dist.barrier()
   self.log('full_stage1_resumed',step=self.progress['step'],world_size=4,global_batch=128,
      encoder_trainable=True,quantizer_trainable=True,decoder_trainable=True,
      generator_lr=cfg['lr'],discriminator_lr=cfg['discriminator_lr'],evaluation_cadence='epoch_end_only')

 def log(self,phase,**values):
  super().log(phase,**values)
  if self.rank==0:
   line=json.dumps(dict(phase=phase,time=time.time(),**values))+'\n'
   with (self.out/'metrics.jsonl').open('a') as f:f.write(line)
   if not self.smoke:
    self.mirror('status.json')
    with (OUT/'metrics.jsonl').open('a') as f:f.write(line)

 def mirror(self,name):
  if self.rank!=0 or self.smoke:return
  p=self.out/name
  if p.exists():
   tmp=OUT/(name+'.pending');shutil.copyfile(p,tmp);tmp.replace(OUT/name)

 def restore(self):
  super().restore();self.learning_dynamics=dict(enabled=False,lr_factor=1.)

 def save(self,publish=False):
  super().save(publish)
  if self.rank==0 and publish:
   for name in ['checkpoint.json','joint-codebook.pt']:self.mirror(name)

 def evaluate(self):
  if self.progress['batch']!=0 or self.progress['step']!=self.progress['epoch']*self.updates:
   raise RuntimeError('Full 50k evaluation is allowed only at an epoch end')
  epoch=self.progress['epoch'];step=self.progress['step']
  marker=self.out/f'full-eval-complete-epoch{epoch:03d}.json'
  finished=broadcast(json.loads(marker.read_text()) if self.rank==0 and marker.exists() else None)
  if finished is not None:
   assert finished['step']==step and finished['count']==50000
   return
  previous=broadcast(json.loads((self.out/'best-reconstruction.json').read_text())['rfid'] if self.rank==0 else None)
  result_file=self.out/f'reconstruction-epoch{epoch:03d}.json'
  result=broadcast(json.loads(result_file.read_text()) if self.rank==0 and result_file.exists() else None)
  cached=result is not None and result.get('step')==step and result.get('count')==50000
  self.log('evaluation_started',step=step,epoch=epoch,count=50000,world_size=4,cadence='epoch_end_only',reuse_completed_metrics=cached)
  started=time.monotonic()
  if not cached:
   self.evaluation_calls+=1
   super().evaluate()
   result=broadcast(json.loads(result_file.read_text()) if self.rank==0 else None)
  assert result['count']==result['reference_count']==50000 and result['step']==step
  self.model.quantize.sync_assignment_tables()
  self.save(publish=True)
  if self.rank==0:
   receipt=json.loads((self.out/'checkpoint.json').read_text())
   if result['rfid']<previous:
    temporary=self.out/'tokenizer-best-rfid.pending';shutil.copyfile(self.out/'tokenizer-last.pt',temporary);temporary.replace(self.out/'tokenizer-best-rfid.pt')
    trainer.atomic(self.out/'best-reconstruction.json',dict(result,checkpoint=receipt))
    if self.run:
     import wandb
     a=wandb.Api().artifact(receipt['artifact'])
     if 'best-rfid' not in a.aliases:a.aliases.append('best-rfid');a.save()
     self.run.summary.update({'reconstruction/best_rfid':result['rfid'],'reconstruction/best_epoch':epoch,'reconstruction/best_artifact':receipt['artifact']})
    self.mirror('best-reconstruction.json')
   record=dict(result,checkpoint=receipt,cadence='epoch_end_only',evaluation_seconds=time.monotonic()-started,
      comparison_baseline_run=self.cfg['fair_comparison']['baseline_run'],baseline_reported_rfid=self.cfg['fair_comparison']['baseline_reported_rfid'],
      endpoint_budget_matched=epoch==10,warm_start_history_disclosed=True)
   trainer.atomic(marker,record);trainer.atomic(self.out/f'reconstruction-step{step:09d}.json',record)
   self.mirror(marker.name);self.mirror(f'reconstruction-step{step:09d}.json')
   if self.run:self.run.summary.update({'evaluation/latest_epoch':epoch,'evaluation/latest_step':step,'evaluation/cadence':'epoch_end_only'})
  self.log('evaluation_completed',step=step,epoch=epoch,rfid=result['rfid'],count=50000,seconds=time.monotonic()-started,cadence='epoch_end_only')
  dist.barrier()


def verify_replicas(experiment):
 maximum=torch.zeros((),device=experiment.device)
 with torch.no_grad():
  for model in [experiment.model,experiment.discriminator]:
   for p in model.parameters():
    reference=p.detach().clone();dist.broadcast(reference,src=0);maximum=torch.maximum(maximum,(reference-p).abs().max())
 dist.all_reduce(maximum,op=dist.ReduceOp.MAX);assert float(maximum)==0.,float(maximum)
 return float(maximum)


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);p.add_argument('--smoke',action='store_true');a=p.parse_args();cfg=json.loads(a.config.read_text())
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 trainer.var_dist.initialize(backend='nccl',timeout=120)
 if a.smoke:trainer.SmokeImages=lambda _:trainer.ArchiveImages(cfg['train_archive'],cfg['train_index'],True,cfg['seed'])
 experiment=None
 try:
  experiment=FullStage1(cfg,a.smoke)
  if a.smoke:
   start=experiment.progress['step'];old_g={k:int(s['step']) for k,s in experiment.optimizer.state_dict()['state'].items() if 'step' in s};old_d={k:int(s['step']) for k,s in experiment.d_optimizer.state_dict()['state'].items() if 'step' in s}
   names=['encoder.conv_in.weight','quant_conv.weight','quantize.atom_directions','quantize.raw_magnitudes','decoder.conv_out.weight','post_quant_conv.weight']
   parameters=dict(experiment.model.named_parameters());before={n:parameters[n].detach().cpu().clone() for n in names}
   try:experiment.evaluate()
   except RuntimeError as e:assert 'epoch end' in str(e)
   else:raise AssertionError('Mid-epoch evaluation was not rejected')
   experiment.train_loop(start+2);assert experiment.progress['step']==start+2 and experiment.evaluation_calls==0
   changed={n:int((parameters[n].detach().cpu()!=before[n]).sum()) for n in names};assert all(v>0 for v in changed.values()),changed
   assert {k:int(s['step']) for k,s in experiment.optimizer.state_dict()['state'].items() if 'step' in s}=={k:v+2 for k,v in old_g.items()}
   assert {k:int(s['step']) for k,s in experiment.d_optimizer.state_dict()['state'].items() if 'step' in s}=={k:v+2 for k,v in old_d.items()}
   assert all(g['lr']==1e-4 for g in experiment.optimizer.param_groups) and all(g['lr']==1e-4 for g in experiment.d_optimizer.param_groups)
   maximum=verify_replicas(experiment);experiment.restore();saved=torch.load(experiment.out/'tokenizer-last.pt',map_location='cpu',mmap=True,weights_only=False)
   rng=trainer.capture_rng();expected=saved['rng'][experiment.rank];assert torch.equal(rng['torch'],expected['torch']) and torch.equal(rng['cuda'],expected['cuda'])
   assert torch.equal(experiment.model.decoder.conv_out.weight.detach().cpu(),saved['model']['decoder.conv_out.weight'])
   report=dict(passed=True,stage='stage1',world_size=4,source_step=start,final_step=start+2,global_batch=128,all_tokenizer_parameters_trainable=True,all_rank_parameters_bit_exact=True,maximum_replica_error=maximum,component_parameter_changes=changed,both_optimizer_ages_advanced_correctly=True,all_rank_rng_bit_exact=True,model_roundtrip_bit_exact=True,generator_lr=1e-4,discriminator_lr=1e-4,mid_epoch_evaluation_rejected=True,full_evaluation_calls=0,evaluation_cadence='epoch_end_only',tokens_per_image=680,vocabulary=4096,peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)
   if experiment.rank==0:trainer.atomic(BASE/'preflight-verification.json',report);print(json.dumps(report),flush=True)
  else:
   experiment.train_loop()
   if experiment.run and experiment.progress['epoch']<cfg['epochs']:experiment.run.summary['training/status']='user_stop_checkpointed';experiment.run.finish()
 except Exception as e:
  if experiment is not None:
   trainer.atomic(experiment.out/f'failure-rank{experiment.rank}.json',dict(type=type(e).__name__,error=str(e),progress=experiment.progress))
   if experiment.run:experiment.run.finish(exit_code=1)
  raise
 finally:
  if dist.is_initialized():dist.destroy_process_group()
if __name__=='__main__':main()
