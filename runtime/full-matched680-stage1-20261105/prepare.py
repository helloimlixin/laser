"""Prepare full stage-1 continuation with the completed VQ control's steady-state recipe."""
import json,hashlib,shutil,sys,tarfile,time
from pathlib import Path
import torch
BASE=Path('/tmp/laser-var-full-stage1-20261105')
OLD=Path('/tmp/laser-var-stage1-4gpu-20261105')
HERE=Path(__file__).resolve().parent
OUT=Path('/workspace/Projects/laser/outputs/imnet-s1-laser-matched680-full-continue-best-4gpu-20261105')
EXCLUDED={'output','wandb','workers','checkpoint_every','preview_every','source_bundle'}
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,default=str)+'\n')
def main():
 torch.set_num_threads(4)
 baseline=json.loads((BASE/'vq-baseline.json').read_text());b=baseline['config'];tok=b['tokenizer']
 assert b['stage']=='stage1' and b['model']['bottleneck']=='vq' and tok['epochs']==10
 assert sum(x*x for x in b['model']['patch_nums'])==680
 source=OLD/'run/tokenizer-best-rfid.pt';receipt=json.loads((OLD/'run/migrated-best-published.json').read_text())
 assert sha(source)==receipt['checkpoint_sha256']
 p=torch.load(source,map_location='cpu',mmap=True,weights_only=False)
 assert p['world_size']==4 and len(p['rng'])==4 and p['progress']['step']==43166
 runtime=BASE/'runtime';assert not runtime.exists(),'Refusing to overwrite a prepared runtime'
 shutil.copytree(OLD/'runtime',runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 script=runtime/'train_matched_stage1.py';original=script.read_text();s=original
 def replace(a,z):
  nonlocal s
  assert s.count(a)==1,(a,s.count(a));s=s.replace(a,z)
 replace("for epoch in itertools.count(self.progress['epoch']):", "for epoch in range(self.progress['epoch'],cfg['epochs']):")
 replace("for group in self.optimizer.param_groups:group['lr']=learning_rate(self,self.progress['step'])", "for group in self.optimizer.param_groups:group['lr']=cfg['lr']*ratio")
 replace("for group in self.d_optimizer.param_groups:group['lr']=max(cfg['discriminator_lr_floor'],cfg['discriminator_lr']*self.learning_dynamics['lr_factor'])", "for group in self.d_optimizer.param_groups:group['lr']=cfg['discriminator_lr']*ratio")
 replace("                if not self.smoke and evaluation_due(cfg['continuation'],self.progress['step']):\n                    self.save(publish=False);self.evaluate()\n                    if self.stop:return\n", '')
 replace("not (self.out/f\"reconstruction-epoch{self.progress['epoch']:03d}.json\").exists():self.evaluate()", "not (self.out/f\"full-eval-complete-epoch{self.progress['epoch']:03d}.json\").exists():self.evaluate()")
 replace("self.progress.update(epoch=epoch+1,batch=0);self.save(publish=not self.smoke)", "self.progress.update(epoch=epoch+1,batch=0);self.save(publish=False)")
 old='''                best=self.out/'best-reconstruction.json'
                if not best.exists() or score<json.loads(best.read_text())['rfid']:
                    atomic(best,{**result,'checkpoint':json.loads((self.out/'checkpoint.json').read_text())})
                    if self.run:self.run.summary.update({'reconstruction/best_rfid':score,'reconstruction/best_epoch':self.progress['epoch']})
'''
 replace(old,'')
 script.write_text(s)
 import difflib
 (BASE/'full-stage1-runtime.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='decoder_refinement/train_matched_stage1.py',tofile='full_stage1/train_matched_stage1.py')))
 shutil.copyfile(HERE/'driver.py',runtime/'full_stage1_driver.py')
 cfg=json.loads((OLD/'run/config.json').read_text())
 for k in ['decoder_refinement','verified_pilot','continuous','continuation','migration','discriminator_lr_floor']:cfg.pop(k,None)
 cfg.update(lr=float(tok['lr']),discriminator_lr=float(tok['discriminator_lr']),warmup_updates=int(tok['warmup_steps']),epochs=int(tok['epochs']),checkpoint_every=1000,preview_every=1000,eval_batch=32,source_bundle=str(BASE/'source.tar.gz'),full_stage1=True,evaluation=dict(cadence='epoch_end_only',images=50000,reference_split='validation',no_mid_epoch_evaluations=True,successful_evaluations_per_epoch=1),stage='stage1',wandb=dict(id=OUT.name,name='ImageNet LASER VAR stage1 | full encoder+quantizer+decoder | matched680 | epoch-end FID'))
 assert cfg['lr']==cfg['discriminator_lr']==1e-4 and cfg['global_batch']==128 and cfg['microbatch']==32 and cfg['accumulation']==1
 cfg['fair_comparison']=dict(baseline_run=baseline['id'],baseline_tokenizer_epochs=10,baseline_reported_rfid=baseline['summary']['reconstruction_full/fid'],training_images=1281167,validation_images=50000,image_size=256,token_budget=680,vocabulary=4096,nominal_bits_per_image=8160,backbone='VAR VQVAE ch160 z32 downsample16',optimizer_g='AdamW betas(0.5,0.9) weight_decay0',optimizer_d='Adam betas(0.5,0.9)',global_batch=128,lr_g=1e-4,lr_d=1e-4,preprocessing='Resize288 Lanczos; deterministic epoch-specific RandomCrop256; no horizontal flip; validation CenterCrop256',losses='L1 + LPIPS + bottleneck + adaptive GAN, coefficient0.5 capped0.1; hinge critic',precision='bf16 CNN; float32 quantizer and perceptual/adversarial losses; TF32 disabled',intentional_representation_difference='LASER four scales with two joint IDs/site; VQ ten scales with one ID/site',limitation='Selected warm-start after prior tuning and decoder refinement; not a fresh equal-compute randomized ablation. Compare endpoint reconstruction metrics with this history disclosed.',historical_critic_repair_difference='The inherited VQ and LASER critics used different past repair activation steps; future steady-state losses and optimizer hyperparameters are matched.')
 cfg['lineage']=dict(source_artifact=receipt['artifact'],source_checkpoint_sha256=sha(source),source_step=43166,source_rfid=2.4608426770367373,reason='User requested full stage1, fair VQ comparison, epoch-end-only 50k evaluation',model_and_optimizer_states_loaded=True,decoder_only_refinement_disabled=True,adaptive_lr_rewind_policy_disabled=True)
 cfg['runtime_sha256']={n:sha(runtime/n) for n in cfg['runtime_sha256'] if n!='four_gpu_driver.py'}
 cfg['runtime_sha256']['full_stage1_driver.py']=sha(runtime/'full_stage1_driver.py')
 progress=dict(p['progress']);progress['batch']=progress['step']-progress['epoch']*10009
 assert progress==dict(epoch=4,batch=3130,step=43166)
 sys.path.insert(0,str(runtime))
 from migrate_stage1_world_size import assert_equal,global_indices
 assert global_indices(1281167,1,16,8,25040,4,0)==global_indices(1281167,4,32,1,3130,4,0)
 best=json.loads((OLD/'run/best-reconstruction.json').read_text())
 for folder in ['run','preflight']:
  c=dict(cfg,output=str(BASE/folder));local=BASE/folder;local.mkdir(exist_ok=True)
  contract={k:v for k,v in c.items() if k not in EXCLUDED};digest=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
  payload=dict(p,progress=progress,contract_sha256=digest,learning_dynamics=dict(enabled=False,lr_factor=1.),full_stage1_lineage=cfg['lineage'])
  torch.save(payload,local/'tokenizer-last.pt')
  check=torch.load(local/'tokenizer-last.pt',map_location='cpu',mmap=True,weights_only=False)
  for k in ['model','optimizer','discriminator','discriminator_optimizer','rng']:assert_equal(p[k],check[k])
  torch.save(payload,local/'tokenizer-best-rfid.pt')
  best_record=dict(best,checkpoint=dict(receipt,progress=progress,contract_sha256=digest,checkpoint_sha256=sha(local/'tokenizer-best-rfid.pt'),inherited_metric=True,source_checkpoint_sha256=sha(source)))
  for name,v in [('config.json',c),('contract.json',contract),('initialization.json',dict(initialization='full_stage1_resume_selected_best',source_artifact=receipt['artifact'],source_step=43166,source_rfid=2.4608426770367373,all_tokenizer_parameters_trainable=True)),('best-reconstruction.json',best_record),('checkpoint.json',dict(progress=progress,contract_sha256=digest,checkpoint_sha256=sha(local/'tokenizer-last.pt'),full_state=True,world_size=4))]:write(local/name,v)
 report=dict(passed=True,stage='stage1',source_artifact=receipt['artifact'],source_step=43166,source_rfid=2.4608426770367373,progress=progress,world_size=4,microbatch=32,accumulation=1,global_batch=128,model_both_optimizers_all_rank_rng_preserved=True,all_tokenizer_parameters_trainable=True,baseline_run=baseline['id'],generator_lr=1e-4,discriminator_lr=1e-4,lr_policy='baseline constant after 500-update warmup; no decoder-only plateau controller',evaluation='exactly one successful full50k evaluation per epoch end',evaluation_steps=[50045,60054,70063,80072,90081,100090],target_epochs=10,mid_epoch_evaluations=False,stage2_enabled=False,warm_start_history_disclosed=True,time=time.time())
 write(BASE/'resume-verification.json',report);write(OUT/'resume-verification.json',report);write(OUT/'fair-comparison.json',cfg['fair_comparison']);write(OUT/'config.json',dict(cfg,output=str(BASE/'run')))
 shutil.copyfile(BASE/'full-stage1-runtime.patch',OUT/'runtime.patch')
 with tarfile.open(BASE/'source.tar.gz','w:gz') as a:
  for f in runtime.rglob('*'):
   if f.is_file() and '__pycache__' not in f.parts:a.add(f,arcname='runtime/'+str(f.relative_to(runtime)))
 print(json.dumps(report),flush=True)
if __name__=='__main__':main()
