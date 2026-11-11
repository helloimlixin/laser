"""Verified LASER VAR stage-1 continuation over the original matched-budget runtime."""
import os,sys,json,runpy,argparse
from pathlib import Path
BASE=Path('/tmp/laser-var-stage1-20261105')
RUNTIME=BASE/'frozen/runtime'
HERE=Path(__file__).resolve().parent
OUT=Path('/workspace/Projects/laser/outputs/imnet-s1-laser-matched680-best24608-resume-20261105')
sys.path[:0]=[str(RUNTIME)]
import torch
import train_matched_stage1 as trainer
import continuation

def freeze_factory():
 original=trainer.FourScaleJointVAE
 class FrozenTokenizerVAE(original):
  def __init__(self,*args,**kwargs):
   super().__init__(*args,**kwargs)
   self.encoder.requires_grad_(False);self.quant_conv.requires_grad_(False);self.quantize.requires_grad_(False)
 trainer.FourScaleJointVAE=FrozenTokenizerVAE

def smoke(cfg):
 torch.set_num_threads(4);trainer.var_dist.initialize(backend='nccl',timeout=120)
 freeze_factory()
 trainer.SmokeImages=lambda _:trainer.ArchiveImages(cfg['train_archive'],cfg['train_index'],True,cfg['seed'])
 continuation.install(trainer)
 experiment=None
 try:
  experiment=trainer.Experiment(cfg,True)
  frozen={n:p.detach().cpu().clone() for n,p in experiment.model.named_parameters() if not p.requires_grad}
  assert frozen and experiment.progress['step']==43166
  experiment.train_loop(max_steps=43168)
  assert experiment.progress['step']==43168
  for n,p in experiment.model.named_parameters():
   if n in frozen:assert torch.equal(frozen[n],p.detach().cpu()),n
  experiment.restore()
  saved=torch.load(experiment.out/'tokenizer-last.pt',map_location='cpu',mmap=True,weights_only=False)
  current=trainer.capture_rng();expected=saved['rng'][0]
  assert torch.equal(current['torch'],expected['torch']) and torch.equal(current['cuda'],expected['cuda'])
  assert torch.equal(experiment.model.decoder.conv_out.weight.detach().cpu(),saved['model']['decoder.conv_out.weight'])
  ga=sorted({int(s['step']) for s in experiment.optimizer.state.values() if 'step' in s})
  da=sorted({int(s['step']) for s in experiment.d_optimizer.state.values() if 'step' in s})
  assert ga==[43136,43168] and da==[33168],(ga,da)
  probe=torch.load(cfg['smoke_cache'],weights_only=True)['images'].to(experiment.device)
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   experiment.model.eval();recon=experiment.model(probe)[0]
  assert torch.isfinite(recon).all() and experiment.model.quantize.last_atom_ids.numel()==len(probe)*680
  report=dict(passed=True,stage='stage1',progress=experiment.progress,optimizer_ages=ga,discriminator_optimizer_ages=da,full_state_roundtrip=True,rng_bit_exact=True,model_bit_exact=True,encoder_quantizer_parameters_unchanged=True,tokens_per_image=680,vocabulary=4096,nominal_bits_per_image=8160,finite_reconstruction=True,stage2_enabled=False,peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)
  trainer.atomic(OUT/'preflight-verification.json',report);print(json.dumps(report),flush=True)
 finally:
  if trainer.td.is_initialized():trainer.td.destroy_process_group()

def production(cfg):
 original_init=trainer.Experiment.__init__
 def verified_init(self,config,smoke=False):
  original_init(self,config,smoke)
  assert self.arm=='laser' and self.world==1 and self.global_batch==128
  assert not any(p.requires_grad for p in self.model.encoder.parameters())
  if self.run:
   self.run.config.update({'stage':'stage1','resume_source':'matched680_joint4096_best'},allow_val_change=True)
   self.run.summary.update({'training/stage':'stage1','training/source_step':43166,'training/source_best_rfid':2.4608426770367373,'training/hardware':'NVIDIA RTX PRO 6000 Blackwell 96GB','baseline/inherited_same_hardware_and_protocol':False,'baseline/inherited_rfid':2.4608426770367373,'stage2/enabled':False})
   self.run.use_artifact(config['continuation']['source_artifact'])
   import wandb
   a=wandb.Artifact(config['wandb']['id']+'-resume-runtime',type='code')
   for p in HERE.glob('*.py'):a.add_file(str(p),name=p.name)
   for name in ['resume-verification.json','preflight-verification.json']:a.add_file(str(OUT/name))
   a=self.run.log_artifact(a);a.wait()
   trainer.atomic(self.out/'resume-runtime-artifact.json',dict(artifact=a.qualified_name,state=str(a.state)))
 trainer.Experiment.__init__=verified_init
 runpy.run_path(str(RUNTIME/'continuous.py'),run_name='__main__')

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 cfg=json.loads(a.config.read_text())
 if a.smoke:smoke(cfg)
 else:production(cfg)
if __name__=='__main__':main()
