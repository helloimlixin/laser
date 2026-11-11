"""Calibrate continuous density units and prepare the frozen beam-aligned stage2."""
from pathlib import Path
import json,shutil,sys,hashlib,time,tarfile,py_compile,math
import torch,numpy as np
from torch.utils.data import DataLoader,Subset
BASE=Path('/tmp/laser-var-stage2-beam-continuous-20261106')
PREVIOUS=Path('/tmp/laser-var-stage2-matched680-20261106')
OUT=Path('/workspace/Projects/laser/outputs/imnet-s2-laser-beam16-continuous-var-d16-4gpu-20261106')

def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,default=str)+'\n')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 runtime=BASE/'runtime';inputs=BASE/'inputs';inputs.mkdir(parents=True,exist_ok=True);OUT.mkdir(exist_ok=True)
 if not runtime.exists():shutil.copytree(PREVIOUS/'runtime',runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 if (BASE/'run/prior-last.pt').exists():raise RuntimeError('Do not replace an active production training contract')
 here=Path(__file__).resolve().parent
 source_file=Path('/workspace/Projects/laser/src/models/continuous_beam_var.py')
 if source_file.exists():shutil.copyfile(source_file,runtime/'src/models/continuous_beam_var.py')
 for name in ['frozen_continuous_tokenizer.py']:shutil.copyfile(here/name,runtime/name)
 sys.path.insert(0,str(runtime))
 from frozen_continuous_tokenizer import load_source,LASER_HASH,digest_state
 from frozen_tokenizer import load_source as load_joint_source
 from train_stage2 import ArchiveImages
 cfg=json.loads((PREVIOUS/'config.json').read_text())
 checkpoint=inputs/'source-stage1.pt'
 if not checkpoint.exists():checkpoint.hardlink_to(Path(cfg['checkpoint']))
 assert sha(checkpoint)==LASER_HASH
 model=load_source(checkpoint,'laser','cuda:0');q=model.quantize
 train=ArchiveImages(cfg['train_archive'],cfg['train_index'],True,0)
 indices=np.random.default_rng(26110621).choice(len(train),512,replace=False).tolist()
 sums=torch.zeros(4,device='cuda');counts=torch.zeros_like(sums)
 with torch.no_grad():
  for first in range(0,len(indices),16):
   images=torch.stack([train[i][0] for i in indices[first:first+16]]).cuda()
   with torch.autocast('cuda',dtype=torch.bfloat16):latent=model.quant_conv(model.encoder(images))
   result=q.decompose(latent);offset=0
   for scale,pn in enumerate(q.v_patch_nums):
    values=result['coefficients'][:,offset:offset+pn*pn];sums[scale]+=values.square().sum();counts[scale]+=values.numel();offset+=pn*pn
   print(json.dumps(dict(phase='calibration',images=first+len(images))),flush=True)
 scales=(sums/counts).sqrt().clamp_min(.01);q.coefficient_scales.copy_(scales)
 spec=dict(kind='beam16_ols_continuous_sparse_codec',format_version=1,source_checkpoint_sha256=LASER_HASH,
  source_artifact=cfg['tokenizer_artifact'],dictionary=q.dictionary.cpu(),coefficient_scales=scales.cpu(),
  beam_width=16,sparsity=2,patch_nums=list(q.v_patch_nums),atoms=1024,coefficient_representation='continuous_fp32',
  coefficient_quantization=False,coefficient_clipping=False,per_image_full_refit=True,
  teacher='complete beam trajectory posterior; exact preceding atom+coefficient conditionals',
  sequence_temperature_relative=.01,max_relative_candidate_error=.01,calibration_images=512,calibration_seed=26110621,
  pairs_per_image=680,categorical_decisions_per_image=680,continuous_values_per_image=680,
  nominal_atom_bits_per_image=6800,float32_coefficient_bits_per_image=21760,total_nominal_bits_per_image=28560)
 torch.save(spec,inputs/'continuous-codebook.pt')
 # Compare actual image reconstructions with the original codec, without
 # presenting a small check as a full-validation rFID measurement.
 original=load_joint_source(checkpoint,'laser','cuda:0')
 val=ArchiveImages(cfg['val_archive'],cfg['val_index'],False,0,labels=cfg['val_labels'])
 val_indices=np.random.default_rng(26110622).choice(len(val),128,replace=False).tolist()
 totals=torch.zeros(4,device='cuda',dtype=torch.float64);count=0
 with torch.no_grad():
  for first in range(0,len(val_indices),8):
   images=torch.stack([val[i][0] for i in val_indices[first:first+8]]).cuda()
   with torch.autocast('cuda',dtype=torch.bfloat16):z=model.quant_conv(model.encoder(images))
   beam=q.decompose(z);joint=original.quantize.decompose(z)
   decoded,teacher_inputs=q.from_codes(beam['atoms'],beam['coefficients'])
   torch.testing.assert_close(decoded,beam['latent'],atol=0,rtol=0);torch.testing.assert_close(teacher_inputs,beam['inputs'],atol=0,rtol=0)
   with torch.autocast('cuda',dtype=torch.bfloat16):
    beam_images=model.decoder(model.post_quant_conv(beam['latent'])).float().clamp(-1,1)
    joint_images=original.decoder(original.post_quant_conv(joint['latent'])).float().clamp(-1,1)
   assert beam_images.isfinite().all()
   totals+=torch.stack(((beam_images-images).square().mean(),(joint_images-images).square().mean(),
     (beam['latent']-z.float()).square().mean(),(joint['latent']-z.float()).square().mean())).double()*len(images);count+=len(images)
  cached=torch.load(cfg['smoke_cache'],map_location='cpu',weights_only=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):z=model.quant_conv(model.encoder(cached['images'][:2].cuda()))
  teacher=q.decompose(z,teacher=True);decoded,teacher_inputs=q.from_codes(teacher['atoms'],teacher['coefficients'])
  torch.testing.assert_close(decoded,teacher['latent'],atol=0,rtol=0);torch.testing.assert_close(teacher_inputs,teacher['inputs'],atol=0,rtol=0)
 quality=(totals/count).cpu().tolist()
 report=dict(passed=True,source_checkpoint_sha256=LASER_HASH,source_stage1_epoch=10,source_joint_codec_rfid=cfg['stage1_rfid'],
  new_codec_full_validation_rfid=None,coefficient_scales=scales.cpu().tolist(),training_calibration_images=512,
  validation_check_images=count,beam_pixel_mse=quality[0],source_joint_pixel_mse=quality[1],
  beam_psnr=-10*math.log10(quality[0]/4),source_joint_psnr=-10*math.log10(quality[1]/4),
  beam_latent_mse=quality[2],source_joint_latent_mse=quality[3],integer_and_continuous_roundtrip_bit_exact=True,
  sampled_teacher_inputs_bit_exact=True,finite_decodes=True,frozen_tokenizer_digest=digest_state(model),
  beam_width=16,per_site_full_ols_refit=True,coefficient_quantization=False,coefficient_clipping=False,
  sparse_pairs_per_image=680,total_nominal_storage_bits_fp32=28560,matched_vq_storage_bits=False,time=time.time())
 write(BASE/'codec-verification.json',report);write(OUT/'codec-verification.json',report)
 print(json.dumps(report),flush=True)
 del original,model;torch.cuda.empty_cache()
 original_code=(runtime/'train_stage2.py').read_text();code=original_code
 def replace(a,b):
  nonlocal code
  assert code.count(a)==1,(a,code.count(a));code=code.replace(a,b)
 replace('from frozen_tokenizer import load_source, LASER_HASH, digest_state','from frozen_continuous_tokenizer import load_source, LASER_HASH, digest_state')
 replace('from src.models.joint_multiscale_var import JointMultiScaleQuantizer, JointLaserVAR','from src.models.continuous_beam_var import ContinuousBeamVAR')
 replace("torch.testing.assert_close(self.q.coefficients.cpu(),spec['coefficients'],rtol=0,atol=0)","torch.testing.assert_close(self.q.coefficient_scales.cpu(),spec['coefficient_scales'],rtol=0,atol=0)")
 replace('        self.q.quantize_scale=functools.partial(self.q.quantize_scale,chunk_size=2048)\n','')
 replace("        if (self.q.vocab_size,self.q.num_directions,self.q.levels)!=(4096,1024,4):raise ValueError('Requires the trained matched680/4096-symbol tokenizer')", "        if self.q.vocab_size!=1024 or self.q.beam_width!=16:raise ValueError('Requires the beam16 continuous-coefficient tokenizer')")
 replace('self.model=JointLaserVAR(self.q,depth=16,num_classes=1000)','self.model=ContinuousBeamVAR(self.q,depth=16,num_classes=1000)')
 replace("group='imagenet-var-matched680-stage2-20261106'","group='imagenet-var-beam16-continuous-stage2-20261106'")
 replace("'stage1/full_validation_rfid':cfg['stage1_rfid']","'tokenizer/source_joint_stage1_rfid':cfg['source_joint_stage1_rfid']")
 replace("'tokenizer/tokens_per_image':680,'tokenizer/vocabulary':4096,'tokenizer/nominal_bits_per_image':8160", "'tokenizer/sparse_pairs_per_image':680,'tokenizer/atom_vocabulary':1024,'tokenizer/coefficient_representation':'continuous_fp32','tokenizer/beam_width':16,'tokenizer/nominal_bits_per_image':28560,'teacher/beam_aligned':True")
 replace('    def tokenize(self,images):','    def tokenize(self,images,*,teacher=False):')
 replace('return self.q.decompose(latent)','return self.q.decompose(latent,teacher=teacher)')
 replace("loss=self.model(labels,code['inputs'],code['codes']);wrong=self.model((labels+1)%1000,code['inputs'],code['codes'])", "loss=self.model(labels,code['inputs'],code['atoms'],code['coefficients']);wrong=self.model((labels+1)%1000,code['inputs'],code['atoms'],code['coefficients'])")
 replace("joint_nll=float(total[0]/total[2]),wrong_label_nll=float(total[1]/total[2])", "joint_mixed_nll=float(total[0]/total[2]),wrong_label_mixed_nll=float(total[1]/total[2])")
 replace("codes=self.tokenize(images.to(self.device,non_blocking=True));sync=", "codes=self.tokenize(images.to(self.device,non_blocking=True),teacher=True);sync=")
 replace("loss=self.wrapped(labels,codes['inputs'],codes['codes'])", "loss=self.wrapped(labels,codes['inputs'],codes['atoms'],codes['coefficients'],codes['targets'])")
 replace("joint_nll=float(loss_sum),uniform_joint_nll=(math.log(4096)+math.log(4092))*.5,gain_vs_uniform=(math.log(4096)+math.log(4092))*.5-float(loss_sum)", "joint_mixed_nll=float(loss_sum),atom_nll=float(self.model.last_losses['atom_nll']),continuous_coefficient_nll=float(self.model.last_losses['coefficient_nll'])")
 replace("tokens_per_image=680,vocabulary=4096", "sparse_pairs_per_image=680,atom_vocabulary=1024,coefficient_representation='continuous_fp32',beam_width=16,beam_aligned_teacher=True")
 # Resume an epoch-end checkpoint with its evaluation still pending; completed
 # receipts prevent repeating a full 50k evaluation on a later process restart.
 replace("        with evaluation_rng(self.device):\n            self.model.eval()\n            if self.inception is None:", "        receipt=self.out/f\"fid-epoch{self.progress['epoch']:03d}.json\"\n        done=torch.tensor(int(receipt.exists()) if self.rank==0 else 0,device=self.device);dist.broadcast(done,src=0)\n        if done.item():return\n        with evaluation_rng(self.device):\n            self.model.eval()\n            if self.inception is None:")
 replace("        if max_steps and self.progress['step']>=max_steps:return", "        if max_steps and self.progress['step']>=max_steps:return\n        if not self.smoke and self.progress['epoch']>0 and self.progress['batch']==0:self.fid()")
 driver=runtime/'train_continuous_stage2.py';driver.write_text(code);py_compile.compile(str(driver),doraise=True)
 cfg.update(output=str(BASE/'run'),checkpoint=str(checkpoint),codebook=str(inputs/'continuous-codebook.pt'),codebook_sha256=sha(inputs/'continuous-codebook.pt'),
  source_bundle=str(BASE/'source.tar.gz'),source_joint_stage1_rfid=cfg.pop('stage1_rfid'),tokenizer_full_validation_rfid=None,
  vocabulary=1024,atoms=1024,coefficient_representation='continuous_fp32',coefficient_quantization=False,coefficient_clipping=False,
  coefficient_density='8-component Gaussian mixture in asinh(coefficient/RMS); exact physical Jacobian; min_std0.03',
  density_coefficient_scales=scales.cpu().tolist(),beam_width=16,teacher='beam_aligned_complete_trajectory_posterior',
  beam_teacher=dict(sequence_temperature_relative=.01,max_relative_candidate_error=.01,condition_on='preceding atom IDs and physical FP32 coefficients',prefixes='sample one terminal beam trajectory per site; same coefficients used for decoder and next-scale inputs'),
  sparse_pairs_per_image=680,categorical_decisions_per_image=680,continuous_values_per_image=680,nominal_atom_bits_per_image=6800,
  nominal_coefficient_bits_per_image_fp32=21760,total_nominal_bits_per_image_fp32=28560,
  fid_reference=str(inputs/'validation-fid-reference.npz'),wandb=dict(id=OUT.name,name='ImageNet LASER | beam16 continuous coefficients | aligned teacher | VAR-d16 | 4 GPUs'))
 for key in ['coefficient_levels','directions','joint_tokens_per_image']:cfg.pop(key,None)
 shutil.copyfile(PREVIOUS/'inputs/validation-fid-reference.npz',cfg['fid_reference'])
 cfg['sampling'].update(coefficient_sampler='normalized class-conditional continuous density',cfg_applies_to='atom logits',coefficient_temperature=1.)
 cfg['comparison'].update(token_budget='680 sparse pairs, each with a 1024-way atom and one physical continuous coefficient',
  representation_difference='680 categorical atom IDs +680 continuous FP32 values; VQ 680 categorical IDs with vocabulary4096',
  nominal_storage_bits=dict(laser_fp32=28560,vq=8160),equal_nominal_bits=False,
  sampler='CFG1.5 top-k900 top-p.96 on atoms; class-conditional coefficient density, temperature1; no rejection')
 keys=['train_continuous_stage2.py','frozen_continuous_tokenizer.py','frozen_tokenizer.py','src/models/continuous_beam_var.py','src/models/trainable_joint_var_tokenizer.py','src/models/joint_multiscale_var.py','src/models/multiscale_laser_var.py','src/rqvae_metrics.py','src/original_rq_training.py','third_party/FoundationVision_VAR/models/var.py','third_party/FoundationVision_VAR/models/basic_var.py','third_party/FoundationVision_VAR/utils/lr_control.py']
 cfg['runtime_sha256']={name:sha(runtime/name) for name in keys}
 write(BASE/'config.json',cfg);write(OUT/'config.json',cfg);write(OUT/'comparison.json',cfg['comparison']);(BASE/'run').mkdir(exist_ok=True)
 for micro,accum in [(192,1),(96,2),(64,3)]:
  trial=dict(cfg,output=str(BASE/f'preflight-mb{micro}'),microbatch=micro,accumulation=accum);Path(trial['output']).mkdir(exist_ok=True);write(BASE/f'preflight-mb{micro}.json',trial)
 with tarfile.open(BASE/'source.tar.gz','w:gz') as tar:
  for p in runtime.rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts:tar.add(p,arcname='runtime/'+str(p.relative_to(runtime)))
 print(json.dumps(dict(prepared=True,output=str(OUT),source_sha256=LASER_HASH,new_codec_sha256=cfg['codebook_sha256'])),flush=True)
if __name__=='__main__':main()
