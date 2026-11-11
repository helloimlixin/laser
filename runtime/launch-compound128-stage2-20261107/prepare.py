"""Calibrate fixed coefficient bins and prepare the compound128 stage2 trial."""
from pathlib import Path
import json,shutil,sys,hashlib,time,tarfile,py_compile,math
import numpy as np
import torch
BASE=Path('/tmp/laser-var-stage2-compound128-20261107')
OLD=Path('/tmp/laser-var-stage2-beam-joint4096-20261106')
OUT=Path('/workspace/Projects/laser/outputs/imnet-s2-laser-compound128-beam16-var-d16-4gpu-20261107')
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,default=str)+'\n')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 here=Path(__file__).resolve().parent;runtime=BASE/'runtime';inputs=BASE/'inputs';inputs.mkdir(parents=True,exist_ok=True);OUT.mkdir(exist_ok=True)
 if (BASE/'run/prior-last.pt').exists():raise RuntimeError('Do not replace an active production training contract')
 shutil.copyfile(here/'frozen_compound128_tokenizer.py',runtime/'frozen_compound128_tokenizer.py')
 sys.path.insert(0,str(runtime))
 from frozen_tokenizer import load_source as load_original,LASER_HASH,digest_state
 from src.models.continuous_beam_var import ContinuousBeamQuantizer
 from src.models.compound_beam128_var import CompoundBeam128Quantizer
 from train_stage2 import ArchiveImages
 cfg=json.loads((OLD/'config.json').read_text())
 checkpoint=inputs/'source-stage1.pt'
 if not checkpoint.exists():checkpoint.hardlink_to(Path(cfg['checkpoint']))
 assert sha(checkpoint)==LASER_HASH
 model=load_original(checkpoint,'laser','cuda:0');original_q=model.quantize
 continuous=ContinuousBeamQuantizer(original_q,beam_width=16,chunk_size=256).eval().requires_grad_(False)
 train=ArchiveImages(cfg['train_archive'],cfg['train_index'],True,0)
 indices=np.random.default_rng(26110713).choice(len(train),512,replace=False).tolist();values=[[] for _ in range(4)]
 with torch.no_grad():
  for first in range(0,len(indices),16):
   images=torch.stack([train[i][0] for i in indices[first:first+16]]).cuda()
   with torch.autocast('cuda',dtype=torch.bfloat16):z=model.quant_conv(model.encoder(images))
   result=continuous.decompose(z);offset=0
   for scale,pn in enumerate(continuous.v_patch_nums):
    values[scale].append(result['coefficients'][:,offset:offset+pn*pn].flatten().cpu());offset+=pn*pn
   print(json.dumps(dict(phase='coefficient_calibration',images=first+len(images))),flush=True)
  quantiles=torch.stack([torch.quantile(torch.cat(v).abs(),.9995) for v in values])
  learned_max=original_q.physical_tables()[1].detach().cpu().abs().flatten(1).amax(-1)
  bounds=torch.maximum(quantiles*1.1,learned_max*1.01).clamp_min(.1)
  model.quantize=CompoundBeam128Quantizer(original_q,beam_width=16,chunk_size=256,sequence_temperature=.01,max_relative_error=.01)
  model.quantize.coefficient_max.copy_(bounds);model.eval().requires_grad_(False);q=model.quantize
  spec=dict(kind='beam16_compound128',format_version=1,source_checkpoint_sha256=LASER_HASH,source_artifact=cfg['tokenizer_artifact'],
   dictionary=q.dictionary.cpu(),coefficient_max=bounds,companding=q.companding.cpu(),coefficient_grid=torch.stack([q.coefficient_values(torch.arange(128,device='cuda'),s).cpu() for s in range(4)]),
   coefficient_bins=128,atoms=1024,sparsity=2,patch_nums=list(q.v_patch_nums),beam_width=16,
   pair_id='atom_id*128+coefficient_bin_id',pair_vocabulary=131072,shared_event_vocabulary=1152,
   coefficient_quantization='signed asinh companding10; fixed per-scale training-calibrated bounds',
   calibration_images=512,calibration_seed=26110713,calibration_quantile=.9995,calibration_margin=1.1,
   beam_cost='physical squared reconstruction error after coefficient quantization',
   teacher='beam16 complete compound-event trajectory posterior; exact sampled event-prefix conditionals')
  torch.save(spec,inputs/'compound128-codebook.pt')
  cache=torch.load(cfg['smoke_cache'],map_location='cpu',weights_only=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):z=model.quant_conv(model.encoder(cache['images'][:2].cuda()))
  for teacher in [False,True]:
   result=q.decompose(z,teacher=teacher);decoded,teacher_inputs=q.from_codes(result['codes'])
   torch.testing.assert_close(decoded,result['latent'],atol=0,rtol=0);torch.testing.assert_close(teacher_inputs,result['inputs'],atol=0,rtol=0)
   assert result['codes'].shape==(2,340,2) and result['coefficients'].dtype==torch.long and (result['coefficients']<128).all()
  val=ArchiveImages(cfg['val_archive'],cfg['val_index'],False,0,labels=cfg['val_labels']);checks=np.random.default_rng(26110714).choice(len(val),64,replace=False).tolist();totals=torch.zeros(5,device='cuda',dtype=torch.float64)
  for first in range(0,64,8):
   images=torch.stack([val[i][0] for i in checks[first:first+8]]).cuda()
   with torch.autocast('cuda',dtype=torch.bfloat16):z=model.quant_conv(model.encoder(images))
   compound=q.decompose(z);joint=original_q.decompose(z)
   with torch.autocast('cuda',dtype=torch.bfloat16):
    a=model.decoder(model.post_quant_conv(compound['latent'])).float().clamp(-1,1)
    b=model.decoder(model.post_quant_conv(joint['latent'])).float().clamp(-1,1)
   assert a.isfinite().all()
   totals+=torch.stack(((a-images).square().mean(),(b-images).square().mean(),(compound['latent']-z.float()).square().mean(),(joint['latent']-z.float()).square().mean(),compound['clip_fraction'])).double()*len(images)
 quality=(totals/64).cpu().tolist()
 report=dict(passed=True,source_checkpoint_sha256=LASER_HASH,source_stage1_epoch=10,source_greedy_rfid=cfg['source_greedy_stage1_rfid'],new_codec_full_validation_rfid=None,
  atoms=1024,coefficient_bins=128,coefficient_max=bounds.tolist(),training_calibration_images=512,companding=10.,beam_width=16,
  compound_pairs_per_image=680,scalar_events_per_image=1360,shared_event_vocabulary=1152,packed_pair_vocabulary=131072,nominal_bits_per_image=11560,
  quantized_beam_rescoring=True,canonical_integer_roundtrip_bit_exact=True,sampled_teacher_inputs_bit_exact=True,
  validation_check_images=64,compound_psnr=-10*math.log10(quality[0]/4),source_joint_psnr=-10*math.log10(quality[1]/4),
  compound_latent_mse=quality[2],source_joint_latent_mse=quality[3],validation_clip_fraction=quality[4],
  frozen_tokenizer_digest=digest_state(model),stage1_adaptation=False,time=time.time())
 write(BASE/'codec-verification.json',report);write(OUT/'codec-verification.json',report);print(json.dumps(report),flush=True)
 del model,original_q,continuous;torch.cuda.empty_cache()
 shutil.copyfile(cfg['fid_reference'],inputs/'validation-fid-reference.npz')
 code=(runtime/'train_beam_joint_stage2.py').read_text()
 def replace(a,b):
  nonlocal code
  assert code.count(a)==1,(a,code.count(a));code=code.replace(a,b)
 replace('from frozen_beam_tokenizer import load_source, LASER_HASH, digest_state','from frozen_compound128_tokenizer import load_source, LASER_HASH, digest_state\nfrom types import SimpleNamespace')
 replace('from src.models.beam_joint_var import BeamAlignedJointVAR','from src.models.compound_beam128_var import BeamAlignedCompound128VAR')
 replace("torch.testing.assert_close(self.q.coefficients.cpu(),spec['coefficients'],rtol=0,atol=0)","torch.testing.assert_close(self.q.coefficient_max.cpu(),spec['coefficient_max'],rtol=0,atol=0)")
 replace("if (self.q.vocab_size,self.q.num_directions,self.q.levels)!=(4096,1024,4):raise ValueError('Requires the trained matched680/4096-symbol tokenizer')", "if self.q.vocab_size!=1024 or self.q.coefficient_bins!=128:raise ValueError('Requires the compound128 tokenizer')")
 replace('self.model=BeamAlignedJointVAR(self.q,depth=16,num_classes=1000)','self.model=BeamAlignedCompound128VAR(SimpleNamespace(quantize=self.q),depth=16,num_classes=1000,token_width=256,token_layers=2,token_heads=4)')
 replace("group='imagenet-var-beam16-joint4096-stage2-20261106'","group='imagenet-var-compound128-stage2-20261107'")
 replace("'tokenizer/tokens_per_image':680,'tokenizer/vocabulary':4096,'tokenizer/nominal_bits_per_image':8160", "'tokenizer/compound_pairs_per_image':680,'tokenizer/scalar_events_per_image':1360,'tokenizer/atom_vocabulary':1024,'tokenizer/coefficient_bins':128,'tokenizer/shared_event_vocabulary':1152,'tokenizer/packed_pair_vocabulary':131072,'tokenizer/nominal_bits_per_image':11560")
 replace("'tokenizer/pair_format':'atom_id*4+level_id','tokenizer/separate_coefficient_stream':False", "'tokenizer/pair_format':'atom_id*128+coefficient_bin_id','training/compound_formulation':True")
 replace("int(result['codes'].max())<4096","int(result['codes'].max())<131072")
 replace("loss=self.model(labels,code['inputs'],code['codes']);wrong=self.model((labels+1)%1000,code['inputs'],code['codes'])", "loss=self.model(labels,code['inputs'],code['atoms'],code['coefficients']);wrong=self.model((labels+1)%1000,code['inputs'],code['atoms'],code['coefficients'])")
 replace("loss=self.wrapped(labels,codes['inputs'],codes['codes'],codes['targets'])", "loss=self.wrapped(labels,codes['inputs'],codes['atoms'],codes['coefficients'],codes['targets'])")
 replace("loss_sum=torch.zeros((),device=self.device);tick", "loss_sum=torch.zeros((),device=self.device);parts_sum=torch.zeros(2,device=self.device);tick")
 replace("                loss_sum+=loss.detach()/self.accum", "                loss_sum+=loss.detach()/self.accum;parts_sum+=torch.stack((self.model.last_losses['atom_nll'],self.model.last_losses['coefficient_nll']))/self.accum")
 replace("                dist.all_reduce(loss_sum);loss_sum/=self.world", "                dist.all_reduce(loss_sum);loss_sum/=self.world;dist.all_reduce(parts_sum);parts_sum/=self.world")
 replace("joint_nll=float(loss_sum),uniform_joint_nll=(math.log(4096)+math.log(4092))*.5,gain_vs_uniform=(math.log(4096)+math.log(4092))*.5-float(loss_sum)", "joint_nll=float(loss_sum),atom_nll=float(parts_sum[0]),coefficient_nll=float(parts_sum[1]),uniform_joint_nll=math.log(128)+(math.log(1024)+math.log(1023))*.5,gain_vs_uniform=math.log(128)+(math.log(1024)+math.log(1023))*.5-float(loss_sum)")
 replace("                loss_sum.zero_();tick", "                loss_sum.zero_();parts_sum.zero_();tick")
 replace("tokens_per_image=680,vocabulary=4096,pair_id='atom_id*4+level_id',beam_width=16,beam_aligned_teacher=True,separate_coefficient_stream=False", "compound_pairs_per_image=680,scalar_events_per_image=1360,atom_vocabulary=1024,coefficient_bins=128,shared_event_vocabulary=1152,packed_pair_vocabulary=131072,pair_id='atom_id*128+bin_id',beam_width=16,beam_aligned_teacher=True,compound_formulation=True")
 driver=runtime/'train_compound128_stage2.py';driver.write_text(code);py_compile.compile(str(driver),doraise=True)
 cfg.update(output=str(BASE/'run'),checkpoint=str(checkpoint),codebook=str(inputs/'compound128-codebook.pt'),codebook_sha256=sha(inputs/'compound128-codebook.pt'),
  source_bundle=str(BASE/'source.tar.gz'),fid_reference=str(inputs/'validation-fid-reference.npz'),lr=1e-4,vocabulary=1152,atoms=1024,coefficient_bins=128,
  coefficient_representation='128 signed asinh-companded bins',coefficient_max=bounds.tolist(),coefficient_companding=10.,
  coefficient_bin_calibration=dict(training_images=512,seed=26110713,quantile=.9995,margin=1.1),
  pair_token_format='atom_id*128+coefficient_bin_id',compound_formulation=True,compound_pairs_per_image=680,scalar_events_per_image=1360,
  shared_event_vocabulary=1152,packed_pair_vocabulary=131072,nominal_bits_per_image=11560,
  local_token_transformer=dict(width=256,layers=2,heads=4),teacher='beam16_aligned_quantized_compound_event_posterior',
  beam_teacher=dict(sequence_temperature_relative=.01,max_relative_candidate_error=.01,cost='physical error AFTER coefficient quantization',conditioning='exact sampled atom/coefficient event prefix'),
  peak_lr_reason='Joint4096 train/validation NLL worsened and gradient norms jumped when warmup reached3e-4; lower peak1e-4 for the compound trial',
  fresh_prior=True,stage1_adaptation=False,wandb=dict(id=OUT.name,name='ImageNet LASER | compound128 | beam16 aligned | VAR-d16 | LR1e-4 | 4 GPUs'))
 for k in ['directions','coefficient_levels','joint_tokens_per_image','separate_coefficient_stream']:cfg.pop(k,None)
 cfg['comparison'].update(optimizer='AdamW .9/.95, LR1e-4, weight decay .05, clip1.0, 4epoch warmup, lin0 floor0.1, 200epochs',
  token_budget='680 compound pairs: 1024-way atom +128-bin coefficient, 11560 nominal bits/image',
  representation_difference='Compound128: 1360 conditional scalar events with shared vocabulary1152; VQ: 680 categorical IDs with vocabulary4096',
  encoding='beam16 continuous-OLS support bank; coefficients quantized to128 bins; candidates rescored with decoded physical bin values',
  nominal_storage_bits=dict(laser=11560,vq=8160),equal_nominal_bits=False,peak_lr=dict(compound=1e-4,vq=3e-4))
 keys=['train_compound128_stage2.py','frozen_compound128_tokenizer.py','frozen_tokenizer.py','src/models/compound_beam128_var.py','src/models/compound_multiscale_var.py','src/models/continuous_beam_var.py','src/models/beam_joint_var.py','src/models/joint_multiscale_var.py','src/models/trainable_joint_var_tokenizer.py','src/models/multiscale_laser_var.py','src/rqvae_metrics.py','src/original_rq_training.py','third_party/FoundationVision_VAR/models/var.py','third_party/FoundationVision_VAR/models/basic_var.py','third_party/FoundationVision_VAR/utils/lr_control.py']
 cfg['runtime_sha256']={name:sha(runtime/name) for name in keys}
 (BASE/'run').mkdir(exist_ok=True);write(BASE/'config.json',cfg);write(OUT/'config.json',cfg);write(OUT/'comparison.json',cfg['comparison'])
 for micro,accum in [(192,1),(96,2),(64,3)]:
  c=dict(cfg,output=str(BASE/f'preflight-mb{micro}'),microbatch=micro,accumulation=accum);Path(c['output']).mkdir(exist_ok=True);write(BASE/f'preflight-mb{micro}.json',c)
 with tarfile.open(BASE/'source.tar.gz','w:gz') as tar:
  for p in runtime.rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts:tar.add(p,arcname='runtime/'+str(p.relative_to(runtime)))
 print(json.dumps(dict(prepared=True,run_id=OUT.name,stage='stage2',atoms=1024,coefficient_bins=128,compound=True,lr=cfg['lr'])),flush=True)
if __name__=='__main__':main()
