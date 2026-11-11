"""Prepare stage2 with one integer per pair and a beam16-aligned categorical teacher."""
from pathlib import Path
import json,shutil,sys,hashlib,time,tarfile,py_compile,math
import numpy as np
import torch
BASE=Path('/tmp/laser-var-stage2-beam-joint4096-20261106')
OLD=Path('/tmp/laser-var-stage2-matched680-20261106')
OUT=Path('/workspace/Projects/laser/outputs/imnet-s2-laser-beam16-joint4096-var-d16-4gpu-20261106')
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,default=str)+'\n')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 here=Path(__file__).resolve().parent;runtime=BASE/'runtime';inputs=BASE/'inputs';inputs.mkdir(parents=True,exist_ok=True);OUT.mkdir(exist_ok=True)
 if not runtime.exists():shutil.copytree(OLD/'runtime',runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 if (BASE/'run/prior-last.pt').exists():raise RuntimeError('Do not replace an active production training contract')
 source=Path('/workspace/Projects/laser/src/models/beam_joint_var.py')
 shutil.copyfile(source if source.exists() else here/'beam_joint_var.py',runtime/'src/models/beam_joint_var.py')
 shutil.copyfile(here/'frozen_beam_tokenizer.py',runtime/'frozen_beam_tokenizer.py')
 sys.path.insert(0,str(runtime))
 from frozen_beam_tokenizer import load_source,LASER_HASH,digest_state
 from frozen_tokenizer import load_source as load_original
 from train_stage2 import ArchiveImages
 cfg=json.loads((OLD/'config.json').read_text())
 checkpoint=inputs/'source-stage1.pt'
 if not checkpoint.exists():checkpoint.hardlink_to(Path(cfg['checkpoint']))
 assert sha(checkpoint)==LASER_HASH
 book=torch.load(cfg['codebook'],map_location='cpu',weights_only=True)
 book.update(kind='beam16_joint_atom_coefficient',encoding_beam_width=16,pair_id='atom_id*4+coefficient_level_id',
  coefficient_representation='learned_four_levels_per_atom_per_scale',continuous_coefficients_quantized=True,
  separate_coefficient_stream=False,teacher='beam16 complete-trajectory posterior with exact sampled joint-prefix conditionals',
  teacher_sequence_temperature_relative=.01,teacher_max_relative_error=.01,source_encoding='greedy',source_greedy_rfid=cfg['stage1_rfid'],beam_rfid=None)
 torch.save(book,inputs/'joint-codebook.pt')
 shutil.copyfile(cfg['fid_reference'],inputs/'validation-fid-reference.npz')
 model=load_source(checkpoint,'laser','cuda:0');original=load_original(checkpoint,'laser','cuda:0');q=model.quantize
 torch.testing.assert_close(q.dictionary.cpu(),book['dictionary'],atol=0,rtol=0)
 torch.testing.assert_close(q.coefficients.cpu(),book['coefficients'],atol=0,rtol=0)
 cache=torch.load(cfg['smoke_cache'],map_location='cpu',weights_only=True)
 with torch.no_grad():
  with torch.autocast('cuda',dtype=torch.bfloat16):z=model.quant_conv(model.encoder(cache['images'][:2].cuda()))
  for teacher in [False,True]:
   result=q.decompose(z,teacher=teacher);decoded,inputs_forced=q.from_codes(result['codes'])
   torch.testing.assert_close(decoded,result['latent'],atol=0,rtol=0);torch.testing.assert_close(inputs_forced,result['inputs'],atol=0,rtol=0)
   assert result['codes'].dtype==torch.long and result['codes'].shape==(2,340,2)
   assert (result['codes']>=0).all() and (result['codes']<4096).all()
  # This small common-subset check is not a full-validation rFID result.
  val=ArchiveImages(cfg['val_archive'],cfg['val_index'],False,0,labels=cfg['val_labels'])
  indices=np.random.default_rng(26110623).choice(len(val),64,replace=False).tolist();totals=torch.zeros(4,device='cuda',dtype=torch.float64)
  for first in range(0,len(indices),8):
   images=torch.stack([val[i][0] for i in indices[first:first+8]]).cuda()
   with torch.autocast('cuda',dtype=torch.bfloat16):z=model.quant_conv(model.encoder(images))
   beam=q.decompose(z);greedy=original.quantize.decompose(z)
   assert beam['beam_assignment_mse']<=beam['greedy_anchor_mse']+1e-6
   with torch.autocast('cuda',dtype=torch.bfloat16):
    a=model.decoder(model.post_quant_conv(beam['latent'])).float().clamp(-1,1)
    b=original.decoder(original.post_quant_conv(greedy['latent'])).float().clamp(-1,1)
   assert a.isfinite().all()
   totals+=torch.stack(((a-images).square().mean(),(b-images).square().mean(),(beam['latent']-z.float()).square().mean(),(greedy['latent']-z.float()).square().mean())).double()*len(images)
 quality=(totals/64).cpu().tolist()
 report=dict(passed=True,source_checkpoint_sha256=LASER_HASH,source_stage1_epoch=10,source_greedy_rfid=cfg['stage1_rfid'],new_beam_codec_full_rfid=None,
  beam_width=16,atoms=1024,coefficient_levels_per_atom=4,vocabulary=4096,tokens_per_image=680,nominal_bits_per_image=8160,
  pair_id='atom_id*4+coefficient_level_id',separate_coefficient_stream=False,decoder_requires_only_joint_ids=True,
  canonical_integer_roundtrip_bit_exact=True,sampled_teacher_inputs_bit_exact=True,beam_assignment_no_worse_than_greedy_anchor=True,
  validation_check_images=64,beam_psnr=-10*math.log10(quality[0]/4),greedy_psnr=-10*math.log10(quality[1]/4),
  beam_latent_mse=quality[2],greedy_latent_mse=quality[3],frozen_tokenizer_digest=digest_state(model),stage1_adaptation=False,time=time.time())
 write(BASE/'codec-verification.json',report);write(OUT/'codec-verification.json',report);print(json.dumps(report),flush=True)
 del model,original;torch.cuda.empty_cache()
 code=(runtime/'train_stage2.py').read_text()
 def replace(a,b):
  nonlocal code
  assert code.count(a)==1,(a,code.count(a));code=code.replace(a,b)
 replace('from frozen_tokenizer import load_source, LASER_HASH, digest_state','from frozen_beam_tokenizer import load_source, LASER_HASH, digest_state')
 replace('from src.models.joint_multiscale_var import JointMultiScaleQuantizer, JointLaserVAR','from src.models.beam_joint_var import BeamAlignedJointVAR')
 replace('        self.q.quantize_scale=functools.partial(self.q.quantize_scale,chunk_size=2048)\n','')
 replace('self.model=JointLaserVAR(self.q,depth=16,num_classes=1000)','self.model=BeamAlignedJointVAR(self.q,depth=16,num_classes=1000)')
 replace("group='imagenet-var-matched680-stage2-20261106'","group='imagenet-var-beam16-joint4096-stage2-20261106'")
 replace("'stage1/full_validation_rfid':cfg['stage1_rfid']","'tokenizer/source_greedy_stage1_rfid':cfg['source_greedy_stage1_rfid']")
 replace("'evaluation/cadence':'epoch_end_only'})","'evaluation/cadence':'epoch_end_only','tokenizer/beam_width':16,'tokenizer/pair_format':'atom_id*4+level_id','tokenizer/separate_coefficient_stream':False,'teacher/beam_aligned':True})")
 replace('    def tokenize(self,images):','    def tokenize(self,images,*,teacher=False):')
 replace("        with torch.autocast('cuda',enabled=False):return self.q.decompose(latent)","        with torch.autocast('cuda',enabled=False):\n            result=self.q.decompose(latent,teacher=teacher)\n            if self.smoke:\n                assert result['codes'].dtype==torch.long and result['codes'].shape[1:]==(340,2)\n                assert int(result['codes'].min())>=0 and int(result['codes'].max())<4096\n                decoded,inputs=self.q.from_codes(result['codes'])\n                torch.testing.assert_close(decoded,result['latent'],atol=0,rtol=0)\n                torch.testing.assert_close(inputs,result['inputs'],atol=0,rtol=0)\n            return result")
 replace("codes=self.tokenize(images.to(self.device,non_blocking=True));sync=","codes=self.tokenize(images.to(self.device,non_blocking=True),teacher=True);sync=")
 replace("loss=self.wrapped(labels,codes['inputs'],codes['codes'])","loss=self.wrapped(labels,codes['inputs'],codes['codes'],codes['targets'])")
 replace("tokens_per_image=680,vocabulary=4096,gradient_clip", "tokens_per_image=680,vocabulary=4096,pair_id='atom_id*4+level_id',beam_width=16,beam_aligned_teacher=True,separate_coefficient_stream=False,actual_integer_roundtrip_bit_exact=True,gradient_clip")
 replace("        with evaluation_rng(self.device):\n            self.model.eval()\n            if self.inception is None:","        receipt=self.out/f\"fid-epoch{self.progress['epoch']:03d}.json\"\n        done=torch.tensor(int(receipt.exists()) if self.rank==0 else 0,device=self.device);dist.broadcast(done,src=0)\n        if done.item():return\n        with evaluation_rng(self.device):\n            self.model.eval()\n            if self.inception is None:")
 replace("        if max_steps and self.progress['step']>=max_steps:return","        if max_steps and self.progress['step']>=max_steps:return\n        if not self.smoke and self.progress['epoch']>0 and self.progress['batch']==0:self.fid()")
 driver=runtime/'train_beam_joint_stage2.py';driver.write_text(code);py_compile.compile(str(driver),doraise=True)
 cfg.update(output=str(BASE/'run'),checkpoint=str(checkpoint),codebook=str(BASE/'inputs/joint-codebook.pt'),codebook_sha256=sha(BASE/'inputs/joint-codebook.pt'),
  source_bundle=str(BASE/'source.tar.gz'),fid_reference=str(BASE/'inputs/validation-fid-reference.npz'),source_greedy_stage1_rfid=cfg.pop('stage1_rfid'),tokenizer_full_validation_rfid=None,
  coefficient_representation='categorical_learned_levels',pair_token_format='atom_id*4+coefficient_level_id',separate_coefficient_stream=False,
  beam_width=16,beam_chunk_size=256,teacher='beam16_aligned_complete_joint_trajectory_posterior',
  beam_teacher=dict(sequence_temperature_relative=.01,max_relative_candidate_error=.01,conditioning='exact sampled joint-ID prefix',prefixes='one sampled complete beam trajectory per site; same codewords form next-scale inputs'),
  stage1_adaptation=False,wandb=dict(id=OUT.name,name='ImageNet LASER | beam16 joint4096 | 1 token per pair | aligned teacher | VAR-d16 | 4 GPUs'))
 cfg['comparison'].update(encoding='beam16 complete joint-ID trajectories with greedy anchor',nominal_storage_bits=dict(laser=8160,vq=8160),equal_nominal_bits=True)
 keys=['train_beam_joint_stage2.py','frozen_beam_tokenizer.py','frozen_tokenizer.py','src/models/beam_joint_var.py','src/models/joint_multiscale_var.py','src/models/trainable_joint_var_tokenizer.py','src/models/multiscale_laser_var.py','src/rqvae_metrics.py','src/original_rq_training.py','third_party/FoundationVision_VAR/models/var.py','third_party/FoundationVision_VAR/models/basic_var.py','third_party/FoundationVision_VAR/utils/lr_control.py']
 cfg['runtime_sha256']={name:sha(runtime/name) for name in keys}
 (BASE/'run').mkdir(exist_ok=True);write(BASE/'config.json',cfg);write(OUT/'config.json',cfg);write(OUT/'comparison.json',cfg['comparison'])
 for micro,accum in [(192,1),(96,2),(64,3)]:
  trial=dict(cfg,output=str(BASE/f'preflight-mb{micro}'),microbatch=micro,accumulation=accum);Path(trial['output']).mkdir(exist_ok=True);write(BASE/f'preflight-mb{micro}.json',trial)
 with tarfile.open(BASE/'source.tar.gz','w:gz') as tar:
  for p in runtime.rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts:tar.add(p,arcname='runtime/'+str(p.relative_to(runtime)))
 print(json.dumps(dict(prepared=True,output=str(OUT),vocabulary=4096,tokens_per_image=680,beam_width=16,teacher='beam_aligned',stage='stage2')),flush=True)
if __name__=='__main__':main()
