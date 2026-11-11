"""Prepare native OMP / 128-bin compound training with ordinary hard CE."""
from pathlib import Path
import json,shutil,sys,hashlib,time,tarfile,py_compile,math
import numpy as np
import torch
BASE=Path('/tmp/laser-var-stage2-compound128-omp-20261107')
OLD=Path('/tmp/laser-var-stage2-compound128-20261107')
OUT=Path('/workspace/Projects/laser/outputs/imnet-s2-laser-compound128-omp-var-d16-4gpu-20261107')
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,default=str)+'\n')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 here=Path(__file__).resolve().parent;runtime=BASE/'runtime';inputs=BASE/'inputs';inputs.mkdir(parents=True,exist_ok=True);OUT.mkdir(exist_ok=True)
 if (BASE/'run/prior-last.pt').exists():raise RuntimeError('Do not replace an active production training contract')
 shutil.copyfile(here/'frozen_native_compound128_tokenizer.py',runtime/'frozen_native_compound128_tokenizer.py')
 sys.path.insert(0,str(runtime))
 from frozen_tokenizer import load_source as load_original,LASER_HASH,digest_state
 from src.models.native_compound128_var import NativeCompound128Quantizer
 from train_stage2 import ArchiveImages
 assert not any('beam' in n for n in sys.modules if n.startswith('src.models.'))
 cfg=json.loads((OLD/'config.json').read_text());checkpoint=inputs/'source-stage1.pt'
 if not checkpoint.exists():checkpoint.hardlink_to(Path(cfg['checkpoint']))
 assert sha(checkpoint)==LASER_HASH
 model=load_original(checkpoint,'laser','cuda:0');original_q=model.quantize
 q=NativeCompound128Quantizer(original_q).eval().requires_grad_(False)
 train=ArchiveImages(cfg['train_archive'],cfg['train_index'],True,0)
 indices=np.random.default_rng(26110715).choice(len(train),512,replace=False).tolist();values=[[] for _ in range(4)]
 with torch.no_grad():
  for first in range(0,len(indices),16):
   images=torch.stack([train[i][0] for i in indices[first:first+16]]).cuda()
   with torch.autocast('cuda',dtype=torch.bfloat16):z=model.quant_conv(model.encoder(images))
   result=q.decompose_raw(z);offset=0
   for scale,pn in enumerate(q.v_patch_nums):
    values[scale].append(result['physical_coefficients'][:,offset:offset+pn*pn].flatten().cpu());offset+=pn*pn
   print(json.dumps(dict(phase='native_omp_coefficient_calibration',images=first+len(images))),flush=True)
  quantiles=torch.stack([torch.quantile(torch.cat(v).abs(),.9995) for v in values])
  learned_max=original_q.physical_tables()[1].detach().cpu().abs().flatten(1).amax(-1)
  bounds=torch.maximum(quantiles*1.1,learned_max*1.01).clamp_min(.1)
  q.coefficient_max.copy_(bounds);model.quantize=q;model.eval().requires_grad_(False)
  spec=dict(kind='native_omp_compound128',format_version=1,source_checkpoint_sha256=LASER_HASH,source_artifact=cfg['tokenizer_artifact'],
   dictionary=q.dictionary.cpu(),coefficient_max=bounds,companding=q.companding.cpu(),
   coefficient_grid=torch.stack([q.coefficient_values(torch.arange(128,device='cuda'),s).cpu() for s in range(4)]),
   coefficient_bins=128,atoms=1024,sparsity=2,patch_nums=list(q.v_patch_nums),encoding='native_greedy_omp',omp_ridge=1e-5,omp_max_coherence=.999,
   coefficient_quantization='signed asinh companding10; fixed per-scale training-calibrated bounds',
   calibration_images=512,calibration_seed=26110715,calibration_quantile=.9995,calibration_margin=1.1,calibration_encoder='single-path native OMP',
   pair_id='atom_id*128+coefficient_bin_id',pair_vocabulary=131072,shared_event_vocabulary=1152,training_labels='hard OMP atom IDs and hard coefficient-bin IDs')
  torch.save(spec,inputs/'compound128-codebook.pt')
  cache=torch.load(cfg['smoke_cache'],map_location='cpu',weights_only=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):z=model.quant_conv(model.encoder(cache['images'][:2].cuda()))
  before=torch.get_rng_state().clone();cuda_before=torch.cuda.get_rng_state().clone();first=q.decompose(z);second=q.decompose(z)
  assert torch.equal(first['codes'],second['codes']) and torch.equal(torch.get_rng_state(),before) and torch.equal(torch.cuda.get_rng_state(),cuda_before)
  assert 'targets' not in first and not hasattr(q,'beam_pool')
  decoded,teacher_inputs=q.from_codes(first['codes']);torch.testing.assert_close(decoded,first['latent'],atol=0,rtol=0);torch.testing.assert_close(teacher_inputs,first['inputs'],atol=0,rtol=0)
  val=ArchiveImages(cfg['val_archive'],cfg['val_index'],False,0,labels=cfg['val_labels']);checks=np.random.default_rng(26110716).choice(len(val),64,replace=False).tolist();totals=torch.zeros(3,device='cuda',dtype=torch.float64)
  for first in range(0,64,8):
   images=torch.stack([val[i][0] for i in checks[first:first+8]]).cuda()
   with torch.autocast('cuda',dtype=torch.bfloat16):z=model.quant_conv(model.encoder(images))
   native=q.decompose(z)
   with torch.autocast('cuda',dtype=torch.bfloat16):a=model.decoder(model.post_quant_conv(native['latent'])).float().clamp(-1,1)
   assert a.isfinite().all();totals+=torch.stack(((a-images).square().mean(),(native['latent']-z.float()).square().mean(),native['clip_fraction'])).double()*len(images)
 quality=(totals/64).cpu().tolist()
 report=dict(passed=True,source_checkpoint_sha256=LASER_HASH,source_stage1_epoch=10,atoms=1024,coefficient_bins=128,coefficient_max=bounds.tolist(),
  native_omp=True,single_greedy_support_path=True,full_least_squares_refit=True,beam_search=False,soft_teacher=False,hard_atom_and_bin_ce=True,
  training_calibration_images=512,calibration_encoding='native_greedy_omp',targets_deterministic=True,target_generation_preserves_cpu_cuda_rng=True,
  no_candidate_sampling=True,canonical_integer_roundtrip_bit_exact=True,quantized_teacher_inputs_bit_exact=True,validation_check_images=64,
  psnr=-10*math.log10(quality[0]/4),latent_mse=quality[1],validation_clip_fraction=quality[2],
  frozen_tokenizer_digest=digest_state(model),new_codec_full_validation_rfid=None,stage1_adaptation=False,time=time.time())
 write(BASE/'codec-verification.json',report);write(OUT/'codec-verification.json',report);print(json.dumps(report),flush=True)
 del model,original_q,q;torch.cuda.empty_cache();shutil.copyfile(cfg['fid_reference'],inputs/'validation-fid-reference.npz')
 code=(runtime/'train_compound128_stage2.py').read_text()
 def replace(a,b):
  nonlocal code
  assert code.count(a)==1,(a,code.count(a));code=code.replace(a,b)
 replace('from frozen_compound128_tokenizer import load_source, LASER_HASH, digest_state','from frozen_native_compound128_tokenizer import load_source, LASER_HASH, digest_state')
 replace('from src.models.compound_beam128_var import BeamAlignedCompound128VAR','from src.models.native_compound128_var import NativeCompound128VAR')
 replace('self.model=BeamAlignedCompound128VAR(', 'self.model=NativeCompound128VAR(')
 replace("group='imagenet-var-compound128-stage2-20261107'", "group='imagenet-var-native-compound128-stage2-20261107'")
 replace("'tokenizer/beam_width':16,", "'tokenizer/encoding':'native_greedy_omp',")
 replace("'teacher/beam_aligned':True", "'teacher/type':'hard_omp_atom_and_bin_labels','teacher/beam_search':False,'teacher/soft_targets':False")
 replace('    def tokenize(self,images,*,teacher=False):','    def tokenize(self,images):')
 replace('result=self.q.decompose(latent,teacher=teacher)','result=self.q.decompose(latent)')
 replace('codes=self.tokenize(images.to(self.device,non_blocking=True),teacher=True)','codes=self.tokenize(images.to(self.device,non_blocking=True))')
 replace("loss=self.wrapped(labels,codes['inputs'],codes['atoms'],codes['coefficients'],codes['targets'])", "loss=self.wrapped(labels,codes['inputs'],codes['atoms'],codes['coefficients'])")
 replace('            assert experiment.updates==1668',"            assert experiment.updates==1668\n            assert not hasattr(experiment.q,'beam_pool')\n            import sys\n            assert not any('beam' in n for n in sys.modules if n.startswith('src.models.'))")
 replace('beam_width=16,beam_aligned_teacher=True,compound_formulation=True','native_omp=True,beam_search=False,soft_teacher=False,hard_atom_and_bin_ce=True,compound_formulation=True')
 driver=runtime/'train_native_compound128_stage2.py';driver.write_text(code);py_compile.compile(str(driver),doraise=True)
 cfg.update(output=str(BASE/'run'),checkpoint=str(checkpoint),codebook=str(inputs/'compound128-codebook.pt'),codebook_sha256=sha(inputs/'compound128-codebook.pt'),
  source_bundle=str(BASE/'source.tar.gz'),fid_reference=str(inputs/'validation-fid-reference.npz'),coefficient_max=bounds.tolist(),
  coefficient_bin_calibration=dict(training_images=512,seed=26110715,quantile=.9995,margin=1.1,encoding='native_greedy_omp'),
  teacher='native_omp_hard_atom_and_coefficient_bin_labels',encoding='native_greedy_omp',omp_ridge=1e-5,omp_max_support_coherence=.999,
  beam_search=False,soft_teacher=False,target_trajectory_sampling=False,loss='ordinary hard atom CE plus coefficient-bin CE',
  fresh_prior=True,stage1_adaptation=False,prior_initialization='fresh random; no prior training checkpoint loaded',
  wandb=dict(id=OUT.name,name='ImageNet LASER | native OMP compound128 | hard CE | VAR-d16 | LR1e-4 | 4 GPUs'))
 for key in ['beam_width','beam_chunk_size','beam_teacher']:cfg.pop(key,None)
 cfg['comparison'].update(encoding='native single-path greedy OMP; full coefficient refit;128-bin quantization',teacher='ordinary hard OMP atom/bin labels',prior_warm_start=False)
 keys=['train_native_compound128_stage2.py','frozen_native_compound128_tokenizer.py','frozen_tokenizer.py','src/models/native_compound128_var.py','src/models/compound_multiscale_var.py','src/models/joint_multiscale_var.py','src/models/trainable_joint_var_tokenizer.py','src/models/multiscale_laser_var.py','src/models/dictionary_learner.py','src/rqvae_metrics.py','src/original_rq_training.py','third_party/FoundationVision_VAR/models/var.py','third_party/FoundationVision_VAR/models/basic_var.py','third_party/FoundationVision_VAR/utils/lr_control.py']
 cfg['runtime_sha256']={name:sha(runtime/name) for name in keys}
 (BASE/'run').mkdir(exist_ok=True);write(BASE/'config.json',cfg);write(OUT/'config.json',cfg);write(OUT/'comparison.json',cfg['comparison'])
 for micro,accum in [(192,1),(96,2),(64,3)]:
  c=dict(cfg,output=str(BASE/f'preflight-mb{micro}'),microbatch=micro,accumulation=accum);Path(c['output']).mkdir(exist_ok=True);write(BASE/f'preflight-mb{micro}.json',c)
 with tarfile.open(BASE/'source.tar.gz','w:gz') as tar:
  for p in runtime.rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts:tar.add(p,arcname='runtime/'+str(p.relative_to(runtime)))
 print(json.dumps(dict(prepared=True,run_id=OUT.name,stage='stage2',encoding='native_greedy_omp',hard_ce=True,beam_search=False)),flush=True)
if __name__=='__main__':main()
