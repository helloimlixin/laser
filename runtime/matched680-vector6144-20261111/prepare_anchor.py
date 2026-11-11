"""Source-preserving vector catalog for the matched680 stage-1 checkpoint."""
import os,sys,json,hashlib,shutil
from pathlib import Path
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
BASE=Path('/tmp/laser-matched680-vector6144-20261111')
ROOT=Path('/workspace/Projects/laser')
OUT=ROOT/'outputs/imnet-s2-matched680-vector6144-anchor-var-d16-4h100-20261111'
sys.path.insert(0,str(BASE/'runtime/runtime'))
from frozen_tokenizer import load_source,digest_state,LASER_HASH
from src.models.residual_vector_var import AnchoredResidualVectorQuantizer
from src.original_rq_training import FeatureMoments
from src.rqvae_metrics import OriginalRQVAEInception,frechet_distance

def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.pending');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)

def main():
 rank=int(os.environ['LOCAL_RANK']);device=torch.device('cuda',rank);torch.cuda.set_device(device);torch.set_num_threads(4);dist.init_process_group('nccl',timeout=timedelta(minutes=30));OUT.mkdir(parents=True,exist_ok=True)
 try:
  cfg=json.loads((BASE/'config.json').read_text());source=load_source(cfg['checkpoint'],'laser',device);before=digest_state(source)
  dictionary,coef=source.quantize.physical_tables();original=(dictionary.T[None,:,None,:]*coef[...,None]).reshape(4,4096,32)
  mid=torch.stack(((coef[...,0]+coef[...,1])*.5,(coef[...,2]+coef[...,3])*.5),-1)
  extra=(dictionary.T[None,:,None,:]*mid[...,None]).reshape(4,2048,32)
  vectors=torch.zeros(4,2,6144,32,device=device);vectors[:,:,1:4097]=original[:,None];vectors[:,:,4097:]=extra[:,None,:2047]
  q=AnchoredResidualVectorQuantizer(source.quantize,vectors,temperature=.01).to(device)
  held=torch.from_numpy(np.load(BASE/'cache'/f'held-rank{rank}.npy')).to(device);probes=[]
  for part in held[:128].split(32):
   r=q.decompose(part);offset=0
   for scale,pn in enumerate(q.v_patch_nums):
    for depth in range(2):
     center=r['residuals'][:,offset:offset+pn*pn,depth].reshape(-1,32)
     probes.append(q.distances(center[::max(1,len(center)//256)],scale,depth))
    offset+=pn*pn
  distance=torch.cat(probes);lo,hi=1e-7,1.
  for _ in range(32):
   mid=(lo+hi)/2;lp=(-distance/mid).log_softmax(-1);h=(-(lp.exp()*lp).sum(-1)).mean();dist.all_reduce(h);h/=4
   if h<2.12:lo=mid
   else:hi=mid
  q.temperature=(lo+hi)/2
  net=OriginalRQVAEInception().to(device).eval().requires_grad_(False);mom=FeatureMoments(device);real=FeatureMoments(device)
  real.update(torch.from_numpy(np.load(BASE/'cache'/f'held-features-rank{rank}.npy')).to(device));torch.manual_seed(2611116144+rank)
  with torch.inference_mode():
   for part in held.split(64):
    old=source.quantize.decompose(part);hard=q.decompose(part)
    torch.testing.assert_close(old['latent'],hard['latent'],rtol=0,atol=0);torch.testing.assert_close(old['inputs'],hard['inputs'],rtol=0,atol=0)
    assert torch.equal(old['codes']+1,hard['codes'])
    r=q.decompose(part,stochastic=True);dec,inp=q.from_codes(r['codes'])
    torch.testing.assert_close(dec,r['latent'],rtol=0,atol=0);torch.testing.assert_close(inp,r['inputs'],rtol=0,atol=0)
    with torch.autocast('cuda',dtype=torch.bfloat16):images=torch.cat([source.decoder(source.post_quant_conv(p)).float().clamp(-1,1) for p in dec.split(16)])
    mom.update(net((images+1)*.5)[0])
  n,mu,cov=real.finish();nn,fmu,fcov=mom.finish();assert n==nn==10000 and before==digest_state(source)
  if rank==0:
   score=frechet_distance(mu,cov,fmu,fcov);baseline=json.loads((BASE/'source-heldout-baseline.json').read_text())['heldout_training_rfid']
   report=dict(passed=score<=baseline+.8,source_preserving_integer_encoding_bit_exact=True,sampled_integer_decoding_bit_exact=True,heldout_training_images=10000,original_heldout_rfid=baseline,stochastic_heldout_rfid=score,temperature=q.temperature,entropy_target=2.12,validation_used_for_fitting=False,original_atoms=1024,original_levels=4,vector_ids=6144,tokens_per_image=680,new_entries='2047 within-sign coefficient midpoints plus zero; all4096 original vectors retained',source_digest=before)
   write(OUT/'anchor-preflight.json',report);print(json.dumps(report),flush=True)
   if not report['passed']:raise RuntimeError('Anchored catalog stochastic teacher failed heldout quality gate')
   spec=dict(kind='multiscale_anchored_vector_ids',vectors=q.vectors.cpu(),source_checkpoint_sha256=LASER_HASH,vocabulary=6144,patch_nums=[2,4,8,16],sparsity=2,zero_id=0,source_preserving=True,teacher='anchor_centered_complete_vector_softmax_with_prefix_error_compensation',teacher_temperature=q.temperature)
   previous=BASE/'quality/selected-codebook.pt'
   if previous.exists() and not (BASE/'quality/rejected-raw-codebook.pt').exists():previous.rename(BASE/'quality/rejected-raw-codebook.pt')
   torch.save(spec,previous)
   cfg.update(teacher_temperature=q.temperature,codebook_sha256=hashlib.sha256(previous.read_bytes()).hexdigest(),output=str(BASE/'train-anchor'),wandb=dict(id=OUT.name,name='ImageNet | matched680 | source-preserving vector6144 | classcond VAR-d16 | 4H100'),tokenization=dict(kind='source_preserving_vector_id_catalog',tables=[4,2,6144,32],zero_id=0,original_vectors_retained=4096,additional_nonzero_vectors=2047,source_hard_codec_bit_exact=True,continuous_coefficient_stream=False),target_mode='anchor_centered_vector_soft_cross_entropy')
   cfg['runtime_sha256']['src/models/residual_vector_var.py']=hashlib.sha256((BASE/'runtime/runtime/src/models/residual_vector_var.py').read_bytes()).hexdigest();cfg['runtime_sha256']['train_vector_stage2.py']=hashlib.sha256((BASE/'runtime/runtime/train_vector_stage2.py').read_bytes()).hexdigest()
   write(BASE/'config.json',cfg);write(OUT/'config.json',cfg)
   oldout=ROOT/'outputs/imnet-s2-matched680-vector6144-var-d16-4h100-20261111'
   for name in ['cache-verification.json','source-verification.json']:
    shutil.copyfile(oldout/name,OUT/name)
   write(OUT/'selection.json',dict(kind=spec['kind'],selection_split='training contributions; disjoint training heldout fidelity check',source_original_vectors_unchanged=True,raw_vector_fit_rejected=5.554609952043791,**report))
   write(OUT/'fit-trace.json',dict(method='exact expansion of source finite vectors with fixed intermediate physical magnitudes',original_dictionary_unchanged=True,encoder_decoder_unchanged=True,anchor_preflight=report))
 finally:dist.destroy_process_group()
if __name__=='__main__':main()
