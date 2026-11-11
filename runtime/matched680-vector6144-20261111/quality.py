"""Distributed training-only vector fitting and independent full validation."""
import argparse, hashlib,json,os,time
from pathlib import Path
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.data import DataLoader,Subset
from frozen_tokenizer import load_source,digest_state,LASER_HASH
from train_vector_stage2 import ArchiveImages
from src.models.residual_vector_var import ResidualVectorQuantizer,AnchoredResidualVectorQuantizer
from src.original_rq_training import FeatureMoments
from src.rqvae_metrics import OriginalRQVAEInception,frechet_distance
BASE=Path('/tmp/laser-matched680-vector6144-20261111')
OUT=Path('/workspace/Projects/laser/outputs/imnet-s2-matched680-vector6144-anchor-var-d16-4h100-20261111')

def write(path,v):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.pending');tmp.write_text(json.dumps(v,indent=2,default=str)+'\n');tmp.replace(path)
def save(path,v):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.pending');torch.save(v,tmp);tmp.replace(path)
def log(phase,**v):
 if dist.get_rank()==0:
  row=dict(phase=phase,time=time.time(),**v);write(OUT/'preparation-status.json',row);print(json.dumps(row),flush=True)
def loader(data,indices,batch=64):
 return DataLoader(Subset(data,list(map(int,indices))),batch_size=batch,num_workers=4,pin_memory=True,multiprocessing_context='spawn',prefetch_factor=2)
def dataset(cfg,train):return ArchiveImages(cfg['train_archive'] if train else cfg['val_archive'],cfg['train_index'] if train else cfg['val_index'],train,0,labels=None if train else cfg['val_labels'])
def make_q(source,path,device,gain=1.):
 v=torch.load(path,map_location=device,weights_only=True)
 anchored=isinstance(v,dict) and v.get('kind')=='multiscale_anchored_vector_ids'
 if isinstance(v,dict):v=v['vectors']
 if anchored:return AnchoredResidualVectorQuantizer(source.quantize,v*gain,temperature=.5).to(device).eval()
 return ResidualVectorQuantizer(source.quantize.source,v*gain,temperature=.5).to(device).eval()

@torch.inference_mode()
def cache(cfg,rank,world,device):
 source=load_source(cfg['checkpoint'],'laser',device);before=digest_state(source)
 cfg=dict(cfg,train_archive=str(BASE/'data/ILSVRC2012_img_train.tar') if (BASE/'data/ILSVRC2012_img_train.tar.verified.json').exists() else '/workspace/Projects/data/imagenet/archives/ILSVRC2012_img_train.verified-20261005.tar')
 data=dataset(cfg,True);rng=np.random.default_rng(2611116144);fit=[];held=[]
 for label in range(1000):
  ids=rng.choice(np.flatnonzero(data.labels==label),60,replace=False);fit.extend(ids[:50]);held.extend(ids[50:])
 rng.shuffle(fit);rng.shuffle(held)
 assert set(fit).isdisjoint(held)
 if rank==0:
  np.save(BASE/'quality/fit-indices.npy',fit);np.save(BASE/'quality/held-indices.npy',held)
 for name,indices in [('fit',fit),('held',held)]:
  part=indices[rank::world];path=BASE/'cache'/f'{name}-rank{rank}.npy';path.parent.mkdir(exist_ok=True)
  arr=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(len(part),32,16,16));first=0;start=time.monotonic()
  network=OriginalRQVAEInception().to(device).eval().requires_grad_(False) if name=='held' else None
  features=np.lib.format.open_memmap(BASE/'cache'/f'held-features-rank{rank}.npy',mode='w+',dtype='float32',shape=(len(part),2048)) if network is not None else None
  for images,_ in loader(data,part):
   with torch.autocast('cuda',dtype=torch.bfloat16):z=source.quant_conv(source.encoder(images.to(device,non_blocking=True)))
   arr[first:first+len(images)]=z.float().cpu().numpy()
   if features is not None:features[first:first+len(images)]=network(images.to(device,non_blocking=True).float().mul(.5).add(.5))[0].cpu().numpy()
   first+=len(images)
   if first%1024==0:log('cache',split=name,images_per_rank=first,target=len(part),aggregate_images_per_second=first*world/(time.monotonic()-start))
  arr.flush();assert first==len(part)
  if features is not None:features.flush()
 assert before==digest_state(source)
 dist.barrier()
 if rank==0:write(OUT/'cache-verification.json',dict(passed=True,fit_images=50000,heldout_training_images=10000,disjoint=True,class_balanced=True,validation_used=False,source_digest=before,augmentation='Resize288 Lanczos, deterministic RandomCrop256, no horizontal flip; matches VAR training'))

@torch.inference_mode()
def initialize(cfg,source,x,rank,world,device):
 qold=source.quantize;dictionary,coefficients=qold.physical_tables()
 base=(dictionary.T[None,:,None,:]*coefficients[...,None]).reshape(4,4096,32)
 vectors=torch.zeros(4,2,6144,32,device=device);vectors[:,:,1:4097]=base[:,None]
 # Draw extra prototypes from actual training residuals, without validation.
 z=x[:1024];acc=torch.zeros_like(z);rng=torch.Generator(device=device).manual_seed(261111+rank)
 for scale,pn in enumerate(qold.v_patch_nums):
  r=F.interpolate(z-acc,(pn,pn),mode='area').flatten(2).transpose(1,2).reshape(-1,32).contiguous();total=torch.zeros_like(r)
  for depth in range(2):
   table=base[scale];d=torch.cdist(r,table).square();errors,ids=d.min(-1)
   count=(2047+world-1)//world
   pick=torch.multinomial(errors.clamp_min(1e-8),count,replacement=len(r)<count,generator=rng)
   proposed=r[pick].contiguous();gathered=[torch.empty_like(proposed) for _ in range(world)];dist.all_gather(gathered,proposed)
   vectors[scale,depth,4097:]=torch.cat(gathered)[:2047]
   val=table[ids];r=r-val;total+=val
  acc+=qold.contribution(total.reshape(len(z),pn*pn,32),scale)
 return vectors

@torch.inference_mode()
def fit(cfg,rank,world,device):
 source=load_source(cfg['checkpoint'],'laser',device)
 x=torch.from_numpy(np.load(BASE/'cache'/f'fit-rank{rank}.npy')).to(device)
 held=torch.from_numpy(np.load(BASE/'cache'/f'held-rank{rank}.npy')).to(device)
 vectors=initialize(cfg,source,x,rank,world,device)
 q=ResidualVectorQuantizer(source.quantize.source,vectors).to(device)
 trace=[];start=time.monotonic();snapshots=[0,2,4,8,16,24]
 for iteration in range(25):
  counts=torch.zeros(4,2,6144,device=device);stored_codes=[];stored_residuals=[]
  total=torch.zeros(4,device=device,dtype=torch.float64)
  for part in x.split(128):
   result=q.decompose(part);error=(part-result['latent']).square();total[0]+=error.sum();total[1]+=error.numel()
   stored_codes.append(result['codes']);stored_residuals.append(result['residuals'][:,:,0].contiguous())
   offset=0
   for scale,pn in enumerate(q.v_patch_nums):
    for depth in range(2):counts[scale,depth]+=torch.bincount(result['codes'][:,offset:offset+pn*pn,depth].flatten(),minlength=6144)
    offset+=pn*pn
  for part in held.split(256):
   result=q.decompose(part);error=(part-result['latent']).square();total[2]+=error.sum();total[3]+=error.numel()
  dist.all_reduce(total);dist.all_reduce(counts)
  row=dict(iteration=iteration,fit_latent_mse=float(total[0]/total[1]),heldout_latent_mse=float(total[2]/total[3]),used_codewords=(counts>0).sum(-1).cpu().tolist(),elapsed_seconds=time.monotonic()-start)
  trace.append(row);log('fit',**row)
  if rank==0:
   write(OUT/'fit-trace.json',trace)
   if iteration in snapshots:save(BASE/'quality'/f'pass-{iteration:03d}.pt',q.vectors.cpu())
  if iteration==24:break
  assignments=torch.cat(stored_codes);residuals=torch.cat(stored_residuals);offset=0
  generator=torch.Generator(device=device).manual_seed(2611116144+rank+iteration*world)
  for scale,pn in enumerate(q.v_patch_nums):
   local=assignments[:,offset:offset+pn*pn].reshape(-1,2)
   error=residuals[:,offset:offset+pn*pn].reshape(-1,32).clone()
   for depth in range(2):error.sub_(q.embed(local[:,depth],scale,depth))
   for depth in range(2):
    old=q.vectors[scale,depth].clone();index=local[:,depth];sums=torch.zeros_like(old)
    for first in range(0,len(error),8192):
     ids=index[first:first+8192];sums.index_add_(0,ids,error[first:first+len(ids)]+old[ids])
    dist.all_reduce(sums)
    proposal=(sums+8.*old)/(counts[scale,depth,:,None]+8.);proposal[0]=0
    unused=torch.nonzero(counts[scale,depth]==0).flatten();unused=unused[unused!=0]
    if len(unused):
     amount=(len(unused)+world-1)//world
     ids=torch.multinomial(error.square().sum(-1).clamp_min(1e-8),amount,replacement=False,generator=generator)
     candidates=(error[ids]+old[index[ids]]).contiguous();parts=[torch.empty_like(candidates) for _ in range(world)]
     dist.all_gather(parts,candidates);proposal[unused]=torch.cat(parts)[:len(unused)]
    assert torch.isfinite(proposal).all()
    q.vectors[scale,depth].copy_(proposal)
    for first in range(0,len(error),8192):
     ids=index[first:first+8192];error[first:first+len(ids)].sub_(proposal[ids]-old[ids])
   offset+=pn*pn
  del stored_codes,stored_residuals,assignments,residuals
 if rank==0:
  selected=min([r for r in trace if r['iteration'] in snapshots],key=lambda r:r['heldout_latent_mse'])
  write(BASE/'quality/selected-pass.json',selected)
 dist.barrier()

@torch.inference_mode()
def select(cfg,rank,world,device):
 source=load_source(cfg['checkpoint'],'laser',device)
 held=torch.from_numpy(np.load(BASE/'cache'/f'held-rank{rank}.npy')).to(device)
 # Select on image reconstruction FID using only the disjoint training holdout.
 trace=json.loads((OUT/'fit-trace.json').read_text());snapshots=[0,2,4,8,16,24]
 top=sorted([r for r in trace if r['iteration'] in snapshots],key=lambda r:r['heldout_latent_mse'])[:2]
 passes=sorted(set([0]+[r['iteration'] for r in top]));results=[]
 real=FeatureMoments(device);real.update(torch.from_numpy(np.load(BASE/'cache'/f'held-features-rank{rank}.npy')).to(device))
 n,real_mu,real_cov=real.finish();assert n==10000
 network=OriginalRQVAEInception().to(device).eval().requires_grad_(False)
 for iteration in passes:
  path=BASE/'quality'/f'pass-{iteration:03d}.pt'
  for gain in [1.,1.025]:
   q=make_q(source,path,device,gain);v=torch.zeros(2,device=device,dtype=torch.float64);moment=FeatureMoments(device)
   for part in held.split(64):
    r=q.decompose(part);v[0]+=(part-r['latent']).square().sum();v[1]+=part.numel()
    with torch.autocast('cuda',dtype=torch.bfloat16):images=torch.cat([source.decoder(source.post_quant_conv(chunk)).float().clamp(-1,1) for chunk in r['latent'].split(16)])
    moment.update(network(images.mul(.5).add(.5))[0])
   dist.all_reduce(v);count,mu,cov=moment.finish();assert count==10000
   score=torch.tensor(frechet_distance(real_mu,real_cov,mu,cov) if rank==0 else 0.,device=device,dtype=torch.float64);dist.broadcast(score,0)
   row=dict(iteration=iteration,gain=gain,heldout_latent_mse=float(v[0]/v[1]),heldout_rfid=float(score));results.append(row)
   log('training_holdout_screen',**row)
   if rank==0:write(OUT/'holdout-screen.json',results)
 best=min(results,key=lambda r:r['heldout_rfid']);selected=dict(iteration=best['iteration']);path=BASE/'quality'/f"pass-{best['iteration']:03d}.pt"
 q=make_q(source,path,device,best['gain'])
 # Select a physical temperature with the same mean entropy as the newest 24k teacher.
 probes=[]
 for part in held[:128].split(32):
  r=q.decompose(part);offset=0
  for scale,pn in enumerate(q.v_patch_nums):
   for depth in range(2):
    residual=r['residuals'][:,offset:offset+pn*pn,depth].reshape(-1,32)
    probes.append(q.distances(residual[::max(1,len(residual)//256)],scale,depth))
   offset+=pn*pn
 distances=torch.cat(probes);lo,hi=1e-7,1.
 for _ in range(32):
  mid=(lo+hi)/2;lp=(-distances/mid).log_softmax(-1);entropy=(-(lp.exp()*lp).sum(-1)).mean();dist.all_reduce(entropy);entropy/=world
  if entropy<2.12:lo=mid
  else:hi=mid
 temperature=(lo+hi)/2
 if rank==0:
  spec=dict(kind='multiscale_residual_vector_ids',vectors=q.vectors.cpu(),source_checkpoint_sha256=LASER_HASH,vocabulary=6144,patch_nums=[2,4,8,16],sparsity=2,zero_id=0,fit_images=50000,heldout_training_images=10000,fit_passes=24,selected_pass=selected['iteration'],gain=best['gain'],teacher_temperature=temperature,teacher_entropy_target=2.12,source_dictionary_unchanged=True)
  save(BASE/'quality/selected-codebook.pt',spec)
  write(OUT/'selection.json',dict(selected_pass=selected['iteration'],gain=best['gain'],candidates=results,selected_by='disjoint_training_image_rfid',heldout_rfid=best['heldout_rfid'],temperature=temperature,mean_teacher_entropy=float(entropy),selection_split='10000 disjoint training images',validation_used=False))
  config=json.loads((BASE/'config.json').read_text());config['teacher_temperature']=temperature
  with (BASE/'quality/selected-codebook.pt').open('rb') as f:config['codebook_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
  write(BASE/'config.json',config);write(OUT/'config.json',config)
 dist.barrier()

@torch.inference_mode()
def verify(cfg,rank,world,device):
 source=load_source(cfg['checkpoint'],'laser',device);before=digest_state(source);q=make_q(source,cfg['codebook'],device);q.temperature=cfg['teacher_temperature']
 inception=OriginalRQVAEInception().to(device).eval().requires_grad_(False)
 moments=[FeatureMoments(device),FeatureMoments(device)];sse=torch.zeros(3,device=device,dtype=torch.float64)
 data=dataset(cfg,False);start=time.monotonic();done=0;torch.manual_seed(2611116144+rank)
 for images,_ in loader(data,range(rank,len(data),world),batch=64):
  images=images.to(device,non_blocking=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):z=source.quant_conv(source.encoder(images))
  for kind in range(2):
   result=q.decompose(z,stochastic=bool(kind))
   decoded,inputs=q.from_codes(result['codes'])
   torch.testing.assert_close(decoded,result['latent'],rtol=0,atol=0);torch.testing.assert_close(inputs,result['inputs'],rtol=0,atol=0)
   with torch.autocast('cuda',dtype=torch.bfloat16):recon=torch.cat([source.decoder(source.post_quant_conv(part)) for part in decoded.split(16)]).float().clamp(-1,1)
   moments[kind].update(inception((recon+1)*.5)[0]);sse[kind]+=(recon-images).square().sum()
  sse[2]+=images.numel();done+=len(images)
  if done%1024==0:log('full_validation',images_per_rank=done,target=12500,elapsed_seconds=time.monotonic()-start)
 dist.all_reduce(sse);scores=[]
 for m in moments:
  n,mu,cov=m.finish()
  if rank==0:
   ref=np.load(cfg['fid_reference']);scores.append(frechet_distance(ref['mu'],ref['sigma'],mu,cov))
 assert before==digest_state(source)
 if rank==0:
  passed=all(np.isfinite(scores)) and scores[0] <= cfg['stage1_rfid']+.4 and scores[1]<=scores[0]+.8
  report=dict(passed=bool(passed),greedy_rfid=scores[0],stochastic_rfid=scores[1],images=50000,source_rfid=cfg['stage1_rfid'],reference_split='validation',reference_count=50000,psnr=[float(-10*torch.log10(sse[i]/sse[2]/4)) for i in range(2)],integer_roundtrip_bit_exact=True,source_frozen=True,source_digest=before,vocabulary=6144,tokens_per_image=680,teacher_temperature=q.temperature,tf32_enabled=False,cnn_precision='bf16',inception_precision='fp32',mean_teacher_entropy_target=2.12,selection_split='disjoint training',validation_used_for_fitting=False)
  write(OUT/'codec-verification.json',report);log('codec_verified',**report)
 flag=torch.tensor(int(passed) if rank==0 else 0,device=device)
 dist.broadcast(flag,0)
 assert flag.item(),'New vector codec failed reconstruction fidelity gate; inspect before starting prior'
 dist.barrier()

def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['cache','fit','select','verify']);a=p.parse_args()
 torch.set_num_threads(4);device=torch.device('cuda',int(os.environ['LOCAL_RANK']));torch.cuda.set_device(device);dist.init_process_group('nccl',timeout=timedelta(hours=2))
 if a.phase=='verify':
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 BASE.joinpath('quality').mkdir(exist_ok=True);cfg=json.loads((BASE/'config.json').read_text())
 try:globals()[a.phase](cfg,dist.get_rank(),dist.get_world_size(),device)
 finally:dist.destroy_process_group()
if __name__=='__main__':main()
