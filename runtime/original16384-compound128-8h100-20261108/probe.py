"""Training-only range calibration and paired full-validation codec evaluation."""
import os,sys,json,time,argparse,hashlib
from pathlib import Path
B=Path(os.environ.get('LASER_BEAM_BASE','/tmp/laser-original16384-compound128-8h100-20261108'));O=Path('/workspace/Projects/laser/outputs/imagenet-original16384-compound128-scratch100-8h100-20261108')
sys.path[:0]=[str(B/'source'),str(B)];os.environ['RQ_UPSTREAM_ROOT']=str(B/'upstream');os.environ['TORCH_HOME']=str(B/'torch-cache')
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader,Subset
from src import rq_upstream
from src.models.laser_tokenizer_adapter import FrozenLaserTokenizer
from src.training.laser_hard_ar import sparse_encoding
from src.training.rq_imagenet_runtime import TarImageNet,write_json
from rqvae.img_datasets.transforms import create_transforms
from rqvae.utils.config import load_config
from runtime.compound_recipe import CompoundRecipe

def dataset(split,transform_split=None):
 c=load_config(B/'inputs/stage2-training-config.yaml')
 return TarImageNet(B/'data',split,create_transforms(c.dataset,split=transform_split or split))

@torch.inference_mode()
def calibrate():
 import torch.distributed as dist
 rank,world,local=[int(os.environ[x]) for x in ['RANK','WORLD_SIZE','LOCAL_RANK']]
 torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(4)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;dist.init_process_group('nccl',device_id=device)
 payload=torch.load(B/'inputs/stage1-model.pt',map_location='cpu',weights_only=False,mmap=True)
 state=payload['state_dict'];dictionary_key='quantizer.dictionary' if 'quantizer.dictionary' in state else 'bottleneck.dictionary'
 raw_dictionary=state[dictionary_key].float();assert raw_dictionary.shape==(256,16384)
 dictionary=torch.nn.functional.normalize(raw_dictionary,dim=0).clone()
 dictionary_sha=hashlib.sha256(dictionary.numpy().tobytes()).hexdigest()
 raw_dictionary_sha=hashlib.sha256(raw_dictionary.numpy().tobytes()).hexdigest()
 with (B/'inputs/stage1-model.pt').open('rb') as stream:checkpoint_sha=hashlib.file_digest(stream,'sha256').hexdigest()
 assert checkpoint_sha=='dd28db9306d526bdc9fbf8016403e106c4f317fd8f5c04792f0953e53310fdab'
 del payload,state,raw_dictionary
 assert dictionary.shape==(256,16384) and (dictionary.norm(dim=0)-1).abs().max()<1e-4
 aux=FrozenLaserTokenizer(B/'inputs/stage1-model.pt',num_atoms=16384,coeff_vocab_size=128,coeff_max=3.,coeff_scales=[1.]*4,sparsity_level=4)
 aux.dictionary=dictionary.clone();aux=aux.to(device).requires_grad_(False).eval();gram=aux.dictionary.T@aux.dictionary
 data=dataset('train');import numpy as np
 rng=np.random.default_rng(261108128)
 indices=np.stack([rng.choice(np.flatnonzero(data.labels==c),50,replace=False) for c in range(1000)]).T.reshape(-1)
 assert len(indices)==len(np.unique(indices))==50000
 if rank==0:np.save(O/'coefficient-calibration-indices.npy',indices)
 values=[]
 for i,(images,_) in enumerate(DataLoader(Subset(data,indices[rank::world].tolist()),batch_size=8,num_workers=4,pin_memory=True)):
  z=aux.quant_conv(aux.encoder(images.to(device))).permute(0,2,3,1).contiguous();_,c=sparse_encoding(z,aux.dictionary,4,gram=gram);values.append(c.reshape(-1,4))
  if rank==0 and i%50==0: print(json.dumps(dict(phase='coefficient_calibration',images=min(50000,(i+1)*8*world),total=50000)),flush=True)
 values=torch.cat(values).contiguous();received=[torch.empty_like(values) for _ in range(world)] if rank==0 else None
 dist.gather(values,gather_list=received,dst=0)
 if rank==0:
  values=torch.cat(received).cpu();quantiles=torch.quantile(values.abs(),.9995,dim=0)
  bounds=(quantiles*1.05).clamp_min(.1);scales=bounds/3.
  codec=dict(dictionary=dictionary.clone(),coefficient_bins=128,coefficient_scales=scales,coefficient_max=3.,dictionary_sha256=dictionary_sha,dictionary_source='original_stage1_checkpoint',dictionary_checkpoint_key=dictionary_key,source_checkpoint_sha256=checkpoint_sha,raw_dictionary_sha256=raw_dictionary_sha,training_images=50000,calibration_seed=261108128)
  torch.save(codec,B/'inputs/compound-codec.pending');(B/'inputs/compound-codec.pending').replace(B/'inputs/compound-codec.pt')
  recipe=OmegaConf.load(B/'inputs/recipe.yaml');recipe.tokenizer.coefficient_scales=scales.tolist();recipe.tokenizer.dictionary_sha256=dictionary_sha;recipe.tokenizer.dictionary_checkpoint_key=dictionary_key;recipe.targets.coefficient_range_calibration='50000 class-balanced training images, 50/class; q0.9995 with 5% margin covering the original 16384-atom Stage1 coefficients';OmegaConf.save(recipe,B/'inputs/recipe.yaml')
  report=dict(images=50000,classes=1000,images_per_class=50,latent_sites=int(len(values)),split='train',validation_used=False,seed=261108128,training_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),quantile=.9995,margin=1.05,physical_bounds=bounds.tolist(),coefficient_scales=scales.tolist(),coefficient_steps=(2*bounds/127).tolist(),calibration_clip_fraction=(values.abs()>bounds).float().mean(0).tolist(),coefficient_bins=128,atoms=16384,source_checkpoint_sha256=checkpoint_sha,dictionary_sha256=dictionary_sha,raw_dictionary_sha256=raw_dictionary_sha,dictionary_checkpoint_key=dictionary_key,dictionary_source='original_stage1_checkpoint',time=time.time())
  write_json(O/'coefficient-calibration.json',report);print(json.dumps(report),flush=True)
 dist.barrier();dist.destroy_process_group()

@torch.inference_mode()
def quality():
 import torch.distributed as dist
 from src import rqvae_metrics as m
 rank,world,local=[int(os.environ[x]) for x in ['RANK','WORLD_SIZE','LOCAL_RANK']];device=torch.device('cuda',local);torch.cuda.set_device(device);torch.set_num_threads(4)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;dist.init_process_group('nccl',device_id=device)
 profile=CompoundRecipe(B);aux,_=profile.tokenizer(device);before=profile.digest(aux)
 m.INCEPTION_PATH=B/'upstream/rqvae/metrics/inception.py';inception=m.OriginalRQVAEInception().to(device).eval().requires_grad_(False)
 metrics={k:m.DistributedOriginalRQVAEMetrics(device,inception=inception) for k in ['hard','soft','continuous']}
 data=dataset('val');loader=DataLoader(Subset(data,range(rank,len(data),world)),batch_size=8,num_workers=4,pin_memory=True)
 errors=torch.zeros(4,dtype=torch.float64,device=device);seen=0;started=time.monotonic()
 for i,(images,_) in enumerate(loader):
  images=images.to(device);atoms,values=profile.clean_components(aux,images)
  ids=(values[...,None]-aux.coeff_bins).abs().argmin(-1)
  sigma=float(profile.recipe.targets.coefficient_sigma_bins)*float(aux.coeff_bins[1]-aux.coeff_bins[0]);prob=(-(values[...,None]-aux.coeff_bins).square()/(2*sigma*sigma)).softmax(-1)
  sampled=torch.multinomial(prob.reshape(-1,128),1).reshape_as(ids)
  vectors=aux.dictionary.T[atoms]
  latents={'continuous':(vectors*(values*aux.coeff_scales)[...,None]).sum(-2),'hard':(vectors*(aux.coeff_bins[ids]*aux.coeff_scales)[...,None]).sum(-2),'soft':(vectors*(aux.coeff_bins[sampled]*aux.coeff_scales)[...,None]).sum(-2)}
  real=images.mul(.5).add(.5).clamp(0,1)
  for name,z in latents.items():
   decoded=aux.decoder(aux.post_quant_conv(z.permute(0,3,1,2).contiguous())).clamp(-1,1);metrics[name].update(real,real=True);metrics[name].update(decoded.mul(.5).add(.5),real=False)
  errors[0]+=((latents['hard']-latents['continuous']).square().mean())*len(images)
  errors[1]+=((latents['soft']-latents['continuous']).square().mean())*len(images)
  errors[2]+=(values.abs()>3).float().mean()*len(images);errors[3]+=len(images);seen+=len(images)
  if rank==0 and i%25==0:write_json(O/'quality-progress.json',dict(images=seen*world,total=50000,seconds=time.monotonic()-started))
 result={k:metric.compute()[0] for k,metric in metrics.items()};dist.all_reduce(errors);assert profile.digest(aux)==before
 if rank==0:
  proof=dict(passed=all(torch.isfinite(torch.tensor(v)) for v in result.values()) and result['hard']<4.73,rfid_hard=result['hard'],rfid_soft=result['soft'],rfid_continuous=result['continuous'],images=50000,real_split='val',hard_quantization_latent_mse=float(errors[0]/errors[3]),soft_quantization_latent_mse=float(errors[1]/errors[3]),clip_fraction=float(errors[2]/errors[3]),original_dictionary=True,dictionary_shape=list(aux.dictionary.shape),dictionary_sha256=hashlib.sha256(aux.dictionary.cpu().numpy().tobytes()).hexdigest(),source_original_reported_rfid=4.210914134979248,stage1_CNN_unchanged=True,dictionary_unchanged=True,frozen_codec_digest=before,coefficient_bins=128,atoms=16384,time=time.time())
  write_json(B/'tokenizer-verification.json',proof);write_json(O/'tokenizer-verification.json',proof);print(json.dumps(proof),flush=True)
 dist.destroy_process_group()

if __name__=='__main__':
 mode=sys.argv[1]
 if mode=='calibrate':calibrate()
 elif mode=='quality':quality()
