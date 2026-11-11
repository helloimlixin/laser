"""Measure sequence-posterior entropy and distortion on fixed training views."""
import os,sys,json,math,hashlib
from pathlib import Path
B=Path(os.environ['LASER_BEAM_BASE']);sys.path[:0]=[str(B/'source'),str(B)]
os.environ['RQ_UPSTREAM_ROOT']=str(B/'upstream')
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader,Subset
from src import rq_upstream
from src.models.rfid_preserved_joint_tokenizer import RFIDPreservedJointTokenizer
from src.training.rq_imagenet_runtime import TarImageNet
from rqvae.img_datasets.transforms import create_transforms
from rqvae.utils.config import load_config
from runtime.training_acceleration import install
install()
from runtime import beam_teacher
rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
torch.cuda.set_device(device);torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
dist.init_process_group('nccl',device_id=device)
aux=RFIDPreservedJointTokenizer(B/'inputs/stage1-model.pt',B/'inputs/joint-codebook.pt',B/'inputs/reconstruction-codebook.pt').to(device)
from runtime.preservation_guard import verify_frozen_codec
from runtime.aligned_recipe import AlignedBeamRecipe
verify_frozen_codec(AlignedBeamRecipe(B),aux)
dataset=TarImageNet(B/'data','train',create_transforms(load_config(B/'inputs/stage2-training-config.yaml').dataset,split='train'))
g=torch.Generator().manual_seed(261108);indices=torch.randperm(len(dataset),generator=g)[:2048].tolist()
loader=DataLoader(Subset(dataset,indices[rank::world]),batch_size=8,num_workers=2,pin_memory=True)
temperatures=[0.,.125,.25,.5,1.,2.,4.,8.,16.,32.,64.,256.]
bounds=[.01,.02,.05,.1,.2,1.]
combinations=[(b,t) for b in bounds for t in temperatures]
stats=torch.zeros(len(combinations),5,device=device,dtype=torch.float64);sites=0
with torch.inference_mode():
 for images,_ in loader:
  latent=aux.encode(images.to(device).float());pool=beam_teacher.beam_pool(aux.physical_book,latent)
  costs=pool['costs'];best=costs.gather(1,pool['canonical_index'][:,None]);relative=(costs-best).clamp_min(0)/best.clamp_min(1e-8)
  for i,(bound,t) in enumerate(combinations):
   w=beam_teacher.posterior(pool,t,max_relative_error=bound)
   entropy=-(w*w.clamp_min(1e-30).log()).sum(-1)
   canonical_probability=w.gather(1,pool['canonical_index'][:,None]).squeeze(1)
   stats[i,0]+=entropy.sum()/4
   stats[i,1]+=(w*relative).sum()
   stats[i,2]+=(1-canonical_probability).sum()
   stats[i,3]+=(w>0).sum()
   stats[i,4]+=(-w.clamp_min(1e-30).log()*w).sum()
  sites+=len(costs)
  if rank==0: print(json.dumps(dict(phase='calibrating',local_sites=sites)),flush=True)
dist.all_reduce(stats);n=torch.tensor(sites,device=device);dist.all_reduce(n);stats/=n
if rank==0:
 rows=[dict(max_relative_candidate_error=b,sequence_temperature=t,conditional_entropy_per_token=float(s[0]),expected_relative_latent_error=float(s[1]),probability_noncanonical=float(s[2]),supported_sequences=float(s[3])) for (b,t),s in zip(combinations,stats.cpu())]
 feasible=[r for r in rows if r['conditional_entropy_per_token']>=.2 and r['expected_relative_latent_error']<=.01 and r['max_relative_candidate_error']<=.1]
 if feasible: selected=min(feasible,key=lambda r:(r['expected_relative_latent_error'],r['conditional_entropy_per_token']))
 else:
  feasible=[r for r in rows if r['expected_relative_latent_error']<=.01 and r['max_relative_candidate_error']<=.1]
  selected=max(feasible,key=lambda r:r['conditional_entropy_per_token'])
 result=dict(training_images=len(indices),training_indices=indices,training_indices_sha256=hashlib.sha256(json.dumps(indices).encode()).hexdigest(),sites=int(n),split='train',validation_used=False,target_entropy=.2,max_mean_relative_latent_error=.01,baseline=next(r for r in rows if r['sequence_temperature']==2 and r['max_relative_candidate_error']==.01),selected=next(r for r in rows if r['max_relative_candidate_error']==.05 and r['sequence_temperature']==2.),grid_selection=selected,rows=rows,selection_rule='Use T=2 with cutoff5% to isolate cutoff versus control1%; temperature sweep verifies saturation; target entropy 0.2 is not attained under 1% expected excess distortion; FID superiority unmeasured')
 target=Path(os.environ['LASER_CALIBRATION_REPORT']);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(selected=selected,baseline=result['baseline'])),flush=True)
dist.destroy_process_group()
