"""Match CC3M teacher corruption to released RQ using training data only."""
import argparse,json,os,sys
from pathlib import Path
base=Path('/tmp/laser-cc3m-rq24576-20261111')
sys.path.insert(0,str(base/'source'))
import numpy as np
import torch
from omegaconf import OmegaConf
from src import rq_upstream
from src.models.native_vector_tokenizer import NativeVectorTokenizer,module_digest,file_digest
from src.data.cc3m_joint import CC3MJointDataset,write_json
from rqvae.models import create_model
from rqvae.utils.config import load_config,augment_arch_defaults

@torch.inference_mode()
def main():
 torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
 cfg=OmegaConf.to_container(OmegaConf.load(base/'config.yaml'),resolve=True)
 aux=NativeVectorTokenizer(cfg['stage1'],cfg['codebook']).cuda()
 assert module_digest(aux.backbone)==aux.spec['source_cnn_digest']
 original_dir=next((base/'reference').glob('**/stage1/config.yaml')).parent
 assert file_digest(original_dir/'model.pt')=='e8eb312e6c0dbd8e490ca95f69e3c3b417162625ff9320bbfff791cbf832b694'
 oldcfg=load_config(original_dir/'config.yaml')
 old,_=create_model(augment_arch_defaults(oldcfg.arch),ema=False)
 old.load_state_dict(torch.load(original_dir/'model.pt',weights_only=False,map_location='cpu')['state_dict'],strict=True)
 old=old.requires_grad_(False).eval().cuda()
 specs=json.loads(Path(cfg['data_manifest']).read_text())['train']
 # Even coverage across eight shards; disjoint fit/heldout, no image validation.
 selected=[specs[i] for i in np.linspace(0,len(specs)-1,8,dtype=int)]
 ds=CC3MJointDataset(selected,cfg['data_source'],cfg['data_local'],train=True,seed=cfg['seed'])
 rng=np.random.default_rng(261111);starts=np.r_[0,np.cumsum([x['items'] for x in selected])[:-1]]
 indices=np.concatenate([start+rng.choice(s['items'],128,replace=False) for start,s in zip(starts,selected)])
 rng.shuffle(indices)
 cache=[[],[]];identities=[]
 for first in range(0,len(indices),8):
  pixels=torch.stack([ds[int(i)][0] for i in indices[first:first+8]]).cuda()
  cache[0].append(aux.encode(pixels).cpu());cache[1].append(old.encode(pixels).cpu())
  if first%128==0: print(json.dumps(dict(encoded=first)),flush=True)
 cache=[torch.cat(x) for x in cache];stats={}
 profiles=[('original_greedy',1,0),('original_soft',1,.5),('native_greedy',0,0)]+[(f'native_{t:g}',0,t) for t in [.06,.075,.085,.1,.125,.15,.2,.25,.5]]
 for name,kind,temp in profiles:
  q=old.quantizer if kind else aux.quantizer;parts=[];entropy=torch.zeros(4,device='cuda',dtype=torch.float64)
  for first in range(0,len(indices),8):
   torch.manual_seed(781111+first)
   x=cache[kind][first:first+8].cuda()
   if temp:
    soft,codes=q.get_soft_codes(x,temp=temp,stochastic=True)
    entropy+=-(soft*soft.clamp_min(1e-30).log()).sum((0,1,2,4)).double();del soft
   else: _,codes=q.quantize(x)
   decoded=q.embed_code(codes)
   parts.append(torch.stack([(x-decoded).square().flatten(1).sum(1),x.square().flatten(1).sum(1)],1).double().cpu())
  values=torch.cat(parts).numpy();stats[name]=dict(temperature=temp,fit_sse=float(values[:512,0].sum()),heldout_sse=float(values[512:,0].sum()),entropy_per_depth=(entropy/(1024*64)).tolist())
  print(json.dumps(dict(profile=name,**stats[name])),flush=True)
 ratios={split:stats['original_soft'][split+'_sse']/stats['original_greedy'][split+'_sse'] for split in ['fit','heldout']}
 for name,v in stats.items():
  if name.startswith('native_') and name!='native_greedy':
   for split in ['fit','heldout']:v[split+'_ratio']=v[split+'_sse']/stats['native_greedy'][split+'_sse']
 choices=[v for k,v in stats.items() if k.startswith('native_') and k!='native_greedy']
 winner=min(choices,key=lambda v:abs(v['fit_ratio']-ratios['fit']))
 # 0.2 percentage point tolerance in relative MSE on an independent half.
 passed=abs(winner['heldout_ratio']-ratios['heldout'])<.002
 result=dict(passed=passed,selected_temperature=winner['temperature'],selected=winner,original_ratios=ratios,
  profiles=stats,fit_images=512,heldout_training_images=512,validation_used=False,
  selection_criterion='relative sampled-to-greedy latent squared error',shards=selected,indices=indices.tolist(),
  frozen_CNN_verified=True,integer_interface=True,codebook_sha256=file_digest(cfg['codebook']))
 write_json(base/'live/reproducibility/noise-calibration.json',result)
 assert passed,result
 cfg['temperatures']=winner['temperature']
 OmegaConf.save(OmegaConf.create(cfg),base/'config.yaml')
 print(json.dumps(dict(passed=passed,temperature=winner['temperature'],original_ratios=ratios,selected=winner)),flush=True)
if __name__=='__main__':main()
