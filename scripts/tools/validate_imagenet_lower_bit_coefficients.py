import argparse,json,os,time,hashlib,math
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader,Subset
from src.training.imagenet_compound_runtime import make_aux
from src.training.rq_imagenet_runtime import TarImageNet,write_json
from src.training.rqtransformer import val_image_transform
from src.rqvae_metrics import DistributedOriginalRQVAEMetrics, _mean_covariance, frechet_distance

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--samples',type=int,default=5000);p.add_argument('--bins',type=int,nargs='+',default=[16,32,64,128,256,512]);a=p.parse_args()
 rank,world,local=[int(os.environ[k]) for k in ['RANK','WORLD_SIZE','LOCAL_RANK']]
 torch.set_num_threads(4);device=torch.device('cuda',local);torch.cuda.set_device(device);dist.init_process_group('nccl',device_id=device)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 base=Path('/tmp/laser-imagenet-compound512-audit-20261026');aux=make_aux(base/'inputs/stage1-tokenizer.pt',device);aux.clamp_coeffs=False
 def digest():
  h=hashlib.sha256()
  for name,v in aux.state_dict().items():h.update(name.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
  return h.hexdigest()
 before=digest();ds=TarImageNet(base/'data','val',val_image_transform());assert a.samples%1000==0
 ix=[]
 for label in range(1000):ix.extend(np.flatnonzero(ds.labels==label)[:a.samples//1000].tolist())
 assert len(ix)==a.samples
 loader=DataLoader(Subset(ds,ix[rank::world]),batch_size=16,num_workers=2,multiprocessing_context='spawn',pin_memory=True)
 bins=sorted(set(a.bins+[512]));real=DistributedOriginalRQVAEMetrics(device)
 metrics={k:DistributedOriginalRQVAEMetrics(device,inception=real.inception) for k in ['continuous']+[str(n) for n in bins]}
 centers={n:torch.linspace(-3,3,n,device=device) for n in bins};errors=torch.zeros(len(bins),3,device=device,dtype=torch.float64);seen=0;started=time.monotonic();a.output.mkdir(parents=True,exist_ok=True)
 with torch.inference_mode():
  for batch,(images,_) in enumerate(loader):
   images=images.to(device);atoms,values=aux.encode_sparse_components(images);vectors=aux.dictionary.T[atoms]
   def decode(c):
    v=c*aux.coeff_scales.view(1,1,1,4);z=(vectors*v[...,None]).sum(-2).permute(0,3,1,2).contiguous()
    return aux.decoder(aux.post_quant_conv(z)).clamp(-1,1)
   real.update(images*.5+.5,real=True);metrics['continuous'].update(decode(values)*.5+.5,real=False)
   for j,n in enumerate(bins):
    ids=(values[...,None]-centers[n]).abs().argmin(-1);q=centers[n][ids];recon=decode(q)
    if batch==0:
     saved=aux.coeff_bins,aux.coeff_vocab_size,aux.vocab_size;aux.coeff_bins,aux.coeff_vocab_size,aux.vocab_size=centers[n],n,16384+n
     expected=aux.decode_tokens(torch.stack((atoms,ids+16384),-1).flatten(-2));torch.testing.assert_close(recon,expected,rtol=1e-5,atol=1e-5)
     aux.coeff_bins,aux.coeff_vocab_size,aux.vocab_size=saved
    metrics[str(n)].update(recon*.5+.5,real=False);d=q-values;errors[j]+=torch.stack((d.square().sum(),(d*aux.coeff_scales).square().sum(),d.new_tensor(d.numel())))
   seen+=len(images)
   if rank==0 and batch%10==0:
    r=dict(images_per_rank=seen,samples=a.samples,bins=bins,seconds=time.monotonic()-started);write_json(a.output/'progress.json',r);print(json.dumps(r),flush=True)
 dist.all_reduce(errors);measured={}
 for key,metric in metrics.items():
  metric.real_sum.copy_(real.real_sum);metric.real_cross.copy_(real.real_cross);metric.real_count.copy_(real.real_count);metric._reduce_fid_state()
  assert int(metric.real_count)==int(metric.fake_count)==a.samples
 for i,(key,metric) in enumerate(metrics.items()):
  if i%world==rank:
   mr,cr=_mean_covariance(metric.real_sum,metric.real_cross,int(metric.real_count));mf,cf=_mean_covariance(metric.fake_sum,metric.fake_cross,int(metric.fake_count));measured[key]=frechet_distance(mr,cr,mf,cf)
   print(json.dumps(dict(codec=key,rfid=measured[key],rank=rank)),flush=True)
 gathered=[None]*world;dist.all_gather_object(gathered,measured);measured={k:v for d in gathered for k,v in d.items()}
 assert digest()==before
 if rank==0:
  results=[]
  for j,n in enumerate(bins):
   delta=measured[str(n)]-measured['512'];results.append(dict(bins=n,coefficient_bits=int(math.log2(n)),nominal_bits_per_image=256*(14+int(math.log2(n))),rfid=measured[str(n)],delta_rfid_vs_512=delta,coefficient_mse=float(errors[j,0]/errors[j,2]),physical_coefficient_mse=float(errors[j,1]/errors[j,2]),full_gate_passed=a.samples==50000 and measured[str(n)]<4.4 and delta<.1))
  result=dict(samples=a.samples,continuous_rfid=measured['continuous'],results=results,frozen_weights_unchanged=True,token_codec_parity_passed=True,world_size=world,seconds=time.monotonic()-started)
  write_json(a.output/'result.json',result);print(json.dumps(result),flush=True)
 dist.destroy_process_group()
if __name__=='__main__':main()
