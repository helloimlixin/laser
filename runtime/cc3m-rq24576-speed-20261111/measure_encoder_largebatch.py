import sys,json,time
from pathlib import Path
b=Path('/tmp/laser-cc3m-rq24576-20261111');sys.path.insert(0,str(b/'source'))
import torch
from omegaconf import OmegaConf
from src.training import cc3m_joint as t

torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(.3)
c=OmegaConf.to_container(OmegaConf.load(b/'config.yaml'),resolve=True);aux=t.build_aux(c,'cuda');ds=t.dataset(c,'validation')
x=torch.stack([ds[i][0] for i in range(128)]).cuda();torch.backends.cudnn.benchmark_limit=3
rows=[]
@torch.inference_mode()
def encode(batch,channels_last):
 return torch.cat([aux.encode(part.contiguous(memory_format=torch.channels_last) if channels_last else part) for part in x.split(batch)])
@torch.inference_mode()
def main():
 reference=encode(8,False);_,refcodes=aux.quantizer.quantize(reference)
 greedy_mse=(reference-aux.quantizer.embed_code(refcodes)).square().mean().item()
 for name,tf32,bench,channels,batch in [('original',False,False,False,8),('autotune_fp32_b32',False,True,False,32),('autotune_fp32_b64',False,True,False,64),('autotune_fp32_b128',False,True,False,128)]:
  torch.backends.cudnn.allow_tf32=tf32;torch.backends.cudnn.benchmark=bench
  aux.backbone.to(memory_format=torch.channels_last if channels else torch.contiguous_format)
  encode(batch,channels);torch.cuda.synchronize();times=[]
  for _ in range(3):
   start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
   start.record();actual=encode(batch,channels);end.record();torch.cuda.synchronize();times.append(start.elapsed_time(end))
  mse=(actual-reference).square().mean().item();_,codes=aux.quantizer.quantize(actual)
  row=dict(name=name,tf32_convolutions=tf32,autotune=bench,channels_last=channels,encoder_batch=batch,
   ms=sum(times)/len(times),latent_mse_vs_fp32=mse,max_abs_latent_delta=(actual-reference).abs().max().item(),
   latent_delta_fraction_of_quantization_error=mse/greedy_mse,greedy_token_agreement=(codes==refcodes).float().mean().item())
  rows.append(row);print(json.dumps(row),flush=True)
 (b/'live/encoder-fp32-largebatch-parity.json').write_text(json.dumps(dict(images=128,greedy_mse=greedy_mse,rows=rows),indent=2)+'\n')
main()
