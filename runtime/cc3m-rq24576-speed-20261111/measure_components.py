import json,sys,time
from pathlib import Path
base=Path('/tmp/laser-cc3m-rq24576-20261111');sys.path.insert(0,str(base/'source'))
import torch
from omegaconf import OmegaConf
from src.training import cc3m_joint as t
from src.training.rq_precision import native_bf16_autocast
from src.training.rq_sdpa_backend import sdpa_attention
from src.scaled_atom_training import soft_cross_entropy

torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(.24)
c=OmegaConf.to_container(OmegaConf.load(base/'config.yaml'),resolve=True)
model=t.build_model(c,'cuda');aux=t.build_aux(c,'cuda');ds=t.dataset(c,'validation')
x=torch.stack([ds[i][0] for i in range(8)]).cuda().repeat(8,1,1,1);ids=torch.stack([ds[i][1] for i in range(8)]).cuda().repeat(8,1)
results=[]
with native_bf16_autocast(),sdpa_attention(model,**t.attention_options(c)):
 for iteration in range(2):
  times={};events=[]
  def mark(name):
   e=torch.cuda.Event(enable_timing=True);e.record();events.append((name,e))
  model.zero_grad(set_to_none=True);mark('start')
  with torch.no_grad(),torch.autocast('cuda',enabled=False):
   latent=torch.cat([aux.encode(part.float()) for part in x.split(c['encoder_batch'])]);mark('encoder')
   targets,codes=t.image_targets(aux,latent,c);mark('teacher')
  with torch.autocast('cuda',dtype=torch.bfloat16):
   logits,text_logits=model(codes,model_aux=aux,cond=ids,amp=True);mark('transformer')
   image_loss=soft_cross_entropy(logits,targets,chunk_size=1024)
   text_loss=torch.nn.functional.cross_entropy(text_logits.flatten(0,1).float(),ids[:,1:].reshape(-1))
   loss=.9*image_loss+.1*text_loss;mark('loss')
  loss.backward();mark('backward');torch.cuda.synchronize()
  for i in range(1,len(events)):times[events[i][0]]=events[i-1][1].elapsed_time(events[i][1])
  results.append(times);print(json.dumps(dict(iteration=iteration,milliseconds=times)),flush=True)
  del targets,logits,text_logits,loss,image_loss,text_loss
 # One representative trace identifies high-cost kernels rather than guessing.
 model.zero_grad(set_to_none=True)
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
  loss,_=t.objective(model,aux,x,ids,c);loss.backward();torch.cuda.synchronize()
 print(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=35),flush=True)
(base/'live/speed-profile.json').write_text(json.dumps(results,indent=2)+'\n')
