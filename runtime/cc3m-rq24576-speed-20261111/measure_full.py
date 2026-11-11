import json,sys,time
from pathlib import Path
b=Path('/tmp/laser-cc3m-rq24576-20261111');sys.path.insert(0,str(b/'source'))
import torch
from omegaconf import OmegaConf
from src.training import cc3m_joint as t
from src.training.rq_precision import native_bf16_autocast
from src.training.rq_sdpa_backend import sdpa_attention

torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(.55)
c=OmegaConf.to_container(OmegaConf.load(b/'config.yaml'),resolve=True);model=t.build_model(c,'cuda');aux=t.build_aux(c,'cuda');opt=t.optimizer_for(model,c);ds=t.dataset(c,'validation')
x=torch.stack([ds[i][0] for i in range(32)]).cuda().repeat(8,1,1,1);ids=torch.stack([ds[i][1] for i in range(32)]).cuda().repeat(8,1)
rows=[]
with native_bf16_autocast(),sdpa_attention(model,**t.attention_options(c)):
 for batch,autotune in [(8,False),(32,True),(64,True),(128,True)]:
  t.configure_execution(dict(encoder_batch=batch,cudnn_benchmark=autotune,cudnn_benchmark_limit=3))
  torch.backends.cudnn.benchmark=autotune;torch.backends.cudnn.benchmark_limit=3
  times=[]
  for iteration in range(4):
   opt.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.monotonic()
   loss,_=t.objective(model,aux,x,ids,c);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();torch.cuda.synchronize()
   elapsed=time.monotonic()-start
   if iteration:times.append(elapsed)
  row=dict(encoder_batch=batch,cudnn_benchmark=autotune,cudnn_tf32=torch.backends.cudnn.allow_tf32,seconds=sum(times)/len(times),images_per_second_single_gpu=256/(sum(times)/len(times)),peak_gib=torch.cuda.max_memory_allocated()/2**30,loss=loss.item(),passed=True)
  rows.append(row);print(json.dumps(row),flush=True)
(b/'live/full-speed-benchmark.json').write_text(json.dumps(rows,indent=2)+'\n')
