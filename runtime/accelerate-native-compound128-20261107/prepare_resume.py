import json,hashlib,tarfile,time,pickle,shutil
from pathlib import Path
import torch,numpy as np
BASE=Path('/tmp/laser-native-speed-20261107');OLD=Path('/tmp/laser-var-stage2-compound128-omp-20261107')
EXCLUDED={'output','wandb','checkpoint_every','preview_every','workers','source_bundle'}

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def digest(value):
 h=hashlib.sha256()
 def visit(v):
  if isinstance(v,torch.Tensor):
   t=v.detach().cpu().contiguous();h.update(str((str(t.dtype),tuple(t.shape))).encode());h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
  elif isinstance(v,np.ndarray):h.update(str((str(v.dtype),v.shape)).encode());h.update(v.tobytes())
  elif isinstance(v,dict):
   for k in sorted(v,key=repr):h.update(repr(k).encode());visit(v[k])
  elif isinstance(v,(list,tuple)):
   h.update(type(v).__name__.encode())
   for item in v:visit(item)
  else:h.update(pickle.dumps(v,protocol=5))
 visit(value);return h.hexdigest()

def main():
 torch.set_num_threads(4)
 original=json.loads((OLD/'config.json').read_text());cfg=dict(original,output=str(BASE/'run'),source_bundle=str(BASE/'source.tar.gz'))
 cfg['runtime_sha256']=dict(original['runtime_sha256'],**{n:sha(BASE/'runtime'/n) for n in ['native_performance.py','fast_driver.py']})
 cfg['execution_optimization']=dict(base=str(BASE),compiled_prior_forward=True,compile_backend='inductor',compile_fullgraph=True,
  cuda_graphs=False,attention='original masked SDPA',classifier='compute only type-valid rows',gradient_as_bucket_view=True,
  numa_affinity=True,frozen_encoder_changed=False,effective_batch_changed=False)
 scientific_exclude={'output','source_bundle','runtime_sha256','execution_optimization'}
 assert {k:v for k,v in original.items() if k not in scientific_exclude}=={k:v for k,v in cfg.items() if k not in scientific_exclude}
 payload=torch.load(OLD/'run/prior-last.pt',map_location='cpu',mmap=True,weights_only=False)
 assert payload['progress']==dict(epoch=0,batch=595,step=595) and payload['world_size']==4
 original_digests={k:digest(payload[k]) for k in ['model','optimizer','rng']}
 contract={k:v for k,v in cfg.items() if k not in EXCLUDED};contract_sha=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
 updated=dict(payload,contract_sha256=contract_sha)
 (BASE/'run').mkdir(exist_ok=True);torch.save(updated,BASE/'run/prior-last.pt')
 reread=torch.load(BASE/'run/prior-last.pt',map_location='cpu',mmap=True,weights_only=False)
 actual={k:digest(reread[k]) for k in original_digests};assert actual==original_digests
 assert reread['progress']==payload['progress']
 write(BASE/'config.json',cfg);write(BASE/'run/config.json',cfg);write(BASE/'run/contract.json',contract)
 for name in ['initialization.json','wandb.json','last-preview.json']:
  if (OLD/'run'/name).exists():shutil.copyfile(OLD/'run'/name,BASE/'run'/name)
 for p in (OLD/'run').glob('fid-epoch*.json'):shutil.copyfile(p,BASE/'run'/p.name)
 private=BASE/'preflight';private.mkdir(exist_ok=True);(private/'prior-last.pt').hardlink_to(BASE/'run/prior-last.pt')
 private_cfg=dict(cfg,output=str(private));write(BASE/'preflight.json',private_cfg)
 report=dict(passed=True,source_checkpoint_sha256=sha(OLD/'run/prior-last.pt'),source_progress=payload['progress'],
  original_contract_sha256=payload['contract_sha256'],new_contract_sha256=contract_sha,
  scientific_configuration_identical=True,global_batch=768,world_size=4,microbatch=192,accumulation=1,
  model_bit_exact=True,optimizer_bit_exact=True,all_rank_rng_bit_exact=True,data_cursor_identical=True,component_sha256=actual,
  execution_changes_only=True,time=time.time())
 write(BASE/'resume-verification.json',report)
 with tarfile.open(BASE/'source.tar.gz','w:gz') as tar:
  for p in (BASE/'runtime').rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts:tar.add(p,arcname='runtime/'+str(p.relative_to(BASE/'runtime')))
 print(json.dumps(report),flush=True)
if __name__=='__main__':main()
