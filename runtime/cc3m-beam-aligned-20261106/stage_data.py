"""Stage the pinned CC3M shards; begin with validation and epoch-zero training groups."""
import sys,json,time,concurrent.futures
from pathlib import Path
import numpy as np
ROOT=Path('/workspace/Projects/laser');sys.path.insert(0,str(ROOT))
from src.data.cc3m_joint import shard_index,write_json
BASE=Path('/tmp/laser-cc3m-aligned-20261106')
manifest=json.loads((BASE/'data-manifest.json').read_text())
order=np.random.default_rng(np.random.SeedSequence([261101,0])).permutation(len(manifest['train']))
specs=manifest['validation']+[manifest['train'][int(i)] for i in order]
def stage(spec):
 shard_index(spec,'/workspace/Projects/data/cc3m/shards',BASE/'data');return spec['name']
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for n,name in enumerate(pool.map(stage,specs),1):
  row=dict(completed=n,total=len(specs),last=name,time=time.time());write_json(BASE/'staging.json',row)
  print(json.dumps(row),flush=True)
  if n==32:write_json(BASE/'initial-shards-ready.json',dict(ready=True,validation_shards=16,initial_training_shards=16,time=time.time()))
write_json(BASE/'all-shards-ready.json',dict(ready=True,shards=len(specs),time=time.time()))
