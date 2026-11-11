"""Prefetch verified CC3M shards in initial epoch order; training can also stage on demand."""
from concurrent.futures import ThreadPoolExecutor
import json,sys
from pathlib import Path
base=Path('/tmp/laser-cc3m-rq24576-20261111');sys.path.insert(0,str(base/'source'))
import numpy as np
from omegaconf import OmegaConf
from src.data.cc3m_joint import stage_shard,shard_index
cfg=OmegaConf.load(base/'config.yaml');manifest=json.loads(Path(cfg.data_manifest).read_text())
shards=manifest['train'];order=np.random.default_rng(np.random.SeedSequence([cfg.seed,0])).permutation(len(shards))
ordered=manifest['validation']+[shards[i] for i in order]
def stage(spec):
 shard_index(spec,cfg.data_source,cfg.data_local)
 print(json.dumps(dict(verified=spec['name'],items=spec['items'])),flush=True)
with ThreadPoolExecutor(4) as pool:list(pool.map(stage,ordered))
