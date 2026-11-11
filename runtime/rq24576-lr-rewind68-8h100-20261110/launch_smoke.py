"""Validate the preserved parent and start a two-update eight-GPU resume test."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
plan=json.loads((HERE/'plan.json').read_text());base=Path(plan['base']);out=Path(plan['output'])
source=Path(plan['source_checkpoint'])
with source.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
assert actual==plan['source_checkpoint_sha256']
receipt=dict(passed=True,path=str(source),bytes=source.stat().st_size,sha256=actual,source_epoch=62,source_step=38812,time=time.time())
(out/'source-checkpoint-verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
spec=importlib.util.spec_from_file_location('parent_environment',Path('/workspace/Projects/laser/runtime/rq24576-native-8h100-20261108/environment.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
env=module.environment()
env.update(LASER_STAGE2_OUTPUT=str(out),LASER_BEAM_BASE=str(base),LASER_PRIVATE_SMOKE_DIR=str(base/'smoke'),
 WANDB_CACHE_DIR=str(base/'wandb-cache'))
command=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc-per-node=8',str(base/'train.py'),
 '--epochs','100','--microbatch','64','--resume',str(source),'--smoke-updates',str(plan['source_step']+2)]
with (out/'resume-smoke.log').open('ab',buffering=0) as stream:
 child=subprocess.Popen(command,env=env,cwd=base,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
state=dict(pid=child.pid,command=command,phase='resume_smoke',source_epoch=62,change_lr_at_epoch=68,world_size=8,time=time.time())
(out/'smoke-launch.json').write_text(json.dumps(state,indent=2)+'\n')
print(json.dumps(state),flush=True)
