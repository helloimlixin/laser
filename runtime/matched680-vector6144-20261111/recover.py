import os,json
from pathlib import Path
import wandb
os.environ['WANDB_API_KEY']=Path('/root/.config/laser/wandb.key').read_text().strip()
os.environ['WANDB_BASE_URL']='https://api.forge.coreweave.com'
base=Path('/tmp/laser-matched680-vector6144-20261111');base.mkdir(exist_ok=True)
api=wandb.Api(timeout=120)
for rid in ['imnet-s1-laser-matched680-full-continue-best-4gpu-20261105','imnet-s2-laser-beam16-joint4096-var-d16-4gpu-20261106']:
 run=api.run('helloimlixin-rutgers/laser/'+rid)
 print('RUN',rid,run.state,flush=True)
 arts=list(run.logged_artifacts());print('ARTIFACTS',[(a.name,a.type,a.size) for a in arts if 'runtime' in a.name],flush=True)
 (base/(rid+'-artifacts.json')).write_text(json.dumps([{'name':a.name,'type':a.type,'size':a.size} for a in arts],indent=2))
 if 'stage1' not in rid and 's1-' not in rid:continue
 for name,dest in [('imnet-s1-laser-matched680-full-continue-best-4gpu-20261105-full-stage1-runtime:v0','source'),('imnet-s1-laser-matched680-full-continue-best-4gpu-20261105-tokenizer:v62','checkpoint')]:
  a=api.artifact('helloimlixin-rutgers/laser/'+name);print('DOWNLOAD',name,a.size,flush=True)
  a.download(root=str(base/dest));print('DONE',dest,flush=True)
