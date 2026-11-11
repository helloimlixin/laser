import os,json,concurrent.futures
from pathlib import Path
os.environ['WANDB_API_KEY']=Path('/root/.config/laser/wandb.key').read_text().strip()
os.environ['WANDB_BASE_URL']='https://api.forge.coreweave.com'
import wandb
api=wandb.Api(timeout=180)
r=api.run('helloimlixin-rutgers/laser/imnet-j16384-best14574-continue100-1b300-20261104')
arts=list(r.logged_artifacts())
for a in arts:
 print('ARTIFACT',a.name,a.type,a.state,a.size,flush=True)
 print('CONTENTS',json.dumps([dict(name=f.name,size=f.size) for f in a.files()]),flush=True)
a=api.artifact('helloimlixin-rutgers/laser/imnet-j16384-best14574-continue100-1b300-20261104-source:v0')
a.download(root='/tmp/laser-resume-h100-20261105/source-artifact')
a=api.artifact('helloimlixin-rutgers/laser/imnet-j16384-best14574-continue100-1b300-20261104-checkpoint:v2')
a.download(root='/tmp/laser-resume-h100-20261105/checkpoint-artifact')
print('DOWNLOADS_COMPLETE',flush=True)
