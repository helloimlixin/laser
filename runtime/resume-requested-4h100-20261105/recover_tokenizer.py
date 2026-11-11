import os,json
from pathlib import Path
os.environ['WANDB_API_KEY']=Path('/root/.config/laser/wandb.key').read_text().strip()
os.environ['WANDB_BASE_URL']='https://api.forge.coreweave.com'
import wandb
api=wandb.Api(timeout=180)
a=api.artifact('helloimlixin-rutgers/laser/imagenet-atoms4096-joint16384-native-teacher-beam-reconstruction-tokenizer:v0')
print(a.name,a.state,a.size,flush=True)
for f in a.files(): print(f.name,f.size,flush=True)
a.download(root='/tmp/laser-resume-h100-20261105/tokenizer-artifact')
