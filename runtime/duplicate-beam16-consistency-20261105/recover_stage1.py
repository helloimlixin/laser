import os,json
from pathlib import Path
base=Path('/tmp/laser-beam16-duplicate-20261105')
os.environ['WANDB_API_KEY']=Path('/root/.config/laser/wandb.key').read_text().strip()
os.environ['WANDB_BASE_URL']='https://api.forge.coreweave.com'
import wandb
api=wandb.Api(timeout=180)
a=api.artifact('helloimlixin-rutgers/laser/imga16384k4altbn64-b128-b300-20260830000755-stage1-checkpoints:v9')
a.get_path('best_rfid_slot1_model.pt').download(root=str(base/'stage1-artifact'))
print('STAGE1_COMPLETE',flush=True)
