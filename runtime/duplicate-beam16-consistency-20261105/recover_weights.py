import os,json,concurrent.futures,zipfile
from pathlib import Path
base=Path('/tmp/laser-beam16-duplicate-20261105')
os.environ['WANDB_API_KEY']=Path('/root/.config/laser/wandb.key').read_text().strip()
os.environ['WANDB_BASE_URL']='https://api.forge.coreweave.com'
import wandb
api=wandb.Api(timeout=180)
artifacts=[('imnet-j16384-a4096-beam16-rfid451-scratch100-2b300-20261104-checkpoint:v52','checkpoint-artifact'),('imagenet-atoms4096-joint16384-native-teacher-beam-reconstruction-tokenizer:v0','tokenizer-artifact')]
def recover(row):
 name,folder=row
 a=api.artifact('helloimlixin-rutgers/laser/'+name)
 print('FILES',name,[(f.name,f.size) for f in a.files()],flush=True)
 a.download(root=str(base/folder))
 print('DONE',name,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(recover,artifacts))
