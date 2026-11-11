import os,json
from pathlib import Path
os.environ['WANDB_API_KEY']=Path('/root/.config/laser/wandb.key').read_text().strip()
os.environ['WANDB_BASE_URL']='https://api.forge.coreweave.com'
import wandb
api=wandb.Api(timeout=120)
r=api.run('helloimlixin-rutgers/laser/imnet-j16384-best14574-continue100-1b300-20261104')
result=dict(run_id=r.id,state=r.state,name=r.name,world_size=r.config.get('world_size'),microbatch=r.config.get('microbatch'),accumulation=r.config.get('accumulation_steps'),lr_amplitude_factor=r.config.get('lr_amplitude_factor'),continuation=r.config.get('continuation'))
a=api.artifact('helloimlixin-rutgers/laser/imnet-j16384-best14574-continue100-1b300-20261104-checkpoint:latest')
result.update(latest_checkpoint_artifact=a.qualified_name,checkpoint_state=a.state,checkpoint_metadata=a.metadata)
b=api.artifact('helloimlixin-rutgers/laser/imnet-j16384-a4096-fid1481-resume100-4h100-20261104-checkpoint:v26')
result.update(original_best_checkpoint_state=b.state)
print(json.dumps(result,indent=2),flush=True)
Path('/tmp/laser-resume-h100-20261105/live-verification.json').write_text(json.dumps(result,indent=2)+'\n')
