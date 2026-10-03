"""Validated epoch-boundary layouts with exact, weighted global batches."""
import hashlib,json,math,os
from pathlib import Path
import torch
LAYOUTS={1:{4,8},2:{2,4},3:{2},4:{2},5:{1,2},6:{2}}
def optimizer_boundary_migration(saved, world, accumulation, batch_idx):
 return (os.environ.get('LASER_ALLOW_OPTIMIZER_BOUNDARY_WORLD_SIZE_CHANGE')=='1'
         and saved.get('exact_global_batch') is True
         and saved.get('world_size')==5 and world==6
         and saved.get('accumulation_steps')==accumulation==2
         and saved.get('total_batch_size')==2048
         and saved.get('batch_size')==205
         and isinstance(batch_idx,int) and batch_idx>=0 and batch_idx%2==0)
def validate_resume_layout(saved,world,accumulation,total_batch,batch_idx):
 same=(saved.get('exact_global_batch') and saved.get('world_size')==world and saved.get('accumulation_steps')==accumulation and saved.get('total_batch_size')==total_batch)
 if same:return False
 old=saved.get('world_size');oldacc=saved.get('accumulation_steps')
 valid=(total_batch==2048 and optimizer_boundary_migration(saved,world,accumulation,batch_idx)) or (os.environ.get('LASER_ALLOW_EPOCH_BOUNDARY_WORLD_SIZE_CHANGE')=='1' and saved.get('exact_global_batch') and old in LAYOUTS and world in LAYOUTS and oldacc in LAYOUTS[old] and accumulation in LAYOUTS[world] and saved.get('total_batch_size')==total_batch==2048 and saved.get('batch_size')==math.ceil(2048/(old*oldacc)) and batch_idx==0)
 if not valid:raise ValueError('Unsupported exact-global-batch epoch-boundary layout change')
 return True

def restore_resized_rng(payload,device,rank,world):
 states=payload['rng_state_by_rank']
 saved=payload.get('config',{})
 if not ((len(states)==saved.get('world_size') and optimizer_boundary_migration(saved,world,2,payload.get('batch_idx',0))) or (os.environ.get('LASER_ALLOW_EPOCH_BOUNDARY_WORLD_SIZE_CHANGE')=='1' and len(states) in LAYOUTS and world in LAYOUTS and payload.get('batch_idx',0)==0)):raise ValueError('Unsupported RNG layout migration')
 seed=None
 if rank<len(states):
  local=states[rank];policy='retained saved CPU and CUDA stream'
 else:
  h=hashlib.sha256(f"laser-world-change:{len(states)}:{world}:{payload['global_step']}:{rank}".encode())
  for state in states:
   h.update(state['torch_cpu'].numpy().tobytes())
   if 'torch_cuda' in state:h.update(state['torch_cuda'].numpy().tobytes())
  seed=int.from_bytes(h.digest()[:8],'big')%(2**63-1)
  local={'torch_cpu':torch.Generator().manual_seed(seed).get_state()}
  if device.type=='cuda':local['torch_cuda']=torch.Generator(device=device).manual_seed(seed).get_state()
  policy='independent stream derived from checkpoint, step, and new rank'
 torch.set_rng_state(local['torch_cpu'])
 if device.type=='cuda':torch.cuda.set_rng_state(local['torch_cuda'],device=device)
 directory=os.environ.get('LASER_LAYOUT_MIGRATION_DIR')
 if directory:
  p=Path(directory);p.mkdir(parents=True,exist_ok=True);(p/f'rng-rank{rank}.json').write_text(json.dumps(dict(passed=True,rank=rank,old_world=len(states),new_world=world,step=payload['global_step'],seed=seed,policy=policy,trajectory_bitwise_identical=False),indent=2)+'\n')
 return 'resized'
