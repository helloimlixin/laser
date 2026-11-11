"""Launch the reviewed CC3M recipe in a detached process with local logs."""
import json,os,subprocess
from pathlib import Path
from omegaconf import OmegaConf
BASE=Path('/tmp/laser-cc3m-rq24576-20261111')
ROOT=Path('/workspace/Projects/laser')
PYTHON='/tmp/laser-cc3m-venv/bin/python'

def environment():
 env=dict(os.environ)
 for key in ['WANDB_SERVICE','_WANDB_SERVICE']:env.pop(key,None)
 env.update(WANDB_API_KEY=Path('/root/.config/laser/wandb.key').read_text().strip(),
  WANDB_BASE_URL='https://api.wandb.ai',WANDB_MODE='online',WANDB_CONSOLE='off',WANDB_DISABLE_GIT='true',
  CUDA_VISIBLE_DEVICES='0,1',TORCH_HOME=str(BASE/'torch-cache'),
  OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',
  NCCL_DEBUG='WARN',NCCL_NVLS_ENABLE='0',NCCL_IB_DISABLE='1',NCCL_SOCKET_IFNAME='eth0',PYTORCH_ALLOC_CONF='expandable_segments:True',
  LASER_CC3M_CHECKPOINT_MIRROR='0')
 return env

def command(smoke=False):
 cmd=[PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=2']
 if smoke:cmd+=['-m','src.training.cc3m_joint','--config',str(BASE/'config.yaml'),'--smoke']
 else:cmd+=['train.py','--config',str(BASE/'config.yaml')]
 return cmd

def main():
 cfg=OmegaConf.load(BASE/'config.yaml');run=cfg.run_id
 receipt=json.loads((BASE/'live/preflight/passed.json').read_text())
 calibration=json.loads((BASE/'live/reproducibility/noise-calibration.json').read_text())
 assert receipt['passed'] and calibration['passed']
 assert cfg.temperatures==calibration['selected_temperature']
 assert cfg.world_size*cfg.microbatch*cfg.accumulation==2048
 assert not (BASE/'checkpoints/last.pt').exists(),'Fresh launch requires a new output directory'
 (BASE/'wandb').mkdir(exist_ok=True)
 log=(BASE/'logs/training.log').open('ab')
 process=subprocess.Popen(command(),cwd=BASE/'source',env=environment(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 status=dict(pid=process.pid,run_id=run,command=command(),base=str(BASE),output=cfg.output,
  wandb_url=f'https://wandb.ai/{cfg.wandb_entity}/laser/runs/{run}',
  world_size=2,microbatch=cfg.microbatch,accumulation=cfg.accumulation,global_batch=2048,
  epochs=cfg.epochs,teacher_temperature=cfg.temperatures,status='launched')
 (BASE/'launch.json').write_text(json.dumps(status,indent=2)+'\n')
 out=Path(cfg.output);out.mkdir(exist_ok=True,parents=True)
 (out/'launch.json').write_text(json.dumps(status,indent=2)+'\n')
 print(json.dumps(status),flush=True)
if __name__=='__main__':main()
