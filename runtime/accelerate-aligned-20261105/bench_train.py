"""Run the production driver against a private output and disposable weights."""
import math
import os
from pathlib import Path
import sys

BASE=Path('/tmp/laser-aligned-beam451-20261105/frozen')
sys.path[:0]=[str(BASE/'source'),str(BASE)]
os.environ['RQ_UPSTREAM_ROOT']=str(BASE/'upstream')
os.environ['WANDB_DISABLE_GIT']='true'
import torch
torch.backends.cuda.matmul.allow_tf32=False
torch.set_float32_matmul_precision('highest')
torch.backends.cudnn.allow_tf32=True
from src import rq_upstream
from runtime.training_acceleration import install
install()
from runtime.media_policy import install as media_policy
media_policy()
if os.environ.get('LASER_PROFILE_TRAINING')=='1':
    import importlib.util
    p=Path(__file__).with_name('profile_hooks.py')
    spec=importlib.util.spec_from_file_location('speed_profile_hooks',p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    mb=int(sys.argv[sys.argv.index('--microbatch')+1])
    module.install(Path(os.environ['LASER_PRIVATE_SMOKE_DIR'])/f"timings-rank{os.environ['RANK']}.json",
                   math.ceil(2048/(4*mb)))
from src.training.rq_imagenet_shared import main

if __name__=='__main__':main('laser_aligned_beam',BASE)
