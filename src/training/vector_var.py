"""Launch the frozen multiscale vector-ID fitting, verification and VAR pipeline."""
import os
from pathlib import Path
import subprocess
import sys


def run(cfg):
    root=Path(__file__).resolve().parents[2]
    expected=dict(vocabulary=6144,tokens_per_image=680,world_size=4,global_batch=768,epochs=200)
    for name,value in expected.items():
        if int(cfg[name]) != value:
            raise ValueError(f"This frozen experiment requires {name}={value}; prepare a separate contract to change it")
    command=[sys.executable,str(root/cfg.pipeline)]
    env=dict(os.environ)
    key=Path('/root/.config/laser/wandb.key')
    if 'WANDB_API_KEY' not in env and key.exists():env['WANDB_API_KEY']=key.read_text().strip()
    env.update(WANDB_BASE_URL='https://api.forge.coreweave.com',WANDB_DISABLE_GIT='true',
               OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',
               PYTHONUNBUFFERED='1',PYTORCH_ALLOC_CONF='expandable_segments:True',NCCL_NVLS_ENABLE='0',
               TORCHINDUCTOR_COMPILE_THREADS='2',
               TORCHINDUCTOR_CACHE_DIR='/tmp/laser-matched680-vector6144-20261111/compile-cache')
    subprocess.run(command,cwd=root,env=env,check=True)
