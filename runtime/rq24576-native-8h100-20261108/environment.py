"""Paths and execution settings for the 24,576-class native RQ experiment."""
import os
from pathlib import Path

ROOT = Path('/workspace/Projects/laser')
SOURCE = Path('/tmp/laser-beam451-native25-8h100-20261108')
BASE = Path('/tmp/laser-rq24576-native-8h100-20261108')
RUN = 'imagenet-rq24576-native-scratch100-8h100-20261108'
OUT = ROOT / 'outputs' / RUN
QUALITY = ROOT / 'outputs/imagenet-rq24576-tokenizer-20261108'
HERE = Path(__file__).resolve().parent
PYTHON = '/opt/laser-venv/bin/python'
VOCABULARY = 24576


def environment():
    env = dict(os.environ)
    for key in ['WANDB_SERVICE', '_WANDB_SERVICE']:
        env.pop(key, None)
    env.update(
        WANDB_API_KEY=Path('/root/.config/laser/wandb.key').read_text().strip(),
        WANDB_BASE_URL='https://api.wandb.ai', WANDB_MODE='online', WANDB_CONSOLE='off',
        WANDB_DISABLE_GIT='true', CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
        LASER_STAGE2_OUTPUT=str(OUT), LASER_BEAM_BASE=str(BASE),
        TORCH_HOME=str(BASE / 'torch-cache'), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1', NCCL_DEBUG='WARN',
        NCCL_NVLS_ENABLE='0', LASER_FUSED_ADAMW='1', LASER_TRAIN_SDPA='1',
        LASER_SDPA_MIN_SEQ='16', LASER_COMPILE_NATIVE_LOSS='1',
        LASER_COMPILE_LOSS_DYNAMIC='1', LASER_EXACT_BEAM_KERNELS='0',
        LASER_FAST_BEAM_TOPK='0', TORCHINDUCTOR_COMPILE_THREADS='2',
        TORCHINDUCTOR_CACHE_DIR=str(BASE / 'compile-cache'),
        PYTORCH_ALLOC_CONF='expandable_segments:True', WANDB_CACHE_DIR=str(BASE / 'wandb-cache'),
    )
    return env
