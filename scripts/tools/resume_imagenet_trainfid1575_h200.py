#!/usr/bin/env python3
"""Launch the verified train-reference FID15.75 checkpoint through epoch100."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/workspace/Projects/laser')
BASE=Path('/tmp/laser-imagenet-trainfid1575-to100-20261019')
OUT=ROOT/'outputs/imagenet-trainfid1575-epoch98-to100-4h200-20261019'


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def environment(microbatch, benchmark):
    return dict(os.environ,
        WANDB_API_KEY=Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip(),
        LASER_RUN_BASE=str(BASE),LASER_PERSISTENT_BASE=str(OUT),LASER_PHASE='production',
        LASER_EXPECTED_WORLD_SIZE='4',LASER_ACCUMULATION=str(512//microbatch),
        LASER_FFHQ_DECODER_OBJECTIVE='1',LASER_SUB_BIN_TARGETS='0',
        LASER_COEFFICIENT_SIGMA_BINS=str(.125/(6/2047)),
        LASER_THROUGHPUT_BENCHMARK='1' if benchmark else '0',
        LASER_CHECKPOINT_STAGING_DIR=str(BASE/'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(BASE/'checkpoint-upload-cache'),
        LASER_CHECKPOINT_IMMUTABLE_FILES='1',
        LASER_REFERENCE_SHA256='3f9c92d15755e76ec312964a819e3e19b9cf3aadc618a7ae99f5e8aa96501260',
        LASER_LAYOUT_MIGRATION_DIR=str(BASE/'verification'/('benchmark'+str(microbatch) if benchmark else 'production')/'layout'),
        LASER_LOSS_BALANCE_VERIFY_DIR=str(BASE/'verification'/('benchmark'+str(microbatch) if benchmark else 'production')),
        LASER_DATA_WORKERS='12',LASER_COMPILE_BLOCKS='1',
        CUDA_VISIBLE_DEVICES='0,1,2,3',
        PYTHONPATH=str(BASE)+':'+str(BASE/'source')+':'+str(BASE/'source/runtime'),
        TORCH_HOME=str(BASE/'torch-cache'),TORCHINDUCTOR_CACHE_DIR=str(BASE/'inductor-cache'),
        TORCHINDUCTOR_COMPILE_THREADS='4',
        WANDB_DIR=str(BASE/'wandb'),WANDB_CACHE_DIR=str(BASE/'wandb-cache'),WANDB_DATA_DIR=str(BASE/'wandb-data'),
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',NCCL_NVLS_ENABLE='0',PYTHONUNBUFFERED='1')


def run(benchmark, microbatch):
    import yaml
    mode='benchmark'+str(microbatch) if benchmark else 'production'
    lock=(BASE/(mode+'.lock')).open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config=BASE/(mode+'.yaml' if benchmark else 'train.yaml')
    if not benchmark:
        ready=json.loads((BASE/'data/ready.json').read_text())
        assert ready['ready'] and ready['training_images']==1281167
        assert json.loads((OUT/'checkpoint-verification.json').read_text())['passed']
    for attempt in range(1 if benchmark else 4):
        if not benchmark and attempt:
            rows=[json.loads(p.read_text()) for p in (OUT/'train/checkpoints/.checkpoint-data').glob('*.json')]
            rows=[r for r in rows if Path(r['local']).is_file()]
            assert rows, 'No committed recovery checkpoint available for retry'
            latest=max(rows,key=lambda r:(r['global_step'],r['time']))
            c=yaml.safe_load(config.read_text())
            c['options']['resume_checkpoint']=latest['local']
            config.write_text(yaml.safe_dump(c,sort_keys=False))
        command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=4',
            str(BASE/'entry.py'),'--config',str(config)]
        log=BASE/(mode+'.log')
        with log.open('a') as stream:
            child=subprocess.Popen(command,cwd=BASE/'source',env=environment(microbatch,benchmark),
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            write(OUT/(mode+'-status.json'),dict(state='running',pid=os.getpid(),child_pid=child.pid,
                attempt=attempt+1,microbatch=microbatch,accumulation=512//microbatch,
                target_epoch=100,target_step=62600,log=str(log),time=time.time()))
            code=child.wait()
        if code==0:
            if not benchmark:
                final=json.loads((OUT/'training-exit.json').read_text())
                assert final['completed'] and final['global_step']==62600,final
                write(OUT/'completed.json',dict(passed=True,epoch=100,global_step=62600,time=time.time()))
            write(OUT/(mode+'-status.json'),dict(state='completed',returncode=0,time=time.time()))
            return
        write(OUT/(mode+'-status.json'),dict(state='failed',returncode=code,attempt=attempt+1,time=time.time()))
    raise SystemExit(code)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['benchmark','production'])
    parser.add_argument('--microbatch',type=int,choices=[128,256],default=256)
    args=parser.parse_args()
    run(args.mode=='benchmark',args.microbatch)
