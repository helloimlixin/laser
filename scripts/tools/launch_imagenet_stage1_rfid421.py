#!/usr/bin/env python3
"""Launch the corrected ten-epoch A1024/K4 run after its verification gates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=Path('/tmp/laser-a1024-rfid421-20261029'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    driver = project/'scripts/tools/train_imagenet_stage1_rfid421.py'
    smoke = json.loads((args.base/'smoke-final/verification.json').read_text())
    assert smoke['passed'] and smoke['checkpoint_reloaded_exactly']
    assert smoke['signature']['driver_sha256'] == hashlib.sha256(driver.read_bytes()).hexdigest()
    assert json.loads((args.base/'data/ready.json').read_text())['ready']
    assert json.loads((args.output/'validation-jpeg-verification.json').read_text())['passed']
    for name,digest in smoke['signature']['source_sha256'].items():
        assert hashlib.sha256((args.base/'runtime'/name).read_bytes()).hexdigest() == digest,name
    run_id = 'imagenet-stage1-a1024-k4-rfid421-fixed-10ep-20261029'
    env = dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
        LASER_VGG16_WEIGHTS=str(args.base/'weights/vgg16.pth'),
        LASER_VGG_LPIPS_DIR=str(args.base/'weights'),TORCH_HOME=str(args.base/'weights'),
        WANDB_MODE='online',WANDB_ENTITY='helloimlixin-rutgers',WANDB_PROJECT='laser',
        WANDB_RUN_ID=run_id,WANDB_NAME='ImageNet stage1 | A1024 K4 continuous | rFID4.21 pipeline fixed | scratch10',
        WANDB_RUN_GROUP='imagenet-small-dictionary-stage1-20261029',
        WANDB_RESUME_MODE='must' if args.resume else 'never',WANDB_CHECKPOINT_UPLOAD='1',
        WANDB_DIR=str(args.base),WANDB_CACHE_DIR=str(args.base/'wandb-cache'),
        WANDB_TAGS='stage1,imagenet,a1024,k4,continuous-coefficients,rfid421-recipe,alternating-dictionary,fp32,4b300')
    command = [sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=4',
        str(driver),'--runtime-root',str(args.base/'runtime'),
        '--config',str(project/'configs/stage1/imagenet-a1024-k4-rfid421-fixed-10ep.yaml'),
        '--output',str(args.output)]
    if args.resume:command.extend(['--resume',str(args.resume)])
    else:assert not (args.output/'last_model.pt').exists(),'Refusing to overwrite an existing run.'
    log_path = args.base/'train.log'
    public_log = args.output/'train.log'
    if not public_log.exists():public_log.symlink_to(log_path)
    status = dict(phase='starting',command=command,run_id=run_id,
        wandb_url=f'https://wandb.ai/helloimlixin-rutgers/laser/runs/{run_id}',
        supervisor_pid=os.getpid(),started_at=time.time(),log=str(log_path))
    def record():
        target = args.output/'launch-status.json'; temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(status,indent=2)+'\n'); temporary.replace(target)
    with log_path.open('a') as log:
        child = subprocess.Popen(command,cwd=project,env=env,stdout=log,stderr=subprocess.STDOUT)
        status.update(phase='running',torchrun_pid=child.pid); record()
        result = child.wait()
    status.update(phase='completed' if result == 0 else 'failed',returncode=result,finished_at=time.time())
    record()
    return result


if __name__ == '__main__':sys.exit(main())
