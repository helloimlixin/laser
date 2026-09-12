#!/usr/bin/env python3
"""Launch the prepared matched pair and its final reporter as detached jobs."""
import json
import argparse
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('outputs/mdctcodec_matched_6kbps'))
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[2]
    os.chdir(repo)
    root=args.root
    for name in ['prepared.json','preflight_verified.json','restore_verified.json','source.tar.gz']:
        if not (root/name).is_file():raise RuntimeError('Missing verified preparation: '+name)
    if (root/'launch.json').exists():raise RuntimeError('Launch already recorded; inspect or explicitly resume existing arms')
    if any((root/arm/'run.json').exists() for arm in ['laser','rvq']):
        raise RuntimeError('An arm already exists; refusing a duplicate launch')
    usage=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    if len(usage.splitlines())<2 or any(int(line.strip())>100 for line in usage.splitlines()[:2]):
        raise RuntimeError('Both GPUs must be free before this two-arm launch')
    jobs={}
    env_base={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','WANDB_MODE':'online',
              'VISQOL_BINARY':str(repo/'outputs/visqol/bin/visqol'),'CUBLAS_WORKSPACE_CONFIG':':4096:8'}
    for gpu,arm in enumerate(['laser','rvq']):
        output=root/arm;output.mkdir(exist_ok=True)
        cmd=[sys.executable,'-u','scripts/train_mdctcodec_matched.py','--root',str(root),'--arm',arm]
        with (output/'train.log').open('ab') as log:
            proc=subprocess.Popen(cmd,env={**env_base,'CUDA_VISIBLE_DEVICES':str(gpu)},
                stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        jobs[arm]={'pid':proc.pid,'gpu':gpu,'command':cmd,'log':str(output/'train.log')}
        (root/'launch.json').write_text(json.dumps({'status':'launching','jobs':jobs},indent=2))
    cmd=[sys.executable,'-u','scripts/report_mdctcodec_matched.py','--root',str(root),'--pids',
         str(jobs['laser']['pid']),str(jobs['rvq']['pid'])]
    with (root/'reporter.log').open('ab') as log:
        proc=subprocess.Popen(cmd,env={**env_base,'CUDA_VISIBLE_DEVICES':'0'},
            stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    record={'status':'launched','started_utc':datetime.now(timezone.utc).isoformat(),
            'jobs':jobs,'reporter':{'pid':proc.pid,'command':cmd,'log':str(root/'reporter.log')}}
    (root/'launch.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
