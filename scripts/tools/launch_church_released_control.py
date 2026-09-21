#!/usr/bin/env python3
"""Launch or resume the verified released-tokenizer Church control."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = 'church-original-rqvae-released-tokenizer-control-20260917'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,default=ROOT/'outputs/church-released-tokenizer-control-20260917')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    root = args.run_root.resolve()
    verified = json.loads((root/'verification.json').read_text())
    assert verified['passed'] and verified['resumed_step3_matches_uninterrupted']
    manifest = root/'runtime-manifest.json'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == verified['runtime_manifest_sha256']
    for name,digest in json.loads(manifest.read_text()).items():
        assert hashlib.sha256((root/'runtime'/name).read_bytes()).hexdigest() == digest,name
    train = root/'train'
    if train.exists() and not args.resume:
        parser.error('Production output exists; use --resume to retain its state')
    if args.resume and not (train/'last.pt').is_file():
        parser.error('No full production last.pt checkpoint to resume')
    for receipt in [root/'launch.json',train/'status.json']:
        if receipt.exists():
            pid = json.loads(receipt.read_text()).get('pid')
            path = Path(f'/proc/{pid}/cmdline')
            if pid and path.exists() and str(root).encode() in path.read_bytes():
                parser.error(f'The control is already running as PID {pid}')
    environment = os.environ.copy()
    key_file = Path('/root/.config/laser/wandb-api-key')
    if not environment.get('WANDB_API_KEY') and key_file.is_file():
        environment['WANDB_API_KEY'] = key_file.read_text().strip()
    if not environment.get('WANDB_API_KEY'):
        parser.error('The existing private W&B credential is unavailable')
    environment.update(CUDA_VISIBLE_DEVICES='0,1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4')
    command = [str(ROOT/'.venv-imagenet-stage2/bin/python'),'-m','torch.distributed.run',
        '--standalone','--nproc-per-node=2',str(root/'runtime/scripts/tools/train_church_released_control.py'),
        '--run-root',str(root),'--output',str(train),'--run-id',RUN_ID,'--batch-size','128','--memory-fraction','.40']
    if args.resume:
        command += ['--resume',str(train/'last.pt')]
    with (root/'training.log').open('ab') as log:
        process = subprocess.Popen(command,cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt = dict(pid=process.pid,command=command,run_id=RUN_ID,resume=args.resume,launched_unix=time.time())
    (root/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
