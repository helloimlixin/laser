#!/usr/bin/env python3
"""Launch the verified matched ordering pilot, preserving prior training state."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
ALLOWED = {'train_church_calibrated.py', 'train_church_support_pattern.py', 'train_church_pattern_order.py'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def command_for(pid):
    path = Path(f'/proc/{pid}/cmdline')
    return [v.decode() for v in path.read_bytes().split(b'\0') if v] if path.exists() else []


def authorized_process(pid):
    command = command_for(pid)
    if not any(Path(value).name in ALLOWED for value in command):
        raise RuntimeError(f'PID {pid} is not a recognized Church trainer')
    if Path(f'/proc/{pid}/cwd').resolve() != ROOT:
        raise RuntimeError(f'PID {pid} is outside the project workspace')
    return command


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'outputs/church-pattern-order-20260911')
    p.add_argument('--credential-pid', type=int)
    p.add_argument('--pause-pids', type=int, nargs='*', default=[])
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    output = args.output.resolve()
    verified = json.loads((output/'verification.json').read_text())
    if not verified['production_ready']:
        raise RuntimeError('GPU, resume, and codec verification must pass before launch')
    for name, value in verified['source_sha256'].items():
        if digest(ROOT/name) != value:
            raise RuntimeError(f'Source changed since verification: {name}')
    calibration = json.loads((ROOT/'outputs/church-support-pattern-integer-20260911/results.json').read_text())
    if digest(Path(calibration['selected_codebook'])) != verified['codebook_sha256']:
        raise RuntimeError('Pattern codebook changed')
    launch_file = output/'launch.json'
    if launch_file.exists():
        previous = json.loads(launch_file.read_text())
        for arm in previous['arms'].values():
            if any(Path(v).name=='train_church_pattern_order.py' for v in command_for(arm['pid'])):
                raise RuntimeError('A matched-order training process is already running')
    for order in ('pattern-first','support-first'):
        training = output/order
        if training.exists() and not args.resume:
            raise FileExistsError(training)
        if args.resume and not (training/'last.pt').exists():
            raise FileNotFoundError(training/'last.pt')
    credential = os.environ.get('WANDB_API_KEY')
    if not credential and args.credential_pid:
        authorized_process(args.credential_pid)
        for value in Path(f'/proc/{args.credential_pid}/environ').read_bytes().split(b'\0'):
            if value.startswith(b'WANDB_API_KEY='):
                credential = value.split(b'=',1)[1].decode()
                break
    if not credential:
        raise RuntimeError('A private W&B credential is required')
    pauses = []
    for pid in args.pause_pids:
        command = authorized_process(pid)
        training = Path(command[command.index('--output')+1])
        if not training.is_absolute():
            training = ROOT/training
        status = json.loads((training/'status.json').read_text())
        if status['pid'] != pid:
            raise RuntimeError('Training status does not match the requested process')
        pauses.append({'pid':pid,'output':str(training),'status_before':status,'command':command})
    if pauses:
        # Authorization and credentials are checked before either signal.
        for pause in pauses:
            os.kill(pause['pid'],signal.SIGTERM)
        deadline = time.monotonic()+300
        while any(command_for(row['pid']) for row in pauses):
            if time.monotonic()>deadline:
                raise RuntimeError('Previous trainers have not completed graceful pause')
            print(json.dumps({'phase':'waiting_for_saved_pause','pids':[row['pid'] for row in pauses]}),flush=True)
            time.sleep(10)
        import torch
        for pause in pauses:
            training = Path(pause['output'])
            status = json.loads((training/'status.json').read_text())
            if status['phase'] != 'paused':
                raise RuntimeError('Previous trainer did not acknowledge a saved pause')
            saved = torch.load(training/'last.pt',map_location='cpu',weights_only=False)
            assert saved['step'] == status['optimizer_step']
            assert all(key in saved for key in ('state_dict','optimizer','stream'))
            pause.update(status_after=status,checkpoint_bytes=(training/'last.pt').stat().st_size,
                         saved_step=saved['step'],optimizer_and_stream_verified=True)
            del saved
        (output/'previous-runs-pause.json').write_text(json.dumps(pauses,indent=2)+'\n')
    for name, value in verified['source_sha256'].items():
        target = output/'source-snapshot'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        if args.resume:
            if not target.exists() or digest(target)!=value:
                raise RuntimeError(f'Launch snapshot changed: {name}')
        else:
            shutil.copy2(ROOT/name,target)
    metadata = {'started_unix':time.time(),'maximum_epochs_per_arm':12,'arms':{}}
    for gpu,order in enumerate(('pattern-first','support-first')):
        run_id = f'church-order-{order}-199m-20260911'
        command = [sys.executable,'-u',str(ROOT/'scripts/train_church_pattern_order.py'),
                   '--ordering',order,'--output',str(output/order),'--wandb-id',run_id]
        if args.resume:
            command.append('--resume')
        environment = {**os.environ,'WANDB_API_KEY':credential,'CUDA_VISIBLE_DEVICES':str(gpu),
            'OMP_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'8','MKL_NUM_THREADS':'8',
            'TORCH_HOME':'/workspace/tmp/official-rqvae-eval-cache',
            'LASER_VGG16_WEIGHTS':'/workspace/tmp/laser-vgg/vgg16-397923af.pth'}
        with (output/f'{order}.log').open('ab') as f:
            child = subprocess.Popen(command,cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,
                stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        metadata['arms'][order] = {'pid':child.pid,'gpu':gpu,'command':command,
            'wandb_url':f'https://wandb.ai/helloimlixin-rutgers/laser/runs/{run_id}'}
        launch_file.write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata),flush=True)


if __name__=='__main__':
    main()
