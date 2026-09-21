#!/usr/bin/env python3
"""Launch the verified Church continuation or the authorized fresh transformer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/church-consistent-rqvae-20260914'
OUTPUT = ROOT / 'outputs/church-ft3ep-resume-20260916'
RUN_ID = 'church-laser-ft3ep-scratch-20260916'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview-on-resume',action='store_true',
                        help='Also publish a current 10 x 10 grid immediately after restoring')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--from-scratch', action='store_true')
    mode.add_argument('--resume-fresh-run', action='store_true',
                      help='Resume the adaptive-LR run from its own saved last.pt')
    mode.add_argument('--checkpoint', type=Path, default=ROOT /
                        'outputs/church-stage2-scratch-20260916/train/last.pt')
    args = parser.parse_args()
    new_run = args.from_scratch or args.resume_fresh_run
    output = ROOT/'outputs/church-ft3ep-scratch-adaptive-20260916' if new_run else OUTPUT
    run_id = 'church-laser-ft3ep-scratch-adaptive-lr-20260916' if new_run else RUN_ID
    checkpoint = (output/'train/last.pt') if args.resume_fresh_run else args.checkpoint.resolve()
    if not args.from_scratch and not checkpoint.is_file():
        parser.error(f'The requested run checkpoint is unavailable: {checkpoint}')
    python = ROOT / '.venv-imagenet-stage2/bin/python'
    driver = ROOT / 'scripts/tools/continue_church_stage2.py'
    if new_run:
        manifest = json.loads((output/'runtime-manifest.json').read_text())
        for name, expected in manifest.items():
            with (output/'runtime'/name).open('rb') as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest()!=expected:
                    raise ValueError(f'Frozen runtime changed: {name}')
        driver = output/'runtime/scripts/tools/continue_church_stage2.py'
        preflight = json.loads((output/'preflight/status.json').read_text())
        recovery = json.loads((output/'preflight-resume/status.json').read_text())
        if preflight.get('phase')!='preflight_complete' or recovery.get('phase')!='preflight_complete':
            raise RuntimeError('Fresh training and checkpoint recovery preflight must pass')
    arguments = [str(driver), '--pipeline-dir', str(BASE),
                 '--cache', str(BASE / 'preparation/cache'),
                 '--calibration', str(BASE / 'preparation/temperature-calibration.json'),
                 '--output', str(output / 'train'), '--run-id', run_id,
                 '--batch-size', '256',
                 '--resume-lr-scale', '.5', '--fid-every', '10',
                 '--validation-every', '5', '--preview-every-steps', '200',
                 '--decode-batch-size', '8', '--keep-best']
    if args.from_scratch:
        arguments += ['--from-scratch','--source-run',f'helloimlixin-rutgers/laser/{RUN_ID}']
    else:
        arguments += ['--resume',str(checkpoint)]
    if args.preview_on_resume:
        arguments += ['--preview-on-resume']
    # Complete CPU compatibility checks before creating/updating any W&B run.
    if not args.from_scratch:
        subprocess.run([str(python), *arguments, '--validate-resume'], cwd=ROOT, check=True)
    environment = os.environ.copy()
    key_path = Path('/root/.config/laser/wandb-api-key')
    if not environment.get('WANDB_API_KEY') and key_path.is_file():
        environment['WANDB_API_KEY'] = key_path.read_text().strip()
    if not environment.get('WANDB_API_KEY'):
        parser.error('Set WANDB_API_KEY or restore the private W&B credential file')
    environment.update(CUDA_VISIBLE_DEVICES='0,1', PYTHONUNBUFFERED='1',
                       OMP_NUM_THREADS='8')
    output.mkdir(parents=True, exist_ok=True)
    command = [str(python), '-m', 'torch.distributed.run', '--standalone',
               '--nproc-per-node=2', *arguments]
    with (output / 'training.log').open('ab') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                   stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    receipt = dict(pid=process.pid, command=command, checkpoint=None if args.from_scratch else str(checkpoint),
                   run_id=run_id, from_scratch=args.from_scratch, launched_unix=time.time())
    (output / 'launch.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
