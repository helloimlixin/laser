"""Finish pinned best/latest uploads on CPU after an optimizer-boundary stop."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--key-file', type=Path, required=True)
    p.add_argument('--directory', type=Path)
    p.add_argument('--stop-reason', default='user-requested checkpoint fork')
    p.add_argument('--completed', action='store_true')
    args = p.parse_args()
    os.environ['WANDB_API_KEY'] = args.key_file.read_text().strip()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path[:0] = [str(args.base / 'source/runtime'), str(args.base / 'support')]
    import torch
    import wandb
    from src.training.full_resume_upload import recovery_metadata
    from verified_wandb_checkpoint_upload import VerifiedCloudUpload
    directory = args.directory or args.base / 'final-cpu-upload'
    paths = []
    epoch = 0
    for name in ('best-fid-resume.pt', 'best-is-resume.pt', 'last.pt'):
        file = directory / name
        payload = torch.load(file, map_location='cpu', mmap=True, weights_only=False)
        metadata = recovery_metadata(payload)
        epoch = max(epoch, metadata['epoch'])
        receipt = file.with_suffix('.json')
        receipt.write_text(json.dumps(metadata, indent=2, default=str) + '\n')
        paths.extend((file, receipt))
        del payload
    run = wandb.Api(timeout=120).run(args.run)
    run.summary.update({'execution/state':'completed' if args.completed else 'stopped_resumable',
                        'execution/stop_reason':args.stop_reason})
    VerifiedCloudUpload(args.run, args.evidence / 'final-cloud-checkpoint-receipt.json')(paths, epoch)


if __name__ == '__main__':
    main()
