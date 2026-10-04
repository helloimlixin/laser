"""Download and verify a full ImageNet recovery checkpoint and its dependencies.

Example (credentials supplied through WANDB_API_KEY or --key-file):
  python scripts/tools/download_wandb_full_resume.py --run ENTITY/PROJECT/RUN \
      --slot latest --destination /workspace/recovery

This prepares recovery files without launching or changing a training job.
The downloaded checkpoint retains its own optimizer, scheduler, cursor and RNG.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.training.full_resume_upload import atomic_json, recovery_metadata


def download_verified(run, name, destination):
    """Do not replace a usable local recovery file with an incomplete download."""
    remote = run.file(name)
    target = destination / name
    if target.is_file() and target.stat().st_size == remote.size:
        with target.open('rb') as stream:
            digest = base64.b64encode(hashlib.file_digest(stream, 'md5').digest()).decode()
        if digest == remote.md5:
            return target
    temporary = Path(tempfile.mkdtemp(prefix='.download-', dir=destination))
    try:
        remote.download(root=str(temporary), replace=True)
        downloaded = temporary / name
        with downloaded.open('rb') as stream:
            digest = base64.b64encode(hashlib.file_digest(stream, 'md5').digest()).decode()
        if downloaded.stat().st_size != remote.size or digest != remote.md5:
            raise ValueError(f'Cloud download checksum mismatch: {name}')
        downloaded.replace(target)
    finally:
        shutil.rmtree(temporary)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--slot', choices=('latest', 'fid', 'is'), default='latest')
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--key-file', type=Path)
    parser.add_argument('--extract-code', action='store_true')
    args = parser.parse_args()
    if args.key_file:
        os.environ['WANDB_API_KEY'] = args.key_file.read_text().strip()
    args.destination.mkdir(parents=True, exist_ok=True)
    import wandb
    import torch
    run = wandb.Api(timeout=120).run(args.run)
    bundle = json.loads(download_verified(run, 'resume-bundle.json', args.destination).read_text())
    for dependency in bundle['dependencies']:
        path = download_verified(run, dependency['file'], args.destination)
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != dependency['sha256']:
            raise ValueError(f'Recovery dependency SHA256 mismatch: {path.name}')
    name = 'last.pt' if args.slot == 'latest' else f'best-{args.slot}-resume.pt'
    path = download_verified(run, name, args.destination)
    payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    record = recovery_metadata(payload)
    record.update(run=args.run, file=name, dependencies_verified=True)
    atomic_json(args.destination / 'downloaded-recovery.json', record)
    if args.extract_code:
        with tarfile.open(args.destination / 'resume-frozen-code.tar.gz') as archive:
            archive.extractall(args.destination / 'frozen-code', filter='data')
    print(json.dumps(dict(file=str(path), epoch=record['epoch'],
                          next_microbatch=record['next_microbatch'],
                          global_step=record['global_step'],
                          adam_parameters=record['adam_parameters'], adam_step=record['adam_step'],
                          learning_rates=record['saved_learning_rates'],
                          rng_ranks=record['rng_ranks'], dependencies_verified=True), indent=2))


if __name__ == '__main__':
    main()
