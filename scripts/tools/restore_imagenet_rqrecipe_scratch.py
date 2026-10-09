"""Restore this online ImageNet scratch run without resetting its Adam or RNG state.

Example:
  python scripts/tools/restore_imagenet_rqrecipe_scratch.py \
    --destination /tmp/laser-recovered --checkpoint-dir /workspace/recovered-checkpoints \
    --data /path/to/verified/imagenet --key-file /path/to/private-wandb-key
Add --launch to resume training on eight GPUs after preparing the files.
"""
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile


def download(run, name, directory):
    remote = run.file(name)
    target = directory / name
    if target.is_file() and target.stat().st_size == remote.size:
        with target.open('rb') as stream:
            if base64.b64encode(hashlib.file_digest(stream, 'md5').digest()).decode() == remote.md5:
                return target
    with tempfile.TemporaryDirectory(prefix='.download-', dir=directory) as temporary:
        remote.download(root=temporary, replace=True)
        source = Path(temporary) / name
        with source.open('rb') as stream:
            digest = base64.b64encode(hashlib.file_digest(stream, 'md5').digest()).decode()
        if source.stat().st_size != remote.size or digest != remote.md5:
            raise ValueError('Recovery download checksum mismatch: ' + name)
        source.replace(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='helloimlixin-rutgers/laser/imagenet-rfid421-rqrecipe-scratch-8gpu-20261004')
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--key-file', type=Path, required=True)
    parser.add_argument('--slot', choices=['latest', 'fid', 'is'], default='latest')
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    os.environ['WANDB_API_KEY'] = args.key_file.read_text().strip()
    import wandb
    import yaml
    import torch
    base = args.destination.expanduser().resolve()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    if not checkpoint_dir.is_relative_to(Path('/workspace')):
        parser.error('--checkpoint-dir must be under /workspace')
    base.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    inputs = base / 'inputs'
    inputs.mkdir(exist_ok=True)
    run = wandb.Api(timeout=120).run(args.run)
    bundle = json.loads(download(run, 'resume-bundle.json', inputs).read_text())
    for dependency in bundle['dependencies']:
        path = download(run, dependency['file'], inputs)
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != dependency['sha256']:
                raise ValueError('Dependency SHA256 mismatch: ' + path.name)
    selected = 'last.pt' if args.slot == 'latest' else f'best-{args.slot}-resume.pt'
    download(run, selected, inputs)
    for name in ['best-fid-resume.pt', 'best-is-resume.pt']:
        if name == selected:
            continue
        try:
            remote = run.file(name)
        except (ValueError, IndexError):
            continue
        if remote.size:
            download(run, name, inputs)
    source_dir = base / 'source'
    source_dir.mkdir(exist_ok=True)
    with tarfile.open(inputs / 'resume-frozen-code.tar.gz') as archive:
        archive.extractall(source_dir, filter='data')
    shutil.copytree(source_dir / 'support', base / 'support', dirs_exist_ok=True)
    shutil.copyfile(source_dir / 'entry.py', base / 'entry.py')
    sys.path[:0] = [str(source_dir / 'runtime'), str(base / 'support')]
    from src.training.full_resume_upload import recovery_metadata
    selected = 'last.pt' if args.slot == 'latest' else f'best-{args.slot}-resume.pt'
    payload = torch.load(inputs / selected, map_location='cpu', weights_only=False, mmap=True)
    metadata = recovery_metadata(payload)
    if metadata['world_size'] != 8:
        raise ValueError('This run requires its saved eight-GPU layout')
    # Filename changes relocate checkpoint references only; tensor, Adam,
    # scheduler, sampler cursor, and RNG values remain byte-for-byte saved.
    for kind, key in [('fid', 'best_fid'), ('is', 'best_inception')]:
        for _, saved in payload.get(key, []):
            target = checkpoint_dir / Path(saved).name
            if target.exists() or target.is_symlink():
                raise FileExistsError('Refuse to overwrite an existing winner: ' + str(target))
            target.symlink_to(inputs / f'best-{kind}-resume.pt')
    latest = checkpoint_dir / 'last.pt'
    if latest.exists() or latest.is_symlink():
        raise FileExistsError('Refuse to overwrite an existing recovery checkpoint: ' + str(latest))
    latest.symlink_to(inputs / selected)
    recipe = yaml.safe_load((inputs / 'resume-active-config.yaml').read_text())
    recipe['options'].update(checkpoint=str(inputs / 'resume-stage1-tokenizer.pt'),
        data=str(args.data.expanduser().resolve()), wandb_id=bundle['run'], output=str(base / 'production/train'),
        checkpoint_dir=str(checkpoint_dir), resume=True, wandb_mode='online')
    recipe['options'].pop('resume_checkpoint', None)
    config = base / 'recipe.yaml'
    config.write_text(yaml.safe_dump(recipe, sort_keys=False))
    accumulation = math.ceil(recipe['options']['total_batch_size'] / (8 * recipe['options']['batch_size']))
    evidence = checkpoint_dir.parent
    (evidence / 'cloud-restore-verification.json').write_text(json.dumps(metadata, indent=2, default=str))
    env = os.environ.copy()
    env.update(LASER_RUN_BASE=str(base), LASER_PERSISTENT_BASE=str(evidence), LASER_PHASE='production',
        LASER_CHECKPOINT_STAGING_DIR=str(base / 'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(base / 'checkpoint-upload-cache'),
        LASER_CHECKPOINT_IMMUTABLE_FILES='1', LASER_ACCUMULATION=str(accumulation),
        LASER_COMPILE_BLOCKS='1', CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
        WANDB_DIR=str(base / 'wandb'), WANDB_CACHE_DIR=str(base / 'wandb-cache'), WANDB_DATA_DIR=str(base / 'wandb-data'),
        TORCHINDUCTOR_CACHE_DIR=str(base / 'inductor-cache'), TORCH_HOME=str(base / 'torch-cache'),
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        PYTHONUNBUFFERED='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', NCCL_NVLS_ENABLE='0')
    for name in ['wandb', 'wandb-cache', 'wandb-data']:
        (base / name).mkdir(exist_ok=True)
    weights = base / 'torch-cache/hub/checkpoints'
    weights.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(inputs / 'resume-weights-inception-2015-12-05-6726825d.pth',
                    weights / 'weights-inception-2015-12-05-6726825d.pth')
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8',
               str(base / 'entry.py'), '--config', str(config)]
    print(json.dumps(dict(checkpoint=selected, epoch=metadata['epoch'],
        next_microbatch=metadata['next_microbatch'], global_step=metadata['global_step'],
        adam_step=metadata['adam_step'], learning_rates=metadata['saved_learning_rates'],
        dependencies_verified=True, command=command), indent=2))
    if args.launch:
        return subprocess.call(command, env=env, cwd=base)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
