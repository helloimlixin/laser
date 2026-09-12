#!/usr/bin/env python3
"""Recover restartable stage 1 checkpoints using standard W&B credentials."""
import argparse
import hashlib
import json
from pathlib import Path

import wandb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='helloimlixin-rutgers/laser/s2er91dm')
    parser.add_argument('--output', type=Path, default=Path('outputs/mdctcodec_recovery/s2er91dm'))
    args = parser.parse_args()
    api = wandb.Api(timeout=120)
    run = api.run(args.run)
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ('config.yaml', 'wandb-metadata.json', 'diff.patch'):
        run.file(name).download(root=str(args.output), replace=True)
    artifact = api.artifact(f'{run.entity}/{run.project}/model-{run.id}-selected-checkpoints:latest')
    directory = Path(artifact.download(root=str(args.output / 'checkpoints')))
    manifest = {
        'source_run': args.run, 'source_url': run.url, 'artifact': artifact.qualified_name,
        'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob('*.ckpt')},
    }
    (args.output / 'recovery.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if not (directory / 'last.ckpt').is_file():
        raise RuntimeError('The selected artifact has no restartable last.ckpt')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
