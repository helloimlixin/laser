#!/usr/bin/env python3
"""Resolve one immutable checkpoint version, then stage it consistently per node."""
import argparse
import json
from pathlib import Path

import wandb

RUN_ID = 'church-original-rqvae-released-tokenizer-control-20260917'
COLLECTION = f'helloimlixin-rutgers/laser/model-{RUN_ID}-selected-checkpoints'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['resolve', 'stage'])
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--local', type=Path)
    p.add_argument('--node-rank', type=int, default=0)
    args = p.parse_args()
    api = wandb.Api(timeout=120)
    if args.action == 'resolve':
        run = api.run(f'helloimlixin-rutgers/laser/{RUN_ID}')
        if run.state == 'running':
            raise RuntimeError('W&B reports an active run; check for another live trainer before launching')
        a = api.artifact(COLLECTION + ':latest')
        assert a.metadata['run_id'] == RUN_ID
        assert a.metadata['optimizer_step'] >= 1984
        resolved = dict(artifact=a.source_qualified_name, digest=a.digest, metadata=a.metadata)
        args.selection.write_text(json.dumps(resolved, indent=2) + '\n')
        print(json.dumps(resolved), flush=True)
        return
    resolved = json.loads(args.selection.read_text())
    a = api.artifact(resolved['artifact'])
    assert a.digest == resolved['digest']
    inputs = args.local / 'inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    a.get_entry('last.pt').download(root=str(inputs), skip_cache=True)
    if args.node_rank == 0:
        output = args.local / 'checkpoints'
        output.mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(resolved['metadata']['ranked_fid_checkpoints'], 1):
            path = Path(a.get_entry(f'best-fid-{index:02d}.pt').download(root=str(inputs), skip_cache=True))
            path.replace(output / row['path'])
    (args.local / 'source-artifact.json').write_text(json.dumps(resolved, indent=2) + '\n')
    print(json.dumps(dict(phase='checkpoint_staged', artifact=resolved['artifact'], node_rank=args.node_rank)), flush=True)


if __name__ == '__main__':
    main()
