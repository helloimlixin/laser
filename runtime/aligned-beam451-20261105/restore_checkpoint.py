"""Restore a chunked W&B checkpoint and verify its complete SHA256."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main(args):
    os.environ['WANDB_API_KEY']=Path('/root/.config/laser/wandb.key').read_text().strip()
    os.environ['WANDB_BASE_URL']='https://api.forge.coreweave.com'
    import wandb
    assert not args.output.exists(),'Refusing to overwrite an existing checkpoint'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    folder=Path(wandb.Api(timeout=120).artifact(args.artifact).download(root=str(args.output.parent/'artifact-parts')))
    manifest=json.loads((folder/'checkpoint-manifest.json').read_text())
    digest=hashlib.sha256();size=0;temp=args.output.with_suffix('.partial')
    with temp.open('wb') as output:
        for part in manifest['parts']:
            part_hash=hashlib.sha256();part_size=0
            with (folder/part['name']).open('rb') as handle:
                while block:=handle.read(16*1024**2):
                    output.write(block);part_hash.update(block);digest.update(block);part_size+=len(block)
            assert part_hash.hexdigest()==part['sha256'] and part_size==part['bytes']
            size+=part_size
    assert digest.hexdigest()==manifest['sha256'] and size==manifest['bytes']
    temp.replace(args.output);print(json.dumps(dict(restored=str(args.output),bytes=size,sha256=digest.hexdigest())))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--artifact',required=True)
    parser.add_argument('--output',type=Path,required=True);main(parser.parse_args())
