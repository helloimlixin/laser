#!/usr/bin/env python3
"""Resume the exact saved compound recipe after cache and input verification."""
import argparse
import json
import os
from pathlib import Path
import sys


def recipe_arguments(base, *, local=Path('/workspace')):
    sys.path.insert(0, str(base / 'source'))
    from src.training.rqtransformer import build_parser
    config = json.loads((base / 'prepared/source-config.json').read_text())
    config.update(checkpoint=str(base / 'prepared/tokenizer.pt'), data=str(base / 'data'),
                  token_cache=str(local / 'cache/compound-cache.pt'),
                  resume_checkpoint=str(local / 'inputs/last.pt'),
                  output=str(local / 'train'), checkpoint_dir=str(local / 'checkpoints'),
                  fid_reference_stats=str(base / 'reference/real-statistics.npz'),
                  wandb_mode='online', upload_checkpoints=True, upload_token_cache=True)
    parser = build_parser()
    argv = []
    for action in parser._actions:
        if action.dest not in config or not action.option_strings:
            continue
        value = config[action.dest]
        if value is None:
            continue
        flag = action.option_strings[0]
        if isinstance(action, argparse.BooleanOptionalAction):
            argv.append(flag if value else '--no-' + flag[2:])
        elif isinstance(action, argparse._StoreTrueAction):
            if value:
                argv.append(flag)
        elif isinstance(action, argparse._StoreFalseAction):
            if not value:
                argv.append(flag)
        elif isinstance(value, list):
            argv.extend([flag, *map(str, value)])
        else:
            argv.extend([flag, str(value)])
    return argv


def main():
    from church_compound_support import sha, atomic_json
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--check-config', action='store_true')
    args = parser.parse_args()
    base = args.base
    argv = recipe_arguments(base)
    from src.training import rqtransformer as recipe
    parsed = recipe.build_parser().parse_args(argv)
    if args.check_config:
        print(json.dumps(vars(parsed), default=str, indent=2))
        return
    import torch
    world = int(os.environ['WORLD_SIZE'])
    if world not in (4, 8):
        raise RuntimeError('The validated continuation requires four or eight ranks')
    ready = json.loads((base / 'prepared/cache-ready.json').read_text())
    if not ready['passed'] or sha(parsed.token_cache) != ready['cache_sha256']:
        raise RuntimeError('Missing, incomplete, or changed compound cache')
    preflight = json.loads((base / 'preflight.json').read_text())
    selection = json.loads(Path('/workspace/source-selection.json').read_text())
    recovered = selection['validated_checkpoint']
    if sha(parsed.resume_checkpoint) != recovered['sha256']:
        raise RuntimeError('Staged resume checkpoint differs from the allocation selection')
    local_rank = int(os.environ['LOCAL_RANK'])
    name = torch.cuda.get_device_name(local_rank)
    properties = torch.cuda.get_device_properties(local_rank)
    if properties.major < 8 or properties.total_memory < 23 * 1024**3:
        raise RuntimeError(f'Resume needs native BF16 and at least a 24 GB class GPU: {name}')
    record = dict(rank=int(os.environ['RANK']), host=os.uname().nodename,
                  gpu=name, gpu_memory_bytes=properties.total_memory,
                  world_size=world, accumulation_steps=128 // (16 * world),
                  torch_version=torch.__version__,
                  resume_step=recovered['step'], cache_sha256=ready['cache_sha256'])
    atomic_json(base / 'train' / f"rank-{record['rank']}.json", record)
    print(json.dumps(record), flush=True)
    recipe.main(argv)


if __name__ == '__main__':
    main()
