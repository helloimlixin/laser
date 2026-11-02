#!/usr/bin/env python3
"""Stage verified CC3M shards on local storage in training access order."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import yaml

ROOT = Path('/workspace/Projects/laser')
sys.path.insert(0, str(ROOT))
from src.data.cc3m_joint import shard_index, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--download', action='store_true',
        help='Fetch the same pinned, checksummed bytes directly to local storage')
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    base = Path(cfg['local_checkpoints']).parent
    manifest = json.loads(Path(cfg['data_manifest']).read_text())
    permutation = np.random.default_rng(np.random.SeedSequence([cfg['seed'], 0])).permutation(len(manifest['train']))
    first = [manifest['train'][int(i)] for i in permutation[:cfg['shuffle_group_shards']]]
    remaining = [manifest['train'][int(i)] for i in permutation[cfg['shuffle_group_shards']:]]
    specs = manifest['validation'] + first + remaining
    if args.download:
        os.environ.setdefault('HF_XET_HIGH_PERFORMANCE', '1')
        os.environ.setdefault('HF_XET_CACHE', str(base/'xet-cache'))
        os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
        from huggingface_hub import hf_hub_download

    def stage(spec):
        if args.download:
            local = Path(cfg['data_local'])
            path = local/spec['name']
            receipt = path.with_suffix('.verified.json')
            with (local/(spec['name']+'.lock')).open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if not (path.is_file() and path.stat().st_size == spec['bytes'] and receipt.is_file()
                        and json.loads(receipt.read_text())['sha256'] == spec['sha256']):
                    fetched = Path(hf_hub_download(repo_id='pixparse/cc3m-wds', repo_type='dataset',
                        revision=cfg['dataset_revision'], filename=spec['name'], local_dir=local))
                    assert fetched == path and path.stat().st_size == spec['bytes']
                    with path.open('rb') as stream:
                        assert hashlib.file_digest(stream, 'sha256').hexdigest() == spec['sha256']
                    write_json(receipt, dict(sha256=spec['sha256'], bytes=spec['bytes'],
                        source='pinned public dataset; direct download to local storage'))
        shard_index(spec, cfg['data_source'], cfg['data_local'])
        return spec['name']

    completed = 0
    with ThreadPoolExecutor(max_workers=8 if args.download else 4) as pool:
        for name in pool.map(stage, specs):
            completed += 1
            row = dict(completed=completed, total=len(specs), last=name, time=time.time())
            write_json(base/'staging.json', row)
            if completed == len(manifest['validation'])+len(first):
                write_json(base/'initial-shards-ready.json', dict(
                    validation_shards=len(manifest['validation']), training_shards=len(first),
                    training_shard_names=[s['name'] for s in first], time=time.time()))
            print(json.dumps(row), flush=True)
    write_json(base/'all-shards-ready.json', dict(completed=completed, time=time.time()))


if __name__ == '__main__':
    main()
