#!/usr/bin/env python3
"""Validate the LSUN corpus and invoke the established continuous pair cache builder."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from church_compound_support import atomic_json, sha


def verify_data(base):
    sys.path.insert(0, str(base / 'source'))
    from src.training.rqtransformer import source_image_dataset, val_image_transform
    protocol = json.loads((base / 'reference/data-protocol.json').read_text())
    records = {}
    for split, name in [('train', 'church')]:
        data = source_image_dataset('lsun_church', base / 'data', val_image_transform(), split=split)
        expected = protocol['cache_stage2_fid']['datasets'][name]
        assert len(data) == expected['images']
        digest = hashlib.sha256()
        for key in data.keys:
            digest.update(len(key).to_bytes(8, 'little'))
            digest.update(key)
        assert digest.hexdigest() == expected['key_order_sha256']
        for index, expected_sha in expected['pixel_probes'].items():
            actual = hashlib.sha256(data[int(index)][0].numpy().tobytes()).hexdigest()
            assert actual == expected_sha, (split, index, actual, expected_sha)
        records[split] = dict(items=len(data), key_order_sha256=digest.hexdigest(),
                              pixel_probes=len(expected['pixel_probes']))
        data.env.close()
    atomic_json(base / 'prepared/data-validation.json', dict(passed=True, **records))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--verify-data-only', action='store_true')
    args = parser.parse_args()
    base = args.base
    verify_data(base)
    if args.verify_data_only:
        print((base / 'prepared/data-validation.json').read_text())
        return
    import torch
    local_rank = int(os.environ['LOCAL_RANK'])
    gpu = torch.cuda.get_device_name(local_rank)
    properties = torch.cuda.get_device_properties(local_rank)
    assert properties.major >= 8 and properties.total_memory >= 23 * 1024**3, gpu
    preflight = json.loads((base / 'preflight.json').read_text())
    assert sha(base / 'prepared/tokenizer.pt') == preflight['tokenizer']['export_sha256']
    spec = importlib.util.spec_from_file_location('cache_builder', base / 'source/scripts/tools/build_official_imagenet_token_cache.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    sys.argv = ['cache_builder', '--checkpoint', str(base / 'prepared/tokenizer.pt'),
                '--data', str(base / 'data'), '--output', str(base / 'prepared/compound-cache.pt'),
                '--dataset', 'lsun_church', '--num-atoms', '16384', '--sparsity-level', '4',
                '--coeff-vocab-size', '2048', '--coeff-max', '3', '--coeff-scale', '6.4',
                '--coeff-scales', *map(str, preflight['coeff_scales']),
                '--batch-size', '16', '--num-workers', '2', '--verify-samples', '256', '--compound']
    atomic_json(base / 'prepared/cache-start.json', dict(gpu=gpu, host=os.uname().nodename,
                 job_id=os.environ.get('SLURM_JOB_ID'), command=sys.argv))
    builder.main()


if __name__ == '__main__':
    main()
