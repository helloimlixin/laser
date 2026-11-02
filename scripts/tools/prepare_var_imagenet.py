"""Prepare verified local ImageNet archives and full VAR image manifests.

Uses the existing archive copier/indexer; VAR augmentation remains in Images.
No image resampling, token cache, or extraction is involved.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def write_manifests(data, output):
    data, output = Path(data).resolve(), Path(output).resolve()
    ready = json.loads((data/'ready.json').read_text())
    if not ready.get('ready'):
        raise ValueError('Archive verification is incomplete')
    for receipt in ready['archives']:
        if not receipt.get('verified'):
            raise ValueError('Unverified archive')
    output.mkdir(parents=True, exist_ok=True)
    shared_classes = None
    for split, expected in [('train', 1281167), ('val', 50000)]:
        index_path = data/f'{split}-index.npz'
        with np.load(index_path) as index:
            classes = index['classes'].tolist()
            names = index['filenames'].tolist()
            labels = index['labels'].tolist()
            offsets, sizes = index['offsets'], index['sizes']
        if (len(names) != expected or len(set(names)) != expected or
                len(classes) != 1000 or classes != sorted(set(classes)) or
                set(labels) != set(range(1000))):
            raise ValueError(f'Incomplete or duplicated {split} index')
        if shared_classes is not None and classes != shared_classes:
            raise ValueError('Training and validation classes differ')
        shared_classes = classes
        archive = data/f'ILSVRC2012_img_{split}.tar'
        if (offsets < 0).any() or (sizes <= 0).any() or (offsets+sizes > archive.stat().st_size).any():
            raise ValueError('Image offsets outside verified archive')
        manifest = dict(classes=classes, samples=[[f'{classes[label]}/{name}', label]
                        for name, label in zip(names, labels)],
                        archive=str(archive), index=str(index_path),
                        data_format='original JPEG bytes; MD5-verified indexed tar',
                        archive_receipts=ready['archives'])
        target = output/f'{split}-manifest.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(manifest, separators=(',', ':'))+'\n')
        temporary.replace(target)
        print(f'{split}: {expected} unique images, 1000 official classes', flush=True)
    ready['training_views'] = 'Fresh deterministic 288px Lanczos resize/256px crop each epoch; no flip'
    (output/'ready.json').write_text(json.dumps(ready, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='/workspace/Projects/data/imagenet/archives')
    parser.add_argument('--data', default='/tmp/laser-imagenet-var/data')
    parser.add_argument('--output', default='outputs/imagenet-var-dictionary-256-20261024/data')
    parser.add_argument('--manifests-only', action='store_true')
    parser.add_argument('--validation-archive', help='Verified replacement for an incomplete source val archive')
    args = parser.parse_args()
    if not args.manifests_only:
        spec = importlib.util.spec_from_file_location('imagenet_archive_preparer',
            Path(__file__).with_name('prepare_original_rq_imagenet_data.py'))
        preparer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(preparer)
        preparer.SOURCE = Path(args.source).resolve()
        preparer.DATA = Path(args.data).resolve()
        preparer.OUT = Path(args.output).resolve().parent
        preparer.main(validation_archive=args.validation_archive)
    write_manifests(args.data, args.output)


if __name__ == '__main__':
    main()
