"""Audit every input, preserve fixed splits, and cache aligned 256px RGB images."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

from PIL import Image


def prepare(root, output, workers=4):
    manifests = output/'manifests'
    manifests.mkdir(parents=True, exist_ok=True)
    rows, receipts, ids, hashes, results = {}, {}, {}, {}, {}
    for split, expected in [('train', 28000), ('val', 2000)]:
        paths = sorted(p for p in (root/split).rglob('*')
                       if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp'))
        if len(paths) != expected:
            raise ValueError(f'{split}: expected {expected} images, found {len(paths)}')

        def inspect(path):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            relative = path.relative_to(root/split).with_suffix('.png')
            target = output/'data'/split/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(path) as im:
                im.load()  # Fail on corruption; never substitute another image.
                size, mode = im.size, im.mode
                if size[0] != size[1] or min(size) < 256:
                    raise ValueError(f'Expected a square aligned face of at least 256px: {path} {size}')
                rgb = im.convert('RGB').resize((256, 256), Image.Resampling.LANCZOS)
                rgb.save(target, compress_level=1)
                pixel_digest = hashlib.sha256(rgb.tobytes()).hexdigest()
            return str(relative), digest, pixel_digest, size, mode

        with ThreadPoolExecutor(max_workers=workers) as pool:
            result = list(pool.map(inspect, paths))
        results[split] = result
        rows[split] = [[x[0], 0] for x in result]
        ids[split] = {Path(x[0]).stem for x in result}
        hashes[split] = {x[2] for x in result}
        if len(ids[split]) != expected:
            raise ValueError(f'Duplicate image identity in {split}')
        receipts[split] = dict(count=len(result), dimensions=dict(Counter(str(x[3]) for x in result)),
                               modes=dict(Counter(x[4] for x in result)),
                               source_sha256={x[0]: x[1] for x in result},
                               pixels_sha256={x[0]: x[2] for x in result})
        print(json.dumps(dict(split=split, count=len(result), dimensions=receipts[split]['dimensions'])), flush=True)
    if ids['train'] & ids['val']:
        raise ValueError('Train/validation overlap by image identity')
    # Preserve the held-out split. Exact pixel duplicates in training can
    # otherwise both leak validation examples and overweight repeated faces.
    seen = set(hashes['val'])
    excluded, retained = [], []
    for name, source_digest, pixel_digest, size, mode in results['train']:
        if pixel_digest in seen:
            excluded.append(dict(path=name, pixels_sha256=pixel_digest,
                                 reason='validation_overlap' if pixel_digest in hashes['val'] else 'duplicate_training_pixels'))
        else:
            retained.append([name, 0])
            seen.add(pixel_digest)
    rows['train'] = retained
    for split in rows:
        (manifests/f'{split}-manifest.json').write_text(json.dumps(dict(classes=['face'], samples=rows[split]))+'\n')
    (output/'data-audit.json').write_text(json.dumps(dict(passed=True, source_root=str(root),
        transform='RGB, full-frame Lanczos resize to 256; no random crop',
        label=0, train_val_identity_overlap=0, train_val_pixel_overlap=0,
        retained_train_images=len(retained), excluded_train_images=excluded,
        splits=receipts), indent=2)+'\n')
    print('DATA_AUDIT_PASSED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    prepare(args.root, args.output, args.workers)
