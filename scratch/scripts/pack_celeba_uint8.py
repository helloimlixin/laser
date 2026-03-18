#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def scan_image_paths(root: Path):
    image_paths = []
    for dirpath, _, filenames in os.walk(root):
        base = Path(dirpath)
        for name in filenames:
            path = base / name
            if path.suffix.lower() in IMG_EXTENSIONS:
                image_paths.append(path)
    image_paths.sort()
    return image_paths


def main() -> None:
    parser = argparse.ArgumentParser(description='Pack CelebA into a single uint8 NumPy memmap file.')
    parser.add_argument('--src_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--flush_every', type=int, default=512)
    args = parser.parse_args()

    src_dir = Path(args.src_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    image_size = int(args.image_size)
    if image_size <= 0:
        raise ValueError('image_size must be > 0')
    if not src_dir.exists():
        raise FileNotFoundError(f'source directory does not exist: {src_dir}')

    image_paths = scan_image_paths(src_dir)
    if not image_paths:
        raise RuntimeError(f'no images found under {src_dir}')
    if int(args.limit) > 0:
        image_paths = image_paths[: int(args.limit)]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'celeba_{image_size}x{image_size}_rgb_uint8.npy'
    meta_path = out_dir / f'celeba_{image_size}x{image_size}_rgb_uint8.json'
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f'packed dataset already exists: {out_path}')
    if out_path.exists() and args.overwrite:
        out_path.unlink()

    count = len(image_paths)
    print(f'[Pack] src_dir={src_dir}')
    print(f'[Pack] out_path={out_path}')
    print(f'[Pack] image_size={image_size}')
    print(f'[Pack] count={count}')

    data = np.lib.format.open_memmap(
        out_path,
        mode='w+',
        dtype=np.uint8,
        shape=(count, image_size, image_size, 3),
    )

    started = time.time()
    flush_every = max(1, int(args.flush_every))
    for idx, path in enumerate(image_paths):
        with Image.open(path) as img:
            img = img.convert('RGB')
            if img.size != (image_size, image_size):
                img = img.resize((image_size, image_size), resample=Image.BILINEAR)
            data[idx] = np.asarray(img, dtype=np.uint8)
        if (idx + 1) % flush_every == 0:
            data.flush()
            elapsed = max(time.time() - started, 1e-6)
            rate = float(idx + 1) / elapsed
            print(f'[Pack] {idx + 1}/{count} images ({rate:.1f} img/s)')

    data.flush()
    elapsed = max(time.time() - started, 1e-6)
    meta = {
        'src_dir': str(src_dir),
        'out_path': str(out_path),
        'count': count,
        'image_size': image_size,
        'dtype': 'uint8',
        'shape': [count, image_size, image_size, 3],
        'elapsed_seconds': elapsed,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'[Pack] done in {elapsed:.1f}s')
    print(f'[Pack] meta_path={meta_path}')


if __name__ == '__main__':
    main()
