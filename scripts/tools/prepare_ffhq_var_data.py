"""Build lossless FFHQ-256 Arrow splits from the official archive and metadata."""
import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import time
import zipfile

from PIL import Image


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def archive_index(archive, metadata, expected_counts):
    counts = Counter(row['category'] for row in metadata.values())
    if dict(counts) != expected_counts:
        raise ValueError(f'Unexpected official FFHQ split counts: {dict(counts)}')
    entries = {}
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            if entry.is_dir() or not entry.filename.lower().endswith('.png'):
                continue
            key = str(int(Path(entry.filename).stem))
            if key in entries or key not in metadata:
                raise ValueError(f'Duplicate or unknown FFHQ image: {entry.filename}')
            if entry.file_size != metadata[key]['image']['file_size']:
                raise ValueError(f'FFHQ archive size mismatch: {key}')
            entries[key] = entry.filename
    if set(entries) != set(metadata):
        raise ValueError('The archive does not contain every metadata image')
    return entries


def generate_rows(shards, archive, metadata_path, split, workers, progress_dir):
    metadata = json.loads(Path(metadata_path).read_text())
    with zipfile.ZipFile(archive) as source:
        names = {str(int(Path(name).stem)): name for name in source.namelist()
                 if name.lower().endswith('.png')}
        keys = sorted((int(key) for key, row in metadata.items() if row['category'] == split))
        for shard in shards:
            done = 0
            for index in keys[shard::workers]:
                entry = metadata[str(index)]['image']
                original = source.read(names[str(index)])  # Includes ZIP CRC validation.
                if hashlib.md5(original).hexdigest() != entry['file_md5']:
                    raise ValueError(f'Official FFHQ MD5 mismatch: {index}')
                with Image.open(io.BytesIO(original)) as image:
                    if image.size != (1024, 1024):
                        raise ValueError(f'Unexpected FFHQ resolution: {index}: {image.size}')
                    resized = image.convert('RGB').resize((256, 256), Image.Resampling.LANCZOS)
                    buffer = io.BytesIO()
                    resized.save(buffer, format='PNG', compress_level=1)
                yield dict(image=dict(bytes=buffer.getvalue(), path=None), label=0,
                           image_id=index, source_md5=entry['file_md5'])
                done += 1
                if done % 100 == 0:
                    Path(progress_dir, f'{split}-worker{shard:02d}.json').write_text(
                        json.dumps(dict(images=done, time=time.time())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value, ClassLabel
    if (args.output / 'manifest.json').exists():
        print('FFHQ data is already prepared.', flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    progress = args.output / 'progress'
    progress.mkdir(exist_ok=True)
    metadata_path = args.source / 'ffhq-dataset-v2.json'
    archive = args.source / 'images1024x1024.zip'
    metadata = json.loads(metadata_path.read_text())
    entries = archive_index(archive, metadata, dict(training=60000, validation=10000))
    print(json.dumps(dict(phase='verified_index', images=len(entries), split_counts=dict(training=60000, validation=10000))), flush=True)
    features = Features(dict(image=HFImage(), label=ClassLabel(names=['face']),
                             image_id=Value('int32'), source_md5=Value('string')))
    datasets = {}
    started = time.time()
    for target, split in [('train', 'training'), ('validation', 'validation')]:
        datasets[target] = Dataset.from_generator(generate_rows, features=features,
            gen_kwargs=dict(shards=list(range(args.workers)), archive=str(archive),
                            metadata_path=str(metadata_path), split=split, workers=args.workers,
                            progress_dir=str(progress)),
            cache_dir=str(args.output / 'build-cache'), num_proc=args.workers,
            keep_in_memory=False)
        print(json.dumps(dict(phase='split_ready', split=target, images=len(datasets[target]))), flush=True)
    dataset = DatasetDict(datasets)
    temporary = args.output / 'hf-building'
    dataset.save_to_disk(str(temporary), max_shard_size='512MB')
    temporary.replace(args.output / 'hf')
    manifest = dict(source_archive=str(archive), source_metadata_sha256=sha256(metadata_path),
        source_archive_bytes=archive.stat().st_size, original_images=70000,
        original_md5_and_zip_crc_verified=True, image_size=256,
        transform='RGB; Pillow LANCZOS resize 1024x1024 -> 256x256; lossless PNG',
        split_source='Official FFHQ metadata category; no random re-split',
        split_counts={name:len(data) for name,data in datasets.items()},
        dataset_fingerprints={name:data._fingerprint for name,data in datasets.items()},
        labels='all zero; unconditional FFHQ', elapsed_seconds=time.time()-started,
        files={str(path.relative_to(args.output)):dict(bytes=path.stat().st_size, sha256=sha256(path))
               for path in sorted((args.output/'hf').rglob('*.arrow'))})
    path = args.output / 'manifest.json'
    path.with_suffix('.tmp').write_text(json.dumps(manifest, indent=2) + '\n')
    path.with_suffix('.tmp').replace(path)
    print(json.dumps(dict(complete=True, **{key:manifest[key] for key in ('split_counts','elapsed_seconds')})), flush=True)


if __name__ == '__main__':
    main()
