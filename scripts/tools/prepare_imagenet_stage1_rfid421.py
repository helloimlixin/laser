#!/usr/bin/env python3
"""Prepare verified local JPEG archives for the corrected small-dictionary run.

The validation manifest is the pinned source receipt from the original A1024 run.
It records SHA-256 for all fourteen parquet containers. JPEG bytes are copied
unchanged; the damaged workspace validation tar is never used for evaluation.
"""
import argparse
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile

import numpy as np
import pyarrow.parquet as pq
import requests

import prepare_original_rq_imagenet_data as preparation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validation-manifest', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path('/workspace/Projects/data/imagenet/archives'))
    parser.add_argument('--cache', type=Path, default=Path('/dev/shm/laser-imagenet-a1024-20261029'))
    args = parser.parse_args()
    data = args.base/'data'; data.mkdir(parents=True,exist_ok=True)
    args.cache.mkdir(parents=True,exist_ok=True)
    preparation.DATA = data; preparation.OUT = args.output
    train = args.cache/'ILSVRC2012_img_train.tar'
    receipt = preparation.copy_archive(args.source/'ILSVRC2012_img_train.verified-20261005.tar',
        train,'1d675b47d978889d74fa0da5fadfb00e',32)
    link = data/train.name
    if link.is_symlink():link.unlink()
    if not link.exists():link.symlink_to(train)
    if not (data/'train-index.npz').exists():
        wnids = preparation.index_train(train)
    else:
        with np.load(data/'train-index.npz') as index:wnids = index['classes'].tolist()
    manifest = json.loads(args.validation_manifest.read_text())
    shards = args.base/'validation-parquet'; shards.mkdir(exist_ok=True)

    def download(record):
        target = shards/Path(record['path']).name
        if not target.exists():
            url = (f'https://huggingface.co/datasets/mrm8488/ImageNet1K-val/resolve/'
                   f'{manifest["validation_revision"]}/{record["path"]}')
            partial = target.with_suffix('.partial')
            with requests.get(url,stream=True,timeout=120) as response:
                response.raise_for_status()
                with partial.open('wb') as handle:
                    for block in response.iter_content(8*1024**2):handle.write(block)
            partial.replace(target)
        assert target.stat().st_size == record['bytes']
        with target.open('rb') as handle:digest = hashlib.file_digest(handle,'sha256').hexdigest()
        assert digest == record['sha256'],target
        return target

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        paths = list(pool.map(download,manifest['validation_parquet_files']))
    clean = args.base/'clean-validation'; clean.mkdir(exist_ok=True)
    if not (clean/'ready.json').exists():
        rows = []; counts = np.zeros(1000,np.int64)
        with tarfile.open(clean/'ILSVRC2012_img_val.tar','w') as archive:
            for path in sorted(paths):
                for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
                    for row in batch.to_pylist():
                        payload = row['image']['bytes']; label = row['label']
                        assert payload[:2] == b'\xff\xd8'
                        name = f'validation_{len(rows)+1:08d}.JPEG'
                        info = tarfile.TarInfo(name); info.size = len(payload)
                        offset = archive.offset+512
                        archive.addfile(info,io.BytesIO(payload))
                        rows.append((offset,len(payload),label,name)); counts[label] += 1
        assert len(rows) == 50000 and (counts == 50).all()
        offsets,sizes,labels,names = zip(*rows)
        np.savez(clean/'val-index.npz',offsets=np.array(offsets,np.int64),
            sizes=np.array(sizes,np.int32),labels=np.array(labels,np.int16),
            filenames=np.array(names),classes=np.array(wnids))
        (clean/'ready.json').write_text(json.dumps(dict(ready=True,images=50000,reencoded=False)))
    link = data/'ILSVRC2012_img_val.tar'
    if link.is_symlink():link.unlink()
    if not link.exists():link.symlink_to(clean/link.name)
    shutil.copyfile(clean/'val-index.npz',data/'val-index.npz')
    with np.load(data/'train-index.npz') as index:
        assert len(index['labels']) == 1281167 and len(index['classes']) == 1000
    report = dict(ready=True,training_images=1281167,validation_images=50000,
        classes=1000,training_archive=receipt,validation_source=manifest['validation_source'],
        validation_revision=manifest['validation_revision'],validation_jpeg_reencoded=False,
        validation_shards_sha256_verified=True,damaged_workspace_validation_tar_used=False)
    preparation.record('ready.json',report)
    print(json.dumps(report),flush=True)


if __name__ == '__main__':main()
