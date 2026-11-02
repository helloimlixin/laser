"""Checksum original ImageNet archives, repair validation locally, and index JPEGs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request

import prepare_original_rq_imagenet_data as upstream

BASE = Path('/tmp/laser-imagenet-compound512-audit-20261026')


def validation():
    data = BASE/'data'
    data.mkdir(parents=True, exist_ok=True)
    target = data/'ILSVRC2012_img_val.tar'
    expected = '29b22e2961454d5413ddabcf34fc5622'
    if target.exists():
        with target.open('rb') as stream:
            assert hashlib.file_digest(stream, 'md5').hexdigest() == expected
        return
    # The workspace validation archive is incomplete (its tar header is zero).
    # Download the official bytes into a separate local file, preserving source.
    partial = data/'validation.https-partial'
    size, block = 6744924160, 64*1024**2
    fd = os.open(partial, os.O_CREAT|os.O_RDWR|os.O_TRUNC, 0o600)
    os.ftruncate(fd, size)
    def fetch(start):
        end = min(start+block, size)
        for attempt in range(4):
            try:
                request = urllib.request.Request(
                    'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar',
                    headers={'Range': f'bytes={start}-{end-1}'})
                with urllib.request.urlopen(request, timeout=60) as response:
                    assert response.status == 206
                    assert response.headers['Content-Range'] == f'bytes {start}-{end-1}/{size}'
                    offset = start
                    while offset < end:
                        payload = response.read(min(8*1024**2, end-offset))
                        if not payload:
                            raise OSError('Truncated ImageNet range')
                        view = memoryview(payload)
                        while view:
                            count = os.pwrite(fd, view, offset)
                            if count <= 0:
                                raise OSError('Incomplete archive write')
                            offset += count
                            view = view[count:]
                return end-start
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(attempt+1)
    try:
        done = 0
        with ThreadPoolExecutor(max_workers=48) as pool:
            for future in as_completed([pool.submit(fetch, x) for x in range(0, size, block)]):
                done += future.result()
                if done//block % 16 == 0:
                    print(json.dumps(dict(phase='validation_download', bytes=done, total=size)), flush=True)
        os.fsync(fd)
    finally:
        os.close(fd)
    with partial.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'md5').hexdigest()
    if actual != expected:
        raise ValueError(f'Validation archive checksum mismatch: {actual}')
    partial.replace(target)
    target.with_suffix('.verified.json').write_text(json.dumps(dict(bytes=size, md5=actual,
        source='official ImageNet HTTPS', replaced_incomplete_workspace_archive=True), indent=2))
    print('Official validation archive checksum verified', flush=True)


def index():
    upstream.BASE = BASE
    upstream.DATA = BASE/'data'
    upstream.OUT = BASE/'records'
    data = upstream.DATA
    train = data/'ILSVRC2012_img_train.tar'
    assert train.is_file() and train.with_suffix('.verified.json').is_file()
    receipt = json.loads(train.with_suffix('.verified.json').read_text())
    assert receipt['md5'] == '1d675b47d978889d74fa0da5fadfb00e'
    validation()
    shutil.copyfile(upstream.SOURCE/'ILSVRC2012_devkit_t12.tar.gz', data/'ILSVRC2012_devkit_t12.tar.gz')
    classes = upstream.index_train(train)
    upstream.index_val(data/'ILSVRC2012_img_val.tar', classes)
    ready = dict(ready=True, training_images=1281167, validation_images=50000, classes=1000,
        training_archive_md5=receipt['md5'], validation_archive_md5='29b22e2961454d5413ddabcf34fc5622',
        augmentation='Resize256 RandomCrop256 RandomHorizontalFlip(0.5), fresh every epoch',
        source='original JPEGs in verified local tar archives')
    (data/'ready.json').write_text(json.dumps(ready, indent=2)+'\n')
    print(json.dumps(ready), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['validation', 'index'])
    args = parser.parse_args()
    validation() if args.action == 'validation' else index()
