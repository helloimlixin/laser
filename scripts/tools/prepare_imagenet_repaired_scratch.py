"""Stage verified ImageNet data and the frozen rFID-4.21 tokenizer locally."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/tmp/laser-imagenet-repaired-scratch-20261008')
OUT = ROOT/'outputs/imagenet-rfid421-repaired-scratch-4h200-20261008'
ASSETS = ROOT/'outputs/imagenet-rfid421-combination-soft-20260927/assets'


def record(**value):
    value['time'] = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    temporary = OUT/'preparation-status.tmp'
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(OUT/'preparation-status.json')
    print(json.dumps(value), flush=True)


def stage(source, target, algorithm, expected, workers=64):
    if target.exists():
        with target.open('rb') as stream:
            if hashlib.file_digest(stream, algorithm).hexdigest() == expected:
                return
        raise ValueError('Existing staged file failed checksum: '+str(target))
    temporary = target.with_suffix('.partial')
    size = source.stat().st_size
    writer = os.open(temporary, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
    reader = os.open(source, os.O_RDONLY)
    os.ftruncate(writer, size)
    block = 32*1024*1024
    def copy(offset):
        remaining = min(block, size-offset)
        while remaining:
            data = os.pread(reader, min(8*1024*1024, remaining), offset)
            if not data:
                raise IOError('Truncated source: '+str(source))
            view = memoryview(data)
            while view:
                count = os.pwrite(writer, view, offset)
                if count <= 0:
                    raise IOError('Incomplete local write')
                offset += count
                remaining -= count
                view = view[count:]
    completed = 0
    last = time.monotonic()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(copy, n):min(block, size-n) for n in range(0, size, block)}
            for future in concurrent.futures.as_completed(futures):
                future.result()
                completed += futures[future]
                if time.monotonic()-last > 20:
                    record(phase='staging', file=source.name, bytes=completed, total_bytes=size)
                    last = time.monotonic()
        os.fsync(writer)
    finally:
        os.close(reader)
        os.close(writer)
    record(phase='checksum', file=source.name, bytes=size)
    with temporary.open('rb') as stream:
        actual = hashlib.file_digest(stream, algorithm).hexdigest()
    if actual != expected:
        raise ValueError(f'{source.name} checksum mismatch: {actual}')
    temporary.replace(target)


def main():
    (BASE/'inputs').mkdir(parents=True, exist_ok=True)
    for name, source, sha in [
        ('stage1-tokenizer.pt', ASSETS/'laser-tokenizer.pt', 'dd28db9306d526bdc9fbf8016403e106c4f317fd8f5c04792f0953e53310fdab'),
        ('imagenet_256_train.npz', ASSETS/'imagenet_256_train.npz', '3f9c92d15755e76ec312964a819e3e19b9cf3aadc618a7ae99f5e8aa96501260'),
    ]:
        record(phase='stage_dependency', file=name)
        stage(source, BASE/'inputs'/name, 'sha256', sha)
    archive = BASE/'ILSVRC2012_img_train.tar'
    stage(Path('/workspace/Projects/data/imagenet/archives/ILSVRC2012_img_train.verified-20261005.tar'),
          archive, 'md5', '1d675b47d978889d74fa0da5fadfb00e')
    train = BASE/'imagenet/train'
    train.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as outer:
        members = outer.getmembers()
    assert len(members) == 1000
    assert all(m.isfile() and re.fullmatch(r'n\d{8}\.tar', m.name) for m in members)
    def extract(member):
        synset = member.name[:-4]
        folder = train/synset
        folder.mkdir(exist_ok=True)
        count = 0
        with archive.open('rb') as stream:
            stream.seek(member.offset_data)
            with tarfile.open(fileobj=stream, mode='r|') as inner:
                for image in inner:
                    if not image.isfile() or not re.fullmatch(synset+r'_\d+\.JPEG', image.name):
                        raise ValueError('Unexpected ImageNet member: '+image.name)
                    target = folder/image.name
                    if not target.exists() or target.stat().st_size != image.size:
                        with inner.extractfile(image) as source, target.open('wb') as dest:
                            shutil.copyfileobj(source, dest)
                    assert target.stat().st_size == image.size
                    count += 1
        return count
    counts = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        for future in concurrent.futures.as_completed([pool.submit(extract, m) for m in members]):
            counts.append(future.result())
            if len(counts) % 25 == 0:
                record(phase='extract', classes=len(counts), images=sum(counts))
    assert sum(counts) == 1281167
    ready = dict(passed=True, classes=1000, training_images=sum(counts),
                 archive_md5='1d675b47d978889d74fa0da5fadfb00e', time=time.time())
    (BASE/'imagenet/training-ready.json').write_text(json.dumps(ready, indent=2)+'\n')
    record(phase='ready', **ready)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        record(phase='failed', error=str(error))
        raise
