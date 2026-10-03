"""Download, verify, and extract official training images for fresh augmentation."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
import threading
import time
import urllib.request


BASE = Path('/tmp/laser-imagenet-stage2/imagenet')
URL = 'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar'
SIZE = 147897477120
MD5 = '1d675b47d978889d74fa0da5fadfb00e'
CHUNK = 128 * 1024 * 1024
WORKERS = int(os.environ.get('LASER_IMAGENET_DOWNLOAD_WORKERS', '64'))
ARCHIVE = BASE / 'ILSVRC2012_img_train.tar'
CHUNKS = BASE / 'training-download-chunks.json'
STATUS = BASE / 'training-preparation-status.json'
READY = BASE / 'training-ready.json'
LOCK = threading.Lock()
started = time.monotonic()


def record(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    if READY.exists():
        print(READY.read_text(), flush=True)
        return
    total = (SIZE + CHUNK - 1) // CHUNK
    done = set(json.loads(CHUNKS.read_text())['completed']) if CHUNKS.exists() else set()
    descriptor = os.open(ARCHIVE, os.O_RDWR | os.O_CREAT, 0o600)
    os.ftruncate(descriptor, SIZE)
    downloaded = sum(min(CHUNK, SIZE-index*CHUNK) for index in done)

    def download(index):
        nonlocal downloaded
        first, last = index * CHUNK, min(SIZE, (index+1)*CHUNK)-1
        for attempt in range(5):
            try:
                request = urllib.request.Request(URL, headers={'Range': f'bytes={first}-{last}'})
                with urllib.request.urlopen(request, timeout=45) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {first}-{last}/{SIZE}':
                        raise RuntimeError('Training server returned the wrong byte range')
                    offset = first
                    while data := response.read(8*1024*1024):
                        view = memoryview(data)
                        while view:
                            count = os.pwrite(descriptor, view, offset)
                            if count <= 0: raise OSError('Incomplete archive write')
                            offset += count
                            view = view[count:]
                    if offset != last+1: raise OSError('Truncated download segment')
                with LOCK:
                    done.add(index)
                    downloaded += last-first+1
                    record(CHUNKS, dict(size=SIZE, chunk_bytes=CHUNK, completed=sorted(done)))
                return
            except Exception:
                if attempt == 4: raise
                time.sleep(1+attempt)

    print(json.dumps(dict(phase='download', total_bytes=SIZE, workers=WORKERS)), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(download, i) for i in range(total) if i not in done]
        pending = set(futures)
        while pending:
            finished, pending = concurrent.futures.wait(pending, timeout=20,
                return_when=concurrent.futures.FIRST_EXCEPTION)
            for future in finished: future.result()
            with LOCK:
                progress = dict(phase='download', completed_chunks=len(done), total_chunks=total,
                    completed_bytes=downloaded, total_bytes=SIZE,
                    elapsed_seconds=time.monotonic()-started)
            record(STATUS, progress)
            print(json.dumps(progress), flush=True)
    os.fsync(descriptor)
    os.close(descriptor)
    record(STATUS, dict(phase='checksum', total_bytes=SIZE))
    print(json.dumps(dict(phase='checksum')), flush=True)
    with ARCHIVE.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'md5').hexdigest()
    if actual != MD5: raise RuntimeError(f'Training archive checksum mismatch: {actual}')
    classes = sorted(p.name for p in (BASE/'val').iterdir() if p.is_dir())
    assert len(classes) == 1000
    with tarfile.open(ARCHIVE, 'r:') as outer:
        members = outer.getmembers()
    assert len(members) == 1000
    assert sorted(m.name[:-4] for m in members) == classes
    assert all(m.isfile() and re.fullmatch(r'n\d{8}\.tar', m.name) for m in members)
    train = BASE/'train'
    markers = BASE/'training-prepared-classes'
    markers.mkdir(exist_ok=True)

    def extract(member):
        wnid = member.name[:-4]
        marker = markers/wnid
        if marker.exists(): return int(marker.read_text())
        folder = train/wnid
        folder.mkdir(parents=True, exist_ok=True)
        count = 0
        with ARCHIVE.open('rb') as stream:
            stream.seek(member.offset_data)
            with tarfile.open(fileobj=stream, mode='r|') as inner:
                for image in inner:
                    assert image.isfile() and re.fullmatch(wnid+r'_\d+\.JPEG', image.name)
                    target = folder/image.name
                    if not target.exists() or target.stat().st_size != image.size:
                        temporary = target.with_suffix('.partial')
                        with inner.extractfile(image) as source, temporary.open('wb') as output:
                            shutil.copyfileobj(source, output, 1024*1024)
                        assert temporary.stat().st_size == image.size
                        temporary.replace(target)
                    count += 1
        marker.write_text(str(count))
        return count

    print(json.dumps(dict(phase='extraction', workers=16)), flush=True)
    counts = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        for future in concurrent.futures.as_completed([pool.submit(extract, m) for m in members]):
            counts.append(future.result())
            if len(counts) % 25 == 0:
                progress = dict(phase='extraction', completed_classes=len(counts), classes=1000,
                                images=sum(counts), elapsed_seconds=time.monotonic()-started)
                record(STATUS, progress)
                print(json.dumps(progress), flush=True)
    assert sum(counts) == 1281167
    assert sorted(p.name for p in train.iterdir() if p.is_dir()) == classes
    report = dict(phase='ready', training_images=sum(counts), classes=1000,
                  md5=actual, archive_bytes=SIZE, training_root=str(train),
                  elapsed_seconds=time.monotonic()-started)
    record(READY, report)
    record(STATUS, report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
