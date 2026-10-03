"""Prepare the official 50k validation reference used by the v8 run."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import time
import urllib.request
from scipy.io import loadmat

BASE = Path('/tmp/laser-imagenet-stage2/imagenet')
BASE.mkdir(parents=True, exist_ok=True)
archive = BASE / 'ILSVRC2012_img_val.tar'
url = 'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar'
expected = '29b22e2961454d5413ddabcf34fc5622'
if not archive.exists():
    from concurrent.futures import ThreadPoolExecutor
    import os
    size = 6744924160
    parts = BASE / 'validation-parts'
    parts.mkdir(exist_ok=True)
    def download(index):
        start = index * size // 16
        end = (index + 1) * size // 16 - 1
        target = parts / (str(index) + '.part')
        if target.exists() and target.stat().st_size == end-start+1:
            return target
        req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
        with urllib.request.urlopen(req, timeout=90) as response, target.open('wb') as out:
            assert response.status == 206 and response.headers['Content-Range'] == f'bytes {start}-{end}/{size}'
            shutil.copyfileobj(response, out, 4 * 1024 * 1024)
        assert target.stat().st_size == end-start+1
        return target
    with ThreadPoolExecutor(max_workers=16) as pool:
        downloads = list(pool.map(download, range(16)))
    partial = archive.with_suffix('.assembled')
    with partial.open('wb') as out:
        for part in downloads:
            with part.open('rb') as stream:
                shutil.copyfileobj(stream,out,16*1024*1024)
    partial.replace(archive)
with archive.open('rb') as stream:
    checksum = hashlib.file_digest(stream, 'md5').hexdigest()
assert checksum == expected, (checksum, expected)
print('Official validation archive checksum verified', flush=True)
devkit = Path('/workspace/Projects/data/imagenet/archives/ILSVRC2012_devkit_t12.tar.gz')
with devkit.open('rb') as stream:
    assert hashlib.file_digest(stream, 'md5').hexdigest() == 'fa75699e90414af021442c21a62c3abf'
with tarfile.open(devkit) as tar:
    synsets = loadmat(io.BytesIO(tar.extractfile('ILSVRC2012_devkit_t12/data/meta.mat').read()), squeeze_me=True)['synsets']
    mapping = {int(s[0]): str(s[1]) for s in synsets if int(s[4]) == 0}
    truth = [int(v) for v in tar.extractfile('ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt').read().split()]
assert len(truth) == 50000 and len(mapping) == 1000
for wnid in mapping.values():
    (BASE / 'val' / wnid).mkdir(parents=True, exist_ok=True)
count = 0
with tarfile.open(archive) as tar:
    for member in tar:
        if not member.isfile():
            continue
        name = Path(member.name).name
        index = int(name.rsplit('_', 1)[1].split('.')[0]) - 1
        dest = BASE / 'val' / mapping[truth[index]] / name
        if not dest.exists():
            with tar.extractfile(member) as source, dest.open('wb') as output:
                shutil.copyfileobj(source, output)
        count += 1
assert count == 50000
assert sum(1 for p in (BASE / 'val').glob('*/*.JPEG')) == 50000
(BASE / 'validation-ready.json').write_text(json.dumps(dict(images=50000, classes=1000, md5=checksum, source=url, prepared_unix=time.time()), indent=2))
print('Validation reference ready: 50,000 images and 1,000 classes', flush=True)
