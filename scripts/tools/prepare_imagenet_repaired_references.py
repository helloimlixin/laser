"""Build complete validation references for companion FID comparisons."""
import hashlib
import concurrent.futures
import json
import os
from pathlib import Path
import re
import sys
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/tmp/laser-imagenet-repaired-scratch-20261008')
OUT = ROOT/'outputs/imagenet-rfid421-repaired-scratch-4h200-20261008'
sys.path[:0] = [str(ROOT), str(ROOT/'runtime'), str(ROOT/'scripts/tools')]
import prepare_imagenet_repaired_scratch as preparation


def record(**value):
    value['time'] = time.time()
    path = OUT/'reference-preparation-status.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)
    print(json.dumps(value), flush=True)


def main():
    archive = BASE/'ILSVRC2012_img_val.tar'
    record(phase='stage_validation_archive')
    preparation.record = record
    try:
        preparation.stage(Path('/workspace/Projects/data/imagenet/archives/ILSVRC2012_img_val.tar'), archive,
              'md5', '29b22e2961454d5413ddabcf34fc5622')
    except ValueError:
        size = 6744924160
        temporary = archive.with_suffix('.https-partial')
        descriptor = os.open(temporary, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
        os.ftruncate(descriptor, size)
        block = 64*1024*1024
        def download(start):
            end = min(size, start+block)
            for attempt in range(4):
                try:
                    request = urllib.request.Request('https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar',
                        headers={'Range':f'bytes={start}-{end-1}'})
                    with urllib.request.urlopen(request, timeout=60) as response:
                        assert response.status == 206
                        assert response.headers['Content-Range'] == f'bytes {start}-{end-1}/{size}'
                        offset = start
                        while offset < end:
                            data = response.read(min(8*1024*1024,end-offset))
                            if not data:
                                raise IOError('Truncated validation range')
                            view = memoryview(data)
                            while view:
                                written = os.pwrite(descriptor,view,offset)
                                if written <= 0:
                                    raise IOError('Incomplete validation staging write')
                                offset += written
                                view = view[written:]
                    return end-start
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(attempt+1)
        record(phase='download_verified_validation_archive')
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=48) as pool:
                total = sum(pool.map(download,range(0,size,block)))
            assert total == size
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        with temporary.open('rb') as stream:
            assert hashlib.file_digest(stream,'md5').hexdigest() == '29b22e2961454d5413ddabcf34fc5622'
        temporary.replace(archive)
    folder = BASE/'validation-reference-images/validation'
    folder.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        assert len(members) == 50000
        assert {m.name for m in members} == {f'ILSVRC2012_val_{i:08d}.JPEG' for i in range(1,50001)}
        assert all(m.isfile() and re.fullmatch(r'ILSVRC2012_val_\d{8}\.JPEG', m.name) for m in members)
        bundle.extractall(folder, filter='data')
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    import torchmetrics
    from torchmetrics.image.fid import FrechetInceptionDistance
    from src.training.rqtransformer import val_image_transform
    from src.rqvae_metrics import DistributedOriginalRQVAEMetrics, _mean_covariance
    from src.training.fid_reference import SCHEMA, load_torchmetrics_reference
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    dataset = ImageFolder(BASE/'validation-reference-images', transform=val_image_transform())
    assert len(dataset) == 50000
    loader = DataLoader(dataset, batch_size=128, num_workers=8, pin_memory=True)
    original = DistributedOriginalRQVAEMetrics(torch.device('cuda:3'))
    companion = FrechetInceptionDistance(feature=2048, normalize=True, sync_on_compute=False).to('cuda:3').eval()
    record(phase='extract_validation_features', images=0)
    with torch.inference_mode():
        for batch, (images, _) in enumerate(loader):
            pixels = ((images.to('cuda:3').float()+1)*.5).clamp(0,1)
            original.update(pixels, real=True)
            companion.update(pixels, real=True)
            if batch % 40 == 0:
                record(phase='extract_validation_features', images=int(original.real_count))
    assert int(original.real_count) == int(companion.real_features_num_samples) == 50000
    mean, covariance = _mean_covariance(original.real_sum, original.real_cross, 50000)
    np.savez(BASE/'inputs/imagenet_val_original.npz', mu=mean, sigma=covariance)
    model_digest = hashlib.sha256()
    for name, tensor in companion.inception.state_dict().items():
        model_digest.update(name.encode())
        model_digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    metadata = dict(samples=50000, real_split='val', original_classes=1000,
        source_archive_md5='29b22e2961454d5413ddabcf34fc5622',
        inception_state_sha256=model_digest.hexdigest(), torchmetrics_version=torchmetrics.__version__,
        transform='Resize256 CenterCrop256; [-1,1] normalization then clamp((x+1)/2,0,1)',
        reference_label_grouping='All 50000 validation pixels; labels are irrelevant to FID moments',
        no_duplicate_indices=True, feature_dim=2048, created_unix=time.time())
    payload = dict(schema=SCHEMA, feature_dim=2048, metadata=metadata,
        **{name:getattr(companion,name).cpu() for name in
           ('real_features_sum','real_features_cov_sum','real_features_num_samples')})
    target = BASE/'inputs/imagenet_val_torchmetrics.pt'
    torch.save(payload, target)
    load_torchmetrics_reference(target, expected_samples=50000)
    for name in ('imagenet_val_original.npz','imagenet_val_torchmetrics.pt'):
        p = BASE/'inputs'/name
        with p.open('rb') as stream:
            sha = hashlib.file_digest(stream,'sha256').hexdigest()
        metadata[name] = dict(bytes=p.stat().st_size, sha256=sha)
    (OUT/'validation-reference-verification.json').write_text(json.dumps(dict(passed=True, **metadata),indent=2)+'\n')
    record(phase='ready', **metadata)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        record(phase='failed', error=str(error))
        raise
