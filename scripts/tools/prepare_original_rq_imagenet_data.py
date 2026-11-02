#!/usr/bin/env python3
"""Copy verified ImageNet archives locally and index JPEGs without extraction."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import time

import numpy as np

BASE = Path('/tmp/laser-original-rq-imagenet-20261014')
DATA = BASE / 'data'
SOURCE = Path('/workspace/Projects/data/imagenet/archives')
OUT = Path('/workspace/Projects/laser/outputs/imagenet-original-rq1400m-rfid473-scratch20-4h100-20261014')


def record(name, value):
    for root in (DATA, OUT / 'data-verification'):
        root.mkdir(parents=True,exist_ok=True)
        path = root / name
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value,indent=2)+'\n')
        temporary.replace(path)


def copy_archive(source, target, expected_md5, workers=24):
    size = source.stat().st_size
    if target.exists() and target.with_suffix('.verified.json').exists():
        receipt = json.loads(target.with_suffix('.verified.json').read_text())
        assert receipt['bytes']==size and receipt['md5']==expected_md5
        return receipt
    partial = target.with_suffix('.partial')
    source_fd = os.open(source,os.O_RDONLY)
    destination_fd = os.open(partial,os.O_CREAT|os.O_RDWR,0o600)
    os.ftruncate(destination_fd,size)
    chunk = 256*1024**2
    started = time.monotonic()
    done = 0
    def copy_segment(first):
        last = min(first+chunk,size)
        offset = first
        while offset<last:
            data = os.pread(source_fd,min(8*1024**2,last-offset),offset)
            if not data: raise OSError(f'Truncated source at {offset}')
            view = memoryview(data)
            while view:
                written = os.pwrite(destination_fd,view,offset)
                if written<=0:raise OSError('Incomplete copy')
                offset += written
                view = view[written:]
        return last-first
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(copy_segment,first) for first in range(0,size,chunk)]
        for future in concurrent.futures.as_completed(futures):
            done += future.result()
            if done//chunk % 8 == 0 or done==size:
                status = dict(phase='copying',archive=source.name,bytes_copied=done,total_bytes=size,
                              elapsed_seconds=time.monotonic()-started,time=time.time())
                record('status.json',status)
                print(json.dumps(status),flush=True)
    os.fsync(destination_fd)
    os.close(source_fd);os.close(destination_fd)
    record('status.json',dict(phase='verifying_md5',archive=source.name,bytes=size))
    with partial.open('rb') as stream:
        md5 = hashlib.file_digest(stream,'md5').hexdigest()
    if md5 != expected_md5:raise RuntimeError(f'{source}: MD5 {md5}, expected {expected_md5}')
    partial.replace(target)
    receipt = dict(source=str(source),target=str(target),bytes=size,md5=md5,verified=True)
    target.with_suffix('.verified.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def index_train(path):
    offsets,sizes,labels,names = [],[],[],[]
    with tarfile.open(path,'r:') as outer:
        classes = sorted(outer.getmembers(),key=lambda m:m.name)
        assert len(classes)==1000
        wnids = [m.name.removesuffix('.tar') for m in classes]
        for label,member in enumerate(classes):
            assert member.isfile() and member.name.endswith('.tar')
            with outer.extractfile(member) as stream,tarfile.open(fileobj=stream,mode='r:') as inner:
                images = sorted(inner.getmembers(),key=lambda m:m.name)
                for image in images:
                    assert image.isfile() and image.name.startswith(wnids[label]+'_') and image.name.endswith('.JPEG')
                    offsets.append(member.offset_data+image.offset_data)
                    sizes.append(image.size);labels.append(label);names.append(image.name)
            if (label+1)%25==0:
                status=dict(phase='indexing_train',classes=label+1,images=len(offsets),time=time.time())
                record('status.json',status);print(json.dumps(status),flush=True)
    assert len(offsets)==1281167
    np.savez(DATA/'train-index.npz',offsets=np.asarray(offsets,dtype=np.int64),
             sizes=np.asarray(sizes,dtype=np.int32),labels=np.asarray(labels,dtype=np.int16),
             filenames=np.asarray(names),classes=np.asarray(wnids))
    return wnids


def index_val(path,wnids):
    from torchvision.datasets.imagenet import parse_devkit_archive,load_meta_file
    parse_devkit_archive(str(DATA))
    _,val_wnids = load_meta_file(str(DATA))
    assert len(val_wnids)==50000
    class_map={name:i for i,name in enumerate(wnids)}
    with tarfile.open(path,'r:') as archive:
        images=sorted(archive.getmembers(),key=lambda m:m.name)
        assert len(images)==50000
        labels=[class_map[val_wnids[int(m.name.removesuffix('.JPEG').rsplit('_',1)[1])-1]] for m in images]
        np.savez(DATA/'val-index.npz',offsets=np.asarray([m.offset_data for m in images],dtype=np.int64),
                 sizes=np.asarray([m.size for m in images],dtype=np.int32),
                 labels=np.asarray(labels,dtype=np.int16),filenames=np.asarray([m.name for m in images]),
                 classes=np.asarray(wnids))


def main(validation_archive=None):
    DATA.mkdir(parents=True,exist_ok=True)
    if (DATA/'ready.json').exists():print((DATA/'ready.json').read_text());return
    train=DATA/'ILSVRC2012_img_train.tar'
    val=DATA/'ILSVRC2012_img_val.tar'
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        a=pool.submit(copy_archive,SOURCE/'ILSVRC2012_img_train.verified-20261005.tar',train,'1d675b47d978889d74fa0da5fadfb00e',24)
        validation_source = Path(validation_archive) if validation_archive else SOURCE/'ILSVRC2012_img_val.tar'
        b=pool.submit(copy_archive,validation_source,val,'29b22e2961454d5413ddabcf34fc5622',8)
        receipts=[a.result(),b.result()]
    shutil.copyfile(SOURCE/'ILSVRC2012_devkit_t12.tar.gz',DATA/'ILSVRC2012_devkit_t12.tar.gz')
    wnids=index_train(train)
    index_val(val,wnids)
    report=dict(ready=True,training_images=1281167,validation_images=50000,classes=1000,
                archives=receipts,data_format='indexed original JPEGs in local uncompressed archives',
                training_views='fresh upstream Resize(256), RandomCrop(256), horizontal flip',time=time.time())
    record('ready.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
