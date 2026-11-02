"""Full-coverage ImageNet JPEG access for the coarse-to-fine pair prior."""
import concurrent.futures
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


def tar_image_records(data, base_offset=0):
    """Read normal JPEG members from one uncompressed class tar payload."""
    position=0
    while position+512<=len(data):
        header=data[position:position+512]
        if not header.strip(b'\0'):break
        name=header[:100].split(b'\0')[0].decode()
        size=int(header[124:136].rstrip(b'\0 ').lstrip(b' ') or b'0',8)
        end=position+512+size
        if end>len(data):raise EOFError('truncated class tar member '+name)
        if name.endswith('.JPEG') and header[156:157] in (b'0',b'\0'):
            if len(name.encode())>63:raise ValueError('ImageNet member name exceeds index width')
            yield base_offset+position+512,size,name
        position+=512+((size+511)//512)*512


def _pwrite_all(fd, data, offset):
    view=memoryview(data)
    while len(view):
        count=os.pwrite(fd,view,offset)
        if count<=0:raise OSError('incomplete local archive write')
        view=view[count:];offset+=count


def build_index(source, classes_path, heldout_manifest, output, *, local_archive=None,
                workers=8, expected_images=1281167):
    source,output=Path(source),Path(output)
    if output.exists():raise FileExistsError(output)
    classes=json.loads(Path(classes_path).read_text())
    heldout={row['name'] for row in json.loads(Path(heldout_manifest).read_text()) if row['split']=='heldout'}
    original_fd=os.open(source,os.O_RDONLY);copy_fd=None
    if local_archive is not None:
        local_archive=Path(local_archive);local_archive.parent.mkdir(parents=True,exist_ok=True)
        copy_fd=os.open(local_archive,os.O_CREAT|os.O_EXCL|os.O_RDWR,0o600)
        os.ftruncate(copy_fd,source.stat().st_size)
    def one(item):
        label,(name,offset,size)=item
        if size%512:raise ValueError('class tar is not block aligned')
        payload=os.pread(original_fd,size,offset)
        if len(payload)!=size:raise EOFError(name)
        records=list(tar_image_records(payload,offset))
        if copy_fd is not None:
            _pwrite_all(copy_fd,os.pread(original_fd,512,offset-512),offset-512)
            _pwrite_all(copy_fd,payload,offset)
        return label,records,hashlib.sha256(payload).hexdigest()
    rows=[];digests={}
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for done,(label,records,digest) in enumerate(pool.map(one,enumerate(classes))):
                rows.extend((offset,size,label,name) for offset,size,name in records)
                digests[classes[label][0]]=digest
                if done%50==0:print(json.dumps(dict(indexed_classes=done+1,images=len(rows))),flush=True)
        if len(rows)!=expected_images:raise ValueError(f'expected {expected_images} images, found {len(rows)}')
        if copy_fd is not None:
            tail=classes[-1][1]+classes[-1][2]
            _pwrite_all(copy_fd,os.pread(original_fd,source.stat().st_size-tail,tail),tail)
            os.fsync(copy_fd)
    finally:
        os.close(original_fd)
        if copy_fd is not None:os.close(copy_fd)
    names=np.asarray([r[3] for r in rows],dtype='S64')
    reserved=np.asarray([r[3] in heldout for r in rows])
    labels=np.asarray([r[2] for r in rows],dtype=np.int16)
    if reserved.sum()!=len(heldout) or len(np.unique(labels[reserved]))!=len(classes):
        raise ValueError('heldout split must reserve exactly one image per class')
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix('.tmp.npz')
    np.savez_compressed(temporary,offsets=np.asarray([r[0] for r in rows],dtype=np.int64),
        sizes=np.asarray([r[1] for r in rows],dtype=np.int32),labels=labels,names=names,
        train=np.flatnonzero(~reserved),heldout=np.flatnonzero(reserved))
    temporary.replace(output)
    metadata=dict(source=str(source),archive=str(local_archive or source),archive_bytes=source.stat().st_size,
        images=len(rows),train_images=int((~reserved).sum()),heldout_images=int(reserved.sum()),
        class_payload_sha256=digests,full_coverage=True)
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


class IndexedPairImages(Dataset):
    def __init__(self, archive, index, transform, *, split='train', seed=261029):
        self.archive=str(archive);self.transform=transform;self.seed=int(seed);self.split=split
        with np.load(index) as data:
            self.offsets=data['offsets'];self.sizes=data['sizes'];self.labels=data['labels']
            self.names=data['names'];self.indices=data[split]
        self.epoch=torch.zeros((),dtype=torch.long).share_memory_();self.descriptor=None

    def __len__(self):return len(self.indices)

    def set_epoch(self, epoch):self.epoch.fill_(epoch)

    def __getitem__(self, index):
        full=int(self.indices[index])
        if self.descriptor is None:self.descriptor=os.open(self.archive,os.O_RDONLY)
        size=int(self.sizes[full]);data=os.pread(self.descriptor,size,int(self.offsets[full]))
        if len(data)!=size:raise OSError('incomplete JPEG at '+str(full))
        with Image.open(io.BytesIO(data)) as image:
            image=image.convert('RGB')
            if self.split=='train':
                seed=(self.seed+1000003*int(self.epoch)+1000000007*full)%(2**63-1)
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(seed)
                    pixels=self.transform(image)
            else:pixels=self.transform(image)
        return pixels,int(self.labels[full])
