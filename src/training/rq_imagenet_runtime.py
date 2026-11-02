"""JPEG access and small runtime helpers for the original ImageNet RQ run."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,default=str)+'\n')
    temporary.replace(path)


def sha256(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def local_training_logs(base, folder):
    """Put per-update diagnostics on NVMe, preserving durable checkpoint paths.

    The public train path still resolves to live diagnostics. Checkpoints have
    a separate persistent alias so recovery does not depend on the NVMe view.
    Call with trainers stopped, or once on rank zero before training starts.
    """
    base,folder=Path(base),Path(folder)
    persistent=folder.parent/'persistent-train'
    local=base/'live-training'/folder.name
    cache=os.environ.get('LASER_CHECKPOINT_UPLOAD_CACHE_DIR')
    receipts=[]
    if not folder.is_symlink() and folder.exists() and cache:
        prefix=str(folder.resolve())+'/checkpoints/'
        for receipt in (Path(cache)/'objects').glob('*.json'):
            metadata=json.loads(receipt.read_text())
            if metadata.get('source','').startswith(prefix):
                receipts.append((receipt,metadata,metadata['source'][len(prefix):]))
    if not folder.is_symlink() and folder.exists():
        if persistent.exists():raise RuntimeError('Persistent training archive already exists')
        folder.rename(persistent)
    persistent.mkdir(parents=True,exist_ok=True)
    (persistent/'checkpoints').mkdir(exist_ok=True)
    local.mkdir(parents=True,exist_ok=True)
    for path in persistent.iterdir():
        if path.is_file() and not path.is_symlink() and not (local/path.name).exists():
            shutil.copyfile(path,local/path.name)
    link=local/'checkpoints'
    if not link.exists():link.symlink_to(persistent/'checkpoints',target_is_directory=True)
    if not folder.is_symlink():folder.symlink_to(local,target_is_directory=True)
    durable=folder.parent/'checkpoints'
    if not durable.exists():durable.symlink_to('persistent-train/checkpoints',target_is_directory=True)
    # A directory rename changes the resolved payload path, not its bytes.
    # Re-key the already validated local serialization without rereading it.
    for receipt,metadata,suffix in receipts:
        source=(persistent/'checkpoints'/suffix).resolve()
        original=receipt.with_suffix('.pt')
        assert original.stat().st_size==source.stat().st_size
        key=hashlib.sha256(str(source).encode()).hexdigest()
        target=receipt.parent/(key+'.pt')
        if not target.exists():os.link(original,target)
        stat=source.stat()
        metadata.update(source=str(source),identity={name:getattr(stat,name) for name in
            ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')})
        write_json(target.with_suffix('.json'),metadata)
    return dict(live_directory=str(local),persistent_checkpoint_directory=str(persistent/'checkpoints'),
        durable_checkpoint_alias=str(durable),live_path=str(folder))


class TarImageNet(Dataset):
    def __init__(self, root, split, transform, seed=261001, *, deterministic_train_views=True):
        root=Path(root)
        self.archive=str(root/f'ILSVRC2012_img_{split}.tar')
        with np.load(root/f'{split}-index.npz') as values:
            self.offsets=values['offsets'];self.sizes=values['sizes'];self.labels=values['labels']
            self.classes=values['classes'].tolist()
        self.transform=transform;self.split=split;self.seed=seed
        self.deterministic_train_views=bool(deterministic_train_views)
        self.epoch=torch.zeros((),dtype=torch.long).share_memory_()
        self.descriptor=None

    def __len__(self):return len(self.labels)

    def set_epoch(self, epoch):self.epoch.fill_(epoch)

    def __getitem__(self,index):
        if self.descriptor is None:self.descriptor=os.open(self.archive,os.O_RDONLY)
        size=int(self.sizes[index]);offset=int(self.offsets[index])
        data=os.pread(self.descriptor,size,offset)
        if len(data)!=size:raise OSError(f'Incomplete ImageNet JPEG at {index}')
        with Image.open(io.BytesIO(data)) as image:
            image=image.convert('RGB')
            if self.split=='train' and self.deterministic_train_views:
                seed=(self.seed+1000003*int(self.epoch)+1000000007*index)%(2**63-1)
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(seed)
                    image=self.transform(image)
            else:image=self.transform(image)
        return image,int(self.labels[index])


def cpu_snapshot(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_snapshot(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_snapshot(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_snapshot(v) for v in value)
    return value


def codebook_digest(tokenizer):
    digest=hashlib.sha256()
    for name,value in tokenizer.quantizer.state_dict().items():
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
