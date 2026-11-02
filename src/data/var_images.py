"""Validated manifest loading and dataset-specific VAR image preprocessing."""
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def load_manifests(directory, dataset):
    manifests = {s: json.loads((Path(directory) / f'{s}-manifest.json').read_text())
                 for s in ('train', 'val')}
    classes = manifests['train']['classes']
    if classes != manifests['val']['classes'] or not classes:
        raise ValueError('Train and validation label mappings differ or are empty')
    if dataset not in ('imagenet', 'celebahq'):
        raise ValueError(f'Unsupported VAR dataset: {dataset}')
    if dataset == 'celebahq' and classes != ['face']:
        raise ValueError('CelebA-HQ is unconditional: every image must use the face label 0')
    identities = {}
    for split, manifest in manifests.items():
        rows = manifest['samples']
        if not rows:
            raise ValueError(f'Empty {split} manifest')
        paths = [row[0] for row in rows]
        if len(set(paths)) != len(paths):
            raise ValueError(f'Duplicate paths in {split} manifest')
        for name, label in rows:
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError(f'Unsafe relative image path: {name}')
            if not isinstance(label, int) or not 0 <= label < len(classes):
                raise ValueError(f'Invalid class label in {split}: {label}')
        if dataset == 'imagenet':
            expected = 1281167 if split == 'train' else 50000
            if len(rows) != expected or len(classes) != 1000:
                raise ValueError(f'Incomplete ImageNet {split} manifest')
        else:
            # The local tree uses gender directories; identity is the source
            # filename, never a gender label or directory-dependent class ID.
            identities[split] = {Path(name).stem for name in paths}
            if len(identities[split]) != len(paths):
                raise ValueError(f'Duplicate CelebA-HQ image identity in {split}')
    if dataset == 'celebahq' and identities['train'] & identities['val']:
        raise ValueError('CelebA-HQ train/validation image identities overlap')
    return manifests


class Images(Dataset):
    def __init__(self, root, manifest, train, seed, dataset='imagenet', horizontal_flip=False):
        self.root, self.rows, self.seed = Path(root), manifest['samples'], int(seed)
        self.epoch = 0
        if dataset == 'imagenet':
            geometry = [transforms.Resize(288, interpolation=InterpolationMode.LANCZOS),
                        transforms.RandomCrop(256) if train else transforms.CenterCrop(256)]
        elif dataset == 'celebahq':
            # Preserve the aligned face frame, also for validation/FID. Never
            # apply ImageNet's 288 -> 256 random crop to aligned faces.
            geometry = [transforms.Resize(256, interpolation=InterpolationMode.LANCZOS),
                        transforms.CenterCrop(256)]
        else:
            raise ValueError(f'Unsupported VAR dataset: {dataset}')
        if train and horizontal_flip:
            geometry.append(transforms.RandomHorizontalFlip())
        self.transform = transforms.Compose(geometry + [transforms.ToTensor(), transforms.Normalize(.5, .5)])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        index = int(index)  # NumPy permutations are used by evaluation subsets.
        name, label = self.rows[index]
        with Image.open(self.root / name) as im:
            image = self.transform_image(im, index)
        return image, label

    def transform_image(self, im, index):
        with torch.random.fork_rng(devices=[]):
            # Seed CPU augmentation only. torch.manual_seed also resets CUDA
            # dropout when num_workers=0, despite fork_rng(devices=[])!
            torch.random.default_generator.manual_seed(self.seed + self.epoch * 10000019 + index)
            image = self.transform(im.convert('RGB'))
        if image.shape != (3, 256, 256) or not torch.isfinite(image).all():
            raise ValueError(f'Invalid image tensor at index {index}')
        return image


class IndexedImageNet(Images):
    """Read original JPEG bytes from a verified, uncompressed ImageNet tar.

    The offset indices are produced by prepare_original_rq_imagenet_data.py.
    Pixel transforms and epoch/index augmentation seeds match folder loading.
    Each process opens its own descriptor; pread does not share seek state.
    """
    def __init__(self, root, split, seed):
        if split not in ('train', 'val'):
            raise ValueError(f'Invalid ImageNet split: {split}')
        super().__init__(root, {'samples': []}, split == 'train', seed)
        self.archive = self.root / f'ILSVRC2012_img_{split}.tar'
        with np.load(self.root / f'{split}-index.npz', allow_pickle=False) as index:
            self.offsets = index['offsets']
            self.sizes = index['sizes']
            self.labels = index['labels']
            self.classes = index['classes'].tolist()
        if (self.offsets.ndim != 1 or self.sizes.shape != self.offsets.shape or
                self.labels.shape != self.offsets.shape or not len(self.offsets) or
                any(a.dtype.kind not in 'iu' for a in (self.offsets, self.sizes, self.labels)) or
                self.classes != sorted(set(self.classes)) or not self.classes or
                np.any(self.labels < 0) or np.any(self.labels >= len(self.classes)) or
                np.any(self.offsets < 0) or np.any(self.sizes <= 0) or
                np.any(self.offsets + self.sizes > self.archive.stat().st_size)):
            raise ValueError(f'Invalid ImageNet {split} archive index')
        self._descriptor = None
        self._pid = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        index = int(index)
        if self._pid != os.getpid():
            if self._descriptor is not None:
                os.close(self._descriptor)
            self._descriptor = os.open(self.archive, os.O_RDONLY)
            self._pid = os.getpid()
        size = int(self.sizes[index])
        data = os.pread(self._descriptor, size, int(self.offsets[index]))
        if len(data) != size:
            raise OSError(f'Truncated ImageNet JPEG at index {index}')
        with Image.open(io.BytesIO(data)) as im:
            image = self.transform_image(im, index)
        return image, int(self.labels[index])

    def __getstate__(self):
        return dict(self.__dict__, _descriptor=None, _pid=None)

    def __del__(self):
        descriptor = getattr(self, '_descriptor', None)
        if descriptor is not None:
            os.close(descriptor)
