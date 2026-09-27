"""Validated manifest loading and dataset-specific VAR image preprocessing."""
import json
from pathlib import Path

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
        with torch.random.fork_rng(devices=[]):
            # Seed CPU augmentation only. torch.manual_seed also resets CUDA
            # dropout when num_workers=0, despite fork_rng(devices=[])!
            torch.random.default_generator.manual_seed(self.seed + self.epoch * 10000019 + index)
            with Image.open(self.root / name) as im:
                image = self.transform(im.convert('RGB'))
        if image.shape != (3, 256, 256) or not torch.isfinite(image).all():
            raise ValueError(f'Invalid image tensor: {self.root / name}')
        return image, label
