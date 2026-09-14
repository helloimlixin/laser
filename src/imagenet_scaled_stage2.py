"""ImageNet adapters for the released class-conditional RQTransformer.

The sparse atom/coefficient integer vocabulary is shared with the Church run.
Cached encoder outputs preserve stochastic residual-quantization training targets.
"""
from contextlib import nullcontext
import json
from pathlib import Path
import random
from types import MethodType

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.data import Dataset

from src.scaled_atom_training import soft_cross_entropy


def load_imagenet_config(upstream, vocab_size=131073):
    from rqvae.utils.config import load_config, augment_arch_defaults, augment_optimizer_defaults
    config = load_config(Path(upstream) / 'configs/imagenet256/stage2/in256-rqtransformer-8x8x4-480M.yaml')
    config.arch = augment_arch_defaults(config.arch)
    config.arch.vocab_size = config.dataset.vocab_size = vocab_size
    config.optimizer = augment_optimizer_defaults(config.optimizer)
    config.seed = 0
    config.sampling = dict(temp=1., top_k=config.experiment.sample.top_k,
                           top_p=config.experiment.sample.top_p)
    return config


class ManifestImages(Dataset):
    """Stable labels from the official sorted ImageNet WNID class order."""
    def __init__(self, root, manifest, transform, view=0, seed=0, indices=None):
        self.root = Path(root)
        self.rows = manifest['samples']
        self.transform = transform
        self.view, self.seed = view, seed
        self.indices = indices if indices is not None else range(len(self.rows))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, offset):
        index = self.indices[offset]
        name, label = self.rows[index]
        # Image augmentation is reproducible across workers, ranks, and restarts.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed + self.view * 10000019 + index)
            with Image.open(self.root / name) as image:
                image = self.transform(image.convert('RGB'))
        return image, label, index


def image_manifest(root):
    root = Path(root)
    classes = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith('n'))
    if len(classes) != 1000:
        raise ValueError(f'Expected all 1,000 ImageNet classes in {root}')
    samples = []
    for label, name in enumerate(classes):
        files = sorted(p.name for p in (root/name).iterdir()
                       if p.suffix.lower() in {'.jpeg', '.jpg', '.png'})
        if not files:
            raise ValueError(f'Empty class {name}')
        samples.extend((f'{name}/{file}', label) for file in files)
    return dict(classes=classes, samples=samples, images=len(samples),
                class_to_idx={name:i for i,name in enumerate(classes)})


class CachedClassLatents(Dataset):
    """One view per image each epoch, alternating the cached augmented views."""
    def __init__(self, cache, split='train', epoch=0, include_hard_codes=True):
        cache = Path(cache)
        self.spec = json.loads((cache/'complete.json').read_text())
        self.views = self.spec['train_views'] if split == 'train' else 1
        self.latents = [np.load(cache/f'{split}-view{v}-latents.npy', mmap_mode='r')
                        for v in range(self.views)]
        self.labels = np.load(cache/f'{split}-labels.npy', mmap_mode='r')
        self.hard = ([np.load(cache/f'{split}-view{v}-codes.npy', mmap_mode='r')
                      for v in range(self.views)] if include_hard_codes else [])
        self.epoch = epoch
        for array in self.latents:
            assert array.dtype == np.float32 and array.shape == (len(self.labels),8,8,256)
        for array in self.hard:
            assert array.dtype == np.uint32 and array.shape == (len(self.labels),8,8,4)
        assert self.labels.min() >= 0 and self.labels.max() < 1000

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        view = (self.epoch + index) % self.views
        hard = (torch.from_numpy(self.hard[view][index].astype(np.int64))
                if self.hard else torch.empty(0, dtype=torch.long))
        return (torch.from_numpy(self.latents[view][index].copy()), int(self.labels[index]), hard)


def sdpa_forward(self, x, caching=False, past_kv=None):
    # Retain released cached inference, including its offset causal mask.
    if caching or past_kv is not None:
        return self._released_forward(x, caching=caching, past_kv=past_kv)
    batch, length, width = x.shape
    def project(layer):
        return layer(x).view(batch,length,self.n_head,width//self.n_head).transpose(1,2)
    y = F.scaled_dot_product_attention(project(self.query), project(self.key), project(self.value),
        dropout_p=self.attn_drop.p if self.training else 0., is_causal=self.mask)
    y = y.transpose(1,2).contiguous().view(batch,length,width)
    return self.resid_drop(self.proj(y))


def enable_sdpa(model):
    """Use fused PyTorch attention without changing parameters or causal semantics."""
    from rqvae.models.rqtransformer.attentions import MultiSelfAttention
    count = 0
    for module in model.modules():
        if isinstance(module, MultiSelfAttention) and not hasattr(module, '_released_forward'):
            module._released_forward = module.forward
            module.forward = MethodType(sdpa_forward, module)
            count += 1
    return count


def conditional_update(ddp, tokenizer, optimizer, scaler, batches, temperature, max_gn=1.,
                       *, inputs_are_images=False, encoder_batch_size=16):
    total = sum(len(batch[0]) for batch in batches)
    device = next(ddp.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    weighted_loss = torch.zeros((),device=device)
    for index,(z,labels,_) in enumerate(batches):
        with (ddp.no_sync() if index+1 < len(batches) else nullcontext()):
            with torch.no_grad():
                if inputs_are_images:
                    z = torch.cat([tokenizer.encode(images.to(device, non_blocking=True))
                                   for images in z.split(encoder_batch_size)])
                targets,codes = tokenizer.quantizer.get_soft_codes(
                    z.to(device,non_blocking=True),temp=temperature,stochastic=True)
            logits = ddp(codes,model_aux=tokenizer,cond=labels.to(device,non_blocking=True),amp=True)
            loss = soft_cross_entropy(logits,targets)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite class-conditional soft CE')
            weight = len(z)/total
            scaler.scale(loss*weight).backward()
            weighted_loss += loss.detach()*weight
            del targets,codes,logits,loss
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(ddp.parameters(),max_gn)
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    success = scaler.get_scale() >= old_scale
    success_count = torch.tensor(int(success),device=device)
    dist.all_reduce(success_count)
    if success_count.item() not in (0,dist.get_world_size()):
        raise RuntimeError('Ranks disagree on AMP optimizer update')
    values = torch.stack([weighted_loss,norm.float()])
    dist.all_reduce(values)
    values /= dist.get_world_size()
    return dict(loss=values[0].item(),gradient_norm=values[1].item(),amp_scale=scaler.get_scale(),
                optimizer_updated=success,global_images=total*dist.get_world_size())
