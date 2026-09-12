"""Physical coefficient targets, reproducible image views, and held-out stopping."""
from dataclasses import asdict, dataclass
import io
import math

import lmdb
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import functional as TF

from src.training.rqtransformer import CompoundLaserRQTransformer
from src.church_ffhq_recipe import recipe_config
from src.coefficient_history_training import EpochStream


@torch.no_grad()
def physical_targets(aux, atoms, coefficients, mode, sigma, stochastic=True):
    if not aux.soft_target_physical:
        raise ValueError('Calibrated targets require physical coefficient distances')
    if mode not in {'soft', 'hard'} or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError('Expected soft/hard mode and a finite positive physical sigma')
    ids, probabilities = aux.compound_coeff_ids(
        coefficients.float() / aux.coeff_scales,
        temp=2 * sigma ** 2, hard=mode == 'hard',
        stochastic=stochastic and mode == 'soft',
    )
    return atoms.long() * aux.coeff_vocab_size + ids, probabilities


def calibrated_prior(dropout=.15):
    config = recipe_config('balanced')
    config.body.block.resid_pdrop = float(dropout)
    config.head.block.resid_pdrop = float(dropout)
    return CompoundLaserRQTransformer(config, 16384, 2048,
        micro_transformer_layers=2, depth_specific_coeff_heads=True,
        pair_autoregressive=True)


def augmented_view(image, seed, index, epoch):
    """Augment pixels before encoding; latent horizontal flips are not valid."""
    rng = torch.Generator().manual_seed((seed + 1000003 * epoch + 9176 * index) % (2 ** 63 - 1))
    image = TF.resize(image, 256)
    width, height = image.size
    top = int(torch.randint(height - 256 + 1, (), generator=rng))
    left = int(torch.randint(width - 256 + 1, (), generator=rng))
    image = TF.crop(image, top, left, 256, 256)
    if float(torch.rand((), generator=rng)) < .5:
        image = TF.hflip(image)
    return TF.to_tensor(image).mul_(2).sub_(1)


class AugmentedChurch(Dataset):
    def __init__(self, path, keys, seed):
        self.path, self.keys, self.seed = str(path), list(keys), int(seed)
        self.environment = None

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        index, epoch = item
        if self.environment is None:
            self.environment = lmdb.open(self.path, readonly=True, lock=False, readahead=False)
        with self.environment.begin() as transaction:
            data = transaction.get(self.keys[index].encode('ascii'))
        if data is None:
            raise KeyError(self.keys[index])
        with Image.open(io.BytesIO(data)) as image:
            view = augmented_view(image.convert('RGB'), self.seed, index, epoch)
        return view, index


class PendingEpochBatches(Sampler):
    """Prefetch from a COPY; only the trainer commits consumed data positions."""
    def __init__(self, stream, batch_size):
        self.stream, self.batch_size = stream, batch_size

    def __iter__(self):
        pending = EpochStream(self.stream.size, 0)
        pending.load_state_dict(self.stream.state_dict())
        while True:
            indices, _, end = pending.next(self.batch_size)
            yield [(int(i), pending.epoch) for i in indices]
            if end:
                break

    def __len__(self):
        remaining = self.stream.size - self.stream.position
        if remaining == 0:
            remaining = self.stream.size
        return math.ceil(remaining / self.batch_size)


@dataclass
class HeldoutStop:
    patience: int = 3
    min_delta: float = .01
    minimum_epoch: int = 8
    best: float | None = None
    best_epoch: int | None = None
    bad_checks: int = 0

    def observe(self, score, epoch):
        if not math.isfinite(score):
            raise FloatingPointError('Nonfinite held-out score')
        improved = self.best is None or score < self.best - self.min_delta
        if improved:
            self.best, self.best_epoch, self.bad_checks = float(score), int(epoch), 0
        else:
            self.bad_checks += 1
        stop = epoch >= self.minimum_epoch and self.bad_checks >= self.patience
        return improved, stop

    def state_dict(self):
        return asdict(self)


def optimizer_groups(model, weight_decay):
    decay, other = [], []
    for name, parameter in model.named_parameters():
        if parameter.ndim < 2 or name.startswith('pos_emb') or name.endswith('_pos'):
            other.append(parameter)
        else:
            decay.append(parameter)
    return [{'params': decay, 'weight_decay': weight_decay},
            {'params': other, 'weight_decay': 0.}]
