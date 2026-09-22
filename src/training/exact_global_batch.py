"""Full-coverage distributed batches when global size is not divisible by world size."""
import math

import torch
from torch.utils.data import Sampler


class ExactGlobalBatchSampler(Sampler):
    """Partition each shuffled global batch and then each rank's microbatches.

    Every image appears once per epoch, including the smaller final batch.
    The cursor counts microbatches and may only resume at an optimizer boundary.
    """

    def __init__(self, dataset, global_batch_size, world_size, rank, accumulation=1, seed=0):
        self.size = len(dataset)
        self.global_batch_size = int(global_batch_size)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.accumulation = int(accumulation)
        self.seed = int(seed)
        self.epoch = 0
        self.start_batch = 0
        if not 0 <= self.rank < self.world_size or self.accumulation < 1:
            raise ValueError('invalid distributed layout')
        if self.size < 1 or self.global_batch_size < self.world_size * self.accumulation:
            raise ValueError('each rank needs a nonempty microbatch')
        remainder = self.size % self.global_batch_size
        if remainder and remainder < self.world_size * self.accumulation:
            raise ValueError('final batch is too small for this distributed layout')
        self.steps_per_epoch = math.ceil(self.size / self.global_batch_size)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def set_start_batch(self, batch):
        if not 0 <= batch <= self.steps_per_epoch * self.accumulation:
            raise ValueError('resume cursor outside epoch')
        if batch % self.accumulation:
            raise ValueError('resume requires an optimizer boundary')
        self.start_batch = int(batch)

    def global_count(self, microbatch):
        step = microbatch // self.accumulation
        return min(self.global_batch_size, self.size - step * self.global_batch_size)

    def backward_scale(self, local_count, microbatch):
        # The trainer already divided its mean loss by accumulation; undo that
        # division and account for DDP's unweighted rank-average reduction.
        return self.accumulation * self.world_size * local_count / self.global_count(microbatch)

    def __len__(self):
        return self.steps_per_epoch * self.accumulation - self.start_batch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.size, generator=generator).tolist()
        for step in range(self.start_batch // self.accumulation, self.steps_per_epoch):
            global_indices = order[step * self.global_batch_size:(step + 1) * self.global_batch_size]
            local_indices = global_indices[self.rank::self.world_size]
            for micro in range(self.accumulation):
                yield local_indices[micro::self.accumulation]
