"""Distributed execution helpers; the corrected Church objective is unchanged."""
from contextlib import nullcontext
import torch
import torch.distributed as dist
from src.church_joint_geometry import objective


class JointObjective(torch.nn.Module):
    """Keep every candidate coefficient branch inside DDP's forward graph."""
    def __init__(self, prior, aux):
        super().__init__()
        self.prior = prior
        self.aux = aux

    def forward(self, atoms, physical, geometry_weight):
        return objective(self.prior, self.aux, atoms, physical, geometry_weight)


def local_chunks(indices, microbatch, rank, world):
    if len(indices) % (microbatch * world):
        raise ValueError('The effective batch must divide into equal per-rank microbatches')
    chunks = list(enumerate(indices.split(microbatch)))
    per_rank = len(chunks) // world
    return chunks[rank*per_rank:(rank+1)*per_rank]


def backward_batch(wrapped, indices, data, geometry_weight, step, seed, microbatch, rank, world):
    chunks = local_chunks(indices, microbatch, rank, world)
    device = next(wrapped.parameters()).device
    totals = {}
    for local_index, (global_micro, chunk) in enumerate(chunks):
        # Preserve the old stream for microbatch32 even when GPUs split work.
        torch.manual_seed(seed + step*10000 + global_micro)
        sync = wrapped.no_sync() if world > 1 and local_index+1 < len(chunks) else nullcontext()
        with sync:
            loss, metrics = wrapped(data['atoms'][chunk].to(device).long(), data['coefficients'][chunk].to(device), geometry_weight)
            # DDP averages gradients across ranks; local loss must average over
            # the local portion, so the final gradient averages all 256 images.
            (loss * (len(chunk)*world/len(indices))).backward()
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.) + value*len(chunk)/len(indices)
        del loss
    keys = sorted(totals)
    values = torch.tensor([totals[k] for k in keys], device=device, dtype=torch.float64)
    if world > 1:
        dist.all_reduce(values)
    return dict(zip(keys, values.cpu().tolist()))


def generation_range(count, rank, world):
    return count*rank//world, count*(rank+1)//world
