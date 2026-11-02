"""Retain the global data cursor when changing accumulation on the same ranks."""
import math


def remap_microbatch_cursor(saved, *, world, accumulation, total_batch, batch_idx):
    old_accumulation = saved.get('accumulation_steps')
    if (saved.get('exact_global_batch') is not True or saved.get('world_size') != world
            or saved.get('total_batch_size') != total_batch or world < 1
            or not isinstance(old_accumulation, int) or old_accumulation < 1
            or not isinstance(accumulation, int) or accumulation < 1
            or not isinstance(batch_idx, int) or batch_idx < 0
            or batch_idx % old_accumulation
            or saved.get('batch_size') != math.ceil(total_batch/(world*old_accumulation))
            or total_batch < world*accumulation):
        raise ValueError('Accumulation changes require identical ranks/global batches and an optimizer boundary')
    return batch_idx//old_accumulation*accumulation
