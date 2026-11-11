"""Verify four-GPU NCCL and the actual BF16 attention kernel on this host."""
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist

torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
dist.init_process_group('nccl')
assert dist.get_world_size() == 4
value = torch.tensor(float(dist.get_rank()+1), device='cuda')
dist.all_reduce(value)
assert value.item() == 10
q = torch.randn(8, 24, 64, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
    y = torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
    y.float().square().mean().backward()
assert torch.isfinite(q.grad).all()
if dist.get_rank() == 0:
    report = dict(passed=True, world_size=4, nccl_sum=10,
                  bf16_flash_attention_forward_backward=True)
    Path('/tmp/laser-beam16-a100-20261105/distributed-verification.json').write_text(
        json.dumps(report, indent=2)+'\n')
    print(report, flush=True)
dist.destroy_process_group()
