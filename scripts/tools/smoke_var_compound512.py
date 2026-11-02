"""Exercise full-size frozen-tokenizer VAR training, recovery, and generation."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.source).resolve()))
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from omegaconf import OmegaConf
    from src.models.scratch_var import build_scratch_tokenizer, build_scratch_prior
    from src.models.compound_multiscale_var import CompoundLaserVAR

    cfg = OmegaConf.load(args.config)
    torch.set_num_threads(4)
    rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', timeout=timedelta(minutes=15))
    device = torch.device('cuda', rank)
    try:
        torch.manual_seed(1001)
        tokenizer = build_scratch_tokenizer(cfg.model, cfg.seed)
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
        tokenizer.load_state_dict(checkpoint['model'])
        del checkpoint
        tokenizer.to(device).eval().requires_grad_(False)
        if cfg.model.bottleneck == 'laser':
            tokenizer.quantize.coefficient_range_decay = None
            model = CompoundLaserVAR(tokenizer, depth=cfg.model.depth)
        else:
            model = build_scratch_prior(tokenizer, 'vq', 1001, depth=cfg.model.depth)
        model.to(device)
        wrapped = DistributedDataParallel(model, device_ids=[rank], broadcast_buffers=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(.9, .95), fused=True)
        # Real original JPEGs, using the same deterministic production transform.
        from src.data.var_images import Images, load_manifests
        manifests = load_manifests(cfg.data.manifests, 'imagenet')
        data = Images(Path(cfg.data.root) / 'train', manifests['train'], True, 0)
        rows = [data[rank * args.batch_size + i] for i in range(args.batch_size)]
        images = torch.stack([r[0] for r in rows]).to(device)
        labels = torch.tensor([r[1] for r in rows], device=device)
        losses = []
        for step in range(2):
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                codes = tokenizer.tokenize(images)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, _ = wrapped(labels, codes['inputs'], codes['atoms'], codes['coefficients'])
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
        # Test actual optimizer serialization/load without another full model.
        if rank == 0:
            path = Path(args.output).with_suffix('.optimizer.pt')
            torch.save(optimizer.state_dict(), path)
            restored = torch.load(path, map_location='cpu', weights_only=False)
            optimizer.load_state_dict(restored)
            assert all(int(v['step']) == 2 for v in optimizer.state.values())
            path.unlink()
        dist.barrier()
        model.eval()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            sampled = model.sample(torch.tensor([0, 1], device=device), seed=13)
            if cfg.model.bottleneck == 'laser':
                sampled = tokenizer.fhat_to_img(sampled)
        if sampled.shape != (2, 3, 256, 256) or not sampled.isfinite().all():
            raise ValueError('Invalid full-size class-conditional generation')
        if rank == 0:
            result = dict(passed=True, arm=cfg.model.bottleneck, world=dist.get_world_size(),
                          batch_per_rank=args.batch_size, steps=2, losses=losses,
                          gradient_norm=float(gradient), optimizer_restored=True,
                          sample_shape=list(sampled.shape),
                          parameters=sum(p.numel() for p in model.parameters()),
                          peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30)
            Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(result), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
