"""Discarded real-image throughput trials for the fresh native joint16k run."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--microbatch', type=int, required=True)
    parser.add_argument('--attention', choices=['native', 'spatial_sdpa'], default='spatial_sdpa')
    parser.add_argument('--updates', type=int, default=5)
    args = parser.parse_args()
    base = args.base
    os.environ['RQ_UPSTREAM_ROOT'] = str(base / 'upstream')
    os.environ['TORCH_HOME'] = str(base / 'torch-cache')
    os.environ['LASER_TRAIN_SDPA'] = '1' if args.attention == 'spatial_sdpa' else '0'
    sys.path[:0] = [str(base / 'source'), str(base / 'upstream'), str(base)]
    import torch
    from runtime.precision import install_bf16_training
    from runtime.training_acceleration import install
    install_bf16_training(); install()
    from src.training import rq_imagenet_shared as shared
    from src.training.laser_joint_recipe import JointLaserRecipe
    from src.training.rq_imagenet_runtime import TarImageNet, write_json
    from rqvae.img_datasets.transforms import create_transforms
    from runtime.amp_replay import recoverable_amp_step
    device = torch.device('cuda', args.gpu)
    torch.cuda.set_device(device); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('highest')
    profile = JointLaserRecipe(base)
    shared.BASE, shared.PROFILE = base, profile
    shared.seed_all(261001)
    report = dict(passed=False, gpu=args.gpu, microbatch=args.microbatch,
                  attention=args.attention, weights_discarded=True,
                  includes_ddp=False, includes_jpeg_decode=False)
    try:
        aux, _ = profile.tokenizer(device)
        before = profile.digest(aux)
        model, optimizer, config = shared.fresh_model(device)
        report['parameter_count'] = sum(p.numel() for p in model.parameters())
        dataset = TarImageNet(base / 'data', 'train', create_transforms(config.dataset, split='train'))
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(261001))[:512]
        rows = [dataset[int(index)] for index in indices]
        images = torch.stack([r[0] for r in rows]).pin_memory()
        labels = torch.tensor([r[1] for r in rows]).pin_memory()
        accumulation = 512 // args.microbatch
        assert accumulation * args.microbatch == 512
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        model.train(); optimizer.zero_grad(set_to_none=True)
        timings, losses, norms = [], [], []
        torch.cuda.reset_peak_memory_stats(device)
        for update in range(args.updates):
            torch.cuda.synchronize(device); start = time.monotonic()
            def backward():
                value = 0.
                for first in range(0, 512, args.microbatch):
                    loss = shared.training_backward(profile, model, aux,
                        images[first:first+args.microbatch].to(device, non_blocking=True),
                        labels[first:first+args.microbatch].to(device, non_blocking=True),
                        scaler, 1. / accumulation, amp=True)
                    value += float(loss) / accumulation
                return value
            loss = backward()
            norm, retries = recoverable_amp_step(model.parameters(), optimizer, scaler,
                max_norm=1., replay_backward=backward)
            assert retries == 0
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            timings.append(time.monotonic()-start); losses.append(loss); norms.append(float(norm))
            print(json.dumps(dict(gpu=args.gpu,update=update+1,seconds=timings[-1],loss=loss,
                                  peak_gib=torch.cuda.max_memory_allocated(device)/1024**3)),flush=True)
        assert profile.digest(aux) == before
        assert len(optimizer.state) == len(list(model.parameters()))
        assert all(int(state['step']) == args.updates for state in optimizer.state.values())
        model.eval()
        with torch.no_grad(), torch.random.fork_rng(devices=[device.index]):
            torch.manual_seed(93017); torch.cuda.manual_seed(93017)
            codes = profile.sample(model, aux, 2, torch.tensor([933,849],device=device))
            pixels = profile.decode(aux, codes)
            assert codes.shape == (2,8,8,4) and codes.min() >= 0 and codes.max() < 16384
            assert torch.isfinite(pixels).all()
        report.update(passed=True,updates=args.updates,seconds=timings,losses=losses,gradient_norms=norms,
            images_per_second_per_gpu=512./statistics.median(timings[2:]),
            peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3,
            optimizer_state_count=len(optimizer.state),frozen_tokenizer_unchanged=True,
            native_sampling_finite=True,encoder_precision='fp32',training_precision='bf16')
    except torch.OutOfMemoryError as error:
        report.update(reason='CUDA out of memory',error=str(error))
    except Exception as error:
        report.update(reason='error',error=str(error),traceback=traceback.format_exc())
    write_json(base / f'benchmark-gpu{args.gpu}.json',report)
    print(json.dumps(report),flush=True)
    if report.get('reason') == 'error': raise RuntimeError(report['error'])


if __name__ == '__main__': main()
