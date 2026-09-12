#!/usr/bin/env python3
"""Read-only throughput probes for the existing corrected Church checkpoint."""
import argparse
import codecs
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from src.church_joint_geometry import make_prior, objective
from src.church_relative_noise import RelativeChurchAux
from src.church_ffhq_archived import full_training_cache
from src.rqvae_metrics import DistributedOriginalRQVAEMetrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--skip-training', action='store_true')
    parser.add_argument('--generation-batches', type=int, nargs='+', default=[32,128,256,512])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.serialization.add_safe_globals([codecs.encode])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    config = saved['config']
    model = make_prior().cuda()
    model.load_state_dict(saved['state_dict'], strict=True)
    aux = RelativeChurchAux(Path(config['stage1']), 16384, 2048, 3.,
        coeff_scales=config['coeff_scales'], sparsity_level=4, soft_target_physical=False,
        sigma_cap=config['coefficient_noise_sigma_cap'], relative_sigma=config['relative_sigma'],
        truncate=config['truncate']).cuda().eval().requires_grad_(False)
    raw = torch.load(config['cache'], map_location='cpu', weights_only=False, mmap=True)
    splits, _ = full_training_cache(raw)
    del raw
    rows = []

    def report(row):
        rows.append(row)
        print(json.dumps(row), flush=True)
        (args.output/'benchmark.json').write_text(json.dumps({'checkpoint_step': saved['step'], 'rows': rows}, indent=2)+'\n')

    for micro in (() if args.skip_training else (32, 64, 128)):
        try:
            model.train()
            times = []
            torch.cuda.reset_peak_memory_stats()
            for repeat in range(3):
                model.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                start = time.monotonic()
                for first in range(0, 128, micro):
                    torch.manual_seed(73101+first)
                    loss, _ = objective(model, aux, splits['train']['atoms'][first:first+micro].cuda().long(),
                        splits['train']['coefficients'][first:first+micro].cuda(), .05)
                    (loss*(micro/128)).backward()
                    del loss
                torch.cuda.synchronize()
                times.append(time.monotonic()-start)
            report({'kind': 'train_forward_backward', 'microbatch': micro, 'images': 128,
                'seconds': times, 'images_per_second': 128/(sum(times[1:])/2),
                'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30})
        except torch.cuda.OutOfMemoryError:
            report({'kind': 'train_forward_backward', 'microbatch': micro, 'oom': True})
        model.zero_grad(set_to_none=True)
        gc.collect(); torch.cuda.empty_cache()

    metric = DistributedOriginalRQVAEMetrics('cuda', reference_stats_path=config['fid_stats'])
    model.eval()
    for batch in args.generation_batches:
        try:
            times = []
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                for repeat in range(2):
                    torch.manual_seed(83101+repeat)
                    torch.cuda.synchronize(); start=time.monotonic()
                    atoms, ids = model.sample_compound(batch, aux, atom_top_k=250,
                        atom_top_p=1., coeff_top_p=.85, atom_temperature=1., coeff_temperature=1., amp=True)
                    for first in range(0, batch, 32):
                        images=((aux.decode_compound(atoms[first:first+32], ids[first:first+32])+1)/2).clamp(0,1)
                        metric.update(images, real=False)
                    torch.cuda.synchronize(); times.append(time.monotonic()-start)
                    del atoms, ids, images
            report({'kind': 'generation_decode_inception', 'batch': batch, 'seconds': times,
                'images_per_second': batch/times[-1], 'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30})
        except torch.cuda.OutOfMemoryError:
            report({'kind': 'generation_decode_inception', 'batch': batch, 'oom': True})
        model.init_cache(); gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
