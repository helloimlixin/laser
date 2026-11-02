"""Matched, class-balanced FID screening for recovered sign/magnitude priors."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--teacher', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=2000)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--wait-for-trial', action='store_true')
    args = p.parse_args()
    if args.samples < 1000 or args.samples % 1000:
        p.error('sample count must be a positive multiple of 1000')
    sys.path[:0] = [str(args.source), str(args.source/'runtime')]
    import torch
    import torch.distributed as dist
    from torchvision.utils import save_image
    from src.models.sign_magnitude_prior import SignMagnitudeRQTransformer
    from src.training.orthogonal_gmm_runtime import OrthogonalGMMAux, model_config
    from src.training.imagenet_compound_runtime import SCALES
    from src.rqvae_metrics import DistributedOriginalRQVAEMetrics
    torch.set_num_threads(8)
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    if world > 1:
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        dist.init_process_group('nccl', device_id=torch.device('cuda', int(os.environ['LOCAL_RANK'])))
    if args.samples % world:
        raise ValueError('sample count must divide evenly across workers')
    device = torch.device('cuda', torch.cuda.current_device())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    if args.wait_for_trial:
        while not (args.checkpoint.parent/'result.json').exists():
            print('Waiting for completed repair checkpoint', flush=True)
            time.sleep(10)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    model = SignMagnitudeRQTransformer(model_config(), 16384).cuda().eval()
    source_config = saved.get('config')
    aux = OrthogonalGMMAux(args.teacher, 16384, 512, 3., coeff_scales=SCALES,
                          sparsity_level=4).cuda().eval()
    if source_config and any(source_config.get(k) for k in ('geometric_atom_head', 'cluster_atom_head', 'projected_atom_head')):
        path = Path(__file__).resolve().parents[2]/'src/models/dictionary_atom_readout.py'
        spec = importlib.util.spec_from_file_location('dictionary_atom_head_policy', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        install = (module.install_clustered_atom_readout if source_config.get('cluster_atom_head') else
                   module.install_projected_dictionary_readout if source_config.get('projected_atom_head') else
                   module.install_dictionary_atom_readout)
        install(model, aux.dictionary)
    model.load_state_dict(saved['state_dict'], strict=True)
    if source_config and source_config.get('innovation_depth_context'):
        path = Path(__file__).resolve().parents[2]/'src/training/innovation_depth_context.py'
        spec = importlib.util.spec_from_file_location('innovation_depth_context_policy', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.install_innovation_depth_context(model)
    if source_config and source_config.get('atom_only_support_context'):
        path = Path(__file__).resolve().parents[2]/'src/training/atom_only_support_context.py'
        spec = importlib.util.spec_from_file_location('atom_only_support_policy', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.install_atom_only_support_context(model)
    del saved
    metrics = DistributedOriginalRQVAEMetrics(device, compute_inception_score=True,
                                             reference_stats_path=args.reference)
    torch.manual_seed(261029 + rank)
    started = time.time()
    saturated, pixel_count = 0, 0
    with torch.no_grad():
        local_samples = args.samples // world
        for first in range(0, local_samples, args.batch_size):
            count = min(args.batch_size, local_samples-first)
            labels = (torch.arange(first, first+count, device=device)*world+rank) % 1000
            atoms, gamma = model.sample_compound(count, aux, cond=labels,
                atom_temperature=.85, atom_top_k=16384, atom_top_p=.9,
                coeff_temperature=1., coeff_top_k=0, coeff_top_p=1., amp=True)
            preview = []
            for a, g in zip(atoms.split(16), gamma.split(16)):
                images = aux.decode_orthogonal(a, g)
                saturated += int((images.abs()>.99).sum())
                pixel_count += images.numel()
                pixels = ((images+1)*.5).clamp(0, 1)
                metrics.update(pixels, real=False)
                if first == 0 and rank == 0:
                    preview.append(pixels.cpu())
            if first == 0 and rank == 0:
                save_image(torch.cat(preview)[:64], args.output/'samples.png', nrow=8)
            print(json.dumps(dict(rank=rank, generated=first+count, total=local_samples,
                                  seconds=time.time()-started)), flush=True)
        fid, score, score_std = metrics.compute()
    counts = torch.tensor([saturated, pixel_count], device=device, dtype=torch.float64)
    if world > 1:
        dist.all_reduce(counts)
    result = dict(fid=fid, inception_score=score, inception_score_std=score_std,
        generated_samples=args.samples, samples_per_class=args.samples//1000,
        seed=261029, world_size=world, saturation_fraction=float(counts[0]/counts[1]),
        checkpoint=str(args.checkpoint), reference=str(args.reference),
        backend='original-rqvae', source_config=source_config,
        official_50000_sample_fid=(args.samples == 50000),
        interpretation='Compare only with matched sample-count/seed/settings screens; sample-count-biased relative to 50k FID.',
        seconds=time.time()-started)
    if rank == 0:
        (args.output/'result.json').write_text(json.dumps(result, indent=2, default=str))
        print(json.dumps(result, default=str), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
