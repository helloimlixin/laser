"""Compare scalar LASER sampling recipes using identical weights and captions."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time

from omegaconf import OmegaConf
import torch
import torch.distributed as dist

from src.training.cc3m_text import build_model, configure_performance, generate
from src.training.cc3m_compound import load_cache, make_aux, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    options = OmegaConf.to_container(OmegaConf.load(args.config).options, resolve=True)
    torch.manual_seed(options['seed'])
    validation = load_cache(options['validation_cache'])
    aux = make_aux(options, options['coeff_scales'], device)
    model = build_model(options).to(device).eval()
    configure_performance(model, options)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=True)
    model.load_state_dict(checkpoint['model'], strict=True)
    step, source_metrics = checkpoint['global_step'], checkpoint['metrics']
    del checkpoint
    recipes = [
        ('current_rq_sampler', {}),
        ('imagenet_physical_pair_sampler', dict(atom_temperature=.9, atom_top_k=0,
            atom_top_p=.9, coeff_temperature=1., coeff_top_k=0, coeff_top_p=.85)),
        ('previous_cc3m_physical_pair_sampler', dict(atom_temperature=1., atom_top_k=16384,
            atom_top_p=.7, coeff_temperature=1., coeff_top_k=0, coeff_top_p=.85)),
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, settings in recipes:
        trial = dict(options, **settings)
        torch.manual_seed(options['eval_seed'] + rank)
        started = time.monotonic()
        metrics = evaluate(model, aux, validation, trial, device, None, step, generator=generate)
        if name == 'current_rq_sampler':
            assert abs(metrics['fid'] - source_metrics['fid']) < 1e-6
            assert abs(metrics['clip_score'] - source_metrics['clip_score']) < 1e-8
        row = dict(name=name, step=step, sampling={k:trial[k] for k in (
            'atom_temperature','atom_top_k','atom_top_p','coeff_temperature','coeff_top_k','coeff_top_p')},
            **metrics, elapsed_seconds=time.monotonic()-started)
        rows.append(row)
        if rank == 0:
            (args.output / (name + '.json')).write_text(json.dumps(row, indent=2))
            print(json.dumps(row), flush=True)
    if rank == 0:
        (args.output / 'comparison.json').write_text(json.dumps(rows, indent=2))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
