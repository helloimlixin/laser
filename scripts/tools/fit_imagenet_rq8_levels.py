#!/usr/bin/env python3
"""Fit eight shared coefficient values against the actual four-step RQ error.

Only the four positive magnitudes are calibrated. The checkpoint, dictionary,
encoder, decoder, vocabulary size, and greedy RQ rule are unchanged.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT/'outputs/church-rq-baseline-scratch-20260912/upstream-source'
sys.path[:0] = [str(ROOT), str(UPSTREAM)]
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from rqvae.img_datasets.transforms import create_transforms
from src.imagenet_scaled_stage2 import ManifestImages, load_imagenet_config
from src.original_rq_training import atomic_json, file_sha256, state_sha256
from src.scaled_atom_rq import FrozenSparseBackbone, ScaledAtomRQ, orthogonal_matching_pursuit


def signed(magnitudes):
    return torch.cat([-magnitudes.flip(0), magnitudes])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=ROOT/'outputs/imagenet-scaled-rq-stage2-20260913')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--iterations', type=int, default=16)
    p.add_argument('--objectives', nargs='+', choices=['final', 'prefix'], default=['final', 'prefix'])
    p.add_argument('--decoder-steps', type=int, default=0)
    args = p.parse_args()
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.set_num_threads(8)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group('nccl', timeout=timedelta(hours=1))
    out = args.output.resolve()
    if rank == 0: out.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    checkpoint = args.source/'assets/best_rfid_slot1_model.pt'
    backbone = FrozenSparseBackbone(checkpoint).to(device).eval()
    before = state_sha256(backbone)
    old = torch.load(args.source/'scaled-atom-codebook.pt', weights_only=True)
    torch.testing.assert_close(old['dictionary'], backbone.dictionary.cpu(), atol=0, rtol=0)
    fit_ids = old['fit_indices']
    manifest = json.loads((args.source/'train-manifest.json').read_text())
    dataset = ManifestImages('/workspace/Projects/data/imagenet2012/train', manifest,
        create_transforms(load_imagenet_config(UPSTREAM).dataset, split='train'), seed=421,
        indices=fit_ids[rank::world])
    loader = DataLoader(dataset, batch_size=16, num_workers=4, pin_memory=True)
    zs = torch.cat([backbone.encode(images.to(device)) for images, _, _ in loader])
    initial = old['levels']['8'][4:].to(device)
    candidates = {'rq8_control': signed(initial).cpu()}
    history = []
    started = time.time()
    # With assignments fixed, each reconstruction is linear in four shared
    # positive magnitudes. Reassign greedily after each least-squares update.
    for objective in args.objectives:
        centers = initial.clone()
        for iteration in range(args.iterations):
            quantizer = ScaledAtomRQ(backbone.dictionary, signed(centers)).to(device)
            lhs = torch.zeros(4, 4, device=device, dtype=torch.float64)
            rhs = torch.zeros(4, device=device, dtype=torch.float64)
            error = torch.zeros(2, device=device, dtype=torch.float64)
            for z in zs.split(16):
                result = quantizer.quantize(z)
                codes = result['codes'].reshape(-1, 4)
                target = z.reshape(-1, 256)
                packed = (codes-1).clamp_min(0)
                bins = packed % 8
                magnitude_ids = torch.where(bins < 4, 3-bins, bins-4)
                signs = torch.where(bins < 4, -1., 1.) * (codes != 0)
                atoms = backbone.dictionary.T[packed//8] * signs[..., None]
                design = z.new_zeros(len(target), 4, 256)
                rows = torch.arange(len(target), device=device)
                for depth in range(4):
                    design[rows, magnitude_ids[:, depth]] += atoms[:, depth]
                    if objective == 'prefix' or depth == 3:
                        lhs += torch.einsum('nac,nbc->ab', design, design).double()
                        rhs += torch.einsum('nac,nc->a', design, target).double()
                error[0] += (result['quantized']-z).square().sum()
                error[1] += z.numel()
            for value in [lhs, rhs, error]: dist.all_reduce(value)
            update = torch.linalg.solve(lhs, rhs).float()
            assert (update > 0).all() and (update[1:] > update[:-1]).all()
            row = dict(objective=objective, iteration=iteration,
                latent_mse=float(error[0]/error[1]), levels=centers.cpu().tolist(),
                next_levels=update.cpu().tolist(), elapsed_seconds=time.time()-started)
            history.append(row)
            if rank == 0:
                print(json.dumps(row), flush=True)
                atomic_json(out/'fit-history.json', history)
            centers = update
            if iteration+1 in [4, 8, 16, 32, 64, 128, args.iterations]:
                candidates[f'rq8_{objective}_{iteration+1}'] = signed(centers).cpu()
    for scale in [.85, .925, 1.075]:
        candidates[f'rq8_scale_{scale}'] = signed(initial*scale).cpu()
    if args.decoder_steps:
        # The frozen OMP reconstruction is the teacher. Gradients only adjust
        # four physical coefficient magnitudes, never any stage-1 parameters.
        gram = backbone.dictionary.T @ backbone.dictionary
        teacher = torch.cat([orthogonal_matching_pursuit(z, backbone.dictionary, gram)['quantized']
                             for z in zs.split(16)])
        log_centers = torch.nn.Parameter(candidates[f'rq8_final_{args.iterations}'][4:].to(device).log())
        optimizer = torch.optim.Adam([log_centers], lr=.003)
        with torch.enable_grad():
            for step in range(args.decoder_steps):
                ids = torch.arange(step*4, step*4+4, device=device) % len(zs)
                z = zs[ids]
                with torch.no_grad():
                    code = ScaledAtomRQ(backbone.dictionary, signed(log_centers.exp())).to(device).quantize(z)['codes']
                    packed = (code-1).clamp_min(0)
                    bins = packed % 8
                    magnitude_ids = torch.where(bins < 4, 3-bins, bins-4)
                    signs = torch.where(bins < 4, -1., 1.) * (code != 0)
                    atoms = backbone.dictionary.T[packed//8] * signs[..., None]
                    target = backbone.decode(teacher[ids]).detach()
                reconstructed = (atoms * log_centers.exp()[magnitude_ids][..., None]).sum(-2)
                decoded = backbone.decode(reconstructed)
                loss = (decoded-target).abs().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                dist.all_reduce(log_centers.grad)
                log_centers.grad.div_(world)
                optimizer.step()
                with torch.no_grad():
                    values = log_centers.exp()
                    assert (values[1:] > values[:-1]).all()
                    if (step+1) % 25 == 0:
                        dist.all_reduce(loss)
                        if rank == 0:
                            print(json.dumps(dict(objective='decoded_omp_teacher_l1', step=step+1,
                                loss=float(loss/world), levels=values.cpu().tolist())), flush=True)
                    if (step+1) % 50 == 0 or step+1 == args.decoder_steps:
                        candidates[f'rq8_decoder_{step+1}'] = signed(values).cpu()
    assert state_sha256(backbone) == before
    if rank == 0:
        torch.save(dict(dictionary=backbone.dictionary.cpu(), candidates=candidates,
            checkpoint_sha256=file_sha256(checkpoint), fit_indices=fit_ids,
            objective='Alternating greedy RQ assignment and shared-magnitude least squares; training images only'),
            out/'candidate-codebooks.pt')
        atomic_json(out/'status.json', dict(phase='complete', candidates=list(candidates),
            fit_images=len(fit_ids), frozen_backbone_unchanged=True))
    dist.destroy_process_group()


if __name__ == '__main__': main()
