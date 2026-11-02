#!/usr/bin/env python3
"""Optimize only the shared joint codewords through the frozen image decoder."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts/tools')]
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import refit_laser_joint20k_depth4 as study

RUN = 'imagenet-laser-sharedjoint20k-depth4-perceptual-20261022'
BASE = Path('/dev/shm/laser-joint20k-depth4-perceptual-20261022')


class TrainingImagesAndLatents(Dataset):
    def __init__(self, base):
        from omegaconf import OmegaConf
        from rqvae.img_datasets.transforms import create_transforms
        from src.training.rq_imagenet_runtime import TarImageNet
        metadata = json.loads((base / 'output/calibration-data.json').read_text())
        self.indices = metadata['fit_indices']
        config = OmegaConf.create(dict(transforms=dict(type='imagenet256x256')))
        self.images = TarImageNet(base / 'data', 'train', create_transforms(config, split='train'), seed=261001)
        self.images.set_epoch(0)
        self.latents = [np.load(base / f'fit-rank{rank}.npy', mmap_mode='r') for rank in range(8)]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        image, _ = self.images[self.indices[position]]
        latent = torch.from_numpy(self.latents[position % 8][position // 8].copy())
        return image, latent


class SharedTrainableBook(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.dictionary = nn.Parameter(spec['dictionary'].float().clone())
        self.coefficients = nn.Parameter(spec['coefficients'][0].float().clone())
        self.register_buffer('atom_ids', spec['atom_ids'][0].long().clone())
        old = self.coefficients.detach()
        lower, upper = torch.full_like(old, -torch.inf), torch.full_like(old, torch.inf)
        layout = self.atom_ids[1:]
        same = layout[1:] == layout[:-1]
        midpoints = (old[1:-1] + old[2:]) * .5
        upper[1:-1] = torch.where(same, midpoints - 1.e-7, torch.inf)
        lower[2:] = torch.where(same, midpoints + 1.e-7, -torch.inf)
        lower[old > 0] = torch.maximum(lower[old > 0], old.new_tensor(1.e-6))
        upper[old < 0] = torch.minimum(upper[old < 0], old.new_tensor(-1.e-6))
        self.register_buffer('lower', lower)
        self.register_buffer('upper', upper)

    def forward(self, codes):
        atoms = self.atom_ids[codes]
        return (self.dictionary.T[atoms] * self.coefficients[codes, None]).sum(-2)

    @torch.no_grad()
    def project(self):
        self.dictionary.div_(self.dictionary.norm(dim=0).clamp_min(1.e-8))
        self.coefficients.copy_(torch.maximum(self.lower, torch.minimum(self.upper, self.coefficients)))
        self.coefficients[0] = 0
        assert torch.isfinite(self.dictionary).all() and torch.isfinite(self.coefficients).all()

    @torch.no_grad()
    def refresh(self, q):
        q.dictionary.copy_(self.dictionary)
        q.coefficients.copy_(self.coefficients.expand_as(q.coefficients))
        q.norms.copy_(q.dictionary.square().sum(0))
        study.assert_shared_codewords(q)


def train(args, device, run):
    from rqvae.losses.vqgan.lpips import LPIPS
    from src.models.laser_joint_tokenizer import FrozenJointLaserTokenizer
    rank, world = dist.get_rank(), dist.get_world_size()
    spec = torch.load(args.baseline, map_location='cpu', weights_only=True)
    learner = SharedTrainableBook(spec).to(device)
    q = study.build(args.baseline, device)
    aux = FrozenJointLaserTokenizer(args.evaluation / 'stage1.pt', args.baseline).to(device).eval().requires_grad_(False)
    perceptual = LPIPS().to(device).eval().requires_grad_(False)
    versions = [v._version for v in aux.state_dict().values()]
    model = DistributedDataParallel(learner, device_ids=[device.index], broadcast_buffers=False)
    optimizer = torch.optim.Adam([
        dict(params=[learner.dictionary], lr=args.lr),
        dict(params=[learner.coefficients], lr=args.lr * .1)], betas=(.9, .99))
    data = TrainingImagesAndLatents(args.base)
    sampler = DistributedSampler(data, num_replicas=world, rank=rank, shuffle=True, seed=26102220486)
    loader = DataLoader(data, batch_size=args.microbatch, sampler=sampler, num_workers=2,
                        pin_memory=True, drop_last=True, persistent_workers=True)
    selected_steps = [50, 150, 300, args.steps]
    step, epoch, started = 0, 0, time.monotonic()
    while step < args.steps:
        sampler.set_epoch(epoch)
        for images, latent in loader:
            images, latent = images.to(device), latent.to(device)
            learner.refresh(q)
            with torch.no_grad():
                codes = q.quantize(latent, chunk_size=1024)['codes']
            optimizer.zero_grad(set_to_none=True)
            z = model(codes)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                reconstructed = aux.decode(z)
                lpips = perceptual(reconstructed, images)
                pixel = (reconstructed.float() - images).abs().mean()
                mse = (z - latent).square().mean()
                loss = lpips + pixel + mse
            loss.backward()
            torch.nn.utils.clip_grad_norm_(learner.parameters(), 1.)
            optimizer.step()
            learner.project()
            step += 1
            factor = .2 + .8 * .5 * (1 + math.cos(math.pi * step / args.steps))
            optimizer.param_groups[0]['lr'] = args.lr * factor
            optimizer.param_groups[1]['lr'] = args.lr * .1 * factor
            if step % 20 == 0 or step == 1:
                values = torch.stack([loss.detach(), lpips.detach(), pixel.detach(), mse.detach()]).float()
                dist.all_reduce(values)
                values /= world
                row = dict(step=step, loss=float(values[0]), lpips=float(values[1]),
                    pixel_l1=float(values[2]), latent_mse=float(values[3]),
                    elapsed_seconds=time.monotonic() - started)
                study.progress(args, 'perceptual_codeword_training', **row)
                if rank == 0 and run:
                    run.log({f'train/{key}':value for key,value in row.items()})
            if step in selected_steps:
                learner.refresh(q)
                study.save_book(args, spec, q, f'perceptual-step{step:04d}',
                    perceptual_refinement=True, perceptual_refinement_updates=step,
                    fitting_objective='Frozen-decoder perceptual reconstruction',
                    perceptual_objective='LPIPS + pixel L1 + latent MSE; frozen CNN',
                    gradient_optimized_parameters=['shared dictionary directions', 'shared coefficient levels'])
                dist.barrier()
            if step >= args.steps:
                break
        epoch += 1
    assert versions == [v._version for v in aux.state_dict().values()]
    if rank == 0:
        study.write(args.out / 'training-verification.json', dict(passed=True, updates=step,
            encoder_decoder_weights_unchanged=True, trained_parameters=sum(p.numel() for p in learner.parameters()),
            physical_vectors_shared_across_all_depths=True, baseline_rfid=args.expected_baseline_rfid,
            objective='LPIPS + pixel L1 + latent MSE', full_training_images=len(data),
            validation_images_used_for_fitting=0, microbatch=args.microbatch, world_size=world,
            snapshots=selected_steps))
    del model, learner, optimizer, aux, perceptual, loader
    torch.cuda.empty_cache()
    dist.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('run', 'train', 'screen', 'full'))
    parser.add_argument('--base', type=Path, default=study.BASE)
    parser.add_argument('--out', type=Path, default=BASE / 'output')
    parser.add_argument('--evaluation', type=Path, default=study.EVALUATION)
    parser.add_argument('--baseline', type=Path, default=ROOT / 'outputs/imagenet-laser-sharedjoint20k-depth4-refit-20261022/joint20k-depth4-codebook.pt')
    parser.add_argument('--expected-baseline-rfid', type=float, default=4.950253483044264)
    parser.add_argument('--fit-images', type=int, default=16384)
    parser.add_argument('--held-images', type=int, default=2048)
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--microbatch', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=.0003)
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()
    args.optimization_objective = 'LPIPS + pixel L1 + latent MSE; shared codeword parameters only'
    os.environ.setdefault('TORCH_HOME', str(args.evaluation / 'torch-cache'))
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(26102220486 + int(os.environ['RANK']))
    dist.init_process_group('nccl', device_id=device)
    study.RUN = RUN
    run, succeeded = None, False
    try:
        if dist.get_rank() == 0:
            shutil.copyfile(__file__, args.out / Path(__file__).name)
            shutil.copyfile(ROOT / 'scripts/tools/refit_laser_joint20k_depth4.py', args.out / 'refit_laser_joint20k_depth4.py')
            shutil.copyfile(args.base / 'output/calibration-data.json', args.out / 'calibration-data.json')
            if args.wandb:
                import wandb
                os.environ['WANDB_API_KEY'] = Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
                os.environ['WANDB_BASE_URL'] = 'https://api.wandb.ai'
                run = wandb.init(entity='helloimlixin-rutgers', project='laser', id=RUN,
                    name='LASER shared joint20k depth4 | perceptual codeword refinement',
                    job_type='tokenizer-fitting', dir=str(BASE), resume='allow',
                    config=dict(depth=4, vocabulary=20481, atoms=8192, tokens_per_image=256,
                        baseline_rfid=args.expected_baseline_rfid, encoder_decoder_frozen=True,
                        objective='LPIPS + pixel L1 + latent MSE', train_images=args.fit_images,
                        heldout_training_selection_images=args.held_images, updates=args.steps,
                        learning_rate=args.lr, microbatch=args.microbatch, world_size=dist.get_world_size(),
                        shared_codewords_across_all_depths=True, validation_used_for_selection=False))
                study.write(args.out / 'wandb.json', dict(run_id=run.id, url=run.url))
        if args.phase in ('run', 'train'):
            train(args, device, run)
        if args.phase in ('run', 'screen'):
            cases = [('baseline', args.baseline, [1.])]
            cases += [(f'perceptual-step{step:04d}', args.out / f'perceptual-step{step:04d}-codebook.pt', [1.])
                      for step in [50, 150, 300, args.steps]]
            study.screen(args, device, run, cases=cases)
        if args.phase in ('run', 'full'):
            study.full(args, device, run)
        succeeded = True
    finally:
        if run:
            run.finish(exit_code=0 if succeeded else 1)
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
