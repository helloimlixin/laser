#!/usr/bin/env python3
"""Re-evaluate the surviving epoch-five LASER baseline on shared upstream RQ."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/workspace/Projects/laser')
PARENT = Path('/tmp/laser-rq-causal-ar-imagenet-20261015')
BASE = Path('/tmp/laser-historical-recovery-20261015')
OUT = ROOT/'outputs/imagenet-laser-historical-recovery-20261015'
sys.path[:0] = [str(BASE/'source'), str(BASE), str(ROOT), str(PARENT/'upstream')]
os.environ['RQ_UPSTREAM_ROOT'] = str(PARENT/'upstream')
os.environ['TORCH_HOME'] = str(PARENT/'torch-cache')

import torch
import torch.distributed as dist
from rqvae.utils.config import load_config, augment_arch_defaults
from src.models.laser_historical_rq_adapter import build_historical_pair_model
from src.training.rq_imagenet_recipes import LaserRecipe
from src.training.rq_imagenet_runtime import TarImageNet, write_json
from src.training import rq_imagenet_shared as shared
from rqvae.img_datasets.transforms import create_transforms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sampler', choices=['matched', 'historical'], required=True)
    args = parser.parse_args()
    rank, world, local = (int(os.environ[k]) for k in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK'])
    assert world == 4
    device = torch.device('cuda', local)
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    dist.init_process_group('nccl', device_id=device)
    local_out, durable = BASE/args.sampler, OUT/args.sampler
    local_out.mkdir(parents=True, exist_ok=True)
    durable.mkdir(parents=True, exist_ok=True)
    source = BASE/'best-epoch5.pt'
    saved = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
    assert len(saved['state_dict']) == 870
    old = saved['config']
    config = load_config(PARENT/'inputs/stage2-training-config.yaml')
    arch = augment_arch_defaults(config.arch)
    arch.vocab_size = 16384
    with torch.device('meta'):
        model = build_historical_pair_model(arch, num_atoms=16384, coefficient_bins=2048)
    model.load_state_dict(saved['state_dict'], strict=True, assign=True)
    assert sum(p.numel() for p in model.parameters()) == 1466384384
    if rank == 0:
        stage1_sha = shared.sha256(PARENT/'inputs/stage1-model.pt')
        reference_sha = shared.sha256(PARENT/'inputs/imagenet_256_train.npz')
        historical_stage1 = json.loads((ROOT/'outputs/imagenet-rfid421-orthogonal-scratch-20260926/orthogonal-views.meta.json').read_text())['stage1_checkpoint_sha256']
        assert stage1_sha == historical_stage1 == 'dd28db9306d526bdc9fbf8016403e106c4f317fd8f5c04792f0953e53310fdab'
        assert reference_sha == '3f9c92d15755e76ec312964a819e3e19b9cf3aadc618a7ae99f5e8aa96501260'
        hasher = hashlib.sha256()
        for name, value in saved['state_dict'].items():
            assert torch.isfinite(value).all(), name
            hasher.update(name.encode())
            hasher.update(value.contiguous().numpy().tobytes())
        checkpoint_proof = dict(passed=True, checkpoint=str(source),
            checkpoint_bytes=source.stat().st_size, model_state_sha256=hasher.hexdigest(),
            model_tensors=870, model_parameters=1466384384, epoch=saved['epoch'],
            global_step=saved['global_step'], coefficient_scales=list(old['coeff_scales']),
            stage1_checkpoint_sha256=stage1_sha,reference_sha256=reference_sha,
            shared_upstream_forward=True, global_coefficient_history_restored=True,
            weights_unmodified=True, training_updates_added=0)
        write_json(OUT/'checkpoint-verification.json', checkpoint_proof)
    del saved
    model.eval().requires_grad_(False).to(device)
    profile = LaserRecipe(PARENT)
    profile.recipe.tokenizer.coefficient_scales = list(old['coeff_scales'])
    aux, _ = profile.tokenizer(device)
    profile.adapt(model, aux)
    frozen_digest = profile.digest(aux)
    if args.sampler == 'historical':
        profile.recipe.sampling.atom_top_k = 2048
        profile.recipe.sampling.atom_top_p = 1.
        profile.recipe.sampling.coefficient_top_p = .85
    validation = TarImageNet(PARENT/'data', 'val', create_transforms(config.dataset, split='val'), seed=261001)
    pairs = [validation[i] for i in range(rank*2, rank*2+2)]
    images = torch.stack([p[0] for p in pairs]).to(device)
    labels = torch.tensor([p[1] for p in pairs], device=device)
    with torch.inference_mode():
        _, truth = profile.targets(aux, images)
        expected = model(truth, aux, labels, amp=True)
        buffer = torch.zeros_like(truth)
        buffer[..., 1::2] = 16384 + 1024
        model.init_cache()
        divergences = []
        for w in range(4):
            for event in range(8):
                actual = model.cached_forward(buffer[:, :1], aux, labels, amp=True, sample_loc=(0, w, event))
                key = 'atom_logits' if event % 2 == 0 else 'coeff_logits'
                actual = actual[:, :16384] if event % 2 == 0 else actual[:, 16384:]
                reference = expected[key][:, 0, w, event//2]
                assert torch.equal(torch.isfinite(actual), torch.isfinite(reference))
                a, b = actual.float().log_softmax(-1), reference.float().log_softmax(-1)
                divergences.append(torch.where(b.isfinite(), b.exp()*(b-a), 0.).sum(-1).mean())
                buffer[:, 0, w, event] = truth[:, 0, w, event]
        maximum = torch.stack(divergences).max()
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        assert maximum < 1.e-4, maximum
        if rank == 0:
            write_json(durable/'sampling-verification.json', dict(passed=True,
                maximum_mean_kl=float(maximum), ranks=4, events_per_rank=32,
                fp16_cache_matches_full_teacher_forcing=True))
        model.init_cache()
    del images, labels, expected, truth, buffer, actual, reference, a, b, divergences
    torch.cuda.empty_cache()
    run_id = 'imagenet-laser-historical-recovery-'+args.sampler+'-20261015'
    run = None
    if rank == 0:
        import wandb
        os.environ['WANDB_API_KEY'] = Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
        run = wandb.init(entity='helloimlixin-rutgers', project='laser', id=run_id,
            name='LASER historical epoch5 recovery | '+args.sampler+' sampler', dir=str(BASE),
            config=dict(experiment_group='laser_historical_recovery_20261015',
                historical_baseline_recovery=True, source_run='imagenet-rfid421-orthogonal-scratch-20260926',
                source_step=3175, fresh_stage2=False, training_updates_added=0,
                auxiliary_loss_terms=0, omp_trajectory_banks=0, shared_upstream_forward=True,
                coefficient_scales=list(old['coeff_scales']), sampler_variant=args.sampler,
                generated_images=50000, reference_images=1281167, reference_split='train'))
        run.define_metric('train/epoch')
        run.define_metric('eval/fid', step_metric='train/epoch', overwrite=True)
        write_json(durable/'wandb.json',dict(id=run.id,url=run.url))
    shared.BASE, shared.OUT, shared.PROFILE, shared.RUN = PARENT, local_out, profile, run_id
    shared.original_metrics.INCEPTION_PATH = PARENT/'upstream/rqvae/metrics/inception.py'
    import rqvae.utils.utils as utils
    original_top_k = utils.top_k_logits
    utils.top_k_logits = lambda logits, k: logits if k == logits.shape[-1] else original_top_k(logits, k)
    started = time.time()
    result = shared.generation(model, aux, device, rank, world, 5, None, batch=128)
    assert profile.digest(aux) == frozen_digest
    if rank == 0:
        settings = profile.recipe.sampling
        result.update(sampler_variant=args.sampler, generated_images=50000,
            reference_images=1281167, reference_split='train', source_epoch=5, source_step=3175,
            reference_sha256=reference_sha,
            training_updates_added=0, historical_baseline_recovery=True,
            same_weights_for_both_sampler_variants=True,
            sampler=dict(atom_temperature=float(settings.atom_temperature),atom_top_k=int(settings.atom_top_k),
                atom_top_p=float(settings.atom_top_p),coefficient_temperature=float(settings.coefficient_temperature),
                coefficient_top_p=float(settings.coefficient_top_p)),elapsed_seconds=time.time()-started)
        write_json(durable/'fid50k.json',result)
        run.log({'train/epoch':5,'eval/fid':result['fid'],'eval/inception_score':result['inception_score']})
        run.summary['evaluation/generated_images']=50000
        run.summary['evaluation/reference_images']=1281167
        run.save(str(durable/'fid50k.json'),base_path=str(OUT),policy='now')
        run.finish()
        print(json.dumps(result),flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__=='__main__':
    main()
