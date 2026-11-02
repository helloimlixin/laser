#!/usr/bin/env python3
"""One full-population, fixed-weight sampler test for the compact joint model."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/workspace/Projects/laser')
BASE = Path('/tmp/laser-rq-native-joint32k-imagenet-20261016')
OUT = ROOT/'outputs/imagenet-laser-joint32k-epoch2-topk512-fid50k-20261016'
RUN = OUT.name
os.environ['RQ_UPSTREAM_ROOT'] = str(BASE/'upstream')
os.environ['TORCH_HOME'] = str(BASE/'torch-cache')
sys.path[:0] = [str(BASE/'source'), str(BASE/'upstream'), str(BASE/'runtime'), str(ROOT)]

import torch
import torch.distributed as dist
from src import rq_upstream
from src.training.laser_joint_recipe import JointLaserRecipe
from src.training.rq_imagenet_runtime import TarImageNet, write_json
from src.training import rq_imagenet_shared as shared
from rqvae.img_datasets.transforms import create_transforms
from rqvae.utils.config import load_config


def main():
    rank, world, local = (int(os.environ[k]) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    assert world == 4
    torch.set_num_threads(4)
    device = torch.device('cuda', local)
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', device_id=device)
    plan = json.loads((OUT/'plan.json').read_text())
    checkpoint = Path(plan['local_checkpoint'])
    profile = JointLaserRecipe(BASE)
    saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    profile.verify_resume(saved)
    assert saved['global_step'] == 1252 and saved['next_epoch'] == 2 and saved['next_microbatch'] == 0
    assert all(int(state['step']) == 1252 for state in saved['optimizer']['state'].values())
    assert saved['scheduler']['after']['T_max'] == 62600
    assert len(saved['rng_state_by_rank']) == 4
    from omegaconf import OmegaConf
    arch = OmegaConf.create(saved['config']['arch'])
    with torch.device('meta'):
        model = profile.model(arch)
    model.load_state_dict(saved['state_dict'], strict=True, assign=True)
    assert type(model) is rq_upstream.RQTransformer
    assert model.vocab_size == [32769]*4
    del saved
    model.requires_grad_(False).eval().to(device)
    aux, _ = profile.tokenizer(device)
    profile.adapt(model, aux)
    config = load_config(BASE/'inputs/stage2-training-config.yaml')
    data = TarImageNet(BASE/'data', 'val', create_transforms(config.dataset, split='val'))
    pairs = [data[rank*2+i] for i in range(2)]
    images = torch.stack([p[0] for p in pairs]).to(device)
    labels = torch.tensor([p[1] for p in pairs], device=device)
    with torch.inference_mode():
        probabilities, codes = profile.targets(aux, images)
        del probabilities
        expected = model(codes, aux, labels, amp=True)
        prefix = torch.zeros_like(codes)
        model.init_cache()
        maximum = torch.zeros((), device=device)
        for h in range(8):
            for w in range(8):
                for d in range(4):
                    actual = model.cached_forward(prefix[:, :h+1], aux, labels, amp=True, sample_loc=(h, w, d))
                    p = expected[:, h, w, d].float().log_softmax(-1)
                    q = actual.float().log_softmax(-1)
                    maximum = torch.maximum(maximum, (p.exp()*(p-q)).sum(-1).max())
                    prefix[:, h, w, d] = codes[:, h, w, d]
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        assert float(maximum) < 1.e-4
    if rank == 0:
        write_json(OUT/'checkpoint-verification.json', dict(passed=True, strict_model_reload=True,
            source_step=1252, optimizer_age=1252, rng_streams=4, cosine_total_updates=62600,
            cache_events_per_rank=256, ranks=world, maximum_fp16_cache_kl=float(maximum),
            native_transformer=True, weights_unmodified=True, training_updates_added=0))
    del images, labels, codes, prefix, expected, actual, p, q, data
    model.init_cache()
    torch.cuda.empty_cache()
    source_sampling = profile.sampling_settings()
    sampling = dict(temperature=1., top_k=[512], top_p=[.95])
    assert source_sampling == dict(temperature=1., top_k=[1024], top_p=[.95])
    # Override this evaluation instance only. Keep the training recipe intact.
    profile.sampling_settings = lambda: dict(sampling)
    profile.name = 'LASER joint32k epoch2 | fixed weights | top-k512 FID50k'
    run = None
    if rank == 0:
        import wandb
        os.environ['WANDB_API_KEY'] = Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
        run = wandb.init(entity='helloimlixin-rutgers', project='laser', id=RUN,
            name=profile.name, dir=str(BASE/'wandb'), config=dict(plan, wandb_id=RUN,
                source_sampling=source_sampling, sampling=sampling, weights_unmodified=True,
                training_updates_added=0, fid_num_samples=50000, fid_real_split='train',
                fid_reference_images=1281167, auxiliary_loss_terms=0))
        run.define_metric('train/epoch')
        run.define_metric('eval/fid', step_metric='train/epoch', overwrite=True)
        write_json(OUT/'wandb.json', dict(id=RUN, url=run.url))
    shared.BASE, shared.OUT, shared.PROFILE, shared.RUN = BASE, OUT, profile, RUN
    shared.original_metrics.INCEPTION_PATH = BASE/'upstream/rqvae/metrics/inception.py'
    if rank == 0:
        shared.fid_workspace(run)
    dist.barrier()
    frozen = profile.digest(aux)
    started = time.time()
    result = shared.generation(model, aux, device, rank, world, 2, run, batch=128, seed=261001)
    assert profile.digest(aux) == frozen
    if rank == 0:
        result.update(generated_images=50000, real_images=1281167, real_split='train',
            source_step=1252, source_epoch=2, source_checkpoint_sha256=plan['checkpoint_sha256'],
            sampling=sampling, source_sampling=source_sampling, weights_unmodified=True,
            training_updates_added=0, generation_seed=261001, rejection_sampling=False,
            reference_fid=plan['reference_fid'], fid_delta=result['fid']-plan['reference_fid'],
            elapsed_seconds=time.time()-started)
        write_json(OUT/'fid50k.json', result)
        run.log({'train/epoch':2, 'train/global_step':1252, 'eval/fid':result['fid'],
            'eval/inception_score':result['inception_score']})
        run.save(str(OUT/'fid50k.json'), base_path=str(OUT), policy='now')
        run.finish()
        print(json.dumps(result), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
