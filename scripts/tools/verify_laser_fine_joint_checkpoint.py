#!/usr/bin/env python3
"""Verify a fresh joint512 checkpoint, every cache event, and production batch sampling."""
import json
import os
from pathlib import Path
import sys

BASE = Path(os.environ['LASER_RQ_BASE'])
os.environ['RQ_UPSTREAM_ROOT'] = str(BASE/'upstream')
sys.path[:0] = [str(BASE/'source'), str(BASE/'runtime'), str(BASE/'upstream'), str(BASE)]
import torch
from src import rq_upstream
from src.training.laser_fine_joint_recipe import FineJointLaserRecipe
from src.training.rq_imagenet_runtime import write_json, TarImageNet
from src.training import k4_checkpoint_io as checkpoint_io
from rqvae.utils.config import load_config, augment_arch_defaults, augment_optimizer_defaults
from rqvae.optimizer.optimizer import create_resnet_optimizer
from rqvae.optimizer.scheduler import create_scheduler
from rqvae.img_datasets.transforms import create_transforms


def main():
    torch.set_num_threads(4)
    profile = FineJointLaserRecipe(BASE)
    out = Path('/workspace/Projects/laser/outputs')/profile.run_id
    assert json.loads((out/'smoke/verification.json').read_text())['passed']
    path = checkpoint_io._checkpoint_upload_source((out/'smoke/checkpoints/last.pt').resolve())
    saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    profile.verify_resume(saved)
    assert saved['global_step'] == 2 and saved['runtime']['fresh_stage2']
    config = load_config(BASE/'inputs/stage2-training-config.yaml')
    arch = augment_arch_defaults(config.arch); arch.vocab_size = config.dataset.vocab_size
    with torch.device('meta'): model = profile.model(arch)
    model.load_state_dict(saved['state_dict'], strict=True, assign=True)
    assert type(model) is rq_upstream.RQTransformer and tuple(model.block_size) == (8, 8, 4)
    assert len(model.body_transformer.blocks) == 42 and len(model.head_transformer.blocks) == 6
    optimizer = create_resnet_optimizer(model, augment_optimizer_defaults(config.optimizer))
    scheduler = create_scheduler(optimizer, config.optimizer.warmup, 626, 100)
    optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
    assert scheduler.after_scheduler.last_epoch == 2 and scheduler.after_scheduler.T_max == 62600
    assert len(optimizer.state) == len(list(model.parameters()))
    assert all(int(s['step']) == 2 for s in optimizer.state.values())
    assert all(torch.isfinite(s['exp_avg']).all() and torch.isfinite(s['exp_avg_sq']).all()
               for s in optimizer.state.values())
    assert len(saved['rng_state_by_rank']) == saved['runtime']['world_size'] == 4
    assert all('coefficient' in state for state in saved['rng_state_by_rank'])
    torch.cuda.set_device(0); device = torch.device('cuda:0')
    aux, _ = profile.tokenizer(device); model.eval().to(device); profile.adapt(model, aux)
    data = TarImageNet(BASE/'data', 'val', create_transforms(config.dataset, split='val'))
    images = torch.stack([data[i][0] for i in range(2)]).to(device)
    with torch.inference_mode():
        probabilities, codes = profile.targets(aux, images)
        target_check = profile.verify_targets(aux, probabilities, codes)
        state = profile.coefficient_generator.get_state()
        repeated, repeated_codes = profile.targets(aux, images)
        profile.coefficient_generator.set_state(state)
        replayed, replayed_codes = profile.targets(aux, images)
        assert torch.equal(repeated_codes, replayed_codes)
        cond = torch.tensor([0, 999], device=device)
        expected = model(codes, aux, cond, amp=True)
        assert torch.isfinite(profile.loss(model, expected, probabilities))
        prefix = torch.zeros_like(codes); model.init_cache()
        maximum_kl = 0.; maximum_context_error = 0.
        for h in range(8):
            for w in range(8):
                for d in range(4):
                    actual = model.cached_forward(prefix[:, :h+1], aux, cond, amp=True, sample_loc=(h, w, d))
                    p = expected[:, h, w, d, :aux.num_atoms].float().log_softmax(-1)
                    q = actual[:, :aux.num_atoms].float().log_softmax(-1)
                    maximum_kl = max(maximum_kl, float((p.exp()*(p-q)).sum(-1).max()))
                    atoms = codes[:, h, w, d]//aux.coeff_vocab_size
                    p = model.classifier.linear.coefficient_logits(expected[:, h, w, d], atoms).float().log_softmax(-1)
                    q = model.classifier.linear.coefficient_logits(actual, atoms).float().log_softmax(-1)
                    maximum_context_error = max(maximum_context_error, float((p.exp()*(p-q)).sum(-1).max()))
                    prefix[:, h, w, d] = codes[:, h, w, d]
        assert maximum_kl < 1e-4 and maximum_context_error < 1e-4
        model.init_cache(); del expected
        count = profile.generation_batch
        generated = profile.sample(model, aux, count, torch.arange(count, device=device)%1000)
        assert generated.shape == (count, 8, 8, 4)
        assert ((generated >= 0) & (generated < aux.vocab_size)).all()
        assert ((generated//aux.coeff_vocab_size).sort(-1).values.diff(dim=-1) > 0).all()
        for chunk in generated.split(8):
            decoded = profile.decode(aux, chunk)
            assert decoded.shape == (len(chunk), 3, 256, 256) and torch.isfinite(decoded).all()
    write_json(out/'gpu-generation-verification.json', dict(passed=True,
        native_transformer=True, cached_events_verified=256, maximum_atom_cache_kl=maximum_kl,
        maximum_coefficient_cache_kl=maximum_context_error, private_coefficient_rng_replays_exactly=True,
        generated_images=count, finite_decoded_images=True, unique_generated_supports=True,
        integer_token_shape=list(generated.shape), targets=target_check,
        peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3))
    write_json(out/'smoke-reload-verification.json', dict(passed=True, global_step=2,
        strict_model_reload=True, optimizer_state_count=len(optimizer.state), all_adam_steps=2,
        scheduler_horizon=62600, scheduler_step=2, rng_streams=len(saved['rng_state_by_rank']),
        private_smoke_discarded_for_fresh_production=True, checkpoint_bytes=path.stat().st_size,
        production_size_gpu_sampling_verified=True))
    print('Joint512 checkpoint, cache, targets and sampler verified', flush=True)


if __name__ == '__main__': main()
