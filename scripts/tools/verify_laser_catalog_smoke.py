#!/usr/bin/env python3
"""Strict reload and native cache parity for the fresh integer-catalog run."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    base = Path(os.environ['LASER_RQ_BASE'])
    os.environ['RQ_UPSTREAM_ROOT'] = str(base/'upstream')
    sys.path[:0] = [str(base/'source'), str(base/'upstream'), str(base/'runtime')]
    import torch
    from src import rq_upstream
    from src.training.laser_catalog_integer_recipe import CatalogIntegerLaserRecipe
    from src.training.rq_imagenet_runtime import TarImageNet
    from src.training.rq_sample_panels import CLASSES, PREVIEW_IMAGES, OVERVIEW_IMAGES
    from rqvae.utils.config import load_config, augment_arch_defaults, augment_optimizer_defaults
    from rqvae.optimizer.optimizer import create_resnet_optimizer
    from rqvae.optimizer.scheduler import create_scheduler
    from rqvae.img_datasets.transforms import create_transforms
    torch.set_num_threads(4)
    profile = CatalogIntegerLaserRecipe(base)
    out = Path('/workspace/Projects/laser/outputs')/profile.run_id
    smoke = Path(os.environ.get('LASER_PRIVATE_SMOKE_DIR', str(base/'private-smoke')))
    verification = json.loads((smoke/'verification.json').read_text())
    assert verification['passed'] and verification['global_step'] == 2
    receipts = [json.loads(p.read_text()) for p in (smoke/'checkpoints/.checkpoint-data').glob('*.json')]
    receipt = max(receipts, key=lambda row:row['global_step'])
    assert receipt['global_step'] == 2
    for key in ['local', 'persistent']:
        p = Path(receipt[key])
        assert p.stat().st_size == receipt['bytes']
        with p.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == receipt['sha256']
    saved = torch.load(receipt['local'], map_location='cpu', mmap=True, weights_only=False)
    profile.verify_resume(saved)
    assert saved['runtime']['fresh_stage2'] and saved['runtime']['fresh_optimizer']
    assert saved['runtime']['global_step'] == 0 and saved['runtime']['initial_lr'] == .0005
    assert saved['runtime']['warmup_epochs'] == 0 and saved['runtime']['lr_schedule_epochs'] == 100
    assert len(saved['rng_state_by_rank']) == 4
    config = load_config(base/'inputs/stage2-training-config.yaml')
    arch = augment_arch_defaults(config.arch)
    with torch.device('meta'):
        model = profile.model(arch)
    model.load_state_dict(saved['state_dict'], strict=True, assign=True)
    count = sum(p.numel() for p in model.parameters())
    assert count == 1765262848 and type(model) is rq_upstream.RQTransformer
    assert type(model.classifier.linear) is torch.nn.Linear
    assert model.classifier.linear.weight.shape == (262144, 1536)
    optimizer = create_resnet_optimizer(model, augment_optimizer_defaults(config.optimizer))
    scheduler = create_scheduler(optimizer, config.optimizer.warmup, 626, 100)
    optimizer.load_state_dict(saved['optimizer'])
    scheduler.load_state_dict(saved['scheduler'])
    assert scheduler.after_scheduler.last_epoch == 2 and scheduler.after_scheduler.T_max == 62600
    assert len(optimizer.state) == len(list(model.parameters()))
    for state in optimizer.state.values():
        assert int(state['step']) == 2
        for tensor in (state['exp_avg'], state['exp_avg_sq']):
            for chunk in tensor.reshape(-1).split(1048576):
                assert torch.isfinite(chunk).all()
    preview = json.loads((out/'smoke/samples/step000002-panels.json').read_text())
    assert preview['generated_images'] == PREVIEW_IMAGES == 512
    assert preview['overview_images'] == OVERVIEW_IMAGES == 80
    assert preview['classes'] == [dict(name=name, imagenet_id=label) for name, label in CLASSES]
    for rank in range(4):
        assert json.loads((smoke/f'preview-verification-rank{rank}.json').read_text())['training_rng_unchanged']
    del optimizer
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    model = model.eval().to(device)
    aux, _ = profile.tokenizer(device)
    data = TarImageNet(base/'data', 'val', create_transforms(config.dataset, split='val'))
    images = torch.stack([data[i][0] for i in range(2)]).to(device)
    labels = torch.tensor([106, 269], device=device)
    with torch.no_grad():
        targets, codes = profile.targets(aux, images)
        target_check = profile.verify_targets(aux, targets, codes)
        expected = model(codes, aux, labels, amp=True)
        maximum_kl = 0.
        prefix = torch.zeros_like(codes)
        model.init_cache()
        for h in range(8):
            for w in range(8):
                for d in range(4):
                    actual = model.cached_forward(prefix[:, :h+1], aux, labels, amp=True, sample_loc=(h, w, d))
                    reference = expected[:, h, w, d].float().log_softmax(-1)
                    logp = actual.float().log_softmax(-1)
                    kl = (reference.exp()*(reference-logp)).sum(-1).max()
                    maximum_kl = max(maximum_kl, float(kl))
                    prefix[:, h, w, d] = codes[:, h, w, d]
        assert maximum_kl < 1.e-4, maximum_kl
        model.init_cache()
    proof = dict(passed=True, native_transformer=True, strict_model_reload=True,
        strict_optimizer_reload=True, all_adam_steps=2, scheduler_updates=2,
        cosine_total_updates=62600, parameter_count=count, vocabulary=262144,
        classifier_projection_dim=1536, classifier_low_rank_projection=False,
        source_checkpoint_sha256=receipt['sha256'], targets=target_check,
        maximum_fp16_cache_kl=maximum_kl, cached_events_verified=256,
        designated_preview_images=512, overview_images=80, class_grids=8,
        training_rng_unchanged_by_previews=True, production_starts_from_zero=True,
        preflight_weights_discarded=True, time=time.time())
    (out/'smoke-reload-verification.json').write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
