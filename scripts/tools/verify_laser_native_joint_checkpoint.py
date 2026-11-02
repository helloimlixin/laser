#!/usr/bin/env python3
"""Strict reload, native cache parity, and full-batch sampling for joint IDs."""
import json
import os
from pathlib import Path
import sys

BASE = Path(os.environ['LASER_RQ_BASE'])
os.environ['RQ_UPSTREAM_ROOT'] = str(BASE/'upstream')
sys.path[:0] = [str(BASE/'source'), str(BASE/'runtime'), str(BASE/'upstream'), str(BASE)]
import torch
from src import rq_upstream
from src.training.laser_joint_recipe import JointLaserRecipe
from src.training.rq_imagenet_runtime import write_json, TarImageNet
from src.training import k4_checkpoint_io as checkpoint_io
from rqvae.utils.config import load_config, augment_arch_defaults, augment_optimizer_defaults
from rqvae.optimizer.optimizer import create_resnet_optimizer
from rqvae.optimizer.scheduler import create_scheduler
from rqvae.img_datasets.transforms import create_transforms


@torch.no_grad()
def verify_compact_targets(profile, aux, config, out, device):
    """Audit fixed temperatures on separate training images without tuning them."""
    from src.models.laser_joint_tokenizer import JointAtomCoefficientRQ
    audit = json.loads((BASE/'inputs/target-audit.json').read_text())
    source = torch.load(BASE/'inputs/stage1-model.pt', map_location='cpu', mmap=True,
                        weights_only=False)['state_dict']
    checked = 0
    for name, tensor in aux.backbone.state_dict().items():
        if not name.startswith(('encoder.', 'decoder.', 'quant_conv.', 'post_quant_conv.')):
            continue
        key = name
        if key not in source:
            key = key.replace('quant_conv.', 'pre_bottleneck.', 1) if key.startswith('quant_conv.') else key
            key = key.replace('post_quant_conv.', 'post_bottleneck.', 1) if key.startswith('post_quant_conv.') else key
        torch.testing.assert_close(tensor.cpu(), source[key], rtol=0, atol=0)
        checked += 1
    assert checked > 0 and all(not p.requires_grad for p in aux.parameters())
    del source
    spec = torch.load(BASE/'inputs/reference-joint-codebook.pt', map_location='cpu', weights_only=True)
    reference = JointAtomCoefficientRQ(spec['dictionary'], spec['atom_ids'], spec['coefficients']).to(device)
    data = TarImageNet(BASE/'data', 'train', create_transforms(config.dataset, split='train'), seed=audit['seed'])
    data.set_epoch(audit['epoch'])
    indices = audit['training_indices']
    temperatures = profile.target_temperature()
    totals = {name: torch.zeros(4, 4, dtype=torch.float64) for name in ('joint_65k', 'joint_32k')}
    errors = {name: torch.zeros(2, dtype=torch.float64) for name in totals}
    with torch.random.fork_rng(devices=[device.index]):
        for first in range(0, len(indices), 4):
            images = torch.stack([data[i][0] for i in indices[first:first+4]]).to(device)
            x = aux.encode(images)
            for name, quantizer in (('joint_65k', reference), ('joint_32k', aux.quantizer)):
                torch.manual_seed(261016+first)
                p, codes = quantizer.get_soft_codes(x, temp=temperatures, stochastic=True)
                assert torch.isfinite(p).all()
                residual = x.reshape(-1, 256).clone()
                for depth in range(4):
                    probability = p[..., depth, :].reshape(-1, quantizer.vocab_size)
                    scores = quantizer.scores(residual, depth)
                    maximum = scores.max(-1).values
                    nearest_error = (residual.square().sum(-1)-maximum).clamp_min(1.e-8)
                    extra_error = (probability*(maximum[:, None]-scores)).sum()
                    totals[name][depth] += torch.stack((
                        -(probability*probability.clamp_min(1.e-30).log()).sum(),
                        probability.max(-1).values.sum(), nearest_error.sum(), extra_error)).double().cpu()
                    residual.sub_(quantizer.embed_depth(codes[..., depth].reshape(-1), depth))
                hard = quantizer.quantize(x)['quantized']
                errors[name] += torch.stack(((x-hard).square().sum(), residual.square().sum())).double().cpu()
                del p
    profiles = {name: [dict(depth=d+1, mean_entropy=float(values[d, 0]/(len(indices)*64)),
        mean_maximum_probability=float(values[d, 1]/(len(indices)*64)),
        expected_extra_error_over_nearest_error=float(values[d, 3]/values[d, 2]))
        for d in range(4)] for name, values in totals.items()}
    ratios = {name: float(v[1]/v[0]) for name, v in errors.items()}
    passed = all(0 < row['mean_entropy'] < 11 and 0 < row['expected_extra_error_over_nearest_error'] < .1
                 for row in profiles['joint_32k']) and .9 < ratios['joint_32k'] < 1.2
    write_json(out/'compact-target-verification.json', dict(passed=passed,
        training_images=len(indices), training_indices=indices,
        disjoint_from_dictionary_fit_and_prior_temperature_calibration=True,
        augmentation='original training RandomCrop/RandomHorizontalFlip seed261001 epoch0',
        temperatures=temperatures, temperatures_refitted=False, profiles=profiles,
        sampled_to_hard_latent_mse_ratio=ratios, frozen_cnn_tensors_exact=checked,
        scope='Numerical and gross-corruption sanity check; generation gain remains untested'))
    assert passed, 'Compact target corruption sanity check failed'


def main():
    torch.set_num_threads(4)
    for name in ('src.training.rqtransformer', 'src.models.rqtransformer.transformers',
                 'src.training.stochastic_omp_graph', 'src.training.omp_joint_targets',
                 'src.training.stochastic_image_pairs'):
        assert name not in sys.modules, name
    profile = JointLaserRecipe(BASE)
    out = Path('/workspace/Projects/laser/outputs')/profile.run_id
    assert json.loads((out/'smoke/verification.json').read_text())['passed']
    checkpoint = checkpoint_io._checkpoint_upload_source((out/'smoke/checkpoints/last.pt').resolve())
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
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
    assert len(saved['rng_state_by_rank']) == saved['runtime']['world_size']
    torch.cuda.set_device(0); device = torch.device('cuda:0')
    aux, _ = profile.tokenizer(device); model.eval().to(device)
    data = TarImageNet(BASE/'data', 'val', create_transforms(config.dataset, split='val'))
    images = torch.stack([data[i][0] for i in range(2)]).to(device)
    with torch.inference_mode(), torch.random.fork_rng(devices=[0]):
        torch.manual_seed(261015); torch.cuda.manual_seed(261015)
        from rqvae.models.rqvae.quantizations import RQBottleneck
        native_q = RQBottleneck([8, 8, 256], [8, 8, 4], aux.quantizer.vocab_size,
                               shared_codebook=False, restart_unused_codes=False).eval().to(device)
        for d, book in enumerate(native_q.codebooks):
            book.weight[:-1].copy_(aux.quantizer.expanded_codebook(d))
        latent = aux.encode(images)
        native_hard, _, native_codes = native_q(latent)
        hard = aux.quantizer.quantize(latent)
        assert torch.equal(hard['codes'], native_codes)
        torch.testing.assert_close(hard['quantized'], native_hard, rtol=1.e-4, atol=2.e-5)
        temperature = profile.target_temperature()
        if isinstance(temperature, list):
            # Native distance calculations and native embeddings, with only
            # the temperature varying at each causal residual depth.
            residual = native_q.to_code_shape(latent).detach().clone()
            probabilities_by_depth, codes_by_depth = [], []
            for book, value in zip(native_q.codebooks, temperature):
                distances = book.compute_distances(residual)
                probability = (-distances / value).softmax(-1)
                chosen = distances.argmin(-1)
                probabilities_by_depth.append(probability)
                codes_by_depth.append(chosen)
                residual -= book.embed(chosen)
            expected_p = torch.stack(probabilities_by_depth, -2)
            expected_c = torch.stack(codes_by_depth, -1)
            del probabilities_by_depth, codes_by_depth, distances, probability, residual
        else:
            expected_p, expected_c = native_q.get_soft_codes(latent, temp=temperature, stochastic=False)
        actual_p, actual_c = aux.quantizer.get_soft_codes(latent, temp=temperature, stochastic=False)
        assert torch.equal(expected_c, actual_c)
        maximum_quantizer_kl = float((expected_p * (expected_p.clamp_min(1.e-30).log() -
            actual_p.clamp_min(1.e-30).log())).sum(-1).max())
        assert maximum_quantizer_kl < 1.e-4
        del expected_p, actual_p, native_q
        probabilities, codes = profile.targets(aux, images)
        target_check = profile.verify_targets(aux, probabilities, codes)
        del probabilities
        cond = torch.tensor([0, 999], device=device)
        unaligned = model(codes, aux, cond, amp=True)
        parameter_ids = [id(p) for p in model.parameters()]
        profile.adapt(model, aux)
        assert parameter_ids == [id(p) for p in model.parameters()]
        expected = model(codes, aux, cond, amp=True)
        log_p, log_q = unaligned.float().log_softmax(-1), expected.float().log_softmax(-1)
        alignment_kl = float((log_p.exp() * (log_p-log_q)).sum(-1).max())
        assert alignment_kl < 1.e-5
        assert expected.shape[-1] == aux.quantizer.vocab_size
        del unaligned, log_p, log_q
        prefix = torch.zeros_like(codes); model.init_cache()
        maximum_kl = 0.
        for h in range(8):
            for w in range(8):
                for d in range(4):
                    actual = model.cached_forward(prefix[:, :h+1], aux, cond, amp=True, sample_loc=(h, w, d))
                    p = expected[:, h, w, d].float().log_softmax(-1)
                    q = actual.float().log_softmax(-1)
                    maximum_kl = max(maximum_kl, float((p.exp()*(p-q)).sum(-1).max()))
                    prefix[:, h, w, d] = codes[:, h, w, d]
        assert maximum_kl < 1.e-4
        model.init_cache(); del expected
        count = profile.generation_batch
        generated = profile.sample(model, aux, count, torch.arange(count, device=device)%1000)
        assert generated.shape == (count, 8, 8, 4)
        assert ((generated >= 0) & (generated < aux.quantizer.vocab_size)).all()
        for chunk in generated.split(8):
            decoded = profile.decode(aux, chunk)
            assert decoded.shape == (len(chunk), 3, 256, 256) and torch.isfinite(decoded).all()
    if profile.recipe.tokenizer.get('learned_compact_dictionary'):
        verify_compact_targets(profile, aux, config, out, device)
    write_json(out/'gpu-generation-verification.json', dict(passed=True,
        native_transformer=True, cached_events_verified=256, maximum_fp16_cache_kl=maximum_kl,
        classifier_alignment_preserves_parameters=True, maximum_fp16_alignment_kl=alignment_kl,
        actual_codebook_native_quantizer_hard_ids_exact=True,
        maximum_soft_target_kl_against_native_quantizer=maximum_quantizer_kl,
        generated_images=count, production_generation_batch=count, finite_decoded_images=True,
        integer_token_shape=list(generated.shape), targets=target_check,
        peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3))
    write_json(out/'smoke-reload-verification.json', dict(passed=True, global_step=2,
        strict_model_reload=True, optimizer_state_count=len(optimizer.state), all_adam_steps=2,
        scheduler_horizon=62600, scheduler_step=2, rng_streams=len(saved['rng_state_by_rank']),
        private_smoke_discarded_for_fresh_production=True, checkpoint_bytes=checkpoint.stat().st_size,
        production_size_gpu_sampling_verified=True, codebook_sha256=profile.codebook_sha256))
    print('Native joint RQ checkpoint, cache, targets and sampler verified', flush=True)


if __name__ == '__main__': main()
