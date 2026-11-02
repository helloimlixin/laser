#!/usr/bin/env python3
"""Matched inference probes treating each codeword as one categorical integer token."""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time


TOKEN_METRICS = (
    'ce', 'target_entropy', 'kl', 'predictive_entropy',
    'target_maximum_probability', 'sample_top1_accuracy',
    'soft_target_top1_accuracy', 'top256_target_coverage',
    'top1024_target_coverage', 'top2048_target_coverage',
    'class_context_gain', 'spatial_context_gain', 'depth_prefix_gain',
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.pending')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def wait_for(path):
    while not path.exists():
        time.sleep(5)


def target_codes(profile, aux, latent, temperatures=None):
    if profile.kind == 'rq':
        return aux.quantizer.get_soft_codes(latent, temp=.5, stochastic=True)
    return aux.quantizer.get_soft_codes(latent,
        temp=temperatures or profile.target_temperature(), stochastic=True,
        chunk_size=int(profile.recipe.targets.chunk_size))


def image_sums(values):
    return values.reshape(values.shape[0], -1, values.shape[-1]).mean(1)


def per_depth_ce(logits, targets):
    return image_sums(-(targets * logits.float().log_softmax(-1)).sum(-1))


def build_cache(plan, device):
    local, out = Path(plan['local']), Path(plan['output'])
    original = RQRecipe(Path(plan['cases'][0]['base']))
    laser = JointLaserRecipe(Path(plan['runtime']))
    profiles = dict(original_rq=original, laser=laser)
    train_ids = np.random.default_rng(plan['seed']).permutation(1281167)[:plan['fit_images']].tolist()
    val_ids = np.random.default_rng(plan['seed'] + 1).permutation(50000)[:plan['images']].tolist()
    config = load_config(Path(plan['runtime']) / 'inputs/stage2-training-config.yaml')
    cache = dict(seed=plan['seed'], fit_indices=train_ids, validation_indices=val_ids,
                 original_rq={}, laser={})
    for name, profile in profiles.items():
        aux, _ = profile.tokenizer(device)
        aux.requires_grad_(False).eval()
        versions = [v._version for v in aux.state_dict().values()]
        for split, indices in [('train', train_ids), ('val', val_ids)]:
            data = TarImageNet(Path(plan['runtime']) / 'data', split,
                create_transforms(config.dataset, split='val'))
            latents, labels, images = [], [], []
            for first in range(0, len(indices), 4):
                records = [data[i] for i in indices[first:first + 4]]
                x = torch.stack([r[0] for r in records]).to(device)
                latents.append(aux.encode(x).cpu())
                labels.extend(r[1] for r in records)
                if split == 'val':
                    images.append(x.cpu())
                write(out / 'noise-status.json', dict(phase='encoding', case=name,
                    split=split, images=first + len(records), time=time.time()))
            cache[name][split] = torch.cat(latents)
            cache[name][split + '_labels'] = torch.tensor(labels)
            if split == 'val':
                cache[name]['images'] = torch.cat(images)
        depth, vocabulary = profile.shape[-1], 16384 if name == 'original_rq' else aux.quantizer.vocab_size
        counts = torch.zeros(depth, vocabulary, dtype=torch.float64, device=device)
        for first in range(0, plan['fit_images'], 4):
            torch.manual_seed(plan['seed'] + 10 + first)
            q, _ = target_codes(profile, aux, cache[name]['train'][first:first + 4].to(device))
            counts += q.reshape(-1, depth, vocabulary).sum(0).double()
        cache[name]['fit_soft_counts'] = counts.cpu()
        assert versions == [v._version for v in aux.state_dict().values()]
        del aux
        gc.collect(); torch.cuda.empty_cache()
    torch.save(cache, local / 'matched-cache.pt')
    write(out / 'matched-data.json', dict(training_fit_images=plan['fit_images'],
        validation_images=plan['images'], fit_indices=train_ids, validation_indices=val_ids,
        transform='Common Resize/CenterCrop validation transform on both populations',
        target_draw_policy='Same explicit seed per batch for every LASER checkpoint',
        unigram_fit_split='train', predictor_evaluation_split='val', time=time.time()))
    write(local / 'cache-ready.json', dict(passed=True, time=time.time()))
    return cache


def noise(plan, device):
    spec = importlib.util.spec_from_file_location('noise_probe_support',
        ROOT / 'src/training/joint_noise_calibration.py')
    support = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(support)
    hard_trajectory, measure_noise, public_profile = (
        support.hard_trajectory, support.measure_noise, support.public_profile)
    cache = build_cache(plan, device)
    out, local = Path(plan['output']), Path(plan['local'])
    profiles = dict(original_rq=RQRecipe(Path(plan['cases'][0]['base'])),
                    laser=JointLaserRecipe(Path(plan['runtime'])))
    calibration = json.loads((ROOT / 'outputs/imagenet-laser-joint20k-depth5-rq-noise-calibration-20261017/summary.json').read_text())
    results = {}
    for family, profile in profiles.items():
        aux, _ = profile.tokenizer(device)
        versions = [v._version for v in aux.state_dict().values()]
        if family == 'original_rq':
            class NativeBook:
                depth = 4

                def scores(self, residual, depth):
                    return -aux.quantizer.codebooks[depth].compute_distances(residual)

                def embed_depth(self, codes, depth):
                    return aux.quantizer.codebooks[depth].embed(codes)
            book = NativeBook()
            policies = {'original_rq': [.5] * 4}
        else:
            book = aux.quantizer
            current = profile.target_temperature()
            policies = dict(laser_current=current,
                laser_half_temperature=[t * .5 for t in current],
                laser_double_temperature=[t * 2 for t in current],
                laser_prior_rq_matched=calibration['proposed_temperatures'])
        latent = cache[family]['val'].to(device)
        hard = hard_trajectory(book, latent)
        # The generic adapter must produce exactly the native deterministic codes.
        native_q, native_codes = aux.quantizer.get_soft_codes(latent[:2],
            temp=.5 if family == 'original_rq' else profile.target_temperature(), stochastic=False)
        assert torch.equal(native_codes.cpu(), hard['codes'][:2])
        del native_q, native_codes
        for name, temperatures in policies.items():
            write(out / 'noise-status.json', dict(phase='noise_profiles', case=name, time=time.time()))
            measured = measure_noise(book, latent, temperatures,
                [plan['seed'] + 100 + k for k in range(4)], hard=hard, chunk_size=256)
            public = public_profile(measured)
            hard_codes = measured['hard_codes']
            sampled = measured['sampled_codes']
            public['sample_disagreement_with_full_hard_trajectory_by_depth'] = torch.stack(
                [(c != hard_codes).float().reshape(-1, book.depth).mean(0) for c in sampled]).mean(0).tolist()
            public['pairwise_stochastic_code_disagreement_by_depth'] = torch.stack(
                [(sampled[a] != sampled[b]).float().reshape(-1, book.depth).mean(0)
                 for a in range(4) for b in range(a)]).mean(0).tolist()
            pixel_count = min(32, plan['images'])
            hard_mse, soft_mse, changes = [], [[], []], [[], []]
            for first in range(0, pixel_count, 4):
                real = cache[family]['images'][first:first + 4].to(device)
                decoded_hard = aux.decode_code(hard_codes[first:first + 4].to(device)).clamp(-1, 1)
                hard_mse.extend((decoded_hard - real).square().flatten(1).mean(-1).cpu().tolist())
                for draw in range(2):
                    decoded = aux.decode_code(sampled[draw][first:first + 4].to(device)).clamp(-1, 1)
                    soft_mse[draw].extend((decoded - real).square().flatten(1).mean(-1).cpu().tolist())
                    changes[draw].extend((decoded - decoded_hard).square().flatten(1).mean(-1).cpu().tolist())
            public['pixel_probe'] = dict(images=pixel_count, stochastic_draws=2,
                hard_reconstruction_mse=float(np.mean(hard_mse)),
                sampled_reconstruction_mse=float(np.mean(soft_mse)),
                sampled_to_hard_reconstruction_mse_ratio=float(np.mean(soft_mse) / np.mean(hard_mse)),
                stochastic_vs_hard_image_mse=float(np.mean(changes)))
            results[name] = public
            write(out / 'noise.json', dict(completed=False, cases=results))
        assert versions == [v._version for v in aux.state_dict().values()]
        del aux, latent
        gc.collect(); torch.cuda.empty_cache()
    report = dict(completed=True, validation_images=plan['images'], stochastic_draws=4,
        cases=results, native_hard_code_parity_passed=True, tokenizer_state_unchanged=True,
        training_modified=False, peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
        limits=['Physical errors are normalized within each tokenizer; raw latent units differ.',
            'Noise and reconstruction probes do not establish generation-quality gains.',
            'Alternate temperatures are diagnostic policies; no new training is launched.'], time=time.time())
    write(out / 'noise.json', report)
    write(out / 'noise-status.json', dict(phase='completed', time=time.time()))


def predict_case(case, plan, cache, device):
    base, out = Path(case['base']), Path(plan['output'])
    profile = RQRecipe(base) if case['kind'] == 'rq' else JointLaserRecipe(base)
    saved = torch.load(case['checkpoint'], mmap=True, map_location='cpu', weights_only=False)
    assert saved['global_step'] == case['step'] and saved['run_id'] == profile.run_id
    profile.verify_resume(saved)
    with torch.device('meta'):
        model = profile.model(OmegaConf.create(saved['config']['arch']))
    model.load_state_dict(saved['state_dict'], strict=True, assign=True)
    model.requires_grad_(False).eval().to(device)
    del saved
    aux, _ = profile.tokenizer(device)
    profile.adapt(model, aux)
    versions = [p._version for p in model.parameters()]
    family = 'original_rq' if case['kind'] == 'rq' else 'laser'
    data = cache[family]
    depth = profile.shape[-1]
    fit_counts = data['fit_soft_counts'].to(device).float()
    unigram = (fit_counts + .01) / (fit_counts.sum(-1, keepdim=True) + .01 * fit_counts.shape[-1])
    all_metrics = {}
    fingerprint = hashlib.sha256()

    def append(name, value):
        all_metrics.setdefault(name, []).append(value.detach().double().cpu())

    for first in range(0, plan['images'], 2):
        latent = data['val'][first:first + 2].to(device)
        labels = data['val_labels'][first:first + 2].to(device)
        torch.manual_seed(plan['seed'] + 200 + first)
        target, codes = target_codes(profile, aux, latent)
        fingerprint.update(codes.cpu().numpy().tobytes())
        fingerprint.update(target.reshape(-1, target.shape[-1])[:8].cpu().numpy().tobytes())
        logits = model(codes, aux, labels, amp=True).float()
        logp = logits.log_softmax(-1)
        p = logp.exp()
        ce = per_depth_ce(logits, target)
        entropy = image_sums(-(target * target.clamp_min(1.e-30).log()).sum(-1))
        append('ce', ce); append('target_entropy', entropy); append('kl', ce - entropy)
        append('predictive_entropy', image_sums(-(p * logp).sum(-1)))
        append('target_maximum_probability', image_sums(target.max(-1).values))
        append('sample_top1_accuracy', image_sums((p.argmax(-1) == codes).float()))
        append('soft_target_top1_accuracy', image_sums(target.gather(-1, p.argmax(-1, keepdim=True)).squeeze(-1)))
        unigram_ce = image_sums(-(target * unigram.log()).sum(-1))
        append('unigram_ce', unigram_ce); append('context_gain_over_unigram', unigram_ce - ce)
        for alpha in (.1, 1.):
            marginal = (fit_counts + alpha) / (fit_counts.sum(-1, keepdim=True) + alpha * fit_counts.shape[-1])
            append(f'unigram_ce_alpha{alpha}', image_sums(-(target * marginal.log()).sum(-1)))
        for k in (256, 1024, 2048):
            index = p.topk(k, -1).indices
            append(f'top{k}_target_coverage', image_sums(target.gather(-1, index).sum(-1)))
        del logits, logp, p
        wrong = model(codes, aux, (labels + 499) % 1000, amp=True)
        append('class_context_gain', per_depth_ce(wrong, target) - ce)
        del wrong
        # Swap only spatial context; retain each image's correct within-site code prefix.
        hook = model.body_transformer.register_forward_hook(lambda _m, _i, output: output.flip(0))
        wrong = model(codes, aux, labels, amp=True)
        hook.remove()
        append('spatial_context_gain', per_depth_ce(wrong, target) - ce)
        del wrong
        # Swap only depth-prefix embeddings; retain the correct spatial transformer output.
        hook = model.head_mlp.register_forward_hook(lambda _m, _i, output: output.flip(0))
        wrong = model(codes, aux, labels, amp=True)
        hook.remove()
        depth_ce = per_depth_ce(wrong, target)
        torch.testing.assert_close(depth_ce[:, 0], ce[:, 0], rtol=2.e-5, atol=2.e-5)
        append('depth_prefix_gain', depth_ce - ce)
        del wrong, target, codes
        write(out / 'predict-status.json', dict(phase='measuring', case=case['name'],
            images=first + len(labels), total_images=plan['images'], time=time.time()))
    assert versions == [p._version for p in model.parameters()]
    assert all(p.grad is None for p in model.parameters())
    matrices = {name:torch.cat(values).numpy() for name, values in all_metrics.items()}
    rng = np.random.default_rng(plan['seed'] + 300)
    draws = rng.integers(0, plan['images'], (2000, plan['images']))
    metrics = {}
    for name, values in matrices.items():
        bootstrap = values[draws].mean(1)
        metrics[name] = dict(per_depth=values.mean(0).tolist(),
            bootstrap95_low=np.quantile(bootstrap, .025, axis=0).tolist(),
            bootstrap95_high=np.quantile(bootstrap, .975, axis=0).tolist(),
            mean=float(values.mean()))
    np.savez_compressed(out / f'{case["name"]}-per-image.npz', **matrices)
    report = dict(completed=True, case=case, images=plan['images'], metrics=metrics,
        target_fingerprint=fingerprint.hexdigest(), model_parameters_unchanged=True,
        training_modified=False, depth_prefix_first_depth_invariance_passed=True,
        peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30, time=time.time())
    write(out / f'{case["name"]}.json', report)
    print(json.dumps(dict(case=case['name'], ce=metrics['ce']['per_depth'],
        entropy=metrics['target_entropy']['per_depth'], kl=metrics['kl']['per_depth'])), flush=True)
    del model, aux
    gc.collect(); torch.cuda.empty_cache()
    return report


def predict(plan, device):
    local, out = Path(plan['local']), Path(plan['output'])
    wait_for(local / 'cache-ready.json')
    cache = torch.load(local / 'matched-cache.pt', weights_only=False, mmap=True, map_location='cpu')
    reports = {}
    for case in plan['cases']:
        reports[case['name']] = predict_case(case, plan, cache, device)
    laser = [r for name, r in reports.items() if name != 'original_epoch2']
    assert len({r['target_fingerprint'] for r in laser}) == 1
    comparisons = {}
    for a, b in [('weighted_epoch2', 'uniform_epoch2'), ('weighted_epoch3', 'weighted_epoch2')]:
        x = np.load(out / f'{a}-per-image.npz'); y = np.load(out / f'{b}-per-image.npz')
        rng = np.random.default_rng(plan['seed'] + 301)
        draws = rng.integers(0, plan['images'], (2000, plan['images']))
        comparisons[a + '_minus_' + b] = {}
        for metric in ['ce', 'kl']:
            difference = x[metric] - y[metric]
            bootstrap = difference[draws].mean(1)
            comparisons[a + '_minus_' + b][metric] = dict(per_depth=difference.mean(0).tolist(),
                bootstrap95_low=np.quantile(bootstrap, .025, axis=0).tolist(),
                bootstrap95_high=np.quantile(bootstrap, .975, axis=0).tolist())
    write(out / 'predictability-summary.json', dict(completed=True,
        matched_epoch2_cases=['original_epoch2', 'uniform_epoch2', 'weighted_epoch2'],
        all_laser_inputs_targets_and_codes_identical=True, comparisons=comparisons,
        cases={name:{k:r['metrics'][k] for k in TOKEN_METRICS} for name,r in reports.items()},
        limits=['128 common held-out images; one checkpoint/training seed per condition.',
            'Bootstrap intervals cover images, not training-seed uncertainty.',
            'Different tokenizers define different likelihood tasks; raw CE/KL are not image likelihoods.',
            'Teacher-forced prediction and context interventions do not measure generation FID.'], time=time.time()))
    write(out / 'predict-status.json', dict(phase='completed', time=time.time()))


def intervene(plan, device):
    """Separate prefix policy from label entropy and check physical error importance."""
    out, local = Path(plan['output']), Path(plan['local'])
    wait_for(local / 'cache-ready.json')
    cache = torch.load(local / 'matched-cache.pt', mmap=True, map_location='cpu', weights_only=False)
    calibration = json.loads((ROOT / 'outputs/imagenet-laser-joint20k-depth5-rq-noise-calibration-20261017/summary.json').read_text())
    geometry_only = plan.get('geometry_only', False)
    previous = json.loads((out / 'interventions.json').read_text()) if geometry_only else None
    reports = {} if previous is None else previous['cases']
    for case in plan['cases']:
        profile = RQRecipe(Path(case['base'])) if case['kind'] == 'rq' else JointLaserRecipe(Path(case['base']))
        saved = torch.load(case['checkpoint'], mmap=True, map_location='cpu', weights_only=False)
        profile.verify_resume(saved)
        with torch.device('meta'):
            model = profile.model(OmegaConf.create(saved['config']['arch']))
        model.load_state_dict(saved['state_dict'], strict=True, assign=True)
        model.requires_grad_(False).eval().to(device)
        del saved
        aux, _ = profile.tokenizer(device)
        profile.adapt(model, aux)
        family = 'original_rq' if case['kind'] == 'rq' else 'laser'
        data, depth = cache[family], profile.shape[-1]
        books = [(aux.quantizer.codebooks[d].weight[:16384] if case['kind'] == 'rq'
                  else aux.quantizer.expanded_codebook(d)) for d in range(depth)]
        metrics = {}

        def add(name, value):
            metrics.setdefault(name, []).append(value.detach().double().cpu())

        def geometry(logits, target, codes, latent):
            rows = {}
            residual = latent.reshape(-1, latent.shape[-1]).clone()
            for d in range(depth):
                book = books[d]
                p = logits[..., d, :].reshape(-1, logits.shape[-1]).float().softmax(-1)
                q = target[..., d, :].reshape_as(p)
                norms = book.square().sum(-1)
                initial = residual.square().sum(-1)
                pmean, qmean = p @ book, q @ book
                p_error = initial + p @ norms - 2 * (residual * pmean).sum(-1)
                q_error = initial + q @ norms - 2 * (residual * qmean).sum(-1)
                distances = initial[:, None] + norms[None] - 2 * residual @ book.T
                minimum, nearest = distances.min(-1)
                top1 = book[p.argmax(-1)]
                # Aggregate physical energy over an image before division. Individual
                # zero-code sites can have zero achievable reduction; dividing each
                # site first produces meaningless arbitrarily large ratios.
                def normalized(numerator, denominator):
                    a = numerator.reshape(len(codes), -1).double().sum(-1)
                    b = denominator.reshape(len(codes), -1).double().sum(-1)
                    assert (b > 1.e-8).all()
                    return a / b
                oracle_reduction = initial - minimum
                values = dict(model_expected_residual_error_over_initial=normalized(p_error, initial),
                    target_expected_residual_error_over_initial=normalized(q_error, initial),
                    greedy_residual_error_over_initial=normalized(minimum, initial),
                    model_fraction_of_oracle_error_reduction=normalized(initial - p_error, oracle_reduction),
                    target_fraction_of_oracle_error_reduction=normalized(initial - q_error, oracle_reduction),
                    model_mean_vector_error_over_initial=normalized((residual - pmean).square().sum(-1), initial),
                    model_top1_error_over_initial=normalized((residual - top1).square().sum(-1), initial))
                assert (values['target_fraction_of_oracle_error_reduction'] <= 1.0001).all()
                for name, value in values.items():
                    rows.setdefault(name, []).append(value)
                residual -= book[codes[..., d].reshape(-1)]
            for name, values in rows.items():
                add('physical/' + name, torch.stack(values, -1))

        for first in range(0, plan['images'], 2):
            latent = data['val'][first:first + 2].to(device)
            labels = data['val_labels'][first:first + 2].to(device)
            torch.manual_seed(plan['seed'] + 200 + first)
            q, codes = target_codes(profile, aux, latent)
            logits = model(codes, aux, labels, amp=True).float()
            add('current_soft_stochastic/ce', per_depth_ce(logits, q))
            add('current_soft_stochastic/entropy', image_sums(-(q * q.clamp_min(1.e-30).log()).sum(-1)))
            add('mode_labels_stochastic_prefix/ce', image_sums(
                -logits.log_softmax(-1).gather(-1, q.argmax(-1, keepdim=True)).squeeze(-1)))
            geometry(logits, q, codes, latent)
            del logits, q, codes
            if geometry_only:
                write(out / 'intervention-status.json', dict(phase='geometry_validation',
                    case=case['name'], images=first + len(labels), time=time.time()))
                continue
            q, codes = aux.quantizer.get_soft_codes(latent,
                temp=.5 if case['kind'] == 'rq' else profile.target_temperature(), stochastic=False)
            logits = model(codes, aux, labels, amp=True).float()
            add('current_soft_hard_prefix/ce', per_depth_ce(logits, q))
            add('current_soft_hard_prefix/entropy', image_sums(-(q * q.clamp_min(1.e-30).log()).sum(-1)))
            add('hard_labels_hard_prefix/ce', image_sums(
                -logits.log_softmax(-1).gather(-1, codes[..., None]).squeeze(-1)))
            del logits, q, codes
            if case['kind'] == 'laser':
                for name, temperatures in [('half_temperature', [t * .5 for t in profile.target_temperature()]),
                                            ('prior_rq_matched', calibration['proposed_temperatures'])]:
                    torch.manual_seed(plan['seed'] + 200 + first)
                    q, codes = target_codes(profile, aux, latent, temperatures)
                    logits = model(codes, aux, labels, amp=True).float()
                    add(name + '/ce', per_depth_ce(logits, q))
                    add(name + '/entropy', image_sums(-(q * q.clamp_min(1.e-30).log()).sum(-1)))
                    del logits, q, codes
            write(out / 'intervention-status.json', dict(phase='measuring', case=case['name'],
                images=first + len(labels), time=time.time()))
        matrices = {name:torch.cat(rows).numpy() for name, rows in metrics.items()}
        if geometry_only:
            old = np.load(out / f'{case["name"]}-interventions-per-image.npz')
            torch.testing.assert_close(torch.tensor(matrices['current_soft_stochastic/ce']),
                torch.tensor(old['current_soft_stochastic/ce']), rtol=1.e-6, atol=1.e-6)
            matrices = {**{key:old[key] for key in old.files if not key.startswith('physical/')}, **matrices}
        for mode in ('current_soft_stochastic', 'current_soft_hard_prefix', 'half_temperature', 'prior_rq_matched'):
            if mode + '/ce' in matrices:
                matrices[mode + '/kl'] = matrices[mode + '/ce'] - matrices[mode + '/entropy']
        np.savez_compressed(out / f'{case["name"]}-interventions-per-image.npz', **matrices)
        result = {name:dict(per_depth=values.mean(0).tolist(), mean=float(values.mean()))
                  for name, values in matrices.items()}
        reports[case['name']] = dict(case=case, metrics=result)
        write(out / 'interventions.json', dict(completed=False, cases=reports))
        del model, aux
        gc.collect(); torch.cuda.empty_cache()
    write(out / 'interventions.json', dict(completed=True, images=plan['images'], cases=reports,
        training_modified=False,
        physical_aggregation='Ratio of sums over 64 latent sites per image, then mean across images',
        physical_zero_denominator_checks_passed=True, limits=[
            'Fixed-model inference interventions; no causal conclusion about retraining with these targets.',
            'Hard-label CE and soft-label CE have different entropy floors.',
            'Physical metrics use full model probabilities, before sampling truncation.',
            'The mode-label/stochastic-prefix condition is a diagnostic denoising objective, not native maximum likelihood.',
            'Within-tokenizer normalized latent error is not a generated-image FID measurement.'], time=time.time()))
    write(out / 'intervention-status.json', dict(phase='completed', time=time.time()))


def summarize(plan):
    """Reuse measured whole-token metrics on CPU, without loading any checkpoint."""
    out = Path(plan['output'])
    reports = {case['name']: json.loads((out / f'{case["name"]}.json').read_text())
               for case in plan['cases']}
    summary_path = out / 'summary.json'
    archive = out / 'summary-superseded-factor-analysis.json'
    if summary_path.exists() and not archive.exists():
        write(archive, json.loads(summary_path.read_text()))
    prediction_path = out / 'predictability-summary.json'
    prediction = json.loads(prediction_path.read_text())
    for comparison in prediction['comparisons'].values():
        for metric in list(comparison):
            if metric not in ('ce', 'kl'):
                del comparison[metric]
    prediction['analysis_unit'] = 'whole_integer_token'
    prediction['cases'] = {name: {key: report['metrics'][key] for key in TOKEN_METRICS}
                           for name, report in reports.items()}
    prediction['time'] = time.time()
    write(prediction_path, prediction)
    noise_report = json.loads((out / 'noise.json').read_text())
    intervention_report = json.loads((out / 'interventions.json').read_text())
    audit_path = out / 'rq-compatibility-audit.json'
    models = {}
    for name, report in reports.items():
        metrics = report['metrics']
        intervention = intervention_report['cases'][name]['metrics']
        prefix_delta = (intervention['current_soft_hard_prefix/kl']['mean'] -
                        intervention['current_soft_stochastic/kl']['mean'])
        models[name] = dict(case=report['case'], metrics=prediction['cases'][name],
            hard_prefix_minus_stochastic_prefix_kl=prefix_delta)
    summary = dict(completed=True, analysis_unit='whole_integer_token',
        experiment_contract=dict(backbone='shared upstream RQTransformer',
            prediction='one categorical integer per spatial position and residual depth',
            baseline_loss='uniform cross entropy over complete codeword IDs',
            representation_change='frozen sparse codewords exposed as an integer codebook',
            codebook_vector_use='native RQ code embeddings and decoder lookup'),
        scope=dict(heldout_images=plan['images'], unigram_fit_images=plan['fit_images'],
            reused_existing_inference=True, additional_gpu_work=False),
        models=models, comparisons=prediction['comparisons'],
        noise={name: {key: values[key] for key in (
            'temperatures', 'profiles', 'final_sampled_to_hard_latent_error_ratio',
            'sample_disagreement_with_full_hard_trajectory_by_depth',
            'pairwise_stochastic_code_disagreement_by_depth', 'pixel_probe')}
            for name, values in noise_report['cases'].items()},
        findings=[
            'Whole-token cross entropy and target entropy are measured separately; LASER targets have lower entropy than the original RQ targets on this sample.',
            'The LASER epoch-three checkpoint improves whole-token KL at every depth compared with epoch two.',
            'Changing only the teacher-forced prefix from stochastic to hard barely changes LASER KL at these fixed model weights.',
            'Top-k coverage is measured on whole integer IDs before nucleus filtering; this is a prediction diagnostic, not a sampler quality score.',
            'The current run uses depth-weighted CE and top-k 2048; the original-RQ recipe uses uniform CE and top-k 256.'],
        next_tests=[
            dict(priority=1, candidate='Compare whole-token predictability and code usage using the existing uniform-CE LASER control and original RQ at matched training steps.', status='diagnostics'),
            dict(priority=2, candidate='Compare top-k 256 and 2048 at identical model weights, seeds, labels, temperature, and top-p; assess generated-image quality.', status='recommendation_only'),
            dict(priority=3, candidate='Test a calibrated target-noise change while holding integer codebook, uniform CE, transformer, and evaluation sampler fixed.', status='recommendation_only')],
        limits=prediction['limits'] + [
            'The 256-image unigram fit is too small for a decisive vocabulary-usage or marginal-entropy comparison.',
            'A numeric target temperature is specific to codebook distance units; copying RQ temperature 0.5 does not match its corruption level.',
            'These diagnostics do not establish intrinsic token unpredictability or a causal explanation for FID.'],
        validation=dict(all_laser_inputs_targets_and_codes_identical=prediction['all_laser_inputs_targets_and_codes_identical'],
            native_hard_code_parity_passed=noise_report['native_hard_code_parity_passed'],
            source_token_metrics_preserved=True, diagnostics_training_updates=0,
            training_configuration_modified=False), time=time.time())
    if audit_path.exists():
        summary['rq_compatibility_audit'] = json.loads(audit_path.read_text())
    write(summary_path, summary)
    print(json.dumps(dict(report=str(summary_path), analysis_unit=summary['analysis_unit'],
        cases=list(models), training_modified=False)), flush=True)


def main():
    global torch, np, OmegaConf, RQRecipe, JointLaserRecipe, TarImageNet, create_transforms, load_config, ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['noise', 'predict', 'intervene', 'summarize'])
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--device', default='cuda:3')
    parser.add_argument('--memory-gib', type=float, default=16.)
    parser.add_argument('--geometry-only', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.phase == 'summarize':
        summarize(plan)
        return
    plan['geometry_only'] = args.geometry_only
    wait_for(Path(plan['local']) / 'assets-ready.json')
    ROOT = Path('/workspace/Projects/laser')
    runtime = Path(plan['runtime'])
    os.environ['RQ_UPSTREAM_ROOT'] = str(runtime / 'upstream')
    sys.path[:0] = [str(runtime / 'source'), str(runtime / 'upstream'), str(runtime)]
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from src import rq_upstream
    from src.training.laser_joint_recipe import JointLaserRecipe
    from src.training.rq_imagenet_recipes import RQRecipe
    from src.training.rq_imagenet_runtime import TarImageNet
    from rqvae.img_datasets.transforms import create_transforms
    from rqvae.utils.config import load_config
    torch.set_num_threads(4)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(args.memory_gib * 2**30 /
        torch.cuda.get_device_properties(device).total_memory, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # This changes only this diagnostic process's autocast cache, keeping model arithmetic FP16.
    rq_upstream.RQTransformer.forward.__globals__['autocast'] = lambda enabled=True: torch.autocast(
        'cuda', enabled=enabled, dtype=torch.float16, cache_enabled=False)
    torch.manual_seed(plan['seed'])
    with torch.no_grad():
        {'noise':noise, 'predict':predict, 'intervene':intervene}[args.phase](plan, device)


if __name__ == '__main__':
    main()
