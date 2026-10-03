#!/usr/bin/env python3
"""Freeze the audited Church runtime for a matched normalized-noise experiment."""
import hashlib
import json
from pathlib import Path
import shutil
import time

REPO = Path('/workspace/Projects/laser')
ROOT = Path('/mnt/laser-church/normalized-noise-comparison')
OLD = REPO/'outputs/church-rqrecipe300-dropout02-b2048-4h100-20260924'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def replace(source, old, new):
    assert source.count(old) == 1, old[:100]
    return source.replace(old, new)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    import torch
    ROOT.mkdir(parents=True, exist_ok=True)
    assert not (ROOT/'prepared.json').exists(), 'Do not overwrite a launched experiment'
    runtime = ROOT/'runtime'
    if not runtime.exists():
        shutil.copytree(OLD/'runtime', runtime)
    trainer = runtime/'src/training/rqtransformer.py'
    source = trainer.read_text()
    if '--coeff-target-space' not in source:
        source = replace(source, '    p.add_argument("--coeff-scales", type=float, nargs="+")',
            '    p.add_argument("--coeff-scales", type=float, nargs="+")\n'
            '    p.add_argument("--coeff-target-space", choices=("auto", "normalized", "physical"), default="auto")')
        source = replace(source, 'soft_target_physical=args.coeff_scales is not None,',
            'soft_target_physical=(args.coeff_target_space == "physical" or\n'
            '                       (args.coeff_target_space == "auto" and args.coeff_scales is not None)),')
        trainer.write_text(source)
    assert 'gc.collect()' in source[source.index('def atomic_torch_save'):source.index('def snapshot_checkpoint')]
    cache_path = Path('/mnt/laser-church/assets/compound-cache.pt')
    cache = torch.load(cache_path, weights_only=True, mmap=True)
    meta = cache['meta']
    assert cache['atoms'].shape == cache['coeffs'].shape == (126227, 8, 8, 4)
    assert meta['format'] == 'laser_compound_pairs_v1' and not meta['clip_coefficients']
    assert meta['encoder_precision'] == 'fp32' and 'coeff_bin_centers' not in meta
    assert digest(cache_path) == '4c03555ebedca1bf7b74f1db30e91752586889337a58547e856416c3aeeae586'
    tokenizer = Path('/mnt/laser-church/assets/tokenizer.pt')
    assert digest(tokenizer) == meta['stage1_checkpoint_sha256'] == '762c51a10267ed6fa55709ff0d6cf997940d21056b16c7a0a0399b3ebf93868d'
    cache_record = dict(passed=True, images=126227, cache=str(cache_path), cache_sha256=digest(cache_path),
                        checkpoint=str(tokenizer), checkpoint_sha256=meta['stage1_checkpoint_sha256'], metadata=meta)
    overrides = dict(checkpoint=str(tokenizer), token_cache=str(cache_path), coeff_max=3., coeff_scale=meta['coeff_scale'],
                     coeff_scales=meta['coeff_scales'], coeff_target_space='normalized', coeff_target_mode='soft')
    driver = (OLD/'train.py').read_text()
    driver = replace(driver, "choices=['benchmark-a2', 'benchmark-a4', 'train']", "choices=['benchmark-a4', 'train']")
    driver = replace(driver, "assert int(os.environ['WORLD_SIZE']) == 4", "assert int(os.environ['WORLD_SIZE']) == 2")
    driver = replace(driver, '2048 / (4 * accumulation)', '2048 / (2 * accumulation)')
    driver = replace(driver, '/mnt/laser-church/dropout-experiment/checkpoints', str(ROOT/'checkpoints'))
    driver = replace(driver, "payload['config']['experiment'] = request", "payload['config']['experiment'] = request\n        assert payload['config']['coeff_target_space'] == 'normalized'")
    driver = replace(driver, "        del restored", "        assert restored['config']['coeff_target_temperature'] == request['coefficient_target_temperature']\n        assert restored['config']['coeff_target_space'] == 'normalized'\n        assert restored['config']['coeff_scales'] == config['coeff_scales']\n        del restored")
    driver = replace(driver, "        if rank == 0:\n            initialization_record", "        if rank == 0:\n            state_hash = hashlib.sha256()\n            for name, value in model.state_dict().items():\n                state_hash.update(name.encode())\n                state_hash.update(value.detach().contiguous().numpy().tobytes())\n            initialization_record")
    driver = replace(driver, "                parameters=sum(p.numel() for p in model.parameters()), dropouts=dropouts,", "                parameters=sum(p.numel() for p in model.parameters()), dropouts=dropouts,\n                full_state_sha256=state_hash.hexdigest(),")
    start = driver.index("                centers = torch.load(cache['cache']")
    end = driver.index('                self._verified_cache = True', start)
    driver = driver[:start] + '''                assert torch.equal(self.coeff_bins, torch.linspace(-3., 3., 2048).to(self.coeff_bins.device))
                assert torch.equal(self.coeff_scales, torch.tensor(config['coeff_scales'], device=self.coeff_scales.device))
                assert not self.clamp_coeffs and not self.soft_target_physical
                assert coeffs.dtype == torch.float32 and torch.isfinite(coeffs).all()
                assert kwargs['temp'] == request['coefficient_target_temperature']
                assert kwargs.get('stochastic', True) and not kwargs.get('hard', False)
                assert not any(p.requires_grad for p in self.parameters())
                sample = coeffs[:2]
                _, q = super().compound_coeff_ids(sample, stochastic=False, temp=kwargs['temp'])
                expected = (-(sample[..., None]-self.coeff_bins).square()/kwargs['temp']).softmax(-1)
                assert torch.equal(q, expected)
                atomic_json(output / f'codec-rank{rank}.json', dict(passed=True, frozen_tokenizer=True,
                    coefficient_target_space='normalized', coefficient_target_temperature=kwargs['temp'],
                    kernel_bitwise_verified=True, no_clipping=True, atom_support='deterministic OMP', rank=rank))
''' + driver[end:]
    driver = replace(driver, "probe, temperature=.125)", "probe, temperature=request['coefficient_target_temperature'])")
    driver = replace(driver, "group='church-rqrecipe-dropout-20260924'", "group='church-detomp-normalized-noise-20260924'")
    start = driver.index("        recovery = BASE / 'memory-recovery.json'")
    end = driver.index('        original_finish = wb.finish', start)
    driver = driver[:start] + driver[end:]
    start = driver.index("        assets = read(output_dir / 'token_cache_artifact.json')")
    end = driver.index("        if not (output / 'provenance-upload.json').exists():", start)
    driver = driver[:start] + '''        tokenizer = api.artifact('helloimlixin-rutgers/laser/church-laser-stage1-selection-20260920-selected-checkpoints:v0')
        assert tokenizer.state == 'COMMITTED'
        wb.use_artifact(tokenizer)
        receipt = output / 'token_cache_artifact.json'
        if receipt.exists():
            artifact = api.artifact(read(receipt)['cache_artifact'])
            assert artifact.state == 'COMMITTED'
            wb.use_artifact(artifact)
        else:
            artifact = wandb.Artifact(wb.id+'-deterministic-token-cache', type='dataset', metadata=cache)
            artifact.add_file(cache['cache'], name='compound-cache.pt', policy='immutable', skip_cache=True)
            logged = wb.log_artifact(artifact, aliases=['latest'])
            logged.wait()
            committed = api.artifact(logged.qualified_name)
            assert committed.state == 'COMMITTED'
            entry = committed.manifest.entries['compound-cache.pt']
            import base64
            with Path(cache['cache']).open('rb') as stream:
                expected_md5 = base64.b64encode(hashlib.file_digest(stream, 'md5').digest()).decode()
            assert entry.digest == expected_md5 and entry.size == Path(cache['cache']).stat().st_size
            atomic_json(receipt, dict(verified_online=True, cache_artifact=logged.qualified_name,
                tokenizer_artifact=tokenizer.qualified_name, cache_sha256=cache['cache_sha256']))
''' + driver[end:]
    driver = replace(driver, "'source-manifest.json', 'runtime-source.tar.gz', 'heldout-probe.pt', 'heldout-probe.json']", "'source-manifest.json', 'runtime-source.tar.gz', 'heldout-probe.pt', 'heldout-probe.json', 'audit-numerical.json']")
    driver = replace(driver, "len(saved['rng_state_by_rank']) == 4", "len(saved['rng_state_by_rank']) == 2")
    benchmark = (OLD/'benchmark_fid.py').read_text()
    benchmark = benchmark.replace('all five H200s', 'two H100s')
    benchmark = replace(benchmark, "coeff_scales=[1.]*4, soft_target_physical=True", "coeff_scales=config['coeff_scales'], soft_target_physical=False")
    benchmark = replace(benchmark, "coeff_bin_centers=config['coeff_bin_centers']", "coeff_bin_centers=None")
    benchmark = benchmark.replace('coeff_top_p=1.', 'coeff_top_p=.85')
    benchmark = benchmark.replace('[256, 512, 1024]', '[1024]')
    benchmark = benchmark.replace('4 * batch_size', '2 * batch_size').replace('4*batch_size', '2*batch_size')
    variants = []
    for temperature, gpus in [(.25, '0,1'), (.5, '2,3')]:
        run_id = f'church-detomp-normt{round(temperature*100):03d}-b2048-2h100-20260924'
        base = ROOT/run_id
        base.mkdir(exist_ok=True)
        for name, target in [('runtime', runtime), ('heldout-probe.pt', ROOT/'heldout-probe.pt'),
                             ('heldout-probe.json', ROOT/'heldout-probe.json'), ('audit-numerical.json', ROOT/'audit-numerical.json'),
                             ('validation.json', ROOT/'validation.json')]:
            if not (base/name).is_symlink():
                (base/name).symlink_to(target)
        link = REPO/'outputs'/run_id
        if not link.exists():
            link.symlink_to(base, target_is_directory=True)
        for name in ['official_metrics.py', 'heldout_monitor.py', 'checkpoint_staging.py', 'official-fid-provenance.json']:
            shutil.copyfile(OLD/name, base/name)
        shutil.copytree(OLD/'official-metrics', base/'official-metrics', dirs_exist_ok=True)
        (base/'train.py').write_text(driver)
        (base/'benchmark_fid.py').write_text(benchmark)
        shutil.copyfile(REPO/'scripts/tools/supervise_church_normalized_comparison.py', base/'supervise.py')
        config = read(OLD/'baseline-request.json')['config']
        for name in ['coeff_bin_centers', 'experiment', 'compound_cache_identity', 'preview_sampling_settings', 'coefficient_quantizer', 'world_size']:
            config.pop(name, None)
        config.update(overrides, coeff_target_temperature=temperature, temp=temperature, coeff_top_p=.85,
                      stochastic_atom_supports=False, compound_cache_variants_per_site=1)
        write(base/'baseline-request.json', dict(config=config))
        (base/'cache').mkdir(exist_ok=True)
        write(base/'cache/complete.json', cache_record)
        write(base/'cache/codec-overrides.json', overrides)
        plan = read(OLD/'plan.json')
        for key in ['historical_control', 'historical_differences', 'memory_recovery', 'accumulation_override']:
            plan.pop(key, None)
        plan.update(run_id=run_id, objective='Matched deterministic OMP K4 normalized coefficient noise comparison',
                    coefficient_target_temperature=temperature, coefficient_target_space='normalized',
                    world_size=2, cuda_visible_devices=gpus, accumulation_steps=4, microbatch_per_gpu=256,
                    fid_batch_size=1024, created_unix=time.time(), checkpoint_policy='Full last and best FID states, optimizer, scheduler, both rank RNG states; online digest verification',
                    normalization='Per-depth training maximum absolute coefficient / 3; not RMS normalization',
                    coeff_scales=meta['coeff_scales'], atom_support='Deterministic OMP, no support bank or atom target noise',
                    coefficient_kernel='q(j|c,d) proportional to exp(-(c/s_d-bin_j)^2 / temperature)',
                    coefficient_bins='2048 uniformly spaced normalized centers in [-3,3]',
                    coefficient_sampling='Fresh categorical history draw each training visit; full soft-target CE',
                    controls='Same frozen stage1, cache, initialization, seed, architecture, optimizer, dropout, batch, schedules, sampler and FID protocol',
                    physical_noise_note='Finite bins truncate the kernel near boundaries; interior normalized sigma is sqrt(temperature/2).')
        plan['laser_specific'].update(coefficient_target_temperature=temperature, stochastic_omp_temperature=0.,
                                     stochastic_support_bank_variants=1, coefficient_top_p=.85, atom_top_p=1.)
        write(base/'plan.json', plan)
        for mode in ['benchmark-a4', 'train']:
            (base/mode).mkdir(exist_ok=True)
            storage = ROOT/'checkpoints'/run_id/mode
            storage.mkdir(parents=True, exist_ok=True)
            (base/mode/'checkpoints').symlink_to(storage, target_is_directory=True)
        variants.append(dict(run_id=run_id, base=str(base), temperature=temperature, gpus=gpus))
    write(ROOT/'prepared.json', dict(variants=variants, source_runtime=str(runtime), created_unix=time.time()))
    link = REPO/'outputs/church-detomp-normalized-comparison-20260924'
    if not link.exists():
        link.symlink_to(ROOT, target_is_directory=True)
    print(json.dumps(dict(root=str(ROOT), variants=variants)))


if __name__ == '__main__':
    main()
