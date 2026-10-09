"""Prepare and supervise a scratch run; retain its twenty verified initial updates."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
SUB_BIN = os.environ.get('LASER_SUB_BIN_TARGETS') == '1'
FFHQ_ADAPT = os.environ.get('LASER_FFHQ_DECODER_OBJECTIVE') == '1'
REQUESTED_SIGMA_BINS = os.environ.get('LASER_COEFFICIENT_SIGMA_BINS')
SIGMA_BINS = float(REQUESTED_SIGMA_BINS) if REQUESTED_SIGMA_BINS is not None else 25.5875
assert math.isfinite(SIGMA_BINS) and SIGMA_BINS > 0
assert not (SUB_BIN and REQUESTED_SIGMA_BINS is not None)
assert not FFHQ_ADAPT or (REQUESTED_SIGMA_BINS is not None and not SUB_BIN)
TARGET_TEMPERATURE = 2 * ((.01 if SUB_BIN else SIGMA_BINS) * (6 / 2047))**2
if FFHQ_ADAPT:
    BASE = Path(f'/tmp/laser-imagenet-ffhq-v4-sigma{SIGMA_BINS:g}bins-scratch-20261009')
    RUN_ID = f'imagenet-rfid421-ffhq-v4-sigma{SIGMA_BINS:g}bins-scratch-4h200-20261009'
elif REQUESTED_SIGMA_BINS is not None:
    BASE = Path(f'/tmp/laser-imagenet-sigma{SIGMA_BINS:g}bins-scratch-20261008')
    RUN_ID = f'imagenet-rfid421-sigma{SIGMA_BINS:g}bins-scratch-4h200-20261008'
else:
    BASE = Path('/tmp/laser-imagenet-repaired-subbin-scratch-20261008' if SUB_BIN else
                '/tmp/laser-imagenet-repaired-scratch-20261008')
    RUN_ID = ('imagenet-rfid421-repaired-subbin-scratch-4h200-20261008' if SUB_BIN else
              'imagenet-rfid421-repaired-scratch-4h200-20261008')
OUT = ROOT/'outputs'/RUN_ID


def record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str)+'\n')
    temporary.replace(path)


def prepare():
    import yaml
    BASE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if SUB_BIN or REQUESTED_SIGMA_BINS is not None:
        previous_base = Path('/tmp/laser-imagenet-repaired-scratch-20261008')
        previous_out = ROOT/'outputs/imagenet-rfid421-repaired-scratch-4h200-20261008'
        assert json.loads((previous_base/'imagenet/training-ready.json').read_text())['passed']
        assert json.loads((previous_out/'validation-reference-verification.json').read_text())['passed']
        for name in ('inputs', 'imagenet', 'torch-cache'):
            (BASE/name).symlink_to(previous_base/name, target_is_directory=True)
        for name in ('validation-reference-verification.json', 'diagnosis.json',
                     'environment.txt', 'tokenizer-gpu-preflight.json', 'test-verification.json'):
            shutil.copyfile(previous_out/name, OUT/name)
    if SUB_BIN:
        record(OUT/'sub-bin-policy.json', dict(interpretation='User constraint applies to coefficient target noise',
            target_distribution='truncated Gaussian jitter before nearest-bin quantization',
            sigma_bins=.01, hard_cap_bins=.025, actual_rms_bins=.009545974862347768,
            normalized_bin_width=6/2047, maximum_in_range_quantized_target_error_bins=.5251,
            model_error_guaranteed=False, coefficient_cross_entropy_units='nats',
            prediction_error_metrics_units='bin widths against clean continuous coefficients',
            saturation='Out-of-range coefficients saturate and may exceed one-bin quantization error; measured separately',
            supersedes=previous_out.name, fresh_stage2=True, reused_data_and_frozen_tokenizer_only=True))
    elif REQUESTED_SIGMA_BINS is not None:
        record(OUT/'coefficient-noise-policy.json', dict(
            requested_sigma_bins=SIGMA_BINS, normalized_bin_width=6/2047,
            coefficient_quantizer='uniform', coefficient_bin_centers=2048,
            coefficient_bin_range=[-3., 3.],
            bin_policy_reference_run='ffhqcmp0804205803',
            normalized_sigma=SIGMA_BINS*6/2047, temperature=TARGET_TEMPERATURE,
            target_distribution='Gaussian kernel evaluated on uniform bin centers',
            edge_behavior='Finite coefficient range truncates the Gaussian; requested sigma is the nominal kernel width',
            supersedes=('imagenet-rfid421-sigma200bins-scratch-4h200-20261008' if FFHQ_ADAPT else
                        'imagenet-rfid421-repaired-subbin-scratch-4h200-20261008'),
            supersedes_sub_bin_noise_constraint=True, fresh_stage2=True,
            reused_data_and_frozen_tokenizer_only=True))
    source = BASE/'source'
    assert not source.exists(), 'Do not overwrite a frozen runtime'
    source.mkdir()
    for name in ('src', 'configs', 'runtime', 'third_party'):
        shutil.copytree(ROOT/name, source/name, copy_function=shutil.copyfile,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '*.pt', '*.pth', '*.ckpt'))
    shutil.copyfile(ROOT/'train.py', source/'train.py')
    shutil.copyfile(ROOT/'scripts/tools/imagenet_repaired_scratch_entry.py', BASE/'entry.py')
    for name in ('checkpoint-staging', 'checkpoint-upload-cache', 'wandb', 'torch-cache'):
        (BASE/name).mkdir(exist_ok=True)
    config = yaml.safe_load((ROOT/'configs/stage2/imagenet-rfid421-tiny-jitter-fresh-8h100-20261008.yaml').read_text())
    options = config['options']
    options.update(checkpoint=str(BASE/'inputs/stage1-tokenizer.pt'), data=str(BASE/'imagenet'),
        token_cache=None, output=str(OUT/'train'), checkpoint_dir=str(OUT/'train/checkpoints'),
        coeff_target_space='normalized', coeff_target_temperature=TARGET_TEMPERATURE,
        compound_tokens=False, compound_pair_autoregressive=False, physical_pair_context=True,
        batch_size=128, total_batch_size=2048, lr=.0005, lr_schedule='cosine', lr_schedule_epochs=100,
        min_lr=0., warmup_epochs=0., epochs=100, atom_temperature=.9, atom_top_k=0, atom_top_p=.9,
        coeff_temperature=1., coeff_top_k=0, coeff_top_p=.85, metric_backend='original-rqvae',
        fid_reference_stats=str(BASE/'inputs/imagenet_256_train.npz'), fid_batch_size=128,
        fid_every=2, fid_seed=261001, save_step_freq=250, keep_best_checkpoints=1,
        model_only_best_checkpoints=False, sample_grid_every=626,
        wandb_id=RUN_ID, wandb_name=(f'ImageNet rFID4.21 K4 | sigma {SIGMA_BINS:g} bins | scratch | 4 H200' if REQUESTED_SIGMA_BINS is not None else
                                    'ImageNet rFID4.21 K4 | repaired scratch physical pairs + sub-bin jitter | 4 H200' if SUB_BIN else
                                    'ImageNet rFID4.21 K4 | repaired scratch physical pairs + fresh views | 4 H200'),
        resume=False, resume_checkpoint=None, init_stage2_checkpoint=None, max_optimizer_steps=0)
    if FFHQ_ADAPT:
        options.update(compound_tokens=True, compound_pair_autoregressive=True,
            physical_pair_context=False, compound_refiner_layers=0,
            compound_micro_transformer_layers=2, compound_depth_specific_coeff_heads=True,
            compound_distribution_geometry=True, geometry_top_k=4,
            atom_loss_weight=1.5, geometry_loss_weight=.05,
            geometry_start_epoch=2., geometry_warmup_epochs=3.,
            coeff_regression_weight=0., coeff_crps_weight=0.,
            wandb_name=f'ImageNet rFID4.21 K4 | FFHQ8.17 decoder+objective | sigma {SIGMA_BINS:g} bins | scratch | 4 H200')
        record(OUT/'ffhq-adaptation-policy.json', dict(
            reference_run='helloimlixin-rutgers/laser/ffhqcmp0804205803',
            archive_sha256='9ba1b49b4e5e339f0076bebee6fbac5629f6c391601de467019a3723c9d3e33f',
            decoder='archived compound-v4, full pair history, micro-transformer2, depth-specific coefficient heads',
            objective='archived atom-weighted CE plus normalized pair/spatial geometry against clean physical contributions',
            atom_weight=1.5, coefficient_weight=1., geometry_weight=.05,
            geometry_start_epoch=2., geometry_warmup_epochs=3., geometry_top_k=4,
            geometry_formulation='expected atom vector times expected coefficient, as in FFHQ8.17 archive',
            training_seen_atom_mask=False, sampling_distinct_atoms=True,
            adaptations=dict(conditional_classes=1000, atoms=16384, sparsity_level=4,
                model_preset='imagenet-1400m', embed_dim=1536, body_layers=42, head_layers=6,
                frozen_stage1='ImageNet rFID4.210914', coefficient_sigma_bins=SIGMA_BINS),
            fresh_stage2_weights=True, fresh_optimizer=True,
            supersedes='imagenet-rfid421-sigma200bins-scratch-4h200-20261008'))
    canonical = '# @package _global_\n'+yaml.safe_dump(config, sort_keys=False)
    (OUT/'train.yaml').write_text(canonical)
    (ROOT/'configs/stage2'/f'{RUN_ID}.yaml').write_text(canonical)
    for phase in ('preflight', 'production'):
        chosen = dict(options)
        if phase == 'preflight':
            chosen.update(max_optimizer_steps=20, fid_every=0, save_step_freq=0,
                sample_grid_every=0, upload_checkpoints=False)
        else:
            chosen.update(resume=True, resume_checkpoint=str(OUT/'train/checkpoints/last.pt'))
        (BASE/f'{phase}.yaml').write_text(yaml.safe_dump(dict(config, options=chosen), sort_keys=False))
    manifest = {str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in source.rglob('*') if p.is_file()}
    manifest['entry.py'] = hashlib.sha256((BASE/'entry.py').read_bytes()).hexdigest()
    record(OUT/'source-manifest.json', manifest)
    with tarfile.open(OUT/'source-code.tar.gz', 'w:gz') as bundle:
        bundle.add(source, arcname='source')
        for name in ('entry.py', 'preflight.yaml', 'production.yaml'):
            bundle.add(BASE/name, arcname=name)
        for name in ('launch_imagenet_repaired_scratch.py', 'prepare_imagenet_repaired_scratch.py'):
            bundle.add(ROOT/'scripts/tools'/name, arcname=name)
    record(OUT/'plan.json', dict(run=RUN_ID, stage2_initialization='scratch', start_step=0,
        frozen_stage1_sha256='dd28db9306d526bdc9fbf8016403e106c4f317fd8f5c04792f0953e53310fdab',
        world_size=4, gpu='H200', global_batch=2048, microbatch=128, accumulation=4,
        epochs=100, steps_per_epoch=626, cosine_total_steps=62600, lr=.0005, min_lr=0.,
        decoder='ffhq-v4-archived-compound' if FFHQ_ADAPT else 'physical-pair-scalar',
        objective='ffhq-v4-archived weighted CE plus clean geometry' if FFHQ_ADAPT else 'scalar cross-entropy',
        augmentation='fresh crop and flip per image/epoch',
        coefficient_target_temperature=options['coeff_target_temperature'],
        coefficient_target_distribution='bounded continuous jitter before quantization' if SUB_BIN else 'Gaussian on centers',
        coefficient_sigma_bins=.01 if SUB_BIN else SIGMA_BINS,
        coefficient_noise_cap_bins=.025 if SUB_BIN else None,
        primary_fid='original-rqvae features, full training reference, generated50k',
        generation_metric_keys=['eval/fid_original_train50k', 'eval/inception_score'],
        companion_fid=[],
        first_fid_epoch=1, fid_every=2, preflight_updates_retained=20,
        limitations=['Target noise changed at user request; its FID benefit is unverified.',
            'Historical K2 TorchMetrics validation FID uses a different protocol.',
            'Fresh-run final FID is unknown until evaluated.']))
    print(json.dumps(dict(prepared=True, runtime=str(source))), flush=True)


def verify_preflight():
    import torch
    os.environ['LASER_CHECKPOINT_UPLOAD_CACHE_DIR'] = str(BASE/'checkpoint-upload-cache')
    os.environ['LASER_CHECKPOINT_IMMUTABLE_FILES'] = '1'
    sys.path[:0] = [str(BASE/'source'), str(BASE/'source/runtime')]
    from src.training.k4_checkpoint_io import _checkpoint_upload_source
    checkpoint = _checkpoint_upload_source(OUT/'train/checkpoints/last.pt')
    payload = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    assert payload['global_step'] == 20 and payload['batch_idx'] == 80
    assert payload['checkpoint_world_size'] == len(payload['rng_state_by_rank']) == 4
    assert payload['scheduler']['last_epoch'] == 20 and payload['scheduler']['T_max'] == 62600
    expected_lr = .0005*(1+math.cos(math.pi*20/62600))/2
    assert math.isclose(payload['optimizer']['param_groups'][0]['lr'], expected_lr, rel_tol=1e-10)
    assert {int(s['step']) for s in payload['optimizer']['state'].values()} == {20}
    for value in payload['state_dict'].values():
        assert torch.isfinite(value).all()
    for state in payload['optimizer']['state'].values():
        assert all(torch.isfinite(value).all() for value in state.values() if isinstance(value, torch.Tensor))
    proofs = []
    for rank in range(4):
        startup = json.loads((OUT/f'verification/preflight/startup-rank{rank}.json').read_text())
        targets = json.loads((OUT/f'verification/preflight/targets-rank{rank}.json').read_text())
        assert startup['empty_optimizer'] and startup['from_scratch'] and targets['passed']
        if SUB_BIN:
            assert targets['coefficient_noise_cap_bins'] == .025
            assert targets['maximum_in_range_quantized_target_error_bins'] < 1
            assert targets['temperature_floor_bypassed']
        else:
            assert math.isclose(targets['coefficient_sigma_bins'], SIGMA_BINS, rel_tol=1e-10)
            assert math.isclose(targets['center_measured_sigma_bins'], SIGMA_BINS, rel_tol=1e-4)
        proofs.append(dict(rank=rank, startup=startup, targets=targets))
    diagnostics = json.loads((OUT/'verification/preflight/coefficient-metrics-step0000020.json').read_text())
    assert diagnostics['globally_reduced'] and diagnostics['continuous_clean_targets']
    record(OUT/'preflight-verification.json', dict(passed=True, global_step=20, fresh_optimizer_verified=True,
        finite_model_and_adam=True, rng_ranks=4, cosine_total_steps=62600, lr=expected_lr,
        checkpoint_bytes=checkpoint.stat().st_size, proofs=proofs, time=time.time()))


def run(resume_only=False):
    with (BASE/'supervisor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = json.loads((OUT/'source-manifest.json').read_text())
        assert all(hashlib.sha256((BASE/name).read_bytes()).hexdigest() == expected for name, expected in manifest.items())
        while not (BASE/'imagenet/training-ready.json').exists():
            preparation = json.loads((OUT/'preparation-status.json').read_text())
            if preparation['phase'] == 'failed':
                raise RuntimeError(preparation['error'])
            record(OUT/'launch-status.json', dict(phase='waiting_for_verified_data', preparation=preparation, time=time.time()))
            time.sleep(5)
        while not (OUT/'validation-reference-verification.json').exists():
            reference = json.loads((OUT/'reference-preparation-status.json').read_text())
            if reference['phase'] == 'failed':
                raise RuntimeError(reference['error'])
            record(OUT/'launch-status.json', dict(phase='waiting_for_validation_references', reference=reference, time=time.time()))
            time.sleep(5)
        env = dict(os.environ, WANDB_API_KEY=Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip(),
            LASER_SUB_BIN_TARGETS='1' if SUB_BIN else '0',
            LASER_FFHQ_DECODER_OBJECTIVE='1' if FFHQ_ADAPT else '0',
            LASER_RUN_BASE=str(BASE), LASER_PERSISTENT_BASE=str(OUT), LASER_ACCUMULATION='4',
            LASER_CHECKPOINT_STAGING_DIR=str(BASE/'checkpoint-staging'),
            LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(BASE/'checkpoint-upload-cache'), LASER_CHECKPOINT_IMMUTABLE_FILES='1',
            PYTHONPATH=str(BASE/'source')+':'+str(BASE/'source/runtime'), CUDA_VISIBLE_DEVICES='0,1,2,3',
            TORCH_HOME=str(BASE/'torch-cache'), TORCHINDUCTOR_CACHE_DIR=str(BASE/'inductor-cache'),
            TORCHINDUCTOR_COMPILE_THREADS='4', WANDB_DIR=str(BASE/'wandb'), WANDB_CACHE_DIR=str(BASE/'wandb-cache'),
            WANDB_DATA_DIR=str(BASE/'wandb-data'), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
            PYTHONUNBUFFERED='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', NCCL_NVLS_ENABLE='0')
        if REQUESTED_SIGMA_BINS is not None:
            env['LASER_COEFFICIENT_SIGMA_BINS'] = str(SIGMA_BINS)
        child = None
        stopping = False
        def stop(*_):
            nonlocal stopping
            stopping = True
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        if resume_only:
            assert json.loads((OUT/'preflight-verification.json').read_text())['passed']
        for phase in (('production',) if resume_only else ('preflight', 'production')):
            if stopping:
                break
            if phase == 'preflight':
                assert not (OUT/'train/checkpoints/last.pt').exists(), 'Fresh start cannot consume an old checkpoint'
            env['LASER_PHASE'] = phase
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=4',
                str(BASE/'entry.py'), '--config', str(BASE/f'{phase}.yaml')]
            with (OUT/'train.log').open('a') as log:
                child = subprocess.Popen(command, cwd=BASE/'source', env=env, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
                record(OUT/'launch-status.json', dict(phase=phase, running=True, torchrun_pid=child.pid,
                    supervisor_pid=os.getpid(), wandb_id=RUN_ID, time=time.time()))
                result = child.wait()
            if result:
                record(OUT/'launch-status.json', dict(phase='failed', failed_phase=phase, returncode=result, time=time.time()))
                raise SystemExit(result)
            if phase == 'preflight':
                verify_preflight()
        record(OUT/'launch-status.json', dict(phase='stopped_resumable' if stopping else 'completed', time=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run', 'verify-preflight', 'resume-production'])
    args = parser.parse_args()
    {'prepare':prepare, 'run':run, 'verify-preflight':verify_preflight,
     'resume-production':lambda:run(resume_only=True)}[args.action]()
