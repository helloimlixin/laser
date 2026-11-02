"""Stage a resumable, gradient-balanced ImageNet objective change."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import time

POLICY_VERSION = 'gradient-balanced-ce-no-geometry-v1'


def make_entry(original):
    additions = """    LOSS_POLICY = json.loads((OUT/'loss-balance-policy.json').read_text())
    assert LOSS_POLICY['version'] == 'gradient-balanced-ce-no-geometry-v1'
    ATOM_LOSS_WEIGHT = float(LOSS_POLICY['atom_weight'])
    assert .25 <= ATOM_LOSS_WEIGHT <= 1. and LOSS_POLICY['coefficient_weight'] == 1.
    assert LOSS_POLICY['geometry_weight'] == 0.
"""
    changes = {
        "VERIFY = OUT/'verification'/PHASE":
            "VERIFY = Path(os.environ.get('LASER_LOSS_BALANCE_VERIFY_DIR', str(OUT/'verification'/PHASE)))",
        '    ffhq_adapter.verify_archive()\n': '    ffhq_adapter.verify_archive()\n'+additions,
        'assert ARGS.compound_depth_specific_coeff_heads and ARGS.compound_distribution_geometry':
            'assert ARGS.compound_depth_specific_coeff_heads and not ARGS.compound_distribution_geometry',
        'assert ARGS.atom_loss_weight == 1.5 and ARGS.geometry_loss_weight == .05':
            'assert ARGS.atom_loss_weight == ATOM_LOSS_WEIGHT and ARGS.geometry_loss_weight == 0',
        '            ARGS.ffhq_decoder_archive_sha256 = ffhq_adapter.ARCHIVE_SHA256':
            "            ARGS.loss_balance_policy_version = LOSS_POLICY['version']\n"
            '            ARGS.loss_balance_policy = LOSS_POLICY\n'
            '            ARGS.coefficient_loss_weight = 1.\n'
            '            ARGS.ffhq_decoder_archive_sha256 = ffhq_adapter.ARCHIVE_SHA256',
        'torch.load = load_checkpoint\n':
            "torch.load = load_checkpoint\n\n"
            "native_rng_restore = training.restore_rank_rng_state\n"
            "def verified_rng_restore(payload, device, seed):\n"
            "    restored = native_rng_restore(payload, device, seed)\n"
            "    if PHASE == 'production':\n"
            "        assert restored\n"
            "        states = payload['rng_state_by_rank']\n"
            "        retained_rank = dist.get_rank() < len(states)\n"
            "        if retained_rank:\n"
            "            state = states[dist.get_rank()]\n"
            "            assert torch.equal(torch.get_rng_state(), state['torch_cpu'].cpu())\n"
            "            assert torch.equal(torch.cuda.get_rng_state(device), state['torch_cuda'].cpu())\n"
            "        record('balanced-rng-resume-rank'+os.environ['RANK']+'.json', dict(\n"
            "            passed=True, global_step=payload['global_step'], epoch=payload['epoch'],\n"
            "            next_microbatch=payload['batch_idx'], cpu_rng_exact=retained_rank,\n"
            "            cuda_rng_exact=retained_rank, new_rank_stream=not retained_rank))\n"
            "    return restored\n"
            "training.restore_rank_rng_state = verified_rng_restore\n",
        'def save(payload, target):\n    checkpoint_io.atomic_torch_save(payload, target, background=WRITER)':
            "def save(payload, target):\n"
            "    smoke_target = os.environ.get('LASER_LOSS_BALANCE_SMOKE_CHECKPOINT')\n"
            "    if smoke_target:\n"
            "        assert PHASE == 'production' and ARGS.wandb_mode == 'disabled'\n"
            "        assert not ARGS.upload_checkpoints and ARGS.max_optimizer_steps == 2\n"
            "        path = Path(smoke_target)\n"
            "        temporary = path.with_suffix('.tmp')\n"
            "        torch.save(payload, temporary)\n"
            "        temporary.replace(path)\n"
            "        record('smoke-endpoint.json', dict(passed=True, global_step=payload['global_step'],\n"
            "            local_checkpoint=str(path), production_checkpoint_modified=False))\n"
            "        return\n"
            "    checkpoint_io.atomic_torch_save(payload, target, background=WRITER)",
        "ARGS.compound_geometry_version = 'ffhq_v4_archived_expected_atom_times_expected_coefficient'":
            "ARGS.compound_geometry_version = 'disabled_gradient_balanced_ce'",
        "objective='archived atom-weighted CE + clean physical pair/spatial geometry'":
            "objective='gradient-balanced atom and coefficient CE; geometry disabled'",
        "correction='ImageNet-calibrated lower noise: sigma .125 with retained q99.5 scales, clean clipping and native sampler; fresh model and Adam',":
            "correction='Gradient-balanced CE continuation; geometry disabled; model, Adam, scheduler and RNG retained',\n"
            "        atom_loss_weight=ATOM_LOSS_WEIGHT, coefficient_loss_weight=1., geometry_loss_weight=0.,\n"
            "        loss_balance_policy=LOSS_POLICY,",
        "'original-fid-only-policy.json', 'original-fid-only-tests.json'):":
            "'original-fid-only-policy.json', 'original-fid-only-tests.json',\n"
            "                 'loss-balance-policy.json', 'loss-balance-gradient-calibration.json',\n"
            "                 'loss-balance-test-verification.json', 'loss-balance-resume-verification.json'):",
    }
    for before, after in changes.items():
        if original.count(before) != 1:
            raise ValueError(f'Unexpected frozen entry structure: {before}')
        original = original.replace(before, after)
    begin = original.index('        if not objective_probed:\n')
    end = original.index('            objective_probed = True\n', begin)
    original = original[:begin]+"""        if not objective_probed:
            with torch.no_grad():
                probe_loss, probe = ffhq_adapter.compound_objective(atom_logits, coeff_logits,
                    prediction, atoms, probabilities, physical, **kwargs)
                expected = (ATOM_LOSS_WEIGHT*probe['atom_nll'].mean()
                    + probe['coeff_cross_entropy'].mean())/(ATOM_LOSS_WEIGHT+1)/kwargs['accumulation']
                assert torch.isfinite(probe_loss) and torch.allclose(probe_loss, expected)
                assert kwargs['atom_weight'] == ATOM_LOSS_WEIGHT and kwargs['geometry_weight'] == 0
                disabled = ('geometry', 'geometry_pair_mse', 'geometry_spatial_mse',
                    'coefficient_regression', 'coefficient_crps', 'causal_prefix', 'causal_prefix_mse')
                assert all(float(probe[name]) == 0 for name in disabled)
                record('balanced-objective-rank'+os.environ['RANK']+'.json', dict(passed=True,
                    archive_sha256=ffhq_adapter.ARCHIVE_SHA256, policy_version=LOSS_POLICY['version'],
                    atom_weight=ATOM_LOSS_WEIGHT, coefficient_weight=1., geometry_weight=0.,
                    normalized_weighted_ce_verified=True, disabled_terms=list(disabled),
                    calibration_shared_gradient_ratio=LOSS_POLICY['calibration_shared_gradient_ratio']))
"""+original[end:]
    compile(original, 'balanced_entry.py', 'exec')
    return original


def prepare(args):
    import yaml
    calibration = json.loads((args.audit/'gradient-calibration.json').read_text())
    assert calibration['passed'] and calibration['images'] == 64
    ratio = calibration['median_coefficient_to_atom_gradient_ratio']
    atom_weight = round(max(.25, min(1., round(ratio/.025)*.025)), 3)
    policy = dict(version=POLICY_VERSION, atom_weight=atom_weight, coefficient_weight=1.,
        geometry_weight=0., coefficient_regression_weight=0., coefficient_crps_weight=0.,
        causal_prefix_weight=0., classification_normalizer=atom_weight+1,
        calibration_checkpoint_step=calibration['global_step'],
        calibration_shared_gradient_ratio=ratio,
        calibration_old_weighted_atom_to_coefficient_gradient_ratio=1.5/ratio,
        calibration_new_weighted_atom_to_coefficient_gradient_ratio=atom_weight/ratio,
        calibration_images=64, calibration_classes=32,
        balancing_representation=calibration['balancing_representation'],
        full_backbone_parameter_gradients_measured=False,
        rationale='Balance measured shared decoder gradients; remove the stagnant geometry proxy. Retain both likelihood objectives.',
        coefficient_ce_target_entropy_is_constant=True,
        weight_selection='Median head-gradient ratio rounded to nearest .025; coefficient/atom weight capped at four.',
        noise_sigma_bins=42.645833333333336, normalized_noise_sigma=.125,
        continue_same_run=True, preserve_model_optimizer_scheduler_rng=True,
        generation_metric_keys=['eval/fid_original_train50k','eval/inception_score'],
        fid_num_samples=50000, fid_real_split='train', prepared_unix=time.time(),
        fid_benefit_verified=False)
    args.stage.mkdir(parents=True, exist_ok=True)
    before = args.audit/'before'
    before.mkdir(exist_ok=False)
    for name in ('entry.py','production.yaml','preflight.yaml'):
        shutil.copyfile(args.base/name, before/name)
    for name in ('source-manifest.json','source-code.tar.gz','train.yaml'):
        shutil.copyfile(args.output/name, before/name)
    recipe = yaml.safe_load((args.base/'production.yaml').read_text())
    options = dict(recipe['options'], atom_loss_weight=atom_weight, geometry_loss_weight=0.,
        compound_distribution_geometry=False, resume=True,
        resume_checkpoint=str(args.output/'train/checkpoints/last.pt'), max_optimizer_steps=0)
    entry = make_entry((args.base/'entry.py').read_text())
    (args.stage/'entry.py').write_text(entry)
    (args.stage/'production.yaml').write_text(yaml.safe_dump(dict(recipe, options=options), sort_keys=False))
    # Native smoke uses the production resume path; its separate local state is discarded.
    smoke = dict(options, max_optimizer_steps=2, fid_every=0, save_step_freq=0,
        sample_grid_every=0, upload_checkpoints=False, wandb_mode='disabled')
    (args.stage/'smoke.yaml').write_text(yaml.safe_dump(dict(recipe, options=smoke), sort_keys=False))
    preflight = yaml.safe_load((args.base/'preflight.yaml').read_text())
    preflight['options'].update(atom_loss_weight=atom_weight, geometry_loss_weight=0., compound_distribution_geometry=False)
    (args.stage/'preflight.yaml').write_text(yaml.safe_dump(preflight, sort_keys=False))
    (args.stage/'loss-balance-policy.json').write_text(json.dumps(policy, indent=2)+'\n')
    (args.audit/'policy.json').write_text(json.dumps(policy, indent=2)+'\n')
    hashes = {name:hashlib.sha256((args.stage/name).read_bytes()).hexdigest()
        for name in ('entry.py','production.yaml','smoke.yaml','preflight.yaml','loss-balance-policy.json')}
    (args.audit/'staged-files.json').write_text(json.dumps(hashes, indent=2)+'\n')
    print(json.dumps(policy, indent=2))


def apply_staged(args):
    """Commit only after a full checkpoint and clean native worker shutdown."""
    root = Path(__file__).resolve().parents[2]
    shutdown = json.loads((args.audit/'shutdown.json').read_text())
    saved = json.loads((args.audit/'pre-change-checkpoint-verification.json').read_text())
    assert saved['passed'] and saved['checkpoint_state']['finite_checked']
    for worker in shutdown['workers']:
        process = Path('/proc')/str(worker['pid'])/'stat'
        assert not process.exists() or process.read_text().rsplit(')', 1)[1].strip().split()[0] == 'Z'
    with (args.base/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ('entry.py', 'production.yaml', 'preflight.yaml'):
            assert (args.base/name).read_bytes() == (args.audit/'before'/name).read_bytes()
        hashes = json.loads((args.audit/'staged-files.json').read_text())
        for name, expected in hashes.items():
            if name != 'loss-balance-policy.json':
                assert hashlib.sha256((args.stage/name).read_bytes()).hexdigest() == expected
        policy = json.loads((args.stage/'loss-balance-policy.json').read_text())
        assert policy['started_at_step'] == saved['checkpoint_state']['global_step']
        compile((args.stage/'entry.py').read_text(), 'balanced_entry.py', 'exec')

        def write(path, content):
            temporary = path.with_suffix(path.suffix+'.loss-change.tmp')
            temporary.write_bytes(content)
            temporary.replace(path)

        write(args.output/'loss-balance-policy.json', (args.stage/'loss-balance-policy.json').read_bytes())
        for source, target in (('gradient-calibration.json', 'loss-balance-gradient-calibration.json'),
                               ('test-verification.json', 'loss-balance-test-verification.json')):
            write(args.output/target, (args.audit/source).read_bytes())
        for name in ('entry.py', 'production.yaml', 'preflight.yaml'):
            write(args.base/name, (args.stage/name).read_bytes())
        for relative in ('scripts/tools/calibrate_imagenet_loss_balance.py',
                         'scripts/tools/rebalance_imagenet_ffhq_losses.py',
                         'tests/test_imagenet_loss_rebalance.py'):
            target = args.base/'source'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root/relative, target)
        canonical = b'# @package _global_\n'+(args.stage/'production.yaml').read_bytes()
        write(args.output/'train.yaml', canonical)
        write(root/'configs/stage2'/f'{args.output.name}.yaml', canonical)
        plan = json.loads((args.output/'plan.json').read_text())
        plan['loss_balance_continuation'] = policy
        write(args.output/'plan.json', (json.dumps(plan, indent=2)+'\n').encode())
        manifest = {str(p.relative_to(args.base)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (args.base/'source').rglob('*')
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
        for name in ('entry.py','production.yaml','preflight.yaml'):
            manifest[name] = hashlib.sha256((args.base/name).read_bytes()).hexdigest()
        write(args.output/'source-manifest.json', (json.dumps(manifest, indent=2)+'\n').encode())
        with tarfile.open(args.stage/'source-code.tar.gz', 'w:gz') as archive:
            archive.add(args.base/'source', arcname='source',
                filter=lambda item: None if '__pycache__' in item.name or item.name.endswith('.pyc') else item)
            for name in ('entry.py','preflight.yaml','production.yaml'):
                archive.add(args.base/name, arcname=name)
            for name in ('loss-balance-policy.json','loss-balance-gradient-calibration.json',
                         'loss-balance-test-verification.json','coefficient-support-policy.json',
                         'support-validation.json','original-fid-only-policy.json','plan.json'):
                archive.add(args.output/name, arcname=name)
        shutil.copyfile(args.stage/'source-code.tar.gz', args.output/'source-code.tar.gz')
        record = dict(applied=True, step=saved['checkpoint_state']['global_step'], policy=policy,
            model_architecture_changed=False, optimizer_reset=False, scheduler_reset=False,
            source_manifest_verified=all(hashlib.sha256((args.base/name).read_bytes()).hexdigest()==h
                for name,h in manifest.items()), time=time.time())
        (args.audit/'applied.json').write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(record, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','apply'], nargs='?', default='prepare')
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--stage', type=Path, required=True)
    arguments = parser.parse_args()
    {'prepare':prepare, 'apply':apply_staged}[arguments.action](arguments)
