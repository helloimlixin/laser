"""Prepare, verify, and run one epoch of stochastic atom/coefficient targets."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.tools.run_compound_energy_trial import environment, patch_trial_runtime, record


def prepare(args):
    import torch
    import yaml
    from src.training.stochastic_image_pairs import POLICY_VERSION
    if (args.output/'preparation-complete.json').exists():
        raise RuntimeError('Prepared trial already exists')
    calibration = json.loads((args.output/'calibration.json').read_text())
    if not calibration['passed'] or calibration['policy'] != POLICY_VERSION:
        raise RuntimeError('Training requires a passed matching calibration')
    source = Path('/tmp/laser-compound-energy-ab-20261007/source-epoch079-full.pt')
    parent = Path('/tmp/laser-compound-energy-ab-20261007/control')
    sys.path.insert(0, str(parent/'support'))
    from src.training.full_resume_upload import recovery_metadata
    payload = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
    metadata = recovery_metadata(payload)
    if (metadata['epoch'], metadata['global_step'], metadata['rng_ranks']) != (79, 49454, 8):
        raise RuntimeError('Protected full-state source identity changed')
    if metadata['saved_learning_rates'] != [8.397530347342202e-7]:
        raise RuntimeError('Source learning rate changed')
    plan = dict(version='stochastic-image-pair-trial-v1', target_policy=POLICY_VERSION,
        atom_temperature=calibration['selected_temperature'], site_chunk_size=calibration['site_chunk_size'],
        coefficient_temperature=.01125, source_checkpoint=str(source), source_epoch=79,
        source_step=49454, source_fid=15.081930549156368,
        source_run='helloimlixin-rutgers/laser/imagenet-rfid421-epoch77-full-compound-history-lr1e6-8h100-20261006',
        initial_lr=metadata['saved_learning_rates'][0], target_epoch=80, target_step=50080,
        updates=626, geometry_loss_enabled=False, additional_losses_enabled=False,
        optimizer_and_lr_policy='all801 Adam states and original absolute cosine to zero unchanged',
        evaluation_seeds=[261001,261101], evaluation_backend='original_rqtransformer',
        real_split='val', real_images=50000, generated_images=50000,
        automatic_promotion=False, automatic_continuation=False,
        sampler=dict(atom_temperature=.9, atom_top_p=.9, coefficient_temperature=1., coefficient_top_p=.85),
        matched_control=str(REPO/'outputs/compound-energy-ab-20261007/control/official-two-seed-summary.json'),
        wandb_id='imagenet-rfid421-epoch79-stochastic-atoms-coeff-official50k-20261007')
    args.base.mkdir(parents=True, exist_ok=True)
    runtime = args.base/'runtime'
    shutil.copytree(parent/'source/runtime', runtime,
        ignore=shutil.ignore_patterns('__pycache__'), dirs_exist_ok=True)
    changed = ['src/training/stochastic_image_pairs.py', 'src/training/stochastic_pair_trial_hook.py']
    for name in changed:
        shutil.copy2(REPO/name, runtime/name)
    patch_trial_runtime(runtime)
    base, output = args.base/'trial', args.output/'trial'
    (base/'source').mkdir(parents=True, exist_ok=True)
    (base/'source/runtime').symlink_to(runtime)
    shutil.copytree(parent/'support', base/'support', ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['compound-transfer.json', 'fixed-teacher-batch.pt']:
        shutil.copy2(parent/name, base/name)
    (base/'inputs').mkdir()
    (base/'inputs/resume-stage1-tokenizer.pt').symlink_to(parent/'inputs/resume-stage1-tokenizer.pt')
    # Reuse derived compilation/model-download caches; no training data download.
    for name in ['inductor-cache', 'torch-cache']:
        (base/name).symlink_to(parent/name)
    template = (parent/'entry.py').read_text()
    old = ("from src.training.compound_geometry_trial_hook import install as install_energy_trial\n"
           "install_energy_trial(globals(), json.loads((BASE.parent/'plan.json').read_text()))")
    new = ("from src.training.stochastic_pair_trial_hook import install as install_stochastic_trial\n"
           "install_stochastic_trial(globals(), json.loads((BASE.parent/'plan.json').read_text()))")
    if template.count(old) != 1:
        raise RuntimeError('Frozen entrypoint changed')
    (base/'entry.py').write_text(template.replace(old, new))
    recipe = yaml.safe_load((parent/'recipe.yaml').read_text())
    options = dict(recipe['options'], checkpoint=str(base/'inputs/resume-stage1-tokenizer.pt'),
        output=str(output/'train'), checkpoint_dir=str(output/'train/checkpoints'),
        resume_checkpoint=str(source), epochs=80, wandb_id=plan['wandb_id'],
        wandb_name='ImageNet K4 | stochastic atoms + coefficients | epoch79',
        save_step_freq=250, upload_checkpoints=False, max_optimizer_steps=0, fid_every=1)
    for phase in ('preflight', 'production'):
        selected = dict(options)
        if phase == 'preflight':
            selected.update(max_optimizer_steps=20, fid_every=0)
        else:
            selected.update(resume_checkpoint=str(output/'train/checkpoints/last.pt'))
        (base/f'{phase}.yaml').write_text(yaml.safe_dump(dict(recipe, options=selected), sort_keys=False))
    output.mkdir(parents=True, exist_ok=True)
    record(args.base/'plan.json', plan)
    record(args.output/'plan.json', plan)
    record(args.output/'source-integrity.json', metadata)
    shutil.copyfile(base/'production.yaml', args.output/'recipe.yaml')
    files = list(runtime.rglob('*.py')) + list((base/'support').rglob('*.py')) + [base/'entry.py']
    manifest = {str(path.relative_to(args.base)):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    record(args.output/'code-manifest.json', manifest)
    with zipfile.ZipFile(args.output/'stochastic-pair-code.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(args.base))
        for name in ['scripts/tools/run_stochastic_image_pair_trial.py',
                     'scripts/tools/calibrate_imagenet_stochastic_pairs.py',
                     'tests/test_stochastic_image_pairs.py']:
            archive.write(REPO/name, name)
    record(args.output/'preparation-complete.json', dict(plan=plan, code_manifest=manifest, time=time.time()))
    print(json.dumps(dict(prepared=True, plan=plan)), flush=True)


def verify_preflight(args):
    import torch
    base, output = args.base/'trial', args.output/'trial'
    sys.path[:0] = [str(args.base/'runtime'), str(base/'support')]
    from src.training.k4_checkpoint_io import _checkpoint_upload_source
    from src.training.full_resume_upload import recovery_metadata
    os.environ['LASER_CHECKPOINT_UPLOAD_CACHE_DIR'] = str(base/'checkpoint-upload-cache')
    os.environ['LASER_CHECKPOINT_IMMUTABLE_FILES'] = '1'
    saved = json.loads((output/'last-local-save.json').read_text())
    payload = torch.load(_checkpoint_upload_source(Path(saved['target'])), map_location='cpu', mmap=True, weights_only=False)
    plan = json.loads((args.base/'plan.json').read_text())
    expected_policy = {key:plan[key] for key in ('target_policy','atom_temperature',
        'site_chunk_size','coefficient_temperature','source_step')}
    if payload.get('stochastic_pair_targets') != expected_policy:
        raise RuntimeError('The checkpoint did not preserve the stochastic target policy')
    metadata = recovery_metadata(payload)
    if metadata['global_step'] != 49474 or metadata['rng_ranks'] != 8:
        raise RuntimeError('Preflight recovery checkpoint is incomplete')
    proofs = []
    for rank in range(8):
        verify = output/'verification/preflight'
        step = json.loads((verify/f'step20-rank{rank}.json').read_text())
        startup = json.loads((verify/f'startup-rank{rank}.json').read_text())
        stochastic = json.loads((verify/f'stochastic-targets-rank{rank}.json').read_text())
        rng = json.loads((verify/f'resume-rng-rank{rank}.json').read_text())
        if not (step['finite'] and startup['optimizer_parameters'] == 801
                and startup['lr'] == 8.397530347342202e-7 and rng['restored_exactly']
                and stochastic['coefficient_rng_replay_exact']):
            raise RuntimeError('Distributed stochastic preflight failed')
        proofs.append(dict(rank=rank, step=step, startup=startup, stochastic=stochastic))
    record(args.output/'preflight-verification.json', dict(passed=True,
        recovery=metadata, rank_proofs=proofs, continuation='resume these20 verified updates exactly'))


def publish(args, plan):
    base, output = args.base/'trial', args.output/'trial'
    sys.path[:0] = [str(args.base/'runtime'), str(base/'support')]
    os.environ['WANDB_API_KEY'] = args.key_file.read_text().strip()
    os.environ['LASER_CHECKPOINT_UPLOAD_CACHE_DIR'] = str(base/'checkpoint-upload-cache')
    os.environ['LASER_CHECKPOINT_IMMUTABLE_FILES'] = '1'
    from src.training.k4_checkpoint_io import _checkpoint_upload_source
    from verified_wandb_checkpoint_upload import VerifiedCloudUpload
    checkpoint_dir = output/'train/checkpoints'
    slots = base/'upload-slots'
    slots.mkdir(exist_ok=True)
    sources = [('last.pt', checkpoint_dir/'last.pt'),
        ('best-fid-resume.pt', next(checkpoint_dir.glob('best_fid_*.pt'))),
        ('best-is-resume.pt', next(checkpoint_dir.glob('best_is_*.pt')))]
    paths = []
    for name, source in sources:
        target = slots/name
        if not target.exists():
            os.link(_checkpoint_upload_source(source), target)
        paths.append(target)
    run_path = 'helloimlixin-rutgers/laser/'+plan['wandb_id']
    VerifiedCloudUpload(run_path, args.output/'cloud-checkpoint-receipt.json')(paths, 80)
    extras = [args.output/name for name in ['plan.json','calibration.json','code-manifest.json',
        'stochastic-pair-code.zip','preflight-verification.json','comparison.json','test-verification.json']]
    extras += list(args.output.glob('code-transition.json'))
    extras += list(args.output.glob('stochastic-pair-code-preflight.zip'))
    extras += list(output.glob('official-*.json'))
    VerifiedCloudUpload(run_path, args.output/'cloud-evidence-receipt.json')(extras, 80)


def run(args):
    args.base.mkdir(parents=True, exist_ok=True)
    lock = (args.base/'supervisor.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan = json.loads((args.base/'plan.json').read_text())
    manifest = json.loads((args.output/'code-manifest.json').read_text())
    if any(hashlib.sha256((args.base/name).read_bytes()).hexdigest() != value for name, value in manifest.items()):
        raise RuntimeError('Frozen trial code changed')
    base, output = args.base/'trial', args.output/'trial'
    try:
        phases = ['preflight', 'production'] if args.action == 'run' else ['production']
        if args.action == 'resume-production':
            verify_preflight(args)
        for phase in phases:
            env = environment(base, output, args.key_file, phase)
            command = [sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8',
                str(base/'entry.py'),'--config',str(base/f'{phase}.yaml')]
            with (output/'training.log').open('a') as log:
                child = subprocess.Popen(command, cwd=base, env=env, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
                while child.poll() is None:
                    record(args.output/'status.json', dict(state='running',phase=phase,
                        supervisor_pid=os.getpid(),torchrun_pid=child.pid,source_step=49454,
                        target_step=49474 if phase=='preflight' else 50080,time=time.time()))
                    time.sleep(5)
            if child.returncode:
                raise RuntimeError(f'{phase} failed with code{child.returncode}; inspect training.log')
            if phase == 'preflight':
                verify_preflight(args)
        result = json.loads((output/'official-two-seed-summary.json').read_text())
        control = json.loads(Path(plan['matched_control']).read_text())
        record(args.output/'comparison.json', dict(stochastic=result, control=control,
            fid_mean_difference=result['fid_mean']-control['fid_mean'],
            inception_score_mean_difference=result['inception_score_mean']-control['inception_score_mean'],
            automatic_promotion=False, source_epoch79_protected=True,
            paired_control='same source, image order, optimizer, scheduler, sampler, evaluation seeds; target RNG draws differ'))
        record(args.output/'status.json', dict(state='publishing', official_evaluations_completed=2, time=time.time()))
        publish(args, plan)
        record(args.output/'status.json', dict(state='completed', official_evaluations_completed=2,
            full_best_and_last_checkpoints_verified_online=True, automatic_promotion=False,
            source_epoch79_protected=True,time=time.time()))
    except BaseException as error:
        record(args.output/'status.json', dict(state='failed',error=str(error),time=time.time()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare','run','resume-production'])
    parser.add_argument('--base', type=Path, default=Path('/tmp/laser-stochastic-image-pairs-20261007'))
    parser.add_argument('--output', type=Path, default=REPO/'outputs/stochastic-image-pairs-20261007')
    parser.add_argument('--key-file', type=Path, default=Path('/tmp/laser-resume-wandb-key'))
    options = parser.parse_args()
    (prepare if options.action == 'prepare' else run)(options)
