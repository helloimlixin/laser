"""Stage the saved ordinary-Adam branch for uninterrupted epoch-100 training."""
import json
from pathlib import Path
import shutil

import torch
import yaml

ROOT = Path('/workspace/Projects/laser')
SEARCH = ROOT / 'outputs/imnet-j16384-best14574-direction-search-1b300-20261104'
SOURCE = Path('/tmp/laser-best-b300-20261104/frozen')
BASE = Path('/tmp/laser-adam-small-continue100-20261105/frozen')
RUN = 'imnet-j16384-adam-small-continue100-1b300-20261105'
OUT = ROOT / 'outputs' / RUN


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + '\n')


def main():
    assert not BASE.exists() and not (OUT / 'launch.json').exists()
    result = json.loads((SEARCH / 'provisional-best.json').read_text())
    measured = json.loads((SEARCH / 'winner-confirmation-50k.json').read_text())
    checkpoint = Path(result['checkpoint']['path'])
    assert checkpoint.stat().st_size == result['checkpoint']['bytes']
    payload = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    assert payload['global_step'] == measured['global_step'] == 45706
    assert payload['next_epoch'] == 73 and payload['next_microbatch'] == 64
    assert payload['direction_search']['mode'] == 'adam'
    assert payload['runtime']['lr_amplitude_factor'] == 0.00375
    assert len(payload['optimizer']['state']) == 780
    assert {int(s['step']) for s in payload['optimizer']['state'].values()} == {45706}
    assert payload['scheduler']['after']['last_epoch'] == 45706
    assert payload['scheduler']['after']['T_max'] == 62600
    assert measured['generated_images'] == 50000 and measured['generation_batch'] == 64
    assert measured['seed'] == 261001 and measured['reference_split'] == 'train'
    assert measured['sampler'] == payload['native_sampling']
    assert all(tuple(g['betas']) == (0.9, 0.95) for g in payload['optimizer']['param_groups'])
    BASE.mkdir(parents=True)
    for folder in ['source', 'upstream', 'runtime']:
        shutil.copytree(SOURCE / folder, BASE / folder, ignore=shutil.ignore_patterns('__pycache__'))
    (BASE / 'inputs').mkdir()
    for path in (SOURCE / 'inputs').iterdir():
        if not path.is_file() or path.name == 'resume-checkpoint.pt':
            continue
        if path.stat().st_size > 50 * 1024**2:
            (BASE / 'inputs' / path.name).hardlink_to(path)
        else:
            shutil.copyfile(path, BASE / 'inputs' / path.name)
    (BASE / 'inputs/resume-checkpoint.pt').hardlink_to(checkpoint)
    for name in ['data', 'torch-cache']:
        (BASE / name).symlink_to(SOURCE / name, target_is_directory=True)
    shutil.copyfile(SOURCE / 'train.py', BASE / 'train.py')
    recipe = dict(payload['recipe'])
    recipe.update(run_id=RUN, name='ImageNet joint16384 | adam small branch | continue100 | 1 B300',
        lineage=dict(source_run_id=payload['run_id'], source_step=45706,
            source_artifact=result['committed_checkpoint']['artifact'],
            reason='User explicitly requested continuing the small-LR Adam branch through epoch100 despite failed improvement confirmation.'))
    (BASE / 'inputs/recipe.yaml').write_text(yaml.safe_dump(recipe, sort_keys=False))
    write(BASE / 'inputs/sampling-continuation.json', dict(source_run_id=payload['run_id'],
        source_global_step=45706, previous_sampler=payload['native_sampling'],
        sampler=payload['native_sampling'], source_checkpoint_artifact=result['committed_checkpoint']['artifact']))
    write(BASE / 'inputs/branch-full50k-evaluation.json', measured)
    execution = json.loads((BASE / 'inputs/execution.json').read_text())
    execution.update(source_run_id=payload['run_id'], source_step=45706, source_epoch=73,
        source_checkpoint_artifact=result['committed_checkpoint']['artifact'],
        lr_amplitude_factor=0.00375, repair_trial=False, source_saved_world_size=1,
        quality_guard=dict(enabled=False, reason='User requested continuous training until epoch100'),
        eval_every_epochs=5, is_preservation_floor=0.0,
        best_checkpoint_policy='Lowest full50k FID in this continuation; original14.5739 checkpoint retained separately',
        validated_initial_fid=measured['fid'], validated_initial_is=measured['inception_score'],
        baseline_measurement_origin='Previously measured full50k evaluation of identical adam_small weights on this B300',
        matched_full50k_initial_evaluation=True,
        rng_policy='Restore the saved one-GPU branch RNG, Adam age45706, epoch73 microbatch64 and unchanged cosine clock.')
    write(BASE / 'inputs/execution.json', execution)
    path = BASE / 'source/src/training/rq_imagenet_shared.py'
    code = path.read_text()
    needle = '    ddp=DDP(model,device_ids=[local],broadcast_buffers=False,gradient_as_bucket_view=True,bucket_cap_mb=100)'
    assert code.count(needle) == 1
    override = """    if args.resume and global_step == 45706:
        measured_branch=json.loads((BASE/'inputs/branch-full50k-evaluation.json').read_text())
        assert measured_branch['global_step']==global_step
        assert measured_branch['generated_images']==50000 and measured_branch['generation_batch']==64
        assert measured_branch['seed']==261001 and measured_branch['sampler']==PROFILE.sampling_settings()
        assert measured_branch['reference_sha256']==sha256(BASE/'inputs/imagenet_256_train.npz')
        restored_evaluation={key:measured_branch[key] for key in ('fid','inception_score','inception_score_std','sampler')}
        restored_best_fid=restored_evaluation['fid']
        inherited_evaluation_world_size=1
        verified_baseline_reused=True
"""
    code = code.replace(needle, override + needle)
    code = code.replace("measurement_hardware='B200' if inherited_baseline else 'B300',", "measurement_hardware='B300',")
    path.write_text(code)
    support = BASE / 'source/src/training/joint_runtime_support.py'
    support.write_text(support.read_text().replace(
        'ImageNet Joint16384 best14574 continue100 1B300 20261104',
        'ImageNet Joint16384 Adam small continue100 1B300 20261105'))
    proof = dict(passed=True, source_step=45706, adam_age=45706, optimizer_states=780,
        next_epoch=73, next_microbatch=64, lr_amplitude_factor=0.00375,
        saved_lr=[g['lr'] for g in payload['optimizer']['param_groups']],
        scheduler_step=45706, scheduler_horizon=62600, native_adam_direction=True,
        beta1=0.9, beta2=0.95, full50k_baseline=measured,
        source_artifact=result['committed_checkpoint']['artifact'], target_epoch=100,
        target_step=62600, early_stop_for_fid=False, full50k_every_epochs=5)
    write(OUT / 'source-verification.json', proof)
    write(BASE / 'continuation-verification.json', proof)
    write(OUT / 'global-best-reference.json', json.loads((SEARCH / 'best-checkpoint.json').read_text()))
    for name in ['recipe.yaml', 'execution.json', 'sampling-continuation.json', 'branch-full50k-evaluation.json']:
        destination = OUT / 'reproducibility' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(BASE / 'inputs' / name, destination)
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
