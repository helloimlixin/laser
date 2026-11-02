#!/usr/bin/env python3
"""Launch a bounded native RQ experiment on verified joint LASER IDs."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

ROOT = Path('/workspace/Projects/laser')
sys.path[:0] = [str(ROOT), str(ROOT/'scripts/tools')]
import launch_laser_hard_ar_imagenet as launcher
from runtime.rq_imagenet_entry import freeze_shared_runtime

RUN = 'imagenet-laser-native-joint-scratch6-4h100-20261015'
BASE = Path('/tmp/laser-rq-native-joint-imagenet-20261015')
OUT = ROOT/'outputs'/RUN
SOURCE = Path('/tmp/laser-rq-clean-scratch-imagenet-20261015')
STUDY = ROOT/'outputs/imagenet-laser-joint-vocabulary-study-20261015'
RECIPE = ROOT/'configs/stage2/imagenet-laser-native-joint-scratch6-20261015.yaml'
launcher.BASE, launcher.OUT, launcher.RUN = BASE, OUT, RUN


def prepare():
    import yaml
    if (OUT/'prepared.json').exists(): raise RuntimeError('Already prepared')
    selected = json.loads((STUDY/'selected.json').read_text())
    assert selected['passed'] and selected['real_images'] == selected['reconstructed_images'] == 50000
    assert selected['source_sha256'] == launcher.digest(SOURCE/'inputs/stage1-model.pt')
    assert selected['drift'] <= .20 and selected['temperature'] > 0
    codebook = STUDY/f"{selected['candidate']}-codebook.pt"
    assert launcher.digest(codebook) == selected['codebook_sha256']
    recipe = yaml.safe_load(RECIPE.read_text())
    assert recipe['run_id'] == RUN and recipe['decoder'] == 'native_joint_integer'
    assert recipe['epochs'] == 6 and recipe['auxiliary_loss_terms'] == 0
    assert recipe['global_batch'] == 2048 and recipe['lr_schedule_epochs'] == 100
    recipe['tokenizer'].update(codebook_sha256=selected['codebook_sha256'],
        vocab_size=selected['vocab_size'], measured_rfid=selected['converted_rfid'],
        source_measured_rfid=selected['source_rfid'], candidate=selected['candidate'])
    recipe['targets']['temperature'] = selected['temperature']
    # Preserve approximately the same codeword budget after vocabulary expansion.
    recipe['sampling']['top_k'] = 256 * ((selected['vocab_size']-1)//16384)
    RECIPE.write_text(yaml.safe_dump(recipe, sort_keys=False))
    (BASE/'inputs').mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    freeze_shared_runtime(BASE, ROOT, include_stochastic_teacher=False)
    shutil.copytree(SOURCE/'upstream', BASE/'upstream',
        ignore=shutil.ignore_patterns('__pycache__', '.git', '*.pyc'))
    (BASE/'train.py').write_text("from runtime.rq_imagenet_entry import launch\nif __name__ == '__main__':\n    launch('laser_joint')\n")
    shutil.copyfile(ROOT/'scripts/tools/verify_laser_native_joint_checkpoint.py', BASE/'verify.py')
    for name in ['stage2-training-config.yaml', 'stage2-archive-config.yaml', 'imagenet_256_train.npz',
                 'rq-upstream-core.json']:
        shutil.copyfile(SOURCE/'inputs'/name, BASE/'inputs'/name)
    config = yaml.safe_load((BASE/'inputs/stage2-training-config.yaml').read_text())
    config['loss'] = dict(type='soft_target_cross_entropy', stochastic_codes=True,
                          temp=selected['temperature'])
    (BASE/'inputs/stage2-training-config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    os.link(SOURCE/'inputs/stage1-model.pt', BASE/'inputs/stage1-model.pt')
    shutil.copyfile(codebook, BASE/'inputs/joint-codebook.pt')
    shutil.copyfile(RECIPE, BASE/'inputs/recipe.yaml')
    assert launcher.digest(BASE/'inputs/imagenet_256_train.npz') == recipe['evaluation']['reference_sha256']
    for name in ['data', 'torch-cache']:
        (BASE/name).symlink_to(SOURCE/name, target_is_directory=True)
    environment = dict(LASER_RQ_BASE=str(BASE), LASER_CHECKPOINT_STAGING_DIR=str(BASE/'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(BASE/'checkpoint-cache'), LASER_CHECKPOINT_IMMUTABLE_FILES='1',
        TORCH_HOME=str(BASE/'torch-cache'), WANDB_DIR=str(BASE/'wandb'),
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    launcher.write(BASE/'environment.json', environment)
    reproducibility = OUT/'reproducibility'; reproducibility.mkdir(exist_ok=True)
    for path in (BASE/'inputs').iterdir():
        if path.suffix in ('.yaml', '.json', '.pt') and path.name != 'stage1-model.pt':
            shutil.copyfile(path, reproducibility/path.name)
    for path in STUDY.glob('*.json'):
        shutil.copyfile(path, reproducibility/path.name)
    files = [BASE/'train.py', BASE/'verify.py', BASE/'environment.json']
    for directory in ['source', 'runtime', 'upstream']:
        files.extend((BASE/directory).rglob('*.py'))
    files.extend((BASE/'inputs').glob('*.yaml'))
    files.extend([BASE/'inputs/rq-upstream-core.json', BASE/'inputs/joint-codebook.pt'])
    launcher.write(reproducibility/'code-manifest.json',
                   {str(p.relative_to(BASE)): launcher.digest(p) for p in files})
    with zipfile.ZipFile(reproducibility/'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files: archive.write(path, path.relative_to(BASE))
        for name in ['launch_laser_native_joint_imagenet.py', 'prepare_laser_joint_vocabulary.py',
                     'verify_laser_joint_vocabulary.py']:
            archive.write(ROOT/'scripts/tools'/name, 'scripts/tools/'+name)
        archive.write(ROOT/'src/training/laser_joint_vocabulary.py', 'src/training/laser_joint_vocabulary.py')
        archive.write(ROOT/'tests/test_laser_joint_vocabulary.py', 'tests/test_laser_joint_vocabulary.py')
    launcher.write(OUT/'prepared.json', dict(prepared=True, run_id=RUN, recipe=recipe,
        fresh_stage2=True, fresh_optimizer=True, source_stage2_checkpoint=None,
        initial_global_step=0, initial_optimizer_state_count=0,
        train_images=1281167, global_batch=2048, updates_per_epoch=626,
        stop_updates=3756, cosine_total_updates=62600, evaluation_epochs=[1, 2, 4, 6],
        sampled_trajectories=0, native_transformer=True, joint_codebook_verification=selected,
        source_manifest_sha256=launcher.digest(reproducibility/'code-manifest.json'), time=time.time()))
    print('Prepared native joint integer run', RUN, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run'])
    args = parser.parse_args()
    if args.action == 'prepare': prepare()
    else:
        os.environ['WANDB_API_KEY'] = Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
        result = launcher.run(resume=False)
        if result == 0:
            curves = {}
            reference = ROOT/'outputs/imagenet-original-rq1400m-rfid473-scratch20-4h100-20261014/evaluations'
            for epoch in [1, 2, 4, 6]:
                filename = f'epoch{epoch:03d}.json'
                joint = json.loads((OUT/'evaluations'/filename).read_text())
                rq = json.loads((reference/filename).read_text())
                assert joint['generated_images'] == rq['generated_images'] == 50000
                assert joint['real_images'] == rq['real_images'] == 1281167
                assert joint['train/global_step'] == rq['train/global_step'] == 626*epoch
                curves[str(epoch)] = dict(joint_fid=joint['eval/fid'], rq_fid=rq['eval/fid'],
                                         joint_minus_rq=joint['eval/fid']-rq['eval/fid'])
            recipe = json.loads((OUT/'prepared.json').read_text())['recipe']
            launcher.write(OUT/'comparison.json', dict(completed_epochs=6, global_step=3756,
                curves=curves, joint_sampler=recipe['sampling'], rq_sampler=dict(temperature=1., top_k=256, top_p=.95),
                continued_beyond_six_epochs=False))
        raise SystemExit(result)


if __name__ == '__main__': main()
