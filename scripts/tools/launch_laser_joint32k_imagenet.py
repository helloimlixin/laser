#!/usr/bin/env python3
"""Fresh two-epoch native RQ trial with a verified compact LASER dictionary."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path('/workspace/Projects/laser')
sys.path[:0] = [str(ROOT), str(ROOT/'scripts/tools')]
import launch_laser_hard_ar_imagenet as launcher
from runtime.rq_imagenet_entry import freeze_shared_runtime

RUN = 'imagenet-laser-native-joint32k-scratch2-4h100-20261016'
BASE = Path('/tmp/laser-rq-native-joint32k-imagenet-20261016')
OUT = ROOT/'outputs'/RUN
SOURCE = Path('/tmp/laser-rq-native-joint-depthtemp-imagenet-20261016')
STUDY = ROOT/'outputs/imagenet-laser-compact-joint-vocabulary-20261016'
RECIPE = ROOT/'configs/stage2/imagenet-laser-native-joint32k-scratch2-20261016.yaml'
launcher.BASE, launcher.OUT, launcher.RUN = BASE, OUT, RUN


def prepare():
    import yaml
    if (OUT/'prepared.json').exists():
        raise RuntimeError('Already prepared; use run')
    recipe = yaml.safe_load(RECIPE.read_text())
    previous = yaml.safe_load((SOURCE/'inputs/recipe.yaml').read_text())
    selected = json.loads((STUDY/'compact8192-refined-gain110-full-rfid-final.json').read_text())
    validation = json.loads((STUDY/'native-quantizer-verification.json').read_text())
    book = Path(selected['codebook_path'])
    assert recipe['run_id'] == RUN and recipe['epochs'] == 2
    assert recipe['decoder'] == previous['decoder'] == 'native_joint_integer'
    for key in ('sampling', 'global_batch', 'microbatch', 'lr_schedule_epochs',
                'seed', 'objective', 'auxiliary_loss_terms', 'backbone', 'target_mode'):
        assert recipe[key] == previous[key], key
    assert recipe['global_batch'] == 2048 and recipe['lr_schedule_epochs'] == 100
    assert recipe['auxiliary_loss_terms'] == 0
    temperatures = recipe['targets']['temperature']
    assert len(temperatures) == 4 and all(math.isfinite(t) and t > 0 for t in temperatures)
    for key in ('temperature', 'chunk_size', 'encoding', 'stochastic_prefixes'):
        assert recipe['targets'][key] == previous['targets'][key], key
    assert validation['passed'] and validation['results']['compact8192-refined-gain110']['passed']
    assert recipe['tokenizer']['vocab_size'] == selected['vocabulary'] == 32769
    assert selected['measured_rfid'] == recipe['tokenizer']['measured_rfid'] <= 4.73
    assert selected['learned_compact_dictionary'] and selected['encoder_decoder_frozen']
    assert recipe['initialization'] == dict(reference_vocab_size=65537, paired_fresh_backbone=True)
    assert selected['passed'] and selected['real_images'] == selected['reconstructed_images'] == 50000
    assert launcher.digest(SOURCE/'inputs/stage1-model.pt') == recipe['tokenizer']['sha256'] == previous['tokenizer']['sha256']
    assert launcher.digest(book) == recipe['tokenizer']['codebook_sha256'] == selected['codebook_sha256']
    assert launcher.digest(SOURCE/'inputs/imagenet_256_train.npz') == recipe['evaluation']['reference_sha256']
    assert recipe['evaluation']['generated_images'] == 50000
    assert recipe['evaluation']['real_images'] == 1281167 and recipe['evaluation']['reference_split'] == 'train'
    assert recipe['evaluation']['every_epochs'] == 2 and recipe['evaluation']['include_epoch_one']
    (BASE/'inputs').mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    freeze_shared_runtime(BASE, ROOT, include_stochastic_teacher=False)
    shutil.copytree(SOURCE/'upstream', BASE/'upstream', ignore=shutil.ignore_patterns('__pycache__', '.git', '*.pyc'))
    (BASE/'train.py').write_text("from runtime.rq_imagenet_entry import launch\nif __name__ == '__main__':\n    launch('laser_joint')\n")
    shutil.copyfile(ROOT/'scripts/tools/verify_laser_native_joint_checkpoint.py', BASE/'verify.py')
    for name in ('stage2-training-config.yaml', 'stage2-archive-config.yaml',
                 'imagenet_256_train.npz', 'rq-upstream-core.json'):
        shutil.copyfile(SOURCE/'inputs'/name, BASE/'inputs'/name)
    shutil.copyfile(book, BASE/'inputs/joint-codebook.pt')
    shutil.copyfile(SOURCE/'inputs/joint-codebook.pt', BASE/'inputs/reference-joint-codebook.pt')
    import numpy as np
    fitted = json.loads((STUDY/'calibration-data.json').read_text())
    prior = json.loads((ROOT/'outputs/imagenet-laser-joint-vocabulary-study-20261015/calibration-data.json').read_text())
    excluded = set(fitted['fit_indices'] + prior['fit_indices'] + prior['temperature_indices'])
    indices = [int(i) for i in np.random.default_rng(26101632769).permutation(1281167) if int(i) not in excluded][:128]
    launcher.write(BASE/'inputs/target-audit.json', dict(training_indices=indices, seed=261001, epoch=0,
        disjoint_from_dictionary_fit_and_prior_temperature_calibration=True,
        source_calibration_sha256=launcher.digest(STUDY/'calibration-data.json'),
        reference_codebook_sha256=launcher.digest(SOURCE/'inputs/joint-codebook.pt')))
    config = yaml.safe_load((BASE/'inputs/stage2-training-config.yaml').read_text())
    config['loss'] = dict(type='soft_target_cross_entropy', stochastic_codes=True, temp=temperatures)
    assert config['optimizer']['init_lr'] == .0005
    assert config['optimizer']['warmup']['epoch'] == 0 and config['experiment']['test_freq'] == 2
    (BASE/'inputs/stage2-training-config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    os.link(SOURCE/'inputs/stage1-model.pt', BASE/'inputs/stage1-model.pt')
    shutil.copyfile(RECIPE, BASE/'inputs/recipe.yaml')
    launcher.write(BASE/'inputs/execution.json', dict(align_classifier=True, alignment_multiple=64,
        valid_classes=32769, padded_projection_width=32832, extra_softmax_classes=0))
    for name in ('data', 'torch-cache'):
        (BASE/name).symlink_to(SOURCE/name, target_is_directory=True)
    environment = dict(LASER_RQ_BASE=str(BASE), LASER_CHECKPOINT_STAGING_DIR=str(BASE/'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(BASE/'checkpoint-cache'), LASER_CHECKPOINT_IMMUTABLE_FILES='1',
        TORCH_HOME=str(BASE/'torch-cache'), WANDB_DIR=str(BASE/'wandb'),
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    launcher.write(BASE/'environment.json', environment)
    reproducibility = OUT/'reproducibility'
    reproducibility.mkdir(exist_ok=True)
    for path in (BASE/'inputs').iterdir():
        if path.suffix in ('.yaml', '.json', '.pt') and path.name != 'stage1-model.pt':
            shutil.copyfile(path, reproducibility/path.name)
    for name in ('compact8192-refined-gain110-full-rfid-final.json', 'native-quantizer-verification.json',
                 'recommended-trial.json', 'calibration-data.json', 'compact8192-refined-fit-final.json'):
        shutil.copyfile(STUDY/name, reproducibility/name)
    files = [BASE/'train.py', BASE/'verify.py', BASE/'environment.json']
    for directory in ('source', 'runtime', 'upstream'):
        files.extend((BASE/directory).rglob('*.py'))
    files.extend((BASE/'inputs').glob('*.yaml'))
    files.extend((BASE/'inputs').glob('*.json'))
    files.extend([BASE/'inputs/joint-codebook.pt', BASE/'inputs/reference-joint-codebook.pt'])
    launcher.write(reproducibility/'code-manifest.json',
        {str(p.relative_to(BASE)): launcher.digest(p) for p in files})
    with zipfile.ZipFile(reproducibility/'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(BASE))
        for path in (Path(__file__), ROOT/'scripts/tools/launch_laser_hard_ar_imagenet.py',
                     ROOT/'tests/test_laser_joint_vocabulary.py', ROOT/'tests/test_laser_joint_execution.py',
                     ROOT/'tests/test_laser_joint_compact_runtime.py'):
            archive.write(path, path.relative_to(ROOT))
    launcher.write(OUT/'prepared.json', dict(prepared=True, run_id=RUN, recipe=recipe,
        fresh_stage2=True, fresh_optimizer=True, source_stage2_checkpoint=None,
        initial_global_step=0, initial_optimizer_state_count=0,
        train_images=1281167, global_batch=2048, updates_per_epoch=626,
        stop_updates=1252, cosine_total_updates=62600, evaluation_epochs=[1, 2],
        sampled_trajectories=0, native_transformer=True,
        ablated_setting='compact dictionary and joint vocabulary', reconstruction_gate=selected,
        paired_initial_backbone=True, learned_compact_dictionary=True,
        source_manifest_sha256=launcher.digest(reproducibility/'code-manifest.json'), time=time.time()))
    test_env = dict(os.environ, PYTHONPATH='.:runtime:scripts/tools',
        OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    test_names = ['tests/test_laser_joint_vocabulary.py', 'tests/test_laser_joint_execution.py',
                  'tests/test_laser_joint_compact_runtime.py']
    with (OUT/'cpu-tests.log').open('w') as log:
        result = subprocess.run([sys.executable, '-m', 'pytest', '-q', *test_names,
            '--junitxml='+str(reproducibility/'cpu-tests.xml')], cwd=ROOT, env=test_env,
            stdout=log, stderr=subprocess.STDOUT)
    suites = ET.parse(reproducibility/'cpu-tests.xml').getroot().findall('testsuite')
    counts = {key: sum(int(s.attrib.get(key, 0)) for s in suites)
              for key in ('tests', 'failures', 'errors', 'skipped')}
    passed = result.returncode == 0 and counts['tests'] >= 21 and not counts['failures'] and not counts['errors']
    launcher.write(OUT/'cpu-verification.json', dict(counts, passed=passed,
        scope='native quantizer/cache and CE/gradient parity, classifier alignment, compact lineage/frozen CNN, paired backbone initialization/meta reload',
        time=time.time()))
    assert passed, 'CPU verification failed; see cpu-tests.log'
    print('Prepared and CPU-verified fresh compact vocabulary trial '+RUN, flush=True)


def compare():
    baselines = dict(
        joint_65k_depthtemp='imagenet-laser-native-joint-depthtemp-scratch2-4h100-20261016',
        joint_global='imagenet-laser-native-joint-scratch6-4h100-20261015',
        compound_replay='imagenet-compound-q995-early-sign-replay2-4h100-20261016',
        signed_joint512='imagenet-laser-signed-joint512-scratch6-4h100-20261015',
        original_rq='imagenet-original-rq1400m-rfid473-scratch20-4h100-20261014')
    curves = {}
    for epoch in (1, 2):
        filename = f'epoch{epoch:03d}.json'
        current = json.loads((OUT/'evaluations'/filename).read_text())
        assert current['generated_images'] == 50000 and current['real_images'] == 1281167
        assert current['train/global_step'] == 626*epoch
        row = dict(compact_joint_fid=current['eval/fid'], baselines={})
        for name, run in baselines.items():
            path = ROOT/'outputs'/run/'evaluations'/filename
            if not path.exists():
                continue
            reference = json.loads(path.read_text())
            assert reference['generated_images'] == 50000 and reference['real_images'] == 1281167
            assert reference['train/global_step'] == 626*epoch
            row['baselines'][name] = dict(fid=reference['eval/fid'],
                compact_joint_minus_baseline=current['eval/fid']-reference['eval/fid'])
        curves[str(epoch)] = row
    launcher.write(OUT/'comparison.json', dict(completed_epochs=2, global_step=1252, curves=curves,
        paired_ablation='joint_65k_depthtemp', paired_baseline_missing_epoch_two=False,
        scope='same FID population and optimizer updates; original RQ and compound branches have different tokenizers and samplers'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    else:
        os.environ['WANDB_API_KEY'] = Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
        result = launcher.run(resume=False)
        if result == 0:
            compare()
        raise SystemExit(result)


if __name__ == '__main__':
    main()
