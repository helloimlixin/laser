#!/usr/bin/env python3
"""Launch a fresh CC3M prior using the audited ImageNet RNG fix and cached data."""
import argparse
import json
import shutil
from pathlib import Path
import sys
import zipfile

import yaml

import launch_cc3m_joint20k as launcher


ROOT = launcher.ROOT
PREVIOUS_BASE = launcher.BASE
PREVIOUS_OUT = launcher.OUT
RUN = 'cc3m-laser-joint20k-depth5-rqnoise-k2048-rngfix-scratch100-8a100-20261017'
BASE = Path('/tmp/laser-cc3m-joint20k-depth5-rngfix-8a100-20261017')
OUT = ROOT/'outputs'/RUN
FIXED_IMAGENET = 'imagenet-laser-joint20k-depth5-rqnoise-k2048-rngfix-scratch100-8h100-20261017'
CHANGED = ('src/training/cc3m_joint.py', 'src/training/rq_rng.py',
           'src/training/cc3m_storage.py',
           'src/training/rq_precision.py', 'src/training/cc3m_preview.py',
           'src/data/cc3m_joint.py', 'src/data/seeded_bpe.py')


def prepare():
    if BASE.exists():
        raise RuntimeError(f'Fresh preparation refuses an existing runtime: {BASE}')
    old = yaml.safe_load((PREVIOUS_BASE/'config.yaml').read_text())
    fixed = yaml.safe_load((ROOT/'configs/stage2'/f'{FIXED_IMAGENET}.yaml').read_text())
    assert old['tokenizer_recipe'] == fixed['tokenizer']
    assert old['temperatures'] == fixed['targets']['temperature']
    assert old['sampling'] == fixed['sampling']
    for key, path in [('stage1_sha256', old['stage1']), ('codebook_sha256', old['codebook'])]:
        assert launcher.sha(path) == old[key]
    BASE.mkdir(parents=True)
    OUT.mkdir(parents=True)
    repro = OUT/'reproducibility'
    repro.mkdir()
    shutil.copytree(PREVIOUS_BASE/'source', BASE/'source', symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(PREVIOUS_BASE/'upstream', BASE/'upstream', symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
    metrics_vendor = BASE/'source/third_party/rq-vae-transformer'
    metrics_vendor.unlink()
    metrics_vendor.symlink_to(BASE/'upstream', target_is_directory=True)
    for name in CHANGED:
        shutil.copyfile(ROOT/name, BASE/'source'/name)
    for name in ('stage1.pt', 'data', 'torch-cache', 'reference'):
        (BASE/name).symlink_to(PREVIOUS_BASE/name, target_is_directory=(PREVIOUS_BASE/name).is_dir())
    for name in ('data-manifest.json', 'official-cc3m.yaml'):
        shutil.copyfile(PREVIOUS_BASE/name, BASE/name)
    for name in ('wandb', 'checkpoints', 'logs'):
        (BASE/name).mkdir()
    for name in ('train', 'preflight'):
        path = BASE/'live'/name
        path.mkdir(parents=True)
        (OUT/name).symlink_to(path, target_is_directory=True)
    cfg = dict(old)
    cfg.update(run_id=RUN, name='CC3M LASER joint20k depth5 | RNG fixed | fresh 650M | 8 A100',
        source_run='helloimlixin-rutgers/laser/'+FIXED_IMAGENET,
        original_source_run=old['source_run'], replaces_run=old['run_id'],
        output=str(OUT), local_checkpoints=str(BASE/'checkpoints'),
        stage1=str(BASE/'stage1.pt'), codebook=str(BASE/'reference/inputs/joint-codebook.pt'),
        data_manifest=str(BASE/'data-manifest.json'), data_local=str(BASE/'data'),
        wandb_dir=str(BASE/'wandb'),
        training_rng=dict(policy='independent_rank_v1', base_seed=old['seed']),
        evaluation_rng_policy='isolated_python_numpy_cpu_and_local_cuda_including_metric_setup',
        dataloader_rng_policy='dedicated_generator_and_cpu_only_augmentation_seed',
        initialization_rng_reset_after_tokenizer=True,
        stage2_initialization='fresh_random', stage2_checkpoint_loaded=False,
        checkpoint_uploads=['last.pt', 'best-fid.pt', 'best-clip_score.pt'])
    for key in ('arch', 'tokenizer_recipe', 'temperatures', 'sampling', 'global_batch',
                'epochs', 'lr', 'betas', 'weight_decay', 'image_transform', 'augmentation_sha256'):
        assert cfg[key] == old[key]
    (BASE/'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    shutil.copyfile(BASE/'config.yaml', ROOT/'configs/stage2'/f'{RUN}.yaml')
    shutil.copyfile(BASE/'config.yaml', repro/'recipe.yaml')
    for name in ('augmentation-verification.json', 'upstream-cc3m-transforms.py',
                 'upstream-cc3m650m.yaml', 'requirements-freeze.txt', 'metric-preflight.json'):
        shutil.copyfile(PREVIOUS_OUT/'reproducibility'/name, repro/name)
    assert launcher.sha(BASE/'upstream/rqvae/txtimg_datasets/transforms.py') == cfg['augmentation_sha256']
    benchmark = json.loads((PREVIOUS_OUT/'reproducibility/benchmark-bf16.json').read_text())
    benchmark['reused_from'] = old['run_id']
    benchmark['reuse_reason'] = 'Same A100 hardware, architecture, BF16 implementation and microbatch'
    launcher.write(OUT/'benchmark.json', benchmark)
    shutil.copyfile('/tmp/cc3m-rngfix-tests.xml', repro/'tests.xml')
    launcher.write(repro/'fix-audit.json', dict(passed=True, fixed_imagenet_run=cfg['source_run'],
        unchanged_tokenizer_dictionary_temperatures_sampler=True, original_augmentation_unchanged=True,
        fresh_transformer=True, fresh_optimizer=True, previous_stage2_weights_loaded=False,
        fixes=['Audited independent rank RNG after DDP',
               'Isolate Python, NumPy, CPU and local CUDA RNG for full preview/evaluation including metric setup',
               'Prevent data-loader and augmentation seeds from advancing training RNG',
               'Initialize fresh Transformer independently of frozen tokenizer construction',
               'Retain true native BF16 and explicit uint8 preview encoding'],
        local_data_staging=json.loads((PREVIOUS_BASE/'staging.json').read_text())))
    files = list((BASE/'source/src').rglob('*.py'))+list((BASE/'upstream').rglob('*.py'))
    launcher.write(repro/'code-manifest.json', {str(p.relative_to(BASE)): launcher.sha(p) for p in files})
    with zipfile.ZipFile(repro/'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(BASE))
        for path in (BASE/'upstream/rqvae/txtimg_datasets/tokenizers/pretrained').iterdir():
            if path.is_file():
                archive.write(path, path.relative_to(BASE))
        for path in (BASE/'config.yaml', BASE/'data-manifest.json', Path(cfg['codebook'])):
            archive.write(path, path.relative_to(BASE))
        archive.write(__file__, Path(__file__).name)
        archive.write(launcher.__file__, 'launch_cc3m_joint20k.py')
    print(json.dumps(dict(prepared=True, run_id=RUN, production_checkpoints=[])), flush=True)


def run():
    launcher.BASE, launcher.OUT, launcher.RUN = BASE, OUT, RUN
    launcher.supervise(resume_script=Path(__file__).resolve())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'run'])
    args = parser.parse_args()
    try:
        {'prepare': prepare, 'run': run}[args.phase]()
    except BaseException as error:
        launcher.write(BASE/'supervisor/status.json', dict(phase='failed', error=str(error)))
        raise
