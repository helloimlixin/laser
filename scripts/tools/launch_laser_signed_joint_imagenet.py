#!/usr/bin/env python3
"""Launch the signed-atom fix from scratch after exact-codec and GPU verification."""
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

RUN = 'imagenet-laser-signed-joint512-scratch6-4h100-20261015'
BASE = Path('/tmp/laser-rq-signed-joint512-imagenet-20261015')
OUT = ROOT/'outputs'/RUN
SOURCE = Path('/tmp/laser-rq-fine-joint512-imagenet-20261015')
OLD = ROOT/'outputs/imagenet-laser-fine-joint512-scratch6-4h100-20261015'
RECIPE = ROOT/'configs/stage2/imagenet-laser-signed-joint512-scratch6-20261015.yaml'
launcher.BASE, launcher.OUT, launcher.RUN = BASE, OUT, RUN


def prepare():
    import yaml
    if (OUT/'prepared.json').exists(): raise RuntimeError('Already prepared')
    recipe = yaml.safe_load(RECIPE.read_text())
    original = yaml.safe_load((SOURCE/'inputs/recipe.yaml').read_text())
    assert recipe['run_id'] == RUN and recipe['decoder'] == 'signed_atom_joint_integer'
    assert recipe['epochs'] == 6 and recipe['global_batch'] == 2048 and recipe['lr_schedule_epochs'] == 100
    assert recipe['auxiliary_loss_terms'] == 0 and recipe['tokenizer'] == original['tokenizer']
    assert recipe['targets'] == original['targets'] and recipe['sampling'] == original['sampling']
    assert recipe['evaluation'] == original['evaluation']
    # Reuse the completed full-50k audit only when the tokenizer and all of its
    # coordinate/encoding dependencies are byte-identical to that audited run.
    tokenizer_files = ['src/models/laser_fine_joint_tokenizer.py','src/models/laser_tokenizer_adapter.py',
                       'src/training/laser_hard_ar.py','src/orthogonal_sparse_codec.py']
    tokenizer_hashes = {}
    for name in tokenizer_files:
        digest = launcher.digest(ROOT/name)
        assert digest == launcher.digest(SOURCE/'source'/name), name
        tokenizer_hashes[name] = digest
    fidelity = json.loads((OLD/'tokenizer-fidelity-verification.json').read_text())
    policy = json.loads((OLD/'coefficient-policy-verification.json').read_text())
    assert fidelity['passed'] and fidelity['real_images'] == fidelity['reconstructed_images'] == 50000
    assert policy['passed'] and policy['selected_sigma'] == recipe['targets']['physical_sigma']
    (BASE/'inputs').mkdir(parents=True); OUT.mkdir(parents=True, exist_ok=True)
    freeze_shared_runtime(BASE, ROOT, include_stochastic_teacher=False)
    shutil.copytree(SOURCE/'upstream', BASE/'upstream',
        ignore=shutil.ignore_patterns('__pycache__','.git','*.pyc'))
    (BASE/'train.py').write_text("from runtime.rq_imagenet_entry import launch\nif __name__ == '__main__':\n    launch('laser_signed_joint')\n")
    shutil.copyfile(ROOT/'scripts/tools/verify_laser_signed_joint_checkpoint.py', BASE/'verify.py')
    for name in ['stage2-training-config.yaml','stage2-archive-config.yaml','imagenet_256_train.npz',
                 'rq-upstream-core.json']:
        shutil.copyfile(SOURCE/'inputs'/name, BASE/'inputs'/name)
    config = yaml.safe_load((BASE/'inputs/stage2-training-config.yaml').read_text())
    assert config['optimizer']['init_lr'] == .0005
    os.link(SOURCE/'inputs/stage1-model.pt', BASE/'inputs/stage1-model.pt')
    assert launcher.digest(BASE/'inputs/stage1-model.pt') == recipe['tokenizer']['sha256'] == fidelity['source_sha256']
    assert launcher.digest(BASE/'inputs/imagenet_256_train.npz') == recipe['evaluation']['reference_sha256']
    shutil.copyfile(RECIPE, BASE/'inputs/recipe.yaml')
    for name in ['data','torch-cache']: (BASE/name).symlink_to(SOURCE/name, target_is_directory=True)
    launcher.write(BASE/'environment.json', dict(LASER_RQ_BASE=str(BASE),
        LASER_PRIVATE_SMOKE_DIR=str(BASE/'private-smoke'),
        LASER_CHECKPOINT_STAGING_DIR=str(BASE/'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(BASE/'checkpoint-cache'), LASER_CHECKPOINT_IMMUTABLE_FILES='1',
        TORCH_HOME=str(BASE/'torch-cache'), WANDB_DIR=str(BASE/'wandb'),
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
    repro = OUT/'reproducibility'; repro.mkdir(exist_ok=True)
    for p in (BASE/'inputs').iterdir():
        if p.suffix in ('.yaml','.json'): shutil.copyfile(p, repro/p.name)
    for name in ['tokenizer-fidelity-verification.json','coefficient-policy-verification.json']:
        shutil.copyfile(OLD/name, repro/name)
    shutil.copyfile(OLD/'diagnosis-epoch001/diagnosis-summary.json', repro/'previous-head-diagnosis.json')
    launcher.write(OUT/'tokenizer-fidelity-verification.json', dict(fidelity,
        inherited_from_run=OLD.name, tokenizer_and_encoding_files_identical=True,
        file_hashes=tokenizer_hashes, coefficient_grid_and_scales_identical=True,
        sign_magnitude_mapping_verified_by_exhaustive_bin_test=True))
    files = [BASE/'train.py',BASE/'verify.py',BASE/'environment.json']
    for directory in ['source','runtime','upstream']: files.extend((BASE/directory).rglob('*.py'))
    files.extend((BASE/'inputs').glob('*.yaml')); files.extend((BASE/'inputs').glob('*.json'))
    launcher.write(repro/'code-manifest.json',{str(p.relative_to(BASE)):launcher.digest(p) for p in files})
    with zipfile.ZipFile(repro/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in files: archive.write(p,p.relative_to(BASE))
        archive.write(Path(__file__),'scripts/tools/'+Path(__file__).name)
        archive.write(ROOT/'tests/test_laser_signed_joint.py','tests/test_laser_signed_joint.py')
    launcher.write(OUT/'prepared.json', dict(prepared=True,run_id=RUN,recipe=recipe,
        fresh_stage2=True,fresh_optimizer=True,source_stage2_checkpoint=None,
        initial_global_step=0,initial_optimizer_state_count=0,train_images=1281167,global_batch=2048,
        updates_per_epoch=626,stop_updates=3756,cosine_total_updates=62600,evaluation_epochs=[1,2,4,6],
        initial_lr=.0005,auxiliary_loss_terms=0,sampled_trajectories=0,native_transformer=True,
        main_signed_atom_classes=32768,magnitude_bins=256,signed_coefficient_values=512,
        packed_id_rule='atom*512+coefficient_bin',soft_coefficient_posterior_preserved_exactly=True,
        tokenizer_fidelity_reused_from_identical_tokenizer=True,private_smoke_checkpoint_storage='local NVMe, discarded for fresh production',
        source_manifest_sha256=launcher.digest(repro/'code-manifest.json'),time=time.time()))
    print('Prepared signed atom / magnitude run', RUN, flush=True)


def run():
    result = launcher.run(resume=False)
    if result == 0:
        reference = ROOT/'outputs/imagenet-original-rq1400m-rfid473-scratch20-4h100-20261014/evaluations'
        comparisons = {}
        for epoch in [1,2,4,6]:
            name = f'epoch{epoch:03d}.json'
            actual = json.loads((OUT/'evaluations'/name).read_text())
            rq = json.loads((reference/name).read_text())
            assert actual['generated_images'] == rq['generated_images'] == 50000
            assert actual['real_images'] == rq['real_images'] == 1281167
            comparisons[str(epoch)] = dict(signed_joint_fid=actual['eval/fid'],rq_fid=rq['eval/fid'],
                difference=actual['eval/fid']-rq['eval/fid'])
        launcher.write(OUT/'comparison.json',dict(completed_epochs=6,updates=3756,curves=comparisons,
            continued_beyond_six_epochs=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run']); args=parser.parse_args()
    if args.action == 'prepare': prepare()
    else:
        os.environ['WANDB_API_KEY'] = Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
        raise SystemExit(run())
