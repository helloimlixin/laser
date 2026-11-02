#!/usr/bin/env python3
"""Freeze a fresh full-pair-ID trial without interrupting running experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    import yaml
    from runtime.rq_imagenet_entry import freeze_shared_runtime
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('/tmp/laser-compound512-origlr-imagenet-20261016-r3'))
    parser.add_argument('--recipe',type=Path,default=ROOT/'configs/stage2/imagenet-laser-packed-pair512-proj128-scratch2-20261017.yaml')
    parser.add_argument('--base',type=Path,default=Path('/tmp/laser-packed-pair512-proj128-imagenet-20261017'))
    parser.add_argument('--audit',type=Path,default=ROOT/'outputs/imagenet-laser-packed-pair512-audit-20261017')
    args=parser.parse_args()
    recipe=yaml.safe_load(args.recipe.read_text())
    run=recipe['run_id'];out=ROOT/'outputs'/run
    if (out/'prepared.json').exists() or (args.base/'inputs/recipe.yaml').exists():
        raise RuntimeError('Preparation already exists; preserve the frozen runtime')
    assert recipe['tokenizer']['num_atoms']==16384 and recipe['tokenizer']['coefficient_bins']==512
    assert recipe['tokenizer']['vocab_size']==8388608 and recipe['auxiliary_loss_terms']==0
    assert recipe['global_batch']==2048 and recipe['lr_schedule_epochs']==100
    assert recipe['optimization']['peak_lr']==.0005 and recipe['optimization']['warmup_epochs']==0
    assert recipe['initialization']['paired_fresh_backbone']
    for name in ['full-vocabulary-preflight.json','real-tokenizer-parity.json']:
        assert json.loads((args.audit/name).read_text())['passed']
    assert digest(args.source/'inputs/stage1-model.pt')==recipe['tokenizer']['sha256']
    assert digest(args.source/'inputs/imagenet_256_train.npz')==recipe['evaluation']['reference_sha256']
    (args.base/'inputs').mkdir(parents=True)
    (out/'reproducibility').mkdir(parents=True)
    freeze_shared_runtime(args.base,ROOT,include_stochastic_teacher=False)
    for name in ['upstream','data','torch-cache']:
        (args.base/name).symlink_to((args.source/name).resolve(),target_is_directory=True)
    for name in ['stage1-model.pt','imagenet_256_train.npz']:
        (args.base/'inputs'/name).symlink_to((args.source/'inputs'/name).resolve())
    for name in ['stage2-training-config.yaml','stage2-archive-config.yaml','rq-upstream-core.json']:
        shutil.copyfile(args.source/'inputs'/name,args.base/'inputs'/name)
    shutil.copyfile(args.recipe,args.base/'inputs/recipe.yaml')
    training=yaml.safe_load((args.base/'inputs/stage2-training-config.yaml').read_text())
    training['loss']=dict(type='single_global_pair_cross_entropy',coefficient_temperature=.01125,
        stochastic_codes=True,auxiliary_loss_terms=0)
    training['experiment'].update(epochs=int(recipe['epochs']),batch_size=int(recipe['microbatch']),
        total_batch_size=2048,save_ckpt_freq=1,test_freq=2)
    assert training['optimizer']['init_lr']==.0005 and training['optimizer']['warmup']['epoch']==0
    (args.base/'inputs/stage2-training-config.yaml').write_text(yaml.safe_dump(training,sort_keys=False))
    (args.base/'train.py').write_text("from runtime.rq_imagenet_entry import launch\nif __name__=='__main__':\n    launch('laser_packed_integer')\n")
    shutil.copyfile(ROOT/'scripts/tools/verify_packed_integer_smoke.py',args.base/'verify.py')
    shutil.copyfile(ROOT/'scripts/tools/configure_rq_sample_panels.py',args.base/'configure_samples.py')
    environment=dict(LASER_RQ_BASE=str(args.base),LASER_RQ_SAFE_CHECKPOINTS='1',
        LASER_CHECKPOINT_STAGING_DIR=str(args.base/'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(args.base/'checkpoint-cache'),
        LASER_CHECKPOINT_IMMUTABLE_FILES='1',TORCH_HOME=str(args.base/'torch-cache'),
        WANDB_DIR=str(args.base/'wandb'),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NCCL_NVLS_ENABLE='0',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    (args.base/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    files=[args.base/'train.py',args.base/'verify.py',args.base/'configure_samples.py',args.base/'environment.json']
    files+=list((args.base/'source').rglob('*.py'))+list((args.base/'runtime').rglob('*.py'))
    files+=list((args.base/'upstream').rglob('*.py'))
    files+=list((args.base/'inputs').glob('*.yaml'))+list((args.base/'inputs').glob('*.json'))
    manifest={str(p.relative_to(args.base)):digest(p) for p in files}
    (out/'reproducibility/code-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for name in ['full-vocabulary-preflight.json','real-tokenizer-parity.json','proj128.json','tile-benchmark.json','final-tests.xml']:
        shutil.copyfile(args.audit/name,out/'reproducibility'/name)
    for path in (args.base/'inputs').glob('*.yaml'):
        shutil.copyfile(path,out/'reproducibility'/path.name)
    with zipfile.ZipFile(out/'reproducibility/source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in files:archive.write(path,path.relative_to(args.base))
        for path in [Path(__file__),ROOT/'tests/test_laser_packed_integer.py']:
            archive.write(path,path.relative_to(ROOT))
    plan=dict(prepared=True,training_launched=False,run_id=run,recipe=recipe,
        fresh_stage2=True,fresh_optimizer=True,stage2_checkpoint_loaded=False,
        initial_global_step=0,optimizer_updates_per_epoch=626,global_batch=2048,
        cosine_total_updates=62600,stop_after_updates=626*int(recipe['epochs']),
        whole_model_gpu_optimizer_smoke_required=True,
        existing_training_unchanged=True,source_manifest_sha256=digest(out/'reproducibility/code-manifest.json'),
        base=str(args.base),created_at=time.time())
    (out/'prepared.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(prepared=True,training_launched=False,run_id=run,base=str(args.base),
        whole_model_gpu_optimizer_smoke_required=True)))


if __name__=='__main__':main()
