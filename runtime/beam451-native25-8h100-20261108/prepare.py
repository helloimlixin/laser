"""Freeze the beam source and explicitly record the teacher continuation."""
import hashlib
import json
from pathlib import Path
import shutil
from omegaconf import OmegaConf
from environment import BASE, OUT, HERE, ROOT, RUN, SOURCE_RUN


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')


def main():
    source = Path('/tmp/laser-beam451-8h100-20261108/arms/cutoff05')
    OUT.mkdir(parents=True, exist_ok=True)
    BASE.mkdir(parents=True, exist_ok=True)
    for name in ['source', 'upstream', 'runtime']:
        target = BASE / name
        assert not target.exists(), f'Inspect existing preparation: {target}'
        shutil.copytree(source / name, target,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.caches'))
    (BASE / 'inputs').mkdir(exist_ok=True)
    for name in ['stage1-model.pt', 'joint-codebook.pt', 'reconstruction-codebook.pt',
                 'imagenet_256_train.npz', 'rq-upstream-core.json']:
        (BASE / 'inputs' / name).hardlink_to(source / 'inputs' / name)
    for name in ['stage2-training-config.yaml', 'stage2-archive-config.yaml',
                 'paired-tokenizer-artifact.json']:
        shutil.copyfile(source / 'inputs' / name, BASE / 'inputs' / name)
    (BASE / 'data').symlink_to(source / 'data', target_is_directory=True)
    (BASE / 'torch-cache').symlink_to('/tmp/laser-beam451-8h100-20261108/torch-cache', target_is_directory=True)
    shutil.copyfile(ROOT / 'runtime/native-beam-mixture-20261107/mixture_teacher.py',
                    BASE / 'runtime/mixture_teacher.py')
    shutil.copyfile(ROOT / 'runtime/original16384-compound128-8h100-20261108/precision.py',
                    BASE / 'runtime/precision.py')
    for path in HERE.glob('*.py'):
        shutil.copyfile(path, BASE / 'runtime' / path.name)

    recipe = OmegaConf.load(ROOT / 'outputs' / SOURCE_RUN / 'reproducibility/recipe.yaml')
    recipe.run_id = RUN
    recipe.name = 'ImageNet | beam75 native25 | BF16 clip0.1 | ep12 to100 | 8H100'
    recipe.version = 'beam-native25-bf16-clip01-continuation-v1'
    recipe.targets = dict(
        encoding='native-soft/beam-best complete-sequence mixture',
        native_weight=.25, native_temperature=.5, stochastic_prefixes=True,
        chunk_size=256, measured_entropy=.6239441336870194,
        full_vocabulary_native_component=True, exact_prefix_conditionals=True,
        calibration='50000 class-balanced training views; previously replicated full50k reconstruction')
    recipe.tokenizer.teacher_sampled_rfid = 4.706071704384215
    recipe.tokenizer.teacher_rfid_independent_seed = 4.728797450372554
    recipe.microbatch = 128
    recipe.lineage = dict(
        source_run=SOURCE_RUN, source_step=7512, source_epoch=12,
        model_loaded=True, optimizer_loaded=True, scheduler_loaded=True,
        new_wandb_run=True, source_evaluation_sampler_differs=True,
        teacher_quality_artifact='helloimlixin-rutgers/laser/imagenet-j16384-a4096-native-beam-mixture-teacher-20261107:v1',
        prior_mixture_training_stopped_at_epoch=2.22, generation_improvement_unproven=True)
    OmegaConf.save(recipe, BASE / 'inputs/recipe.yaml')
    config = OmegaConf.load(BASE / 'inputs/stage2-training-config.yaml')
    config.optimizer.max_gn = .1
    config.experiment.batch_size = 64
    OmegaConf.save(config, BASE / 'inputs/stage2-training-config.yaml')
    execution = json.loads((source / 'inputs/execution.json').read_text())
    execution.update(
        fresh_stage2=False, fresh_optimizer=False, model_and_optimizer_restored=True,
        continuation_from_step=7512, continuation_from_epoch=12, epochs=100,
        training_autocast_dtype='bfloat16', grad_scaler_enabled=False,
        gradient_clip_max_norm=.1, teacher='75% beam-best + 25% native-soft',
        native_temperature=.5, generation_batch_size=64)
    execution.pop('teacher_sequence_temperature', None)
    execution.pop('teacher_max_relative_candidate_error', None)
    write(BASE / 'inputs/execution.json', execution)

    path = BASE / 'source/src/training/rq_imagenet_shared.py'
    code = path.read_text()
    replacements = [
        ("    if kind == 'laser_aligned_beam':", "    if kind == 'laser_mixture':\n        from runtime.mixture_recipe import MixtureRecipe\n        PROFILE=MixtureRecipe(BASE)\n    elif kind == 'laser_aligned_beam':"),
        ("assert payload['run_id'] in (RUN,PROFILE.run_id)", "assert payload['run_id'] in (RUN,PROFILE.run_id,*PROFILE.allowed_source_runs())"),
        ("scaler=torch.amp.GradScaler('cuda',enabled=bool(config.experiment.amp))", "scaler=torch.amp.GradScaler('cuda',enabled=False)"),
        ("restored_evaluation=payload.get('evaluation')", "restored_evaluation=payload.get('evaluation')\n        if payload['run_id'] != RUN:\n            restored_evaluation=None\n            restored_best_fid=math.inf\n            write_json(folder/f'source-transition-rank{rank}.json',PROFILE.source_transition)"),
        ("training_autocast_dtype='float16'", "training_autocast_dtype='bfloat16'"),
        ("group='beam451-noise-20261108',job_type='noise-ablation'", "group='beam451-native25-20261108',job_type='beam-continuation'"),
        ("if not args.smoke_updates and global_step==1:", "if not args.smoke_updates and global_step in (1,7513):"),
    ]
    for old, new in replacements:
        assert code.count(old) == 1, old
        code = code.replace(old, new)
    compile(code, str(path), 'exec')
    path.write_text(code)
    path = BASE / 'source/src/training/joint_runtime_support.py'
    code = path.read_text()
    old = "metadata={'native_rq':True,'fresh':True,'source_epoch':0,'warmup_epochs':0}"
    new = "metadata={'native_rq':True,'fresh':False,'source_epoch':12,'source_step':7512,'warmup_epochs':0}"
    assert code.count(old) == 1
    path.write_text(code.replace(old, new))
    train = (source / 'train.py').read_text()
    train = train.replace("main('laser_aligned_beam',root)", "main('laser_mixture',root)")
    train = train.replace("from runtime.media_policy", "from runtime.precision import install_bf16\ninstall_bf16()\nfrom runtime.media_policy")
    (BASE / 'train.py').write_text(train)
    # Publish reconstruction evidence without redoing two completed full50k checks.
    quality = OUT / 'quality'
    quality.mkdir(exist_ok=True)
    for name in ['full-results.json', 'replicate-results.json']:
        shutil.copyfile(Path('/tmp/laser-prior-mixture-quality/verification') / name,
                        quality / ('prior-' + name))
    proof = dict(
        source_run=SOURCE_RUN, source_step=7512, source_epoch=12,
        prior_full50k_teacher_rfid=[4.706071704384215, 4.728797450372554],
        native_weight=.25, native_temperature=.5, gpus=8, epochs_total=100,
        changes=['teacher native25/beam75', 'BF16 training', 'gradient clip0.1',
                 'all8 H100', 'standard sampler T1 k1024 p0.95'],
        dictionary='preserved beam learned4096 atoms x4 coefficient levels =16384 joint IDs',
        discarded_run='imagenet-original16384-compound128-scratch100-8h100-20261108',
        generation_improvement_unproven=True)
    write(OUT / 'plan.json', proof)
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
