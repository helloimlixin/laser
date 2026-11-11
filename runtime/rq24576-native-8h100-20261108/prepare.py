"""Freeze source, native RQ training code and verified data for the new vocabulary."""
import json
from pathlib import Path
import shutil
from omegaconf import OmegaConf
from environment import ROOT, BASE, SOURCE, HERE, OUT, QUALITY, RUN


def main():
    template = Path('/tmp/laser-beam451-8h100-20261108/arms/cutoff05')
    BASE.mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    QUALITY.mkdir(parents=True, exist_ok=True)
    for name in ['source', 'upstream', 'runtime']:
        assert not (BASE / name).exists(), f'Inspect existing prepared directory {BASE/name}'
        shutil.copytree(template / name, BASE / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.caches'))
    (BASE / 'inputs').mkdir(exist_ok=True)
    for name in ['stage1-model.pt', 'imagenet_256_train.npz', 'rq-upstream-core.json']:
        (BASE / 'inputs' / name).hardlink_to(SOURCE / 'inputs' / name)
    for name in ['stage2-training-config.yaml', 'stage2-archive-config.yaml']:
        shutil.copyfile(SOURCE / 'inputs' / name, BASE / 'inputs' / name)
    data = BASE / 'data'
    data.mkdir()
    for name in ['ILSVRC2012_img_train.tar', 'ILSVRC2012_img_val.tar', 'train-index.npz']:
        (data / name).symlink_to(SOURCE / 'data' / name)
    canonical = ROOT.parent / 'data/imagenet/validation-verified-20261107'
    shutil.copyfile(canonical / 'val-index.npz', data / 'val-index.npz')
    shutil.copyfile(canonical / 'verified.json', data / 'verified.json')
    shutil.copyfile(SOURCE / 'data/ready.json', data / 'ready.json')
    shutil.copyfile('/tmp/laser-rq24576-validation-verification.json', QUALITY / 'local-validation-archive.json')
    (BASE / 'torch-cache').symlink_to(SOURCE / 'torch-cache', target_is_directory=True)
    for path in HERE.glob('*.py'):
        shutil.copyfile(path, BASE / 'runtime' / path.name)
    shutil.copyfile(ROOT / 'runtime/original16384-compound128-8h100-20261108/precision.py',
                    BASE / 'runtime/precision.py')
    recipe = OmegaConf.create(dict(
        version='native-rq24576-vectors-v1', run_id=RUN,
        name='ImageNet | native RQ24576 | no beam | fresh100 | 8H100',
        seed=261001, epochs=100, global_batch=2048, microbatch=64, lr_schedule_epochs=100,
        backbone='upstream_rqtransformer', decoder='native_joint_integer',
        target_mode='native_stochastic_residual_codes',
        tokenizer=dict(sha256='dd28db9306d526bdc9fbf8016403e106c4f317fd8f5c04792f0953e53310fdab',
                       codebook_sha256='', vocab_size=24576, depth=4, channels=256,
                       final_codewords=24576, separate_coefficient_tokens=False,
                       learned_compact_dictionary=False, learned_vector_codebooks=True,
                       encoder_decoder_frozen=True, measured_rfid=0., teacher_sampled_rfid=0.),
        targets=dict(encoding='upstream RQBottleneck.get_soft_codes', temperature=.5,
                     stochastic_prefixes=True, measured_entropy=0., beam_search=False,
                     mixture_teacher=False, full_vocabulary=True),
        objective='native_soft_target_cross_entropy', auxiliary_loss_terms=0,
        sampling=dict(temperature=1., top_k=1024, top_p=.95),
        evaluation=dict(generated_images=50000, real_images=1281167, reference_split='train',
                        every_epochs=2, include_epoch_one=True, batch_size=64),
        initialization=dict(reference_vocab_size=24576, paired_fresh_backbone=False,
                            policy='native vocabulary and Stage2Model._init_weights',
                            stage2_checkpoint_loaded=False),
        lineage=dict(source_CNN_checkpoint='original LASER rFID4.21', source_CNN_frozen=True,
                     dictionary_request='24576 final RQ codewords',
                     source_codeword_initialization='16384 existing vectors plus8191 training-residual prototypes and zero',
                     fitting_images=50000, heldout_training_images=10000,
                     validation_used_for_fitting=False, fresh_stage2=True, fresh_optimizer=True)))
    OmegaConf.save(recipe, BASE / 'inputs/recipe.yaml')
    for name in ['stage2-training-config.yaml', 'stage2-archive-config.yaml']:
        config = OmegaConf.load(BASE / 'inputs' / name)
        config.arch.vocab_size = 24576
        config.dataset.vocab_size = 24576
        if name == 'stage2-training-config.yaml':
            config.optimizer.max_gn = .1
            config.experiment.epochs = 100
            config.experiment.batch_size = 64
        OmegaConf.save(config, BASE / 'inputs' / name)
    execution = dict(world_size=8, hardware='8 NVIDIA H100 80GB HBM3', global_batch=2048,
                     microbatch=64, accumulation_steps=4, epochs=100, fresh_stage2=True,
                     fresh_optimizer=True, model_and_optimizer_restored=False,
                     training_autocast_dtype='bfloat16', gradient_clip_max_norm=.1,
                     training_sdpa=True, compiled_native_soft_loss=True, fused_adamw=True,
                     native_quantizer=True, native_stochastic_prefixes=True, temperature=.5,
                     final_codewords=24576, beam_search=False, mixture_teacher=False)
    (BASE / 'inputs/execution.json').write_text(json.dumps(execution, indent=2) + '\n')
    path = BASE / 'source/src/training/rq_imagenet_shared.py'
    code = path.read_text()
    changes = [
        ("    if kind == 'laser_aligned_beam':", "    if kind == 'native24576':\n        from runtime.native_recipe import NativeRecipe\n        PROFILE=NativeRecipe(BASE)\n    elif kind == 'laser_aligned_beam':"),
        ("    from runtime.preservation_guard import verify_frozen_codec\n    preservation_metadata=verify_frozen_codec(PROFILE,aux)",
         "    preservation_metadata=PROFILE.verify_codec(aux)"),
        ("    model,optimizer,config=fresh_model(device)", "    seed_all(int(PROFILE.recipe.seed))\n    model,optimizer,config=fresh_model(device)"),
        ("scaler=torch.amp.GradScaler('cuda',enabled=bool(config.experiment.amp))", "scaler=torch.amp.GradScaler('cuda',enabled=False)"),
        ("training_autocast_dtype='float16'", "training_autocast_dtype='bfloat16'"),
        ("group='beam451-noise-20261108',job_type='noise-ablation'", "group='native-rq24576-20261108',job_type='native-rq-training'"),
    ]
    for old, new in changes:
        assert code.count(old) == 1, old
        code = code.replace(old, new)
    compile(code, str(path), 'exec')
    path.write_text(code)
    path = BASE / 'runtime/checkpoint_store.py'
    code = path.read_text()
    assert code.count('shutil.copyfile(path, temporary)') == 1
    code = code.replace('import torch', 'import torch\nfrom runtime.persistent_copy import copy_file')
    code = code.replace('shutil.copyfile(path, temporary)', 'copy_file(path, temporary)')
    path.write_text(code)
    train = (template / 'train.py').read_text()
    train = train.replace("main('laser_aligned_beam',root)", "main('native24576',root)")
    train = train.replace('from runtime.media_policy',
                          'from runtime.precision import install_bf16\ninstall_bf16()\nfrom runtime.media_policy')
    (BASE / 'train.py').write_text(train)
    path = BASE / 'runtime/generation_interface.py'
    code = path.read_text().replace('<16384', '<aux.quantizer.vocab_size').replace('vocabulary=16384', 'vocabulary=aux.quantizer.vocab_size')
    compile(code, str(path), 'exec')
    path.write_text(code)
    (QUALITY / 'plan.json').write_text(json.dumps(dict(
        final_codewords=24576, residual_depth=4, classes_per_softmax=24576,
        source_CNN_frozen=True, beam_search=False, mixture_teacher=False,
        fit_images=50000, heldout_training_images=10000, validation_images=50000,
        native_temperature=.5, epochs_stage2=100, world_size=8), indent=2) + '\n')
    print(json.dumps(dict(base=str(BASE), quality=str(QUALITY), output=str(OUT))), flush=True)


if __name__ == '__main__':
    main()
