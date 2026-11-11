"""Freeze the earlier compound path and the original tokenizer for a fresh 128-bin run."""
import json
from pathlib import Path
import shutil

from omegaconf import OmegaConf

ROOT = Path('/tmp/laser-compound128-20261107')
BASE = ROOT / 'frozen'
SOURCE = Path('/tmp/laser-aligned-beam451-20261105/frozen')
REPO = Path('/workspace/Projects/laser')
RUNTIME = REPO / 'runtime/compound128-20261107'
RUN = 'imagenet-laser-compound128-native-scratch100-4a100-20261107'


def main():
    assert not BASE.exists(), 'Already prepared; inspect before replacing frozen inputs'
    (BASE / 'inputs').mkdir(parents=True)
    (ROOT / 'quality').mkdir()
    (ROOT / 'production-nvme').mkdir()
    for name in ['source', 'upstream', 'runtime']:
        shutil.copytree(SOURCE / name, BASE / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
    for name in ['laser_native_compound_adapter.py', 'laser_compound_rq_adapter.py']:
        shutil.copyfile(REPO / 'src/models' / name, BASE / 'source/src/models' / name)
    for name in ['stage1-model.pt', 'imagenet_256_train.npz', 'rq-upstream-core.json']:
        (BASE / 'inputs' / name).hardlink_to(SOURCE / 'inputs' / name)
    for name in ['stage2-training-config.yaml', 'stage2-archive-config.yaml']:
        shutil.copyfile(SOURCE / 'inputs' / name, BASE / 'inputs' / name)
    recipe = OmegaConf.load(REPO / 'configs/stage2/imagenet-laser-compound512-native-scratch2-20261016.yaml')
    recipe.run_id = RUN
    recipe.name = 'ImageNet original LASER | compound128 | fresh100'
    recipe.version = 'original-compound128-hard-coefficients-v1'
    recipe.epochs = 100
    recipe.microbatch = 128
    recipe.target_mode = 'hard_sparse_tokens'
    recipe.tokenizer.coefficient_bins = 128
    recipe.targets = dict(encoder_batch_size=8, coefficient_quantization='nearest_bin',
        stochastic_prefixes=False, coefficient_noise=False)
    recipe.objective = 'historical_weighted_atom_and_hard_coefficient_cross_entropy'
    recipe.verification.maximum_reconstruction_rfid = 4.73
    recipe.initialization = dict(fresh_stage2=True, fresh_optimizer=True, stage2_checkpoint_loaded=False)
    recipe.lineage = dict(earlier_compound_recipe='imagenet-laser-compound512-native-scratch2-4h100-20261016',
        replaced_run='imnet-j16384-a4096-native-beam-mix-scratch4-4a100-20261107',
        change='128 coefficient bins, deterministic source sparse support and nearest-bin coefficient targets',
        teacher_mixture=False, atom_reduction=False)
    OmegaConf.save(recipe, BASE / 'inputs/recipe.yaml')
    config = OmegaConf.load(BASE / 'inputs/stage2-training-config.yaml')
    config.loss = dict(type=recipe.objective, atom_ce_weight=1.5, coefficient_ce_weight=1.,
        coefficient_bins=128, stochastic_codes=False, soft_targets=False)
    OmegaConf.save(config, BASE / 'inputs/stage2-training-config.yaml')
    (BASE / 'data').symlink_to('/tmp/laser-recon-4096-20261105/data', target_is_directory=True)
    (ROOT / 'torch-cache').symlink_to('/tmp/laser-recon-4096-20261105/torch-cache', target_is_directory=True)
    for p in RUNTIME.glob('*.py'):
        shutil.copyfile(p, BASE / 'runtime' / p.name)
    shutil.copyfile(SOURCE / 'train.py', BASE / 'train.py')
    p = BASE / 'train.py'
    p.write_text(p.read_text().replace("main('laser_aligned_beam',root)", "main('laser_compound128',root)"))
    p = BASE / 'source/src/training/rq_imagenet_shared.py'
    code = p.read_text().replace("    if kind == 'laser_aligned_beam':",
        "    if kind == 'laser_compound128':\n        from runtime.compound_recipe import Compound128Recipe\n        PROFILE=Compound128Recipe(BASE)\n    elif kind == 'laser_aligned_beam':")
    start = code.index('    with torch.no_grad():\n        detached=logits.detach()')
    end = code.index('\n    if not torch.isfinite(loss)', start)
    code = code[:start] + code[end:]
    code = code.replace("PROFILE.name='ImageNet4096 | beam-aligned soft teacher | continue100'", "PROFILE.name=str(PROFILE.recipe.name)")
    code = code.replace('if not args.smoke_updates and (global_step==10 or global_step%250==0):',
        'if not args.smoke_updates and (global_step in (1,10) or global_step%250==0):')
    code = code.replace('                    runtime_support.sample_panels(PROFILE,model,aux,device,rank,world,global_step,run,OUT)',
        '                    if global_step != 1:runtime_support.sample_panels(PROFILE,model,aux,device,rank,world,global_step,run,OUT)')
    compile(code, str(p), 'exec')
    p.write_text(code)
    p = BASE / 'source/src/training/joint_runtime_support.py'
    code = p.read_text().replace('codes.max()<aux.quantizer.vocab_size', 'codes.max()<aux.vocab_size')
    code = code.replace('vocabulary=aux.quantizer.vocab_size', 'vocabulary=aux.vocab_size')
    code = code.replace('Beam451 aligned soft teacher 100 epochs 4A100 20261105', 'Original LASER compound128 fresh100 4A100 20261107')
    code = code.replace("title='Token cross entropy'", "title='Atom and coefficient cross entropy'")
    first = code.index("        wr.LinePlot(title='Cross entropy by depth")
    last = code.index("        wr.LinePlot(title='Learning rate", first)
    code = code[:first] + "        wr.LinePlot(title='Atom and coefficient CE',x='train/global_step',y=['train/atom_ce','train/coefficient_ce']),\n" + code[last:]
    p.write_text(code)
    public = REPO / 'outputs' / RUN
    public.mkdir(parents=True, exist_ok=True)
    for name in ['status.json', 'launch.json', 'training.log', 'checkpoint.json', 'cloud-sync-status.json',
                 'evaluations', 'source-artifact.json', 'wandb.json', 'samples']:
        (public / name).symlink_to(ROOT / 'production-nvme' / name,
            target_is_directory=name in ('evaluations', 'samples'))
    for name in ['quality', 'tokenizer-verification.json', 'supervisor.log', 'pipeline-status.json']:
        (public / name).symlink_to(ROOT / ('status.json' if name == 'pipeline-status.json' else name),
            target_is_directory=name == 'quality')
    (public / 'live-nvme').symlink_to(ROOT / 'production-nvme', target_is_directory=True)
    print(json.dumps(dict(base=str(BASE), run=RUN, atoms=16384, coefficient_bins=128)), flush=True)


if __name__ == '__main__':
    main()
