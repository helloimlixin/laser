"""Prepare the verified 4.512 book and the native FP16 AR execution path."""
import hashlib
import json
from pathlib import Path
import shutil
from omegaconf import OmegaConf

ROOT = Path('/workspace/Projects/laser')
SOURCE = Path('/tmp/laser-native4096-generation-20261105/frozen')
BOOK_SOURCE = Path('/tmp/laser-beam16-a100-20261105/frozen')
BASE = Path('/tmp/laser-aligned-beam451-20261105/frozen')
OUT = BASE.parent/'production-nvme'
RUN = 'imnet-j16384-a4096-beam16-aligned-soft-scratch2-4a100-20261105'
PUBLIC = ROOT/'outputs'/RUN


def sha(path):
    with path.open('rb') as handle: return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    assert not BASE.exists(), 'Inspect existing preparation first'
    BASE.mkdir(parents=True); OUT.mkdir(); PUBLIC.mkdir(parents=True, exist_ok=True)
    for name in ['source','upstream','runtime']:
        shutil.copytree(SOURCE/name, BASE/name,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc','.caches'))
    (BASE/'inputs').mkdir()
    for name in ['stage1-model.pt','imagenet_256_train.npz','rq-upstream-core.json']:
        (BASE/'inputs'/name).hardlink_to(SOURCE/'inputs'/name)
    for name in ['joint-codebook.pt','reconstruction-codebook.pt','measured-beam-stage1-rfid.json','paired-tokenizer-artifact.json']:
        shutil.copyfile(BOOK_SOURCE/'inputs'/name, BASE/'inputs'/name)
    for name in ['laser_joint_tokenizer.py','laser_joint_search.py','native_joint_tokenizer.py','rfid_preserved_joint_tokenizer.py']:
        shutil.copyfile(BOOK_SOURCE/'source/src/models'/name, BASE/'source/src/models'/name)
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copyfile(path, BASE/'runtime'/path.name)
    for name in ['stage2-training-config.yaml','stage2-archive-config.yaml']:
        config = OmegaConf.load(SOURCE/'inputs'/name)
        config.arch.vocab_size = config.dataset.vocab_size = 16384
        if name == 'stage2-training-config.yaml':
            config.loss = dict(type='soft_target_cross_entropy',stochastic_codes=True)
            assert config.optimizer.max_gn == 1.
        OmegaConf.save(config, BASE/'inputs'/name)
    recipe = OmegaConf.load(SOURCE/'inputs/recipe.yaml')
    recipe.run_id = RUN
    recipe.name = 'ImageNet4096 | rFID4.512 book | beam-aligned conditional soft teacher | fresh2'
    recipe.version = 'beam451-conditional-soft-v1'
    recipe.epochs = 2
    recipe.tokenizer = dict(sha256=sha(BASE/'inputs/stage1-model.pt'),
        codebook_sha256=sha(BASE/'inputs/joint-codebook.pt'),
        reconstruction_codebook_sha256=sha(BASE/'inputs/reconstruction-codebook.pt'),
        atoms=4096,vocab_size=16384,depth=4,coefficient_levels_per_atom=4,
        measured_rfid=4.5122860099749005,teacher_sampled_rfid=float('nan'),
        learned_compact_dictionary=True,encoder_decoder_frozen=True,
        reconstruction_beam_width=16,teacher_beam_width=16,
        candidate='original verified 4.512 fixed-product book')
    recipe.targets = dict(encoding='beam16 complete-sequence posterior and exact prefix conditionals',
        stochastic_prefixes=True,sequence_temperature=.01,chunk_size=256,
        calibration='training only; pending',max_relative_candidate_error=.01,
        target_mean_conditional_entropy=.20,max_mean_relative_latent_error=.001)
    recipe.initialization = dict(reference_vocab_size=16384,paired_fresh_backbone=False,
        policy='native vocabulary and Stage2Model._init_weights',stage2_checkpoint_loaded=False)
    recipe.lineage = dict(tokenizer='verified rFID4.512 beam16 book; physical values unchanged',
        source_checkpoint_loaded=False,
        source_89fid_run='imnet-j32769-a8192-upstream-native-scratch-2b300-20261031',
        source_epoch1_fid=89.12546967078106,source_epoch2_fid=73.73770937849014,
        earlier_hard_beam_run='imnet-j16384-a4096-beam16-rfid451-scratch100-2b300-20261104',
        comparison_epochs=[1,2],generation_quality_unmeasured=True)
    OmegaConf.save(recipe, BASE/'inputs/recipe.yaml')
    execution = dict(align_classifier=False,valid_classes=16384,padded_projection_width=16384,
        extra_softmax_classes=0,plain_upstream_classifier=True,native_compute_loss=True,
        world_size=4,global_batch=2048,hardware='4 NVIDIA A100-SXM4-80GB',
        checkpoint_storage='NVMe atomic checkpoints; separate retrying W&B uploader',
        training_autocast_dtype='float16',gradient_clip_max_norm=1.,
        training_targets='beam-aligned conditional soft posterior',
        reconstruction_beam_width=16,teacher_beam_width=16,fresh_stage2=True,
        fresh_optimizer=True,pilot_stop_epoch=2,automatic_restart=True)
    (BASE/'inputs/execution.json').write_text(json.dumps(execution,indent=2)+'\n')
    (BASE/'data').symlink_to('/tmp/laser-recon-4096-20261105/data',target_is_directory=True)
    (BASE.parent/'torch-cache').symlink_to('/tmp/laser-recon-4096-20261105/torch-cache',target_is_directory=True)
    (BASE.parent/'quality').mkdir()
    p=BASE/'train.py'
    p.write_text((SOURCE/'train.py').read_text().replace("main('laser_joint',root)","main('laser_aligned_beam',root)"))
    p=BASE/'source/src/training/rq_imagenet_shared.py';code=p.read_text()
    code=code.replace("    if kind == 'laser_signed_joint':", "    if kind == 'laser_aligned_beam':\n        from runtime.aligned_recipe import AlignedBeamRecipe\n        PROFILE=AlignedBeamRecipe(BASE)\n    elif kind == 'laser_signed_joint':")
    compile(code,str(p),'exec');p.write_text(code)
    p=BASE/'source/src/training/joint_runtime_support.py';code=p.read_text()
    code=code.replace("'Native4096 soft AR pilot epochs1 and2 4A100 20261105'", "'Beam451 aligned soft teacher epochs1 and2 4A100 20261105'")
    start=code.index('def save_checkpoint(');stop=code.index('\n\n@torch.no_grad()',start)
    code=code[:start]+'from runtime.checkpoint_store import save_checkpoint\n'+code[stop:]
    code=code.replace('Native teacher entropy by depth','Aligned teacher entropy by depth')
    compile(code,str(p),'exec');p.write_text(code)
    for name in ['status.json','launch.json','training.log','checkpoint.json','supervisor.log','execution-benchmarks.json','selected-execution.json','evaluations','cloud-sync-status.json']:
        (PUBLIC/name).symlink_to(OUT/name,target_is_directory=name=='evaluations')
    (PUBLIC/'live-nvme').symlink_to(OUT,target_is_directory=True)
    (PUBLIC/'quality').symlink_to(BASE.parent/'quality',target_is_directory=True)
    print(json.dumps(dict(run_id=RUN,base=str(BASE),source_rfid=4.5122860099749005)),flush=True)


if __name__=='__main__':main()
