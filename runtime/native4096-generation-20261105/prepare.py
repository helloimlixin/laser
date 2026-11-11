"""Fresh compact-vocabulary pilot, retaining the 89-FID run's native AR recipe."""
import hashlib
import json
from pathlib import Path
import shutil

from omegaconf import OmegaConf

ROOT = Path('/workspace/Projects/laser')
SOURCE = Path('/tmp/laser-j32769-a8192-20261105/frozen')
BASE = Path('/tmp/laser-native4096-generation-20261105/frozen')
OUT = BASE.parent / 'production-nvme'
TOKENIZER = ROOT / 'outputs/imagenet-native4096-below473-20261105'
RUN = 'imnet-native4096-soft-scratch2-4a100-20261105'
PUBLIC = ROOT / 'outputs' / RUN


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    assert not BASE.exists(), 'Inspect the existing pilot before preparing again'
    proof = json.loads((TOKENIZER / 'paired-teacher-verification.json').read_text())
    assert proof['passed'] and proof['full_validation_rfid'] < 4.73
    receipt = json.loads((TOKENIZER / 'cloud-verification.json').read_text())
    assert receipt['state'] == 'COMMITTED'
    BASE.mkdir(parents=True)
    OUT.mkdir()
    PUBLIC.mkdir(parents=True, exist_ok=True)
    for name in ['source', 'upstream', 'runtime']:
        shutil.copytree(SOURCE / name, BASE / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.caches'))
    (BASE / 'inputs').mkdir()
    for name in ['stage1-model.pt', 'imagenet_256_train.npz', 'rq-upstream-core.json']:
        (BASE / 'inputs' / name).hardlink_to(SOURCE / 'inputs' / name)
    for name in ['stage2-training-config.yaml', 'stage2-archive-config.yaml']:
        config = OmegaConf.load(SOURCE / 'inputs' / name)
        config.arch.vocab_size = 16385
        config.dataset.vocab_size = 16385
        OmegaConf.save(config, BASE / 'inputs' / name)
    for name in ['joint-codebook.pt', 'tokenizer-policy.json', 'paired-teacher-verification.json']:
        shutil.copyfile(TOKENIZER / name, BASE / 'inputs' / name)
    for name in ['paired_tokenizer.py', 'beam_search.py']:
        shutil.copyfile(TOKENIZER / name, BASE / 'runtime' / name)
    (BASE / 'inputs/paired-tokenizer-artifact.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (BASE / 'data').symlink_to('/tmp/laser-recon-4096-20261105/data', target_is_directory=True)
    recipe = OmegaConf.load(SOURCE / 'inputs/recipe.yaml')
    recipe.version = 'laser-native-joint16k-paired-reconstruction-v1'
    recipe.run_id = RUN
    recipe.name = 'ImageNet 4096 atoms | native soft AR | fresh2 | rFID4.624 paired recon'
    recipe.epochs = 2
    recipe.tokenizer.codebook_sha256 = digest(BASE / 'inputs/joint-codebook.pt')
    recipe.tokenizer.vocab_size = 16385
    recipe.tokenizer.atoms = 4096
    recipe.tokenizer.measured_rfid = proof['full_validation_rfid']
    recipe.tokenizer.candidate = 'compact4096-unitgain-reconstruction-beam4'
    recipe.tokenizer.reconstruction_beam_width = 4
    recipe.tokenizer.teacher_encoding_beam_width = 1
    recipe.tokenizer.source_reported_rfid = 4.6317537087924165
    recipe.targets.calibration.policy = 'Same fixed per-depth temperatures as native8192 source; not refitted'
    recipe.initialization.reference_vocab_size = 16385
    recipe.lineage = dict(source_run='imnet-j32769-a8192-upstream-native-scratch-2b300-20261031',
        source_checkpoint_loaded=False, source_epoch1_fid=89.12546967078106,
        source_epoch2_fid=73.73770937849014, comparison_epochs=[1, 2],
        requested_change='4096 fixed atoms and unit-gain coefficient book; preserve native soft AR recipe',
        generation_quality_unmeasured=True, reconstruction_encoding_only='beam4',
        ar_teacher='native stochastic soft targets; no hard beam targets')
    OmegaConf.save(recipe, BASE / 'inputs/recipe.yaml')
    execution = dict(align_classifier=False, valid_classes=16385, padded_projection_width=16385,
        extra_softmax_classes=0, plain_upstream_classifier=True, native_compute_loss=True,
        world_size=4, global_batch=2048, hardware='4 NVIDIA A100-SXM4-80GB',
        checkpoint_storage='Local NVMe and committed W&B cloud artifacts',
        training_autocast_dtype='float16', gradient_clip_max_norm=1.,
        training_targets='Native stochastic soft targets; source per-depth temperatures',
        ar_teacher_encoding_beam_width=1, reconstruction_beam_width=4,
        fresh_stage2=True, fresh_optimizer=True, pilot_stop_epoch=2,
        source_run_id=str(recipe.lineage.source_run), automatic_restart=True)
    (BASE / 'inputs/execution.json').write_text(json.dumps(execution, indent=2) + '\n')
    shutil.copyfile(SOURCE / 'train.py', BASE / 'train.py')
    path = BASE / 'source/src/training/laser_joint_recipe.py'
    code = path.read_text()
    code = code.replace('from src.models.laser_joint_tokenizer import FrozenJointLaserTokenizer',
                        'from runtime.paired_tokenizer import PairedReconstructionTokenizer')
    code = code.replace('return FrozenJointLaserTokenizer(', 'return PairedReconstructionTokenizer(')
    code = code.replace("self.base/'inputs/joint-codebook.pt').to(device), None",
                        "self.base/'inputs/joint-codebook.pt', reconstruction_beam_width=4).to(device), None")
    compile(code, str(path), 'exec'); path.write_text(code)
    path = BASE / 'source/src/training/joint_runtime_support.py'
    code = path.read_text().replace("'fresh':False,'source_epoch':2", "'fresh':True,'source_epoch':0")
    code = code.replace('Joint32769 native continuation epoch2 to4 4A100 20261105',
                        'Native4096 soft AR pilot epochs1 and2 4A100 20261105')
    path.write_text(code)
    # Exercise cached autoregressive sampling, not just random-code decoding.
    path = BASE / 'source/src/training/rq_imagenet_shared.py'
    code = path.read_text()
    old = "                    assert PROFILE.digest(aux)==before\n                    checkpoint(epoch,epoch,absolute+1)"
    new = """                    assert PROFILE.digest(aux)==before
                    from runtime.generation_interface import verify_generation
                    write_json(folder/f'generation-interface-rank{rank}.json',
                               verify_generation(PROFILE,model,aux,device))
                    checkpoint(epoch,epoch,absolute+1)"""
    assert code.count(old) == 1
    code = code.replace(old, new)
    # The generic media feed is disabled; only the selected class panels are saved.
    start = code.index('                if first==0 and run is not None:')
    stop = code.index('        assert int(metric.fake_count)', start)
    code = code[:start] + code[stop:]
    # Avoid making the same step-10 checkpoint twice.
    code = code.replace('global_step==initial[\'global_step\']+10 and global_step%250:',
                        "global_step==initial['global_step']+10 and global_step not in (10,) and global_step%250:")
    compile(code, str(path), 'exec'); path.write_text(code)
    shutil.copyfile(Path(__file__).with_name('generation_interface.py'), BASE / 'runtime/generation_interface.py')
    for name in ['status.json', 'launch.json', 'training.log', 'checkpoint.json', 'supervisor.log',
                 'execution-benchmarks.json', 'selected-execution.json']:
        path = PUBLIC / name
        if not path.is_symlink(): path.symlink_to(OUT / name)
    (PUBLIC / 'live-nvme').symlink_to(OUT, target_is_directory=True)
    for name in ['recipe.yaml', 'tokenizer-policy.json', 'paired-teacher-verification.json']:
        shutil.copyfile(BASE / 'inputs' / name, PUBLIC / name)
    print(json.dumps(dict(run_id=RUN, output=str(OUT), fresh=True, epochs=2,
                         atoms=4096, vocabulary=16385, source_epoch1_fid=89.12546967078106)), flush=True)


if __name__ == '__main__': main()
