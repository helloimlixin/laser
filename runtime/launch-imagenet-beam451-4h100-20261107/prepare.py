"""Recover the published beam451 source, changing only execution and run identity."""
from pathlib import Path
import difflib
import hashlib
import json
import shutil
from omegaconf import OmegaConf

ROOT = Path('/workspace/Projects/laser')
BASE = Path('/tmp/laser-beam451-stage2-4h100-20261107')
RUN = 'imagenet-beam451-aligned-joint16k-scratch100-4h100-20261107'
OUT = ROOT / 'outputs' / RUN
SOURCE = Path('/tmp/beam451-source/unpacked')


def main():
    assert not (BASE / 'prepared.json').exists()
    OUT.mkdir(exist_ok=True)
    for name in ('source', 'upstream', 'runtime'):
        shutil.copytree(SOURCE / name, BASE / name, dirs_exist_ok=True)
    for path in (SOURCE / 'inputs').iterdir():
        shutil.copyfile(path, BASE / 'inputs' / path.name)
    for name in ('wandb', 'torch-cache', 'checkpoints', 'smoke'):
        (BASE / name).mkdir(exist_ok=True)
    recipe = OmegaConf.load(BASE / 'inputs/recipe.yaml')
    recipe.run_id = RUN
    recipe.name = 'ImageNet | beam451 aligned joint16k | RQ 1.4B | fresh100 | 4H100'
    recipe.epochs = recipe.lr_schedule_epochs = 100
    recipe.microbatch = 128
    recipe.evaluation.batch_size = 64
    recipe.tokenizer.measured_rfid = 4.514300868542591
    recipe.lineage = dict(source_artifact='helloimlixin-rutgers/laser/imnet-j16384-a4096-beam16-aligned-soft-scratch2-4a100-20261105-source:latest',
        stage1_validation_report='outputs/imagenet-beam16-validation50k-20261107/results.json',
        fresh_stage2=True, stage2_checkpoint_loaded=False, optimizer_loaded=False)
    OmegaConf.save(recipe, BASE / 'inputs/recipe.yaml')
    for name in ('stage2-training-config.yaml', 'stage2-archive-config.yaml'):
        config = OmegaConf.load(BASE / 'inputs' / name)
        if 'experiment' in config:
            config.experiment.batch_size = 128
            config.experiment.epochs = 100
        OmegaConf.save(config, BASE / 'inputs' / name)
    execution = dict(world_size=4, hardware='4 NVIDIA H100 80GB HBM3',
        microbatch=128, accumulation_steps=4, global_batch=2048, epochs=100,
        fresh_stage2=True, fresh_optimizer=True, training_sdpa=True,
        compiled_native_soft_loss=True, exact_fused_beam_cost=True,
        exact_hierarchical_beam_topk=False, reconstruction_beam_width=16,
        teacher_beam_width=16, teacher_sequence_temperature=2.0,
        generation_batch_size=64, checkpoint_every_updates=250,
        checkpoint_storage='local atomic serialization followed by atomic persistent copy; keep latest two and best')
    (BASE / 'inputs/execution.json').write_text(json.dumps(execution, indent=2)+'\n')
    worker = (SOURCE / 'train.py').read_text().replace(
        "os.environ['WANDB_BASE_URL']='https://api.forge.coreweave.com'",
        "os.environ.setdefault('WANDB_BASE_URL','https://api.wandb.ai')")
    (BASE / 'train.py').write_text(worker)
    driver = BASE / 'source/src/training/rq_imagenet_shared.py'
    old = driver.read_text()
    code = old.replace('        runtime_support.workspace(run,PROFILE,OUT)',
                       '        # Metrics and media use the regular W&B run dashboard.')
    code = code.replace('            name=PROFILE.name,',
        "            name=PROFILE.name,settings=wandb.Settings(console='off',disable_git=True),")
    code = code.replace("dataset=TarImageNet(BASE/'data','train',", "dataset=TarImageNet(BASE/'data','val' if args.smoke_updates else 'train',")
    # Save before the first preview, then every 250 updates as in the source.
    code = code.replace("                if not args.smoke_updates and global_step==initial['global_step']+10 and global_step not in (10,) and global_step%250:",
        "                if not args.smoke_updates and global_step==1:")
    driver.write_text(code)
    (BASE / 'runtime/checkpoint_store.py').write_text(Path(__file__).with_name('checkpoint_store.py').read_text())
    # Bound generation activation memory on the 80 GB cards.
    support = BASE / 'source/src/training/joint_runtime_support.py'
    support.write_text(support.read_text().replace('labels.split(128)', 'labels.split(32)').replace('codes.split(32)', 'codes.split(8)'))
    metric_driver = driver.read_text().replace('codes.split(32)', 'codes.split(8)')
    driver.write_text(metric_driver)
    repro = OUT / 'reproducibility'; repro.mkdir(exist_ok=True)
    for name in ('recipe.yaml', 'execution.json', 'stage2-training-config.yaml', 'stage2-archive-config.yaml', 'joint-codebook.pt', 'reconstruction-codebook.pt'):
        shutil.copyfile(BASE / 'inputs' / name, repro / name)
    for name in ('prepare.py', 'checkpoint_store.py', 'supervise.py'):
        shutil.copyfile(Path(__file__).with_name(name), repro / name)
    (repro / 'runtime.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True), metric_driver.splitlines(True), fromfile='published/rq_imagenet_shared.py', tofile='launched/rq_imagenet_shared.py')))
    shutil.copyfile('/tmp/beam451-source/source.zip', repro / 'original-source.zip')
    hashes = {}
    for name, expected in json.loads((ROOT / 'outputs/imagenet-beam16-validation50k-20261107/results.json').read_text())['hashes'].items():
        with (BASE / 'inputs' / name).open('rb') as stream: actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert actual == expected, name
        hashes[name] = actual
    (BASE / 'prepared.json').write_text(json.dumps(dict(run=RUN, base=str(BASE), output=str(OUT), hashes=hashes), indent=2)+'\n')
    print((BASE / 'prepared.json').read_text(), flush=True)


if __name__ == '__main__': main()
