#!/usr/bin/env python3
"""Prepare, validate, and supervise CC3M joint20k training on eight A100s."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile

ROOT = Path('/workspace/Projects/laser')
BASE = Path('/tmp/laser-cc3m-joint20k-depth5-8a100-20261017')
SOURCE = 'imagenet-laser-joint20k-depth5-rqnoise-k2048-scratch100-4h100-20261017'
RUN = 'cc3m-laser-joint20k-depth5-rqnoise-k2048-scratch100-8a100-20261017'
OUT = ROOT/'outputs'/RUN
KEY = Path('/root/.config/laser/wandb.key')


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def environment():
    return dict(os.environ, WANDB_API_KEY=KEY.read_text().strip(), WANDB_DISABLE_GIT='true',
        WANDB_CACHE_DIR=str(BASE/'wandb-cache'), WANDB_DATA_DIR=str(BASE/'wandb-data'),
        WANDB_ARTIFACT_DIR=str(BASE/'wandb-artifacts'), TORCH_HOME=str(BASE/'torch-cache'),
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false',
        PYTHONWARNINGS='ignore::FutureWarning', RQ_UPSTREAM_ROOT=str(BASE/'upstream'),
        PYTHONPATH=str(BASE/'source')+':'+str(BASE/'upstream'),
        CUDA_VISIBLE_DEVICES=','.join(map(str, range(8))))


def prepare():
    import yaml
    import urllib.request
    sys.path.insert(0, str(ROOT))
    from src import rq_upstream
    OUT.mkdir(parents=True, exist_ok=True)
    repro = OUT/'reproducibility'
    repro.mkdir(exist_ok=True)
    base_reference = BASE/'reference'
    if not base_reference.exists():
        with zipfile.ZipFile(ROOT/'outputs'/SOURCE/'reproducibility/source.zip') as archive:
            archive.extractall(base_reference)
    source_manifest = json.loads((ROOT/'outputs'/SOURCE/'reproducibility/code-manifest.json').read_text())['files']
    for name, expected in source_manifest.items():
        assert sha(base_reference/name) == expected, name
    original = json.loads((BASE/'source-wandb-config.json').read_text())
    recipe = yaml.safe_load((base_reference/'inputs/recipe.yaml').read_text())
    assert original['recipe'] == recipe
    assert sha(BASE/'stage1.pt') == recipe['tokenizer']['sha256']
    assert sha(base_reference/'inputs/joint-codebook.pt') == recipe['tokenizer']['codebook_sha256']
    for name, expected in original['shared_upstream']['core_files'].items():
        assert sha(ROOT/'third_party/rq-vae-transformer'/name) == expected, name
    if not (BASE/'source').exists():
        shutil.copytree(base_reference/'source', BASE/'source')
    if not (BASE/'upstream').exists():
        shutil.copytree(ROOT/'third_party/rq-vae-transformer', BASE/'upstream',
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
    for name in ('src/training/cc3m_joint.py', 'src/training/rq_sdpa_backend.py',
                 'src/training/rq_precision.py',
                 'src/training/rq_rng.py',
                 'src/training/cc3m_preview.py',
                 'src/training/cc3m_storage.py',
                 'src/data/__init__.py', 'src/data/seeded_bpe.py', 'src/data/cc3m_joint.py'):
        dest = BASE/'source'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, dest)
    metrics_vendor = BASE/'source/third_party'
    metrics_vendor.mkdir(exist_ok=True)
    if not (metrics_vendor/'rq-vae-transformer').exists():
        (metrics_vendor/'rq-vae-transformer').symlink_to(BASE/'upstream', target_is_directory=True)
    official_path = BASE/'official-cc3m.yaml'
    official_url = ('https://raw.githubusercontent.com/kakaobrain/rq-vae-transformer/'
                    'main/configs/cc3m/cc3m-rqtransformer-8x8x4-650M.yaml')
    if not official_path.exists():
        official_path.write_bytes(urllib.request.urlopen(official_url, timeout=45).read())
    official = yaml.safe_load(official_path.read_text())
    augmentation_url = ('https://raw.githubusercontent.com/kakaobrain/rq-vae-transformer/'
                        'main/rqvae/txtimg_datasets/transforms.py')
    augmentation = urllib.request.urlopen(augmentation_url, timeout=45).read()
    assert augmentation == (BASE/'upstream/rqvae/txtimg_datasets/transforms.py').read_bytes()
    (repro/'upstream-cc3m-transforms.py').write_bytes(augmentation)
    arch = official['arch']
    arch.update(vocab_size=recipe['tokenizer']['vocab_size'], block_size=[8, 8, 5])
    cfg = dict(run_id=RUN, name='CC3M LASER joint20k depth5 | RQ noise + K2048 | 650M | 8 A100',
        wandb_entity='helloimlixin-rutgers', wandb_dir=str(BASE/'wandb'),
        source_run='helloimlixin-rutgers/laser/'+SOURCE,
        stage1=str(BASE/'stage1.pt'), codebook=str(base_reference/'inputs/joint-codebook.pt'),
        stage1_sha256=recipe['tokenizer']['sha256'], codebook_sha256=recipe['tokenizer']['codebook_sha256'],
        tokenizer_recipe=recipe['tokenizer'], arch=arch, temperatures=recipe['targets']['temperature'],
        target_chunk=128, encoder_batch=8, sampling=recipe['sampling'],
        data_source='/workspace/Projects/data/cc3m/shards', data_local=str(BASE/'data'),
        data_manifest=str(BASE/'data-manifest.json'), dataset_revision='46f3d69f840e59d77d52e8decfe5baec97e94c7f',
        train_items=2905954, validation_items=13443, world_size=8,
        global_batch=official['experiment']['total_batch_size'], microbatch=64, accumulation=4,
        epochs=official['experiment']['epochs'], lr=official['optimizer']['init_lr'],
        betas=official['optimizer']['betas'], weight_decay=official['optimizer']['weight_decay'],
        grad_clip=official['optimizer']['max_gn'], warmup_epochs=0, min_lr=0.,
        image_loss_weight=official['loss']['img_weight'], text_loss_weight=official['loss']['txt_weight'],
        bpe_dropout=.1, text_length=32, text_vocab_size=16384,
        image_transform='upstream dalle-vqvae random augmentation; eval resize256 square',
        augmentation_source_url=augmentation_url,
        augmentation_sha256=hashlib.sha256(augmentation).hexdigest(),
        augmentation_matches_upstream_byte_for_byte=True,
        image_target_mode='fresh stochastic residual prefixes and full soft distributions on each visit',
        shuffle_group_shards=16, order_policy='shuffle shard order and images within 16-shard groups each epoch; exact distributed split',
        precision='bf16', frozen_encoder_precision='fp32', attention='SDPA; unchanged upstream parameters',
        seed=261017, workers=4, eval_batch=32, save_every=500, preview_every=500,
        output=str(OUT), local_checkpoints=str(BASE/'checkpoints'),
        optimizer_recipe_url=official_url, official_config_sha256=sha(official_path),
        stage2_initialization='fresh_random', stage2_checkpoint_loaded=False)
    config_path = BASE/'config.yaml'
    if config_path.exists():
        # A completed benchmark chooses the production microbatch exactly once.
        previous = yaml.safe_load(config_path.read_text())
        cfg['microbatch'], cfg['accumulation'] = previous['microbatch'], previous['accumulation']
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    (BASE/'wandb').mkdir(exist_ok=True)
    for src, name in [(config_path, 'recipe.yaml'), (official_path, 'upstream-cc3m650m.yaml'),
                      (BASE/'data-manifest.json', 'data-manifest.json'),
                      (base_reference/'inputs/joint-codebook.pt', 'joint-codebook.pt'),
                      (BASE/'source-wandb-config.json', 'source-wandb-config.json')]:
        shutil.copyfile(src, repro/name)
    shutil.copyfile(config_path, ROOT/'configs/stage2'/f'{RUN}.yaml')
    write(repro/'provenance.json', dict(source_run=cfg['source_run'],
        source_snapshot_files_verified=len(source_manifest),
        source_stage1_sha256=cfg['stage1_sha256'], source_codebook_sha256=cfg['codebook_sha256'],
        upstream=rq_upstream.provenance(), official_recipe=official_url,
        changes=['Text conditioning using the official CC3M 650M architecture and loss weights',
                 'Native joint vocabulary 20481 and depth 5, copied from the requested ImageNet setup',
                 'Exact source temperatures and K2048/p0.95 sampling',
                 'Eight A100 GPUs with BF16, SDPA, fused AdamW, and gradient accumulation',
                 'Online upstream image augmentations and seeded BPE dropout',
                 'Shard-group shuffle and available 2,905,954/13,443 image-caption subset'],
        fresh_transformer=True, fresh_optimizer=True))
    files = list((BASE/'source/src').rglob('*.py'))+list((BASE/'upstream').rglob('*.py'))
    write(repro/'code-manifest.json', {str(p.relative_to(BASE)): sha(p) for p in files})
    with zipfile.ZipFile(repro/'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(BASE))
        for path in (BASE/'upstream/rqvae/txtimg_datasets/tokenizers/pretrained').iterdir():
            if path.is_file():
                archive.write(path, path.relative_to(BASE))
        for path in (config_path, BASE/'data-manifest.json', base_reference/'inputs/joint-codebook.pt', official_path):
            archive.write(path, path.relative_to(BASE))
        archive.write(__file__, 'launch_cc3m_joint20k.py')
    local_requirements = BASE/'requirements-freeze.txt'
    with local_requirements.open('w') as writer:
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=writer, check=True)
    shutil.copyfile(local_requirements, repro/'requirements-freeze.txt')
    for name in ('train', 'preflight'):
        local = BASE/'live'/name
        local.mkdir(parents=True, exist_ok=True)
        target = OUT/name
        if target.exists() and not target.is_symlink():
            target.rename(OUT/(name+'-before-local-logs'))
        if not target.is_symlink():
            target.symlink_to(local, target_is_directory=True)
    print(json.dumps(dict(prepared=True, run_id=RUN, output=str(OUT))), flush=True)


def stage():
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from src.data.cc3m_joint import epoch_order, shard_index
    manifest = json.loads((BASE/'data-manifest.json').read_text())
    # Follow the entire epoch order, rather than reverting to filename order
    # after the first group. Otherwise the next random group stalls on reads.
    shard_order = np.random.default_rng(np.random.SeedSequence([261017, 0])).permutation(
        len(manifest['train']))
    specs = manifest['validation']+[manifest['train'][int(i)] for i in shard_order]
    def copy(spec):
        shard_index(spec, '/workspace/Projects/data/cc3m/shards', BASE/'data')
        return spec['name']
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i, name in enumerate(pool.map(copy, specs)):
            write(BASE/'staging.json', dict(completed=i+1, total=len(specs), last=name, time=time.time()))
            print(json.dumps(dict(completed=i+1, shard=name)), flush=True)


def supervise(resume_script=None):
    import yaml
    sys.path.insert(0, str(BASE/'source'))
    from src.training.cc3m_storage import WorkspaceMirror
    lock = (BASE/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    local_status = BASE/'supervisor'
    local_status.mkdir(exist_ok=True)
    logs = BASE/'logs'
    logs.mkdir(exist_ok=True)
    mirror = WorkspaceMirror([(local_status, OUT), (logs, OUT)]).start()
    env = environment()
    child = None
    def stop(*_):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        raise KeyboardInterrupt('CC3M supervisor interrupted; training saves at an optimizer boundary')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    def execute(command, phase):
        nonlocal child
        local_log = logs/(phase+'.log')
        with local_log.open('ab', buffering=0) as log:
            child = subprocess.Popen(command, cwd=BASE, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
            write(local_status/'status.json', dict(phase=phase, pid=os.getpid(), child_pid=child.pid,
                run_id=RUN, command=command, time=time.time()))
            result = child.wait()
            if result:
                write(local_status/'status.json', dict(phase='failed', failed_phase=phase,
                    exit_code=result, run_id=RUN, time=time.time()))
                raise RuntimeError(f'{phase} failed with code {result}; inspect {local_log}')
    command = [sys.executable, '-m', 'src.training.cc3m_joint', '--config', str(BASE/'config.yaml')]
    # Cache established launch prerequisites once, before starting production.
    benchmark = BASE/'live/benchmark.json'
    if not benchmark.exists() and (OUT/'benchmark.json').exists():
        shutil.copyfile(OUT/'benchmark.json', benchmark)
    if not benchmark.exists():
        execute([*command, '--benchmark'], 'benchmark')
    repro = BASE/'live/reproducibility'
    repro.mkdir(exist_ok=True)
    shutil.copyfile(BASE/'config.yaml', repro/'recipe.yaml')
    distributed = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8',
                   '-m', 'src.training.cc3m_joint', '--config', str(BASE/'config.yaml')]
    if not (BASE/'live/preflight/passed.json').exists():
        execute([*distributed, '--smoke'], 'gpu_preflight')
    assert json.loads((BASE/'live/preflight/passed.json').read_text())['passed']
    cfg = yaml.safe_load((BASE/'config.yaml').read_text())
    write(local_status/'launch.json', dict(run_id=RUN, world_size=8, global_batch=cfg['global_batch'],
        microbatch=cfg['microbatch'], accumulation=cfg['accumulation'], epochs=100,
        preflight_passed=True, production_starts_from_zero=not (BASE/'checkpoints/last.pt').exists(),
        resume_command=f'{sys.executable} {Path(resume_script or __file__).resolve()} run', time=time.time()))
    try:
        execute(distributed, 'training')
        phase = 'completed' if (BASE/'live/completed.json').is_file() else 'stopped'
        write(local_status/'status.json', dict(phase=phase, run_id=RUN, time=time.time()))
    finally:
        mirror.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'stage', 'run'])
    args = parser.parse_args()
    try:
        {'prepare': prepare, 'stage': stage, 'run': supervise}[args.phase]()
    except BaseException as error:
        write(BASE/'supervisor/status.json', dict(phase='failed', error=str(error), time=time.time()))
        raise
