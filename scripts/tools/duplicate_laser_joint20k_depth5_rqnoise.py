#!/usr/bin/env python3
"""Restore the original run's frozen runtime and duplicate it on eight H100s."""
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
import urllib.request
import zipfile

ROOT = Path('/workspace/Projects/laser')
SOURCE_RUN = 'imagenet-laser-joint20k-depth5-rqnoise-k2048-scratch100-4h100-20261017'
RUN = SOURCE_RUN.replace('4h100', '8h100')
ORIGINAL = ROOT/'outputs'/SOURCE_RUN
OUT = ROOT/'outputs'/RUN
BASE = Path('/tmp/laser-rq-joint20k-depth5-rqnoise-k2048-8h100-imagenet-20261017')
KEY = Path('/root/.config/laser/imagenet-stage2-wandb.key')
WORLD = 8


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str)+'\n')
    temporary.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def environment():
    return dict(os.environ, **json.loads((BASE/'environment.json').read_text()),
                WANDB_API_KEY=KEY.read_text().strip())


def prepare():
    import torch
    import yaml
    assert torch.cuda.device_count() == WORLD
    assert all(torch.cuda.get_device_properties(i).total_memory > 79*2**30 for i in range(WORLD))
    assert not BASE.exists(), 'Runtime already exists; use the prepared run'
    assert not OUT.exists(), 'Refusing to overwrite an existing run'
    original_config = json.loads(Path('/tmp/laser-original-config.json').read_text())
    archive = ORIGINAL/'reproducibility/source.zip'
    manifest = json.loads((ORIGINAL/'reproducibility/code-manifest.json').read_text())['files']
    BASE.mkdir(parents=True)
    OUT.mkdir(parents=True)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(BASE)
    for name, expected in manifest.items():
        assert sha(BASE/name) == expected, name
    recipe = yaml.safe_load((BASE/'inputs/recipe.yaml').read_text())
    assert recipe == original_config['recipe']
    recipe.update(run_id=RUN, name='LASER joint20k depth5 | RQ target noise + K2048 | fresh100 | 8 H100')
    (BASE/'inputs/recipe.yaml').write_text(yaml.safe_dump(recipe, sort_keys=False))
    (ROOT/'configs/stage2'/f'{RUN}.yaml').write_text(yaml.safe_dump(recipe, sort_keys=False))
    shutil.copytree(ROOT/'third_party/rq-vae-transformer', BASE/'upstream',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
    for name, expected in original_config['shared_upstream']['core_files'].items():
        assert sha(BASE/'upstream'/name) == expected, name
    assets = ROOT/'outputs/imagenet-rfid421-combination-soft-20260927/assets'
    specifications = [
        (assets/'laser-tokenizer.pt', 'stage1-model.pt', recipe['tokenizer']['sha256']),
        (assets/'imagenet_256_train.npz', 'imagenet_256_train.npz', recipe['evaluation']['reference_sha256']),
    ]
    def copy_verified(specification):
        source, name, expected = specification
        destination = BASE/'inputs'/name
        shutil.copyfile(source, destination)
        assert sha(destination) == expected, name
        return dict(source=str(source), name=name, bytes=destination.stat().st_size, sha256=expected)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        verified = list(pool.map(copy_verified, specifications))
    assert sha(BASE/'inputs/joint-codebook.pt') == recipe['tokenizer']['codebook_sha256']
    verify = (BASE/'verify.py').read_text()
    assert verify.count("len(saved['rng_state_by_rank'])==4") == 1
    assert verify.count('for rank in range(4):') == 1
    verify = verify.replace("len(saved['rng_state_by_rank'])==4", "len(saved['rng_state_by_rank'])==8")
    verify = verify.replace('for rank in range(4):', 'for rank in range(8):')
    (BASE/'verify.py').write_text(verify)
    old_environment = json.loads((BASE/'environment.json').read_text())
    old_base = old_environment['LASER_RQ_BASE']
    env = {k:v.replace(old_base, str(BASE)) for k,v in old_environment.items()}
    env.update(WANDB_DISABLE_GIT='true', CUDA_VISIBLE_DEVICES=','.join(map(str, range(WORLD))),
               WANDB_CACHE_DIR=str(BASE/'wandb-cache'), WANDB_DATA_DIR=str(BASE/'wandb-data'),
               WANDB_ARTIFACT_DIR=str(BASE/'wandb-artifacts'), MPLCONFIGDIR=str(BASE/'matplotlib'))
    write(BASE/'environment.json', env)
    for name in ('wandb', 'torch-cache/hub/checkpoints', 'data'):
        (BASE/name).mkdir(parents=True, exist_ok=True)
    repro = OUT/'reproducibility'
    repro.mkdir()
    for name in ('noise-calibration-report.json', 'sampler-selection.json'):
        shutil.copyfile(ORIGINAL/'reproducibility'/name, repro/name)
    shutil.copyfile(BASE/'inputs/recipe.yaml', repro/'recipe.yaml')
    shutil.copyfile(BASE/'inputs/stage2-training-config.yaml', repro/'stage2-training-config.yaml')
    shutil.copyfile('/tmp/laser-original-config.json', repro/'source-wandb-config.json')
    shutil.copyfile(Path(__file__), repro/Path(__file__).name)
    official_url = ('https://raw.githubusercontent.com/kakaobrain/rq-vae-transformer/'
                    'main/configs/imagenet256/stage2/in256-rqtransformer-8x8x4-1400M.yaml')
    official_text = urllib.request.urlopen(official_url, timeout=45).read().decode()
    (repro/'upstream-imagenet1400m.yaml').write_text(official_text)
    official = yaml.safe_load(official_text)
    training = yaml.safe_load((BASE/'inputs/stage2-training-config.yaml').read_text())
    assert training['optimizer'] == official['optimizer']
    assert training['arch'] == official['arch']
    assert training['experiment']['total_batch_size'] == official['experiment']['total_batch_size'] == 2048
    assert training['experiment']['epochs'] == official['experiment']['epochs'] == 100
    audit = dict(passed=True, source_run=f'helloimlixin-rutgers/laser/{SOURCE_RUN}',
                 upstream_recipe=official_url, upstream_recipe_sha256=sha(repro/'upstream-imagenet1400m.yaml'),
                 optimizer=training['optimizer'], architecture_matches_upstream=True,
                 source_snapshot_files_verified=len(manifest),
                 training_runtime_python_unchanged=True, upstream_core_unchanged=True,
                 tokenizer_and_codebook_unchanged=True, assets=verified,
                 changes=['New run identity', 'Eight ranks instead of four',
                          'Four accumulation steps instead of eight', 'Local runtime paths',
                          'Verification expects eight RNG states and previews'],
                 exact_trajectory_guaranteed=False)
    write(repro/'recipe-audit.json', audit)
    plan = dict(prepared=True, duplicate_of=audit['source_run'], run_id=RUN, base=str(BASE),
                world_size=WORLD, microbatch=64, accumulation_steps=4, global_batch=2048,
                epochs=100, lr_schedule_epochs=100, optimizer_updates_per_epoch=626,
                cosine_total_updates=62600, fresh_stage2=True, fresh_optimizer=True,
                stage2_checkpoint_loaded=False, initial_global_step=0, recipe=recipe)
    write(OUT/'prepared.json', plan)
    files = [p for p in BASE.rglob('*') if p.is_file() and
             p.suffix in ('.py', '.yaml', '.json') and '__pycache__' not in p.parts]
    write(repro/'code-manifest.json', {str(p.relative_to(BASE)):sha(p) for p in files})
    with zipfile.ZipFile(repro/'source.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, path.relative_to(BASE))
        bundle.write(BASE/'inputs/joint-codebook.pt', 'inputs/joint-codebook.pt')
    print(json.dumps(plan), flush=True)


def prepare_data():
    sys.path.insert(0, str(ROOT/'scripts/tools'))
    import prepare_original_rq_imagenet_data as data
    data.BASE, data.DATA, data.OUT = BASE, BASE/'data', OUT
    data.DATA.mkdir(parents=True, exist_ok=True)
    # The persistent validation archive is an incomplete historical download.
    # Fetch the official bytes and retain the same checksum required upstream.
    validation = data.DATA/'ILSVRC2012_img_val.tar'
    receipt = validation.with_suffix('.verified.json')
    url = 'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar'
    if not receipt.exists():
        subprocess.run(['aria2c', '--continue=true', '--allow-overwrite=false',
            '--auto-file-renaming=false', '--file-allocation=none',
            '--max-connection-per-server=16', '--split=16', '--min-split-size=16M',
            '--console-log-level=warn', '--summary-interval=15', '--check-integrity=true',
            '--checksum=md5=29b22e2961454d5413ddabcf34fc5622',
            f'--dir={data.DATA}', f'--out={validation.name}', url], check=True)
        with validation.open('rb') as stream:
            assert hashlib.file_digest(stream, 'md5').hexdigest() == '29b22e2961454d5413ddabcf34fc5622'
        write(receipt, dict(source=url, target=str(validation), bytes=validation.stat().st_size,
                           md5='29b22e2961454d5413ddabcf34fc5622', verified=True))
    data.main()


def configure_focused_panels():
    """Restore the sample gallery in this run's dedicated workspace."""
    import importlib.util
    from urllib.parse import parse_qs, urlparse
    os.environ.update(environment())
    spec = importlib.util.spec_from_file_location('duplicate_sample_panels', BASE/'configure_samples.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    publisher = module.WorkspacePublisher(RUN)
    while not (OUT/'workspace.json').exists():
        time.sleep(10)
    while True:
        try:
            if publisher.verify_images():
                break
        except Exception as error:
            print(json.dumps(dict(waiting_for_samples=True, error=str(error))), flush=True)
        time.sleep(15)
    workspace = json.loads((OUT/'workspace.json').read_text())
    suffix = parse_qs(urlparse(workspace['url']).query)['nw'][0]
    focused = next(v for v in publisher.views('project-view') if v['name'] == f'nw-{suffix}-v')
    publisher.save(focused, module.add_sample_sections(json.loads(focused['spec'])))
    proof = dict(passed=True, url=workspace['url'], sample_panels=9,
                 project_default_workspace_changed=False, time=time.time())
    write(OUT/'samples/workspace/verified.json', proof)
    print(json.dumps(proof), flush=True)


def supervise():
    lock = (BASE/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (OUT/'train').exists(), 'Use the recorded checkpoint to resume existing training'
    child = None
    def stop(*_):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)
        raise KeyboardInterrupt('Duplicate run interrupted')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    env = environment()
    def launch(command, phase, extra=None):
        nonlocal child
        with (OUT/f'{phase}.log').open('ab', buffering=0) as log:
            child = subprocess.Popen(command, cwd=BASE, env=dict(env, **(extra or {})),
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write(OUT/'launch-status.json', dict(phase=phase, pid=os.getpid(), child_pid=child.pid,
                  command=command, run_id=RUN, world_size=WORLD, global_batch=2048,
                  microbatch=64, accumulation_steps=4, time=time.time()))
            result = child.wait()
        if result:
            raise RuntimeError(f'{phase} returned {result}; see {OUT/phase}.log')
    launch([sys.executable, str(Path(__file__).resolve()), 'data'], 'data_preparation')
    # Download and exercise the exact FID backend before committing to training.
    launch([sys.executable, '-c',
        "import os,sys; from pathlib import Path; b=Path(os.environ['LASER_RQ_BASE']); "
        "sys.path[:0]=[str(b/'source'),str(b/'upstream')]; "
        "from src import rqvae_metrics as m; m.INCEPTION_PATH=b/'upstream/rqvae/metrics/inception.py'; "
        "import torch; net=m.OriginalRQVAEInception().eval().cuda(); "
        "x=net(torch.zeros(2,3,256,256,device='cuda'),return_logits=True); "
        "assert x[0].shape==(2,2048); assert torch.isfinite(x[0]).all(); print('FID backend passed')"], 'fid_preflight')
    distributed = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8']
    launch([*distributed, str(BASE/'train.py'), '--microbatch', '64', '--smoke-updates', '2'],
           'gpu_preflight', dict(LASER_PRIVATE_SMOKE_DIR=str(BASE/'private-smoke')))
    launch([sys.executable, str(BASE/'verify.py')], 'checkpoint_verification',
           dict(LASER_PRIVATE_SMOKE_DIR=str(BASE/'private-smoke')))
    proof = json.loads((OUT/'smoke-reload-verification.json').read_text())
    assert proof['passed'] and proof['cached_events_verified'] == 320
    assert proof['targets']['temperature'] == json.loads((OUT/'prepared.json').read_text())['recipe']['targets']['temperature']
    # Keep run lineage available in W&B without changing the training driver.
    import wandb
    os.environ['WANDB_API_KEY'] = KEY.read_text().strip()
    run = wandb.init(entity='helloimlixin-rutgers', project='laser', id=RUN,
        name=json.loads((OUT/'prepared.json').read_text())['recipe']['name'], resume='never',
        dir=str(BASE/'wandb'), config=dict(duplicate_of=f'helloimlixin-rutgers/laser/{SOURCE_RUN}'),
        settings=wandb.Settings(disable_git=True))
    run.finish()
    launch([*distributed, str(BASE/'train.py'), '--microbatch', '64', '--epochs', '100'], 'training')
    write(OUT/'launch-status.json', dict(phase='completed', run_id=RUN, time=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'data', 'run', 'panels'])
    args = parser.parse_args()
    try:
        {'prepare':prepare, 'data':prepare_data, 'run':supervise,
         'panels':configure_focused_panels}[args.phase]()
    except BaseException as error:
        if args.phase not in ('prepare', 'panels'):
            write(OUT/'launch-status.json', dict(phase='failed', error_type=type(error).__name__,
                  error=str(error), time=time.time()))
        raise
