#!/usr/bin/env python3
"""Restore the measured epoch-six LASER checkpoint and its frozen training code."""
import argparse
import copy
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
SOURCE_RUN = 'imagenet-laser-joint20k-depth5-scratch100-4h100-20261017'
SOURCE = ROOT/'outputs'/SOURCE_RUN
ASSETS = Path('/tmp/laser-rq-joint20k-depth5-rqnoise-k2048-rngfix-8h100-imagenet-20261017')
SWEEP = ROOT/'outputs/imagenet-laser-joint20k-depth5-epoch6-sampler-sweep-20261017'
RUN = 'imagenet-laser-joint20k-depth5-earlierbest-continue100-4h100-20261018'
OUT = ROOT/'outputs'/RUN
BASE = Path('/tmp/laser-joint20k-earlier-best-20261018')
KEY = Path('/root/.config/laser/imagenet-stage2-wandb.key')


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str)+'\n')
    temporary.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tensor_hash(value):
    import torch
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(str(item.dtype).encode())
            digest.update(str(tuple(item.shape)).encode())
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            for key, child in item.items():
                digest.update(str(key).encode())
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def prepare():
    import yaml
    if BASE.exists() or OUT.exists():
        raise RuntimeError('Refusing to replace a prepared continuation')
    selection = json.loads((SWEEP/'selected-sampling.json').read_text())
    sweep = json.loads((SWEEP/'report.json').read_text())
    receipt = sweep['plan']['checkpoint']
    assert selection['source_run'] == SOURCE_RUN and selection['source_step'] == 3756
    assert selection['sampling'] == dict(temperature=1., top_k=2048, top_p=.95)
    assert selection['checkpoint_sha256'] == receipt['sha256']
    assert selection['generated_images'] == 50000 and selection['reference_images'] == 1281167
    confirmation = next(row for row in sweep['confirmation'] if row['policy'] == sweep['selected_policy'])
    assert confirmation['fid'] == selection['fid50k'] < selection['baseline_fid50k']
    assert confirmation['weights_unmodified'] and confirmation['training_updates_added'] == 0
    assert confirmation['world_size'] == 4 and confirmation['source_epoch'] == 6
    BASE.mkdir(parents=True)
    OUT.mkdir(parents=True)
    with zipfile.ZipFile(SOURCE/'reproducibility/source.zip') as archive:
        for name in archive.namelist():
            destination = (BASE/name).resolve()
            if not destination.is_relative_to(BASE.resolve()):
                raise RuntimeError('Unsafe snapshot archive path')
        archive.extractall(BASE)
    original_manifest = json.loads((SOURCE/'reproducibility/code-manifest.json').read_text())['files']
    for name, expected in original_manifest.items():
        if sha(BASE/name) != expected:
            raise RuntimeError(f'Original frozen file mismatch: {name}')
    for name in ('upstream', 'data', 'torch-cache'):
        (BASE/name).symlink_to(ASSETS/name, target_is_directory=True)
    for name in ('stage1-model.pt', 'imagenet_256_train.npz'):
        os.link(ASSETS/'inputs'/name, BASE/'inputs'/name)
    original = yaml.safe_load((BASE/'inputs/recipe.yaml').read_text())
    recipe = copy.deepcopy(original)
    recipe.update(run_id=RUN, name='LASER joint20k depth5 | earlier best epoch6 | continue100',
        lineage=dict(source_run=SOURCE_RUN, source_step=3756, source_epoch=6,
            checkpoint_sha256=receipt['sha256'], reason='User requested sticking to the earlier best',
            stage2_initialization='continued', training_recipe='original frozen epoch-six recipe'))
    recipe['sampling'] = copy.deepcopy(selection['sampling'])
    before, after = copy.deepcopy(original), copy.deepcopy(recipe)
    for value in (before, after):
        for key in ('run_id', 'name', 'lineage', 'sampling'):
            value.pop(key, None)
    if before != after:
        raise RuntimeError('Continuation changes the earlier training recipe')
    (BASE/'inputs/recipe.yaml').write_text(yaml.safe_dump(recipe, sort_keys=False))
    for name in ('stage2-training-config.yaml', 'stage2-archive-config.yaml'):
        config = yaml.safe_load((BASE/'inputs'/name).read_text())
        if config.get('experiment') is not None:
            config['experiment']['sample'] = dict(temp=1., top_k=2048, top_p=.95)
        if 'sampling' in config:
            config['sampling'] = dict(temp=1., top_k=2048, top_p=.95)
        (BASE/'inputs'/name).write_text(yaml.safe_dump(config, sort_keys=False))
    env = {key:value.replace('/tmp/laser-rq-joint20k-depth5-imagenet-20261017', str(BASE))
        for key,value in json.loads((BASE/'environment.json').read_text()).items()}
    env.update(CUDA_VISIBLE_DEVICES='0,1,2,3', WANDB_DISABLE_GIT='true',
        WANDB_CACHE_DIR=str(BASE/'wandb-cache'), WANDB_DATA_DIR=str(BASE/'wandb-data'))
    for name in ('wandb', 'wandb-cache', 'wandb-data'):
        (BASE/name).mkdir(exist_ok=True)
    write(BASE/'environment.json', env)
    for name, expected in [('stage1-model.pt', recipe['tokenizer']['sha256']),
                           ('joint-codebook.pt', recipe['tokenizer']['codebook_sha256']),
                           ('imagenet_256_train.npz', recipe['evaluation']['reference_sha256'])]:
        assert sha(BASE/'inputs'/name) == expected
    repro = OUT/'reproducibility'
    repro.mkdir()
    shutil.copyfile(Path(__file__), repro/Path(__file__).name)
    shutil.copyfile(SOURCE/'reproducibility/code-manifest.json', repro/'original-code-manifest.json')
    shutil.copyfile(SWEEP/'selected-sampling.json', repro/'selected-sampling.json')
    shutil.copyfile(SWEEP/'report.json', repro/'sampler-validation.json')
    for name in ('recipe.yaml', 'stage2-training-config.yaml', 'stage2-archive-config.yaml'):
        shutil.copyfile(BASE/'inputs'/name, repro/name)
    code_files = [BASE/name for name in original_manifest]
    write(repro/'code-manifest.json', {str(p.relative_to(BASE)):sha(p) for p in code_files})
    code_unchanged = all(sha(BASE/name) == expected for name,expected in original_manifest.items()
        if name.endswith('.py'))
    assert code_unchanged
    write(repro/'frozen-runtime-verification.json', dict(passed=True,
        all_original_python_files_byte_identical=True, original_python_files=sum(
            name.endswith('.py') for name in original_manifest),
        training_recipe_unchanged=True, sampler_is_measured_earlier_best=True,
        source_checkpoint=receipt, original_four_rank_layout_preserved=True))
    with zipfile.ZipFile(repro/'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in code_files:
            archive.write(path, path.relative_to(BASE))
    plan = dict(run_id=RUN, source_run=SOURCE_RUN, source_checkpoint=receipt,
        url=f'https://wandb.ai/helloimlixin-rutgers/laser/runs/{RUN}',
        source_step=3756, source_epoch=6, initial_fid50k=selection['fid50k'],
        source_original_sampler_fid50k=selection['baseline_fid50k'], sampling=selection['sampling'],
        training_recipe_unchanged=True, all_original_python_files_byte_identical=True,
        world_size=4, global_batch=2048, microbatch=64, accumulation_steps=8,
        rationale_for_four_gpus='Restore the exact earlier four-rank optimizer and RNG continuation',
        restored_optimizer=True, restored_rng=True, total_training_epochs=100,
        lr_schedule_epochs=100, stop_step=62600, fresh_stage2=False,
        experiments_cancelled=True, time=time.time())
    write(OUT/'plan.json', plan)
    write(OUT/'status.json', dict(phase='prepared', time=time.time()))
    print(json.dumps(plan), flush=True)


def restore():
    import torch
    from omegaconf import OmegaConf
    torch.set_num_threads(8)
    plan = json.loads((OUT/'plan.json').read_text())
    receipt = plan['source_checkpoint']
    source = Path(receipt['persistent'])
    local = BASE/'source-epoch006.pt'
    if not local.exists():
        shutil.copyfile(source, local)
    assert local.stat().st_size == receipt['bytes'] and sha(local) == receipt['sha256']
    saved = torch.load(local, map_location='cpu', mmap=True, weights_only=False)
    assert saved['run_id'] == SOURCE_RUN and saved['global_step'] == 3756
    assert saved['next_epoch'] == 6 and saved['next_microbatch'] == 0
    assert len(saved['rng_state_by_rank']) == 4
    assert all(int(value['step']) == 3756 for value in saved['optimizer']['state'].values())
    protected = {name:tensor_hash(saved[name]) for name in
        ('state_dict', 'optimizer', 'scheduler', 'scaler', 'rng_state_by_rank')}
    recipe = OmegaConf.to_container(OmegaConf.load(BASE/'inputs/recipe.yaml'), resolve=True)
    # Change identity and the already measured sampler only. Training tensors,
    # Adam moments, LR age, AMP scaler and all four RNG streams remain intact.
    saved.update(run_id=RUN, recipe=recipe, best_fid=plan['initial_fid50k'])
    sweep = json.loads((SWEEP/'report.json').read_text())
    saved['evaluation'] = next(row for row in sweep['confirmation'] if row['policy'] == sweep['selected_policy'])
    saved['continuation_origin'] = dict(source_run=SOURCE_RUN, source_checkpoint=str(source),
        source_checkpoint_sha256=receipt['sha256'], source_step=3756, source_epoch=6,
        training_state_hashes=protected, reason='Return to earlier best at user request')
    saved['runtime'] = copy.deepcopy(saved['runtime'])
    saved['runtime'].update(fresh_stage2=False, fresh_optimizer=False, global_step=3756,
        native_sampling=dict(temperature=1., top_k=[2048], top_p=[.95]))
    destination = BASE/'resume-epoch006.pt'
    temporary = destination.with_suffix('.tmp')
    torch.save(saved, temporary)
    temporary.replace(destination)
    del saved
    check = torch.load(destination, map_location='cpu', mmap=True, weights_only=False)
    actual = {name:tensor_hash(check[name]) for name in protected}
    if protected != actual:
        raise RuntimeError('Repackaged checkpoint changed training state')
    proof = dict(passed=True, source_sha256=receipt['sha256'],
        resumed_checkpoint=str(destination), resumed_checkpoint_sha256=sha(destination),
        weights_optimizer_scheduler_scaler_and_rng_bit_identical=True,
        protected_training_state_hashes=protected, source_step=3756, source_epoch=6,
        rank_rng_states=4, best_fid=plan['initial_fid50k'], time=time.time())
    write(OUT/'checkpoint-restoration.json', proof)
    print(json.dumps(proof), flush=True)


def verify():
    import torch
    from omegaconf import OmegaConf
    os.environ['RQ_UPSTREAM_ROOT'] = str(BASE/'upstream')
    sys.path[:0] = [str(BASE/'source'), str(BASE/'upstream'), str(BASE/'runtime')]
    from src.training.laser_joint_recipe import JointLaserRecipe
    from src.training.rq_imagenet_runtime import TarImageNet
    from rqvae.utils.config import load_config, augment_optimizer_defaults
    from rqvae.optimizer.optimizer import create_resnet_optimizer
    from rqvae.optimizer.scheduler import create_scheduler
    from rqvae.img_datasets.transforms import create_transforms
    torch.set_num_threads(8)
    saved = torch.load(BASE/'resume-epoch006.pt', map_location='cpu', mmap=True, weights_only=False)
    profile = JointLaserRecipe(BASE)
    profile.verify_resume(saved)
    with torch.device('meta'):
        model = profile.model(OmegaConf.create(saved['config']['arch']))
    model.load_state_dict(saved['state_dict'], strict=True, assign=True)
    config = load_config(BASE/'inputs/stage2-training-config.yaml')
    optimizer = create_resnet_optimizer(model, augment_optimizer_defaults(config.optimizer))
    scheduler = create_scheduler(optimizer, config.optimizer.warmup, 626, 100)
    optimizer.load_state_dict(saved['optimizer'])
    scheduler.load_state_dict(saved['scheduler'])
    assert scheduler.after_scheduler.last_epoch == 3756 and scheduler.after_scheduler.T_max == 62600
    assert len(optimizer.state) == len(list(model.parameters()))
    assert all(int(state['step']) == 3756 for state in optimizer.state.values())
    assert profile.target_temperature() == list(profile.recipe.targets.temperature)
    assert profile.sampling_settings() == dict(temperature=1., top_k=[2048], top_p=[.95])
    initial_lr = optimizer.param_groups[0]['lr']
    del optimizer, scheduler, saved
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(16*2**30/torch.cuda.get_device_properties(0).total_memory, 0)
    model = model.requires_grad_(False).eval().to(device)
    aux, _ = profile.tokenizer(device)
    data = TarImageNet(BASE/'data', 'val', create_transforms(config.dataset, split='val'))
    images = torch.stack([data[i][0] for i in range(2)]).to(device)
    labels = torch.tensor([data[i][1] for i in range(2)], device=device)
    with torch.inference_mode():
        targets, codes = profile.targets(aux, images)
        target_proof = profile.verify_targets(aux, targets, codes)
        del targets
        expected = model(codes, aux, labels, amp=True)
        prefix = torch.zeros_like(codes)
        model.init_cache()
        maximum_kl = 0.
        for h in range(8):
            for w in range(8):
                for d in range(5):
                    actual = model.cached_forward(prefix[:,:h+1], aux, labels, amp=True, sample_loc=(h,w,d))
                    ref = expected[:,h,w,d].float().log_softmax(-1)
                    logp = actual.float().log_softmax(-1)
                    maximum_kl = max(maximum_kl, float((ref.exp()*(ref-logp)).sum(-1).max()))
                    prefix[:,h,w,d] = codes[:,h,w,d]
        assert maximum_kl < 1.e-4, maximum_kl
    proof = dict(passed=True, strict_model_reload=True, strict_optimizer_reload=True,
        all_adam_steps=3756, cosine_schedule_age=3756, cosine_total_updates=62600,
        restored_lr=initial_lr, full_vs_cached_events=320, maximum_cache_kl=maximum_kl,
        target_proof=target_proof, original_rng_states_restored_by_training_driver=True,
        preflight_updates=0, time=time.time())
    write(OUT/'resume-verification.json', proof)
    print(json.dumps(proof), flush=True)


def run():
    lock = (BASE/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert json.loads((OUT/'checkpoint-restoration.json').read_text())['passed']
    assert json.loads((OUT/'resume-verification.json').read_text())['passed']
    for name, expected in json.loads((OUT/'reproducibility/code-manifest.json').read_text()).items():
        assert sha(BASE/name) == expected, name
    if (OUT/'train').exists():
        raise RuntimeError('Refusing to overwrite an existing production continuation')
    occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], text=True).strip()
    if occupied:
        raise RuntimeError('Earlier experiment workers still hold CUDA resources')
    env = dict(os.environ, **json.loads((BASE/'environment.json').read_text()),
        WANDB_API_KEY=KEY.read_text().strip())
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=4',
        str(BASE/'train.py'), '--epochs', '100', '--resume', str(BASE/'resume-epoch006.pt'), '--new-wandb-run']
    with (OUT/'training.log').open('ab', buffering=0) as log:
        child = subprocess.Popen(command, cwd=BASE, env=env, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
        def stop(*_):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGINT)
            raise KeyboardInterrupt('Earlier-best continuation stopped')
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        write(OUT/'status.json', dict(phase='training', pid=os.getpid(), child_pid=child.pid,
            command=command, restored_step=3756, target_epochs=100, time=time.time()))
        result = child.wait()
    if result:
        raise RuntimeError(f'Training exited {result}; see training.log')
    write(OUT/'status.json', dict(phase='completed', time=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare','restore','verify','run'])
    args = parser.parse_args()
    try:
        {'prepare':prepare,'restore':restore,'verify':verify,'run':run}[args.phase]()
    except BaseException as error:
        if OUT.exists():
            write(OUT/'failure.json', dict(error_type=type(error).__name__, error=str(error), time=time.time()))
        raise
