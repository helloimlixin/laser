"""Create an isolated random-initialized LASER prior using the CC3M release config."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import tarfile
import time

from omegaconf import OmegaConf


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-base', type=Path, required=True)
    parser.add_argument('--source-local', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--local', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--diagnosis-dir', type=Path,
                        help='Use the matched sampler comparison and a fresh adaptive schedule')
    args = parser.parse_args()
    source, previous = args.source_base.resolve(), args.source_local.resolve()
    base, local = args.base.resolve(), args.local.resolve()
    root = Path(__file__).resolve().parents[2]
    if (base / 'recipe.yaml').exists() or (local / 'checkpoints/last.pt').exists():
        raise FileExistsError('Scratch destination already contains training state')
    archive_config = local / 'archive-configs/cc3m/stage2/config.yaml'
    release = OmegaConf.load(archive_config)
    training = OmegaConf.load(local / 'official-training.yaml')
    inspection = json.loads((local / 'archive-inspection.json').read_text())
    assert digest(archive_config) == inspection['stage2_config_sha256']
    assert release.arch.embed_dim == 1280 and release.arch.block_size == [8, 8, 4]
    assert release.dataset.context_length == 32
    config = OmegaConf.load(source / 'recipe.yaml')
    options = OmegaConf.to_container(config.options, resolve=True)
    for key in list(options):
        if ('migration' in key or key.startswith('resume_')
                or key in ('fid_lr_policy', 'evaluate_on_resume', 'finish_epoch_on_resume',
                           'checkpoint_best_fid_on_resume', 'preview_on_resume', 'checkpoint_on_resume')):
            options.pop(key)
    base.mkdir(parents=True, exist_ok=True)
    (base / 'assets').mkdir(exist_ok=True)
    (local / 'assets').mkdir(exist_ok=True)
    for name, expected in [('stage1.pt', options['stage1_sha256']),
                           ('train.pt', options['cache_sha256']['train']),
                           ('validation.pt', options['cache_sha256']['validation'])]:
        path = previous / 'assets' / name
        if digest(path) != expected:
            raise ValueError('Frozen asset checksum mismatch: ' + name)
        target = local / 'assets' / name
        if not target.exists():
            target.hardlink_to(path)
        durable = base / 'assets' / name
        if not durable.exists():
            durable.symlink_to(source / 'assets' / name)
    shutil.copyfile(source / 'assets/cc3m-validation-fid.npz', base / 'assets/cc3m-validation-fid.npz')
    shutil.copyfile(archive_config, base / 'official-stage2.yaml')
    shutil.copyfile(local / 'official-training.yaml', base / 'official-training.yaml')
    runtime = local / 'runtime'
    runtime.mkdir(exist_ok=True)
    original = json.loads((source / 'runtime-manifest.json').read_text())
    with tarfile.open(source / 'runtime.tar.gz') as archive:
        archive.extractall(runtime, filter='data')
    for name, expected in original.items():
        if digest(runtime / name) != expected:
            raise ValueError('Frozen source runtime mismatch: ' + name)
    patches = ['src/training/cc3m_text.py', 'src/training/warmup_cosine_schedule.py',
               'src/training/fid_adaptive_schedule.py',
               'src/training/checkpoint_upload_queue.py', 'scripts/tools/resume_cc3m_text.py',
               'scripts/tools/prepare_cc3m_scratch.py']
    for name in patches:
        target = runtime / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    manifest = {name: digest(runtime / name) for name in sorted(set(original) | set(patches))}
    sampling = release.sampling
    options.update(wandb_id=args.run_id,
        wandb_name='CC3M LASER scratch | archive 650M | paper LR 5e-4 cosine | 8 H100',
        checkpoint=str(local / 'assets/stage1.pt'), token_cache=str(local / 'assets/train.pt'),
        validation_cache=str(local / 'assets/validation.pt'), output=str(base / 'train'),
        local_checkpoints=str(local / 'checkpoints'), official_stage2_config=str(base / 'official-stage2.yaml'),
        fid_reference_stats=str(base / 'assets/cc3m-validation-fid.npz'),
        lr=float(training.optimizer.init_lr), min_lr=float(training.optimizer.warmup.min_lr),
        lr_schedule='cosine', warmup_epochs=float(training.optimizer.warmup.epoch),
        epochs=int(training.experiment.epochs), betas=list(training.optimizer.betas),
        weight_decay=float(training.optimizer.weight_decay),
        grad_clip_norm=float(training.optimizer.max_gn),
        seed=2610041830, resume=True, resume_checkpoint=None,
        stage2_initialization='fresh_random', runtime_sha256=manifest,
        atom_top_k=int(sampling.top_k[0]), atom_top_p=float(sampling.top_p[0]),
        atom_temperature=float(sampling.temp), coeff_top_k=int(sampling.top_k[0]),
        coeff_top_p=float(sampling.top_p[0]), coeff_temperature=float(sampling.temp),
        evaluate_at_first_step=True, checkpoint_async_upload=True, track_inception_score=False,
        wandb_mode='online', upload_checkpoints=True, full_state_best_checkpoints=True,
        archive_config_url='https://twg.kakaocdn.net/brainrepo/models/RQVAE/dcd95e8f08408e113aab6451fae895f5/cc3m.tar.gz',
        archive_config_sha256=inspection['stage2_config_sha256'],
        optimizer_recipe_url=inspection['official_training_config_url'],
        optimizer_recipe_sha256=digest(local / 'official-training.yaml'),
        optimizer_paper_url='https://arxiv.org/html/2203.01941#A3',
        archive_config_adaptations=['LASER physical atom/coefficient vocabulary and depth',
            'Existing frozen LASER Stage 1 and verified center-crop token caches',
            'H100 microbatch/accumulation with the published global batch size'])
    assert options['warmup_epochs'] == 0
    assert options['total_batch_size'] == int(training.experiment.total_batch_size)
    if args.diagnosis_dir:
        comparison = json.loads((args.diagnosis_dir / 'sampling/comparison.json').read_text())
        baseline = next(row for row in comparison if row['name'] == 'current_rq_sampler')
        candidates = [row for row in comparison if row['clip_score'] >= baseline['clip_score']]
        selected = min(candidates, key=lambda row: row['fid'])
        epochs, warmup_epochs = 40, 1
        updates = options['train_items'] // options['total_batch_size']
        options.update(selected['sampling'])
        options.update(wandb_name='CC3M LASER scratch | matched sampler | adaptive warmup cosine | 8 H100',
            epochs=epochs, warmup_epochs=warmup_epochs, lr=.0002, min_lr=.00001,
            lr_schedule='fid_adaptive_cosine', fid_lr_policy=dict(
                baseline_fid=None, patience=3, min_delta=.1, factor=.5, cooldown=2,
                warmup_steps=updates, decay_start_step=updates,
                decay_steps=(epochs-warmup_epochs)*updates),
            optimizer_recipe='diagnosed_cc3m_warmup_cosine_with_fid_plateau_reductions_v1',
            matched_sampling_diagnosis=selected,
            comparison_run='helloimlixin-rutgers/laser/imagenet-rfid421-pairfix-scratch-5h200-20261001')
        options['archive_config_adaptations'].append(
            'Diagnosed sampler and shorter adaptive LR schedule, authorized by the fresh-relaunch request')
        write_json(base / 'sampler-comparison.json', comparison)
        shutil.copyfile(args.diagnosis_dir / 'imagenet-rfid421-pairfix-scratch-5h200-20261001.json',
                        base / 'reference-run.json')
    config.options = options
    OmegaConf.save(config, base / 'recipe.yaml')
    OmegaConf.save(config, root / 'configs/stage2' / (args.run_id + '.yaml'))
    with tarfile.open(base / 'runtime.tar.gz', 'w:gz') as archive:
        for name in manifest:
            archive.add(runtime / name, arcname=name)
    write_json(base / 'runtime-manifest.json', manifest)
    shutil.copyfile(root / 'scripts/tools/resume_cc3m_text.py', base / 'resume.py')
    packages = ['torch', 'torchvision', 'wandb', 'hydra-core', 'omegaconf', 'numpy', 'scipy',
                'tokenizers', 'openai-clip', 'torchmetrics', 'torch-fidelity', 'triton', 'einops', 'easydict', 'lmdb']
    write_json(base / 'environment.json', {name: importlib.metadata.version(name) for name in packages})
    write_json(base / 'scratch-provenance.json', dict(timestamp=time.time(), **inspection,
        stage2_initialization='fresh_random', stage2_weights_loaded=False,
        optimizer_state_loaded=False, scheduler_state_loaded=False, global_step_start=0,
        frozen_stage1_sha256=options['stage1_sha256'], cache_sha256=options['cache_sha256'],
        run_id=args.run_id, training_adaptations=options['archive_config_adaptations'],
        optimizer_recipe_url=options['optimizer_recipe_url'],
        optimizer_recipe_sha256=options['optimizer_recipe_sha256'],
        lr_policy=dict(initial_lr=options['lr'], min_lr=options['min_lr'],
                       warmup_epochs=options['warmup_epochs'], epochs=options['epochs'],
                       schedule=options['lr_schedule'], fid_lr_policy=options.get('fid_lr_policy')),
        sampling_diagnosis=options.get('matched_sampling_diagnosis'), metrics=['fid', 'clip_score']))
    print(json.dumps(dict(prepared=True, run_id=args.run_id, recipe=str(base / 'recipe.yaml'),
                          fresh_stage2=True, frozen_assets_verified=True)), flush=True)


if __name__ == '__main__':
    main()
