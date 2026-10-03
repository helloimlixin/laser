"""Prepare an explicit weights-only epoch-six restart with fresh ImageNet views."""
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile

import torch
import yaml

REPO = Path('/workspace/Projects/laser')
PARENT = Path('/tmp/laser-imagenet-pairfix-stage2')
PARENT_EVIDENCE = REPO / 'outputs/imagenet-rfid421-pairfix-scratch-5h200-20261001'
BASE = Path('/tmp/laser-imagenet-epoch6-aug-stage2')
RUN_ID = 'imagenet-rfid421-epoch6-augcosine-5h200-20261002'
EVIDENCE = REPO / 'outputs' / RUN_ID
SOURCE = PARENT / 'checkpoint-upload-cache/objects/22613de2cf4ae949fc6708d9341c5c5a9b3e25ff2365d395ce2bab3a78b7d468.pt'


def record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def replace_once(source, before, after):
    assert source.count(before) == 1, before[:100]
    return source.replace(before, after, 1)


def persist_tree(source, target):
    """Shared storage does not support copystat; preserve contents explicitly."""
    target.mkdir(parents=True, exist_ok=True)
    for path in source.rglob('*'):
        relative = path.relative_to(source)
        if path.is_file():
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    runtime = BASE / 'runtime'
    assert not runtime.exists(), 'Refuse to replace an already prepared runtime'
    shutil.copytree(PARENT / 'runtime', runtime, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    trainer = runtime / 'src/training/rqtransformer.py'
    original = trainer.read_text()
    source = replace_once(original,
        '                    else:\n                        tokens, soft_targets = aux.encode_sparse(\n',
        '                    elif args.physical_pair_context:\n'
        '                        atoms, coeffs = aux.encode_sparse_components(images)\n'
        '                        tokens, compact_targets = aux.sparse_targets(\n'
        '                            atoms, coeffs, temp=args.coeff_target_temperature,\n'
        '                            stochastic=args.coeff_target_mode == "soft",\n'
        '                            compact=True, hard=args.coeff_target_mode == "hard",\n'
        '                        )\n'
        '                    else:\n                        tokens, soft_targets = aux.encode_sparse(\n')
    source = replace_once(source,
        '                    or args.orthogonal_compound_tokens\n                ):\n',
        '                    or args.orthogonal_compound_tokens\n                    or args.physical_pair_context\n                ):\n')
    # The dictionary is frozen: calculate its Gram matrix once per rank rather
    # than repeating the identical 16384-by-16384 product for every microbatch.
    source = replace_once(source, '            gram = dictionary.t() @ dictionary\n',
        '            gram = getattr(self, "_frozen_dictionary_gram", None)\n'
        '            if gram is None:\n'
        '                gram = dictionary.t() @ dictionary\n'
        '                self._frozen_dictionary_gram = gram\n')
    source = replace_once(source,
        '                        target_atoms, target_coeff_probs = compact_targets\n'
        '                        atom_log_probs = F.log_softmax(atom_logits.float(), dim=-1)\n'
        '                        coeff_log_probs = F.log_softmax(coeff_logits.float(), dim=-1)\n'
        '                        atom_loss = -atom_log_probs.gather(\n'
        '                            -1, target_atoms.long().unsqueeze(-1)\n'
        '                        ).squeeze(-1)\n'
        '                        coeff_loss = -(target_coeff_probs * coeff_log_probs).sum(dim=-1)\n'
        '                        depth = target_atoms.shape[-1]\n'
        '                        loss = (\n'
        '                            atom_loss.sum(dim=-1) + coeff_loss.sum(dim=-1)\n'
        '                        ).mean() / (2 * depth * accumulation)\n',
        '                        loss = compiled_sparse_objective(\n'
        '                            atom_logits, coeff_logits, *compact_targets, accumulation\n'
        '                        )\n')
    trainer.write_text(source)
    (BASE / 'online-augmentation.patch').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), source.splitlines(True), fromfile='parent/rqtransformer.py', tofile='augmented/rqtransformer.py')))
    # Preserve the tested per-image, per-epoch RNG stream independently of
    # DataLoader prefetch and process recovery. No fixed crop is stored.
    shutil.copyfile(REPO / 'src/training/fresh_images.py', runtime / 'src/training/fresh_images.py')
    entry = (PARENT / 'entry-production.py').read_text()
    entry = replace_once(entry, "BASE = Path(os.environ['LASER_RUN_BASE'])\n",
        "BASE = Path(os.environ['LASER_RUN_BASE'])\n"
        "from src.training.fresh_images import EpochImageFolder\n"
        "original_source_images = training.source_image_dataset\n"
        "def source_images(dataset, root, transform, *, split='train'):\n"
        "    if dataset == 'imagenet' and split == 'train':\n"
        "        ready = json.loads((root / 'training-ready.json').read_text())\n"
        "        assert ready['training_images'] == 1281167 and ready['classes'] == 1000\n"
        "        dataset_view = EpochImageFolder(root / split, transform=transform, augmentation_seed=261001)\n"
        "        assert len(dataset_view) == 1281167 and len(dataset_view.classes) == 1000\n"
        "        return dataset_view\n"
        "    return original_source_images(dataset, root, transform, split=split)\n"
        "training.source_image_dataset = source_images\n"
        "original_sampler_init = training.ExactGlobalBatchSampler.__init__\n"
        "def sampler_init(self, dataset, *args, **kwargs):\n"
        "    self.dataset = dataset\n"
        "    return original_sampler_init(self, dataset, *args, **kwargs)\n"
        "training.ExactGlobalBatchSampler.__init__ = sampler_init\n"
        "original_sampler_epoch = training.ExactGlobalBatchSampler.set_epoch\n"
        "def set_sampler_epoch(self, epoch):\n"
        "    if hasattr(self.dataset, 'set_epoch'):\n"
        "        self.dataset.set_epoch(epoch)\n"
        "    return original_sampler_epoch(self, epoch)\n"
        "training.ExactGlobalBatchSampler.set_epoch = set_sampler_epoch\n"
        "original_aux_init = training.LaserAux.__init__\n"
        "def aux_init(self, *args, **kwargs):\n"
        "    kwargs['clamp_coeffs'] = False\n"
        "    return original_aux_init(self, *args, **kwargs)\n"
        "training.LaserAux.__init__ = aux_init\n"
        "original_dataloader = training.DataLoader\n"
        "def data_loader(*args, **kwargs):\n"
        "    if isinstance(args[0], EpochImageFolder):\n"
        "        kwargs.update(num_workers=12, prefetch_factor=3)\n"
        "    return original_dataloader(*args, **kwargs)\n"
        "training.DataLoader = data_loader\n")
    # A new W&B run is needed because its training clock rolls back. A warm
    # start is not a W&B resume until that new run actually exists.
    entry = replace_once(entry, "    kwargs['allow_val_change'] = True\n",
        "    c.update(stage2_parent_weights_loaded=True, first_checkpoint_step=3800,\n"
        "             stage2_initialization='epoch6-model-only-warm-start',\n"
        "             source_checkpoint_epoch=6, source_checkpoint_step=3756, source_checkpoint_fid=52.734622955322266,\n"
        "             optimizer_initialization='fresh Adam: epoch6 optimizer backup unavailable',\n"
        "             rng_initialization='fresh rank seeds: epoch6 RNG backup unavailable',\n"
        "             learning_rate_schedule='original absolute 100-epoch cosine, base LR increased 10 percent',\n"
        "             training_augmentation='Resize256 RandomCrop256 RandomHorizontalFlip(p=0.5), fresh each epoch',\n"
        "             online_stage1_encoding=True, coefficient_clipping=False,\n"
        "             augmentation_rng='deterministic per-image per-epoch; independent of worker prefetch',\n"
        "             source_noise_audit='parent cached-view audit; fresh-view audit saved separately')\n"
        "    if PHASE == 'production' and not (BASE / 'wandb-run-created.json').exists():\n"
        "        kwargs['resume'] = 'allow'\n"
        "    kwargs['allow_val_change'] = True\n")
    entry = replace_once(entry, '    WB = original_wandb_init(*args, **kwargs)\n',
        '    WB = original_wandb_init(*args, **kwargs)\n'
        "    if PHASE == 'production':\n"
        "        json_record(BASE / 'wandb-run-created.json', dict(id=WB.id, url=WB.url))\n"
        "        if not WB.summary.get('rollback/baseline_logged', False):\n"
        "            baseline = json.loads((Path(os.environ['LASER_PERSISTENT_BASE']) / 'epoch6-baseline-evaluation.json').read_text())\n"
        "            WB.log({'train/global_step':3756, 'val/fid':baseline['fid'],\n"
        "                    'val/inception_score':baseline['inception_score'],\n"
        "                    'val/inception_score_std':baseline['inception_score_std']})\n"
        "            WB.summary['rollback/baseline_logged'] = True\n")
    start = entry.index("    for name in ['recipe.yaml'")
    end = entry.index('        artifact.add_file', start)
    entry = entry[:start] + (
        "    for name in ['recipe.yaml', 'launch-provenance.json', 'preflight-summary.json',\n"
        "                 'fresh-augmentation-audit.json', 'noise-audit.json', 'validation-reference.json',\n"
        "                 'runtime-manifest.json', 'online-augmentation.patch', 'frozen-runtime.tar.gz',\n"
        "                 'epoch6-baseline-evaluation.json', 'objective-parity.json']:\n") + entry[end:]
    entry = replace_once(entry, '                      fresh=not parsed_args.resume, parameter_count=',
        '                      fresh=(len(optimizer.state) == 0), parameter_count=')
    (BASE / 'entry-production.py').write_text(entry)
    launcher = (PARENT / 'launch.py').read_text().replace(str(PARENT), str(BASE)).replace(str(PARENT_EVIDENCE), str(EVIDENCE)).replace(PARENT_EVIDENCE.name, RUN_ID)
    launcher = replace_once(launcher, "    env = os.environ.copy()\n",
        "    assert json.loads((EVIDENCE / 'preflight-summary.json').read_text())['passed']\n"
        "    assert json.loads((EVIDENCE / 'fresh-augmentation-audit.json').read_text())['acceptable']\n"
        "    ready = json.loads(Path('/tmp/laser-imagenet-stage2/imagenet/training-ready.json').read_text())\n"
        "    assert ready['training_images'] == 1281167 and ready['md5'] == '1d675b47d978889d74fa0da5fadfb00e'\n"
        "    env = os.environ.copy()\n")
    (BASE / 'launch.py').write_text(launcher)
    for name in ('inductor-cache', 'torch-cache'):
        (BASE / name).symlink_to(PARENT / name, target_is_directory=True)
    recipe = yaml.safe_load((PARENT / 'recipe.yaml').read_text())
    options = recipe['options']
    options.update(token_cache=None, output=str(BASE / 'production/train'),
        checkpoint_dir=str(EVIDENCE / 'train/checkpoints'), lr=0.00066,
        continuation_lr_policy=None, wandb_id=RUN_ID,
        wandb_name='ImageNet rFID4.21 K4 | epoch6 rollback | fresh crop+flip | cosine+10% | 5 H200',
        resume=True, save_step_freq=200)
    for path in (BASE / 'recipe.yaml', EVIDENCE / 'recipe.yaml', REPO / 'configs/stage2/imagenet-rfid421-epoch6-augcosine-5h200.yaml'):
        path.write_text(yaml.safe_dump(recipe, sort_keys=False))

    sys.path[:0] = [str(runtime), str(runtime / 'runtime')]
    from src.training.rqtransformer import build_model
    with torch.device('meta'):
        model = build_model(18432, 16384, coeff_vocab_size=2048, sparsity_level=4,
                            physical_pair_context=True, model_preset='imagenet-1400m')
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.00066, weight_decay=1e-4,
                                      betas=(0.9, 0.95), fused=True)
    optimizer_state = optimizer.state_dict()
    assert len(optimizer_state['state']) == 0 and len(optimizer_state['param_groups'][0]['params']) == 782
    payload = torch.load(SOURCE, map_location='cpu', mmap=True, weights_only=False)
    assert payload['epoch'] == 6 and payload['global_step'] == 3756
    assert abs(payload['fid'] - 52.734622955322266) < 1e-9
    assert payload['checkpoint_kind'] == 'model_only_best' and 'optimizer' not in payload
    model.load_state_dict(payload['state_dict'], strict=True, assign=True)
    with SOURCE.open('rb') as stream:
        source_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    provenance = dict(source_checkpoint=str(SOURCE), source_checkpoint_sha256=source_sha,
        source_checkpoint_kind='model_only_best', source_epoch=6, source_step=3756,
        source_fid=payload['fid'], source_original_base_lr=payload['config']['lr'],
        resumed_base_lr=0.00066, original_cosine_schedule_epochs=100,
        lr_curve_multiplier=1.1, min_lr=3e-7, scheduler_epoch=6, scheduler_step=3756,
        optimizer_state_available=False, optimizer_reset=True, rng_state_available=False, rng_reset=True,
        parent_latest_checkpoint=str(PARENT_EVIDENCE / 'epoch6-rollback-20261002/pre-rollback-step025394.pt'),
        new_run_id=RUN_ID, world_size=5, global_batch=2048, physical_microbatch_max=205,
        accumulation=2, training_images=1281167, frozen_stage1=True,
        training_augmentation=['Resize(256)', 'RandomCrop(256)', 'RandomHorizontalFlip(0.5)'],
        online_encoding=True, stage1_encoder_precision='BF16 autocast, OMP FP32',
        dictionary_gram_reused=True, token_cache_used_for_training=False,
        training_workers_per_rank=12, prefetch_factor=3, coefficients_clipped=False)
    warm = dict(payload)
    warm.update(optimizer=optimizer_state, scheduler=None, epoch=6, batch_idx=0, global_step=3756,
                best_fid=[], best_inception=[], checkpoint_kind='model_only_warm_start_with_fresh_optimizer',
                resume_capable=True, warm_start_provenance=provenance)
    warm['config'] = {**payload['config'], **options, 'world_size':5, 'accumulation':2,
                      'warm_start_provenance':provenance}
    warm_path = BASE / 'epoch6-fresh-adam.pt'
    torch.save(warm, warm_path)
    destination = EVIDENCE / 'train/checkpoints/last.pt'
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(warm_path, destination)
    for path in (BASE / 'launch-provenance.json', EVIDENCE / 'launch-provenance.json'):
        record(path, provenance)
    for name in ('noise-audit.json', 'validation-reference.json', 'objective-parity.json'):
        shutil.copyfile(PARENT_EVIDENCE / name, EVIDENCE / name)
    shutil.copyfile(BASE / 'online-augmentation.patch', EVIDENCE / 'online-augmentation.patch')
    persist_tree(PARENT / 'gap-audit-20261002', PARENT_EVIDENCE / 'gap-audit-20261002')
    manifest = {str(p.relative_to(runtime)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in runtime.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    record(EVIDENCE / 'runtime-manifest.json', dict(files=manifest, count=len(manifest)))
    with tarfile.open(BASE / 'frozen-runtime.tar.gz', 'w:gz') as archive:
        archive.add(runtime, arcname='runtime', filter=lambda info: None if '__pycache__' in info.name else info)
        archive.add(BASE / 'entry-production.py', arcname='entry-production.py')
        archive.add(BASE / 'launch.py', arcname='launch.py')
    shutil.copyfile(BASE / 'frozen-runtime.tar.gz', EVIDENCE / 'frozen-runtime.tar.gz')
    print(json.dumps(dict(prepared=True, source_sha256=source_sha, base=str(BASE), evidence=str(EVIDENCE),
                          warm_start_bytes=warm_path.stat().st_size, optimizer_states=0, epoch=6, global_step=3756)), flush=True)


if __name__ == '__main__':
    main()
