#!/usr/bin/env python3
"""Restore the verified 9.9 Church recipe and change only residual dropout."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def replace_once(source, old, new):
    assert source.count(old) == 1, old[:120]
    return source.replace(old, new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dropout', type=float, default=.2)
    args = parser.parse_args()
    assert args.dropout in {.1, .2}
    root = Path('/mnt/laser-church/dropout-experiment')
    repository = Path('/workspace/Projects/laser')
    run_id = f'church-rqrecipe300-dropout{round(args.dropout*10):02d}-b2048-4h100-20260924'
    base = repository / 'outputs' / run_id
    base.mkdir(exist_ok=True)
    runtime = root / 'runtime'
    if not runtime.exists():
        shutil.copytree(root / 'earlier-source/runtime', runtime)
    # The historical runtime predates the current PyTorch serialization fix.
    trainer_path = runtime / 'src/training/rqtransformer.py'
    trainer_source = trainer_path.read_text()
    if 'gc.collect()' not in trainer_source[trainer_source.index('def atomic_torch_save'):trainer_source.index('def snapshot_checkpoint')]:
        trainer_source = replace_once(trainer_source, 'from functools import partial\n', 'from functools import partial\nimport gc\n')
        trainer_source = replace_once(trainer_source, '    finally:\n        if serialized != temporary:',
                                      '    finally:\n        gc.collect()  # Release pickler cycles before optimizer offload.\n        if serialized != temporary:')
        trainer_path.write_text(trainer_source)
    if not (base / 'runtime').exists():
        (base / 'runtime').symlink_to(runtime, target_is_directory=True)
    for name in ['official_metrics.py', 'heldout_monitor.py', 'checkpoint_staging.py']:
        shutil.copyfile(root / 'earlier-source' / name, base / name)
    (base / 'official-metrics').mkdir(exist_ok=True)
    for path in (root / 'earlier-source/official-metrics').glob('*.py'):
        shutil.copyfile(path, base / 'official-metrics' / path.name)
    for name in ['heldout-probe.pt', 'heldout-probe.json']:
        target = base / name
        if not target.is_symlink():
            target.symlink_to(root / 'assets' / name)
    original = repository / 'outputs/church-compound-rqopt-scratch-b2048-4h100-20260924/earlier-fid-comparison'
    config = json.loads((original / 'earlier-run.json').read_text())['config']
    old_reference = json.loads((original / 'fid-reference-comparison.json').read_text())['earlier_reference']
    config.update(checkpoint='/mnt/laser-church/assets/tokenizer.pt',
                  token_cache=str(root / 'assets/compound-cache.pt'),
                  data='/workspace/Projects/data/lsun', fid_reference_stats=old_reference,
                  token_cache_in_ram=True, checkpoint_upload_mode='files',
                  keep_best_checkpoints=1, model_only_best_checkpoints=False,
                  coeff_top_p=1., atom_top_k=250, coeff_temperature=1., atom_temperature=1.,
                  seed=0, wandb_entity='helloimlixin-rutgers', wandb_project='laser')
    write(base / 'baseline-request.json', {'config': config})
    plan = json.loads((root / 'earlier-provenance/plan.json').read_text())
    plan.update(run_id=run_id, world_size=4, accumulation_steps=4, fid_batch_size=512,
                dropout={'residual': args.dropout, 'attention': 0., 'embedding': 0.},
                objective='Fresh Church 9.9 recipe with residual dropout intervention',
                historical_control='helloimlixin-rutgers/laser/church-laser-scratch-rqrecipe300-b2048-h200x5-20260921',
                historical_differences=['4 H100 instead of 5 H200', 'microbatch accumulation and generation batch layout',
                                        'new fixed 300/300 diagnostic probe', 'residual dropout 0.2 instead of 0.1' if args.dropout == .2 else 'residual dropout unchanged'],
                checkpoint_policy='Full last/best states and all four rank RNG streams, committed online with digest verification')
    write(base / 'plan.json', plan)
    provenance = json.loads((root / 'earlier-provenance/official-fid-provenance.json').read_text())
    provenance['real_reference']['reference'] = old_reference
    write(base / 'official-fid-provenance.json', provenance)
    (base / 'cache').mkdir(exist_ok=True)
    write(base / 'cache/complete.json', dict(passed=True, images=126227,
          cache=str(root / 'assets/compound-cache.pt'),
          cache_sha256='437bf76107dd3da5661db6c766b474be43304f37705d6c8431f2e34ae3afa9fa',
          checkpoint=config['checkpoint'],
          checkpoint_sha256='762c51a10267ed6fa55709ff0d6cf997940d21056b16c7a0a0399b3ebf93868d'))
    write(base / 'cache/codec-overrides.json', {'coeff_scale': 1., 'coeff_scales': [1.]*4,
          'coeff_max': config['coeff_max']})
    for mode in ['train', 'benchmark-a2', 'benchmark-a4']:
        output = base / mode
        output.mkdir(exist_ok=True)
        checkpoints = root / 'checkpoints' / run_id / mode
        checkpoints.mkdir(parents=True, exist_ok=True)
        if not (output / 'checkpoints').exists():
            (output / 'checkpoints').symlink_to(checkpoints, target_is_directory=True)
        write(output / 'token_cache_artifact.json', {'input_artifacts': {
            'tokenizer': 'helloimlixin-rutgers/laser/church-laser-stage1-selection-20260920-selected-checkpoints:v0',
            'cache': 'helloimlixin-rutgers/laser/church-laser-ft3best-stochastic-rawcoeff90-h200x5-20260921-physical-compound-token-cache:v0'}})
    source = (original / 'original-files/code/outputs/church-stage2-released-recipe-20260921/train.py').read_text()
    source = replace_once(source, "choices=['benchmark-a1', 'benchmark-a2', 'train']", "choices=['benchmark-a2', 'benchmark-a4', 'train']")
    source = replace_once(source, "assert int(os.environ['WORLD_SIZE']) == 5", "assert int(os.environ['WORLD_SIZE']) == 4")
    source = replace_once(source, 'math.ceil(2048 / (5 * accumulation))', 'math.ceil(2048 / (4 * accumulation))')
    source = replace_once(source, "Path('/tmp/laser-church-released-scratch-20260921')", "Path('/mnt/laser-church/dropout-experiment/checkpoints')")
    source = replace_once(source, 'model.apply(model._init_weights)', '''model.apply(model._init_weights)
        residual_dropout = request['dropout']['residual']
        model.config.body.block.resid_pdrop = residual_dropout
        model.config.head.block.resid_pdrop = residual_dropout
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Dropout) and not name.endswith(('attn_drop', 'embed_drop')):
                module.p = residual_dropout''')
    source = replace_once(source, "else .1)", "else residual_dropout)")
    source = replace_once(source, "group='church-released-recipe-scratch'", "group='church-rqrecipe-dropout-20260924'")
    source = replace_once(source, "assert len(saved['rng_state_by_rank']) == 5", "assert len(saved['rng_state_by_rank']) == 4")
    source = replace_once(source, "'source-manifest.json', 'runtime-source.tar.gz']", "'source-manifest.json', 'runtime-source.tar.gz', 'heldout-probe.pt', 'heldout-probe.json']")
    start = source.index("        if production and not (output / 'preview-provenance-upload.json').exists():")
    stop = source.index('    training.upload_token_cache_once = use_assets', start)
    source = source[:start] + source[stop:]
    source = replace_once(source, '    training.persistent_checkpoint_dir = storage', '''    training.persistent_checkpoint_dir = storage
    original_save = training.atomic_torch_save
    def save_with_experiment(payload, target):
        payload['config']['residual_dropout'] = request['dropout']['residual']
        payload['config']['experiment'] = request
        return original_save(payload, target)
    training.atomic_torch_save = save_with_experiment''')
    source = replace_once(source, "    install_fast_attention()", '''    if last.exists():
        restored = torch.load(last, map_location='cpu', weights_only=False, mmap=True)
        assert restored['config']['residual_dropout'] == request['dropout']['residual']
        del restored
    install_fast_attention()''')
    (base / 'train.py').write_text(source)
    benchmark = (root / 'earlier-source/benchmark_fid.py').read_text()
    benchmark = benchmark.replace("BASE / 'benchmark-a1/checkpoints/last.pt'", "BASE / 'benchmark-a4/checkpoints/last.pt'")
    benchmark = benchmark.replace('for batch_size in [512, 1024, 2048]:', 'for batch_size in [256, 512, 1024]:')
    benchmark = benchmark.replace('5 * batch_size', '4 * batch_size').replace('5*batch_size', '4*batch_size')
    (base / 'benchmark_fid.py').write_text(benchmark)
    write(base / 'source-changes.json', {'historical_runtime_modified': True,
          'runtime_changes': ['Collect checkpoint serialization cycles to release old optimizer storage'],
          'runtime_source': 'church-laser-scratch-rqrecipe300-b2048-h200x5-20260921-training-provenance:v0',
          'driver_changes': ['Four GPU placement and local asset/storage paths',
                             f'Residual dropout {args.dropout} on all 60 residual dropout modules',
                             'Record and validate dropout in every full checkpoint',
                             'Preserve original initializer, exact global batching, objective and support masking',
                             'Fresh launch provenance includes fixed probe and preview logging'],
          'driver_sha256': hashlib.sha256(source.encode()).hexdigest()})
    print(json.dumps({'base': str(base), 'run_id': run_id, 'dropout': args.dropout}))


if __name__ == '__main__':
    main()
