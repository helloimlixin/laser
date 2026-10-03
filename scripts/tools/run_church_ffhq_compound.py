#!/usr/bin/env python3
"""Supervise the verified Church cache, CUDA preflight, and online stage-2 run."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    os.replace(temporary, path)


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--runtime-manifest', type=Path)
    parser.add_argument('--preflight-report', type=Path)
    args = parser.parse_args()
    base = args.base.resolve()
    base.mkdir(parents=True, exist_ok=True)
    runtime_manifest = args.runtime_manifest or base/'runtime-manifest.json'
    lock = open('/mnt/laser-church/supervisor.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from omegaconf import OmegaConf
    from src.training.cli import load_config
    cfg = load_config(args.config)
    options = OmegaConf.to_container(cfg.options, resolve=True)
    os.environ.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                      OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1',
                      WANDB_DIR='/mnt/laser-church/wandb',
                      LASER_CHECKPOINT_STAGING_DIR='/mnt/laser-church/checkpoint-staging',
                      LASER_CHECKPOINT_IMMUTABLE_FILES='1',
                      LASER_CHECKPOINT_UPLOAD_CACHE_DIR='/mnt/laser-church/upload-cache')
    Path(os.environ['WANDB_DIR']).mkdir(parents=True, exist_ok=True)
    child = None
    stopped = False
    wb = None

    def stop(number, _frame):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None:
            child.send_signal(number)

    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, stop)

    def status(phase, **values):
        row = dict(phase=phase, timestamp=time.time(), pid=os.getpid(),
                   child_pid=None if child is None else child.pid, **values)
        write_json(base/'status.json', row)
        print(json.dumps(row), flush=True)
        if wb is not None:
            wb.summary['pipeline/phase'] = phase
            wb.log({'pipeline/heartbeat_unix': row['timestamp']})

    def command(phase, argv, devices='0,1,2,3'):
        nonlocal child
        with (base/f'{phase}.log').open('a') as output:
            child = subprocess.Popen(argv, cwd=ROOT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=devices),
                stdout=output, stderr=subprocess.STDOUT)
            (base/f'{phase}.pid').write_text(str(child.pid)+'\n')
            status(phase, command=argv)
            last = time.monotonic()
            while child.poll() is None:
                time.sleep(5)
                if time.monotonic() - last >= 60:
                    status(phase)
                    last = time.monotonic()
            code = child.returncode
            child = None
            if code:
                raise RuntimeError(f'{phase} exited {code}; inspect {base}/{phase}.log')
            if stopped:
                raise InterruptedError('Supervisor stopped by signal')

    def train_command(config, ranks=4):
        return [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                f'--nproc-per-node={ranks}', str(ROOT/'train.py'), '--config', str(config)]

    try:
        manifest = json.loads(runtime_manifest.read_text())
        for relative, expected in manifest.items():
            if sha256(ROOT/relative) != expected:
                raise RuntimeError(f'Runtime source changed: {relative}')
        provenance = json.loads((base/'assets/stage1-provenance.json').read_text())
        if sha256(Path(options['checkpoint'])) != provenance['sha256']:
            raise RuntimeError('Tokenizer provenance mismatch')
        import wandb
        recipe = OmegaConf.to_container(cfg.get('recipe', {}), resolve=True)
        recipe_metadata = {
            'recipe_reference': recipe.get('reference', 'helloimlixin-rutgers/laser/ffhqcmp0804205803'),
            'dictionary_atom_vector_conditioning': recipe.get('dictionary_atom_vector_conditioning', True),
            'recipe': recipe,
        }
        if recipe.get('reference_fid') is not None:
            recipe_metadata['recipe_reference_fid'] = recipe['reference_fid']
        wb = wandb.init(entity=options['wandb_entity'], project=options['wandb_project'],
            id=options['wandb_id'], name=options['wandb_name'], resume='allow', mode='online',
            config={**recipe_metadata,
                    'stage1_provenance': provenance, 'options': options})
        cache = Path(options['token_cache'])
        validation = cache.with_suffix('.validation.json')
        cache_pid_path = base/'cache.pid'
        cache_pid = int(cache_pid_path.read_text()) if cache_pid_path.is_file() else 0
        while cache_pid and alive(cache_pid):
            status('building_token_cache', cache_pid=cache_pid)
            for _ in range(12):
                if stopped:
                    raise InterruptedError('Supervisor stopped during cache extraction')
                if not alive(cache_pid):
                    break
                time.sleep(5)
        if not cache.is_file() or not validation.is_file():
            command('cache', json.loads((base/'cache-command.json').read_text()), '0,1,2,3,4')
        report = json.loads(validation.read_text())
        if not report['passed'] or report['items'] != 126227:
            raise RuntimeError('Full Church token cache validation failed')
        import torch
        payload = torch.load(cache, map_location='cpu', weights_only=True, mmap=True)
        meta = payload['meta']
        if (meta['stage1_checkpoint_sha256'] != provenance['sha256']
                or meta['clip_coefficients'] or meta['coefficient_storage'] != 'fp32'
                or meta['encoder_precision'] != 'fp32'
                or tuple(payload['atoms'].shape) != (126227, 8, 8, 4)
                or bool(payload['labels'].any())):
            raise RuntimeError('Cache provenance, dimensions, labels, or precision mismatch')
        wb.summary['cache/validation'] = report
        wb.summary['cache/coeff_scales'] = meta['coeff_scales']
        del payload
        import shutil
        for name in ['compound-cache.pt', 'compound-cache.validation.json']:
            if not (base/'assets'/name).is_file():
                shutil.copyfile(cache.parent/name, base/'assets'/name)
        preflight = args.preflight_report or base/'preflight.json'
        if not preflight.is_file():
            test = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            test.defaults = ['_self_']
            test.options.update(output='/mnt/laser-church/preflight/train',
                checkpoint_dir=str(base/'preflight/checkpoints'),
                wandb_mode='disabled', wandb_id=None, upload_checkpoints=False,
                resume=False, save_step_freq=1, max_optimizer_steps=2,
                sample_grid_every=0, fid_every=0, geometry_start_epoch=0,
                geometry_warmup_epochs=0, async_checkpoint_copy=False)
            path = ROOT/'preflight.yaml'
            OmegaConf.save(test, path)
            training_receipt = Path('/mnt/laser-church/preflight/training-verified.json')
            if not training_receipt.is_file():
                command('preflight-training', train_command(path))
            else:
                # Reuse completed numerical checks while validating a repaired
                # persistence path, without introducing these weights into training.
                from src.training.rqtransformer import atomic_torch_save
                previous = json.loads(training_receipt.read_text())
                state = torch.load(previous['path'], map_location='cpu', weights_only=False)
                atomic_torch_save(state, base/'preflight/checkpoints/last.pt')
                del state
            state = torch.load(base/'preflight/checkpoints/last.pt',
                               map_location='cpu', weights_only=False, mmap=True)
            assert state['global_step'] == 2 and state['optimizer']['state']
            assert len(state['rng_state_by_rank']) == 4 and state['scheduler']
            assert all(bool(torch.isfinite(v).all()) for v in state['state_dict'].values())
            parameters = sum(v.numel() for v in state['state_dict'].values())
            del state
            test.options.pop('max_optimizer_steps')
            test.options.update(resume=True, generation_smoke_test=True)
            OmegaConf.save(test, path)
            command('preflight-generation', train_command(path))
            write_json(preflight, dict(passed=True, parameters=parameters,
                training_updates=2, geometry_enabled=True, ranks=4,
                generation_batch_per_rank=options['fid_batch_size']))
        check = json.loads(preflight.read_text())
        if not check['passed']:
            raise RuntimeError('Preflight has not passed')
        if args.preflight_report and check.get('runtime_manifest_sha256') != sha256(runtime_manifest):
            raise RuntimeError('Preflight does not match the selected runtime manifest')
        wb.summary.update({'preflight/passed': True, 'model/parameters': check['parameters'],
            'checkpoints/latest_online_file': 'last.pt',
            'checkpoints/best_fid_online_file': 'best-fid-01.pt'})
        artifact = wandb.Artifact(options['wandb_id']+'-launch', type='run-config')
        for path in [args.config, preflight, runtime_manifest,
                     base/'assets/stage1-provenance.json', validation, base/'focused-tests.log']:
            artifact.add_file(str(path), name=path.name)
        wb.log_artifact(artifact, aliases=['latest']).wait()
        status('starting_training')
        wb.summary['pipeline/phase'] = 'training'
        wb.finish()
        wb = None
        command('training', train_command(args.config))
        status('completed')
    except BaseException as error:
        status('stopped' if isinstance(error, InterruptedError) else 'failed', error=str(error))
        (base/'failure.txt').write_text(traceback.format_exc())
        if wb is not None:
            wb.finish(exit_code=1)
        raise


if __name__ == '__main__':
    main()
