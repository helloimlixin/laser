"""Run matched epoch-8 full-state forks with 200 versus 25.5875 bin noise."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time

REPO = Path('/workspace/Projects/laser')
PARENT = Path('/tmp/laser-imagenet-ffhq-v4-sigma200bins-scratch-20261009')
PARENT_ID = 'imagenet-rfid421-ffhq-v4-sigma200bins-scratch-4h200-20261009'
PARENT_OUT = REPO/'outputs'/PARENT_ID
BASE = Path('/tmp/laser-imagenet-ffhq-noise-ab-epoch8-20261009')
OUT = REPO/'outputs/imagenet-ffhq-noise-ab-epoch8-20261009'
SOURCE_STEP, END_STEP = 5008, 5634
BRANCHES = {'low25': 25.5875, 'control200': 200.}


def record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str)+'\n')
    temporary.replace(path)


def owned_process(pid, entry):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().split()
        command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        return stat[2] != 'Z' and str(entry).encode() in command
    except FileNotFoundError:
        return False


def fork_entry(template):
    replacements = {
        "assert ARGS.epochs == ARGS.lr_schedule_epochs == 100 and ARGS.min_lr == 0":
        "assert ARGS.epochs == 9 and ARGS.lr_schedule_epochs == 100 and ARGS.min_lr == 0",
        "kwargs['resume'] = 'never' if PHASE == 'preflight' else 'must'":
        "kwargs['resume'] = 'allow'",
        "payload = original_logging.rebase_checkpoint_fids(payload, policy['previous_evaluations'])":
        "payload = original_logging.rebase_checkpoint_fids(payload, policy['previous_evaluations'])\n"
        "        assert payload['global_step'] in (5008, 5634)\n"
        "        # Each fork owns its checkpoint rankings; parent winners stay untouched.\n"
        "        payload = dict(payload, best_fid=[], best_inception=[])\n",
        "encoded = tuple(torch.cat(values) for values in results)":
        "encoded = tuple(torch.cat(values) for values in results)\n"
        "    if not hasattr(self, '_noise_trial_first_batch_recorded'):\n"
        "        def digest(value):\n"
        "            return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()\n"
        "        record('matched-first-batch-rank'+os.environ['RANK']+'.json', dict(\n"
        "            images_sha256=digest(image_batch), atoms_sha256=digest(encoded[0]),\n"
        "            clean_coefficients_sha256=digest(encoded[1]),\n"
        "            torch_cpu_rng_sha256=digest(torch.get_rng_state()),\n"
        "            torch_cuda_rng_sha256=digest(torch.cuda.get_rng_state(image_batch.device))))\n"
        "        self._noise_trial_first_batch_recorded = True",
    }
    for old, new in replacements.items():
        if template.count(old) != 1:
            raise ValueError(f'Frozen entry boundary changed: {old}')
        template = template.replace(old, new)
    start = template.index("    if PHASE == 'production':\n        checkpoint_io.upload_selected_checkpoint_files")
    end = template.index("    record('wandb.json'", start)
    metadata = (
        "    run.config.update(dict(stage2_initialization='full_state_epoch8_fork',\n"
        f"        parent_run='helloimlixin-rutgers/laser/{PARENT_ID}',\n"
        "        trial_source_step=5008, trial_end_step=5634, trial_updates=626,\n"
        "        trial_branch=os.environ['LASER_NOISE_TRIAL_BRANCH'],\n"
        "        trial_only_training_change='coefficient_target_temperature',\n"
        "        trial_optimizer_reset=False, trial_scheduler_reset=False,\n"
        "        trial_sampler_changed=False, supersedes_run=None,\n"
        "        correction='Matched epoch8 full-state noise trial; archived FFHQ decoder/objective retained'),\n"
        "        allow_val_change=True)\n"
        "    run.save(str(OUT/'source-integrity.json'), base_path=str(OUT), policy='now')\n"
    )
    return template[:start]+metadata+template[end:]


def prepare():
    import torch
    import yaml
    sys.path[:0] = [str(PARENT/'source'), str(PARENT/'source/runtime')]
    from src.training.full_resume_upload import recovery_metadata
    if (OUT/'preparation-complete.json').exists():
        raise ValueError('Already prepared; run the immutable prepared trial')
    preservation = json.loads((OUT/'source-preservation.json').read_text())
    anchor = BASE/'source-step5008.pt'
    payload = torch.load(anchor, map_location='cpu', mmap=True, weights_only=False)
    metadata = recovery_metadata(payload)
    assert metadata['global_step'] == SOURCE_STEP and metadata['epoch'] == 8
    assert metadata['rng_ranks'] == 4 and metadata['adam_step'] == SOURCE_STEP
    assert payload['scheduler']['last_epoch'] == SOURCE_STEP
    assert payload['scheduler']['T_max'] == 62600
    for tensor in payload['state_dict'].values():
        assert torch.isfinite(tensor).all()
    for state in payload['optimizer']['state'].values():
        assert all(torch.isfinite(v).all() for v in state.values() if isinstance(v, torch.Tensor))
    record(OUT/'source-integrity.json', dict(passed=True, finite_model_and_adam=True,
        metadata=metadata, source_preservation=preservation))
    del payload
    runtime = BASE/'runtime'
    shutil.copytree(PARENT/'source', runtime,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    entry = fork_entry((PARENT/'entry.py').read_text())
    source_manifest = json.loads((PARENT_OUT/'source-manifest.json').read_text())
    for name, expected in source_manifest.items():
        if name.startswith('source/'):
            actual = hashlib.sha256((runtime/name.removeprefix('source/')).read_bytes()).hexdigest()
            assert actual == expected, name
    recipe = yaml.safe_load((PARENT/'production.yaml').read_text())
    plan = dict(source_run='helloimlixin-rutgers/laser/'+PARENT_ID,
        source_step=SOURCE_STEP, source_epoch=8, target_step=END_STEP, target_epoch=9,
        updates_per_branch=626, global_batch=2048, world_size=4,
        source_checkpoint=str(OUT/'source-step5008.pt'),
        local_source_checkpoint=str(anchor), source_fid=68.86146409589696,
        branches=BRANCHES, coefficient_bins=dict(count=2048, range=[-3, 3], uniform=True),
        sampler=dict(atom_temperature=.9, atom_top_p=.9, atom_top_k=0,
                     coeff_temperature=1., coeff_top_p=.85, coeff_top_k=0),
        schedule='saved Adam and absolute 100-epoch cosine; no reset',
        metrics=dict(backend='original-rqvae', fake_images=50000,
                     reference='released full ImageNet training statistics',
                     reference_sha256=hashlib.sha256((PARENT/'inputs/imagenet_256_train.npz').read_bytes()).hexdigest(),
                     keys=['eval/fid_original_train50k', 'eval/inception_score'], seed=261001),
        decoder_objective='unchanged archived FFHQ compound-v4 ImageNet adaptation',
        automatic_full_restart=False, restore_parent_after_trial=True,
        limitations='One-epoch continuation ablation; does not establish eventual scratch-training FID.')
    record(OUT/'plan.json', plan)
    configs = {}
    for branch, sigma in BRANCHES.items():
        base, output = BASE/branch, OUT/branch
        base.mkdir()
        output.mkdir()
        (base/'source').symlink_to(runtime, target_is_directory=True)
        for name in ('inputs', 'imagenet', 'torch-cache', 'inductor-cache'):
            (base/name).symlink_to(PARENT/name, target_is_directory=True)
        for name in ('checkpoint-staging', 'checkpoint-upload-cache', 'wandb', 'wandb-cache', 'wandb-data'):
            (base/name).mkdir()
        (base/'entry.py').write_text(entry)
        checkpoints = output/'train/checkpoints'
        checkpoints.mkdir(parents=True)
        (checkpoints/'last.pt').symlink_to(anchor)
        options = dict(recipe['options'], epochs=9, resume=True,
            resume_checkpoint=str(checkpoints/'last.pt'), output=str(output/'train'),
            checkpoint_dir=str(checkpoints), coeff_target_temperature=2*(sigma*6/2047)**2,
            fid_every=1, save_ckpt_freq=1, save_step_freq=626, max_optimizer_steps=0,
            model_only_best_checkpoints=True, upload_checkpoints=True,
            wandb_id=f'imagenet-ffhq-noise-{branch}-epoch8-ab-4h200-20261009',
            wandb_name=f'ImageNet FFHQ K4 | matched epoch8→9 | sigma {sigma:g} bins')
        configs[branch] = options
        text = '# @package _global_\n'+yaml.safe_dump(dict(recipe, options=options), sort_keys=False)
        (base/'production.yaml').write_text(text)
        (output/'train.yaml').write_text(text)
        record(output/'original-fid-only-policy.json', dict(previous_evaluations=[],
            generation_metric_keys=plan['metrics']['keys'], metric_protocol='original_rqvae_training_reference'))
        record(output/'plan.json', dict(plan, branch=branch, sigma_bins=sigma))
        shutil.copyfile(OUT/'source-integrity.json', output/'source-integrity.json')
        manifest = {str(p.relative_to(runtime)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in runtime.rglob('*') if p.is_file()}
        record(output/'source-manifest.json', dict(runtime=manifest,
            entry_sha256=hashlib.sha256(entry.encode()).hexdigest()))
    differing = {key for key in configs['low25'] if configs['low25'][key] != configs['control200'][key]}
    assert differing == {'coeff_target_temperature', 'output', 'checkpoint_dir',
                         'resume_checkpoint', 'wandb_id', 'wandb_name'}, differing
    record(OUT/'config-difference-proof.json', dict(passed=True, differing_keys=sorted(differing),
        only_training_difference='coeff_target_temperature', immutable_runtime_verified=True))
    bundle = OUT/'source-code.tar.gz'
    with tarfile.open(bundle, 'w:gz') as archive:
        archive.add(runtime, arcname='source')
        archive.add(Path(__file__), arcname='trial-controller.py')
        archive.add(OUT/'plan.json', arcname='plan.json')
        for branch in BRANCHES:
            archive.add(BASE/branch/'entry.py', arcname=branch+'/entry.py')
            archive.add(BASE/branch/'production.yaml', arcname=branch+'/production.yaml')
    for branch in BRANCHES:
        shutil.copyfile(bundle, OUT/branch/'source-code.tar.gz')
    record(OUT/'preparation-complete.json', dict(passed=True, plan=plan,
        controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), time=time.time()))
    print(json.dumps(dict(prepared=True, output=str(OUT))), flush=True)


def environment(branch):
    base, output = BASE/branch, OUT/branch
    return dict(os.environ, LASER_RUN_BASE=str(base), LASER_PERSISTENT_BASE=str(output),
        LASER_PHASE='production', LASER_ACCUMULATION='4', LASER_FFHQ_DECODER_OBJECTIVE='1',
        LASER_SUB_BIN_TARGETS='0', LASER_COEFFICIENT_SIGMA_BINS=str(BRANCHES[branch]),
        LASER_NOISE_TRIAL_BRANCH=branch, LASER_COMPILE_BLOCKS='1',
        LASER_CHECKPOINT_STAGING_DIR=str(base/'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(base/'checkpoint-upload-cache'),
        LASER_CHECKPOINT_IMMUTABLE_FILES='1', CUDA_VISIBLE_DEVICES='0,1,2,3',
        PYTHONPATH=str(base/'source')+':'+str(base/'source/runtime'),
        WANDB_API_KEY=Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip(),
        WANDB_DIR=str(base/'wandb'), WANDB_CACHE_DIR=str(base/'wandb-cache'),
        WANDB_DATA_DIR=str(base/'wandb-data'), TORCH_HOME=str(base/'torch-cache'),
        TORCHINDUCTOR_CACHE_DIR=str(base/'inductor-cache'), TORCHINDUCTOR_COMPILE_THREADS='4',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        PYTHONUNBUFFERED='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', NCCL_NVLS_ENABLE='0')


def preserve_parent():
    import torch
    sys.path[:0] = [str(PARENT/'source'), str(PARENT/'source/runtime')]
    from src.training.k4_checkpoint_io import _checkpoint_upload_source
    from src.training.full_resume_upload import recovery_metadata
    os.environ['LASER_CHECKPOINT_UPLOAD_CACHE_DIR'] = str(PARENT/'checkpoint-upload-cache')
    os.environ['LASER_CHECKPOINT_IMMUTABLE_FILES'] = '1'
    request = json.loads((OUT/'parent-stop-request.json').read_text())
    deadline = time.monotonic()+900
    while any(owned_process(pid, PARENT/'entry.py') for pid in request['workers']):
        if time.monotonic() > deadline:
            raise RuntimeError('Parent workers have not completed their checkpointed shutdown')
        record(OUT/'status.json', dict(state='waiting_for_parent_checkpoint_and_upload', time=time.time()))
        time.sleep(5)
    source = (PARENT_OUT/'train/checkpoints/last.pt').resolve()
    assert str(source) != request['previous_last_payload']
    local = _checkpoint_upload_source(source)
    payload = torch.load(local, map_location='cpu', mmap=True, weights_only=False)
    metadata = recovery_metadata(payload)
    assert metadata['global_step'] > SOURCE_STEP and metadata['rng_ranks'] == 4
    assert payload['scheduler']['last_epoch'] == metadata['global_step']
    for tensor in payload['state_dict'].values():
        assert torch.isfinite(tensor).all()
    for state in payload['optimizer']['state'].values():
        assert all(torch.isfinite(v).all() for v in state.values() if isinstance(v, torch.Tensor))
    anchor = BASE/f'preserved-parent-step{metadata["global_step"]}.pt'
    if not anchor.exists():
        os.link(local, anchor)
    record(OUT/'preserved-parent.json', dict(passed=True, finite_model_and_adam=True,
        metadata=metadata, persistent_checkpoint=str(source), local_anchor=str(anchor)))


def restore_parent():
    script = PARENT/'original-fid-only-deployment/launch_imagenet_repaired_scratch.py'
    env = dict(os.environ, LASER_FFHQ_DECODER_OBJECTIVE='1',
               LASER_COEFFICIENT_SIGMA_BINS='200', LASER_PROJECT_ROOT=str(REPO))
    with (PARENT_OUT/'supervisor.log').open('a') as stream:
        child = subprocess.Popen([sys.executable, str(script), 'resume-production'],
            cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    record(OUT/'parent-restoration.json', dict(requested=True, supervisor_pid=child.pid,
        checkpoint=str(PARENT_OUT/'train/checkpoints/last.pt'), time=time.time()))


def run():
    import torch
    prepared = json.loads((OUT/'preparation-complete.json').read_text())
    assert prepared['controller_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    lock = (BASE/'trial.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    preserve_parent()
    completed = False
    try:
        for branch in BRANCHES:
            base, output = BASE/branch, OUT/branch
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                       '--nproc-per-node=4', str(base/'entry.py'), '--config', str(base/'production.yaml')]
            with (output/'train.log').open('a') as stream:
                child = subprocess.Popen(command, cwd=base/'source', env=environment(branch),
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                while child.poll() is None:
                    record(OUT/'status.json', dict(state='running', branch=branch,
                        supervisor_pid=os.getpid(), torchrun_pid=child.pid,
                        source_step=SOURCE_STEP, target_step=END_STEP, time=time.time()))
                    time.sleep(5)
            if child.returncode:
                raise RuntimeError(f'{branch} failed with return code {child.returncode}')
            score = output/f'train/evaluations/original_generation_step_{END_STEP:07d}.json'
            assert score.exists(), 'The required original FID50k/IS evaluation is missing'
            for rank in range(4):
                proof = json.loads((output/f'verification/production/startup-rank{rank}.json').read_text())
                assert proof['restored_steps'] == [SOURCE_STEP] and not proof['empty_optimizer']
            record(OUT/branch/'completed.json', dict(passed=True, evaluation=json.loads(score.read_text()),
                checkpoint=str(output/'train/checkpoints/last.pt'), time=time.time()))
        first_batches = {}
        for rank in range(4):
            proofs = {branch: json.loads((OUT/branch/f'verification/production/matched-first-batch-rank{rank}.json').read_text())
                      for branch in BRANCHES}
            a, b = (proofs[branch] for branch in BRANCHES)
            keys = [key for key in a if key != 'time']
            assert all(a[key] == b[key] for key in keys), f'Matched initial data/RNG differs on rank {rank}'
            first_batches[rank] = a
        scores = {branch: json.loads((OUT/branch/'completed.json').read_text())['evaluation']
                  for branch in BRANCHES}
        fid_change = scores['low25']['eval/fid_original_train50k']-scores['control200']['eval/fid_original_train50k']
        comparison = dict(plan=prepared['plan'], scores=scores, candidate_minus_control_fid=fid_change,
            lower_noise_better=fid_change < 0, first_batch_and_rng_matching_verified=True,
            matched_initial_batches=first_batches,
            automatic_full_restart=False, time=time.time())
        record(OUT/'comparison.json', comparison)
        completed = True
    finally:
        restore_parent()
        record(OUT/'status.json', dict(state='completed' if completed else 'failed_parent_resume_requested',
            comparison=str(OUT/'comparison.json') if completed else None,
            parent_resume_requested=True, time=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run'])
    args = parser.parse_args()
    {'prepare': prepare, 'run': run}[args.action]()
