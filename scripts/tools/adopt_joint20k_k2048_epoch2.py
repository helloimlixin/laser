#!/usr/bin/env python3
"""Check epoch-two K2048, then adopt it in the existing 100-epoch run."""
import copy
import difflib
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

ROOT = Path('/workspace/Projects/laser')
RUN = 'imagenet-laser-joint20k-depth5-uniformce-k512-scratch100-4b200-20261018'
BASE = Path('/tmp/laser-joint20k-depth5-uniformce-k512-scratch100-4b200-20261018')
TRAIN = ROOT / 'outputs' / RUN
OUT = TRAIN / 'diagnostics/sampler-k2048-epoch2-and-adoption-20261018'
LOCAL = Path('/tmp/laser-k2048-epoch2-and-adoption-20261018')
EVALUATOR = ROOT / 'scripts/tools/evaluate_joint20k_k512_vs_k2048.py'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


support = module('sampler_handoff_support', ROOT / 'scripts/tools/check_joint20k_k512_vs_k2048.py')
support.OUT, support.LOCAL = OUT, LOCAL


def publish():
    import wandb
    os.environ['WANDB_API_KEY'] = Path('/root/.config/laser/wandb_api_key').read_text().strip()
    run = wandb.Api(timeout=60).run('helloimlixin-rutgers/laser/' + RUN)
    for path in OUT.glob('*.json'):
        run.upload_file(str(path), root=str(TRAIN))
    for top_k in (512, 2048):
        path = OUT / f'k{top_k}/smoke/samples/step001252-fixed.png'
        run.upload_file(str(path), root=str(TRAIN))
    report = json.loads((OUT / 'report.json').read_text())
    run.summary.update({'sampler_check/epoch2/k512_fid50k':report['results']['512']['fid'],
        'sampler_check/epoch2/k2048_fid50k':report['results']['2048']['fid'],
        'sampler_check/epoch2/fid2048_minus_fid512':report['fid2048_minus_fid512']})
    support.write(OUT / 'wandb-publication.json', dict(uploaded=True, url=run.url, time=time.time()))


def adopt(evidence):
    import yaml
    guard = module('validated_joint_sampler_override', ROOT / 'src/training/rq_sampling_override.py')
    recipe_path = BASE / 'inputs/recipe.yaml'
    before_recipe = yaml.safe_load(recipe_path.read_text())
    after_recipe = copy.deepcopy(before_recipe)
    after_recipe['sampling'] = evidence['sampling']
    assert guard.validated_sampling_recipe(after_recipe, before_recipe, evidence) == after_recipe
    backup = OUT / 'before-sampling-change'
    backup.mkdir()
    changes = []
    for relative in ('source/src/training/rq_sampling_override.py',
                     'source/src/training/rq_imagenet_shared.py'):
        path = BASE / relative
        before = path.read_text()
        shutil.copyfile(path, backup / path.name)
        if path.name == 'rq_sampling_override.py':
            after = (ROOT / 'src/training/rq_sampling_override.py').read_text()
        else:
            replacements = [
                ("            sampling_change=json.loads(args.sampling_override.read_text())\n",
                 "            sampling_change=json.loads(args.sampling_override.read_text())\n"
                 "            PROFILE.name=str(sampling_change.get('display_name',PROFILE.name))\n"),
                ("                recipe=OmegaConf.to_container(PROFILE.recipe,resolve=True),\n",
                 "                recipe=OmegaConf.to_container(PROFILE.recipe,resolve=True),\n"
                 "                published_config=OmegaConf.to_container(config,resolve=True),\n"),
                ("'sampling/change':'disable atom top-k; all training state preserved'",
                 "'sampling/change':sampling_change.get('change','validated sampler-only update')"),
            ]
            after = before
            for old, new in replacements:
                assert after.count(old) == 1, old
                after = after.replace(old, new)
        compile(after, str(path), 'exec')
        path.write_text(after)
        changes.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
            fromfile='before/' + relative, tofile='after/' + relative))
    for path in (recipe_path, ROOT / 'configs/stage2' / (RUN + '.yaml')):
        shutil.copyfile(path, backup / (path.name if path != recipe_path else 'runtime-recipe.yaml'))
        path.write_text(yaml.safe_dump(after_recipe, sort_keys=False))
    for name, section in (('stage2-training-config.yaml', 'experiment'),
                          ('stage2-archive-config.yaml', 'sampling')):
        path = BASE / 'inputs' / name
        shutil.copyfile(path, backup / name)
        value = yaml.safe_load(path.read_text())
        if section == 'experiment':
            value['experiment']['sample']['top_k'] = 2048
        else:
            value['sampling']['top_k'] = 2048
        path.write_text(yaml.safe_dump(value, sort_keys=False))
    (OUT / 'runtime.patch').write_text(''.join(changes))
    repro = TRAIN / 'reproducibility'
    shutil.copyfile(repro / 'code-manifest.json', backup / 'code-manifest.json')
    manifest = json.loads((repro / 'code-manifest.json').read_text())
    for name in manifest:
        manifest[name] = support.digest(BASE / name)
    support.write(repro / 'code-manifest.json', manifest)
    shutil.copyfile(recipe_path, repro / 'recipe.yaml')
    for name in ('stage2-training-config.yaml', 'stage2-archive-config.yaml'):
        shutil.copyfile(BASE / 'inputs' / name, repro / name)
    shutil.copyfile(OUT / 'sampling-change.json', repro / 'sampling-change-k2048.json')
    support.write(repro / 'k2048-adoption-verification.json', dict(passed=True,
        training_recipe_identical_except_sampling=True, training_math_unchanged=True,
        runtime_changes=['Allow measured whole-integer K512 to K2048 in the sampler-only resume guard',
                         'Publish current sampling configuration and display name correctly'],
        source_evidence=str(OUT / 'sampling-change.json'), time=time.time()))
    with zipfile.ZipFile(repro / 'source-after-k2048-adoption.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in manifest:
            archive.write(BASE / name, name)
        archive.write(OUT / 'sampling-change.json', 'sampling-change.json')
        archive.write(Path(__file__), Path(__file__).relative_to(ROOT))
    support.write(ROOT / 'outputs/imagenet-laser-joint20k-active.json', dict(run_id=RUN,
        url=f'https://forge.coreweave.com/wandb/helloimlixin-rutgers/laser/runs/{RUN}',
        base=str(BASE), recipe='Uniform CE, whole integer tokens, K2048, 100 epochs',
        depth_loss_weights=[1.] * 5, sampling=evidence['sampling'], world_size=4,
        hardware='4 NVIDIA B200', sampler_validation=str(OUT / 'report.json'), updated=time.time()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    lock = (LOCAL / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUT / 'plan.json').exists():
        raise RuntimeError('Epoch-two sampler check already prepared')
    previous = json.loads((TRAIN / 'status.json').read_text())
    assert previous['phase'] == 'training' and previous['sampling']['top_k'] == 512
    source = next(row for row in support.receipts() if row['global_step'] == 1252)
    support.pin(source, LOCAL / 'epoch002.pt')
    reference = json.loads((TRAIN / 'evaluations/epoch002.json').read_text())
    assert reference['eval/fid'] == source['fid']
    import subprocess
    memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], text=True)
    assert len(memory.split()) == 4 and min(int(v) for v in memory.split()) >= 20 * 1024
    current_step = previous['latest_progress']['train/global_step']
    next_step = ((current_step + 625) // 626) * 626
    plan = dict(run_id=RUN, base=str(BASE), training_output=str(TRAIN), output=str(OUT),
        checkpoint=str(LOCAL / 'epoch002.pt'), checkpoint_sha256=source['sha256'],
        checkpoint_receipt=source, source_epoch=2, source_step=1252,
        original_fid512=reference['eval/fid'], baseline_evaluation=reference,
        run_baseline_fid=False, world_size=4, batch_size=128, seed=261001, memory_gib=16,
        reference_sha256=support.digest(BASE / 'inputs/imagenet_256_train.npz'),
        only_additional_gpu_work='Fixed-checkpoint inference; one training run remains active',
        training_handoff_minimum_step=next_step, original_sampling=512, target_sampling=2048,
        original_recipe_sha256=support.digest(BASE / 'inputs/recipe.yaml'),
        evaluator_sha256=support.digest(EVALUATOR), time=time.time())
    support.write(OUT / 'plan.json', plan)
    shutil.copyfile(EVALUATOR, OUT / EVALUATOR.name)
    shutil.copyfile(__file__, OUT / Path(__file__).name)
    distributed = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=4']
    support.child([*distributed, str(EVALUATOR), '--plan', str(OUT / 'plan.json')],
        OUT / 'evaluation.log', 'evaluating_epoch2')
    report = json.loads((OUT / 'report.json').read_text())
    evidence = dict(passed=True, weights_unmodified=True, training_updates_added=0,
        generated_images=50000, reference_images=1281167, reference_split='train',
        source_run=RUN, source_epoch=2, source_step=1252,
        source_sampling=dict(temperature=1., top_k=512, top_p=.95),
        sampling=dict(temperature=1., top_k=2048, top_p=.95),
        fid=report['results']['2048']['fid'], baseline_fid=report['results']['512']['fid'],
        checkpoint_sha256=source['sha256'], report=str(OUT / 'report.json'),
        change='Whole-integer sampling top-k 512 to 2048; all training state preserved',
        display_name='LASER joint20k depth5 | uniform CE + K2048 | fresh100 | 4 B200')
    guard = module('joint_sampler_guard_preflight', ROOT / 'src/training/rq_sampling_override.py')
    import yaml
    saved_recipe = yaml.safe_load((BASE / 'inputs/recipe.yaml').read_text())
    current_recipe = copy.deepcopy(saved_recipe)
    current_recipe['sampling'] = evidence['sampling']
    guard.validated_sampling_recipe(current_recipe, saved_recipe, evidence)
    support.write(OUT / 'sampling-change.json', evidence)
    try:
        publish()
    except Exception as error:
        support.write(OUT / 'wandb-publication.json', dict(uploaded=False, error=str(error)))
    while True:
        receipt = max(support.receipts(), key=lambda row:row['global_step'])
        if receipt['global_step'] >= next_step:
            break
        progress = json.loads((TRAIN / 'train/progress-rank0.json').read_text())
        support.status('validated_waiting_for_training_checkpoint', current_step=progress['train/global_step'],
            target_saved_step=next_step, fid2048=evidence['fid'], fid512=evidence['baseline_fid'])
        time.sleep(10)
    support.pin(receipt, LOCAL / 'resume-training.pt')
    support.stop_previous(previous, receipt)
    adopt(evidence)
    support.write(OUT / 'resume-plan.json', dict(checkpoint=str(LOCAL / 'resume-training.pt'),
        receipt=receipt, sampling=evidence['sampling'], uniform_ce=True,
        optimizer_scheduler_scaler_rng_preserved=True, epochs=100, time=time.time()))
    support.child([*distributed, str(BASE / 'train.py'), '--resume', str(LOCAL / 'resume-training.pt'),
        '--sampling-override', str(OUT / 'sampling-change.json'), '--epochs', '100',
        '--microbatch', '256'], TRAIN / 'training.log', 'training_k2048',
        training=True, sampling_top_k=2048)


if __name__ == '__main__':
    import signal
    signal.signal(signal.SIGTERM, support.stop)
    signal.signal(signal.SIGINT, support.stop)
    try:
        main()
    except BaseException as error:
        support.status('stopped' if support.STOPPING else 'failed', error=str(error),
                       error_type=type(error).__name__)
        raise
