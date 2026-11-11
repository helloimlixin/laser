"""Prepare a best-checkpoint continuation using the recovered frozen runtime."""
import difflib
import json
from pathlib import Path
import shutil
import subprocess

import yaml

ROOT = Path('/workspace/Projects/laser')
BASE = Path('/tmp/laser-best-b300-20261104/frozen')
RUN = 'imnet-j16384-best14574-continue100-1b300-20261104'
OUT = ROOT / 'outputs' / RUN
REQUESTED = 'imnet-j16384-a4096-fid14574-lrquarter-trial64-1b200-20261104'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    assert not (OUT / 'launch.json').exists(), 'Inspect the existing launch first'
    repro = OUT / 'reproducibility'
    repro.mkdir(parents=True, exist_ok=True)
    recipe_path = BASE / 'inputs/recipe.yaml'
    recipe = yaml.safe_load(recipe_path.read_text())
    recipe.update(run_id=RUN, name='ImageNet joint16384 | best14574 | quarter LR | continue100 | 1 B300')
    recipe['lineage'].update(requested_run_id=REQUESTED,
        reason='User requested continuation from the current best of the quarter-LR run; restore its preserved source best, Adam, RNG and cosine clock.')
    recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False))
    execution_path = BASE / 'inputs/execution.json'
    execution = json.loads(execution_path.read_text())
    execution.update(hardware='1 NVIDIA B300 SXM6 AC 275040 MiB', repair_trial=False,
        quality_guard=dict(enabled=False, reason='Continue the existing user-authorized run through epoch100'),
        user_resume_override=None, requested_run_id=REQUESTED,
        baseline_measurement_origin='Inherited verified FID50k of exact source weights measured on B200; not remeasured on B300',
        matched_full50k_initial_evaluation=False,
        rng_policy='Restore source rank0 RNG and preserve global sample order and per-image augmentation. Hardware/topology changes are not bitwise replay.')
    for key in ['peak_allocated_gib', 'benchmark_median_images_per_second', 'pilot_gate',
                'pilot_optimizer_updates', 'lr_multiplier_vs_source', 'pilot_optimizer_updates']:
        execution.pop(key, None)
    write(execution_path, execution)
    write(BASE / 'inputs/hardware.json', dict(gpu=subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], text=True).strip(),
        world_size=1, microbatch=256, accumulation_steps=8, global_batch=2048))

    # The old measurement remains useful, but the same-hardware claim must not
    # carry across from the B200 runtime to this B300 continuation.
    path = BASE / 'source/src/training/rq_imagenet_shared.py'
    before = path.read_text()
    after = before.replace('    verified_baseline_reused=False\n',
        '    verified_baseline_reused=False\n    inherited_evaluation_world_size=saved_world if args.resume else world\n', 1)
    after = after.replace('        verified_baseline_reused=True\n',
        '        inherited_evaluation_world_size=measured[\'world_size\']\n'
        '        verified_baseline_reused=False  # Inherited B200 measurement, not a B300 remeasurement.\n', 1)
    after = after.replace('world_size=saved_world if inherited_baseline and not verified_baseline_reused else world,',
        'world_size=inherited_evaluation_world_size if inherited_baseline else world,\n'
        "                measurement_hardware='B200' if inherited_baseline else 'B300',\n"
        "                continuation_hardware='B300', remeasured_on_continuation_hardware=not inherited_baseline,")
    assert after != before
    path.write_text(after)
    patch = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
        fromfile='recovered/source/src/training/rq_imagenet_shared.py',
        tofile='continuation/source/src/training/rq_imagenet_shared.py'))
    (repro / 'b300-provenance.patch').write_text(patch)
    support = BASE / 'source/src/training/joint_runtime_support.py'
    support.write_text(support.read_text().replace(
        'ImageNet Joint16384 best73 quarter LR trial64 1B200 20261104',
        'ImageNet Joint16384 best14574 continue100 1B300 20261104'))
    for name in ['recipe.yaml', 'execution.json', 'hardware.json', 'sampling-continuation.json',
                 'matched-baseline-evaluation.json']:
        shutil.copyfile(BASE / 'inputs' / name, repro / name)
    shutil.copyfile(BASE.parent / 'checkpoint-verification.json', repro / 'source-checkpoint-verification.json')
    best = dict(artifact=recipe['lineage']['source_artifact'], state='COMMITTED',
        path=str(BASE / 'inputs/resume-checkpoint.pt'), global_step=45698,
        fid=14.573915136162327, inception_score=94.00654602050781,
        measurement_hardware='B200', remeasured_on_continuation_hardware=False)
    write(OUT / 'best-checkpoint.json', best)
    write(OUT / 'continuation-plan.json', dict(run_id=RUN, requested_run_id=REQUESTED,
        best=best, target_epoch=100, target_step=62600, lr_amplitude_factor=0.0375,
        initial_lr=3.175201262590684e-6, optimizer_and_scheduler_restored=True,
        global_batch=2048, microbatch=256, accumulation_steps=8,
        fresh_fid_every_epochs=1, quality_early_stop=False,
        gradient_guidance_enabled=False, gradient_clip_max_norm=0.1))


if __name__ == '__main__':
    main()
