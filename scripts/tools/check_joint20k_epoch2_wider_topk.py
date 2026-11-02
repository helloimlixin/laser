#!/usr/bin/env python3
"""Evaluate K2500, K3000, and K4096 on the saved epoch-two model."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from wandb_config_only import update_config_only

ROOT = Path('/workspace/Projects/laser')
RUN = 'imagenet-laser-joint20k-depth5-uniformce-k512-scratch100-4b200-20261018'
BASE = Path('/tmp/laser-joint20k-depth5-uniformce-k512-scratch100-4b200-20261018')
TRAIN = ROOT / 'outputs' / RUN
PRIOR = TRAIN / 'diagnostics/sampler-k2048-epoch2-and-adoption-20261018'
OUT = TRAIN / 'diagnostics/sampler-k2500-k3000-k4096-epoch2-20261018'
LOCAL = Path('/tmp/laser-k2500-k3000-k4096-epoch2-20261018')
EVALUATOR = ROOT / 'scripts/tools/evaluate_joint20k_k512_vs_k2048.py'
spec = importlib.util.spec_from_file_location('wider_topk_support', ROOT / 'scripts/tools/check_joint20k_k512_vs_k2048.py')
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)
support.OUT, support.LOCAL = OUT, LOCAL


def publish():
    import wandb
    os.environ['WANDB_API_KEY'] = Path('/root/.config/laser/wandb_api_key').read_text().strip()
    run = wandb.Api(timeout=60).run('helloimlixin-rutgers/laser/' + RUN)
    for path in OUT.glob('*.json'):
        run.upload_file(str(path), root=str(TRAIN))
    report = json.loads((OUT / 'report.json').read_text())
    for top_k in (2500, 3000, 4096):
        run.upload_file(str(OUT / f'k{top_k}/fid50k.json'), root=str(TRAIN))
        run.upload_file(str(OUT / f'k{top_k}/smoke/samples/step001252-fixed.png'), root=str(TRAIN))
    # The active training writer replaces summary state, so keep the fixed-checkpoint
    # comparison in config metadata as well as the uploaded report.
    metadata = dict(source_epoch=2, source_step=1252,
        generated_images_per_policy=50000, seed=261001, temperature=1., top_p=.95,
        fid50k={key:value['fid'] for key,value in report['results'].items()},
        lowest_fid_top_k=report['lower_fid_policy'], active_training_top_k=2048,
        report=str((OUT / 'report.json').relative_to(TRAIN)))
    update_config_only(run, {'sampler_check_epoch2': metadata})
    support.write(OUT / 'wandb-publication.json', dict(uploaded=True,
        results_config_key='sampler_check_epoch2', url=run.url, time=time.time()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    lock = (LOCAL / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUT / 'plan.json').exists():
        raise RuntimeError('Wider sampler check already prepared')
    training_status = json.loads((TRAIN / 'status.json').read_text())
    assert training_status['phase'] == 'training' and training_status['sampling']['top_k'] == 2048
    memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], text=True)
    assert len(memory.split()) == 4 and min(int(v) for v in memory.split()) >= 20 * 1024
    prior_plan = json.loads((PRIOR / 'plan.json').read_text())
    prior_report = json.loads((PRIOR / 'report.json').read_text())
    assert prior_report['completed'] and prior_report['source_step'] == 1252
    support.pin(prior_plan['checkpoint_receipt'], LOCAL / 'epoch002.pt')
    baseline = json.loads((TRAIN / 'evaluations/epoch002.json').read_text())
    plan = dict(run_id=RUN, base=str(BASE), training_output=str(TRAIN), output=str(OUT),
        checkpoint=str(LOCAL / 'epoch002.pt'), checkpoint_sha256=prior_plan['checkpoint_sha256'],
        source_epoch=2, source_step=1252, original_fid512=baseline['eval/fid'],
        baseline_evaluation=baseline, run_baseline_fid=False, world_size=4,
        batch_size=128, seed=261001, memory_gib=16,
        top_k_values=[512, 2500, 3000, 4096],
        prior_results={'2048':prior_report['results']['2048']},
        sampling_override_evidence=str(PRIOR / 'sampling-change.json'),
        reference_sha256=prior_plan['reference_sha256'],
        only_additional_gpu_work='Fixed-checkpoint inference; one training run remains active',
        training_sampler_unchanged=2048, original_recipe_sha256=support.digest(BASE / 'inputs/recipe.yaml'),
        evaluator_sha256=support.digest(EVALUATOR), time=time.time())
    support.write(OUT / 'plan.json', plan)
    shutil.copyfile(EVALUATOR, OUT / EVALUATOR.name)
    shutil.copyfile(__file__, OUT / Path(__file__).name)
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=4',
               str(EVALUATOR), '--plan', str(OUT / 'plan.json')]
    support.child(command, OUT / 'evaluation.log', 'evaluating_epoch2_wider_topk')
    assert support.digest(BASE / 'inputs/recipe.yaml') == plan['original_recipe_sha256']
    assert support.digest(LOCAL / 'epoch002.pt') == plan['checkpoint_sha256']
    try:
        publish()
    except Exception as error:
        support.write(OUT / 'wandb-publication.json', dict(uploaded=False, error=str(error)))
    support.status('completed', training_sampler=2048, training_recipe_modified=False)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, support.stop)
    signal.signal(signal.SIGINT, support.stop)
    try:
        main()
    except BaseException as error:
        support.status('stopped' if support.STOPPING else 'failed',
                       error=str(error), error_type=type(error).__name__)
        raise
