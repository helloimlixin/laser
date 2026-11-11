"""Evaluate samplers for higher Inception Score on the epoch 76 EMA."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from sweep import read, write


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = Path('/tmp/laser-rq24576-rewind60-teacher010-8h100-20261110')
TRAIN = ROOT / 'outputs/imagenet-rq24576-rewind60-teacher010-8h100-20261110'
RUN = 'imagenet-rq24576-noisematched-ema76-is-sampler-20261111'
LOCAL = Path('/tmp') / RUN
OUT = ROOT / 'outputs' / RUN
EPOCH = 76
STEP = EPOCH * 626
SEEDS = [261001, 271001, 281001]


def freeze_checkpoint():
    evaluation_path = TRAIN / f'evaluations/epoch{EPOCH:03}.json'
    live_checkpoint = BASE / 'checkpoints' / f'step-{STEP:07}.pt'
    durable_checkpoint = TRAIN / 'persistent-train/checkpoints' / live_checkpoint.name
    frozen = LOCAL / 'frozen-checkpoint.pt'
    deadline = time.monotonic() + 600
    while not frozen.exists():
        source = live_checkpoint if live_checkpoint.exists() else durable_checkpoint
        if evaluation_path.exists() and source.is_file():
            try:
                os.link(source, frozen)
            except FileNotFoundError:
                continue
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f'Epoch {EPOCH} evaluation and checkpoint did not both become available')
        time.sleep(5)
    evaluation = read(evaluation_path)
    assert evaluation['train/global_step'] == STEP
    assert evaluation['generated_images'] == 50000 and evaluation['real_images'] == 1281167
    assert evaluation['real_split'] == 'train' and not evaluation['rejection_sampling']
    assert evaluation['seed'] == SEEDS[0] and evaluation['is_shuffle_seed'] == 261002
    with frozen.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    index = read(TRAIN / 'epoch-checkpoints.json')
    receipt = index.get(str(STEP))
    if receipt:
        assert receipt['sha256'] == digest and receipt['bytes'] == frozen.stat().st_size
    return evaluation['ema'], frozen, digest, receipt


def build_plan():
    baseline, checkpoint, digest, receipt = freeze_checkpoint()
    standard_k = [1536, 2048, 3072, 6144]
    wide_k = [2048, 2560, 4096, 8192]
    standard_p = [.95, .95, .98, .99]

    def sampler(temperature, top_k, top_p=standard_p):
        return dict(temperature=temperature, top_k=top_k, top_p=top_p)

    presets = {
        'narrow_semantic': sampler(.95, [1024, 1536, 3072, 6144], [.93, .95, .98, .99]),
        'depth_cool_front': sampler([.925, .925, .975, .975], wide_k),
        'current': baseline['sampler'],
        'earlier_tuned': sampler(.95, [1024, 1536, 2048, 4096]),
        'wide_t09375': sampler(.9375, wide_k),
        'coarse_k512': sampler([.90, .925, .975, .975], [512, 1536, 3072, 6144], [.93, .95, .98, .99]),
    }
    assert presets['current'] == sampler(.975, standard_k)
    cases = dict(presets)
    seeds = {}
    routes = {}
    for name, settings in presets.items():
        for seed in SEEDS:
            case = f'{name}__s{seed}'
            cases[case] = settings
            seeds[case] = seed
            routes[case] = name
    classes = [
        dict(name=name, imagenet_id=index)
        for name, index in [('cheeseburger', 933), ('teapot', 849), ('samoyed', 258),
                            ('bald eagle', 22), ('red fox', 277), ('hotpot', 926),
                            ('wombat', 106), ('timber wolf', 269)]
    ]
    plan = dict(
        base=str(BASE), local=str(LOCAL), output=str(OUT), source_training_output=str(TRAIN),
        source_run=TRAIN.name, source_epoch=EPOCH, source_step=STEP, checkpoint=str(checkpoint),
        checkpoint_bytes=checkpoint.stat().st_size, checkpoint_sha256=digest,
        weights='ema', state_dict_key='state_dict', ema_decay=baseline['ema_decay'],
        ema_updates=baseline['ema_updates'], policies=cases, policy_seeds=seeds,
        case_base_policy=routes, screen_policies=[], screen_images=0, screen_seed=26111086,
        prespecified_primary_policies=list(presets), confirm_images=50000,
        confirm_seed=SEEDS[0], is_shuffle_seed=261002, generation_seeds=SEEDS,
        replication_seeds=SEEDS[1:], world_size=8, batch=64, decode_batch=8, memory_gib=12,
        study_kind='Inception Score priority', teacher_temperature_at_checkpoint=.085,
        reference_images=1281167, reference_split='train',
        reference_sha256='3f9c92d15755e76ec312964a819e3e19b9cf3aadc618a7ae99f5e8aa96501260',
        active_recipe_sha256=hashlib.sha256((BASE / 'inputs/recipe.yaml').read_bytes()).hexdigest(),
        logged_baseline=baseline, training_continues=True, no_extra_hardware=True,
        inference_sampler_precision='native FP16', decoder_precision='FP32',
        inception_precision='FP32', rejection_sampling=False,
        adapter_probe=presets['depth_cool_front'], published_target_fid=11.56,
        selection='Replicate highest IS and strongest joint FID/IS result, or minimum FID if no primary config improves both.',
        recommendation_rule='Prefer repeatable IS improvement with mean FID no higher than the control.',
        previews=dict(classes=classes, images_per_class=8, seed_offset=86000000,
                      primary_seed=SEEDS[0], unfiltered=True, excluded_from_metric_images=True),
    )
    retention = dict(pinned_local_checkpoint=str(checkpoint), bytes=checkpoint.stat().st_size,
                     sha256=digest, source_epoch=EPOCH, source_step=STEP, canonical_receipt=receipt)
    for folder in (LOCAL, OUT):
        write(folder / 'plan.json', plan)
        write(folder / 'checkpoint-retention.json', retention)
    return plan


def main():
    LOCAL.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    with (LOCAL / 'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(OUT / 'queue-status.json', dict(phase='waiting_for_epoch76_checkpoint'))
        plan = build_plan()
        write(OUT / 'queue-status.json', dict(phase='memory_preflight', source_epoch=EPOCH))
        worker = [sys.executable, '-u', str(HERE / 'worker.py'), '--plan', str(LOCAL / 'plan.json')]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
        with (LOCAL / 'preflight.log').open('ab', buffering=0) as log:
            subprocess.run(worker + ['--rank', '0', '--preflight'], env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        proof = read(LOCAL / 'preflight-rank-0.json')
        assert proof['passed'] and proof['peak_reserved_gib'] < plan['memory_gib']
        for folder in (LOCAL, OUT):
            write(folder / 'memory-preflight.json', proof)
        command = [sys.executable, '-u', str(HERE / 'sweep.py'), '--plan', str(LOCAL / 'plan.json')]
        with (LOCAL / 'coordinator.log').open('ab', buffering=0) as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        write(OUT / 'queue-status.json', dict(phase='search_running', source_epoch=EPOCH, coordinator=child.pid))
        assert child.wait() == 0, 'Sampler coordinator failed'
        write(OUT / 'queue-status.json', dict(phase='completed', source_epoch=EPOCH))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        write(OUT / 'queue-status.json', dict(phase='failed', error=str(error)))
        raise
