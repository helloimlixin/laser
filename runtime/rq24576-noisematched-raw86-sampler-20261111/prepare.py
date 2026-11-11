"""Search depth sampling settings on the epoch 86 raw checkpoint."""

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
RUN = 'imagenet-rq24576-noisematched-raw86-sampler-20261111'
LOCAL = Path('/tmp') / RUN
OUT = ROOT / 'outputs' / RUN
EPOCH = 86
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
    return evaluation['raw'], evaluation['ema'], frozen, digest, receipt


def build_plan():
    baseline, ema, checkpoint, digest, receipt = freeze_checkpoint()
    standard_k = [1536, 2048, 3072, 6144]
    wide_k = [2048, 2560, 4096, 8192]
    standard_p = [.95, .95, .98, .99]

    def sampler(temperature, top_k, top_p=standard_p):
        return dict(temperature=temperature, top_k=top_k, top_p=top_p)

    presets = {
        'current': baseline['sampler'],
        'depth_cool_front': sampler([.925, .925, .975, .975], wide_k),
        'wide_depth_ramp': sampler([.925, .95, .975, 1.0], wide_k),
        'wide_t090': sampler(.90, wide_k),
        'wide_t09375': sampler(.9375, wide_k),
        'current_t090': sampler(.90, standard_k),
        'current_t0875': sampler(.875, standard_k),
        'tight_late_p': sampler(.975, standard_k, [.95, .95, .95, .95]),
        'wide_t0925_p095': sampler(.925, wide_k, [.95, .95, .95, .95]),
        'coarse_k512': sampler([.90, .925, .975, .975], [512, 1536, 3072, 6144], [.93, .95, .98, .99]),
        'cool_coarse_warm_tail': sampler([.875, .925, 1.0, 1.025], wide_k),
        'cool_two_warm_tail': sampler([.90, .90, 1.0, 1.05], wide_k),
        'cool_all_wide': sampler([.875, .90, .925, .95], wide_k),
        'wider_cool': sampler([.875, .90, .95, .975], [3072, 4096, 6144, 12288]),
        'medium_cool': sampler([.90, .925, .95, .975], [1536, 2048, 4096, 8192]),
        'coarse_k1024_warm_tail': sampler([.90, .925, 1.0, 1.025], [1024, 2048, 4096, 8192]),
        'wide_late_p100': sampler([.90, .925, .975, 1.0], wide_k, [.95, .95, 1.0, 1.0]),
        'wide_p098': sampler([.90, .925, .975, 1.0], wide_k, [.98, .98, .98, .99]),
        'current_t0925': sampler(.925, standard_k),
        'current_t095': sampler(.95, standard_k),
        'wide_t095_p095': sampler(.95, wide_k, [.95, .95, .95, .95]),
    }
    assert presets['current'] == sampler(.975, standard_k)
    cases = dict(presets)
    screen_seed = 26111186
    seeds = {name: screen_seed for name in presets}
    routes = {name: name for name in presets}
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
        weights='raw', state_dict_key='state_dict_raw', ema_decay=ema['ema_decay'],
        ema_updates=ema['ema_updates'], policies=cases, policy_seeds=seeds,
        case_base_policy=routes, screen_policies=list(presets)[:18], screen_images=10000, screen_seed=screen_seed,
        additional_primary_policies=['current_t090', 'cool_all_wide', 'current_t0925', 'current_t095', 'wide_t095_p095'],
        confirm_images=50000,
        confirm_seed=SEEDS[0], is_shuffle_seed=261002, generation_seeds=SEEDS,
        replication_seeds=SEEDS[1:], world_size=8, batch=64, decode_batch=8, memory_gib=12,
        study_kind='FID and Inception Score', teacher_temperature_at_checkpoint=.085,
        reference_images=1281167, reference_split='train',
        reference_sha256='3f9c92d15755e76ec312964a819e3e19b9cf3aadc618a7ae99f5e8aa96501260',
        active_recipe_sha256=hashlib.sha256((BASE / 'inputs/recipe.yaml').read_bytes()).hexdigest(),
        logged_baseline=baseline, training_continues=True, no_extra_hardware=True,
        inference_sampler_precision='native FP16', decoder_precision='FP32',
        inception_precision='FP32', rejection_sampling=False,
        adapter_probe=presets['depth_cool_front'], published_target_fid=11.56,
        selection='Confirm the control, the two lowest screen FIDs, minimum moment MMD, highest IS at no higher screen FID, and highest overall screen IS. Replicate minimum FID and highest IS at no higher full50k FID.',
        recommendation_rule='Prefer lower mean FID with no IS loss on both the two fresh seeds averaged and all three seeds averaged.',
        earlier_epoch86_same_sampler=dict(fid=13.05785712569508, inception_score=100.89720153808594),
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
        write(OUT / 'queue-status.json', dict(phase='freezing_epoch86_raw_checkpoint'))
        plan = build_plan()
        for name in ['primary-selection.json', 'replicate-selection.json']:
            (LOCAL / name).unlink(missing_ok=True)
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
