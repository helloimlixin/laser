"""Compare raw and EMA sampling at epoch 88, then confirm on fresh seeds."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = Path('/tmp/laser-rq24576-rewind60-teacher010-8h100-20261110')
TRAIN = ROOT / 'outputs/imagenet-rq24576-rewind60-teacher010-8h100-20261110'
RUN = 'imagenet-rq24576-noisematched-epoch88-sampler-20261111'
LOCAL = Path('/tmp') / RUN
OUT = ROOT / 'outputs' / RUN
STEP = 55088
REFERENCE_SHA = '3f9c92d15755e76ec312964a819e3e19b9cf3aadc618a7ae99f5e8aa96501260'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f'{path.name}.{os.getpid()}.pending')
    pending.write_text(json.dumps(value, indent=2) + '\n')
    pending.replace(path)


def main():
    LOCAL.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    lock = (LOCAL / 'study.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ.update(
        RQ_UPSTREAM_ROOT=str(BASE / 'upstream'), TORCH_HOME=str(BASE / 'torch-cache'),
        OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4', WANDB_BASE_URL='https://api.wandb.ai',
        WANDB_DISABLE_GIT='true', WANDB_DIR=str(LOCAL / 'wandb'),
        WANDB_CACHE_DIR=str(LOCAL / 'wandb-cache'),
        WANDB_API_KEY=Path('/root/.config/laser/wandb.key').read_text().strip(),
    )
    sys.path[:0] = [str(BASE / 'source'), str(BASE), str(BASE / 'upstream')]
    import numpy as np
    import torch
    import wandb
    from src import rqvae_metrics as metrics
    from image_log import ImageLogger

    torch.set_num_threads(4)
    checkpoint = LOCAL / 'frozen-checkpoint.pt'
    evaluation = read(TRAIN / 'evaluations/epoch088.json')
    receipt = read(TRAIN / 'epoch-checkpoints.json')[str(STEP)]
    assert evaluation['train/global_step'] == STEP
    assert evaluation['generated_images'] == 50000 and evaluation['real_images'] == 1281167
    assert evaluation['real_split'] == 'train' and not evaluation['rejection_sampling']
    assert evaluation['seed'] == 261001 and evaluation['is_shuffle_seed'] == 261002
    with checkpoint.open('rb') as stream:
        checkpoint_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert checkpoint_sha == receipt['sha256'] and checkpoint.stat().st_size == receipt['bytes']
    reference = BASE / 'inputs/imagenet_256_train.npz'
    assert hashlib.sha256(reference.read_bytes()).hexdigest() == REFERENCE_SHA
    mean, covariance = metrics.load_reference_statistics(reference)
    ema = evaluation['ema']
    policies = {
        'current': evaluation['sampler'],
        'cool_all_wide': {
            'temperature': [.875, .90, .925, .95],
            'top_k': [2048, 2560, 4096, 8192], 'top_p': [.95, .95, .98, .99],
        },
        'current_t0925': dict(evaluation['sampler'], temperature=.925),
    }
    classes = [
        dict(name=name, imagenet_id=index)
        for name, index in [('cheeseburger', 933), ('teapot', 849), ('samoyed', 258),
                            ('bald eagle', 22), ('red fox', 277), ('hotpot', 926),
                            ('wombat', 106), ('timber wolf', 269)]
    ]
    common = dict(
        base=str(BASE), source_run=TRAIN.name, source_epoch=88, source_step=STEP,
        checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
        checkpoint_bytes=checkpoint.stat().st_size, ema_decay=ema['ema_decay'],
        ema_updates=ema['ema_updates'], world_size=8, batch=64, decode_batch=8,
        memory_gib=12, reference_images=1281167, reference_split='train',
        reference_sha256=REFERENCE_SHA, confirm_images=50000, confirm_seed=261001,
        screen_images=0, screen_seed=26111188, screen_policies=[], is_shuffle_seed=261002,
        active_recipe_sha256=hashlib.sha256((BASE / 'inputs/recipe.yaml').read_bytes()).hexdigest(),
        adapter_probe=policies['cool_all_wide'], no_extra_hardware=True, training_continues=True,
        teacher_temperature=.085, rejection_sampling=False, sampling_precision='native FP16',
        decoder_precision='FP32', inception_precision='FP32',
        previews=dict(classes=classes, images_per_class=8, seed_offset=88000000,
                      primary_seed=261001, unfiltered=True, excluded_from_metric_images=True),
    )
    rows = []
    loggers = []
    children = []
    streams = []
    run = None

    def persist(name, value):
        write(LOCAL / name, value)
        write(OUT / name, value)

    def status(phase, plan=None, **fields):
        workers = []
        if plan:
            folder = Path(plan['local'])
            workers = [read(p) for p in sorted(folder.glob('worker-*.json'))]
        persist('status.json', dict(phase=phase, pid=os.getpid(), source_epoch=88,
            time=time.time(), workers=workers, completed_full50k=len(rows), **fields))

    def aggregate(plan, case):
        target = OUT / 'evaluations' / plan['weights'] / f'{case}.json'
        if target.exists():
            row = read(target)
            assert row['checkpoint_sha256'] == checkpoint_sha and row['sampling'] == plan['policies'][case]
            assert row['weights'] == plan['weights'] and row['generated_images'] == 50000
            return row
        total = np.zeros(2048)
        cross = np.zeros((2048, 2048))
        probabilities = []
        receipts = []
        count = 0
        for rank in range(8):
            path = Path(plan['local']) / 'confirm' / case / f'rank-{rank}.npz'
            proof = read(path.with_suffix('.json'))
            assert proof['checkpoint_sha256'] == checkpoint_sha
            assert proof['sampling'] == plan['policies'][case]
            assert proof['seed'] == plan['policy_seeds'][case] and proof['count'] == 6250
            with np.load(path) as data:
                total += data['total']
                cross += data['cross']
                count += int(data['count'])
                probabilities.append(torch.from_numpy(data['probabilities'].copy()))
            receipts.append(proof)
        assert count == 50000
        probabilities = torch.cat(probabilities)
        order = torch.randperm(count, generator=torch.Generator().manual_seed(261002))
        is_mean, is_std = metrics.inception_score(probabilities[order])
        fake_mean = total / count
        fake_covariance = (cross - np.outer(total, total) / count) / (count - 1)
        fid = metrics.frechet_distance(mean, covariance, fake_mean, fake_covariance)
        assert np.isfinite(fid)
        row = dict(
            policy=case, base_policy=plan['case_base_policy'][case], weights=plan['weights'],
            sampling=plan['policies'][case], source_epoch=88, source_step=STEP,
            checkpoint_sha256=checkpoint_sha, state_dict_key=plan['state_dict_key'],
            seed=plan['policy_seeds'][case], fid=fid, inception_score=is_mean,
            inception_score_std=is_std, is_shuffle_seed=261002,
            generated_images=count, reference_images=1281167, reference_split='train',
            reference_sha256=REFERENCE_SHA, world_size=8, sampling_batch_per_rank=64,
            rejection_sampling=False, weights_frozen=True,
            initial_cuda_rng_sha256_by_rank=[r['initial_cuda_rng_sha256'] for r in receipts],
        )
        write(target, row)
        run.log({
            'evaluation/weights': plan['weights'], 'evaluation/policy': row['base_policy'],
            'evaluation/seed': row['seed'], 'evaluation/fid50k': fid,
            'evaluation/inception_score50k': is_mean, 'evaluation/is_split_std': is_std,
        })
        print(json.dumps(dict(event='evaluation_complete', **row)), flush=True)
        return row

    def evaluate_round(weights, stage, cases):
        folder = LOCAL / weights / stage
        output = OUT / weights / stage
        folder.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True, exist_ok=True)
        settings = dict(policies)
        seeds = {}
        routes = {}
        names = []
        for policy, seed in cases:
            name = f'{policy}__s{seed}'
            names.append(name)
            settings[name] = policies[policy]
            seeds[name] = seed
            routes[name] = policy
        plan = dict(common, local=str(folder), output=str(output), weights=weights,
            state_dict_key='state_dict_raw' if weights == 'raw' else 'state_dict',
            policies=settings, policy_seeds=seeds, case_base_policy=routes,
            wandb_prefix=f'sampling/{weights}/{stage}')
        write(folder / 'plan.json', plan)
        write(output / 'plan.json', plan)
        write(folder / 'primary-selection.json', dict(policies=names, checkpoint_sha256=checkpoint_sha))
        write(folder / 'replicate-selection.json', dict(policies=[], checkpoint_sha256=checkpoint_sha))
        logger = ImageLogger(plan, run)
        loggers.append(logger)
        worker = [sys.executable, '-u', str(HERE / 'worker.py'), '--plan', str(folder / 'plan.json')]
        status('memory_preflight', plan, weights=weights, stage=stage)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2')
        with (folder / 'preflight.log').open('ab', buffering=0) as log:
            subprocess.run(worker + ['--rank', '0', '--preflight'], env=env,
                stdout=log, stderr=subprocess.STDOUT, check=True)
        proof = read(folder / 'preflight-rank-0.json')
        assert proof['passed'] and proof['peak_reserved_gib'] < 12 and proof['weights'] == weights
        write(output / 'memory-preflight.json', proof)
        for rank in range(8):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank), OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2')
            stream = (folder / f'worker-{rank}.log').open('ab', buffering=0)
            streams.append(stream)
            children.append(subprocess.Popen(worker + ['--rank', str(rank)], env=env,
                stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True))
        persist('worker-pids.json', dict(coordinator_pid=os.getpid(), weights=weights,
            stage=stage, workers=[child.pid for child in children]))
        completed = {}
        while len(completed) < len(names):
            logger.ready()
            for child in children:
                if child.poll() is not None and child.returncode != 0:
                    raise RuntimeError(f'Worker {child.pid} exited {child.returncode}')
            for case in names:
                receipts = [folder / 'confirm' / case / f'rank-{r}.json' for r in range(8)]
                if case not in completed and all(p.exists() for p in receipts):
                    status('computing_metrics', plan, weights=weights, stage=stage, policy=case)
                    row = aggregate(plan, case)
                    completed[case] = row
                    rows.append(row)
                    persist('results-so-far.json', dict(source_epoch=88, results=rows, run_url=run.url))
                    run.summary[f'{weights}/best_observed_fid50k'] = min(r['fid'] for r in rows if r['weights'] == weights)
                    run.summary[f'{weights}/highest_inception_score50k'] = max(r['inception_score'] for r in rows if r['weights'] == weights)
            status('sampling', plan, weights=weights, stage=stage, completed_cases=list(completed), cases=len(names))
            if len(completed) < len(names):
                time.sleep(5)
        for rank, child in enumerate(children):
            assert child.wait(timeout=60) == 0
            for stem in ['verification', 'adapter-verification']:
                proof = read(folder / f'{stem}-rank-{rank}.json')
                assert proof['passed']
                write(output / f'{stem}-rank-{rank}.json', proof)
        children.clear()
        for stream in streams:
            stream.close()
        streams.clear()
        logger.table([], list(completed.values()))
        return list(completed.values())

    def stop(sig, _):
        raise InterruptedError(f'Sampler study received signal {sig}')

    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, stop)
    try:
        (LOCAL / 'wandb').mkdir(exist_ok=True)
        config = dict(common, policies=policies, primary_weights=['ema', 'raw'],
            primary_seed=261001, fresh_seeds=[271001, 281001],
            source_evaluation=evaluation, canonical_receipt=receipt,
            earlier_epoch86_same_sampler=dict(fid=13.05785712569508, inception_score=100.89720153808594))
        persist('plan.json', config)
        run = wandb.init(entity='helloimlixin-rutgers', project='laser', id=RUN, resume='allow',
            name='RQ24576 epoch88 | raw and EMA | sampler search', job_type='sampling-config-search',
            config=config, dir=str(LOCAL / 'wandb'), settings=wandb.Settings(init_timeout=60))
        persist('wandb.json', dict(id=run.id, url=run.url))
        run.use_artifact('helloimlixin-rutgers/laser/imagenet-native-rq24576-tokenizer-20261108:v0')
        for name in ['study.py', 'worker.py', 'image_log.py']:
            shutil.copyfile(HERE / name, OUT / name)
            run.save(str(OUT / name), base_path=str(OUT), policy='now')
        persist('evaluator-sha256.json', {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
            for name in ['study.py', 'worker.py', 'image_log.py']})
        baseline = dict(
            policy='current__s261001', base_policy='current', weights='ema',
            sampling=policies['current'], source_epoch=88, source_step=STEP,
            checkpoint_sha256=checkpoint_sha, state_dict_key='state_dict', seed=261001,
            fid=ema['fid'], inception_score=ema['inception_score'], inception_score_std=ema['inception_score_std'],
            generated_images=50000, reference_images=1281167, reference_split='train',
            reference_sha256=REFERENCE_SHA, is_shuffle_seed=261002,
            reused_logged_evaluation=True, source_evaluation_path=str(TRAIN / 'evaluations/epoch088.json'),
        )
        rows.append(baseline)
        write(OUT / 'evaluations/ema/current__s261001.json', baseline)
        evaluate_round('ema', 'primary', [('cool_all_wide', 261001), ('current_t0925', 261001)])
        evaluate_round('raw', 'primary', [('current', 261001), ('cool_all_wide', 261001), ('current_t0925', 261001)])
        primary = list(rows)
        controls = {r['weights']: r for r in primary if r['base_policy'] == 'current'}
        candidates = [r for r in primary if r['base_policy'] != 'current']
        joint = [r for r in candidates if r['fid'] < controls[r['weights']]['fid']
                 and r['inception_score'] >= controls[r['weights']]['inception_score']]
        selected = min(joint or candidates, key=lambda r: r['fid'])
        persist('primary-summary.json', dict(results=primary, selected=selected))
        weights = selected['weights']
        policy = selected['base_policy']
        cases = [(name, seed) for seed in [271001, 281001] for name in ['current', policy]]
        fresh = evaluate_round(weights, 'fresh-seeds', cases)
        groups = {}
        for name in ['current', policy]:
            results = [r for r in rows if r['weights'] == weights and r['base_policy'] == name]
            assert sorted(r['seed'] for r in results) == [261001, 271001, 281001]
            groups[name] = dict(policy=name, weights=weights, sampling=policies[name], results=results,
                fid_mean=float(np.mean([r['fid'] for r in results])),
                fid_seed_std=float(np.std([r['fid'] for r in results], ddof=1)),
                inception_score_mean=float(np.mean([r['inception_score'] for r in results])),
                inception_score_seed_std=float(np.std([r['inception_score'] for r in results], ddof=1)))
        paired = []
        for seed in [261001, 271001, 281001]:
            control = next(r for r in groups['current']['results'] if r['seed'] == seed)
            candidate = next(r for r in groups[policy]['results'] if r['seed'] == seed)
            if seed != 261001 or weights == 'raw':
                assert control['initial_cuda_rng_sha256_by_rank'] == candidate['initial_cuda_rng_sha256_by_rank']
            paired.append(dict(seed=seed, fid_improvement=control['fid'] - candidate['fid'],
                is_improvement=candidate['inception_score'] - control['inception_score']))
        fid_gain = float(np.mean([r['fid_improvement'] for r in paired[1:]]))
        is_gain = float(np.mean([r['is_improvement'] for r in paired[1:]]))
        passed = (fid_gain > 0 and is_gain >= 0 and
            groups[policy]['fid_mean'] < groups['current']['fid_mean'] and
            groups[policy]['inception_score_mean'] >= groups['current']['inception_score_mean'])
        recommendation = groups[policy] if passed else groups['current']
        panels = sum(len(logger.records) for logger in loggers)
        report = dict(source_epoch=88, source_step=STEP, primary_results=primary, fresh_results=fresh,
            all_results=rows, selected_weights=weights, selected_policy=policy,
            current=groups['current'], challenger=groups[policy], paired=paired,
            fresh_mean_fid_improvement=fid_gain, fresh_mean_is_improvement=is_gain,
            mean_improvement_on_fresh_seeds=passed, independent_seed_validation_complete=True,
            recommended=recommendation, run_url=run.url, image_panels=panels,
            generated_metric_images=50000 * (len(rows) - 1), canonical_ema_baseline_reused=True,
            limits='Primary selection uses seed 261001. Two fresh generation seeds validate the selected sampler and weight type. IS split standard deviation and across-seed standard deviation measure different quantities.')
        persist('report.json', report)
        run.summary.update(dict(recommended_weights=weights, recommended_policy=recommendation['policy'],
            recommended_sampling=recommendation['sampling'], recommended_fid_mean=recommendation['fid_mean'],
            recommended_inception_score_mean=recommendation['inception_score_mean'],
            image_panel_count=panels, independent_seed_validation_complete=True))
        for name in ['plan.json', 'report.json', 'primary-summary.json', 'evaluator-sha256.json']:
            run.save(str(OUT / name), base_path=str(OUT), policy='now')
        run.finish()
        run = None
        media = []
        for attempt in range(6):
            online = wandb.Api(timeout=45).run(f'helloimlixin-rutgers/laser/{RUN}')
            media = [f.name for f in online.files() if f.name.startswith('media/images/')]
            if len(media) >= panels:
                break
            time.sleep(10)
        assert len(media) >= panels
        persist('wandb-media-verification.json', dict(passed=True, uploaded_image_files=len(media), expected_panels=panels))
        status('completed', recommended=recommendation)
    except BaseException as error:
        status('failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for stream in streams:
            stream.close()
        if run is not None:
            run.finish(exit_code=1)


if __name__ == '__main__':
    main()
