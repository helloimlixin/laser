"""Refine the epoch80 sampler and audit independent generation seeds."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
LOCAL = Path('/tmp/laser-rq24576-sampler-refine-epoch80-20261109')


def read(path):
    return Path(path).read_text()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.pending')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def moment_mmd(total, cross, fourth, count, reference_second):
    """Generated-sample U statistic for k(x,y)=(1+x.y/d)^2; reference moments fixed."""
    import numpy as np
    dimension = len(total)
    linear = 2*(total@total-np.trace(cross))/(dimension*count*(count-1))
    quadratic = (np.square(cross).sum()-fourth)/(dimension**2*count*(count-1))
    target = np.square(reference_second).sum()/dimension**2
    mixed = -2*np.sum(cross*reference_second)/(dimension**2*count)
    return float(linear+quadratic+target+mixed)


def main():
    lock = (LOCAL/'sweep.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads(read(LOCAL/'plan.json'))
    base, output = Path(plan['base']), Path(plan['output'])
    assert json.loads(read(LOCAL/'memory-preflight.json'))['passed']
    os.environ.update(RQ_UPSTREAM_ROOT=str(base/'upstream'), TORCH_HOME=str(base/'torch-cache'),
        OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4', WANDB_BASE_URL='https://api.wandb.ai',
        WANDB_DISABLE_GIT='true', WANDB_DIR=str(LOCAL/'wandb'), WANDB_CACHE_DIR=str(LOCAL/'wandb-cache'))
    sys.path[:0] = [str(base/'source'), str(base), str(base/'upstream')]
    import numpy as np
    import torch
    from src import rqvae_metrics as metrics
    torch.set_num_threads(4)
    reference = base/'inputs/imagenet_256_train.npz'
    assert hashlib.sha256(reference.read_bytes()).hexdigest() == plan['reference_sha256']
    mean, covariance = metrics.load_reference_statistics(reference)
    std = np.sqrt(np.maximum(np.diag(covariance), 1e-8))
    reference_second = covariance/np.outer(std, std)*(plan['reference_images']-1)/plan['reference_images']
    children, logs = [], []
    run = None

    def persist(name, value):
        write(LOCAL/name, value)
        write(output/name, value)

    def status(phase, **values):
        progress = []
        for rank in range(plan['world_size']):
            path = LOCAL/f'worker-{rank}.json'
            if path.exists():
                progress.append(json.loads(read(path)))
        persist('status.json', dict(phase=phase, pid=os.getpid(), time=time.time(), workers=progress, **values))

    def log(values):
        if run is not None:
            run.log(values)

    def aggregate(stage, name):
        seed = int(plan.get('policy_seeds', {}).get(name, plan[f'{stage}_seed']))
        destination = LOCAL/stage/(name+'.json')
        if destination.exists():
            result = json.loads(read(destination))
            assert result['checkpoint_sha256'] == plan['checkpoint_sha256']
            return result
        count, fourth = 0, 0.
        total, cross = np.zeros(2048), np.zeros((2048, 2048))
        probabilities, receipts = [], []
        for rank in range(plan['world_size']):
            path = LOCAL/stage/name/f'rank-{rank}.npz'
            receipt = json.loads(read(path.with_suffix('.json')))
            assert receipt['checkpoint_sha256'] == plan['checkpoint_sha256']
            assert receipt['seed'] == seed and receipt['sampling'] == plan['policies'][name]
            with np.load(path) as stats:
                total += stats['total']
                cross += stats['cross']
                fourth += float(stats['fourth'])
                count += int(stats['count'])
                probabilities.append(torch.from_numpy(stats['probabilities'].copy()))
            receipts.append(receipt)
        assert count == plan[f'{stage}_images']
        probs = torch.cat(probabilities)
        assert len(probs) == count
        permutation = torch.randperm(count, generator=torch.Generator().manual_seed(plan['is_shuffle_seed']))
        is_mean, is_std = metrics.inception_score(probs[permutation])
        result = dict(policy=name, sampling=plan['policies'][name], source_step=plan['source_step'],
            source_epoch=plan['source_epoch'], checkpoint_sha256=plan['checkpoint_sha256'],
            generated_images=count, reference_images=plan['reference_images'], reference_split='train',
            reference_sha256=plan['reference_sha256'], seed=seed, base_policy=plan.get('case_base_policy', {}).get(name, name),
            inception_score=is_mean, inception_score_std=is_std, world_size=plan['world_size'],
            sampling_batch_per_rank=plan['batch'], weights_frozen=True, rejection_sampling=False,
            initial_cuda_rng_sha256_by_rank=[r['initial_cuda_rng_sha256'] for r in receipts],
            max_worker_peak_gib=max(r['peak_allocated_gib'] for r in receipts),
            max_worker_seconds=max(r['seconds'] for r in receipts))
        if stage == 'screen':
            normalized_total = (total-count*mean)/std
            normalized_cross = (cross-np.outer(total, mean)-np.outer(mean, total)+count*np.outer(mean, mean))/np.outer(std, std)
            result.update(not_fid=True, moment_mmd=moment_mmd(normalized_total, normalized_cross, fourth, count, reference_second))
        else:
            generated_mean = total/count
            generated_cov = (cross-np.outer(total, total)/count)/(count-1)
            result['fid'] = metrics.frechet_distance(mean, covariance, generated_mean, generated_cov)
            assert np.isfinite(result['fid'])
        persist(f'{stage}/{name}.json', result)
        log({f'{stage}/policy':name, f'{stage}/inception_score':is_mean,
            f'{stage}/'+('moment_mmd' if stage == 'screen' else 'fid50k'):
                result['moment_mmd' if stage == 'screen' else 'fid']})
        print(json.dumps(dict(event='aggregated', stage=stage, **result)), flush=True)
        return result

    def wait_for(stage, names, phase_label=None):
        results = {}
        while len(results) < len(names):
            for rank, child in enumerate(children):
                code = child.poll()
                if code is not None and code != 0:
                    raise RuntimeError(f'Inference worker {rank} exited {code}; live training is independent')
            for name in names:
                if name not in results and all((LOCAL/stage/name/f'rank-{r}.json').exists() for r in range(plan['world_size'])):
                    status('computing_'+stage+'_metrics', policy=name)
                    results[name] = aggregate(stage, name)
            status(phase_label or stage, completed_policies=list(results), policies=len(names))
            if len(results) < len(names):
                time.sleep(10)
        return [results[name] for name in names]

    def interrupt(signum, _):
        raise InterruptedError(f'Sweep received signal {signum}')

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        output.mkdir(exist_ok=True)
        for name in ['worker.py', 'sweep.py']:
            shutil.copyfile(HERE/name, output/name)
        persist('memory-preflight.json', json.loads(read(LOCAL/'memory-preflight.json')))
        persist('evaluator-sha256.json', {name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
            for name in ['worker.py', 'sweep.py']})
        try:
            import wandb
            os.environ['WANDB_API_KEY'] = Path('/root/.config/laser/wandb.key').read_text().strip()
            (LOCAL/'wandb').mkdir(exist_ok=True)
            run = wandb.init(entity='helloimlixin-rutgers', project='laser', id=output.name,
                name=f'RQ24576 epoch80 | sampler refinement + three generation seeds',
                job_type='sampler-refinement-and-seed-audit', resume='allow', dir=str(LOCAL/'wandb'),
                config=plan, settings=wandb.Settings(init_timeout=45))
            persist('wandb.json', dict(id=run.id, url=run.url))
            run.use_artifact('helloimlixin-rutgers/laser/imagenet-native-rq24576-tokenizer-20261108:v0')
            for name in ['plan.json', 'worker.py', 'sweep.py', 'memory-preflight.json']:
                run.save(str(output/name), base_path=str(output), policy='now')
        except Exception as error:
            persist('wandb-error.json', dict(error_type=type(error).__name__, error=str(error)))
            run = None
        status('launching_inference_workers')
        for rank in range(plan['world_size']):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank), OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2')
            stream = (LOCAL/f'worker-{rank}.log').open('ab', buffering=0)
            child = subprocess.Popen([sys.executable, '-u', str(HERE/'worker.py'), '--plan', str(LOCAL/'plan.json'),
                '--rank', str(rank)], env=env, stdout=stream, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
            children.append(child)
            logs.append(stream)
        persist('worker-pids.json', dict(coordinator_pid=os.getpid(), workers=[p.pid for p in children]))
        screens = wait_for('screen', plan['screen_policies'], 'screen')
        challengers = [row for row in screens if row['policy'] != 'current']
        best_mmd = min(challengers, key=lambda row:(row['moment_mmd'],row['policy']))
        best_is = max([row for row in challengers if row['policy'] != best_mmd['policy']], key=lambda row:row['inception_score'])
        primary_names = [best_mmd['policy']+'__s261001', best_is['policy']+'__s261001']
        persist('screen-summary.json', dict(results=screens, not_fid=True, selected=primary_names, selection=plan['selection']))
        persist('primary-selection.json', dict(policies=primary_names,checkpoint_sha256=plan['checkpoint_sha256']))
        primary_results = wait_for('confirm', primary_names, 'primary_confirmation')
        chosen_primary = min(primary_results, key=lambda row:row['fid'])
        chosen = plan['case_base_policy'][chosen_primary['policy']]
        replicate_names = [name+'__s'+str(seed) for seed in plan['replication_seeds'] for name in ['current',chosen]]
        persist('primary-summary.json', dict(results=primary_results,chosen_challenger=chosen,
            current=json.loads(read(LOCAL/'confirm/current__s261001.json'))))
        persist('replicate-selection.json', dict(policies=replicate_names,checkpoint_sha256=plan['checkpoint_sha256']))
        fresh_results = wait_for('confirm', replicate_names, 'independent_seed_replication')
        cached_current = json.loads(read(LOCAL/'confirm/current__s261001.json'))
        confirmations = [cached_current]+primary_results+fresh_results
        status('verifying_frozen_weights')
        for rank, child in enumerate(children):
            assert child.wait(timeout=120) == 0
            proof = json.loads(read(LOCAL/f'verification-rank-{rank}.json'))
            assert proof['passed']
            persist(f'verification-rank-{rank}.json', proof)
        def summarize(name):
            rows = [row for row in confirmations if plan['case_base_policy'].get(row['policy'],row.get('base_policy')) == name]
            by_seed = {row['seed']:row for row in rows}
            assert sorted(by_seed) == sorted(plan['generation_seeds'])
            rows = [by_seed[seed] for seed in plan['generation_seeds']]
            fids = np.array([row['fid'] for row in rows])
            scores = np.array([row['inception_score'] for row in rows])
            fresh = [row for row in rows if row['seed'] in plan['replication_seeds']]
            return dict(policy=name,sampling=plan['policies'][name],results=rows,
                fid_mean=float(fids.mean()),fid_seed_std=float(fids.std(ddof=1)),
                inception_score_mean=float(scores.mean()),inception_score_seed_std=float(scores.std(ddof=1)),
                within_seed_is_split_stds=[row['inception_score_std'] for row in rows],
                fresh_fid_mean=float(np.mean([row['fid'] for row in fresh])),
                fresh_is_mean=float(np.mean([row['inception_score'] for row in fresh])))
        baseline_summary, challenger_summary = summarize('current'), summarize(chosen)
        baseline_rows = {row['seed']:row for row in baseline_summary['results']}
        challenger_rows = {row['seed']:row for row in challenger_summary['results']}
        paired = [dict(seed=seed,current_fid=baseline_rows[seed]['fid'],challenger_fid=challenger_rows[seed]['fid'],
            fid_improvement=baseline_rows[seed]['fid']-challenger_rows[seed]['fid']) for seed in plan['generation_seeds']]
        mean_improvement = baseline_summary['fid_mean']-challenger_summary['fid_mean']
        fresh_improvement = baseline_summary['fresh_fid_mean']-challenger_summary['fresh_fid_mean']
        rule = plan['recommendation_rule']
        improved = mean_improvement>=rule['minimum_mean_fid_improvement'] and fresh_improvement>=rule['minimum_fresh_mean_fid_improvement'] and all(row['fid_improvement']>0 for row in paired) and challenger_summary['fresh_is_mean']>=rule['minimum_mean_is_ratio']*baseline_summary['fresh_is_mean']
        selected = challenger_summary if improved else baseline_summary
        best_single = min(confirmations,key=lambda row:row['fid'])
        report = dict(plan=plan,screen=screens,primary_confirmation=primary_results,
            current=baseline_summary,challenger=challenger_summary,paired_seed_comparisons=paired,
            mean_fid_improvement=mean_improvement,fresh_mean_fid_improvement=fresh_improvement,
            improvement_confirmed_on_fresh_seeds=bool(improved),recommended_policy=selected['policy'],
            recommended_sampling=selected['sampling'],recommended_fid_mean=selected['fid_mean'],
            recommended_fid_seed_std=selected['fid_seed_std'],recommended_inception_score_mean=selected['inception_score_mean'],
            recommended_inception_score_seed_std=selected['inception_score_seed_std'],
            best_observed_single_seed=best_single,weights_unmodified=True,tokenizer_unmodified=True,
            active_training_recipe_unchanged=True,active_training_sampler_changed=False,training_seed_changed=False,
            limits='Three generation seeds, including the seed used to select the challenger. Two fresh seeds are reported separately and used to confirm improvement. Seed standard deviations are descriptive; no significance claim from three runs. Best single-seed score is exploratory.')
        persist('report.json', report)
        if run is not None:
            run.summary.update({key:report[key] for key in ['recommended_policy','recommended_sampling',
                'recommended_fid_mean','recommended_fid_seed_std','recommended_inception_score_mean',
                'recommended_inception_score_seed_std','mean_fid_improvement','fresh_mean_fid_improvement',
                'improvement_confirmed_on_fresh_seeds','active_training_sampler_changed']})
            for name in ['report.json','screen-summary.json','primary-summary.json']:
                run.save(str(output/name),base_path=str(output),policy='now')
            run.finish()
            run = None
        status('completed',selected_policy=selected['policy'],recommended_fid_mean=selected['fid_mean'],
            recommended_fid_seed_std=selected['fid_seed_std'],fresh_mean_fid_improvement=fresh_improvement)
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
        for stream in logs:
            stream.close()
        if run is not None:
            run.finish(exit_code=1)


if __name__ == '__main__':
    main()
