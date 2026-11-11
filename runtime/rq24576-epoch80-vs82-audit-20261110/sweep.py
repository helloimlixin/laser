"""Compare frozen epoch80 and82 checkpoints on three matched generation seeds."""
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
LOCAL = Path('/tmp/laser-rq24576-epoch80-vs82-audit-20261110')


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
    assert plan['confirmation_only'] and plan['source_epoch'] == 82
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
        seed = int(plan['policy_seeds'][name])
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
            reference_sha256=plan['reference_sha256'], seed=seed,
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

    def wait_for(stage, names):
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
            status(stage, completed_policies=list(results), policies=len(names))
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
        persist('prior-memory-preflight.json', dict(
            prior_epoch26_preflight=json.loads(read(Path('/tmp/laser-rq24576-sampler-sweep-20261109/memory-preflight.json'))),
            same_architecture_vocabulary_batch_and_memory_cap=True, current_weights_will_be_checked_by_all_eight_workers=True))
        persist('evaluator-sha256.json', {name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
            for name in ['worker.py', 'sweep.py']})
        try:
            import wandb
            os.environ['WANDB_API_KEY'] = Path('/root/.config/laser/wandb.key').read_text().strip()
            (LOCAL/'wandb').mkdir(exist_ok=True)
            run = wandb.init(entity='helloimlixin-rutgers', project='laser', id=output.name,
                name='RQ24576 | epoch80 vs82 | three matched generation seeds',
                job_type='checkpoint-quality-audit', resume='allow', dir=str(LOCAL/'wandb'),
                config=plan, settings=wandb.Settings(init_timeout=45))
            persist('wandb.json', dict(id=run.id, url=run.url))
            run.use_artifact('helloimlixin-rutgers/laser/imagenet-native-rq24576-tokenizer-20261108:v0')
            for name in ['plan.json', 'worker.py', 'sweep.py', 'prior-memory-preflight.json']:
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
        names = plan['new_cases']
        confirmations = wait_for('confirm', names)
        status('verifying_frozen_weights')
        for rank, child in enumerate(children):
            assert child.wait(timeout=120) == 0
            proof = json.loads(read(LOCAL/f'verification-rank-{rank}.json'))
            assert proof['passed']
            persist(f'verification-rank-{rank}.json', proof)
        audit80 = json.loads(read(Path(plan['epoch80_results'])))
        rows80 = audit80['challenger']['results']
        primary82 = json.loads(read(Path(plan['epoch82_primary_result'])))
        rows82 = [primary82]+confirmations
        assert sorted(row['seed'] for row in rows80)==sorted(plan['generation_seeds'])
        assert sorted(row['seed'] for row in rows82)==sorted(plan['generation_seeds'])
        assert all(row['sampling']==plan['policies'][names[0]] and row['generated_images']==50000 for row in rows80+rows82)
        def summarize(rows):
            rows = sorted(rows,key=lambda row:row['seed'])
            return dict(results=rows,fid_mean=float(np.mean([row['fid'] for row in rows])),
                fid_seed_std=float(np.std([row['fid'] for row in rows],ddof=1)),
                inception_score_mean=float(np.mean([row['inception_score'] for row in rows])),
                inception_score_seed_std=float(np.std([row['inception_score'] for row in rows],ddof=1)))
        earlier,later = summarize(rows80),summarize(rows82)
        old_by_seed = {row['seed']:row for row in rows80}
        paired = [dict(seed=row['seed'],epoch80_fid=old_by_seed[row['seed']]['fid'],epoch82_fid=row['fid'],
            fid_delta_epoch82_minus_epoch80=row['fid']-old_by_seed[row['seed']]['fid'],
            epoch80_is=old_by_seed[row['seed']]['inception_score'],epoch82_is=row['inception_score']) for row in later['results']]
        for row in confirmations:
            assert row['initial_cuda_rng_sha256_by_rank']==old_by_seed[row['seed']]['initial_cuda_rng_sha256_by_rank']
        training_audit = json.loads(read(LOCAL/'training-state-audit.json'))
        metrics_audit = json.loads(read(LOCAL/'metric-recalculation-audit.json'))
        assert training_audit['passed'] and metrics_audit['passed']
        report = dict(plan=plan,epoch80=earlier,epoch82=later,paired_seed_comparisons=paired,
            fid_delta_epoch82_minus_epoch80=later['fid_mean']-earlier['fid_mean'],
            fresh_seed_mean_fid_delta=float(np.mean([row['fid_delta_epoch82_minus_epoch80'] for row in paired if row['seed']!=261001])),
            best_mean_fid_epoch=80 if earlier['fid_mean']<=later['fid_mean'] else 82,
            training_state_audit=training_audit,metric_recalculation_audit=metrics_audit,
            sampler_identical=True,generation_seeds_matched=True,initial_cuda_rng_states_matched=True,
            weights_unmodified=True,tokenizer_unmodified=True,active_training_recipe_unchanged=True,
            active_training_sampler_changed=False,training_seed_changed=False,
            limits='Three paired generation seeds and one fixed sampler. The sampler was selected at epoch80; report both per-seed and mean scores. A later training checkpoint is not guaranteed to improve FID or IS.')
        persist('report.json', report)
        if run is not None:
            run.summary.update(dict(epoch80_fid_mean=earlier['fid_mean'],epoch80_fid_seed_std=earlier['fid_seed_std'],
                epoch82_fid_mean=later['fid_mean'],epoch82_fid_seed_std=later['fid_seed_std'],
                epoch80_is_mean=earlier['inception_score_mean'],epoch82_is_mean=later['inception_score_mean'],
                fid_delta_epoch82_minus_epoch80=report['fid_delta_epoch82_minus_epoch80'],
                fresh_seed_mean_fid_delta=report['fresh_seed_mean_fid_delta'],best_mean_fid_epoch=report['best_mean_fid_epoch']))
            for name in ['report.json','training-state-audit.json','metric-recalculation-audit.json']:
                run.save(str(output/name),base_path=str(output),policy='now')
            run.finish()
            run = None
        status('completed',epoch80_fid_mean=earlier['fid_mean'],epoch82_fid_mean=later['fid_mean'],
            best_mean_fid_epoch=report['best_mean_fid_epoch'])
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
