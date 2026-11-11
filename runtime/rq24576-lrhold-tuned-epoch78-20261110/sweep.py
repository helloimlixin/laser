"""Search epoch94 samplers, log class images, and validate joint FID/IS improvement."""
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
LOCAL = Path('/tmp/laser-rq24576-lrhold-tuned-epoch78-20261110')


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
    image_logger = None

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
            assert result['reference_sha256']==plan['reference_sha256'] and result['sampling']==plan['policies'][name]
            assert result['seed']==seed and result['generated_images']==plan[f'{stage}_images']
            log({'cached/policy':name,'cached/fid50k':result['fid'],'cached/inception_score':result['inception_score']})
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
            image_logger.ready()
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
        for name in ['worker.py', 'sweep.py', 'image_log.py']:
            shutil.copyfile(HERE/name, output/name)
        persist('memory-preflight.json', json.loads(read(LOCAL/'memory-preflight.json')))
        persist('evaluator-sha256.json', {name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
            for name in ['worker.py', 'sweep.py', 'image_log.py']})
        try:
            import wandb
            os.environ['WANDB_API_KEY'] = Path('/root/.config/laser/wandb.key').read_text().strip()
            (LOCAL/'wandb').mkdir(exist_ok=True)
            run = wandb.init(entity='helloimlixin-rutgers', project='laser', id=output.name,
                name=f'RQ24576 LR trial epoch78 | three strong samplers + images',
                job_type='sampling-config-search-with-images', resume='allow', dir=str(LOCAL/'wandb'),
                config=plan, settings=wandb.Settings(init_timeout=45))
            persist('wandb.json', dict(id=run.id, url=run.url))
            run.use_artifact('helloimlixin-rutgers/laser/imagenet-native-rq24576-tokenizer-20261108:v0')
            for name in ['plan.json', 'worker.py', 'sweep.py', 'image_log.py', 'memory-preflight.json']:
                run.save(str(output/name), base_path=str(output), policy='now')
        except Exception as error:
            persist('wandb-error.json', dict(error_type=type(error).__name__, error=str(error)))
            raise RuntimeError('W&B initialization failed; image logging is required') from error
        from image_log import ImageLogger
        image_logger = ImageLogger(plan,run)
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
        primary_names=[name+'__s261001' for name in plan['primary_policies']]
        persist('primary-selection.json',dict(policies=primary_names,checkpoint_sha256=plan['checkpoint_sha256']))
        persist('replicate-selection.json',dict(policies=[],checkpoint_sha256=plan['checkpoint_sha256']))
        results=wait_for('confirm',primary_names,'full50k_confirmation')
        status('verifying_frozen_weights')
        for rank,child in enumerate(children):
            assert child.wait(timeout=120)==0
            for stem in ['verification','adapter-verification']:
                proof=json.loads(read(LOCAL/f'{stem}-rank-{rank}.json'));assert proof['passed']
                persist(f'{stem}-rank-{rank}.json',proof)
        image_logger.table([],results)
        assert len(image_logger.records)==len(primary_names)
        best_fid=min(results,key=lambda r:r['fid'])
        best_is=max(results,key=lambda r:r['inception_score'])
        report=dict(plan=plan,results=results,best_fid=best_fid,best_inception_score=best_is,
            routine_baseline=plan['logged_baseline'],frozen_checkpoint_verified=True,
            weights_unmodified=True,tokenizer_unmodified=True,training_continues=True,
            generated_images_per_setting=50000,reference_images=1281167,reference_split='train',
            image_logging=json.loads(read(LOCAL/'image-logging.json')),
            fresh_seed_validation_complete=False,
            limits='One fixed epoch78 checkpoint and one matched generation seed for three prespecified samplers. IS split standard deviation is not between-seed uncertainty. Routine baseline uses its historical IS shuffle.')
        persist('report.json',report)
        parent=Path(plan['source_training_output'])
        write(parent/'sampler-evaluations/epoch078.json',report)
        write(parent/'tuned-sampler-evaluation-latest.json',dict(epoch=78,report=str(output/'report.json'),
            url=run.url,best_fid=best_fid,best_inception_score=best_is))
        run.summary.update(dict(best_fid50k=best_fid['fid'],best_fid_policy=best_fid['base_policy'],
            corresponding_inception_score=best_fid['inception_score'],corresponding_is_split_std=best_fid['inception_score_std'],
            highest_inception_score=best_is['inception_score'],images_logged=True,training_recipe_unchanged=True))
        for name in ['report.json','image-logging.json']:
            run.save(str(output/name),base_path=str(output),policy='now')
        run.finish();run=None
        uploaded=wandb.Api(timeout=45).run('helloimlixin-rutgers/laser/'+output.name)
        media=[f.name for f in uploaded.files() if f.name.startswith('media/images/')]
        assert len(media)>=len(primary_names)
        persist('wandb-media-verification.json',dict(passed=True,uploaded_image_files=len(media),
            expected_panels=len(primary_names),run_url=uploaded.url))
        status('completed',best_fid=best_fid['fid'],corresponding_inception_score=best_fid['inception_score'],
            best_fid_policy=best_fid['base_policy'])
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
