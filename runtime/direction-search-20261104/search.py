"""Bounded, paired full-FID search from the immutable epoch-73 best checkpoint."""
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from contextlib import nullcontext
from datetime import timedelta

BASE = Path('/tmp/laser-direction-search-20261104/frozen')
RUN = 'imnet-j16384-best14574-direction-search-1b300-20261104'
OUT = Path('/workspace/Projects/laser/outputs') / RUN
LOCAL = BASE / 'run-output'
SOURCE_STEP = 45698
SOURCE_EPOCH = 73
UPDATES = 8
TRAIN_SEED = 261001
PRIMARY_SEED = 261002
CONFIRM_SEED = 261001
SCREEN_IMAGES = 2000
SCREEN_BATCH = 256
HISTORICAL_FID = 14.573915136162327
SOURCE_ARTIFACT = 'helloimlixin-rutgers/laser/imnet-j16384-a4096-fid1481-resume100-4h100-20261104-checkpoint:v26'
TRIALS = [
    dict(name='adam_small', mode='adam', factor=0.00375),
    dict(name='no_momentum_small', mode='no_momentum', factor=0.00375),
    dict(name='gradient_small', mode='gradient', factor=0.00375),
    dict(name='reverse_adam_small', mode='reverse_adam', factor=0.00375),
]
for key, value in dict(RQ_UPSTREAM_ROOT=str(BASE / 'upstream'), TORCH_HOME=str(BASE / 'torch-cache'),
        WANDB_BASE_URL='https://api.forge.coreweave.com', WANDB_DISABLE_GIT='true',
        WANDB_CACHE_DIR=str(BASE / 'wandb-cache'), LASER_STAGE2_OUTPUT=str(LOCAL),
        LASER_FUSED_ADAMW='1', LASER_TRAIN_SDPA='0', LASER_COMPILE_NATIVE_LOSS='1',
        LASER_MAX_RECOVERY_LOSS='8.25', TORCHINDUCTOR_COMPILE_THREADS='4').items():
    os.environ[key] = value
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'source'))

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from runtime.precision import install_bf16_training
from runtime.training_acceleration import install, configure_optimizer
install_bf16_training()
install()
from runtime.search_directions import configure as configure_direction, step as direction_step, self_test
from src.training import rq_imagenet_shared as shared
from src.training.laser_joint_recipe import JointLaserRecipe
from src.training.rq_imagenet_runtime import TarImageNet, cpu_snapshot, sha256
from src.training.exact_global_batch import ExactGlobalBatchSampler
from rqvae.img_datasets.transforms import create_transforms
from rqvae.optimizer.scheduler import create_scheduler
from runtime.lr_recovery import apply_lr_amplitude


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def record(name, value):
    for directory in [LOCAL, OUT]:
        atomic(directory / name, value)


def stage(phase, **extra):
    record('stage.json', dict(phase=phase, time=time.time(), worker_pid=os.getpid(), **extra))


def stopped():
    if (OUT / 'STOP').exists() or (LOCAL / 'STOP').exists():
        raise KeyboardInterrupt('Checkpoint-preserving search stop requested')


class Engine:
    def __init__(self):
        self.device = torch.device('cuda', 0)
        torch.cuda.set_device(self.device)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        torch.backends.cudnn.allow_tf32 = True
        dist.init_process_group('nccl', timeout=timedelta(minutes=90), device_id=self.device)
        shared.seed_all(TRAIN_SEED)
        self.profile = JointLaserRecipe(BASE)
        self.profile.generation_batch = SCREEN_BATCH
        shared.BASE, shared.OUT, shared.RUN, shared.PROFILE = BASE, LOCAL, RUN, self.profile
        shared.original_metrics.INCEPTION_PATH = BASE / 'upstream/rqvae/metrics/inception.py'
        self.aux, _ = self.profile.tokenizer(self.device)
        self.tokenizer_digest = self.profile.digest(self.aux)
        self.model, self.optimizer, self.config = shared.fresh_model(self.device)
        self.ddp = DDP(self.model, device_ids=[0], broadcast_buffers=False,
                       gradient_as_bucket_view=True, bucket_cap_mb=100)
        self.payload = torch.load(BASE / 'inputs/resume-checkpoint.pt', map_location='cpu',
                                  weights_only=False, mmap=True)
        self.profile.verify_resume(self.payload)
        assert self.payload['global_step'] == SOURCE_STEP
        assert self.payload['next_epoch'] == SOURCE_EPOCH and self.payload['next_microbatch'] == 0
        assert self.payload['stage1_checkpoint_sha256'] == sha256(BASE / 'inputs/stage1-model.pt')
        assert self.payload['runtime']['codebook_sha256'] == self.profile.codebook_sha256
        self.dataset = TarImageNet(BASE / 'data', 'train',
            create_transforms(self.config.dataset, split='train'), seed=TRAIN_SEED)
        assert len(self.dataset) == 1281167
        self.sampler = ExactGlobalBatchSampler(self.dataset, 2048, 1, 0, 8, TRAIN_SEED)
        self.loader = DataLoader(self.dataset, batch_sampler=self.sampler, num_workers=12,
            pin_memory=True, persistent_workers=True, prefetch_factor=1, multiprocessing_context='spawn')
        self.scaler = torch.amp.GradScaler('cuda', enabled=False)
        self.global_step = SOURCE_STEP
        self.current = None

    def restore(self, spec):
        self.model.load_state_dict(self.payload['state_dict'], strict=True)
        self.optimizer = None
        gc.collect()
        self.optimizer = shared.create_resnet_optimizer(self.model, self.config.optimizer)
        self.scheduler = create_scheduler(self.optimizer, self.config.optimizer.warmup, 626, 100)
        self.optimizer.load_state_dict(self.payload['optimizer'])
        configure_optimizer(self.optimizer)
        self.scheduler.load_state_dict(self.payload['scheduler'])
        self.lr_recovery = apply_lr_amplitude(self.optimizer, self.scheduler,
            previous_factor=float(self.payload['runtime']['lr_amplitude_factor']),
            desired_factor=spec['factor'])
        configure_direction(self.optimizer, spec['mode'])
        assert len(self.optimizer.state) == 780
        assert {int(x['step']) for x in self.optimizer.state.values()} == {SOURCE_STEP}
        assert self.scheduler.after_scheduler.last_epoch == SOURCE_STEP
        assert self.scheduler.after_scheduler.T_max == 62600
        saved = self.payload['rng_state_by_rank'][0]
        random.setstate(saved['python'])
        np.random.set_state(saved['numpy'])
        torch.set_rng_state(saved['cpu'])
        torch.cuda.set_rng_state(saved['cuda'], self.device)
        self.global_step = SOURCE_STEP
        self.current = dict(spec)
        self.optimizer.zero_grad(set_to_none=True)
        self.sampler.set_epoch(SOURCE_EPOCH)
        self.sampler.set_start_batch(0)

    def train(self, updates, phase):
        self.model.train()
        iterator = iter(self.loader)
        first_batch_hash = None
        first_labels = None
        last = None
        for update in range(updates):
            stopped()
            started = time.monotonic()
            total_loss = 0.0
            for micro in range(8):
                images, labels = next(iterator)
                if update == micro == 0:
                    first_batch_hash = hashlib.sha256(images.numpy().tobytes()).hexdigest()
                    first_labels = labels[:16].tolist()
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                with self.ddp.no_sync() if micro < 7 else nullcontext():
                    loss = shared.training_backward(self.profile, self.ddp, self.aux, images,
                        labels, self.scaler, 1 / 8, amp=True)
                total_loss += float(loss)
                del images, labels, loss
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1, error_if_nonfinite=True)
            direction = direction_step(self.optimizer, self.current['mode'])
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1
            assert int(next(iter(self.optimizer.state.values()))['step']) == self.global_step
            last = dict(candidate=self.current['name'], phase=phase, updates=update + 1,
                global_step=self.global_step, adam_age=self.global_step, loss=total_loss / 8,
                learning_rate=float(self.optimizer.param_groups[0]['lr']),
                unclipped_gradient_norm=float(norm), **direction,
                images_per_second=2048 / (time.monotonic() - started), time=time.time())
            record('training-progress.json', last)
            print(json.dumps(last), flush=True)
        assert {int(x['step']) for x in self.optimizer.state.values()} == {self.global_step}
        return dict(last_update=last, first_batch_sha256=first_batch_hash, first_labels=first_labels,
                    source_rng_restored=True, global_batch=2048, accumulation=8, microbatch=256)

    def evaluate(self, name, seed, count=SCREEN_IMAGES, batch=SCREEN_BATCH):
        stopped()
        os.environ['LASER_EVAL_IMAGES'] = str(count)
        stage('evaluating', candidate=name, seed=seed, generated_images=count, generation_batch=batch,
              optimizer_updates_frozen=True, global_step=self.global_step)
        result = shared.generation(self.model, self.aux, self.device, 0, 1, SOURCE_EPOCH,
                                   None, batch=batch, seed=seed)
        assert all(math.isfinite(result[k]) for k in ['fid', 'inception_score', 'inception_score_std'])
        assert self.profile.digest(self.aux) == self.tokenizer_digest
        return dict(result, seed=seed, generated_images=count, generation_batch=batch, hardware='B300',
            reference_split='train', reference_images=1281167,
            reference_sha256=sha256(BASE / 'inputs/imagenet_256_train.npz'),
            global_step=self.global_step, remeasured=True)

    def save(self, name, evaluation, proof):
        stage('saving_checkpoint', candidate=name, global_step=self.global_step)
        folder = LOCAL / 'checkpoints'
        folder.mkdir(exist_ok=True)
        path = folder / (name + '.pt')
        runtime = dict(self.payload['runtime'], world_size=1, microbatch=256,
            accumulation_steps=8, global_batch=2048, lr_amplitude_factor=self.current['factor'],
            direction_search=dict(self.current, requires_direction_search_driver=True))
        state = cpu_snapshot(dict(run_id=RUN, state_dict=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(), scheduler=self.scheduler.state_dict(),
            scaler=self.scaler.state_dict(), epoch=SOURCE_EPOCH, global_step=self.global_step,
            next_epoch=SOURCE_EPOCH, next_microbatch=(self.global_step - SOURCE_STEP) * 8,
            rng_state_by_rank=[dict(python=random.getstate(), numpy=np.random.get_state(),
                cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(self.device))],
            config=OmegaConf.to_container(self.config, resolve=True), runtime=runtime,
            native_sampling=self.profile.sampling_settings(),
            stage1_checkpoint_sha256=self.payload['stage1_checkpoint_sha256'],
            **self.profile.checkpoint_extra(), evaluation=evaluation,
            last_full_evaluation=evaluation, best_fid=evaluation['fid'] if evaluation else HISTORICAL_FID,
            comparison_baseline=read(LOCAL / 'baseline-primary.json'),
            preservation_floor=0.0, direction_search=dict(self.current, training_proof=proof,
                requires_direction_search_driver=True, source_artifact=SOURCE_ARTIFACT)))
        temporary = path.with_suffix('.partial')
        torch.save(state, temporary)
        temporary.replace(path)
        del state
        gc.collect()
        check = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        assert check['global_step'] == self.global_step
        assert len(check['optimizer']['state']) == 780
        assert {int(s['step']) for s in check['optimizer']['state'].values()} == {self.global_step}
        assert check['scheduler']['after']['last_epoch'] == self.global_step
        assert check['next_microbatch'] == (self.global_step - SOURCE_STEP) * 8
        assert check['direction_search']['mode'] == self.current['mode']
        del check
        return dict(path=str(path), bytes=path.stat().st_size, global_step=self.global_step,
                    complete_model_optimizer_scheduler_rng_and_cursor=True,
                    direction=self.current, requires_direction_search_driver=True)

    def load_candidate(self, receipt):
        path = Path(receipt['path'])
        assert path.stat().st_size == receipt['bytes']
        state = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        self.model.load_state_dict(state['state_dict'], strict=True)
        self.global_step = state['global_step']
        self.current = receipt['direction']
        self.optimizer.load_state_dict(state['optimizer'])
        configure_optimizer(self.optimizer)
        self.scheduler.load_state_dict(state['scheduler'])
        self.scaler.load_state_dict(state['scaler'])
        rng = state['rng_state_by_rank'][0]
        random.setstate(rng['python']); np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['cpu']); torch.cuda.set_rng_state(rng['cuda'], self.device)
        del rng
        del state
        gc.collect()


def publish(run, result, final=False):
    import wandb
    receipt = result['checkpoint']
    artifact = wandb.Artifact(RUN + ('-best' if final else '-candidates'), type='model',
        metadata=dict(candidate=result['candidate'], evaluation=result['evaluation'],
            confirmation=result.get('confirmation'), direction=result['spec'],
            source_artifact=SOURCE_ARTIFACT, source_step=SOURCE_STEP,
            requires_direction_search_driver=True))
    artifact.add_file(receipt['path'], name='best.pt' if final else 'candidate.pt',
                      policy='immutable', skip_cache=True)
    aliases = ['best', 'best-confirmed'] if final else [result['candidate'], 'provisional-best']
    committed = run.log_artifact(artifact, aliases=aliases)
    committed.wait()
    assert str(committed.state) == 'COMMITTED'
    return dict(receipt, artifact=committed.qualified_name, state='COMMITTED',
                fid=result['evaluation']['fid'], confirmation=result.get('confirmation'))


def main():
    import wandb
    LOCAL.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if (LOCAL / 'completed.json').exists():
        print('Search already completed; existing best checkpoint retained', flush=True)
        return
    os.environ['WANDB_API_KEY'] = Path('/root/.config/laser/wandb.key').read_text().strip()
    plan = dict(run_id=RUN, protocol='fast-screen-v2', source_step=SOURCE_STEP,
        source_epoch=SOURCE_EPOCH, source_artifact=SOURCE_ARTIFACT, historical_fid=HISTORICAL_FID,
        updates_per_trial=UPDATES, primary_seed=PRIMARY_SEED, confirmation_seed=CONFIRM_SEED,
        trials=TRIALS, screen_images=SCREEN_IMAGES, screen_batch=SCREEN_BATCH,
        confirmation_images=50000, confirmation_batch=64, confirmation_candidates_maximum=1,
        promotion='Beat paired source FID2k screen, then beat historical14.5739 on winner-only FID50k using original seed and batch64',
        full50k_baseline_is_inherited_from_b200=True, sampling_unchanged=True,
        tokenizer_frozen=True, classifier_guidance=False, automatic_long_training=False,
        maximum_full50k_evaluations=1)
    record('plan.json', plan)
    stage('initializing')
    engine = Engine()
    preflight = read(LOCAL / 'preflight.json')
    assert preflight and preflight['passed'] and preflight['same_first_batch_and_source_rng']
    engine.restore(TRIALS[0])
    assert engine.global_step == SOURCE_STEP
    run = wandb.init(entity='helloimlixin-rutgers', project='laser', id=RUN,
        name='ImageNet joint16384 | fast direction search | 2k screen then winner50k',
        resume='allow', dir=str(BASE / 'wandb'))
    run.config.update(dict(plan, training_recipe=engine.profile.checkpoint_extra(),
        native_sampling=engine.profile.sampling_settings(), hardware='1 B300'), allow_val_change=True)
    run.define_metric('search/trial')
    run.define_metric('screen/*', step_metric='search/trial')
    run.define_metric('confirmation/*', step_metric='search/trial')
    run.summary.update({'search/status': 'screening', 'search/source_best_fid50k': HISTORICAL_FID,
                        'search/automatic_long_training': False})
    record('wandb.json', dict(id=RUN, url=run.url))
    if not (LOCAL / 'source-artifact.json').exists():
        from src.training import joint_runtime_support
        joint_runtime_support.publish_source(run, BASE, LOCAL)
    baseline = read(LOCAL / 'baseline-screen-2k.json')
    if baseline is None:
        baseline = engine.evaluate('source_best_screen_2k', PRIMARY_SEED)
        record('baseline-screen-2k.json', baseline)
        run.log({'search/trial': 0, 'search/candidate': 'source_best_screen_2k',
                 'screen/fid_2k': baseline['fid'], 'screen/inception_score_2k': baseline['inception_score']})
    record('best-checkpoint.json', dict(artifact=SOURCE_ARTIFACT, state='COMMITTED',
        path=str(BASE / 'inputs/resume-checkpoint.pt'), global_step=SOURCE_STEP,
        fid50k=HISTORICAL_FID, fid2k=baseline['fid'], historical_fid=HISTORICAL_FID,
        full50k_measurement_hardware='B200', source_retained=True))
    results = []
    best_screen = baseline['fid']
    best_result = None
    for index, spec in enumerate(TRIALS, 1):
        stopped()
        result = read(LOCAL / 'screen-results' / (spec['name'] + '.json'))
        if result is None:
            stage('training_trial', candidate=spec['name'], trial=index, total_trials=len(TRIALS))
            engine.restore(spec)
            proof = engine.train(UPDATES, 'trial')
            evaluation = engine.evaluate(spec['name'] + '_screen_2k', PRIMARY_SEED)
            result = dict(candidate=spec['name'], spec=spec, evaluation=evaluation,
                delta_vs_baseline=evaluation['fid'] - baseline['fid'],
                source_step=SOURCE_STEP, updates=UPDATES, training_proof=proof,
                primary_improved=evaluation['fid'] < baseline['fid'],
                screen_only=True, full_fid50k_confirmed=False)
            if evaluation['fid'] < best_screen:
                result['checkpoint'] = engine.save(spec['name'], evaluation, proof)
            record('screen-results/' + spec['name'] + '.json', result)
            run.log({'search/trial': index, 'search/candidate': spec['name'],
                'screen/fid_2k': evaluation['fid'], 'screen/inception_score_2k': evaluation['inception_score'],
                'screen/fid_delta_vs_source': result['delta_vs_baseline'],
                'screen/best_so_far_2k': min(best_screen, evaluation['fid'])})
        results.append(result)
        if result['evaluation']['fid'] < best_screen:
            assert result.get('checkpoint'), 'Improving candidate must have a complete checkpoint'
            best_result = result
            best_screen = result['evaluation']['fid']
            record('screen-winner.json', result)
        record('ranking.json', sorted(results, key=lambda r: r['evaluation']['fid']))
    winner = None
    if best_result:
        result = best_result
        provisional = read(LOCAL / 'provisional-best.json')
        if provisional is None or provisional['candidate'] != result['candidate']:
            receipt = publish(run, result)
            record('provisional-best.json', dict(result, committed_checkpoint=receipt,
                screen_only=True, full_fid50k_confirmed=False))
        confirmation = read(LOCAL / 'winner-confirmation-50k.json')
        if confirmation is None:
            engine.load_candidate(result['checkpoint'])
            confirmation = engine.evaluate(result['candidate'] + '_confirmation_50k',
                                           CONFIRM_SEED, count=50000, batch=64)
            record('winner-confirmation-50k.json', confirmation)
        passed = confirmation['fid'] < HISTORICAL_FID
        result = dict(result, screen_evaluation=result['evaluation'],
            confirmation=confirmation, confirmation_delta=confirmation['fid'] - HISTORICAL_FID,
            confirmation_passed=passed, full_fid50k_confirmed=passed)
        run.log({'search/trial': len(TRIALS) + 1,
            'confirmation/fid_50k': confirmation['fid'],
            'confirmation/fid_delta_vs_historical': result['confirmation_delta'],
            'confirmation/passed': passed})
        if passed:
            engine.load_candidate(result['checkpoint'])
            result['checkpoint'] = engine.save('confirmed-' + result['candidate'],
                                               confirmation, result['training_proof'])
            result['evaluation'] = confirmation
            receipt = publish(run, result, final=True)
            record('best-checkpoint.json', dict(receipt, candidate=result['candidate'],
                source_artifact=SOURCE_ARTIFACT, source_retained=True,
                full_fid50k_confirmed=True, historical_baseline_hardware='B200'))
            winner = result
    summary = dict(completed=True, protocol='fast-screen-v2', winner=winner,
        original_best_preserved=True, baseline_screen=baseline,
        ranking=sorted(results, key=lambda r: r['evaluation']['fid']),
        no_improvement=winner is None, auto_long_training=False, time=time.time())
    record('completed.json', summary)
    run.summary.update({'search/status': 'completed_with_confirmed_improvement' if winner else 'completed_no_confirmed_improvement',
        'search/winner': winner['candidate'] if winner else None,
        'search/best_fid50k': winner['confirmation']['fid'] if winner else HISTORICAL_FID})
    stage('completed', winner=winner['candidate'] if winner else None)
    run.finish()
    dist.destroy_process_group()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        stage('stopped', original_best_preserved=True)
        raise
    except Exception as error:
        stage('failed', error_type=type(error).__name__, error=str(error))
        raise
