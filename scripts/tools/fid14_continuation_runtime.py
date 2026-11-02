"""I/O adaptations for the frozen, fully resumed historical ImageNet trainer.

The historical training implementation, objective, targets and LR controller
stay in its archived source. Validation FID remains the controller's input;
the public eval/fid uses all ImageNet training-reference statistics.
"""
import json
import time

import torch
import torch.distributed as dist


def install(namespace):
    g = namespace
    training, base, evidence = g['training'], g['BASE'], g['EVIDENCE']
    from imagenet_tar import TarImageNet

    def source_images(dataset, root, transform, *, split='train'):
        assert dataset == 'imagenet'
        result = TarImageNet(base/'data', split, transform, seed=261001)
        assert len(result) == (1281167 if split == 'train' else 50000)
        assert len(result.classes) == 1000
        return result

    training.source_image_dataset = source_images
    original_loader = g['original_loader']

    def loader(*args, **kwargs):
        if isinstance(args[0], TarImageNet):
            kwargs.update(num_workers=8, prefetch_factor=2)
        return original_loader(*args, **kwargs)

    training.DataLoader = loader
    native_init = g['original_init']

    class LoggedRun:
        def __init__(self, run):
            self.run = run
            self.resumed_logged_step = int(getattr(run, 'summary', {}).get('train/global_step', 0))

        def __getattr__(self, name):
            return getattr(self.run, name)

        def define_metric(self, name, **kwargs):
            if 'fid' in name.lower():
                kwargs.pop('summary', None)
            if name in ('val/fid', 'val/fid_rqvae_imagenet_train'):
                return None
            return self.run.define_metric(name, **kwargs)

        def log(self, values, *args, **kwargs):
            values = dict(values)
            if ('train/loss' in values
                    and values.get('train/global_step', 0) <= self.resumed_logged_step):
                return None
            if 'val/fid' in values:
                values['eval/fid'] = values.pop('val/fid')
                values.pop('val/fid_rqvae_imagenet_train', None)
                values['eval/inception_score'] = values.pop('val/inception_score')
                values['eval/inception_score_std'] = values.pop('val/inception_score_std')
            return self.run.log(values, *args, **kwargs)

    def open_run(*args, **kwargs):
        kwargs['resume'] = 'allow'
        kwargs['dir'] = str(base/'wandb')
        kwargs.setdefault('config', {}).update(
            historical_source_run='imagenet-rfid421-lossdecay-warm200-lr1e5-3epoch-8h100-20261006',
            resumed_global_step=g['INITIAL_RECOVERY']['global_step'],
            resumed_adam_age=g['INITIAL_RECOVERY'].get('adam_step', 14317),
            target_epoch=100, target_global_step=62600,
            fid_num_samples=50000, fid_real_split='train', fid_reference_images=1281167,
            lr_controller_reference='validation50000; separate diagnostic from eval/fid',
            original_objective_preserved=True,
            original_rng_streams_retained=True,
            trajectory_bitwise_identical=False,
            continuation_note='Four H100s with smaller microbatches; exact global image batch and cursor retained.')
        run = native_init(*args, **kwargs)
        run.define_metric('train/global_step')
        run.define_metric('eval/fid', step_metric='train/global_step', overwrite=True)
        run.define_metric('diagnostics/fid_validation', step_metric='train/global_step', overwrite=True)
        if g['ARGS'].wandb_mode == 'online':
            g['record'](evidence/'wandb.json', dict(id=run.id, url=run.url))
        return LoggedRun(run)

    g['original_init'] = open_run

    @torch.no_grad()
    def evaluate(model, aux, val_loader, num_samples, batch_size, **kwargs):
        from src import rqvae_metrics as metrics
        assert kwargs['metric_backend'] == 'original-rqvae' and num_samples == 50000
        device = next(model.parameters()).device
        rank, world = dist.get_rank(), dist.get_world_size()
        was_training = model.training
        model.eval()
        start = time.monotonic()
        local_count = num_samples//world + (rank < num_samples % world)
        with torch.random.fork_rng(devices=[device.index]):
            torch.manual_seed(261001+rank)
            torch.cuda.manual_seed(261001+rank)
            metric = metrics.DistributedOriginalRQVAEMetrics(device, compute_inception_score=True,
                reference_stats_path=base/'inputs/imagenet_256_train.npz')
            for first in range(0, local_count, batch_size):
                count = min(batch_size, local_count-first)
                labels = (torch.arange(first, first+count, device=device)*world+rank) % 1000
                tokens = model.sample_sparse(count, aux, cond=labels, amp=True)
                for chunk in tokens.split(4):
                    images = aux.decode_tokens(chunk).float().mul(.5).add(.5).clamp(0, 1)
                    metric.update(images, real=False)
                if rank == 0:
                    g['record'](evidence/'generation-progress.json', dict(
                        global_step=g['INITIAL_RECOVERY']['global_step']+g['UPDATES'],
                        generated_on_rank=first+count, samples_on_rank=local_count, time=time.time()))
            assert int(metric.fake_count) == local_count
            fid, inception, inception_std = metric.compute(shuffle_for_inception=True)
            assert int(metric.fake_count) == 50000
            validation = torch.zeros((), dtype=torch.float64, device=device)
            if rank == 0:
                mu, sigma = metrics._mean_covariance(metric.fake_sum, metric.fake_cross, 50000)
                real_mu, real_sigma = metrics.load_reference_statistics(base/'inputs/imagenet_256_val.npz')
                validation.fill_(metrics.frechet_distance(real_mu, real_sigma, mu, sigma))
            dist.broadcast(validation, src=0)
            del metric
        model.train(was_training)
        step = g['INITIAL_RECOVERY']['global_step']+g['UPDATES']
        decision = g['observe_official_fid'](float(validation))
        row = dict(global_step=step, fid=fid, inception_score=inception,
            inception_score_std=inception_std, fid_validation=float(validation),
            metric_backend='original_rqtransformer', real_images=1281167,
            generated_images=50000, real_split='train', inception_splits=10,
            seed=261001, sampler='joint32-depth-strong', elapsed_seconds=time.monotonic()-start,
            lr_controller_input='fid_validation', lr_controller_decision=decision,
            evaluation_protocol_version='fid50k-fulltrain-with-validation-controller-v1')
        g['OFFICIAL_METRICS'] = row
        g['record'](g['VERIFY']/('lr-controller-rank'+str(rank)+'.json'), decision)
        if rank == 0:
            g['record'](evidence/'evaluations'/f'step{step}.json', row)
            if g['WB'] is not None:
                g['WB'].log({'train/global_step':step, 'diagnostics/fid_validation':float(validation),
                    'lr_controller/reductions':decision['reductions'], 'train/lr':decision['lr_after']})
            print(json.dumps(dict(phase='fid_complete', **row)), flush=True)
        return fid, inception, inception_std

    training.evaluate_generation_metrics = evaluate
