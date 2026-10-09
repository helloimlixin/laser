"""A bounded epoch79 fork changing only the online atom teacher policy."""
import os
from pathlib import Path

import torch
import torch.distributed as dist

from src.training.compound_geometry_trial_hook import seeded_official_evaluation
from src.training.stochastic_image_pairs import POLICY_VERSION, stochastic_image_components, validate_policy


def install(namespace, plan):
    validate_policy(plan['atom_temperature'], plan['site_chunk_size'])
    if plan['target_policy'] != POLICY_VERSION:
        raise RuntimeError('Stochastic target policy version changed')
    training = namespace['training']
    record, evidence, verify = namespace['record'], namespace['EVIDENCE'], namespace['VERIFY']
    policy = {key: plan[key] for key in ('target_policy', 'atom_temperature',
        'site_chunk_size', 'coefficient_temperature', 'source_step')}
    source_load = namespace['original_torch_load']

    def load(path, *args, **kwargs):
        payload = source_load(path, *args, **kwargs)
        options = namespace['ARGS']
        if (options is not None and options.resume and isinstance(path, (str, Path))
                and Path(path).resolve() == options.resume_checkpoint.resolve()):
            payload = dict(payload)
            previous = payload.get('stochastic_pair_targets')
            if previous is None:
                if (payload['epoch'], payload['global_step']) != (79, plan['source_step']):
                    raise RuntimeError('A new target trial must start from the protected epoch79 checkpoint')
                payload.update(best_fid=[], best_inception=[])
                payload.pop('train_loss_tracking', None)
                payload.pop('train_epoch_loss_tracking', None)
            elif previous != policy:
                raise RuntimeError('Target policy changed across resume')
        return payload
    namespace['original_torch_load'] = load

    def encode(aux, images, *, return_prefix_coeffs=False):
        if return_prefix_coeffs:
            raise RuntimeError('This trial preserves interleaved compound pairs without prefix refits')
        return stochastic_image_components(aux, images, temperature=plan['atom_temperature'],
            site_chunk_size=plan['site_chunk_size'])
    training.LaserAux.encode_sparse_components = encode
    native_targets = training.LaserAux.sparse_targets
    audited = False

    def targets(aux, atoms, coefficients, **kwargs):
        nonlocal audited
        if kwargs.get('stochastic') is not True or kwargs.get('hard', False):
            raise RuntimeError('Both atom and coefficient training inputs must be stochastic')
        if kwargs.get('temp') != plan['coefficient_temperature']:
            raise RuntimeError('Coefficient target temperature changed')
        cpu = torch.get_rng_state()
        cuda = torch.cuda.get_rng_state(atoms.device)
        result = native_targets(aux, atoms, coefficients, **kwargs)
        if not audited:
            with torch.random.fork_rng(devices=[atoms.device.index]):
                torch.set_rng_state(cpu)
                torch.cuda.set_rng_state(cuda, atoms.device)
                replay = native_targets(aux, atoms, coefficients, **kwargs)
                repeated = native_targets(aux, atoms, coefficients, **kwargs)
                if not torch.equal(result[0], replay[0]):
                    raise RuntimeError('Stochastic coefficient RNG replay failed')
                changed = (result[0][..., 1::2] != repeated[0][..., 1::2]).float().mean().item()
                if changed <= 0:
                    raise RuntimeError('Stochastic coefficient inputs did not change')
            record(verify / ('stochastic-targets-rank'+os.environ['RANK']+'.json'), dict(
                policy=policy, coefficient_rng_replay_exact=True,
                coefficient_changed_fraction=changed, sampled_atom_labels=True,
                support_and_coefficients_from_same_trajectory=True,
                full_vocabulary_atom_sampling=True, coefficient_clipping=False))
            audited = True
        return result
    training.LaserAux.sparse_targets = targets

    native_init = namespace['original_init']
    def init(*args, **kwargs):
        kwargs['resume'] = 'allow'
        kwargs['name'] = 'ImageNet K4 | epoch79 | stochastic atoms + coefficients'
        kwargs['config'].update(stochastic_pair_trial=plan, stochastic_pair_targets=policy,
            stochastic_atom_supports=True, stochastic_coefficients=True,
            training_target_estimator='sampled hard atom labels; conditional soft coefficient labels',
            objective_changed=False, target_distribution_changed=True,
            objective='existing equal atom/coeff CE + 0.05 coefficient CRPS',
            bounded_trial=True, trial_epochs=1, trial_target_epoch=80, continuation_target_epoch=80,
            automatic_fid_rewind=False, automatic_promotion=False,
            resume_source_epoch=79, source_checkpoint_epoch=79,
            resume_source_global_step=plan['source_step'], source_checkpoint_step=plan['source_step'],
            source_checkpoint_fid=plan['source_fid'], source_run=plan['source_run'],
            source_adam_step=15024, source_scheduler_step=plan['source_step'],
            new_scheduler_step=plan['source_step'], restart_initial_lr=plan['initial_lr'],
            optimizer_initialization='all801 saved AdamW moments/counters retained; no parameter changes',
            learning_rate_schedule='saved epoch79 absolute cosine to zero at epoch100; unchanged',
            lr_continuation='exact source optimizer and scheduler state restored',
            warmup_restarted=False, learning_rate_changed=False,
            evaluation_protocol='official RQ-Transformer; val50k/fake50k; two matched seeds; native ancestral sampler')
        return native_init(*args, **kwargs)
    namespace['original_init'] = init

    native_save = training.atomic_torch_save
    def save(payload, target):
        payload = dict(payload, stochastic_pair_targets=policy,
            config=dict(payload['config'], stochastic_pair_targets=policy,
                stochastic_atom_supports=True, stochastic_coefficients=True,
                coefficient_clipping=False))
        return native_save(payload, target)
    training.atomic_torch_save = save

    def before_evaluation(model, optimizer, parameter_names, device, epoch, global_step,
                          scheduler, runtime_config, best_fid, best_inception, last_checkpoint):
        state, opt = training.full_checkpoint_states(model, optimizer, parameter_names)
        rng = training.gather_rank_rng_states(device)
        if dist.get_rank() == 0:
            training.atomic_torch_save(dict(epoch=epoch+1, batch_idx=0, global_step=global_step,
                fid=None, inception_score=None, inception_score_std=None,
                state_dict=state, optimizer=opt, rng_state_by_rank=rng,
                checkpoint_world_size=dist.get_world_size(), scheduler=scheduler.state_dict(),
                config=runtime_config, best_fid=best_fid, best_inception=best_inception), last_checkpoint)
            namespace['WRITER'].wait()
            record(evidence/'pre-evaluation-checkpoint.json', dict(global_step=global_step,
                epoch=epoch+1, durable=True, rng_ranks=len(rng), evaluation_pending=True))
        dist.barrier()
    training.save_before_official_evaluation = before_evaluation

    def evaluate(*args, **kwargs):
        if kwargs['metric_backend'] != 'original-rqvae':
            raise RuntimeError('Only official evaluation is authorized')
        results = []
        for seed in plan['evaluation_seeds']:
            result = seeded_official_evaluation(namespace['original_evaluate'], *args,
                seed=seed, process_rank=dist.get_rank(), **kwargs)
            metric = dict(global_step=namespace['INITIAL_RECOVERY']['global_step']+namespace['UPDATES'],
                seed=seed, fid=result[0], inception_score=result[1], inception_score_std=result[2],
                metric_backend='original_rqtransformer', real_images=50000, generated_images=50000,
                real_split='val', inception_splits=10, sampler='native ancestral')
            results.append(metric)
            if dist.get_rank() == 0:
                record(evidence/f'official-seed{seed}.json', metric)
                namespace['WB'].log({'train/global_step':metric['global_step'],
                    f'eval/seed{seed}/fid_original_rqtransformer':result[0],
                    f'eval/seed{seed}/inception_score_original_rqtransformer':result[1],
                    f'eval/seed{seed}/inception_score_std_original_rqtransformer':result[2]})
        namespace['OFFICIAL_METRICS'] = results[0]
        summary = dict(evaluations=results, fid_mean=sum(r['fid'] for r in results)/len(results),
            inception_score_mean=sum(r['inception_score'] for r in results)/len(results))
        if dist.get_rank() == 0:
            record(evidence/'official-two-seed-summary.json', summary)
            namespace['WB'].summary['trial/official_two_seed_results'] = summary
        return tuple(results[0][key] for key in ('fid', 'inception_score', 'inception_score_std'))
    training.evaluate_generation_metrics = evaluate
