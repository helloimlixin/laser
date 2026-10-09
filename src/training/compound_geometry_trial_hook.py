"""Bounded, matched compound-energy/control forks of the preserved trainer."""
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist

from src.training.physical_compound_geometry import compound_energy_distance


def seeded_official_evaluation(evaluator, model, *args, seed, process_rank, **kwargs):
    """Seed the frozen evaluator without changing its supported arguments."""
    device = next(model.parameters()).device
    devices = [device.index] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed + process_rank)
        if device.type == 'cuda':
            torch.cuda.manual_seed(seed + process_rank)
        return evaluator(model, *args, **kwargs)


def install(namespace, plan):
    training = namespace['training']
    record, verify, evidence = namespace['record'], namespace['VERIFY'], namespace['EVIDENCE']
    branch = os.environ['LASER_GEOMETRY_BRANCH']
    maximum_weight = None
    latest_output = None
    diagnostic_sum = None
    additional_objective = 0.
    source_load = namespace['original_torch_load']
    from src.training.training_loss_tracker import TrainingLossTracker
    tracker_update = TrainingLossTracker.update
    def update_tracker(self, *args, **kwargs):
        return tracker_update(self, *args, **kwargs, additional_loss=additional_objective)
    TrainingLossTracker.update = update_tracker

    def load(path, *args, **kwargs):
        nonlocal maximum_weight
        payload = source_load(path, *args, **kwargs)
        args_object = namespace['ARGS']
        if (args_object is not None and isinstance(path, (str, Path)) and args_object.resume and
                Path(path).resolve() == args_object.resume_checkpoint.resolve()):
            payload = dict(payload)
            if not payload.get('compound_energy_trial'):
                # This is a new experiment, not a model/optimizer migration.
                payload.update(best_fid=[], best_inception=[])
                payload.pop('train_loss_tracking', None)
                payload.pop('train_epoch_loss_tracking', None)
            else:
                trial = payload['compound_energy_trial']
                if trial['version'] != plan['version'] or trial['branch'] != branch:
                    raise RuntimeError('A different geometry experiment cannot be resumed here')
                maximum_weight = trial['maximum_weight']
                namespace['COMPOUND_MODEL'].compound_geometry_cursor = trial['geometry_cursor']
        return payload
    namespace['original_torch_load'] = load

    original_build = training.build_model
    def build(*args, **kwargs):
        model = original_build(*args, **kwargs)
        model.compound_geometry_config = dict(top_k=plan['candidate_top_k'],
                                             sites_per_image=plan['sites_per_image'])
        model.compound_geometry_cursor = plan['source_step'] * 4
        def capture(module, inputs, output):
            nonlocal latest_output
            if module.training:
                latest_output = output
        model.register_forward_hook(capture)
        return model
    training.build_model = build

    actual_init = namespace['original_init']
    def init(*args, **kwargs):
        kwargs['resume'] = 'allow'
        kwargs['name'] = 'ImageNet K4 | epoch79 | ' + branch + ' | matched energy trial'
        kwargs['config'].update(
            compound_energy_trial=plan, objective_changed=branch == 'geometry',
            objective='CE + coefficient CRPS' + (' + compound energy distance' if branch == 'geometry' else ''),
            objective_revision='compound-energy-distance-ab-v1',
            bounded_trial=True, trial_epochs=1, trial_target_epoch=80,
            continuation_target_epoch=80, resume_source_epoch=79,
            resume_source_global_step=plan['source_step'], source_checkpoint_epoch=79,
            source_checkpoint_step=plan['source_step'], source_checkpoint_fid=plan['source_fid'],
            source_run=plan['source_run'], source_adam_step=plan['common_adam_step'],
            source_scheduler_step=plan['source_step'], new_scheduler_step=plan['source_step'],
            optimizer_initialization='all801 saved AdamW moments/counters retained; no parameter changes',
            learning_rate_schedule='saved epoch79 absolute cosine to zero at epoch100, unchanged in both forks',
            warmup_restarted=False, learning_rate_changed=False,
            geometry_branch=branch, geometry_candidate_top_k=plan['candidate_top_k'],
            geometry_coefficient_groups=plan['coefficient_groups'],
            geometry_target_gradient_ratio=plan['target_gradient_ratio'],
            geometry_max_weight_cap=plan['weight_cap'], geometry_ramp_updates=plan['ramp_updates'],
            geometry_sampling='two rotating spatial sites, all four depths, per image',
            geometry_rng='private fork; native training RNG streams unchanged',
            full_optimizer_checkpoint_uploads=True, automatic_fid_rewind=False,
            evaluation_protocol='official RQ-Transformer; val50k/fake50k; two matched seeds; native ancestral sampler')
        return actual_init(*args, **kwargs)
    namespace['original_init'] = init

    original_objective = training.physical_pair_objective
    def objective(atom_logits, coeff_logits, atoms, probabilities, bins, weight, accumulation):
        nonlocal maximum_weight, diagnostic_sum, latest_output
        total = original_objective(atom_logits, coeff_logits, atoms, probabilities, bins, weight, accumulation)
        output = latest_output
        if output is None:
            raise RuntimeError('Compound geometry forward output is missing')
        rows = output['geometry_selected_rows']
        depths = atoms.shape[-1]
        target = probabilities.reshape(-1, depths, probabilities.shape[-1])[rows].flatten(0, 1)
        aux = namespace['PROBE_AUX']
        def geometry(a, c):
            return compound_energy_distance(a, c, output['geometry_candidate_atoms'],
                output['geometry_teacher_atoms'], target, aux.dictionary, bins,
                groups=plan['coefficient_groups'])
        raw = geometry(output['geometry_atom_logits'], output['geometry_coefficient_logits'])
        if maximum_weight is None:
            # Compare logit gradients on exactly the same selected examples.
            # This diagnostic graph does not retain or backprop through the
            # compiled native training graph or disturb its donated buffers.
            a = output['geometry_atom_logits'].detach().clone().requires_grad_(True)
            c = output['geometry_coefficient_logits'].detach().clone().float().requires_grad_(True)
            geo = geometry(a, c)
            ag, cg = torch.autograd.grad(geo, (a, c))
            matching = output['geometry_candidate_atoms'] == output['geometry_teacher_atoms'][:, None]
            teacher_index = matching.float().argmax(-1)
            teacher_c = c[torch.arange(c.shape[0], device=c.device), teacher_index]
            # Full atom CE gradients, rather than the truncated candidate CE.
            full_a = atom_logits.reshape(-1, depths, atom_logits.shape[-1])[rows].flatten(0, 1).detach().clone().float().requires_grad_(True)
            ce = (-full_a.log_softmax(-1).gather(-1, output['geometry_teacher_atoms'][:, None]).mean()
                  -(target * teacher_c.float().log_softmax(-1)).sum(-1).mean()) / 2
            ac, cc = torch.autograd.grad(ce, (full_a, c))
            norms = torch.stack((ac.square().sum() + cc.square().sum(), ag.square().sum() + cg.square().sum()))
            dist.all_reduce(norms)
            ce_norm, geometry_norm = norms.sqrt().tolist()
            if not all(math.isfinite(x) and x > 0 for x in (ce_norm, geometry_norm)):
                raise RuntimeError('Geometry gradient calibration is nonfinite or zero')
            maximum_weight = min(plan['weight_cap'], plan['target_gradient_ratio'] * ce_norm / geometry_norm)
            calibration = dict(maximum_weight=maximum_weight, ce_logit_gradient_norm=ce_norm,
                geometry_logit_gradient_norm=geometry_norm,
                maximum_weighted_gradient_ratio=maximum_weight * geometry_norm / ce_norm,
                target_ratio=plan['target_gradient_ratio'], weight_cap=plan['weight_cap'],
                global_step=plan['source_step'], branch=branch, parameter_gradient_ratio_not_measured=True)
            record(verify / ('geometry-calibration-rank'+os.environ['RANK']+'.json'), calibration)
            shared = Path(plan['calibration_file'])
            if dist.get_rank() == 0 and (branch == 'control' or os.environ.get('LASER_GEOMETRY_PREFLIGHT') == '1'):
                record(shared, calibration)
            dist.barrier()
            if branch == 'geometry':
                import json
                reference = json.loads(shared.read_text())
                if not math.isclose(maximum_weight, reference['maximum_weight'], rel_tol=2e-3):
                    raise RuntimeError('Matched fork geometry calibration differs at its source')
                maximum_weight = reference['maximum_weight']
            del a, c, ag, cg, full_a, ac, cc, geo, ce
        update = namespace['INITIAL_RECOVERY']['global_step'] - plan['source_step'] + namespace['UPDATES']
        applied_weight = maximum_weight * min(1., (update + 1) / plan['ramp_updates']) if branch == 'geometry' else 0.
        total = total + applied_weight * raw / accumulation
        namespace['TRAIN_OBJECTIVE_SUM'][0].add_(applied_weight * raw.detach() * atoms.shape[0])
        with torch.no_grad():
            selected_a = atom_logits.reshape(-1, depths, atom_logits.shape[-1])[rows].float()
            selected_c = coeff_logits.reshape(-1, depths, coeff_logits.shape[-1])[rows].float()
            selected_atoms = atoms.reshape(-1, depths)[rows]
            target_depth = target.reshape(-1, depths, target.shape[-1])
            atom_nll = -selected_a.log_softmax(-1).gather(-1, selected_atoms[..., None]).squeeze(-1).mean(0)
            coefficient_ce = -(target_depth * selected_c.log_softmax(-1)).sum(-1).mean(0)
            entropy = -(target_depth * target_depth.clamp_min(1e-30).log()).sum(-1).mean(0)
            values = torch.cat((raw.detach()[None], raw.new_tensor([applied_weight]),
                                atom_nll, coefficient_ce, entropy, coefficient_ce - entropy,
                                raw.new_tensor([1.]))) * atoms.shape[0]
            diagnostic_sum = values if diagnostic_sum is None else diagnostic_sum + values
        latest_output = None
        return total
    training.physical_pair_objective = objective

    original_step = torch.optim.AdamW.step
    def step(optimizer, *args, **kwargs):
        nonlocal diagnostic_sum, additional_objective
        dist.all_reduce(diagnostic_sum)
        values = (diagnostic_sum[:-1] / diagnostic_sum[-1]).tolist()
        diagnostic_sum = None
        additional_objective = values[0] * values[1]
        result = original_step(optimizer, *args, **kwargs)
        update = namespace['INITIAL_RECOVERY']['global_step'] - plan['source_step'] + namespace['UPDATES']
        report = dict(global_step=plan['source_step']+update, updates=update,
            raw_energy_distance=values[0], applied_weight=values[1], maximum_weight=maximum_weight,
            atom_nll_by_depth=values[2:6], coefficient_ce_by_depth=values[6:10],
            coefficient_target_entropy_by_depth=values[10:14], coefficient_kl_by_depth=values[14:18],
            diagnostics='rotating training subset, not image-quality evaluation',
            optimizer_lr=optimizer.param_groups[0]['lr'], adam_reset=False)
        if update in (1, 20):
            record(verify / (f'energy-step{update}-rank'+os.environ['RANK']+'.json'), report)
        if dist.get_rank() == 0 and (update in (1, 20) or update % 10 == 0):
            payload = {'train/global_step':report['global_step'], 'train/compound_energy':values[0],
                       'train/compound_energy_weight':values[1]}
            for depth in range(4):
                payload.update({f'train/atom_nll_depth{depth}':values[2+depth],
                    f'train/coefficient_ce_depth{depth}':values[6+depth],
                    f'train/coefficient_target_entropy_depth{depth}':values[10+depth],
                    f'train/coefficient_kl_depth{depth}':values[14+depth]})
            namespace['WB'].log(payload)
            record(evidence / 'latest-training-diagnostics.json', report)
        return result
    torch.optim.AdamW.step = step

    original_save = training.atomic_torch_save
    def save(payload, target):
        payload = dict(payload, compound_energy_trial=dict(plan, branch=branch,
            maximum_weight=maximum_weight, geometry_cursor=namespace['COMPOUND_MODEL'].compound_geometry_cursor))
        return original_save(payload, target)
    training.atomic_torch_save = save

    def save_before_official_evaluation(model, optimizer, parameter_names, device,
            epoch, global_step, scheduler, runtime_config, best_fid, best_inception,
            last_checkpoint):
        model_state, optimizer_state = training.full_checkpoint_states(model, optimizer, parameter_names)
        rng = training.gather_rank_rng_states(device)
        if dist.get_rank() == 0:
            training.atomic_torch_save(dict(epoch=epoch + 1, batch_idx=0,
                global_step=global_step, fid=None, inception_score=None, inception_score_std=None,
                state_dict=model_state, optimizer=optimizer_state, rng_state_by_rank=rng,
                checkpoint_world_size=dist.get_world_size(),
                scheduler=None if scheduler is None else scheduler.state_dict(),
                config=runtime_config, best_fid=best_fid, best_inception=best_inception), last_checkpoint)
            namespace['WRITER'].wait()
            record(evidence / 'pre-evaluation-checkpoint.json', dict(
                epoch=epoch + 1, global_step=global_step, durable=True, rng_ranks=len(rng),
                evaluation_pending=True, checkpoint=str(last_checkpoint)))
        dist.barrier()
    training.save_before_official_evaluation = save_before_official_evaluation

    def evaluate(*args, **kwargs):
        if kwargs['metric_backend'] != 'original-rqvae':
            raise RuntimeError('Only the official evaluation implementation is authorized')
        results = []
        for seed in plan['evaluation_seeds']:
            result = seeded_official_evaluation(namespace['original_evaluate'], *args,
                seed=seed, process_rank=dist.get_rank(), **kwargs)
            metric = dict(global_step=namespace['INITIAL_RECOVERY']['global_step']+namespace['UPDATES'], seed=seed,
                fid=result[0], inception_score=result[1], inception_score_std=result[2],
                metric_backend='original_rqtransformer', real_images=50000, generated_images=50000,
                real_split='val', inception_splits=10, sampler='native ancestral', branch=branch)
            results.append(metric)
            if dist.get_rank() == 0:
                record(evidence / f'official-seed{seed}.json', metric)
                namespace['WB'].log({'train/global_step':metric['global_step'],
                    f'eval/seed{seed}/fid_original_rqtransformer':result[0],
                    f'eval/seed{seed}/inception_score_original_rqtransformer':result[1],
                    f'eval/seed{seed}/inception_score_std_original_rqtransformer':result[2]})
        namespace['OFFICIAL_METRICS'] = results[0]
        summary = dict(branch=branch, evaluations=results,
            fid_mean=sum(x['fid'] for x in results)/len(results),
            inception_score_mean=sum(x['inception_score'] for x in results)/len(results),
            maximum_geometry_weight=maximum_weight)
        if dist.get_rank() == 0:
            record(evidence / 'official-two-seed-summary.json', summary)
            namespace['WB'].summary['trial/official_two_seed_results'] = summary
        return tuple(results[0][key] for key in ('fid','inception_score','inception_score_std'))
    training.evaluate_generation_metrics = evaluate
