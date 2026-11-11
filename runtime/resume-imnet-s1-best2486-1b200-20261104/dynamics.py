"""Serializable learning-rate and full-validation regression policy."""
import math


def initial_state():
    return dict(lr_factor=1., nonimproving_checks=0, evaluations=0, reductions=0)


def learning_rate(schedule, state, step):
    fraction = min(1., max(0., (step-schedule['start_step']) / schedule['decay_updates']))
    base = schedule['minimum'] + .5*(schedule['maximum']-schedule['minimum'])*(
        1.+math.cos(math.pi*fraction))
    return max(schedule['absolute_floor'], base*state['lr_factor'])


def evaluation_due(policy, step):
    elapsed = step-policy['source_step']
    return elapsed == policy['first_evaluation_updates'] or (
        elapsed > 0 and elapsed % policy['evaluation_every_updates'] == 0)


def observe(policy, state, previous_best, score):
    if not math.isfinite(score):
        raise ValueError('Nonfinite full-validation rFID')
    updated = dict(state)
    updated['evaluations'] += 1
    significant = score < previous_best-policy['minimum_improvement']
    updated['nonimproving_checks'] = 0 if significant else state['nonimproving_checks']+1
    limit = previous_best*(1.+policy['maximum_relative_regression'])
    action = 'continue'
    if score > limit:
        action = 'stop_regression'
    elif updated['nonimproving_checks'] >= policy['stop_after_nonimproving_checks']:
        action = 'stop_plateau'
    elif updated['nonimproving_checks'] and updated['nonimproving_checks'] % policy['reduce_after_nonimproving_checks'] == 0:
        updated['lr_factor'] *= policy['lr_reduction_factor']
        updated['reductions'] += 1
        action = 'reduce_learning_rate'
    return updated, dict(action=action, score=score, previous_best=previous_best,
        best_rfid=min(score, previous_best), regression_limit=limit,
        significant_improvement=significant, **updated)
