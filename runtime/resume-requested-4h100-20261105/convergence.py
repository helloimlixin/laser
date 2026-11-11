"""Checkpointed late-stage FID plateau control without restarting the cosine clock."""
from runtime.lr_recovery import apply_lr_amplitude


def observe(state, evaluation, optimizer, scheduler, factor, *, epoch, step):
    state = dict(state or {})
    fid, score = float(evaluation['fid']), float(evaluation['inception_score'])
    # A changed GPU topology changes sampling streams; establish a fresh baseline.
    if not state:
        state = dict(best_fid=fid, best_is=score, bad_evaluations=0, evaluations=1,
                     patience=2, minimum_delta=0.05, minimum_factor=0.0015625)
        return state, factor, None
    state['evaluations'] += 1
    improved = fid < state['best_fid'] - state['minimum_delta']
    if improved:
        state.update(best_fid=fid, best_is=score, bad_evaluations=0)
    else:
        state['bad_evaluations'] += 1
    event = None
    if state['bad_evaluations'] >= state['patience']:
        desired = max(state['minimum_factor'], factor * 0.5)
        if desired < factor:
            event = dict(epoch=epoch, global_step=step, fid=fid, inception_score=score,
                         reason='Two full FID50k evaluations without a 0.05 improvement',
                         **apply_lr_amplitude(optimizer, scheduler,
                             previous_factor=factor, desired_factor=desired))
            factor = desired
        state['bad_evaluations'] = 0
    return state, factor, event
