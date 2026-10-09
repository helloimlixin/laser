"""Consistent generation metric names, axes, durable results, and summaries."""
import json
import math
from pathlib import Path


def generation_payload(fid, inception_mean, inception_std, *, epoch, step):
    if not math.isfinite(fid):
        raise ValueError('Non-finite generation FID')
    payload = {'train/global_step': step, 'train/epoch': epoch,
               'eval/epoch': epoch, 'eval/global_step': step,
               'eval/fid': fid, 'val/fid': fid}
    if inception_mean is not None:
        if inception_std is None or not all(math.isfinite(x) for x in (inception_mean, inception_std)):
            raise ValueError('Non-finite generation Inception Score')
        payload.update({'eval/inception_score': inception_mean,
                        'eval/inception_score_std': inception_std,
                        'val/inception_score': inception_mean,
                        'val/inception_score_std': inception_std})
    return payload


def define_generation_metrics(run):
    run.define_metric('train/global_step')
    run.define_metric('eval/epoch', step_metric='train/global_step')
    for name in ('eval/fid', 'val/fid'):
        run.define_metric(name, step_metric='train/global_step', summary='min,last')
    for name in ('eval/inception_score', 'val/inception_score'):
        run.define_metric(name, step_metric='train/global_step', summary='max,last')
    for name in ('eval/inception_score_std', 'val/inception_score_std'):
        run.define_metric(name, step_metric='train/global_step', summary='last')


def log_generation_metrics(run, payload, output):
    """Save and log once evaluation returns, before checkpoint persistence."""
    folder = Path(output) / 'evaluations'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"generation_step_{int(payload['train/global_step']):07d}.json"
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(payload, indent=2) + '\n')
    temp.replace(path)
    if run is not None:
        run.log(payload)
        summary = {'evaluation/status': 'completed', 'evaluation/last_epoch': payload['eval/epoch'],
                   'evaluation/last_global_step': payload['train/global_step'],
                   'evaluation/last_fid': payload['eval/fid']}
        previous = run.summary.get('evaluation/best_fid')
        summary['evaluation/best_fid'] = min(payload['eval/fid'], previous) if previous is not None else payload['eval/fid']
        if 'eval/inception_score' in payload:
            score = payload['eval/inception_score']
            previous = run.summary.get('evaluation/best_inception_score')
            summary.update({'evaluation/last_inception_score': score,
                'evaluation/last_inception_score_std': payload['eval/inception_score_std'],
                'evaluation/best_inception_score': max(score, previous) if previous is not None else score})
        run.summary.update(summary)
        run.save(str(path), base_path=str(output), policy='now')
    print(json.dumps(dict(phase='generation_metrics', **payload)), flush=True)
