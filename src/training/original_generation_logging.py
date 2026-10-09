"""One original train-reference FID and one corresponding IS per evaluation."""
import json
import math
from pathlib import Path

FID_KEY = 'eval/fid_original_train50k'
IS_KEY = 'eval/inception_score'
FID_PROTOCOL = 'original_rqvae_training_reference'


def generation_payload(values):
    fid = float(values['val/fid'])
    score = float(values['val/inception_score'])
    if not all(math.isfinite(x) for x in (fid, score)):
        raise ValueError('Non-finite original generation metrics')
    return {'train/global_step': values['train/global_step'],
            'train/epoch': values['train/epoch'], FID_KEY: fid, IS_KEY: score}


def log_generation(run, values, output):
    payload = generation_payload(values)
    folder = Path(output) / 'evaluations'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"original_generation_step_{int(payload['train/global_step']):07d}.json"
    result = dict(payload, fid_protocol=FID_PROTOCOL, generated_images=50000,
                  inception_score_std=values.get('val/inception_score_std'))
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(path)
    run.log(payload)
    run.summary.update({FID_KEY: payload[FID_KEY], IS_KEY: payload[IS_KEY]})
    run.save(str(path), base_path=str(output), policy='now')


def define_metrics(run):
    run.define_metric('train/global_step')
    for key in ('val/fid', 'val/inception_score', 'val/inception_score_std',
                'eval/fid', 'eval/fid_original_val50k', 'eval/fid_torchmetrics_val50k',
                'eval/inception_score_std', 'eval/epoch', 'eval/global_step'):
        run.define_metric(key, hidden=True, summary='none', overwrite=True)
    for key in (FID_KEY, IS_KEY):
        # The explicit scalar summaries below avoid old min/max/last aliases.
        run.define_metric(key, step_metric='train/global_step', summary='none', overwrite=True)


def restore_summary(run, previous_results, output):
    candidates = [{'train/global_step':row['global_step'],
                   FID_KEY:row['original_train_fid'], IS_KEY:row['inception_score']}
                  for row in previous_results]
    for path in (Path(output)/'evaluations').glob('original_generation_step_*.json'):
        candidates.append(json.loads(path.read_text()))
    if candidates:
        latest = max(candidates, key=lambda row:row['train/global_step'])
        run.summary.update({key:latest[key] for key in (FID_KEY, IS_KEY)})


def clear_legacy_summaries(run):
    for key in list(run.summary.keys()):
        if ((key.startswith('eval/') and key not in (FID_KEY, IS_KEY))
                or key.startswith(('val/fid', 'val/inception_score'))
                or key.startswith(('evaluation/last_', 'evaluation/best_',
                                   'evaluation/validation_', 'evaluation/companion_'))
                or key == 'evaluation/status'):
            del run.summary[key]


def rebase_checkpoint_fids(payload, previous_results):
    """Translate saved rankings using original FIDs from the same generated sets."""
    config = payload.get('config', {})
    protocol = config.get('checkpoint_fid_metric')
    if protocol == FID_PROTOCOL:
        return payload
    if protocol != 'torchmetrics_validation50k':
        raise ValueError(f'Unknown previous checkpoint FID protocol: {protocol}')

    def original(value):
        matches = [row['original_train_fid'] for row in previous_results
                   if math.isclose(float(value), row['previous_selected_fid'],
                                   rel_tol=0., abs_tol=1e-7)]
        if len(matches) != 1:
            raise ValueError(f'No unique original FID recorded for checkpoint score {value}')
        return float(matches[0])

    rankings = [(original(score), path) for score, path in payload.get('best_fid', [])]
    fid = None if payload.get('fid') is None else original(payload['fid'])
    payload['best_fid'] = sorted(rankings, key=lambda item: item[0])
    payload['fid'] = fid
    payload['config'] = dict(config, checkpoint_fid_metric=FID_PROTOCOL,
                             fid_metric_rebased_from=protocol)
    return payload
