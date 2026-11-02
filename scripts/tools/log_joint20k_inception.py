#!/usr/bin/env python3
"""Add IS views, confirmation summaries and comparison columns without touching training."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import time

ROOT = Path('/workspace/Projects/laser')
EXPERIMENT = ROOT / 'outputs/imagenet-laser-joint20k-depth-loss-epoch1to3-20261018'
KEY = Path('/root/.config/laser/imagenet-stage2-wandb.key')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def inception(row, confirmation=False):
    prefix = '' if confirmation else 'eval/'
    mean, std = float(row[prefix + 'inception_score']), float(row[prefix + 'inception_score_std'])
    if not math.isfinite(mean) or mean <= 0 or not math.isfinite(std) or std < 0:
        raise ValueError('Invalid Inception Score or split standard deviation')
    if row['generated_images'] != 50000:
        raise ValueError('Expected the matched 50,000 generated images')
    return dict(mean=mean, std=std)


def enrich_comparison(plan, result):
    result = copy.deepcopy(result)
    for row in result['rows']:
        scores = {}
        for arm, info in plan['arms'].items():
            confirmation = row['seed'] != 261001
            name = (f'evaluations/confirmation-seed{row["seed"]}.json' if confirmation
                    else f'evaluations/epoch{row["epoch"]:03d}.json')
            value = json.loads((Path(info['out']) / name).read_text())
            step = value['global_step'] if confirmation else value['train/global_step']
            if step != row['epoch'] * 626:
                raise ValueError('Mismatched evaluation update')
            scores[arm] = inception(value, confirmation)
        row['inception_score'] = scores
        row['inception_score_improvement'] = scores['weighted']['mean'] - scores['control']['mean']
    result['inception_score_logging'] = dict(mean_and_split_std=True,
        primary_history_metrics=['eval/inception_score', 'eval/inception_score_std'],
        confirmation_summary_metrics=['eval/inception_score_confirmation', 'eval/inception_score_confirmation_std'])
    return result


def panels(wr):
    return [wr.LinePlot(title='Inception Score mean', x='train/epoch', y=['eval/inception_score']),
            wr.LinePlot(title='Inception Score standard deviation', x='train/epoch', y=['eval/inception_score_std']),
            wr.ScalarChart(title='Confirmation Inception Score mean', metric='eval/inception_score_confirmation'),
            wr.ScalarChart(title='Confirmation Inception Score standard deviation', metric='eval/inception_score_confirmation_std')]


def setup_views(plan, state):
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr
    from wandb_workspaces.expr import Or

    if 'comparison_workspace' not in state:
        view = ws.Workspace(entity='helloimlixin-rutgers', project='laser',
            name='LASER depth-loss epoch1 to3 - FID and Inception Score',
            runset_settings=ws.RunsetSettings(filters=Or(*[
                ws.Config('wandb_id') == info['run_id'] for info in plan['arms'].values()])),
            sections=[ws.Section(name='Generation quality', is_open=True, pinned=True, panels=[
                wr.LinePlot(title='FID50k', x='train/epoch', y=['eval/fid']), *panels(wr),
                wr.LinePlot(title='Unweighted token cross entropy', x='train/global_step', y=['train/unweighted_ce'])])])
        view.save()
        state['comparison_workspace'] = view.url
        write(EXPERIMENT / 'inception-logging.json', state)
    for arm, info in plan['arms'].items():
        key = arm + '_workspace'
        receipt = Path(info['out']) / 'workspace.json'
        if key in state or not receipt.exists():
            continue
        view = ws.Workspace.from_url(json.loads(receipt.read_text())['url'])
        if not any(section.name == 'Inception Score' for section in view.sections):
            view.sections.append(ws.Section(name='Inception Score', is_open=True, pinned=True, panels=panels(wr)))
            view.save()
        state[key] = view.url
        write(EXPERIMENT / 'inception-logging.json', state)


def cycle(plan, state):
    import wandb

    setup_views(plan, state)
    api = wandb.Api()
    for arm, info in plan['arms'].items():
        key = arm + '_confirmation'
        folder = Path(info['out'])
        confirmation = folder / 'evaluations/confirmation-seed261101.json'
        if key in state or not confirmation.exists() or not (folder / 'completed.json').exists():
            continue
        # Wait for the training writer to finish before adding summary fields.
        run = api.run('helloimlixin-rutgers/laser/' + info['run_id'])
        if run.state != 'finished':
            continue
        row = json.loads(confirmation.read_text())
        score = inception(row, confirmation=True)
        run.summary.update({'eval/inception_score_confirmation': score['mean'],
            'eval/inception_score_confirmation_std': score['std'],
            'evaluation/confirmation_seed': 261101, 'evaluation/confirmation_images': 50000})
        state[key] = score
        write(EXPERIMENT / 'inception-logging.json', state)
    comparison = EXPERIMENT / 'comparison.json'
    if comparison.exists():
        value = json.loads(comparison.read_text())
        if 'inception_score_logging' not in value:
            write(comparison, enrich_comparison(plan, value))
        state['comparison_includes_inception_score'] = True
    state['updated'] = time.time()
    state['complete'] = all(arm + '_confirmation' in state for arm in plan['arms']) and bool(
        state.get('comparison_includes_inception_score'))
    write(EXPERIMENT / 'inception-logging.json', state)
    return state['complete']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['WANDB_API_KEY'] = KEY.read_text().strip()
    plan = json.loads((EXPERIMENT / 'plan.json').read_text())
    path = EXPERIMENT / 'inception-logging.json'
    state = json.loads(path.read_text()) if path.exists() else {}
    while True:
        try:
            complete = cycle(plan, state)
            state.pop('last_error', None)
            write(path, state)
            if complete or not args.watch:
                print(json.dumps(state), flush=True)
                return
            status = json.loads((EXPERIMENT / 'status.json').read_text())
            if status['phase'] == 'failed':
                state['stopped_because_experiment_failed'] = True
                write(path, state)
                return
        except Exception as error:
            state.update(last_error=str(error), updated=time.time())
            write(path, state)
            if not args.watch:
                raise
        time.sleep(20)


if __name__ == '__main__':
    main()
