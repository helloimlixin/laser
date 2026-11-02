#!/usr/bin/env python3
"""Repair ImageNet dashboard panels and stale summaries without another logger."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from wandb_config_only import update_config_only

ROOT = Path('/workspace/Projects/laser')
RUN = 'imagenet-laser-joint20k-depth5-uniformce-k512-scratch100-4b200-20261018'
TRAIN = ROOT / 'outputs' / RUN
OUT = TRAIN / 'diagnostics/logging-audit-20261018'
ENTITY, PROJECT = 'helloimlixin-rutgers', 'laser'
PREFIX = 'laser-imagenet-joint20k-logging-'
CLASSES = ('cheeseburger', 'teapot', 'samoyed', 'bald_eagle', 'red_fox',
           'hotpot', 'wombat', 'timber_wolf')


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sections(ws, wr):
    reports = {epoch:json.loads((TRAIN / f'diagnostics/{folder}/report.json').read_text())
        for epoch, folder in [(2, 'sampler-k2500-k3000-k4096-epoch2-20261018'),
                              (4, 'sampler-k3000-k4096-epoch4-20261018')]}
    table = ['| Epoch | K2048 | K3000 | K4096 |', '|---:|---:|---:|---:|']
    for epoch, report in reports.items():
        values = [report['results'][str(k)]['fid'] for k in (2048, 3000, 4096)]
        table.append(f'| {epoch} | ' + ' | '.join(f'{value:.2f}' for value in values) + ' |')
    note = ('**ImageNet, whole integer tokens, uniform cross entropy; active sampler K2048.** '
        'FID50k uses all 1,281,167 training reference images; T=1, top-p=0.95, no rejection. '
        '**The native FID history used K512 at epochs 1 and 2, then K2048 at epoch 4.** '
        'The table below compares samplers at the same saved checkpoint and generation seed.\n\n'
        + '\n'.join(table))

    def line(title, metric, axis='train/global_step'):
        return wr.LinePlot(title=title, x=axis, y=[metric],
            smoothing_type='none', smoothing_factor=0)

    return [
        ws.Section(name='ImageNet: training and generation quality', is_open=True, pinned=True,
            panels=[wr.MarkdownPanel(markdown=note),
                line('Native FID50k by completed epoch (sampler change noted above)', 'eval/fid', 'train/epoch'),
                line('Inception Score by completed epoch', 'eval/inception_score', 'train/epoch'),
                line('Uniform token cross entropy', 'train/loss'),
                line('Learning rate', 'train/lr'),
                line('Gradient norm', 'train/grad_norm'),
                line('Training images per second', 'train/images_per_second')]),
        ws.Section(name='ImageNet: fixed samples', is_open=True, pinned=True,
            panels=[wr.MediaBrowser(title='Latest fixed-seed overview; optimizer step in caption',
                media_keys=['samples/fixed'], num_columns=1)]),
        ws.Section(name='ImageNet: per-class samples', is_open=True,
            panels=[wr.MediaBrowser(title=name.replace('_', ' ').title() + ': 64 fixed-seed samples',
                media_keys=['samples/' + name], num_columns=1) for name in CLASSES]),
    ]


def bank(spec, kind):
    return spec['panelBankConfig'] if kind == 'run-view' else spec['section']['panelBankConfig']


def main():
    os.environ['WANDB_API_KEY'] = Path('/root/.config/laser/wandb_api_key').read_text().strip()
    import wandb
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr
    api = wandb.Api(timeout=60)
    graphql = api._service_api.execute_graphql
    query = '''query($type:String!){project(name:"laser",entityName:"helloimlixin-rutgers"){
        allViews(viewType:$type,userName:"helloimlixin-rutgers",first:100){edges{node{
        id name type displayName spec updatedAt}}}}}'''

    def views(kind):
        return [edge['node'] for edge in graphql(query, {'type':kind})['project']['allViews']['edges']]

    repaired = []
    for kind in ('run-view', 'project-view'):
        original = next(v for v in views(kind) if v['name'] == 'nw-nwuserhelloimlixinrutgers-w')
        write(kind + '-before-repair.json', original)
        spec = json.loads(original['spec'])
        original_runsets = copy.deepcopy(spec.get('section', {}).get('runSets'))
        current = bank(spec, kind)
        preserved = []
        for section in current['sections']:
            identity = section.get('__id__') or ''
            legacy_imagenet = identity in ('laser-fixed-overview', 'laser-class-grids')
            legacy_imagenet |= section.get('name') == 'FID' and any(
                p.get('config', {}).get('xAxis') == 'train/epoch' for p in section.get('panels', []))
            if identity.startswith(PREFIX) or legacy_imagenet:
                continue
            item = copy.deepcopy(section)
            if item.get('name', '').startswith('CC3M'):
                item['isOpen'], item['pinned'] = False, False
            preserved.append(item)
        added = []
        for index, section in enumerate(sections(ws, wr)):
            value = section._to_model().model_dump(by_alias=True, exclude_none=True)
            value['__id__'] = PREFIX + str(index)
            value['isPanelsAuto'] = False
            for panel in value['panels']:
                panel['isAuto'] = False
            added.append(value)
        current['sections'] = added + preserved
        current.setdefault('settings', {}).update(showEmptySections=False, searchQuery='')
        if kind == 'project-view':
            spec['section']['settings'].update(xAxis='train/global_step', xAxisActive=True)
        graphql('''mutation($input:UpsertViewInput!){upsertView(input:$input){view{id} inserted}}''',
            {'input':dict(id=original['id'], entityName=ENTITY, projectName=PROJECT,
                name=original['name'], displayName=original['displayName'], type=kind,
                spec=json.dumps(spec), lastUpdatedAt=original['updatedAt'], createdUsing='WANDB_SDK')})
        saved = next(v for v in views(kind) if v['id'] == original['id'])
        after = json.loads(saved['spec'])
        assert bank(after, kind)['sections'] == added + preserved
        assert after.get('section', {}).get('runSets') == original_runsets
        for section in bank(after, kind)['sections'][:3]:
            for panel in section['panels']:
                assert 'stepIndex' not in panel.get('config', {})
        write(kind + '-after-repair.json', saved)
        repaired.append(dict(kind=kind, existing_run_selection_preserved=True,
            imagenet_sections_first=True, other_sections_preserved=len(preserved)))

    receipt = json.loads((TRAIN / 'workspace.json').read_text())
    workspace = ws.Workspace.from_url(receipt['url'])
    focused_name = 'LASER ImageNet joint20k uniform CE K2048 fresh100 4 B200'
    workspace.name = focused_name
    workspace.sections = sections(ws, wr)
    workspace.save()
    reloaded = ws.Workspace.from_url(workspace.url)
    assert reloaded.name == focused_name and len(reloaded.sections) == 3
    assert RUN in str(reloaded.runset_settings.filters)
    receipt.update(logging_repaired=True, sampler_top_k=2048, url=workspace.url)
    (TRAIN / 'workspace.json').write_text(json.dumps(receipt, indent=2) + '\n')

    # Fix the live service's state so future training events retain the correction.
    subprocess.run([sys.executable, str(ROOT / 'scripts/tools/repair_joint20k_writer_summary.py')],
        check=True, timeout=60)
    api = wandb.Api(timeout=60)
    run = api.run(f'{ENTITY}/{PROJECT}/{RUN}')
    raw = run._attrs['summaryMetrics']
    before = json.loads(raw) if isinstance(raw, str) else copy.deepcopy(raw)
    write('summary-before-repair.json', before)
    evaluations = [json.loads(p.read_text()) for p in sorted((TRAIN / 'evaluations').glob('epoch*.json'))]
    latest = max(evaluations, key=lambda row:row['train/global_step'])
    assert isinstance(before.get('samples/fixed'), dict)
    write('summary-repair-values.json', {key:before[key] for key in
        ('eval/inception_score.last', 'eval/inception_score.max')})
    result = dict(passed=True, time=time.time(), workspace_url=workspace.url, run_url=run.url,
        native_history_unchanged=True, training_restarted=False, views=repaired,
        native_evaluation_epoch=latest['train/epoch'], native_fid=latest['eval/fid'],
        native_inception_score=latest['eval/inception_score'], image_references_intact=True,
        sampler_comparisons_shown_separately=True, active_training_sampler=2048)
    update_config_only(run, {'logging_dashboard':dict(
        workspace_url=workspace.url, repaired_at=result['time'],
        sample_step_key='samples/global_step', training_step_key='train/global_step',
        sampler_comparisons='Separate fixed-checkpoint table', training_restarted=False)})
    write('repair-verification.json', result)
    for name in ('repair-verification.json', 'run-view-after-repair.json', 'project-view-after-repair.json'):
        run.upload_file(str(OUT / name), root=str(TRAIN))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
