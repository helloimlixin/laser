#!/usr/bin/env python3
"""Expose the live CC3M run's existing telemetry in W&B without touching training."""
import copy
import json
import os
from pathlib import Path
import secrets
import time


ENTITY = 'helloimlixin-rutgers'
PROJECT = 'laser'
RUN = 'cc3m-laser-joint20k-depth5-rqnoise-k2048-rngfix-scratch100-8a100-20261017'
BASE = Path('/tmp/laser-cc3m-joint20k-depth5-rngfix-8a100-20261017')
OUT = BASE/'recovery/logging-visibility'
DEFAULT = 'nw-nwuserhelloimlixinrutgers-w'
PREFIX = 'laser-cc3m-joint20k-logging-'


def write(name, data):
    path = OUT/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2)+'\n')


def bank(spec, kind):
    return spec['panelBankConfig'] if kind == 'run-view' else spec['section']['panelBankConfig']


def panel(name, title, metrics=None, media=None):
    config = dict(chartTitle=title)
    if media:
        config.update(mediaKeys=[media], columnCount=1)
    else:
        config.update(metrics=metrics, xAxis='train/global_step',
                      smoothingType='none', smoothingWeight=0, showMarkers=True)
    return dict(__id__=PREFIX+name, viewType='Media Browser' if media else 'Run History Line Plot',
                isAuto=False, layout=dict(x=0, y=0, w=12, h=6), config=config)


def sections(prototype):
    definitions = [
        ('metrics', 'CC3M training and validation', [
            panel('loss', 'Training loss (0.9 image + 0.1 text)', ['train/loss']),
            panel('components', 'Image and caption cross entropy', ['train/image_loss', 'train/text_loss']),
            panel('fid', 'Generation FID on the 13,443-image CC3M validation subset', ['eval/fid']),
            panel('clip', 'CLIP text-image alignment', ['eval/clip_score']),
            panel('lr', 'Learning rate', ['train/lr']),
            panel('gradient', 'Gradient norm', ['train/gradient_norm']),
            panel('speed', 'Training images per second', ['train/images_per_second']),
            panel('memory', 'Peak GPU memory (GiB)', ['train/peak_gib']),
        ]),
        ('samples', 'CC3M generated samples: nine prompts, eight images each', [
            panel('grid', 'Text-to-image grid: latest logged step', media='samples/text_grid_9x8'),
        ]),
    ]
    result = []
    for identity, title, panels in definitions:
        section = copy.deepcopy(prototype)
        section.update(__id__=PREFIX+identity, name=title, isOpen=True, pinned=True,
                       isPanelsAuto=False, panels=panels)
        section.setdefault('flowConfig', {}).update(columnsPerPage=2, rowsPerPage=2)
        result.append(section)
    return result


def main():
    os.environ['WANDB_API_KEY'] = Path('/root/.config/laser/wandb.key').read_text().strip()
    import wandb
    api = wandb.Api(timeout=45)
    graphql = api._service_api.execute_graphql
    query = '''query($type:String!){project(name:"laser",entityName:"helloimlixin-rutgers"){
        allViews(viewType:$type,userName:"helloimlixin-rutgers",first:100){edges{node{
        id name type displayName spec updatedAt}}}}}'''

    def views(kind):
        return [e['node'] for e in graphql(query, {'type': kind})['project']['allViews']['edges']]

    def save(original, spec, kind):
        value = dict(entityName=ENTITY, projectName=PROJECT, name=original['name'],
            displayName=original['displayName'], type=kind, spec=json.dumps(spec), createdUsing='WANDB_SDK')
        if original.get('id'):
            value.update(id=original['id'], lastUpdatedAt=original['updatedAt'])
        result = graphql('''mutation($input:UpsertViewInput!){upsertView(input:$input){
            view{id name type displayName spec updatedAt} inserted}}''', {'input': value})
        return result['upsertView']['view']

    run = api.run(f'{ENTITY}/{PROJECT}/{RUN}')
    assert run.state == 'running'
    # Query each event family independently: training, eval, and media are
    # intentionally separate history rows and cannot be joined by row index.
    evaluations = list(run.scan_history(keys=['train/global_step', 'eval/epoch', 'eval/fid', 'eval/clip_score']))
    samples = list(run.scan_history(keys=['train/global_step', 'samples/text_grid_9x8']))
    assert evaluations and samples
    for row in evaluations:
        local = json.loads((BASE/f'live/train/eval-epoch-{int(row["eval/epoch"]):03d}.json').read_text())
        assert row['eval/fid'] == local['fid'] and row['eval/clip_score'] == local['clip_score']
    published = {f.name: f for f in run.files()}
    latest = samples[-1]
    image = latest['samples/text_grid_9x8']
    assert image['path'] in published and published[image['path']].size == image['size']
    assert image['width'] > 0 and image['height'] > 0
    expected_metrics = {'train/loss', 'train/image_loss', 'train/text_loss', 'eval/fid', 'eval/clip_score',
                        'train/lr', 'train/gradient_norm', 'train/images_per_second', 'train/peak_gib'}
    assert expected_metrics <= set(run.summary.keys())
    receipts = []
    for kind in ('run-view', 'project-view'):
        original = next(v for v in views(kind) if v['name'] == DEFAULT)
        write(f'{kind}-before-repair.json', original)
        before = json.loads(original['spec'])
        spec = copy.deepcopy(before)
        current = bank(spec, kind)
        preserved = [s for s in current['sections'] if not s.get('__id__', '').startswith(PREFIX)]
        current['sections'] = sections(current['sections'][0])+preserved
        save(original, spec, kind)
        saved = next(v for v in views(kind) if v['name'] == DEFAULT)
        saved_spec = json.loads(saved['spec'])
        assert saved_spec == spec
        assert [s for s in bank(saved_spec, kind)['sections']
                if not s.get('__id__', '').startswith(PREFIX)] == preserved
        write(f'{kind}-after-repair.json', saved)
        receipts.append(dict(kind=kind, id=saved['id'], existing_sections_preserved=len(preserved),
                             added_panels=9, verified_by_readback=True))

    # A focused view also avoids unrelated selected runs and old ImageNet panels.
    all_project = views('project-view')
    state_path = OUT/'focused-workspace.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        state = dict(name='nw-'+secrets.token_hex(6)+'-v',
                     displayName='CC3M joint20k RNG fixed: live metrics and samples')
        write(state_path.name, state)
    existing = next((v for v in all_project if v['name'] == state['name']), state)
    default = next(v for v in all_project if v['name'] == DEFAULT)
    spec = json.loads(default['spec'])
    current = spec['section']
    current['panelBankConfig']['sections'] = sections(current['panelBankConfig']['sections'][0])
    current['settings']['maxRuns'] = 1
    runset = current['runSets'][0]
    current['runSets'] = [runset]
    runset['filters'] = dict(filterFormat='filterV2', filters=[dict(
        key=dict(section='run', name='name'), op='=', value=RUN, disabled=False)])
    runset['search'] = dict(query='')
    runset['selections'] = dict(root=1, bounds=[], tree=[])
    save(existing, spec, 'project-view')
    saved = next(v for v in views('project-view') if v['name'] == state['name'])
    assert json.loads(saved['spec']) == spec
    write('focused-workspace-after.json', saved)
    workspace_url = f'https://forge.coreweave.com/wandb/{ENTITY}/{PROJECT}?nw={state["name"][3:-2]}'
    receipt = dict(passed=True, run_url=run.url, workspace_url=workspace_url,
        run_state=run.state, evaluation_points=evaluations,
        latest_sample_step=latest['train/global_step'], latest_sample_uploaded=True,
        telemetry_summary_step=run.summary['train/global_step'], dashboard_views=receipts,
        correct_axes='train/global_step', correct_media_key='samples/text_grid_9x8',
        focused_run_filter=runset['filters'], training_restarted=False, time=time.time())
    write('verification.json', receipt)
    # Keep the repair receipt with the run's durable artifacts.
    local_receipt = BASE/'live/reproducibility/logging-workspace-verification.json'
    local_receipt.write_text(json.dumps(receipt, indent=2)+'\n')
    run.upload_file(str(local_receipt), root=str(BASE/'live'))
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
