#!/usr/bin/env python3
"""Add visible FID charts to existing W&B workspaces without replacing layouts."""
import copy
import json
import os
from pathlib import Path
import time

import requests
import wandb
import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2 as wr

ENTITY = 'helloimlixin-rutgers'
PROJECT = 'laser'
RUN = 'imagenet-rfid421-tiny-jitter-fresh-4h100-20261014'
OUT = Path('/workspace/Projects/laser/outputs') / RUN / 'fid-workspace'
SECTION_ID = 'laser-visible-generation-fid'
KEY = Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
os.environ['WANDB_API_KEY'] = KEY


def graphql(query, variables=None):
    response = requests.post('https://api.wandb.ai/graphql', auth=('api', KEY),
                             json={'query': query, 'variables': variables or {}}, timeout=90)
    response.raise_for_status()
    result = response.json()
    if result.get('errors'): raise RuntimeError(json.dumps(result['errors']))
    return result['data']


def read_view(view_type):
    result = graphql('''query($type:String!){project(name:"laser",entityName:"helloimlixin-rutgers"){
        allViews(viewType:$type,userName:"helloimlixin-rutgers",first:100){edges{node{
        id name type displayName spec updatedAt}}}}}''', {'type': view_type})
    views = [edge['node'] for edge in result['project']['allViews']['edges']]
    return next(view for view in views if view['name'] == 'nw-nwuserhelloimlixinrutgers-w')


def charts():
    section = ws.Section(name='FID', is_open=True, pinned=True, panels=[
        wr.LinePlot(title='FID (50,000 generated images) by optimizer update',
                    x='train/global_step', y=['eval/fid'], title_x='Optimizer update',
                    title_y='FID', smoothing_type='none', smoothing_factor=0),
    ])
    model = section._to_model().model_dump(by_alias=True, exclude_none=True)
    model['__id__'] = SECTION_ID
    model['isPanelsAuto'] = False
    for panel in model['panels']: panel['isAuto'] = False
    return model


def update_personal_view(view_type):
    original = read_view(view_type)
    before = json.loads(original['spec'])
    (OUT / f'{view_type}-before.json').write_text(json.dumps(original, indent=2))
    spec = copy.deepcopy(before)
    bank = spec['panelBankConfig'] if view_type == 'run-view' else spec['section']['panelBankConfig']
    preserved = [section for section in bank['sections'] if section.get('__id__') != SECTION_ID]
    bank['sections'] = [charts(), *preserved]
    bank.setdefault('settings', {})['searchQuery'] = ''
    # The mutation uses the existing view's ID/name/type. All other saved UI
    # state, custom charts, and selected runs pass through unchanged.
    result = graphql('''mutation($input:UpsertViewInput!){upsertView(input:$input){
        view{id name type spec updatedAt} inserted}}''', {'input': dict(
        id=original['id'], entityName=ENTITY, projectName=PROJECT,
        name=original['name'], displayName=original['displayName'], type=view_type,
        spec=json.dumps(spec), lastUpdatedAt=original['updatedAt'], createdUsing='WANDB_SDK')})
    saved = read_view(view_type)
    after = json.loads(saved['spec'])
    saved_bank = after['panelBankConfig'] if view_type == 'run-view' else after['section']['panelBankConfig']
    assert saved_bank['sections'][0]['name'] == 'FID'
    assert saved_bank['sections'][0]['isOpen'] and saved_bank['sections'][0]['pinned']
    assert [section for section in saved_bank['sections'] if section.get('__id__') != SECTION_ID] == preserved
    assert all(panel['config']['metrics'] == ['eval/fid']
               for panel in saved_bank['sections'][0]['panels'])
    (OUT / f'{view_type}-after.json').write_text(json.dumps(saved, indent=2))
    return dict(type=view_type, id=saved['id'], section='FID',
                original_sections_preserved=len(preserved), charts=1,
                chart_metrics=['eval/fid'], verified_by_readback=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=60)
    run = api.run(f'{ENTITY}/{PROJECT}/{RUN}')
    assert run.config['fid_num_samples'] == 50000
    assert run.config['fid_real_split'] == 'train'
    assert run.config['metric_backend'] == 'original-rqvae'
    assert Path(run.config['fid_reference_stats']).name == 'imagenet_256_train.npz'
    points = list(run.scan_history(keys=['train/epoch', 'train/global_step', 'eval/fid'], page_size=1000))
    assert len(points) >= 5
    result = dict(passed=True, run=run.url, checked_at=time.time(),
                  generated_images=50000, training_reference_images=1281167,
                  reference='published ImageNet-256 training statistics',
                  fid_reference_sha256=run.summary['evaluation/fid_reference_sha256'],
                  completed_fid_points=len(points), latest_point=points[-1],
                  views=[update_personal_view('run-view'), update_personal_view('project-view')])
    # Also create a focused saved workspace with an explicit run filter and
    # protocol label, so the user has a direct link independent of UI state.
    workspace = ws.Workspace(entity=ENTITY, project=PROJECT,
        name='ImageNet tiny jitter: FID50k against full training set',
        runset_settings=ws.RunsetSettings(filters=[ws.Config('wandb_id') == RUN]),
        sections=[ws.Section(name='FID — 50k generated vs full ImageNet training set',
            is_open=True, pinned=True, panels=[
                wr.MarkdownPanel(markdown=(
                    '**FID protocol:** 50,000 generated images (50 per ImageNet class) '
                    'against the published ImageNet-256 training statistics for the full '
                    '1,281,167-image training population. Original RQ-VAE Inception; '
                    'no rejection sampling. Charts update as evaluations finish.')),
                wr.LinePlot(title='FID50k vs full ImageNet train by update',
                    x='train/global_step', y=['eval/fid'], title_x='Optimizer update',
                    title_y='FID', smoothing_type='none', smoothing_factor=0),
            ])]).save()
    loaded = ws.Workspace.from_url(workspace.url)
    assert len(loaded.sections[0].panels) == 2 and loaded.sections[0].is_open
    result['focused_workspace_url'] = workspace.url
    result['focused_workspace_filter'] = loaded.runset_settings.filters
    (OUT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    run.upload_file(str(OUT / 'verification.json'), root=str(OUT.parent))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
