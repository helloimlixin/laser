#!/usr/bin/env python3
"""Show the matched sampler evaluation in existing and focused W&B views."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import wandb
import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2 as wr

ROOT = Path('/workspace/Projects/laser')
PARENT = 'imagenet-rfid421-tiny-jitter-fresh-4h100-20261014'
EVAL = 'imagenet-tiny-jitter-epoch8-ancestral-joint32-fid50k-20261014'
OUT = ROOT / 'outputs' / PARENT / 'joint-fid50k-epoch8'
UI = OUT / 'workspace'
SECTION_ID = 'laser-epoch8-sampler-comparison'
spec = importlib.util.spec_from_file_location('laser_existing_fid_workspace',
    ROOT / 'scripts/tools/fix_imagenet_fid_workspace.py')
helper = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helper
spec.loader.exec_module(helper)


def panels():
    return [
        wr.BarPlot(title='Epoch 8: ancestral vs joint32 — FID50k',
                   metrics=['eval/ancestral_fid', 'eval/joint32_t1_fid'],
                   orientation='v', title_y='FID (lower is better)'),
        wr.LinePlot(title='Sampler evaluation: images completed',
                    x='evaluation/generated_images',
                    y=['progress/ancestral_images', 'progress/joint32_t1_images'],
                    smoothing_type='none', smoothing_factor=0,
                    title_y='Images per sampler', range_y=(0,50000)),
    ]


def update_view(view_type):
    view = helper.read_view(view_type)
    (UI / f'{view_type}-before.json').write_text(json.dumps(view,indent=2))
    specification = copy.deepcopy(json.loads(view['spec']))
    bank = specification['panelBankConfig'] if view_type == 'run-view' else specification['section']['panelBankConfig']
    preserved = [s for s in bank['sections'] if s.get('__id__') != SECTION_ID]
    section = ws.Section(name='Sampler comparison — epoch 8',is_open=True,pinned=True,panels=panels())
    section = section._to_model().model_dump(by_alias=True,exclude_none=True)
    section['__id__'] = SECTION_ID
    section['isPanelsAuto'] = False
    for panel in section['panels']: panel['isAuto'] = False
    bank['sections'] = [section,*preserved]
    helper.graphql('''mutation($input:UpsertViewInput!){upsertView(input:$input){view{id} inserted}}''',
        {'input':dict(id=view['id'],entityName=helper.ENTITY,projectName=helper.PROJECT,
            name=view['name'],displayName=view['displayName'],type=view_type,
            spec=json.dumps(specification),lastUpdatedAt=view['updatedAt'],createdUsing='WANDB_SDK')})
    saved = helper.read_view(view_type)
    result = json.loads(saved['spec'])
    result_bank = result['panelBankConfig'] if view_type == 'run-view' else result['section']['panelBankConfig']
    assert result_bank['sections'][0]['__id__'] == SECTION_ID
    assert result_bank['sections'][0]['isOpen'] and result_bank['sections'][0]['pinned']
    assert result_bank['sections'][1:] == preserved
    assert result_bank['sections'][0]['panels'][0]['config']['metrics'] == ['eval/ancestral_fid','eval/joint32_t1_fid']
    (UI / f'{view_type}-after.json').write_text(json.dumps(saved,indent=2))
    return dict(view_type=view_type,verified=True,previous_sections_preserved=len(preserved))


def main():
    UI.mkdir(parents=True,exist_ok=True)
    api = wandb.Api(timeout=60)
    run = api.run(f'{helper.ENTITY}/{helper.PROJECT}/{EVAL}')
    assert run.config['generated_images_per_sampler'] == 50000
    assert run.config['training_reference_images'] == 1281167
    assert run.config['checkpoint_step'] == 5008
    views = [update_view('run-view'),update_view('project-view')]
    workspace = ws.Workspace(entity=helper.ENTITY,project=helper.PROJECT,
        name='Epoch 8: ancestral vs joint32 — matched FID50k',
        runset_settings=ws.RunsetSettings(filters=[ws.Config('wandb_id') == EVAL]),
        sections=[ws.Section(name='Ancestral vs joint32',is_open=True,pinned=True,panels=[
            wr.MarkdownPanel(markdown=(
                '**Same epoch-8 checkpoint (update 5,008).** '
                '50,000 generated images per sampler, 50 per class, against the full '
                '1,281,167-image ImageNet training reference. Original RQ-VAE Inception; '
                'no rejection sampling. Temperature 1 for atoms and coefficients in both arms. '
                '**Final FID appears only after all 50,000 images for that sampler are measured.**')),
            *panels(),
        ])]).save()
    loaded = ws.Workspace.from_url(workspace.url)
    assert len(loaded.sections[0].panels) == 3 and loaded.sections[0].is_open
    receipt = dict(verified=True,wandb_run=run.url,workspace_url=workspace.url,views=views)
    (UI/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    run.upload_file(str(UI/'verification.json'),root=str(OUT))
    run.upload_file(str(OUT/'smoke-verification.json'),root=str(OUT))
    print(json.dumps(receipt),flush=True)


if __name__ == '__main__': main()
