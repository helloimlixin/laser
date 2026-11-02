#!/usr/bin/env python3
"""Create a visible FID workspace for the original RQ-Transformer trial."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import wandb
import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2 as wr

ROOT=Path('/workspace/Projects/laser')
RUN='imagenet-original-rq1400m-rfid473-scratch20-4h100-20261014'
OUT=ROOT/'outputs'/RUN
spec=importlib.util.spec_from_file_location('original_rq_existing_workspace',ROOT/'scripts/tools/fix_imagenet_fid_workspace.py')
helper=importlib.util.module_from_spec(spec);sys.modules[spec.name]=helper;spec.loader.exec_module(helper)


def main():
    ui=OUT/'workspace';ui.mkdir(exist_ok=True)
    api=wandb.Api(timeout=60)
    run=api.run(f'helloimlixin-rutgers/laser/{RUN}')
    assert run.state=='running' and run.config['parameter_count']==1387529728
    assert run.config['fid_num_samples']==50000 and run.config['fid_reference_images']==1281167
    # Put the existing FID section first while preserving all panels and settings.
    views=[]
    for kind in ('run-view','project-view'):
        original=helper.read_view(kind)
        (ui/f'{kind}-before.json').write_text(json.dumps(original,indent=2))
        specification=copy.deepcopy(json.loads(original['spec']))
        bank=specification['panelBankConfig'] if kind=='run-view' else specification['section']['panelBankConfig']
        sections=bank['sections']
        fid=[s for s in sections if s.get('__id__')=='laser-visible-generation-fid']
        assert len(fid)==1
        preserved=[s for s in sections if s.get('__id__')!='laser-visible-generation-fid']
        bank['sections']=[*fid,*preserved]
        helper.graphql('''mutation($input:UpsertViewInput!){upsertView(input:$input){view{id} inserted}}''',
            {'input':dict(id=original['id'],entityName=helper.ENTITY,projectName=helper.PROJECT,
                name=original['name'],displayName=original['displayName'],type=kind,
                spec=json.dumps(specification),lastUpdatedAt=original['updatedAt'],createdUsing='WANDB_SDK')})
        saved=helper.read_view(kind)
        value=json.loads(saved['spec'])
        saved_bank=value['panelBankConfig'] if kind=='run-view' else value['section']['panelBankConfig']
        assert saved_bank['sections']==[*fid,*preserved]
        (ui/f'{kind}-after.json').write_text(json.dumps(saved,indent=2))
        views.append(dict(kind=kind,fid_first=True,existing_sections_preserved=len(sections)))
    workspace=ws.Workspace(entity=helper.ENTITY,project=helper.PROJECT,
        name='Original RQ-Transformer 1.4B: fresh 20-epoch trial',
        runset_settings=ws.RunsetSettings(filters=[ws.Config('wandb_id')==RUN]),
        sections=[ws.Section(name='FID',is_open=True,pinned=True,panels=[
            wr.MarkdownPanel(markdown=(
                '**Original RQ-Transformer 1.4B, fresh stage 2, original RQ-VAE tokenizer.** '
                'Tokenizer rFID measured at **4.7235** on all 50,000 ImageNet validation images '
                '(published: 4.73). **Generation FID:** 50,000 generated images against the full '
                '1,281,167-image training reference, original RQ-VAE Inception, no rejection. '
                'First generation FID after epoch 1; then epochs 2, 4, …, 20. '
                'Training executes the first 20 epochs of the original 100-epoch cosine schedule.')),
            wr.LinePlot(title='Generated-image FID50k vs full ImageNet train',x='train/epoch',
                y=['eval/fid'],title_x='Completed epoch',title_y='FID',smoothing_type='none',smoothing_factor=0),
            wr.LinePlot(title='Generated-image FID50k by optimizer update',x='train/global_step',
                y=['eval/fid'],title_x='Optimizer update',title_y='FID',smoothing_type='none',smoothing_factor=0),
            wr.LinePlot(title='Native stochastic RQ soft-target training loss',x='train/global_step',
                y=['train/loss'],smoothing_type='none',smoothing_factor=0),
            wr.LinePlot(title='Published learning rate schedule',x='train/global_step',
                y=['train/lr'],smoothing_type='none',smoothing_factor=0),
        ])]).save()
    loaded=ws.Workspace.from_url(workspace.url)
    assert loaded.sections[0].is_open and len(loaded.sections[0].panels)==5
    receipt=dict(verified=True,run=run.url,workspace_url=workspace.url,views=views)
    (ui/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    run.upload_file(str(ui/'verification.json'),root=str(OUT))
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
