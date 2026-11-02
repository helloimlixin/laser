#!/usr/bin/env python3
"""Keep one canonical FID chart in the actual default W&B workspaces."""
import copy
import json
import os
from pathlib import Path
import time

ENTITY='helloimlixin-rutgers'
PROJECT='laser'
RUN='imagenet-fid14-continue100-4h100-20261016'
OUT=Path('/workspace/Projects/laser/outputs')/RUN/'fid-logging-cleanup'
ALIASES={'eval/fid','eval/fid.last','eval/fid.min'}


def canonical(panel):
    values=panel.get('config',{}).get('metrics',[])
    return bool(values) and set(values).issubset(ALIASES)


def bank(spec,kind):
    return spec['panelBankConfig'] if kind=='run-view' else spec['section']['panelBankConfig']


def main():
    import requests
    import wandb
    key=Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
    os.environ['WANDB_API_KEY']=key
    OUT.mkdir(exist_ok=True)
    def graphql(query,variables):
        response=requests.post('https://api.wandb.ai/graphql',auth=('api',key),
            json={'query':query,'variables':variables},timeout=60)
        response.raise_for_status();body=response.json()
        if body.get('errors'):raise RuntimeError(body['errors'])
        return body['data']
    query='''query($type:String!){project(name:"laser",entityName:"helloimlixin-rutgers"){
        allViews(viewType:$type,userName:"helloimlixin-rutgers",first:100){edges{node{
        id name type displayName spec updatedAt}}}}}'''
    def views(kind):
        return [edge['node'] for edge in graphql(query,{'type':kind})['project']['allViews']['edges']]
    receipts=[]
    for kind in ('run-view','project-view'):
        names=['nw-nwuserhelloimlixinrutgers-w']
        if kind=='project-view':names.append('nw-ampnidmo8t6-v')
        for name in names:
            original=next(node for node in views(kind) if node['name']==name)
            target=OUT/(kind+'-'+name+'-apply-before.json')
            if not target.exists():target.write_text(json.dumps(original,indent=2)+'\n')
            spec=json.loads(original['spec']);before=copy.deepcopy(spec)
            current=bank(spec,kind)
            candidates=[(section,panel) for section in current['sections']
                for panel in section.get('panels',[]) if canonical(panel)]
            assert candidates,'Missing canonical FID panel'
            preferred=next((p for s,p in candidates if s.get('name')!='Hidden Panels'
                and p.get('config',{}).get('xAxis')=='train/global_step'),candidates[0][1])
            keep=preferred['__id__'];removed=[]
            for section in current['sections']:
                panels=[]
                for panel in section.get('panels',[]):
                    if canonical(panel) and panel['__id__']!=keep:removed.append(panel['__id__'])
                    else:panels.append(panel)
                section['panels']=panels
            preferred.setdefault('config',{}).update(metrics=['eval/fid'],xAxis='train/global_step',
                chartTitle='FID50k vs full ImageNet training set',xAxisTitle='Optimizer update')
            preferred['isAuto']=False
            for section in current['sections']:
                if any(p.get('__id__')==keep for p in section.get('panels',[])):
                    section['isPanelsAuto']=False
            if spec!=before:
                graphql('''mutation($input:UpsertViewInput!){upsertView(input:$input){
                    view{id name type spec updatedAt} inserted}}''',{'input':dict(id=original['id'],
                    entityName=ENTITY,projectName=PROJECT,name=name,displayName=original['displayName'],
                    type=kind,spec=json.dumps(spec),lastUpdatedAt=original['updatedAt'],createdUsing='WANDB_SDK')})
            saved=next(node for node in views(kind) if node['name']==name)
            after=json.loads(saved['spec']);assert after==spec
            remaining=[p for s in bank(after,kind)['sections'] for p in s.get('panels',[]) if canonical(p)]
            assert len(remaining)==1 and remaining[0]['config']['metrics']==['eval/fid']
            unrelated=lambda value:[p for s in bank(value,kind)['sections']
                for p in s.get('panels',[]) if not canonical(p)]
            assert unrelated(before)==unrelated(after)
            (OUT/(kind+'-'+name+'-after.json')).write_text(json.dumps(saved,indent=2)+'\n')
            receipts.append(dict(kind=kind,name=name,canonical_fid_panels=1,
                removed_duplicate_panels=len(removed),unrelated_panels_unchanged=True))
    run=wandb.Api(timeout=60).run(f'{ENTITY}/{PROJECT}/{RUN}')
    points=list(run.scan_history(keys=['train/global_step','eval/fid'],page_size=1000))
    assert len(points)==len({row['train/global_step'] for row in points})
    for row in points:
        record=json.loads((OUT.parent/'evaluations'/f"step{row['train/global_step']}.json").read_text())
        assert record['fid']==row['eval/fid']
    result=dict(passed=True,views=receipts,canonical_fid_history_points=len(points),
        once_per_optimizer_step=True,no_measurements_changed=True,training_restarted=False,time=time.time())
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
