#!/usr/bin/env python3
"""Consolidate FID summaries and panels without changing measurement history."""
import copy
import json
import os
from pathlib import Path
import time

ROOT=Path('/workspace/Projects/laser')
OUT=ROOT/'outputs/imagenet-compound512-atom-identity-pilot-20261016'
AUDIT=OUT/'fid-logging-cleanup'
ENTITY='helloimlixin-rutgers'
PROJECT='laser'
ALIASES={'eval/fid','eval/fid.last','eval/fid.min','pilot/fid_control','pilot/fid_candidate'}


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    import requests
    import wandb
    import wandb_workspaces.workspaces as ws
    key=Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
    os.environ['WANDB_API_KEY']=key
    AUDIT.mkdir(exist_ok=True)
    api=wandb.Api(timeout=30)
    results={}
    for branch in ['control','atom_identity']:
        run_id=json.loads((OUT/branch/'wandb.json').read_text())['id']
        run=api.run(f'{ENTITY}/{PROJECT}/{run_id}')
        assert run.state=='finished'
        before=copy.deepcopy(run.summary._json_dict)
        path=AUDIT/(branch+'-summary-before.json')
        if not path.exists():write(path,before)
        def fid_points(history):
            return [{k:row[k] for k in ['_step','train/global_step','eval/fid']}
                    for row in history if row.get('eval/fid') is not None]
        points=fid_points(run.history(samples=100,pandas=False))
        assert len(points)==1 and points[0]['train/global_step']==1380
        fid=json.loads((OUT/branch/'fid50k.json').read_text())['fid']
        assert abs(points[0]['eval/fid']-fid)<1.e-10
        removed=[]
        for name in sorted(ALIASES-{'eval/fid'}):
            if name in run.summary:del run.summary[name];removed.append(name)
        if branch=='control' and 'pilot/fid_candidate_minus_control' in run.summary:
            del run.summary['pilot/fid_candidate_minus_control'];removed.append('pilot/fid_candidate_minus_control')
        run.summary.update({'eval/fid':fid})
        check=wandb.Api(timeout=30).run(f'{ENTITY}/{PROJECT}/{run_id}')
        after=copy.deepcopy(check.summary._json_dict)
        assert not (set(after)&(ALIASES-{'eval/fid'}))
        assert abs(after['eval/fid']-fid)<1.e-10
        assert check.state=='finished' and fid_points(check.history(samples=100,pandas=False))==points
        assert {k:v for k,v in before.items() if k not in removed and k!='eval/fid'}=={k:v for k,v in after.items() if k not in removed and k!='eval/fid'}
        write(AUDIT/(branch+'-summary-after.json'),after)
        results[branch]=dict(fid=fid,canonical_metric='eval/fid',removed_summary_fields=removed,
            fid_history_points=len(points),history_unchanged=True,state=check.state)

    def graphql(query,variables):
        response=requests.post('https://api.wandb.ai/graphql',auth=('api',key),
            json={'query':query,'variables':variables},timeout=30)
        response.raise_for_status();body=response.json()
        if body.get('errors'):raise RuntimeError(body['errors'])
        return body['data']
    def view(kind):
        query='''query($type:String!){project(name:"laser",entityName:"helloimlixin-rutgers"){
            allViews(viewType:$type,userName:"helloimlixin-rutgers",first:100){edges{node{
            id name type displayName spec updatedAt}}}}}'''
        nodes=graphql(query,{'type':kind})['project']['allViews']['edges']
        return next(edge['node'] for edge in nodes if edge['node']['name']=='nw-nwuserhelloimlixinrutgers-w')
    def bank(spec,kind):return spec['panelBankConfig'] if kind=='run-view' else spec['section']['panelBankConfig']
    def is_alias_panel(panel):
        metrics=panel.get('config',{}).get('metrics',[])
        return bool(metrics) and set(metrics).issubset(ALIASES)
    views=[]
    for kind in ['run-view','project-view']:
        original=view(kind);spec=json.loads(original['spec']);prior=copy.deepcopy(spec)
        write(AUDIT/(kind+'-apply-before.json'),original)
        current=bank(spec,kind)
        panels=[p for s in current['sections'] for p in s.get('panels',[]) if is_alias_panel(p)]
        preferred=next(p for p in panels if p.get('config',{}).get('xAxis')=='train/global_step')
        keep_id=preferred['__id__'];removed=[]
        for section in current['sections']:
            kept=[]
            for panel in section.get('panels',[]):
                if is_alias_panel(panel) and panel['__id__']!=keep_id:removed.append(panel['__id__'])
                else:kept.append(panel)
            section['panels']=kept
        preferred['config'].update(metrics=['eval/fid'],xAxis='train/global_step',
            chartTitle='FID50k vs full ImageNet training set',xAxisTitle='Optimizer update')
        preferred['isAuto']=False
        graphql('''mutation($input:UpsertViewInput!){upsertView(input:$input){
            view{id name type spec updatedAt} inserted}}''',{'input':dict(id=original['id'],
            entityName=ENTITY,projectName=PROJECT,name=original['name'],displayName=original['displayName'],
            type=kind,spec=json.dumps(spec),lastUpdatedAt=original['updatedAt'],createdUsing='WANDB_SDK')})
        saved=view(kind);after=json.loads(saved['spec'])
        assert after==spec
        remaining=[p for s in bank(after,kind)['sections'] for p in s.get('panels',[]) if is_alias_panel(p)]
        assert len(remaining)==1 and remaining[0]['config']['metrics']==['eval/fid']
        # Every panel for a different measurement remains byte-for-byte intact.
        unaffected=lambda value:[p for s in bank(value,kind)['sections'] for p in s.get('panels',[]) if not is_alias_panel(p)]
        assert unaffected(prior)==unaffected(after)
        write(AUDIT/(kind+'-after.json'),saved)
        views.append(dict(kind=kind,canonical_fid_panels=1,removed_duplicate_panels=len(removed),
            unrelated_panels_preserved=True,verified_by_readback=True))
    focused=ws.Workspace.from_url(json.loads((OUT/'workspace.json').read_text())['url'])
    model=focused._to_model().model_dump(by_alias=True,exclude_none=True)
    focused_panels=[p for s in model['spec']['section']['panelBankConfig']['sections'] for p in s.get('panels',[]) if is_alias_panel(p)]
    assert len(focused_panels)==1
    result=dict(passed=True,runs=results,personal_views=views,focused_workspace_fid_panels=1,
        no_fid_evaluations_repeated=True,time=time.time())
    write(AUDIT/'verification.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
