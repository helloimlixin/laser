#!/usr/bin/env python3
"""Show the continuation's latest sample grid in its existing W&B views."""
import copy
import json
from pathlib import Path
import time

import requests
import wandb_workspaces.reports.v2 as wr

from deduplicate_fid14_workspace import bank, ENTITY, PROJECT, RUN

OUT = Path('/workspace/Projects/laser/outputs')/RUN/'samples/workspace'


def main():
    key = Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
    OUT.mkdir(parents=True, exist_ok=True)

    def graphql(query, variables):
        result = requests.post('https://api.wandb.ai/graphql', auth=('api', key),
            json={'query': query, 'variables': variables}, timeout=60)
        result.raise_for_status()
        value = result.json()
        if value.get('errors'):
            raise RuntimeError(value['errors'])
        return value['data']

    query = '''query($type:String!){project(name:"laser",entityName:"helloimlixin-rutgers"){
        allViews(viewType:$type,userName:"helloimlixin-rutgers",first:100){edges{node{
        id name type displayName spec updatedAt}}}}}'''

    def views(kind):
        return [e['node'] for e in graphql(query, {'type': kind})['project']['allViews']['edges']]

    receipts = []
    section_id = 'laser-fid14-sample-preview'
    for kind in ('run-view', 'project-view'):
        names = ['nw-nwuserhelloimlixinrutgers-w']
        if kind == 'project-view':
            names.append('nw-ampnidmo8t6-v')
        for name in names:
            original = next(v for v in views(kind) if v['name'] == name)
            before_path = OUT/f'{kind}-{name}-before.json'
            if not before_path.exists():
                before_path.write_text(json.dumps(original, indent=2)+'\n')
            spec = json.loads(original['spec'])
            original_sections = copy.deepcopy(bank(spec, kind)['sections'])
            unrelated = [s for s in original_sections if s.get('__id__') != section_id]
            section = copy.deepcopy(unrelated[0])
            panel = wr.MediaBrowser(title='Generated samples — latest completed epoch',
                media_keys=['samples/continuation'], num_columns=1)._to_model().model_dump(
                    by_alias=True, exclude_none=True)
            panel.update(__id__='laser-fid14-preview-image', isAuto=False)
            section.update(name='Samples', __id__=section_id, isOpen=True, pinned=True,
                isPanelsAuto=False, panels=[panel])
            bank(spec, kind)['sections'] = [*unrelated[:1], section, *unrelated[1:]]
            graphql('''mutation($input:UpsertViewInput!){upsertView(input:$input){view{id} inserted}}''',
                {'input': dict(id=original['id'], entityName=ENTITY, projectName=PROJECT,
                    name=name, displayName=original['displayName'], type=kind,
                    spec=json.dumps(spec), lastUpdatedAt=original['updatedAt'], createdUsing='WANDB_SDK')})
            saved = next(v for v in views(kind) if v['name'] == name)
            sections = bank(json.loads(saved['spec']), kind)['sections']
            assert [s for s in sections if s.get('__id__') != section_id] == unrelated
            added = [s for s in sections if s.get('__id__') == section_id]
            assert len(added) == 1 and added[0]['panels'][0]['config']['mediaKeys'] == ['samples/continuation']
            (OUT/f'{kind}-{name}-after.json').write_text(json.dumps(saved, indent=2)+'\n')
            receipts.append(dict(kind=kind, name=name, sample_panels=1, unrelated_sections_unchanged=True))
    result = dict(passed=True, views=receipts, time=time.time())
    (OUT/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
