#!/usr/bin/env python3
"""Publish clean W&B views only after this run's sample files are available."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
from urllib.parse import parse_qs, urlparse

import requests

ROOT = Path('/workspace/Projects/laser')
sys.path.insert(0, str(ROOT))
if os.environ.get('LASER_RQ_BASE'):
    sys.path.insert(0, str(Path(os.environ['LASER_RQ_BASE'])/'source'))
from src.training.rq_sample_panels import CLASSES, MEDIA_KEYS

ENTITY, PROJECT = 'helloimlixin-rutgers', 'laser'
DEFAULT = 'nw-nwuserhelloimlixinrutgers-w'


def bank(spec, kind):
    return spec['panelBankConfig'] if kind == 'run-view' else spec['section']['panelBankConfig']


def add_sample_sections(spec):
    import wandb_workspaces.reports.v2 as wr
    result = copy.deepcopy(spec)
    sections = bank(result, 'project-view')['sections']
    sections[:] = [s for s in sections if s.get('__id__') not in ('laser-fixed-overview', 'laser-class-grids')]
    template = copy.deepcopy(sections[0])

    def section(name, identity, titles, keys, columns):
        value = copy.deepcopy(template)
        panels = []
        for index, (title, key) in enumerate(zip(titles, keys, strict=True)):
            panel = wr.MediaBrowser(title=title, media_keys=[key], num_columns=1)._to_model().model_dump(
                by_alias=True, exclude_none=True)
            panel.update(__id__=f'{identity}-{index}', isAuto=False)
            panels.append(panel)
        value.update(name=name, __id__=identity, isOpen=True, pinned=True,
            isPanelsAuto=False, panels=panels)
        value['flowConfig'].update(columnsPerPage=columns, rowsPerPage=(len(keys)+columns-1)//columns)
        return value

    sections.append(section('Fixed classes — 10 images each', 'laser-fixed-overview',
        [' · '.join(name.title() for name, _ in CLASSES)], [MEDIA_KEYS[0]], 1))
    sections.append(section('Per-class samples — 8 × 8', 'laser-class-grids',
        [f'{name.title()} — 64 fixed-seed samples' for name, _ in CLASSES], MEDIA_KEYS[1:], 2))
    return result


class WorkspacePublisher:
    def __init__(self, run):
        self.run = run
        self.out = ROOT/'outputs'/run
        self.receipts = self.out/'samples/workspace'
        self.receipts.mkdir(parents=True, exist_ok=True)
        self.key = Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
        os.environ['WANDB_API_KEY'] = self.key

    def graphql(self, query, variables):
        response = requests.post('https://api.wandb.ai/graphql', auth=('api', self.key),
            json={'query':query, 'variables':variables}, timeout=45)
        response.raise_for_status()
        value = response.json()
        if value.get('errors'):
            raise RuntimeError(value['errors'])
        return value['data']

    def views(self, kind):
        query = '''query($type:String!){project(name:"laser",entityName:"helloimlixin-rutgers"){
            allViews(viewType:$type,userName:"helloimlixin-rutgers",first:100){edges{node{
            id name type displayName spec updatedAt}}}}}'''
        return [edge['node'] for edge in self.graphql(query, {'type':kind})['project']['allViews']['edges']]

    def save(self, original, spec):
        before = self.receipts/f"{original['type']}-{original['name']}-before.json"
        if not before.exists():
            before.write_text(json.dumps(original, indent=2)+'\n')
        self.graphql('''mutation($input:UpsertViewInput!){upsertView(input:$input){view{id} inserted}}''',
            {'input':dict(id=original['id'], entityName=ENTITY, projectName=PROJECT,
                name=original['name'], displayName=original['displayName'], type=original['type'],
                spec=json.dumps(spec), lastUpdatedAt=original['updatedAt'], createdUsing='WANDB_SDK')})
        saved = next(v for v in self.views(original['type']) if v['name'] == original['name'])
        assert json.loads(saved['spec']) == spec
        (self.receipts/f"{original['type']}-{original['name']}-after.json").write_text(json.dumps(saved, indent=2)+'\n')

    def configure(self, samples=False):
        workspace = json.loads((self.out/'workspace.json').read_text())
        suffix = parse_qs(urlparse(workspace['url']).query)['nw'][0]
        focused_name = f'nw-{suffix}-v'
        project_views = self.views('project-view')
        focused = next(v for v in project_views if v['name'] == focused_name)
        source = json.loads(focused['spec'])
        if samples:
            source = add_sample_sections(source)
        # This fresh run's workspace is the source of truth. Do not copy media
        # requirements or sampler-comparison charts from a different run.
        fid = [p for s in bank(source, 'project-view')['sections'] for p in s.get('panels', [])
               if 'eval/fid' in p.get('config', {}).get('metrics', [])]
        media = [p for s in bank(source, 'project-view')['sections'] for p in s.get('panels', [])
                 if p.get('viewType') == 'Media Browser']
        assert len(fid) == 1
        assert [p['config']['mediaKeys'] for p in media] == ([[key] for key in MEDIA_KEYS] if samples else [])
        self.save(focused, source)
        for kind in ('run-view', 'project-view'):
            default = next(v for v in self.views(kind) if v['name'] == DEFAULT)
            if kind == 'project-view':
                target = copy.deepcopy(source)
            else:
                target = json.loads(default['spec'])
                target['panelBankConfig'] = copy.deepcopy(bank(source, 'project-view'))
                target.setdefault('settings', {})['shouldAutoGeneratePanels'] = False
                target.setdefault('workspaceSettings', {})['shouldAutoGeneratePanels'] = False
            self.save(default, target)
        proof = dict(passed=True, canonical_fid_panels=1, sample_panels=len(media),
            media_keys=list(MEDIA_KEYS) if samples else [], url=workspace['url'], time=time.time())
        (self.receipts/('verified.json' if samples else 'initialized.json')).write_text(json.dumps(proof, indent=2)+'\n')
        workspace.update(canonical_fid_panels=1, sample_panels=len(media))
        (self.out/'workspace.json').write_text(json.dumps(workspace, indent=2)+'\n')
        print(json.dumps(proof), flush=True)

    def verify_images(self):
        import wandb
        run = wandb.Api(timeout=45).run(f'{ENTITY}/{PROJECT}/{self.run}')
        values = dict(run.summary)
        if not all(values.get(key, {}).get('path') for key in MEDIA_KEYS):
            return False
        paths = [values[key]['path'] for key in MEDIA_KEYS]
        files = {f.name:f for f in run.files(names=paths, per_page=100)}
        if not all(path in files and files[path].size > 0 for path in paths):
            return False
        points = list(run.scan_history(keys=['samples/global_step', *MEDIA_KEYS], page_size=100))
        if not points:
            return False
        proof = dict(passed=True, run=self.run, history_points=len(points),
            sample_steps=[p['samples/global_step'] for p in points],
            files=[dict(name=path, bytes=files[path].size) for path in paths], time=time.time())
        (self.receipts/'uploaded-images.json').write_text(json.dumps(proof, indent=2)+'\n')
        return True

    def watch(self):
        while not (self.out/'workspace.json').exists():
            time.sleep(10)
        self.configure(samples=False)
        while True:
            try:
                if self.verify_images():
                    break
            except Exception as error:
                # An upload/API delay must not leave the run permanently
                # without its panels. Keep retries visible in the watcher log.
                print(json.dumps(dict(state='waiting_for_upload',error=str(error),time=time.time())),flush=True)
            time.sleep(15)
        self.configure(samples=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    publisher = WorkspacePublisher(args.run)
    if args.watch:
        publisher.watch()
    else:
        assert publisher.verify_images(), 'Images must be uploaded and logged before panels are added'
        publisher.configure(samples=True)
