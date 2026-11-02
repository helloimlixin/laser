#!/usr/bin/env python3
"""Verify and expose completed sampler results without changing training."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import time
from urllib.parse import parse_qs, urlparse

ROOT = Path('/workspace/Projects/laser')
OUT = ROOT/'outputs/imagenet-laser-joint20k-depth5-epoch6-sampler-sweep-20261017'
ACTIVE = ROOT/'outputs/imagenet-laser-joint20k-depth5-scratch100-4h100-20261017'
LOCAL = Path('/tmp/laser-joint20k-depth5-sampler-sweep-20261017')


def main():
    assert json.loads((OUT/'final-verification.json').read_text())['passed']
    report = json.loads((OUT/'report.json').read_text())
    assert all(r['generated_images'] == 50000 and r['real_images'] == 1281167
               for r in report['confirmation'])
    os.environ['WANDB_API_KEY'] = Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
    import wandb
    import wandb_workspaces.reports.v2 as wr
    spec = importlib.util.spec_from_file_location('comparison_panels', ROOT/'scripts/tools/configure_rq_sample_panels.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    publisher = module.WorkspacePublisher(OUT.name)
    run = wandb.Api(timeout=45).run(f'helloimlixin-rutgers/laser/{OUT.name}')
    assert run.summary['selected_fid50k'] == report['selected_fid50k']
    paths = [run.summary[key]['path'] for key in module.MEDIA_KEYS]
    uploaded = {f.name:f for f in run.files(names=paths, per_page=100)}
    assert all(path in uploaded and uploaded[path].size > 0 for path in paths)
    workspace = json.loads((OUT/'workspace.json').read_text())
    suffix = parse_qs(urlparse(workspace['url']).query)['nw'][0]
    focused = next(v for v in publisher.views('project-view') if v['name'] == f'nw-{suffix}-v')
    value = json.loads(focused['spec'])
    sections = module.bank(value, 'project-view')['sections']
    sections[:] = [s for s in sections if s.get('__id__') != 'sampler-class-grids']
    section = copy.deepcopy(sections[0])
    panels = []
    for index, ((name, _), key) in enumerate(zip(module.CLASSES, module.MEDIA_KEYS[1:], strict=True)):
        panel = wr.MediaBrowser(title=f'{name.title()} — 64 fixed-seed samples',
            media_keys=[key], num_columns=1)._to_model().model_dump(by_alias=True, exclude_none=True)
        panel.update(__id__=f'sampler-class-{index}', isAuto=False)
        panels.append(panel)
    section.update(name='Selected sampler — per-class 8 × 8 grids', __id__='sampler-class-grids',
                   isOpen=True, pinned=True, isPanelsAuto=False, panels=panels)
    section['flowConfig'].update(columnsPerPage=2, rowsPerPage=4)
    sections.append(section)
    all_panels = [p for s in sections for p in s.get('panels', [])]
    media = [p for p in all_panels if p.get('viewType') == 'Media Browser']
    fid = [p for p in all_panels if 'confirm/fid50k' in p.get('config', {}).get('metrics', [])]
    assert len(fid) == 1 and [p['config']['mediaKeys'] for p in media] == [[key] for key in module.MEDIA_KEYS]
    publisher.save(focused, value)
    # Add an experiment file to the training run; preserve its live metrics,
    # config, default workspace and main FID history.
    destination = LOCAL/'publish/sampler-evaluation/epoch006-report.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2)+'\n')
    training = wandb.Api(timeout=45).run(f'helloimlixin-rutgers/laser/{ACTIVE.name}')
    training.upload_file(str(destination), root=str(LOCAL/'publish'))
    selected = dict(source_run=ACTIVE.name, source_epoch=6, source_step=3756,
        sampling=report['selected_sampling'], fid50k=report['selected_fid50k'],
        baseline_fid50k=report['baseline_fid50k'], improvement=report['improvement'],
        checkpoint_sha256=report['plan']['checkpoint']['sha256'],
        generated_images=50000, reference_images=1281167, reference_split='train',
        active_training_sampler_changed=False)
    (OUT/'selected-sampling.json').write_text(json.dumps(selected, indent=2)+'\n')
    proof = dict(passed=True, uploaded_media_keys=9, fid_panels=1,
        comparison_workspace=workspace['url'], training_metrics_and_config_unchanged=True,
        report_uploaded_to_training_run=True, selected_fid50k=report['selected_fid50k'],
        time=time.time())
    (OUT/'online-verification.json').write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(proof))


if __name__ == '__main__':
    main()
